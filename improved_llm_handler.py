"""Enhanced LLM Handler with robust JSON extraction and network resilience."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Suppress FutureWarnings and add time budget helpers
import warnings

import ollama


warnings.filterwarnings("ignore", category=FutureWarning)

# Import niche normalization and seed topics from config
from config import normalize_niche, settings


# JSON parsing utilities for robust LLM response handling
@dataclass
class VideoPlan:
    video_title: str
    target_duration_minutes: str  # örn "15-20"
    word_count: int
    script_sections: Dict[str, str]  # {"hook": "...", "body": "...", "outro": "..."}


JSON_SCHEMA_HINT = {
    "video_title": "string (concise, youtube-optimized)",
    "target_duration_minutes": "string range like '12-15'",
    "word_count": "integer 800-1400",
    "script_sections": {
        "hook": "80-120 words",
        "body": "bullet-like paragraphs",
        "outro": "30-60 words",
    },
}


def _extract_json_block(text: str) -> str:
    # 1) ```json ... ``` bloklarını tercih et
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        # ilk { ile son } arasını al
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
        else:
            candidate = text
    else:
        candidate = m.group(1)
    return candidate.strip()


def _force_json(candidate: str) -> Dict[str, Any]:
    s = candidate

    # 2) naive onarım: 15-20 -> "15-20"
    s = re.sub(r":\s*(\d+)\s*-\s*(\d+)\s*(,|})", r': "\1-\2"\3', s)

    # 3) True/False -> true/false ; None -> null
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    # 4) sondaki virgülleri temizle
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 5) çift tırnaksız anahtar varsa düzelt (basit durumlar)
    s = re.sub(r"(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)

    return json.loads(s)

    def _get_ollama_client(self):
        """Get Ollama client wrapper for the new JSON parsing system"""

        class OllamaClient:
            def chat(self, prompt):
                response = ollama.chat(
                    model=settings.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return type(
                    "Response",
                    (),
                    {"text": response.get("message", {}).get("content", "")},
                )()

        return OllamaClient()

    def _generate_video_plan_with_retry(self, niche: str, topic: str, llm) -> VideoPlan:
        """Generate video plan using robust JSON parsing with retry logic"""
        BASE_PROMPT = f"""
You are a disciplined planner. Return ONLY minified JSON, no commentary, no markdown.

STRICT SCHEMA (keys and types must match exactly):
{json.dumps(JSON_SCHEMA_HINT)}

Rules:
- If you need ranges, use strings like "12-15" (never 12-15 without quotes)
- Do not include code fences unless asked, but JSON is allowed in a ```json block too.
- No trailing commas. No extra keys. English only.

Now produce the object for niche="{niche}", topic="{topic}".
"""
        last_err = None
        for attempt in range(1, 4):
            resp = llm.chat(
                BASE_PROMPT
                if attempt == 1
                else BASE_PROMPT + f"\nATTEMPT {attempt}: Ensure strict JSON."
            )
            raw = resp.text if hasattr(resp, "text") else str(resp)

            try:
                jtxt = _extract_json_block(raw)
                data = _force_json(jtxt)
                # minimal validation
                assert isinstance(data.get("video_title"), str)
                assert isinstance(data.get("target_duration_minutes"), str)
                assert isinstance(data.get("word_count"), int)
                assert isinstance(data.get("script_sections"), dict)
                return VideoPlan(**data)
            except Exception as e:
                last_err = e
                time.sleep(0.8)
                continue
        raise ValueError(f"LLM JSON parse failed after retries: {last_err}")


# Pydantic models for topic scoring
try:
    from pydantic import BaseModel, Field, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️ Pydantic not available, using basic validation")

if PYDANTIC_AVAILABLE:

    class TopicScore(BaseModel):
        topic: str = Field(..., description="Topic text")
        score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0")

        @validator("score")
        def validate_score(cls, v):
            return max(0.0, min(1.0, v))

    class TopicCache(BaseModel):
        niche: str = Field(..., description="Content niche")
        topics: List[str] = Field(..., description="List of topics")
        timestamp: float = Field(..., description="Unix timestamp")
        selected_today: List[str] = Field(
            default_factory=list, description="Topics selected today"
        )
        source: str = Field(default="mixed", description="Source: online/offline/seed")


def niche_from_channel(channel_name: str) -> str:
    """
    Helper function to automatically resolve niche from channel name.

    Args:
        channel_name: Channel name (e.g., "CKDrive", "cklegends", "CKIronWill")

    Returns:
        Normalized niche string (e.g., "automotive", "history", "motivation")

    Examples:
        >>> niche_from_channel("CKDrive")
        'automotive'
        >>> niche_from_channel("cklegends")
        'history'
        >>> niche_from_channel("CKIronWill")
        'motivation'

    Usage in pipeline:
        # Instead of manually specifying niche:
        # topics = handler.get_topics_resilient("automotive", timeframe="today 1-m")

        # Use the helper for automatic resolution:
        niche = niche_from_channel(channel_name)  # "CKDrive" -> "automotive"
        topics = handler.get_topics_resilient(niche, timeframe="today 1-m", geo="US")
    """
    return normalize_niche(channel_name)


@contextmanager
def time_budget(seconds: float):
    start = time.monotonic()
    yield
    if (time.monotonic() - start) > seconds:
        raise TimeoutError(f"Time budget exceeded ({seconds}s)")


# --- Daily Cache + 7-day Dedupe Helpers ---
def _today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _cache_dir():
    d = os.path.join("data", "cache", "topics")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(niche: str):
    base = f"{_today_str()}::{niche.strip().lower()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]


def _cache_path(niche: str):
    return os.path.join(_cache_dir(), f"{_cache_key(niche)}.json")


def _load_recent_topics(niche: str, days: int = 7) -> list[str]:
    """Collect topics from last <days> cache files for dedupe."""
    recent = []
    root = _cache_dir()
    cutoff = datetime.utcnow() - timedelta(days=days)
    for fn in os.listdir(root):
        if not fn.endswith(".json"):
            continue
        try:
            date_part = fn.split("_")[0]  # backward-safe; ignore if missing
        except Exception:
            date_part = None
        fpath = os.path.join(root, fn)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            dt = datetime.utcfromtimestamp(data.get("ts", 0))
            if dt >= cutoff and data.get("niche") == niche.lower():
                recent.extend(data.get("topics", []))
        except Exception:
            continue
    # dedupe recent
    seen = set()
    out = []
    for t in recent:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out


def _save_topics_cache(niche: str, topics: list[str]):
    payload = {
        "ts": int(time.time()),
        "date": _today_str(),
        "niche": niche.lower(),
        "topics": topics,
    }
    with open(_cache_path(niche), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --- Augment Cache Helpers ---
def _day_salt(niche: str) -> str:
    base = f"{_today_str()}::{niche.strip().lower()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:6]


def _augment_cache_dir():
    d = os.path.join("data", "cache", "augment")
    os.makedirs(d, exist_ok=True)
    return d


def _augment_cache_path(niche: str):
    return os.path.join(_augment_cache_dir(), f"{_today_str()}_{niche.lower()}.json")


def _load_augment_cache(niche: str) -> list[str]:
    p = _augment_cache_path(niche)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("topics", [])
        except Exception:
            return []
    return []


def _save_augment_cache(niche: str, topics: list[str]):
    try:
        with open(_augment_cache_path(niche), "w", encoding="utf-8") as f:
            json.dump(
                {"date": _today_str(), "niche": niche, "topics": topics},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    from pytrends_offline import PyTrendsOffline
except Exception:
    PyTrendsOffline = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import AI_CONFIG, CHANNELS_CONFIG
except ImportError:
    print("❌ No configuration file found - using minimal defaults")
    CHANNELS_CONFIG = {}
    AI_CONFIG = {"ollama_model": "llama3:8b"}


def _get_trend_client():
    if TrendReq is not None:
        try:
            client = TrendReq(hl="en-US", tz=0)
            # Set timeout configuration for fast failure
            try:
                if hasattr(client, "timeout"):
                    client.timeout = (5, 10)  # (connect, read) timeout
                if hasattr(client, "requests_args") and isinstance(
                    client.requests_args, dict
                ):
                    client.requests_args.update({"allow_redirects": True})
            except Exception:
                pass  # Safe to ignore timeout config errors
            return client
        except Exception:
            pass
    if PyTrendsOffline is not None:
        return PyTrendsOffline()
    return None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> callable:
    """Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2**attempt)  # 1s, 2s, 4s
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


class ImprovedLLMHandler:
    """Enhanced LLM handler with robust JSON extraction and network resilience."""

    def __init__(self, model: Optional[str] = None, max_retries: int = 3) -> None:
        """Initialize the LLM handler.

        Args:
            model: Ollama model to use, defaults to config value
            max_retries: Maximum retry attempts for network operations
        """
        self.model = model or AI_CONFIG.get("ollama_model", "llama3:8b")
        self.max_retries = max_retries

        # Initialize PyTrends with timeout
        self.pytrends = _get_trend_client()
        if self.pytrends is None:
            logging.warning("PyTrends unavailable; using cached/fallback keywords.")

        self.setup_logging()

    def setup_logging(self) -> None:
        """Set up enhanced logging with standardized levels."""
        self.log_file = f"llm_handler_{int(time.time())}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📝 LLM Handler logging to: {self.log_file}")
        self.logger.info(f"🤖 Using Ollama model: {self.model}")

    def log_message(self, message: str, level: str = "INFO") -> None:
        """Log message with standardized levels.

        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        level_map = {
            "DEBUG": self.logger.debug,
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
        }

        log_func = level_map.get(level, self.logger.info)
        log_func(message)

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text using multiple fallback methods.

        Args:
            text: Text containing JSON data

        Returns:
            Extracted JSON string or None if extraction fails

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Text cannot be empty")

        # Method 1: Extract from fenced code blocks
        json_block = self._extract_from_fenced_blocks(text)
        if json_block:
            return json_block

        # Method 2: Use stack-based parser for balanced JSON
        balanced_json = self._extract_balanced_json(text)
        if balanced_json:
            return balanced_json

        self.log_message(
            f"Failed to extract valid JSON from text: {text[:100]}...", "ERROR"
        )
        return None

    def _extract_from_fenced_blocks(self, text: str) -> Optional[str]:
        """Extract JSON from ```json ... ``` fenced blocks.

        Args:
            text: Text containing fenced JSON blocks

        Returns:
            Extracted JSON string or None
        """
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    # Try to parse as-is
                    json.loads(match)
                    self.log_message("JSON extracted from fenced block", "DEBUG")
                    return match
                except json.JSONDecodeError:
                    # Try with fixes
                    fixed = self._fix_json_string(match)
                    if fixed:
                        try:
                            json.loads(fixed)
                            self.log_message(
                                "JSON extracted from fenced block after fixes", "DEBUG"
                            )
                            return fixed
                        except json.JSONDecodeError:
                            continue

        return None

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Stack-based parser to find the first complete JSON object/array.

        Args:
            text: Text containing JSON data

        Returns:
            Extracted JSON string or None
        """
        try:
            # Find first opening brace or bracket
            start_chars = {"{": "}", "[": "]"}
            start_idx = -1
            start_char = None

            for i, char in enumerate(text):
                if char in start_chars:
                    start_idx = i
                    start_char = char
                    break

            if start_idx == -1:
                return None

            # Use stack to find matching closing character
            stack = []
            for i in range(start_idx, len(text)):
                char = text[i]

                if char == start_char:
                    stack.append(char)
                elif char == start_chars[start_char]:
                    stack.pop()
                    if not stack:  # Found complete structure
                        json_text = text[start_idx : i + 1]

                        # Try to parse
                        try:
                            json.loads(json_text)
                            self.log_message(
                                "JSON extracted using balanced parser", "DEBUG"
                            )
                            return json_text
                        except json.JSONDecodeError:
                            # Try with fixes
                            fixed = self._fix_json_string(json_text)
                            if fixed:
                                try:
                                    json.loads(fixed)
                                    self.log_message(
                                        "JSON extracted using balanced parser after fixes",
                                        "DEBUG",
                                    )
                                    return fixed
                                except json.JSONDecodeError:
                                    continue

            return None

        except Exception as e:
            self.log_message(f"Balanced parser error: {e}", "ERROR")
            return None

    def _fix_json_string(self, json_text: str) -> Optional[str]:
        """Apply common JSON fixes for malformed JSON.

        Args:
            json_text: Potentially malformed JSON string

        Returns:
            Fixed JSON string or None
        """
        try:
            # Remove trailing commas
            json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)

            # Convert single quotes to double quotes (preserving escapes)
            json_text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', json_text)

            # Fix Python literals (only if they're not in strings)
            json_text = re.sub(r"\bTrue\b", "true", json_text)
            json_text = re.sub(r"\bFalse\b", "false", json_text)
            json_text = re.sub(r"\bNone\b", "null", json_text)

            return json_text

        except Exception as e:
            self.log_message(f"JSON fixing error: {e}", "ERROR")
            return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _get_ollama_response(self, prompt_template: str) -> Optional[Dict[str, Any]]:
        """Get response from Ollama LLM with retry logic.

        Args:
            prompt_template: Prompt to send to the LLM

        Returns:
            Parsed JSON response or None if failed

        Raises:
            ValueError: If response is empty or invalid
            RuntimeError: If LLM communication fails
        """
        try:
            self.log_message("LLM communication attempt...", "DEBUG")

            # Note: ollama library doesn't support timeout directly
            # We'll implement timeout at the application level
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt_template}],
            )

            raw_text = response.get("message", {}).get("content")
            if not raw_text:
                raise ValueError("Empty response from LLM")

            # Try the existing extraction method first
            clean_json_text = self._extract_json_from_text(raw_text)
            if not clean_json_text:
                # Fallback to the new robust parsing system
                try:
                    jtxt = _extract_json_block(raw_text)
                    data = _force_json(jtxt)
                    self.log_message(
                        "JSON extracted using robust parsing fallback", "INFO"
                    )
                    return data
                except Exception as e:
                    self.log_message(
                        f"Robust parsing fallback also failed: {e}", "ERROR"
                    )
                    raise ValueError("No valid JSON found in response")

            parsed_json = json.loads(clean_json_text)
            self.log_message("LLM response successfully parsed", "INFO")
            return parsed_json

        except (ValueError, json.JSONDecodeError) as e:
            self.log_message(f"LLM response parsing failed: {e}", "ERROR")
            raise
        except Exception as e:
            self.log_message(f"LLM communication failed: {e}", "ERROR")
            raise RuntimeError(f"LLM communication failed: {e}") from e

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _get_pytrends_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Get trending topics using PyTrends API with timeout and retry.

        Args:
            niche: Content niche to search for
            timeframe: Time range for trends
            geo: Geographic location

        Returns:
            List of trending topics

        Raises:
            RuntimeError: If PyTrends API fails
        """
        # Import pandas and configure to suppress silent downcasting warnings
        import pandas as pd

        pd.set_option("future.no_silent_downcasting", True)

        if not self.pytrends:
            self.log_message(
                "PyTrends not available - skipping trending topics", "WARNING"
            )
            return []

        try:
            # Map niches to relevant search terms
            niche_queries = {
                "history": [
                    "ancient mysteries",
                    "historical discoveries",
                    "archaeology news",
                ],
                "science": [
                    "scientific breakthroughs",
                    "space discoveries",
                    "technology trends",
                ],
                "mystery": [
                    "unsolved mysteries",
                    "conspiracy theories",
                    "paranormal news",
                ],
                "true_crime": ["crime documentaries", "cold cases", "forensic science"],
                "nature": [
                    "wildlife discoveries",
                    "nature mysteries",
                    "environmental news",
                ],
            }

            queries = niche_queries.get(niche, [niche])
            trending_topics = []

            # Try only the first query to avoid long loops - fast fail approach
            query = queries[0] if queries else niche
            try:
                if hasattr(self.pytrends, "build_payload"):
                    # Online PyTrends - single attempt
                    self.pytrends.build_payload(
                        [query], timeframe=timeframe, geo=geo, gprop=""
                    )
                    trends_data = self.pytrends.interest_over_time()

                    if not trends_data.empty:
                        # Get top trending terms
                        if hasattr(self.pytrends, "trending_searches"):
                            top_terms = self.pytrends.trending_searches(
                                pn="united_states"
                            )
                            if not top_terms.empty:
                                trending_topics.extend(top_terms[0].head(5).tolist())
                elif hasattr(self.pytrends, "get_trending_topics"):
                    # Offline PyTrends
                    offline_topics = self.pytrends.get_trending_topics(
                        niche, max_results=10
                    )
                    if offline_topics:
                        trending_topics.extend(offline_topics[:5])

            except Exception as e:
                self.log_message(f"PyTrends query failed for '{query}': {e}", "WARNING")
                # Fast fail - don't continue with more queries

            # Remove duplicates and return
            unique_topics = list(dict.fromkeys(trending_topics))
            self.log_message(
                f"Found {len(unique_topics)} trending topics for niche '{niche}'",
                "INFO",
            )
            return unique_topics[:10]  # Limit to top 10

        except Exception as e:
            self.log_message(f"Error in PyTrends topics: {e}", "ERROR")
            raise RuntimeError(f"PyTrends API failed: {e}") from e

    # OLD FALLBACK FUNCTION - DISABLED (using get_topics_resilient instead)
    # def _get_trending_topics_with_fallback(
    #     self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    # ) -> List[str]:
    #     """Get trending topics with PyTrends fallback to local JSON cache.
    #
    #     Args:
    #         niche: Content niche to search for
    #         timeframe: Time range for trends
    #         geo: Geographic location
    #
    #     Returns:
    #         List of trending topics
    #     """
    #     topics = []
    #     try:
    #         # Try online PyTrends first
    #         topics = self._get_pytrends_topics(niche=niche, timeframe=timeframe, geo=geo)
    #     except Exception as e:
    #         self.log_message(f"PyTrends online failed: {e}; switching to offline.", "WARNING")
    #         topics = []
    #
    #     # If empty, try offline fallback
    #     if not topics:
    #         try:
    #             if hasattr(self.pytrends, "get_trending_topics"):
    #                 topics = self.pytrends.get_trending_topics(niche, max_results=20)
    #         except Exception as e:
    #             self.log_message(f"Offline trends failed: {e}", "WARNING")
    #         topics = []
    #
    #     # Seed fallback: if still empty, provide default set based on niche
    #     if not topics:
    #         SEED = {
    #             "history": [
    #                 "ancient civilizations","lost cities","roman empire","greek mythology",
    #                 "egyptian pharaohs","archaeology discoveries","medieval knights",
    #                 "viking history","mysteries of history","ancient inventions"
    #             ],
    #         "motivation": [
    #                 "discipline tips","productivity habits","mental toughness","morning routine",
    #                 "goal setting","sports motivation","habit building","focus techniques",
    #                 "mindset shift","success stories"
    #         ],
    #         "science": [
    #             "space discoveries","quantum physics","evolution mysteries","climate science",
    #                 "medical breakthroughs","technology trends","scientific controversies",
    #                 "unexplained phenomena","research findings","future predictions"
    #         ],
    #         "mystery": [
    #             "unsolved mysteries","conspiracy theories","paranormal events",
    #                 "cryptid sightings","ancient artifacts","lost treasures","urban legends",
    #                 "supernatural stories","mysterious disappearances","occult history"
    #         ],
    #         "true_crime": [
    #             "cold cases","forensic breakthroughs","criminal psychology",
    #                 "unsolved murders","mysterious deaths","criminal investigations",
    #                 "justice system","crime prevention","victim stories","detective work"
    #         ],
    #         "nature": [
    #             "wildlife discoveries","environmental mysteries","natural phenomena",
    #                 "animal behavior","plant adaptations","ecosystem changes",
    #                 "climate effects","biodiversity","natural disasters","conservation"
    #         ]
    #     }
    #         topics = SEED.get(niche.lower(), SEED["history"])
    #         self.log_message(f"PyTrends online failed: {e}; switching to offline.", "WARNING")
    #
    #     return topics

    def get_topics_by_channel(
        self, channel_name: str, timeframe: str | None = None, geo: str | None = None
    ) -> list[str]:
        """
        Convenience method to get topics by channel name with automatic niche resolution.

        Args:
            channel_name: Channel name (e.g., "CKDrive", "cklegends", "CKIronWill")
            timeframe: Time range for trends (e.g., "today 1-m", "now 7-d")
            geo: Geographic location (e.g., "US", "GB", "CA")

        Returns:
            List of trending topics for the channel's niche

        Examples:
            >>> handler = ImprovedLLMHandler()
            >>> topics = handler.get_topics_by_channel("CKDrive", geo="US")
            >>> # Automatically resolves "CKDrive" -> "automotive" niche
        """
        niche = niche_from_channel(channel_name)
        return self.get_topics_resilient(niche, timeframe, geo)

    def get_topics_resilient(
        self, niche: str, timeframe: str | None = None, geo: str | None = None
    ) -> list[str]:
        """
        Get topics with resilient fallback: online (pytrends) → offline (pytrends_offline) → seed list
        Guarantees 24 topics with 7-day dedupe and daily augmentation
        """
        # Niche normalization
        niche = normalize_niche(niche)

        # Check today's cache first
        today_cache = self._load_today_cache(niche)
        if today_cache and len(today_cache.get("topics", [])) >= 24:
            self.log_message(
                f"Using today's cached topics for {niche}: {len(today_cache['topics'])} topics",
                "INFO",
            )
            return today_cache["topics"][:24]

        topics: list[str] = []
        source = "unknown"
        online_warned = False

        # ---- ONLINE (pytrends) ----
        try:
            with time_budget(6.0):
                topics = self._get_online_topics(niche, timeframe, geo)
                if topics and len(topics) >= 8:
                    source = "online"
                    self.log_message(
                        f"PyTrends online OK: {len(topics)} topics", "INFO"
                    )
                else:
                    raise Exception("Insufficient online topics")
        except Exception as e:
            if not online_warned:
                self.log_message(
                    f"PyTrends online failed (404/429/timeout): {e}", "WARNING"
                )
                online_warned = True
            topics = []

        # ---- OFFLINE (pytrends_offline) ----
        if not topics or len(topics) < 8:
            try:
                offline_topics = self._get_offline_topics(niche)
                if offline_topics and len(offline_topics) >= 8:
                    topics = offline_topics
                    source = "offline"
                    self.log_message(
                        f"Offline trends used: {len(topics)} topics", "INFO"
                    )
            except Exception as e:
                self.log_message(f"Offline trends failed: {e}", "WARNING")

        # ---- SEED LIST (guaranteed fallback) ----
        if not topics or len(topics) < 8:
            seed_topics = self._get_seed_topics(niche)
            topics = seed_topics
            source = "seed"
            self.log_message(
                f"Using seed fallback for '{niche}': {len(topics)} topics", "WARNING"
            )

        # ---- ENSURE 24 TOPICS MINIMUM ----
        if len(topics) < 24:
            topics = self._ensure_minimum_topics(niche, topics, 24)

        # ---- 7-DAY DEDUPE + AUGMENT ----
        topics = self._apply_dedupe_and_augment(niche, topics)

        # ---- CACHE WITH TODAY'S SELECTION TRACKING ----
        self._save_today_cache(niche, topics, source)

        self.log_message(
            f"Final topics for {niche}: {len(topics)} (source: {source})", "INFO"
        )
        return topics[:24]  # Guarantee max 24 topics

    def _get_online_topics(
        self, niche: str, timeframe: str | None = None, geo: str | None = None
    ) -> list[str]:
        """Get topics from online PyTrends with error handling for 404/429"""
        try:
            geos_to_try = (
                [geo]
                if geo
                else (
                    getattr(settings, "TIER1_GEOS", ["US"])
                    + getattr(settings, "TIER2_GEOS", ["GB", "CA"])
                )
            )
            frames_to_try = (
                [timeframe]
                if timeframe
                else getattr(settings, "DEFAULT_TIMEFRAMES", ["today 1-m", "now 7-d"])
            )

            for g in geos_to_try:
                for tf in frames_to_try:
                    try:
                        if hasattr(self, "pytrends") and hasattr(
                            self.pytrends, "build_payload"
                        ):
                            self.pytrends.build_payload(
                                [niche], timeframe=tf, geo=g, gprop=""
                            )
                            trends_data = self.pytrends.interest_over_time()
                            if not trends_data.empty and hasattr(
                                self.pytrends, "trending_searches"
                            ):
                                top_terms = self.pytrends.trending_searches(
                                    pn="united_states"
                                )
                                if not top_terms.empty:
                                    topics = top_terms[0].head(8).tolist()
                                    if topics:
                                        return topics
                    except Exception as e:
                        if "404" in str(e) or "429" in str(e):
                            self.log_message(
                                f"PyTrends rate limit/not found ({g}, {tf}): {e}",
                                "WARNING",
                            )
                            continue
                        else:
                            self.log_message(
                                f"PyTrends error ({g}, {tf}): {e}", "WARNING"
                            )
                            continue
            return []
        except Exception as e:
            self.log_message(f"Online topics failed: {e}", "WARNING")
            return []

    def _get_offline_topics(self, niche: str) -> list[str]:
        """Get topics from offline PyTrends fallback"""
        try:
            if hasattr(self, "pytrends") and hasattr(
                self.pytrends, "get_trending_topics"
            ):
                topics = self.pytrends.get_trending_topics(
                    niche, max_results=getattr(settings, "MAX_TOPICS", 24)
                )
                if topics:
                    return topics[:24]
        except Exception as e:
            self.log_message(f"Offline topics failed: {e}", "WARNING")
        return []

    def _get_seed_topics(self, niche: str) -> list[str]:
        """Get guaranteed seed topics for niche"""
        try:
            seed_topics = getattr(settings, "SEED_TOPICS", {}).get(niche.lower(), [])
            if not seed_topics:
                seed_topics = getattr(settings, "SEED_TOPICS", {}).get("history", [])

            # Ensure we have at least 24 topics
            if len(seed_topics) < 24:
                # Add generic topics
                generic_topics = [
                    "Ancient mysteries revealed",
                    "Lost civilizations found",
                    "Historical secrets exposed",
                    "Forgotten stories uncovered",
                    "Mysterious artifacts discovered",
                    "Hidden truths revealed",
                    "Ancient wisdom decoded",
                    "Lost knowledge recovered",
                    "Historical puzzles solved",
                    "Mysterious events explained",
                    "Ancient technology revealed",
                    "Lost treasures found",
                    "Historical controversies resolved",
                    "Mysterious disappearances solved",
                    "Ancient rituals explained",
                    "Lost cities discovered",
                    "Historical myths debunked",
                    "Mysterious symbols decoded",
                    "Ancient prophecies fulfilled",
                    "Lost documents found",
                    "Historical mysteries solved",
                    "Mysterious phenomena explained",
                    "Ancient legends proven",
                    "Lost knowledge revealed",
                ]
                seed_topics.extend(generic_topics)

            return seed_topics[:24]
        except Exception as e:
            self.log_message(f"Seed topics failed: {e}", "WARNING")
            # Ultimate fallback
            return [
                f"{niche} mystery",
                f"{niche} revealed",
                f"{niche} secrets",
                f"{niche} facts",
            ] * 6

    def _ensure_minimum_topics(
        self, niche: str, topics: list[str], minimum: int
    ) -> list[str]:
        """Ensure minimum number of topics by augmenting if needed"""
        if len(topics) >= minimum:
            return topics

        # Augment with variations
        augmented = self.augment_seed_topics(niche, topics, want=minimum - len(topics))
        topics.extend(augmented)

        # If still not enough, add generic topics
        if len(topics) < minimum:
            generic = [f"{niche} {i}" for i in range(1, minimum - len(topics) + 1)]
            topics.extend(generic)

        return topics[:minimum]

    def _apply_dedupe_and_augment(self, niche: str, topics: list[str]) -> list[str]:
        """Apply 7-day dedupe and augmentation"""
        try:
            # Load recent topics for dedupe
            previous = _load_recent_topics(niche, days=7)
            prevset = {p.strip().lower() for p in previous} if previous else set()

            # Dedupe current topics
            seen = set()
            deduped = []
            for t in topics:
                k = t.strip().lower()
                if k and k not in seen and k not in prevset:
                    seen.add(k)
                    deduped.append(t.strip())

            # If not enough novel topics, augment
            if len(deduped) < 12:
                augmented = self.augment_seed_topics(niche, topics, want=16)
                for t in augmented:
                    k = t.strip().lower()
                    if k and k not in seen and k not in prevset:
                        seen.add(k)
                        deduped.append(t.strip())

            # Shuffle and return
            random.shuffle(deduped)
            return deduped[:24]

        except Exception as e:
            self.log_message(f"Dedupe/augment failed: {e}", "WARNING")
            return topics[:24]

    def _load_today_cache(self, niche: str) -> dict | None:
        """Load today's topic cache"""
        try:
            cache_path = _cache_path(niche)
            if os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                    # Check if it's from today
                    if data.get("date") == _today_str():
                        return data
        except Exception as e:
            self.log_message(f"Cache load failed: {e}", "WARNING")
        return None

    def _save_today_cache(self, niche: str, topics: list[str], source: str):
        """Save today's topic cache with selection tracking"""
        try:
            cache_data = {
                "niche": niche,
                "date": _today_str(),
                "timestamp": time.time(),
                "topics": topics,
                "source": source,
                "selected_today": [],  # Will be populated when topics are selected
                "count": len(topics),
            }

            if PYDANTIC_AVAILABLE:
                # Validate with pydantic
                validated = TopicCache(**cache_data)
                cache_data = validated.dict()

            cache_path = _cache_path(niche)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.log_message(
                f"Cached {len(topics)} topics for {niche} (source: {source})", "INFO"
            )

        except Exception as e:
            self.log_message(f"Cache save failed: {e}", "WARNING")

    def score_topics_with_llm(
        self, niche: str, topics: list[str], top_k: int = 8
    ) -> list[tuple[str, float]]:
        """
        Returns list of (topic, score) sorted desc by score. Falls back to simple heuristics.
        """
        if not topics:
            return []
        prompt = (
            "You are a YouTube growth strategist. Score each topic from 0.0 to 1.0 based on:\n"
            "1. CTR potential (click-through rate)\n"
            "2. Curiosity gap (mystery, intrigue)\n"
            "3. Evergreen appeal (timeless relevance)\n"
            "4. Niche alignment\n\n"
            f"Niche: {niche}\n"
            'Return ONLY a JSON array: [{"topic": "text", "score": 0.85}]\n'
            f"Topics: {topics[:24]}"
        )
        scored = []
        try:
            # Use your existing LLM call helper if you have one; else a minimal call:
            resp = self._get_ollama_response(prompt)  # your safe JSON extractor
            for item in resp:
                t = str(item.get("topic", "")).strip()
                s = float(item.get("score", 0))
                if t:
                    scored.append((t, max(0.0, min(1.0, s))))
        except Exception:
            # Heuristic fallback: prefer shorter, high-curiosity tokens
            def heuristic(t):
                base = 0.5
                if any(
                    k in t.lower()
                    for k in [
                        "mystery",
                        "unknown",
                        "secret",
                        "revealed",
                        "ancient",
                        "lost",
                        "why",
                        "how",
                    ]
                ):
                    base += 0.2
                base += max(0, (40 - len(t))) / 100.0  # shorter titles slightly higher
                return min(1.0, base)

            scored = [(t, heuristic(t)) for t in topics]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _heuristic_scoring(
        self, topics: list[str], count: int
    ) -> list[tuple[str, float]]:
        """Heuristic scoring fallback for topics"""
        scored = []

        for topic in topics:
            if len(scored) >= count:
                break

            base_score = 0.5

            # Boost for curiosity keywords
            curiosity_keywords = [
                "mystery",
                "unknown",
                "secret",
                "revealed",
                "ancient",
                "lost",
                "why",
                "how",
                "hidden",
                "forgotten",
                "discovered",
                "uncovered",
            ]
            if any(keyword in topic.lower() for keyword in curiosity_keywords):
                base_score += 0.2

            # Boost for shorter titles (better CTR)
            if len(topic) <= 40:
                base_score += 0.1

            # Boost for action words
            action_words = [
                "revealed",
                "exposed",
                "discovered",
                "found",
                "solved",
                "explained",
            ]
            if any(word in topic.lower() for word in action_words):
                base_score += 0.1

            # Ensure score is within bounds
            final_score = max(0.0, min(1.0, base_score))
            scored.append((topic, final_score))

        return scored

    def augment_seed_topics(
        self, niche: str, topics: list[str], want: int = 16
    ) -> list[str]:
        """Parafraz + varyant üretir. LLM yoksa heuristikle üretir. Günlük cache vardır."""
        cached = _load_augment_cache(niche)
        if cached:
            return cached[:want]

        # LLM dene (kısa, hızlı yanıt)
        prompt = (
            "Rewrite each topic into 1 new catchy variant for YouTube titles.\n"
            "Keep meaning, change phrasing. Avoid clickbait.\n"
            "Return ONLY JSON array of strings with the same order and length as input.\n"
            f"Niche: {niche}\nTopics: {topics[:want]}"
        )
        variants = []
        try:
            resp = self._get_ollama_response(prompt)  # your safe JSON extractor
            if isinstance(resp, list):
                variants = [str(x).strip() for x in resp if str(x).strip()]
        except Exception:
            variants = []

        # Heuristik fallback (güvenli ve hızlı)
        if not variants or len(variants) < max(8, want // 2):
            suffixes = [
                "explained",
                "revealed",
                "in 5 minutes",
                "you should know",
                "that changed history",
                "the untold story",
                "debunked",
                "guide",
                "timeline",
                "top facts",
            ]
            out = []
            for t in topics:
                s = random.choice(suffixes)
                out.append(f"{t} — {s}")
                if len(out) >= want:
                    break
            variants = variants or out

        # dedupe + trim
        seen = set()
        uniq = []
        for v in variants:
            k = v.lower().strip()
            if k and k not in seen:
                seen.add(k)
                uniq.append(v.strip())
        uniq = uniq[:want]

        _save_augment_cache(niche, uniq)
        return uniq

    def _cache_trending_topics(
        self, niche: str, topics: List[str], timeframe: str = "today 1-m", geo: str = ""
    ) -> None:
        """Cache trending topics with extended key: f"{niche}:{geo}:{timeframe}".

        Args:
            niche: Content niche
            topics: List of trending topics
            timeframe: Time range
            geo: Geographic location
        """
        try:
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)

            # Extended cache key format
            cache_key = f"{niche}:{geo}:{timeframe}".replace(":", "_").replace(" ", "_")
            cache_file = os.path.join(cache_dir, f"trending_topics_{cache_key}.json")

            cache_data = {
                "niche": niche,
                "timeframe": timeframe,
                "geo": geo,
                "timestamp": time.time(),
                "topics": topics,
                "source": "pytrends",
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.log_message(
                f"Cached {len(topics)} topics with key '{cache_key}'", "INFO"
            )

        except OSError as e:
            self.log_message(f"Failed to cache topics: {e}", "ERROR")
        except Exception as e:
            self.log_message(f"Unexpected error caching topics: {e}", "ERROR")

    def _load_cached_trending_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Load cached trending topics using extended key format.

        Args:
            niche: Content niche
            timeframe: Time range
            geo: Geographic location

        Returns:
            List of cached topics or empty list if cache miss/expired
        """
        try:
            # Extended cache key format
            cache_key = f"{niche}:{geo}:{timeframe}".replace(":", "_").replace(" ", "_")
            cache_file = os.path.join("cache", f"trending_topics_{cache_key}.json")

            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Check if cache is still valid (24 hours)
                cache_timestamp = cache_data.get("timestamp", 0)
                if time.time() - cache_timestamp < 24 * 60 * 60:  # 24 hours
                    self.log_message(f"Cache HIT for key '{cache_key}'", "DEBUG")
                    return cache_data.get("topics", [])
                else:
                    self.log_message(f"Cache EXPIRED for key '{cache_key}'", "DEBUG")
            else:
                self.log_message(f"Cache MISS for key '{cache_key}'", "DEBUG")

            return []

        except (OSError, json.JSONDecodeError) as e:
            self.log_message(f"Cache loading failed: {e}", "WARNING")
            return []
        except Exception as e:
            self.log_message(f"Unexpected error loading cache: {e}", "ERROR")
            return []

    def get_trending_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Get trending topics with enhanced fallback system and extended cache keys.

        Args:
            niche: Content niche to search for
            timeframe: Time range for trends
            geo: Geographic location

        Returns:
            List of trending topics
        """
        return self._get_trending_topics_with_fallback(niche, timeframe, geo)

    def generate_viral_ideas(
        self, channel_name: str, idea_count: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate viral video ideas with enhanced trending topics integration.

        Args:
            channel_name: Name of the YouTube channel
            idea_count: Number of ideas to generate

        Returns:
            List of viral video ideas or None if generation fails
        """
        try:
            # Use the helper function for automatic niche resolution
            niche = niche_from_channel(channel_name)

            # Get trending topics with resilient fallback system
            trending_topics = self.get_topics_resilient(niche=niche)
            self.log_message(
                f"Selected topics for '{niche}': {len(trending_topics)}", "INFO"
            )

            # Score and select top 8 topics
            topics_scored = self.score_topics_with_llm(niche, trending_topics, top_k=8)
            best_topics = [t for t, _ in topics_scored]
            self.logger.info(f"Selected top {len(best_topics)} topics: {best_topics}")

            trending_context = ""
            if best_topics:
                trending_context = f"Current trending topics in this niche: {', '.join(best_topics[:5])}. "

            prompt = f"""You are a master content strategist for viral YouTube documentaries. Generate {idea_count} viral video idea for a YouTube channel about '{niche}'.

{trending_context}

Focus on creating DEEP, ENGAGING content that can sustain 10+ minute videos. Each idea must include:
- A compelling mystery or untold story
- Multiple cliffhangers and suspense elements
- Engagement hooks to keep viewers engaged
- Global appeal for English-speaking audiences

REQUIRED JSON FORMAT - Each idea must have:
{{
  "ideas": [
    {{
      "title": "Compelling title that creates curiosity",
      "description": "Detailed description of the story/mystery",
      "duration_minutes": 12-18,
      "engagement_hooks": [
        "Hook that shocks viewers at 2 minutes",
        "Cliffhanger at 5 minutes",
        "Revelation at 8 minutes",
        "Twist at 12 minutes",
        "Final shock at 15 minutes"
      ],
      "trending_relevance": "How this connects to current trends",
      "global_appeal": "Why this appeals to international audiences",
      "subtitle_languages": ["English", "Spanish", "French", "German"]
    }}
  ]
}}

Make each idea highly specific and researchable. Focus on creating genuine curiosity and engagement."""

            result = self._get_ollama_response(prompt)
            if result and "ideas" in result:
                self.log_message(
                    f"Generated {len(result['ideas'])} viral ideas for '{niche}'",
                    "INFO",
                )
                return result["ideas"]
            else:
                self.log_message("Failed to generate viral ideas", "ERROR")
                return None

        except Exception as e:
            self.log_message(f"Error in generate_viral_ideas: {e}", "ERROR")
            return None

    def write_script(
        self, video_idea: Dict[str, Any], channel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed script with exact niche match for visual prevention.

        Args:
            video_idea: Video idea dictionary
            channel_name: Name of the YouTube channel

        Returns:
            Generated script dictionary or None if generation fails
        """
        try:
            # Use the helper function for automatic niche resolution
            niche = niche_from_channel(channel_name)

            prompt = f"""You are a master scriptwriter for viral YouTube documentaries. Write a highly detailed, long-form script for a 15-20 minute video on: '{video_idea.get('title', 'N/A')}' using the "Hook → Promise → Proof → Preview" template.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 80-120 words total (no less, no more)
- Follow the 4-part structure: Hook (20-30 words) → Promise (20-30 words) → Proof (20-30 words) → Preview (20-30 words)
- Each part must be a complete, engaging thought
- Include multiple cliffhangers and suspense elements
- Create engagement hooks at specific time intervals
- Optimize Pexels queries for cinematic, high-quality visuals
- EXACT NICHE MATCH: Use precise, specific visual queries that match the exact niche '{niche}' to prevent irrelevant visuals

REQUIRED JSON FORMAT:
{{
  "video_title": "{video_idea.get('title', 'N/A')}",
  "target_duration_minutes": 15-20,
  "word_count": 0,
  "script_structure": {{
    "hook": {{
      "content": "Opening hook to grab attention (20-30 words)",
      "visual_query": "cinematic 4K [exact {niche} scene] with dramatic lighting",
      "timing_seconds": 0,
      "engagement_hook": "Shocking opening statement or question"
    }},
    "promise": {{
      "content": "What viewers will learn/discover (20-30 words)",
      "visual_query": "cinematic 4K [exact {niche} scene] building anticipation",
      "timing_seconds": 8,
      "engagement_hook": "Building curiosity and expectation"
    }},
    "proof": {{
      "content": "Evidence and credibility building (20-30 words)",
      "visual_query": "cinematic 4K [exact {niche} scene] authoritative mood",
      "timing_seconds": 16,
      "engagement_hook": "Establishing trust and expertise"
    }},
    "preview": {{
      "content": "Teaser for what's coming next (20-30 words)",
      "visual_query": "cinematic 4K [exact {niche} scene] mysterious atmosphere",
      "timing_seconds": 24,
      "engagement_hook": "Creating anticipation for next section"
    }}
  }},
  "metadata": {{
    "subtitle_languages": ["English", "Spanish", "French", "German"],
    "target_audience": "Global English-speaking viewers",
    "engagement_strategy": "Hook → Promise → Proof → Preview structure with cliffhangers",
    "visual_prevention": "Exact niche matching for '{niche}' to prevent irrelevant visuals"
  }}
}}

Focus on creating genuine suspense and curiosity. Each section should advance the story while maintaining viewer engagement. Use EXACT niche matching in visual queries. Ensure total word count is exactly 80-120 words.

CRITICAL: Return ONLY valid JSON. Do not include any explanatory text, markdown formatting, or additional content outside the JSON structure."""

            # Use the new robust JSON parsing system
            try:
                # First, try to get a video plan using the new system
                video_plan = self._generate_video_plan_with_retry(
                    niche=niche,
                    topic=video_idea.get("title", "N/A"),
                    llm=self._get_ollama_client(),
                )

                # Convert VideoPlan to the expected script format
                result = {
                    "video_title": video_plan.video_title,
                    "target_duration_minutes": video_plan.target_duration_minutes,
                    "word_count": video_plan.word_count,
                    "script_structure": {
                        "hook": {
                            "content": video_plan.script_sections.get("hook", ""),
                            "visual_query": f"cinematic 4K {niche} scene with dramatic lighting",
                            "timing_seconds": 0,
                            "engagement_hook": "Shocking opening statement or question",
                        },
                        "promise": {
                            "content": video_plan.script_sections.get("body", ""),
                            "visual_query": f"cinematic 4K {niche} scene building anticipation",
                            "timing_seconds": 8,
                            "engagement_hook": "Building curiosity and expectation",
                        },
                        "proof": {
                            "content": video_plan.script_sections.get("body", ""),
                            "visual_query": f"cinematic 4K {niche} scene authoritative mood",
                            "timing_seconds": 16,
                            "engagement_hook": "Establishing trust and expertise",
                        },
                        "preview": {
                            "content": video_plan.script_sections.get("outro", ""),
                            "visual_query": f"cinematic 4K {niche} scene mysterious atmosphere",
                            "timing_seconds": 24,
                            "engagement_hook": "Creating anticipation for next section",
                        },
                    },
                    "metadata": {
                        "subtitle_languages": [
                            "English",
                            "Spanish",
                            "French",
                            "German",
                        ],
                        "target_audience": "Global English-speaking viewers",
                        "engagement_strategy": "Hook → Promise → Proof → Preview structure with cliffhangers",
                        "visual_prevention": f"Exact niche matching for '{niche}' to prevent irrelevant visuals",
                    },
                }

                self.log_message(
                    f"Generated script with {video_plan.word_count} words using robust JSON parsing for '{video_idea.get('title', 'N/A')}'",
                    "INFO",
                )

                # Validate word count
                if video_plan.word_count < 800 or video_plan.word_count > 1400:
                    self.log_message(
                        f"Warning: Script has {video_plan.word_count} words (target: 800-1400 words)",
                        "WARNING",
                    )

                return result

            except Exception as e:
                self.log_message(
                    f"Robust JSON parsing failed, falling back to old method: {e}",
                    "WARNING",
                )

                # Fallback to old method
                result = self._get_ollama_response(prompt)
                if result and "script_structure" in result:
                    # Calculate total word count
                    total_words = 0
                    if "hook" in result["script_structure"]:
                        total_words += len(
                            result["script_structure"]["hook"]["content"].split()
                        )
                    if "promise" in result["script_structure"]:
                        total_words += len(
                            result["script_structure"]["promise"]["content"].split()
                        )
                    if "proof" in result["script_structure"]:
                        total_words += len(
                            result["script_structure"]["proof"]["content"].split()
                        )
                    if "preview" in result["script_structure"]:
                        total_words += len(
                            result["script_structure"]["preview"]["content"].split()
                        )

                    # Update word count in result
                    result["word_count"] = total_words

                    self.log_message(
                        f"Fallback: Generated script with {total_words} words using Hook→Promise→Proof→Preview structure for '{video_idea.get('title', 'N/A')}'",
                        "INFO",
                    )

                    # Validate word count
                    if total_words < 80 or total_words > 120:
                        self.log_message(
                            f"Warning: Script has {total_words} words (target: 80-120 words)",
                            "WARNING",
                        )

                    return result
                else:
                    self.log_message(
                        "Failed to generate script with both methods", "ERROR"
                    )
                    return None

        except Exception as e:
            self.log_message(f"Error in write_script: {e}", "ERROR")
            return None

    def enhance_script_with_metadata(
        self, script_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add enhanced metadata, optimization suggestions, and self-improve with Ollama.

        Args:
            script_data: Script data dictionary

        Returns:
            Enhanced script data with metadata
        """
        try:
            if not script_data or "script" not in script_data:
                return script_data

            # Calculate estimated duration
            total_sentences = len(script_data["script"])
            estimated_duration = total_sentences * 8  # Assume 8 seconds per sentence

            # Self-improve: Use Ollama to generate extra sentences if duration is low
            if estimated_duration < 600:  # Less than 10 minutes
                self.log_message(
                    f"Self-improving script: Duration {estimated_duration}s is low, generating extra sentences",
                    "INFO",
                )
                extra_sentences = self._generate_extra_sentences_with_ollama(
                    script_data, estimated_duration
                )
                if extra_sentences:
                    script_data["script"].extend(extra_sentences)
                    total_sentences = len(script_data["script"])
                    estimated_duration = total_sentences * 8
                    self.log_message(
                        f"Added {len(extra_sentences)} extra sentences, new total: {total_sentences}",
                        "INFO",
                    )

            # Add enhanced metadata
            enhanced_metadata = {
                "estimated_duration_seconds": estimated_duration,
                "estimated_duration_minutes": round(estimated_duration / 60, 1),
                "sentence_count": total_sentences,
                "optimization_suggestions": [
                    "Use cinematic 4K footage for maximum visual impact",
                    "Implement smooth transitions between scenes",
                    "Add atmospheric background music",
                    "Include text overlays for key points",
                    "Use color grading for dramatic effect",
                ],
                "subtitle_optimization": {
                    "English": "Primary language, optimize for clarity",
                    "Spanish": "Latin American and European Spanish",
                    "French": "International French with clear pronunciation",
                    "German": "Standard German with proper grammar",
                },
                "self_improvement": {
                    "extra_sentences_generated": total_sentences
                    - len(script_data.get("script", [])),
                    "duration_improvement": f"{estimated_duration}s (target: 600s+)",
                    "quality_enhancement": "Ollama-powered content expansion",
                },
            }

            script_data["enhanced_metadata"] = enhanced_metadata
            self.log_message(
                f"Enhanced script metadata added for {total_sentences} sentences",
                "INFO",
            )

            return script_data

        except Exception as e:
            self.log_message(f"Error in enhance_script_with_metadata: {e}", "ERROR")
            return script_data

    def _generate_extra_sentences_with_ollama(
        self, script_data: Dict[str, Any], current_duration: float
    ) -> List[Dict[str, Any]]:
        """Use Ollama to generate extra sentences for low duration scripts.

        Args:
            script_data: Current script data
            current_duration: Current script duration in seconds

        Returns:
            List of extra sentences to add
        """
        try:
            target_duration = 600  # 10 minutes minimum
            extra_duration_needed = target_duration - current_duration
            extra_sentences_needed = int(
                extra_duration_needed / 8
            )  # 8 seconds per sentence

            if extra_sentences_needed <= 0:
                return []

            video_title = script_data.get("video_title", "Unknown")

            prompt = f"""Generate {extra_sentences_needed} extra sentences for low duration.

Video title: {video_title}
Current duration: {current_duration:.1f} seconds
Target duration: {target_duration} seconds
Extra sentences needed: {extra_sentences_needed}

Generate sentences to add to the end of the current script.
Each sentence should:
- Continue the existing story
- Take 8 seconds
- Be visually rich
- Contain engagement hooks

Return in JSON format:
            {{
              "extra_sentences": [
                {{
                  "sentence": "Extra sentence text",
                  "visual_query": "cinematic 4K [scene]",
                  "timing_seconds": {current_duration + 8},
                  "engagement_hook": "Hook description"
                }}
              ]
            }}"""

            result = self._get_ollama_response(prompt)
            if result and "extra_sentences" in result:
                extra_sentences = result["extra_sentences"]
                self.log_message(
                    f"Ollama generated {len(extra_sentences)} extra sentences", "INFO"
                )
                return extra_sentences
            else:
                self.log_message(
                    "Failed to generate extra sentences with Ollama", "WARNING"
                )
                return []

        except Exception as e:
            self.log_message(f"Error generating extra sentences: {e}", "ERROR")
            return []

    def _get_ollama_client(self):
        """Get Ollama client wrapper for the new JSON parsing system"""

        class OllamaClient:
            def chat(self, prompt):
                response = ollama.chat(
                    model=settings.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return type(
                    "Response",
                    (),
                    {"text": response.get("message", {}).get("content", "")},
                )()

        return OllamaClient()

    def _generate_video_plan_with_retry(self, niche: str, topic: str, llm) -> VideoPlan:
        """Generate video plan using robust JSON parsing with retry logic"""
        BASE_PROMPT = f"""
You are a disciplined planner. Return ONLY minified JSON, no commentary, no markdown.

STRICT SCHEMA (keys and types must match exactly):
{json.dumps(JSON_SCHEMA_HINT)}

Rules:
- If you need ranges, use strings like "12-15" (never 12-15 without quotes)
- Do not include code fences unless asked, but JSON is allowed in a ```json block too.
- No trailing commas. No extra keys. English only.

Now produce the object for niche="{niche}", topic="{topic}".
"""
        last_err = None
        for attempt in range(1, 4):
            resp = llm.chat(
                BASE_PROMPT
                if attempt == 1
                else BASE_PROMPT + f"\nATTEMPT {attempt}: Ensure strict JSON."
            )
            raw = resp.text if hasattr(resp, "text") else str(resp)

            try:
                jtxt = _extract_json_block(raw)
                data = _force_json(jtxt)
                # minimal validation
                assert isinstance(data.get("video_title"), str)
                assert isinstance(data.get("target_duration_minutes"), str)
                assert isinstance(data.get("word_count"), int)
                assert isinstance(data.get("script_sections"), dict)
                return VideoPlan(**data)
            except Exception as e:
                last_err = e
                time.sleep(0.8)
                continue
        raise ValueError(f"LLM JSON parse failed after retries: {last_err}")


# Convenience functions for backward compatibility
def generate_viral_ideas(
    channel_name: str, idea_count: int = 1
) -> Optional[List[Dict[str, Any]]]:
    """Backward compatibility function for generating viral ideas.

    Args:
        channel_name: Name of the YouTube channel
        idea_count: Number of ideas to generate

    Returns:
        List of viral video ideas or None if generation fails
    """
    handler = ImprovedLLMHandler()
    return handler.generate_viral_ideas(channel_name, idea_count)


def write_script(
    video_idea: Dict[str, Any], channel_name: str
) -> Optional[Dict[str, Any]]:
    """Backward compatibility function for writing scripts.

    Args:
        video_idea: Video idea dictionary
        channel_name: Name of the YouTube channel

    Returns:
        Generated script dictionary or None if generation fails
    """
    handler = ImprovedLLMHandler()
    script = handler.write_script(video_idea, channel_name)
    if script:
        return handler.enhance_script_with_metadata(script)
    return None


if __name__ == "__main__":
    # Test the improved handler
    print("Testing Improved LLM Handler...")

    handler = ImprovedLLMHandler()

    # Test viral ideas generation
    print("\nTesting viral ideas generation...")
    ideas = handler.generate_viral_ideas("test_channel", 2)
    if ideas:
        print(f"Generated {len(ideas)} viral ideas")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i}. {idea.get('title', 'No title')}")

    # Test script generation if ideas exist
    if ideas:
        print("\nTesting script generation...")
        script = handler.write_script(ideas[0], "test_channel")
        if script:
            print(f"Generated script with {len(script.get('script', []))} sentences")
            print(
                f"   Enhanced metadata: {script.get('enhanced_metadata', {}).get('estimated_duration_minutes', 'N/A')} minutes"
            )

    print("\nImproved LLM Handler test completed!")
