# core_engine/llm_handler.py (Usta Yönetmen Sürümü)

import json
import os
import re
import sys
import time

import ollama


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import settings

    CHANNELS_CONFIG = settings.CHANNELS_CONFIG
except ImportError:
    pass


def _extract_json_from_text(text: str) -> str | None:
    # Bu yardımcı fonksiyon, LLM'in çıktısını temizler.
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _get_ollama_response(prompt_template: str, max_retries: int = 3, model: str = None):
    if model is None:
        try:
            from config import settings

            model = settings.OLLAMA_MODEL
        except ImportError:
            model = "llama3:8b"
    # Bu fonksiyon, LLM ile güvenli bir şekilde iletişim kurar.
    for attempt in range(max_retries):
        print(f"    - LLM ile iletişim denemesi ({attempt + 1}/{max_retries})...")
        try:
            response = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt_template}]
            )
            raw_text = response.get("message", {}).get("content")
            if not raw_text:
                time.sleep(1)
                continue
            clean_json_text = _extract_json_from_text(raw_text)
            if not clean_json_text:
                time.sleep(1)
                continue
            parsed_json = json.loads(clean_json_text)
            print(f"    + Attempt {attempt + 1} SUCCESS: Geçerli JSON alındı.")
            return parsed_json
        except Exception as e:
            print(f"      Attempt {attempt + 1} FAILED: Hata: {e}")
            time.sleep(1)
            continue
    print(f"--- Tüm ({max_retries}) deneme başarısız oldu. ---")
    return None


def generate_viral_ideas(channel_name: str, idea_count: int = 1) -> list | None:
    # Fikir üretimini daha odaklı hale getiriyoruz.
    channel_info = CHANNELS_CONFIG.get(channel_name, {})
    niche = channel_info.get("niche", "history")
    prompt = f"Generate {idea_count} viral video idea for a YouTube channel about '{niche}'. Focus on a deep, specific mystery or an untold story. Your response MUST be ONLY a single, valid JSON array of objects."
    return _get_ollama_response(prompt)


def write_script(video_idea: dict, channel_name: str) -> dict | None:
    """
    YAPAY ZEKAYI 10+ DAKİKA İÇİN CÜMLE CÜMLE SENARYO YAZMAYA ZORLAR.
    """
    prompt = f"""
    You are a master scriptwriter for a viral YouTube documentary. Your task is to write a highly detailed, long-form script for a 10-15 minute video on the topic: '{video_idea.get('title', 'N/A')}'.
    The script MUST be structured as a list of individual sentences. Each sentence will be a separate scene in the video.
    Generate at least 40 to 50 sentences to ensure the video is long enough.
    For each sentence, create a highly specific and visually interesting search query for Pexels.

    Your response MUST be ONLY a single, valid JSON object.

    # REQUIRED JSON FORMAT:
    {{
      "video_title": "{video_idea.get('title', 'N/A')}",
      "script": [
        {{ "sentence": "The first sentence of the narration.", "visual_query": "A specific Pexels query for the first sentence." }},
        {{ "sentence": "The second sentence of the narration.", "visual_query": "A specific Pexels query for the second sentence." }}
      ]
    }}
    """
    return _get_ollama_response(prompt)
