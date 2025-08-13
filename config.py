# config.py - Centralized Configuration with Pydantic BaseSettings

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration using Pydantic BaseSettings"""

    # Output and Directory Settings
    OUTPUT_DIR: str = Field(
        default="outputs", description="Output directory for generated content"
    )

    # Ollama Configuration
    OLLAMA_URL: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    OLLAMA_MODEL: str = Field(
        default="llama3:8b", description="Default Ollama model to use"
    )

    # Whisper Configuration
    WHISPER_MODEL: str = Field(
        default="base", description="Whisper model size for transcription"
    )

    # Rendering Settings
    ALLOW_SILENT_RENDER: bool = Field(
        default=True, description="Allow silent rendering mode"
    )

    # Video Processing Settings
    FPS: int = Field(default=30, description="Default video FPS")
    VIDEO_CODEC: str = Field(default="libx264", description="Video codec for output")
    AUDIO_CODEC: str = Field(default="aac", description="Audio codec for output")
    BITRATE: str = Field(default="6M", description="Default video bitrate")

    # Language Tiers
    LANGS_TIER1: List[str] = Field(
        default=["en", "es", "pt", "fr", "de", "ja"],
        description="Primary target languages",
    )
    LANGS_TIER2: List[str] = Field(
        default=["tr", "ar", "hi", "ru", "it", "nl", "ko", "zh"],
        description="Secondary target languages",
    )

    # Random Seed
    SEED: int = Field(default=42, description="Random seed for reproducibility")

    # API Keys (optional)
    PEXELS_API_KEY: Optional[str] = Field(default=None, description="Pexels API key")
    ELEVENLABS_API_KEY: Optional[str] = Field(
        default=None, description="ElevenLabs API key"
    )
    ELEVENLABS_VOICE_ID: Optional[str] = Field(
        default=None, description="ElevenLabs voice ID"
    )

    # Quality Standards
    MINIMUM_QUALITY_SCORE: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality score threshold"
    )
    SCENE_VARIETY_THRESHOLD: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Scene variety threshold"
    )
    ENGAGEMENT_SCORE_THRESHOLD: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Engagement score threshold"
    )
    MINIMUM_DURATION_MINUTES: int = Field(
        default=10, gt=0, description="Minimum video duration in minutes"
    )
    TARGET_RESOLUTION: str = Field(
        default="1920x1080", description="Target video resolution"
    )

    # AI Configuration
    SELF_IMPROVEMENT_ENABLED: bool = Field(
        default=True, description="Enable AI self-improvement"
    )
    CODE_GENERATION_ENABLED: bool = Field(
        default=True, description="Enable code generation"
    )
    CONFIG_UPDATE_ENABLED: bool = Field(
        default=True, description="Enable config updates"
    )
    QUALITY_ANALYSIS_ENABLED: bool = Field(
        default=True, description="Enable quality analysis"
    )
    LEARNING_RATE: float = Field(
        default=0.1, ge=0.0, le=1.0, description="AI learning rate"
    )
    MAX_ITERATIONS: int = Field(
        default=5, ge=1, le=100, description="Maximum AI iterations"
    )
    IMPROVEMENT_THRESHOLD: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Improvement threshold"
    )

    # Self-Update Configuration
    SELF_UPDATE_ENABLED: bool = Field(
        default=True, description="Enable self-update functionality"
    )
    UPDATE_FREQUENCY: str = Field(
        default="after_each_video", description="Update frequency"
    )

    # Pexels Configuration
    PEXELS_BASE_URL: str = Field(
        default="https://api.pexels.com", description="Pexels API base URL"
    )
    PEXELS_SEARCH_ENDPOINT: str = Field(
        default="/videos/search", description="Pexels search endpoint"
    )
    PEXELS_REQUESTS_PER_HOUR: int = Field(
        default=200, description="Pexels rate limit per hour"
    )
    PEXELS_REQUESTS_PER_DAY: int = Field(
        default=5000, description="Pexels rate limit per day"
    )

    # Niche Configuration
    NICHE_ALIASES: Dict[str, str] = Field(
        default={
            "cklegends": "history",
            "ckironwill": "motivation",
            "ckfinancecore": "finance",
            "ckdrive": "automotive",
            "ckcombat": "combat",
        },
        description="Channel name to niche mapping",
    )

    # Geographic and Timeframe Settings
    TIER1_GEOS: List[str] = Field(
        default=[
            "US",
            "GB",
            "CA",
            "AU",
            "DE",
            "FR",
            "IT",
            "ES",
            "NL",
            "SE",
            "NO",
            "DK",
            "CH",
        ],
        description="Tier 1 geographic markets",
    )
    TIER2_GEOS: List[str] = Field(
        default=["BR", "MX", "IN", "PL", "TR", "RU", "ID", "MY", "TH", "ZA"],
        description="Tier 2 geographic markets",
    )
    DEFAULT_TIMEFRAMES: List[str] = Field(
        default=["now 7-d", "today 1-m", "today 3-m"],
        description="Default timeframes for trend analysis",
    )
    MAX_TOPICS: int = Field(
        default=24, description="Maximum number of topics to generate"
    )

    # Seed Topics for each niche
    SEED_TOPICS: Dict[str, List[str]] = Field(
        default={
            "history": [
                "ancient civilizations",
                "lost cities",
                "roman empire",
                "greek mythology",
                "egyptian pharaohs",
                "archaeology discoveries",
                "medieval knights",
                "viking history",
                "mysteries of history",
                "ancient inventions",
                "silk road",
                "mesopotamia",
                "maya civilization",
                "pompeii",
                "stonehenge",
                "alexander the great",
                "genghis khan",
                "byzantine empire",
                "ottoman history",
                "world war myths",
                "ancient engineering",
                "forgotten languages",
                "temples and ruins",
                "artifact mysteries",
                "rosetta stone",
                "hittite empire",
                "indus valley",
            ],
            "motivation": [
                "discipline tips",
                "productivity habits",
                "mental toughness",
                "morning routine",
                "goal setting",
                "sports motivation",
                "habit building",
                "focus techniques",
                "mindset shift",
                "success stories",
                "growth mindset",
                "overcoming procrastination",
                "dopamine detox",
                "confidence building",
                "resilience training",
                "habit stacking",
                "time management",
                "cold showers",
                "gym motivation",
                "study motivation",
                "deep work",
                "atomic habits",
                "self improvement",
                "stoic principles",
                "consistency challenge",
                "no excuses mindset",
            ],
            "finance": [
                "inflation explained",
                "interest rates impact",
                "recession signals",
                "dividend investing",
                "index funds vs ETFs",
                "value vs growth stocks",
                "real estate vs stocks",
                "emergency fund tips",
                "compound interest power",
                "financial freedom steps",
                "budgeting frameworks",
                "credit score hacks",
                "side hustles 2025",
                "ai stocks outlook",
                "crypto regulation watch",
                "gold vs bitcoin",
                "market bubble signs",
                "earnings season guide",
                "dollar cost averaging",
                "tax optimization basics",
                "retirement planning 101",
                "FIRE movement",
                "risk management rules",
                "portfolio rebalancing",
                "behavioral finance biases",
                "hedge against inflation",
            ],
            "automotive": [
                "ev vs hybrid comparison",
                "battery tech breakthroughs",
                "solid state batteries",
                "fast charging myths",
                "self driving levels",
                "best sports cars 2025",
                "affordable performance cars",
                "car maintenance hacks",
                "engine types explained",
                "turbo vs supercharger",
                "aerodynamics basics",
                "track day essentials",
                "car detailing secrets",
                "resale value tips",
                "car insurance tricks",
                "winter driving tips",
                "top road trip cars",
                "classic car legends",
                "racing history moments",
                "motorsport tech transfer",
                "ev charging etiquette",
                "range anxiety fixes",
                "home charger setup",
                "hydrogen vs electric",
                "otonom sürüş güvenliği",
                "infotainment comparisons",
            ],
            "combat": [
                "mma striking basics",
                "wrestling takedown chains",
                "bjj submissions explained",
                "boxing footwork drills",
                "muay thai knees and elbows",
                "counter punching theory",
                "southpaw vs orthodox tactics",
                "defense fundamentals",
                "conditioning for fighters",
                "injury prevention tips",
                "fight IQ examples",
                "legendary comebacks",
                "greatest rivalries",
                "weight cutting science",
                "octagon control",
                "ground and pound efficiency",
                "clinch fighting secrets",
                "kick checking techniques",
                "karate in mma",
                "sambo influence on grappling",
                "daily training routine",
                "fight camp nutrition",
                "mental preparation",
                "corner advice breakdown",
                "scoring criteria myths",
                "judging controversies",
            ],
        },
        description="Seed topics for each niche",
    )

    # Centralized channel configuration
    CHANNELS: Dict[str, Dict[str, str]] = Field(
        default={
            "CKLegends": {"niche": "history"},
            "CKIronWill": {"niche": "motivation"},
            "CKFinanceCore": {"niche": "finance"},
            "CKDrive": {"niche": "automotive"},
            "CKCombat": {"niche": "combat"},
        },
        description="Centralized channel configuration with niches",
    )

    DEFAULT_CHANNELS: List[str] = Field(
        default=["CKLegends", "CKIronWill", "CKFinanceCore", "CKDrive", "CKCombat"],
        description="Default list of available channels",
    )

    # Channel-specific configurations (extended)
    CHANNELS_CONFIG: Dict[str, Dict[str, Any]] = Field(
        default={
            "CKLegends": {
                "name": "CKLegends",
                "niche": "history",
                "niche_keywords": [
                    "ancient mysteries",
                    "historical discoveries",
                    "archaeology",
                    "mythology",
                    "legends",
                ],
                "self_improvement": True,
                "self_update": True,
                "target_duration_minutes": 15,
                "style_preference": "cinematic",
                "narrator_style": "morgan_freeman",
                "music_style": "epic_historical",
                "visual_style": "ancient_civilizations",
                "engagement_strategy": "mystery_cliffhangers",
                "subtitle_languages": ["English", "Spanish", "French", "German"],
                "quality_threshold": 0.8,
            },
            "CKIronWill": {
                "name": "CKIronWill",
                "niche": "motivation",
                "niche_keywords": [
                    "motivation",
                    "willpower",
                    "personal development",
                    "success stories",
                    "inspiration",
                ],
                "self_improvement": True,
                "self_update": True,
                "target_duration_minutes": 12,
                "style_preference": "inspirational",
                "narrator_style": "tony_robbins",
                "music_style": "uplifting_motivational",
                "visual_style": "achievement_success",
                "engagement_strategy": "emotional_peaks",
                "subtitle_languages": ["English", "Spanish", "French", "German"],
                "quality_threshold": 0.85,
            },
            "CKFinanceCore": {
                "name": "CKFinanceCore",
                "niche": "finance",
                "niche_keywords": [
                    "investment strategies",
                    "market analysis",
                    "financial planning",
                    "wealth building",
                    "economic trends",
                ],
                "self_improvement": True,
                "self_update": True,
                "target_duration_minutes": 15,
                "style_preference": "professional",
                "narrator_style": "morgan_freeman",
                "music_style": "corporate_ambient",
                "visual_style": "financial_charts",
                "engagement_strategy": "data_insights",
                "subtitle_languages": ["English", "Spanish", "French", "German"],
                "quality_threshold": 0.8,
            },
            "CKDrive": {
                "name": "CKDrive",
                "niche": "automotive",
                "niche_keywords": [
                    "car reviews",
                    "automotive technology",
                    "racing",
                    "classic cars",
                    "car maintenance",
                ],
                "self_improvement": True,
                "self_update": True,
                "target_duration_minutes": 15,
                "style_preference": "dynamic",
                "narrator_style": "jeremy_clarkson",
                "music_style": "energetic_rock",
                "visual_style": "high_speed_action",
                "engagement_strategy": "thrilling_sequences",
                "subtitle_languages": ["English", "Spanish", "French", "German"],
                "quality_threshold": 0.8,
            },
            "CKCombat": {
                "name": "CKCombat",
                "niche": "combat",
                "niche_keywords": [
                    "martial arts",
                    "self defense",
                    "combat sports",
                    "military tactics",
                    "weapon training",
                ],
                "self_improvement": True,
                "self_update": True,
                "target_duration_minutes": 12,
                "style_preference": "intense",
                "narrator_style": "morgan_freeman",
                "music_style": "epic_battle",
                "visual_style": "action_sequences",
                "engagement_strategy": "adrenaline_rush",
                "subtitle_languages": ["English", "Spanish", "French", "German"],
                "quality_threshold": 0.85,
            },
        },
        description="Extended channel-specific configurations",
    )

    @field_validator("UPDATE_FREQUENCY")
    @classmethod
    def validate_update_frequency(cls, v: str) -> str:
        allowed = ["after_each_video", "daily", "weekly", "monthly", "never"]
        if v not in allowed:
            raise ValueError(f"update_frequency must be one of {allowed}")
        return v

    @field_validator("OLLAMA_MODEL")
    @classmethod
    def validate_ollama_model(cls, v: str) -> str:
        allowed = ["llama3:8b", "llama3:70b", "mistral:7b", "codellama:7b", "llama2:7b"]
        if v not in allowed:
            raise ValueError(f"ollama_model must be one of {allowed}")
        return v

    @field_validator("FPS")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        if v <= 0 or v > 120:
            raise ValueError("fps must be between 1 and 120")
        return v

    @field_validator("MINIMUM_DURATION_MINUTES")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v <= 0 or v > 60:
            raise ValueError("minimum_duration_minutes must be between 1 and 60")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create cached settings instance
@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create global settings instance
settings = get_settings()


# Helper functions
def normalize_niche(niche_or_channel: str) -> str:
    """Normalize niche or channel name to standard niche"""
    key = (niche_or_channel or "").strip().lower()
    # First check if it's already a niche name
    if key in {"history", "motivation", "finance", "automotive", "combat"}:
        return key
    # Then check if it's a channel name
    return settings.NICHE_ALIASES.get(key, "history")


# Export all settings and helper functions
__all__ = ["settings", "get_settings", "normalize_niche", "Settings"]

# Print configuration status on import
if __name__ == "__main__":
    print("✅ Configuration loaded successfully")
    print(f"   Output Directory: {settings.OUTPUT_DIR}")
    print(f"   Ollama URL: {settings.OLLAMA_URL}")
    print(f"   Ollama Model: {settings.OLLAMA_MODEL}")
    print(f"   Whisper Model: {settings.WHISPER_MODEL}")
    print(f"   FPS: {settings.FPS}")
    print(f"   Video Codec: {settings.VIDEO_CODEC}")
    print(f"   Audio Codec: {settings.AUDIO_CODEC}")
    print(f"   Bitrate: {settings.BITRATE}")
    print(f"   Seed: {settings.SEED}")
    print(f"   Pexels Enabled: {settings.PEXELS_API_KEY is not None}")
    print(f"   ElevenLabs Enabled: {settings.ELEVENLABS_API_KEY is not None}")
