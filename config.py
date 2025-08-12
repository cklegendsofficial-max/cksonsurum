# config_new.py - Enhanced Configuration with Environment Variables and Validation

import os
import json
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Multi-Channel niche normalization & rich seeds ===

# Kanal adları / niş isimleri eşlemesi (case-insensitive)
NICHE_ALIASES = {
    "cklegends": "history",
    "ckironwill": "motivation",
    "ckfinancecore": "finance",
    "ckdrive": "automotive",
    "ckcombat": "combat",
}

def normalize_niche(niche_or_channel: str) -> str:
    key = (niche_or_channel or "").strip().lower()
    # önce direkt niş adı olabilir
    if key in {"history","motivation","finance","automotive","combat"}:
        return key
    # kanal adı olabilir
    return NICHE_ALIASES.get(key, "history")

# Tier1/Tier2 geo ve timeframe setleri (varsa mevcutlarınızla birleştirin/tekilleştirin)
TIER1_GEOS = ["US","GB","CA","AU","DE","FR","IT","ES","NL","SE","NO","DK","CH"]
TIER2_GEOS = ["BR","MX","IN","PL","TR","RU","ID","MY","TH","ZA"]
DEFAULT_TIMEFRAMES = ["now 7-d","today 1-m","today 3-m"]
MAX_TOPICS = 24

# Tüm kanallar için zengin seed listeleri (en az 24'er)
SEED_TOPICS = {
    "history": [
        "ancient civilizations","lost cities","roman empire","greek mythology",
        "egyptian pharaohs","archaeology discoveries","medieval knights",
        "viking history","mysteries of history","ancient inventions",
        "silk road","mesopotamia","maya civilization","pompeii","stonehenge",
        "alexander the great","genghis khan","byzantine empire","ottoman history","world war myths",
        "ancient engineering","forgotten languages","temples and ruins","artifact mysteries",
        "rosetta stone","hittite empire","indus valley"
    ],
    "motivation": [
        "discipline tips","productivity habits","mental toughness","morning routine",
        "goal setting","sports motivation","habit building","focus techniques",
        "mindset shift","success stories","growth mindset","overcoming procrastination",
        "dopamine detox","confidence building","resilience training","habit stacking",
        "time management","cold showers","gym motivation","study motivation",
        "deep work","atomic habits","self improvement","stoic principles",
        "consistency challenge","no excuses mindset"
    ],
    "finance": [
        "inflation explained","interest rates impact","recession signals","dividend investing",
        "index funds vs ETFs","value vs growth stocks","real estate vs stocks","emergency fund tips",
        "compound interest power","financial freedom steps","budgeting frameworks","credit score hacks",
        "side hustles 2025","ai stocks outlook","crypto regulation watch","gold vs bitcoin",
        "market bubble signs","earnings season guide","dollar cost averaging","tax optimization basics",
        "retirement planning 101","FIRE movement","risk management rules","portfolio rebalancing",
        "behavioral finance biases","hedge against inflation"
    ],
    "automotive": [
        "ev vs hybrid comparison","battery tech breakthroughs","solid state batteries","fast charging myths",
        "self driving levels","best sports cars 2025","affordable performance cars","car maintenance hacks",
        "engine types explained","turbo vs supercharger","aerodynamics basics","track day essentials",
        "car detailing secrets","resale value tips","car insurance tricks","winter driving tips",
        "top road trip cars","classic car legends","racing history moments","motorsport tech transfer",
        "ev charging etiquette","range anxiety fixes","home charger setup","hydrogen vs electric",
        "otonom sürüş güvenliği","infotainment comparisons"
    ],
    "combat": [
        "mma striking basics","wrestling takedown chains","bjj submissions explained","boxing footwork drills",
        "muay thai knees and elbows","counter punching theory","southpaw vs orthodox tactics","defense fundamentals",
        "conditioning for fighters","injury prevention tips","fight IQ examples","legendary comebacks",
        "greatest rivalries","weight cutting science","octagon control","ground and pound efficiency",
        "clinch fighting secrets","kick checking techniques","karate in mma","sambo influence on grappling",
        "daily training routine","fight camp nutrition","mental preparation","corner advice breakdown",
        "scoring criteria myths","judging controversies"
    ],
}

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigValidator:
    """Configuration validation and bounds checking"""
    
    @staticmethod
    def validate_quality_standards(config: Dict[str, Any]) -> None:
        """Validate quality standards configuration"""
        # 0-1 range validation
        for key in ['minimum_quality_score', 'scene_variety_threshold', 'engagement_score_threshold']:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise ConfigError(f"{key} must be between 0 and 1, got {value}")
        
        # Positive number validation
        for key in ['minimum_duration_minutes', 'target_fps']:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ConfigError(f"{key} must be positive, got {value}")
        
        # Reasonable maximum validation
        if 'target_fps' in config and config['target_fps'] > 120:
            raise ConfigError(f"target_fps {config['target_fps']} is unreasonably high")
        
        if 'minimum_duration_minutes' in config and config['minimum_duration_minutes'] > 60:
            raise ConfigError(f"minimum_duration_minutes {config['minimum_duration_minutes']} is unreasonably high")
    
    @staticmethod
    def validate_ai_config(config: Dict[str, Any]) -> None:
        """Validate AI configuration"""
        # Learning rate bounds
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr < 0 or lr > 1:
                raise ConfigError(f"learning_rate must be between 0 and 1, got {lr}")
        
        # Max iterations bounds
        if 'max_iterations' in config:
            max_iter = config['max_iterations']
            if not isinstance(max_iter, int) or max_iter < 1 or max_iter > 100:
                raise ConfigError(f"max_iterations must be between 1 and 100, got {max_iter}")
    
    @staticmethod
    def validate_self_update_config(config: Dict[str, Any]) -> None:
        """Validate self-update configuration"""
        # Allowed update frequencies
        allowed_frequencies = ['after_each_video', 'daily', 'weekly', 'monthly', 'never']
        if 'update_frequency' in config:
            freq = config['update_frequency']
            if freq not in allowed_frequencies:
                raise ConfigError(f"update_frequency must be one of {allowed_frequencies}, got {freq}")
    
    @staticmethod
    def validate_model_names(config: Dict[str, Any]) -> None:
        """Validate model names and patterns"""
        # Allowed Ollama models
        allowed_models = ['llama3:8b', 'llama3:70b', 'mistral:7b', 'codellama:7b', 'llama2:7b']
        if 'ollama_model' in config:
            model = config['ollama_model']
            if model not in allowed_models:
                raise ConfigError(f"ollama_model must be one of {allowed_models}, got {model}")

# Environment variable configuration with graceful degradation
def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with logging and graceful degradation"""
    value = os.getenv(key, default)
    
    if value is None and required:
        print(f"⚠️ Required environment variable {key} not found")
        return None
    elif value is None:
        print(f"ℹ️ Optional environment variable {key} not found, using default: {default}")
        return default
    else:
        print(f"✅ Environment variable {key} loaded successfully")
        return value

# API Keys from environment variables
PEXELS_API_KEY = get_env_var("PEXELS_API_KEY", None, required=False)
ELEVENLABS_API_KEY = get_env_var("ELEVENLABS_API_KEY", None, required=False)
ELEVENLABS_VOICE_ID = get_env_var("ELEVENLABS_VOICE_ID", None, required=False)
OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", "http://localhost:11434", required=False)

# Enhanced CHANNELS_CONFIG with environment variable support
CHANNELS_CONFIG = {
    "CKLegends": {
        "name": "CKLegends",
        "niche": "history",
        "niche_keywords": ["ancient mysteries", "historical discoveries", "archaeology", "mythology", "legends"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 15,
        "style_preference": "cinematic",
        "narrator_style": "morgan_freeman",
        "music_style": "epic_historical",
        "visual_style": "ancient_civilizations",
        "engagement_strategy": "mystery_cliffhangers",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.8
    },
    "CKIronWill": {
        "name": "CKIronWill",
        "niche": "motivation",
        "niche_keywords": ["motivation", "willpower", "personal development", "success stories", "inspiration"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 12,
        "style_preference": "inspirational",
        "narrator_style": "tony_robbins",
        "music_style": "uplifting_motivational",
        "visual_style": "achievement_success",
        "engagement_strategy": "emotional_peaks",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.85
    }
}

# Normalize channel keys (trim): avoid accidental whitespace/case drifts
CHANNELS_CONFIG = {k.strip(): v for k, v in CHANNELS_CONFIG.items()}

# AI Configuration with validation
AI_CONFIG = {
    "ollama_model": get_env_var("OLLAMA_MODEL", "llama3:8b", required=False),
    "ollama_base_url": OLLAMA_BASE_URL,
    "self_improvement_enabled": True,
    "code_generation_enabled": True,
    "config_update_enabled": True,
    "quality_analysis_enabled": True,
    "learning_rate": 0.1,
    "max_iterations": 5,
    "improvement_threshold": 0.1
}

# Quality Standards with validation
QUALITY_STANDARDS = {
    "minimum_duration_minutes": 10,
    "target_fps": 30,
    "target_resolution": "1920x1080",
    "target_codec": "libx264",
    "audio_codec": "aac",
    "minimum_quality_score": 0.7,
    "scene_variety_threshold": 0.6,
    "engagement_score_threshold": 0.75
}

# Pexels Configuration with graceful degradation
PEXELS_CONFIG = {
    "api_key": PEXELS_API_KEY,
    "base_url": "https://api.pexels.com",
    "search_endpoint": "/videos/search",
    "default_params": {
        "per_page": 1,
        "orientation": "landscape",
        "size": "large",
        "quality": "high"
    },
    "rate_limit": {
        "requests_per_hour": 200,
        "requests_per_day": 5000
    },
    "enabled": PEXELS_API_KEY is not None
}

# Self-Update Configuration with validation
SELF_UPDATE_CONFIG = {
    "enabled": True,
    "update_frequency": "after_each_video",
    "learning_metrics": [
        "video_quality_score",
        "engagement_metrics",
        "scene_variety_score",
        "audio_quality_score",
        "visual_quality_score"
    ],
    "improvement_strategies": [
        "enhance_visual_effects",
        "optimize_audio_processing",
        "improve_scene_transitions",
        "enhance_narration_style",
        "optimize_music_selection"
    ]
}

# Validate configurations
try:
    ConfigValidator.validate_quality_standards(QUALITY_STANDARDS)
    ConfigValidator.validate_ai_config(AI_CONFIG)
    ConfigValidator.validate_self_update_config(SELF_UPDATE_CONFIG)
    ConfigValidator.validate_model_names(AI_CONFIG)
    print("✅ All configuration validations passed")
except ConfigError as e:
    print(f"❌ Configuration validation failed: {e}")
    # Use safe defaults
    QUALITY_STANDARDS["minimum_quality_score"] = 0.5
    QUALITY_STANDARDS["target_fps"] = 25
    AI_CONFIG["ollama_model"] = "llama3:8b"
    print("⚠️ Using safe default values")

# Graceful degradation for missing API keys
if not PEXELS_API_KEY:
    print("⚠️ PEXELS_API_KEY not found - Pexels features will be disabled")
    PEXELS_CONFIG["enabled"] = False

if not ELEVENLABS_API_KEY:
    print("⚠️ ELEVENLABS_API_KEY not found - ElevenLabs features will be disabled")

if not OLLAMA_BASE_URL or OLLAMA_BASE_URL == "http://localhost:11434":
    print("ℹ️ Using default Ollama URL: http://localhost:11434")

print(f"✅ Configuration loaded successfully")
print(f"   Pexels enabled: {PEXELS_CONFIG['enabled']}")
print(f"   ElevenLabs enabled: {ELEVENLABS_API_KEY is not None}")
print(f"   Ollama model: {AI_CONFIG['ollama_model']}")

# Export configurations
__all__ = [
    'CHANNELS_CONFIG',
    'AI_CONFIG',
    'QUALITY_STANDARDS',
    'PEXELS_CONFIG',
    'SELF_UPDATE_CONFIG',
    'ConfigError',
    'ConfigValidator',
    'NICHE_ALIASES',
    'normalize_niche',
    'TIER1_GEOS',
    'TIER2_GEOS',
    'DEFAULT_TIMEFRAMES',
    'MAX_TOPICS',
    'SEED_TOPICS'
]
