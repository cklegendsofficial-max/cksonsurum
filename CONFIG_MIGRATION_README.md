# Configuration System Migration Guide

## Overview

The configuration system has been completely rewritten using **Pydantic BaseSettings** to provide:
- Centralized configuration management
- Environment variable overrides
- Type validation and constraints
- Cached settings for performance

## Key Changes

### 1. New Configuration Structure

All configuration is now accessed through `config.settings`:

```python
from config import settings

# Access any configuration value
output_dir = settings.OUTPUT_DIR
fps = settings.FPS
ollama_model = settings.OLLAMA_MODEL
```

### 2. Environment Variable Support

Create a `.env` file in your project root to override defaults:

```bash
# .env
OUTPUT_DIR=builds
FPS=60
VIDEO_CODEC=h264
OLLAMA_MODEL=llama3:70b
```

### 3. Available Configuration Options

#### Core Settings
- `OUTPUT_DIR`: Output directory (default: "outputs")
- `OLLAMA_URL`: Ollama server URL (default: "http://localhost:11434")
- `OLLAMA_MODEL`: Ollama model (default: "llama3:8b")
- `WHISPER_MODEL`: Whisper model size (default: "base")
- `ALLOW_SILENT_RENDER`: Silent rendering mode (default: True)

#### Video Processing
- `FPS`: Video frame rate (default: 30)
- `VIDEO_CODEC`: Video codec (default: "libx264")
- `AUDIO_CODEC`: Audio codec (default: "aac")
- `BITRATE`: Video bitrate (default: "6M")

#### Language Support
- `LANGS_TIER1`: Primary languages (default: ["en", "es", "pt", "fr", "de", "ja"])
- `LANGS_TIER2`: Secondary languages (default: ["tr", "ar", "hi", "ru", "it", "nl", "ko", "zh"])

#### Quality & AI
- `SEED`: Random seed (default: 42)
- `MINIMUM_QUALITY_SCORE`: Quality threshold (default: 0.7)
- `SELF_IMPROVEMENT_ENABLED`: AI self-improvement (default: True)

## Migration Steps

### 1. Update Imports

**Before:**
```python
from config import AI_CONFIG, CHANNELS_CONFIG, PEXELS_API_KEY
model = AI_CONFIG.get("ollama_model", "llama3:8b")
```

**After:**
```python
from config import settings
model = settings.OLLAMA_MODEL
```

### 2. Update Configuration Access

**Before:**
```python
FPS = 30
CODEC = 'libx264'
AUDIO_CODEC = 'aac'
```

**After:**
```python
FPS = settings.FPS
CODEC = settings.VIDEO_CODEC
AUDIO_CODEC = settings.AUDIO_CODEC
```

### 3. Update Channel Configuration

**Before:**
```python
channels = list(CHANNELS_CONFIG.keys())
```

**After:**
```python
channels = list(settings.CHANNELS_CONFIG.keys())
```

## Testing

### Basic Configuration Test
```bash
python -c "import config; print(config.settings.OUTPUT_DIR)"
# Output: outputs
```

### Environment Variable Override Test
```bash
# Windows PowerShell
$env:OUTPUT_DIR="builds"
python -c "import config; print(config.settings.OUTPUT_DIR)"
# Output: builds

# Linux/macOS
export OUTPUT_DIR=builds
python -c "import config; print(config.settings.OUTPUT_DIR)"
# Output: builds
```

### Full Configuration Display
```bash
python config.py
```

## Benefits

1. **Centralized Management**: All settings in one place
2. **Environment Overrides**: Easy configuration per environment
3. **Type Safety**: Pydantic validation prevents invalid values
4. **Performance**: Cached settings with `@lru_cache`
5. **Documentation**: Self-documenting configuration with descriptions
6. **Validation**: Automatic bounds checking and constraint validation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `pydantic` and `pydantic-settings` are installed
2. **Validation Errors**: Check that environment variable values match expected types
3. **Cache Issues**: Restart Python process if settings don't update

### Dependencies

Add to `requirements.txt`:
```
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

## Examples

### Custom Configuration
```python
from config import get_settings

# Get fresh settings (bypasses cache)
custom_settings = get_settings()
print(custom_settings.OUTPUT_DIR)
```

### Environment-Specific Configs
```bash
# Development
OUTPUT_DIR=dev_outputs
FPS=24
VIDEO_CODEC=h264

# Production
OUTPUT_DIR=prod_outputs
FPS=30
VIDEO_CODEC=libx264
```

### Programmatic Override
```python
import os
os.environ["OUTPUT_DIR"] = "custom_outputs"

from config import get_settings
settings = get_settings()
print(settings.OUTPUT_DIR)  # custom_outputs
```

## Support

For issues or questions about the new configuration system:
1. Check the validation error messages
2. Verify environment variable syntax
3. Ensure all required dependencies are installed
4. Test with minimal configuration first
