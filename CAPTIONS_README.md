# 🎬 Automatic Multi-Language Captions System

This system automatically generates captions in 15+ languages for your videos using AI-powered transcription and translation.

## 🚀 Features

- **Automatic Transcription**: Uses OpenAI Whisper for high-quality English transcription
- **Multi-Language Translation**: Supports 15+ languages using MarianMT models
- **LLM Fallback**: Falls back to Ollama for translation if MarianMT fails
- **Smart Audio Extraction**: Automatically extracts audio from video files
- **SRT Format**: Generates standard SRT subtitle files
- **Pipeline Integration**: Seamlessly integrated into your video creation pipeline

## 🌍 Supported Languages

### Tier 1 (Primary)
- **en** - English (source)
- **es** - Spanish
- **pt** - Portuguese  
- **fr** - French
- **de** - German
- **ja** - Japanese

### Tier 2 (Secondary)
- **tr** - Turkish
- **ar** - Arabic
- **hi** - Hindi
- **ru** - Russian
- **it** - Italian
- **nl** - Dutch
- **ko** - Korean
- **zh** - Chinese

## 📦 Installation

### Quick Setup (Recommended)
```bash
python setup_captions.py
```

### Manual Installation

#### 1. Install Python Dependencies
```bash
pip install -r requirements_captions.txt
```

#### 2. Install ffmpeg
**Windows:**
- Download from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
- Extract and add `bin` folder to PATH

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

#### 3. Set Environment Variable (Optional)
```bash
# Windows
setx WHISPER_MODEL base

# macOS/Linux
export WHISPER_MODEL=base
```

## 🔧 Usage

### Basic Usage
```python
from auto_captions import generate_multi_captions

# Generate captions for all supported languages
captions = generate_multi_captions("video.mp4")
print(f"Generated {len(captions)} caption files")

# Generate captions for specific languages
specific_langs = ["es", "fr", "de"]
captions = generate_multi_captions("video.mp4", langs=specific_langs)
```

### Pipeline Integration
The system is automatically integrated into your main pipeline. After video rendering completes, captions are generated automatically.

### Manual Caption Generation
```python
from auto_captions import transcribe_to_srt, translate_srt

# Step 1: Generate English SRT
en_srt = transcribe_to_srt("video.mp4")
if en_srt:
    print(f"English SRT: {en_srt}")
    
    # Step 2: Translate to Spanish
    es_srt = translate_srt(en_srt, "es")
    if es_srt:
        print(f"Spanish SRT: {es_srt}")
```

## 📁 Output Files

For each video, the system generates:
```
video.mp4
├── video.srt          # English (source)
├── video.es.srt       # Spanish
├── video.fr.srt       # French
├── video.de.srt       # German
├── video.ja.srt       # Japanese
├── video.pt.srt       # Portuguese
├── video.tr.srt       # Turkish
├── video.ar.srt       # Arabic
├── video.hi.srt       # Hindi
├── video.ru.srt       # Russian
├── video.it.srt       # Italian
├── video.nl.srt       # Dutch
├── video.ko.srt       # Korean
└── video.zh.srt       # Chinese
```

## ⚙️ Configuration

### Environment Variables
- `WHISPER_MODEL`: Whisper model size (default: "base")
  - Options: "tiny", "base", "small", "medium", "large"
  - Larger models = better accuracy, slower processing

### Model Selection
The system automatically selects the best available translation method:
1. **MarianMT** (fastest, highest quality)
2. **Ollama LLM** (fallback, good quality)
3. **Copy** (last resort, no translation)

## 🔍 How It Works

### 1. Audio Extraction
- Automatically detects video files
- Extracts 16kHz mono WAV audio using ffmpeg
- Optimized for Whisper processing

### 2. Transcription
- Uses OpenAI Whisper for English transcription
- Generates SRT format with precise timestamps
- Handles various audio formats (WAV, MP3, M4A, FLAC)

### 3. Translation Pipeline
- **MarianMT**: Fast, high-quality neural translation
- **LLM Fallback**: Uses Ollama for complex translations
- **Copy Fallback**: Preserves original if translation fails

### 4. SRT Processing
- Preserves timing information
- Handles multi-line captions
- Maintains proper SRT formatting

## 🚨 Error Handling

The system is designed to be **non-blocking**:
- Missing dependencies → Logs warning, continues pipeline
- Transcription failure → Logs error, skips captions
- Translation failure → Falls back to copy
- File I/O errors → Logs error, continues

## 📊 Performance

### Processing Times (approximate)
- **Audio extraction**: 10-30 seconds (depends on video length)
- **Whisper transcription**: 1-3x real-time (depends on model size)
- **Translation**: 2-5 seconds per language (MarianMT)
- **LLM translation**: 10-30 seconds per language

### Resource Usage
- **Memory**: 2-4GB RAM (Whisper + translation models)
- **Storage**: 100-500MB (model cache)
- **GPU**: Optional acceleration for Whisper

## 🧪 Testing

### Test the System
```python
# Test basic functionality
python -c "
from auto_captions import generate_multi_captions
print('✅ Captions system imported successfully')
"

# Test with a sample video
python -c "
from auto_captions import transcribe_to_srt
result = transcribe_to_srt('test_video.mp4')
print(f'Transcription result: {result}')
"
```

### Verify Dependencies
```bash
python -c "
import whisper
import transformers
print('✅ All dependencies available')
"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "ffmpeg not found"
- Install ffmpeg and add to PATH
- Restart terminal/IDE after installation

#### 2. "whisper not available"
- Install: `pip install openai-whisper`
- Check Python environment

#### 3. "transformers import failed"
- Install: `pip install transformers sentencepiece`
- May require restart after installation

#### 4. "CUDA out of memory"
- Use smaller Whisper model: `set WHISPER_MODEL=tiny`
- Close other GPU applications

#### 5. "Translation failed"
- Check internet connection (for model downloads)
- Verify Ollama is running (for LLM fallback)

### Performance Optimization

#### For Faster Processing
```bash
# Use smaller Whisper model
set WHISPER_MODEL=tiny

# Use CPU-only processing
set CUDA_VISIBLE_DEVICES=""
```

#### For Better Quality
```bash
# Use larger Whisper model
set WHISPER_MODEL=medium

# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Future Enhancements

- [ ] Real-time caption generation
- [ ] Custom language models
- [ ] Caption quality scoring
- [ ] Automatic language detection
- [ ] Caption timing optimization
- [ ] Multi-format export (VTT, ASS)

## 🤝 Contributing

To improve the captions system:
1. Test with different video types
2. Report translation quality issues
3. Suggest new language support
4. Optimize performance bottlenecks

## 📄 License

This captions system is part of Project Chimera and follows the same license terms.

---

**🎯 Goal**: Make every video accessible to global audiences through high-quality, automatically generated captions in 15+ languages.
