# Enhanced Master Director - Video Pipeline Control Center

## 🚀 New Features

### 🎬 Advanced Video Analysis with MoviePy
- **Real Duration Calculation**: Uses MoviePy to extract actual video duration instead of simulated values
- **Visual Quality Analysis**: Analyzes frame variety using numpy to detect static/black frames
- **Audio Quality Assessment**: Evaluates audio levels and variety for comprehensive quality scoring
- **Black Frame Detection**: Automatically detects videos with >10% black frames for regeneration

### 🤖 AI-Powered Regeneration with Ollama
- **Smart Script Generation**: Uses Ollama to create new scripts for low-quality videos
- **Turkish Language Support**: Generates scripts in Turkish as requested
- **Quality-Based Prompts**: Creates targeted prompts based on specific video quality issues
- **Automated Workflow**: Integrates regeneration directly into the analysis pipeline

### 🎥 Visual Preview System
- **Frame Preview**: Display video frames using tkinter Canvas and MoviePy
- **Interactive Analysis**: Select videos from list and analyze in real-time
- **Quality Metrics Display**: Shows comprehensive analysis results with recommendations
- **Frame-by-Frame Analysis**: Navigate through video frames for detailed inspection

## 🚀 Sıfırdan Kur → Tek Komut

### 🎯 Tek Komut ile Tüm Akış
```bash
# Sadece bu komutu yazın:
python main.py --channel CKFinanceCore

# Sistem otomatik olarak:
# ✅ Topics üretir (24 konu + top 8 skorlu)
# ✅ Video render eder (ffmpeg kontrolü ile)
# ✅ Altyazı üretir (EN + 13 dil)
# ✅ Shorts oluşturur (4 horizontal + 3 vertical)
# ✅ Report.md oluşturur
# ✅ Metrics.jsonl kaydeder
```

### 🪟 Windows Hızlı Başlatma
```batch
# Batch dosyası ile:
run_finance_today.bat

# PowerShell ile:
.\run_all_channels.ps1
```

## 🔧 Installation

2. **Environment Setup** - Create environment configuration:
```bash
# Copy env_example.txt to .env and configure your API keys
cp env_example.txt .env
# Edit .env with your actual API keys
```

3. **Run Smoke Test** - Verify system functionality:
```bash
# Test environment loading, JSON parsing, and video processing
python scripts/smoke_test.py
```

4. **Code Quality Setup** - Install development tools:
```bash
# Install code formatting and linting tools
pip install -e ".[dev]"

# Format code with Black
black .

# Check code quality with Ruff
ruff check .

# Type checking with MyPy
mypy .
```

5. **TODO: Pre-commit Setup** - Automate code quality checks:
```bash
# Install pre-commit hooks for automated code quality
pip install pre-commit
pre-commit install

# This will automatically run Black, Ruff, and other checks before each commit
```

6. Ensure Ollama is running locally for AI features:
```bash
ollama serve
```

7. Run the enhanced frontend:
```bash
python frontend.py
```

## 📊 Quality Analysis Features

### Video Quality Metrics
- **Overall Quality Score**: Combined score from duration, visual, and audio analysis
- **Duration Score**: Normalized duration assessment (target: 60+ seconds)
- **Visual Score**: Frame variety analysis with static/black frame penalties
- **Audio Score**: Audio level and variety assessment
- **Black Frame Ratio**: Percentage of dark/black frames detected

### Automatic Quality Checks
- **Black Screen Prevention**: Flags videos with >10% black frames
- **Static Content Detection**: Identifies videos with low frame variety
- **Duration Validation**: Ensures videos meet minimum length requirements
- **Audio Quality Verification**: Checks for proper audio levels and variety

## 🎯 Regeneration Workflow

1. **Quality Analysis**: Automatically analyzes all videos in the pipeline
2. **Issue Detection**: Identifies videos below quality thresholds
3. **AI Script Generation**: Uses Ollama to create improved content
4. **Script Storage**: Saves new scripts in `regenerated_scripts/` folder
5. **Pipeline Integration**: Ready for integration with video creation pipeline

## 🎨 GUI Enhancements

### New Controls
- **Video Preview Button**: Opens interactive preview window
- **Enhanced Analysis**: Real-time quality metrics display
- **Frame Navigation**: Browse through video frames for detailed inspection
- **Quality Indicators**: Visual status for MoviePy and Ollama availability

### Preview Window Features
- **Video Selection**: Dropdown list of available videos
- **Canvas Display**: High-quality frame preview with aspect ratio handling
- **Analysis Results**: Comprehensive quality metrics and recommendations
- **Action Buttons**: Analyze, regenerate, and frame analysis options

## 🔍 Technical Implementation

## 📝 Code Style Standards

### Code Quality Tools
- **Black**: Automatic code formatting with 88-character line length
- **Ruff**: Fast Python linter with E, F, I, W, B, C4, UP rules
- **MyPy**: Static type checking for improved code quality
- **Pre-commit**: Automated hooks for consistent code quality

### Style Guidelines
- **Type Hints**: All public functions include proper type annotations
- **Logging**: Standardized levels (DEBUG, INFO, WARNING, ERROR)
- **Exception Handling**: Specific exception types, not generic `Exception`
- **Documentation**: Comprehensive docstrings with Args/Returns/Raises
- **Import Organization**: Standard library, third-party, local imports

### Example Code Style
```python
def process_video(
    video_path: str,
    quality: Optional[float] = None
) -> Dict[str, Any]:
    """Process video with quality enhancement.

    Args:
        video_path: Path to the video file
        quality: Optional quality threshold (0.0-1.0)

    Returns:
        Dictionary containing processing results

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If quality value is invalid
    """
    try:
        # Implementation here
        pass
    except FileNotFoundError as e:
        logger.error(f"Video file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid quality value: {e}")
        raise
```

### MoviePy Integration
- **Frame Extraction**: Efficient sampling of video frames for analysis
- **Memory Management**: Proper clip closing to prevent memory leaks
- **Error Handling**: Graceful fallback when MoviePy is unavailable
- **Performance Optimization**: Samples frames strategically for speed

### Ollama Integration
- **Local AI Processing**: Uses local Ollama instance for privacy
- **Turkish Language**: Generates content in Turkish as specified
- **Context-Aware Prompts**: Creates targeted prompts based on video analysis
- **Error Recovery**: Handles Ollama connection issues gracefully

### Frame Analysis
- **PIL Integration**: Uses Pillow for image processing and display
- **Canvas Rendering**: Efficient tkinter Canvas for frame display
- **Aspect Ratio Handling**: Maintains video proportions in preview
- **Memory Optimization**: Efficient image conversion and display

## 📁 File Structure

```
Project_Chimera/
├── main.py                  # Ana pipeline (tek komut ile çalışır)
├── run_finance_today.bat    # Windows batch (CKFinanceCore)
├── run_all_channels.ps1     # PowerShell (tüm kanallar)
├── frontend.py              # Enhanced GUI with new features
├── requirements.txt          # Dependencies list
├── README_ENHANCED.md       # This documentation
├── assets/
│   └── videos/              # Video files for analysis
└── regenerated_scripts/     # AI-generated scripts (created automatically)
```

## 📊 Pipeline Çıktıları

### 🎬 Video Çıktıları
```
outputs/<channel>/<YYYY-MM-DD>/
├── final_video.mp4          # Ana video (ffmpeg ile render)
├── captions/                # Altyazı dosyaları
│   ├── video.en.srt         # İngilizce (Whisper)
│   ├── video.es.srt         # İspanyolca (MarianMT/Ollama)
│   ├── video.tr.srt         # Türkçe
│   └── ... (13 dil)
└── shorts/                  # Kısa videolar
    ├── short_1.mp4          # 15s horizontal
    ├── short_2.mp4          # 30s horizontal
    ├── vshort_1.mp4         # 15s vertical (9:16)
    └── ... (7 short)
```

### 📈 Raporlar ve Metrikler
```
outputs/<channel>/<YYYY-MM-DD>/
├── report.md                # Pipeline özet raporu
├── metrics.jsonl            # Performans metrikleri
└── logs/                    # Detaylı log dosyaları
```

## 🚦 Usage Instructions

### 🎯 Tek Komut ile Tam Pipeline
```bash
# Sadece bu komutu yazın:
python main.py --channel CKFinanceCore

# Sistem otomatik olarak tüm adımları tamamlar:
# 1. Topics üretir (24 konu + top 8 skorlu)
# 2. Video render eder (ffmpeg kontrolü ile)
# 3. Altyazı üretir (EN + 13 dil çevirisi)
# 4. Shorts oluşturur (4 horizontal + 3 vertical)
# 5. Report.md oluşturur
# 6. Metrics.jsonl kaydeder
```

### 🪟 Windows Hızlı Başlatma
```batch
# Tek kanal için:
run_finance_today.bat

# Tüm kanallar için:
.\run_all_channels.ps1

# Dry-run (test) için:
.\run_all_channels.ps1 -DryRun
```

### 🔧 Detaylı Kontrol
```bash
# Sadece belirli adımlar:
python main.py --channel CKFinanceCore --steps topics render

# Belirli tarih için:
python main.py --channel CKFinanceCore --date 2025-01-15

# Dry-run (dosya yazmadan test):
python main.py --channel CKFinanceCore --dry-run
```

### Basic Video Analysis

### Video Preview and Analysis
1. Click "🎬 Video Preview" to open the preview window
2. Select a video from the list to preview frames
3. Click "🔍 Analyze Selected" for detailed quality analysis
4. Use "📊 Show Frames" for frame-by-frame inspection

### AI Regeneration
1. Ensure Ollama is running locally
2. Use regeneration features to create improved content
3. Check `regenerated_scripts/` folder for new scripts
4. Integrate with video creation pipeline as needed

## 🔧 Configuration

### Quality Thresholds
- **Minimum Quality Score**: 0.7 (configurable in QUALITY_STANDARDS)
- **Black Frame Threshold**: 10% (configurable in code)
- **Duration Target**: 60+ seconds for optimal scoring

### Ollama Settings
- **Model**: llama2 (configurable in `_generate_ollama_script`)
- **Language**: Turkish (as specified in prompts)
- **Context**: Video-specific quality issues and recommendations

## 🐛 Troubleshooting

### Common Issues
- **MoviePy Not Available**: Install with `pip install moviepy`
- **Ollama Connection**: Ensure Ollama service is running locally
- **Memory Issues**: Videos are automatically closed after analysis
- **Frame Display**: Check PIL/Pillow installation for preview features

### Performance Tips
- **Large Videos**: Analysis samples frames strategically for speed
- **Memory Usage**: Videos are processed and closed immediately
- **Preview Loading**: Frames are loaded on-demand for efficiency

## 🔮 Future Enhancements

- **Real-time Video Processing**: Live quality monitoring during creation
- **Advanced AI Models**: Support for multiple Ollama models
- **Batch Processing**: Parallel analysis of multiple videos
- **Quality History**: Track quality improvements over time
- **Export Reports**: Generate detailed quality analysis reports

## 📞 Support

For issues or questions about the enhanced features:
1. Check the logs for detailed error messages
2. Verify all dependencies are properly installed
3. Ensure Ollama service is running for AI features
4. Check video file formats and accessibility
