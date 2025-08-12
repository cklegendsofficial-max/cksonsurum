# Logger and Metrics System

## Overview

The project now includes a comprehensive logging and metrics system that provides:
- **Rich Console Output**: Colorful, formatted console logging with Rich library
- **File Logging**: Timestamped log files (YYYYMMDD_HHMMSS.log)
- **Pipeline Metrics**: JSONL format metrics for each production run
- **Error Throttling**: Prevents spam of repeated error messages
- **Progress Tracking**: Visual progress bars for long-running operations

## Features

### 1. Enhanced Logger (`logger.py`)

#### Rich Console Output
- Color-coded log levels (INFO: blue, SUCCESS: green, WARNING: yellow, ERROR: red)
- Progress bars with spinners and time tracking
- Formatted tables for metrics and error summaries

#### File Logging
- Automatic log file creation with timestamp format: `YYYYMMDD_HHMMSS.log`
- Configurable log levels for console vs file
- UTF-8 encoding support

#### Error Throttling
- Prevents repeated error messages from flooding logs
- Configurable throttle time (default: 60 seconds)
- Error count tracking and summaries

### 2. Metrics Collection

#### Pipeline Steps Tracked
- **topics**: Topic generation and trending analysis
- **script**: Script writing and enhancement
- **assets**: Visual and audio asset collection
- **render**: Video rendering and compilation
- **captions**: Multi-language subtitle generation
- **score**: Quality scoring and analysis

#### Metrics Data Structure
```json
{
  "timestamp": "2025-08-12T14:04:58.035045",
  "step": "topics",
  "duration_ms": 150.5,
  "input_path": "input_trends.json",
  "output_path": "output_topics.json",
  "status": "success",
  "fallback": false,
  "channel": "TestChannel"
}
```

#### File Organization
```
outputs/
├── <kanal>/
│   └── <tarih>/
│       └── metrics.jsonl
```

### 3. Integration with Main Pipeline

The main pipeline automatically tracks metrics for each step:
- Each phase is timed using `time.monotonic()`
- Metrics are logged with input/output paths
- Fallback scenarios are tracked
- Channel-specific metrics are saved

## Usage

### Basic Logger Setup

```python
from logger import setup_logger

# Setup logger with custom name and log directory
logger = setup_logger("MyApp", "logs")

# Basic logging
logger.log_info("Starting process...")
logger.log_success("Process completed")
logger.log_warning("Something to watch out for")
logger.log_error("error_key", "Error message")
```

### Metrics Collection

```python
# Log a metric
logger.log_metric(
    step="render",
    duration_ms=5000.3,
    input_path="assets_folder",
    output_path="final_video.mp4",
    status="success",
    channel="CKLegends"
)

# Save metrics to file
metrics_file = logger.save_metrics("CKLegends", "20241201")
```

### Progress Tracking

```python
# Start progress bar
logger.start_progress("Processing items...", total=10)

# Update progress
for i in range(10):
    # Process item
    logger.update_progress()

# Stop progress
logger.stop_progress()
```

### Error Throttling

```python
# Errors are automatically throttled
for i in range(100):
    logger.log_error("api_error", "API rate limit exceeded")
    # Only first occurrence and every 10th will be logged
```

### Convenience Functions

```python
from logger import log_metric, log_info, log_success, log_warning, log_error

# Use global logger instance
log_info("Message")
log_metric("step", 150.5, status="success")
```

## Configuration

### Log Levels
- **Console**: INFO (default) - Shows important messages
- **File**: DEBUG (default) - Shows all messages for debugging

### Metrics Directory
- Default: `outputs/`
- Structure: `outputs/<channel>/<date>/metrics.jsonl`
- Date format: `YYYYMMDD`

### Error Throttling
- Default throttle time: 60 seconds
- Log every 10th occurrence after first
- Configurable via `ErrorThrottler` class

## Dependencies

Add to `requirements.txt`:
```
rich>=13.0.0  # For rich console output and progress bars
```

## Testing

### Manual Test
```bash
python logger.py
```

### Integration Test
```python
from main import EnhancedMasterDirector

# This will automatically use the enhanced logger
director = EnhancedMasterDirector()
```

## Output Examples

### Console Output
```
📝 Logging to: logs\20241201_143022.log
ℹ️ Enhanced logging system initialized
✅ Basic logging works
📊 topics: 150.5ms [success]
📊 script: 450.2ms [success]
  Processing items... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
```

### Metrics Summary Table
```
     Pipeline Metrics Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric              ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Steps         │ 6        │
│ Total Duration      │ 7802.5ms │
│ Fallbacks           │ 1        │
│ Step: topics        │ 1        │
│ Step: script        │ 1        │
│ Step: assets        │ 1        │
│ Step: render        │ 1        │
│ Step: captions      │ 1        │
│ Step: score         │ 1        │
└─────────────────────┴──────────┘
```

### Metrics JSONL File
```jsonl
{"timestamp": "2025-08-12T14:04:58.035045", "step": "topics", "duration_ms": 150.5, "input_path": "input_trends.json", "output_path": "output_topics.json", "status": "success", "fallback": false, "channel": "TestChannel"}
{"timestamp": "2025-08-12T14:04:58.137369", "step": "script", "duration_ms": 450.2, "input_path": "topics.json", "output_path": "script.json", "status": "success", "fallback": false, "channel": "TestChannel"}
```

## Benefits

1. **Performance Monitoring**: Track execution time for each pipeline step
2. **Debugging**: Detailed logs with timestamps and context
3. **Quality Assurance**: Monitor fallback usage and error rates
4. **Production Insights**: Analyze pipeline performance over time
5. **User Experience**: Rich console output with progress indicators
6. **Data Analysis**: Structured metrics for performance analysis

## Troubleshooting

### Common Issues

1. **Rich Library Not Available**: Falls back to basic logging
2. **Log Directory Permissions**: Ensure write access to log directory
3. **Metrics File Creation**: Check output directory permissions
4. **Import Errors**: Verify logger.py is in Python path

### Debug Mode

```python
# Enable debug logging
logger = setup_logger("DebugApp", "logs", console_level=logging.DEBUG)
```

### Custom Metrics

```python
# Add custom fields to metrics
logger.log_metric(
    step="custom_step",
    duration_ms=100.0,
    custom_field="custom_value",
    extra_data={"key": "value"}
)
```

## Future Enhancements

- **Metrics Aggregation**: Daily/weekly/monthly summaries
- **Performance Alerts**: Threshold-based notifications
- **Metrics Dashboard**: Web-based visualization
- **Export Formats**: CSV, Excel, JSON export options
- **Real-time Monitoring**: Live metrics streaming
