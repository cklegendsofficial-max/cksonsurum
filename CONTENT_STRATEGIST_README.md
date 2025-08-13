# ContentStrategist and ProductionCoordinator

This document describes the new refactored content generation system that separates script creation from production coordination.

## Overview

The refactoring introduces two new classes:

1. **ContentStrategist** - Handles script generation with structured JSON output
2. **ProductionCoordinator** - Manages production workflow and visual planning

## ContentStrategist Class

### Purpose
Generates structured scripts in JSON format with both script text and visual keywords for each scene.

### Key Features
- Generates 15-20 scenes for 10-15 minute videos
- Each scene contains 2-3 sentences of voiceover text
- Provides specific visual keywords for each scene
- Uses Ollama for AI-powered script generation
- Returns structured JSON instead of plain text

### JSON Output Format
```json
[
  {
    "script_text": "The first scene voiceover text goes here. This should be 2-3 sentences that flow naturally together.",
    "visual_keywords": "ancient roman soldiers marching, battlefield smoke, dramatic sunset lighting"
  },
  {
    "script_text": "The second scene voiceover text continues the story. Each scene should advance the narrative.",
    "visual_keywords": "glowing brain neurons, scientific laboratory, blue light effects"
  }
]
```

### Methods
- `write_script(video_idea, channel_name)` - Generates structured script JSON

## VisualAssetManager Class

### Purpose
Manages searching and downloading video assets from the Pexels API.

### Key Features
- Searches for videos using Pexels API
- Downloads high-quality video files
- Handles API authentication and error handling
- Provides video information and metadata
- Includes cleanup functionality for downloaded files

### Methods
- `search_videos(query, per_page=1)` - Searches for videos based on keywords
- `download_video(video_url, save_folder)` - Downloads video to specified folder
- `get_video_info(query)` - Gets detailed information about videos
- `cleanup_downloads(folder_path)` - Removes all downloaded video files

## ProductionCoordinator Class

### Purpose
Coordinates production workflow and creates visual plans from structured scripts.

### Key Features
- Processes JSON script data
- Creates detailed visual plans for each scene
- Provides production summaries and statistics
- **NEW**: Integrates with VisualAssetManager for actual video downloads
- **NEW**: Downloads video assets from Pexels API based on visual keywords

### Methods
- `create_visual_plan(script_json)` - Generates visual plan output (no downloads)
- `create_visual_plan_and_download_assets(script_json)` - **NEW**: Creates plan AND downloads videos
- `get_scene_summary(script_json)` - Returns script statistics and summary
- `get_download_folder()` - **NEW**: Returns path to downloaded assets folder
- `cleanup_assets()` - **NEW**: Removes all downloaded video files

## Integration with Main Workflow

### Before (Old System)
```python
# Old approach using improved_llm_handler
script_data = director.llm_handler.write_script(video_idea, channel)
```

### After (New System)
```python
# New approach using ContentStrategist and ProductionCoordinator
content_strategist = ContentStrategist()
script_json = content_strategist.write_script(video_idea, channel)

production_coordinator = ProductionCoordinator()
production_coordinator.create_visual_plan(script_json)
```

## Benefits of the New System

1. **Separation of Concerns** - Script generation is separate from production coordination
2. **Structured Data** - JSON format provides clear structure for each scene
3. **Visual Planning** - Dedicated visual keywords for each scene
4. **Extensibility** - Easy to add new production features
5. **Testing** - Each component can be tested independently

## Usage Example

```python
from content_strategist import ContentStrategist
from production_coordinator import ProductionCoordinator
from visual_asset_manager import VisualAssetManager

# Generate script
strategist = ContentStrategist()
video_idea = {"title": "The Mystery of Ancient Roman Engineering"}
script_json = strategist.write_script(video_idea, "history_channel")

# Create production plan with video downloads
pexels_api_key = "your_pexels_api_key_here"
coordinator = ProductionCoordinator(pexels_api_key=pexels_api_key)

# Download video assets for each scene
scene_assets = coordinator.create_visual_plan_and_download_assets(script_json)

# Get summary
summary = coordinator.get_scene_summary(script_json)
print(f"Total scenes: {summary['total_scenes']}")
print(f"Estimated duration: {summary['estimated_duration_minutes']:.1f} minutes")

# Clean up downloaded assets when done
coordinator.cleanup_assets()
```

## Testing

Run the test script to verify functionality:

```bash
python test_content_strategist.py
```

## Future Enhancements

- **Pexels/Pixabay Integration** - Implement actual API calls for visual assets
- **Scene Timing** - Add timing information for each scene
- **Music Suggestions** - Include music recommendations for each scene
- **Production Workflow** - Add more production coordination features

## File Structure

```
Project_Chimera/
├── content_strategist.py          # ContentStrategist class
├── production_coordinator.py      # ProductionCoordinator class
├── visual_asset_manager.py        # VisualAssetManager class
├── test_content_strategist.py     # Test script
├── test_visual_assets.py          # Visual assets test script
├── CONTENT_STRATEGIST_README.md   # This documentation
└── main.py                        # Updated main workflow
```
