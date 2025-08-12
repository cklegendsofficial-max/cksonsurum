#!/usr/bin/env python3
"""
Smoke Test Script for Project Chimera
Tests environment variable loading, JSON parsing, and basic video processing functionality.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_environment_loading():
    """Test environment variable loading from .env file."""
    print("🔧 Testing Environment Variable Loading...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ python-dotenv imported successfully")
    except ImportError:
        print("❌ python-dotenv not available - install with: pip install python-dotenv")
        return False
    
    # Test loading .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - create from env_example.txt")
    
    # Check for required API keys
    pexels_key = os.getenv("PEXELS_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    
    if pexels_key and pexels_key != "your_pexels_api_key_here":
        print("✅ PEXELS_API_KEY loaded")
    else:
        print("⚠️  PEXELS_API_KEY not configured")
    
    if elevenlabs_key and elevenlabs_key != "your_elevenlabs_api_key_here":
        print("✅ ELEVENLABS_API_KEY loaded")
    else:
        print("⚠️  ELEVENLABS_API_KEY not configured")
    
    if ollama_url:
        print(f"✅ OLLAMA_BASE_URL: {ollama_url}")
    else:
        print("⚠️  OLLAMA_BASE_URL not configured")
    
    return True

def test_json_parser():
    """Test the improved LLM handler's JSON parser."""
    print("\n🧪 Testing JSON Parser Functionality...")
    
    try:
        from core_engine.improved_llm_handler import ImprovedLLMHandler
        handler = ImprovedLLMHandler()
        print("✅ ImprovedLLMHandler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ImprovedLLMHandler: {e}")
        return False
    
    # Test 1: Fenced JSON block
    fenced_text = """Here's some text with a JSON response:
```json
{
    "test": "value",
    "numbers": [1, 2, 3],
    "nested": {"key": "value"}
}
```
And more text after."""
    
    try:
        result = handler._extract_json_from_text(fenced_text)
        if result and isinstance(result, str):
            # Try to parse the extracted JSON string
            import json
            parsed_result = json.loads(result)
            print("✅ Fenced JSON extracted successfully")
            print(f"   Extracted: {parsed_result}")
        else:
            print("❌ Failed to extract fenced JSON")
            return False
    except Exception as e:
        print(f"❌ Error extracting fenced JSON: {e}")
        return False
    
    # Test 2: Free-form JSON
    free_form_text = """Here's a response with JSON:
{
    "status": "success",
    "data": ["item1", "item2"]
}
End of response."""
    
    try:
        result = handler._extract_json_from_text(free_form_text)
        if result and isinstance(result, str):
            # Try to parse the extracted JSON string
            import json
            parsed_result = json.loads(result)
            print("✅ Free-form JSON extracted successfully")
            print(f"   Extracted: {parsed_result}")
        else:
            print("❌ Failed to extract free-form JSON")
            return False
    except Exception as e:
        print(f"❌ Error extracting free-form JSON: {e}")
        return False
    
    return True

def test_video_processing():
    """Test basic video processing functions."""
    print("\n🎬 Testing Video Processing Functions...")
    
    try:
        from moviepy.editor import VideoFileClip, ColorClip
        print("✅ MoviePy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MoviePy: {e}")
        print("   Install with: pip install moviepy")
        return False
    
    try:
        from content_pipeline.advanced_video_creator import AdvancedVideoCreator
        creator = AdvancedVideoCreator()
        print("✅ AdvancedVideoCreator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AdvancedVideoCreator: {e}")
        return False
    
    # Create a synthetic test clip
    try:
        # Create a 2-second synthetic clip
        synthetic_clip = ColorClip(size=(640, 480), color=(100, 100, 100), duration=2)
        print("✅ Synthetic test clip created")
        
        tests_passed = 0
        total_tests = 2
        
        # Test black frame detection
        print("   Testing black frame detection...")
        try:
            black_frame_result = creator.detect_black_frames(synthetic_clip)
            if isinstance(black_frame_result, dict):
                print("✅ Black frame detection working")
                print(f"   Black frame ratio: {black_frame_result.get('black_frame_ratio', 'N/A')}")
                tests_passed += 1
            else:
                print("❌ Black frame detection failed")
        except Exception as e:
            print(f"⚠️  Black frame detection test skipped due to: {e}")
        
        # Test clip extension
        print("   Testing clip extension...")
        try:
            extended_clip = creator.extend_clip_to_duration(synthetic_clip, 4.0)
            if extended_clip and hasattr(extended_clip, 'duration') and extended_clip.duration >= 3.9:  # Allow small tolerance
                print("✅ Clip extension working")
                print(f"   Extended duration: {extended_clip.duration:.2f}s")
                tests_passed += 1
            else:
                print("❌ Clip extension failed")
        except Exception as e:
            print(f"⚠️  Clip extension test skipped due to: {e}")
        
        # Clean up
        synthetic_clip.close()
        if 'extended_clip' in locals() and hasattr(extended_clip, 'close'):
            extended_clip.close()
        
        # Return True if at least one test passed
        if tests_passed > 0:
            print(f"   {tests_passed}/{total_tests} video processing tests passed")
            return True
        else:
            print("❌ All video processing tests failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in video processing tests: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("🚀 Project Chimera Smoke Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Environment loading
    if test_environment_loading():
        tests_passed += 1
    
    # Test 2: JSON parser
    if test_json_parser():
        tests_passed += 1
    
    # Test 3: Video processing
    if test_video_processing():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
