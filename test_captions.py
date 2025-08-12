#!/usr/bin/env python3
"""
Quick test script for the captions system
"""

import logging
from auto_captions import generate_multi_captions, TARGET_LANGS

# Set up logging to see all messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_captions_system():
    """Test the captions system with a non-existent video file."""
    print("🎬 Captions System Test")
    print("=" * 40)
    
    print(f"✅ System imported successfully")
    print(f"🌍 Supported languages: {len(TARGET_LANGS)}")
    print(f"   Tier 1: {TARGET_LANGS[:6]}")
    print(f"   Tier 2: {TARGET_LANGS[6:]}")
    print()
    
    # Test with non-existent file (expected behavior)
    print("🧪 Testing with non-existent video file...")
    test_video = "test_video.mp4"
    
    try:
        result = generate_multi_captions(test_video, audio_path=None)
        print(f"📊 Result: {len(result)} caption files generated")
        
        if result:
            print("✅ SUCCESS: Captions generated!")
            for caption in result:
                print(f"   📝 {caption}")
        else:
            print("⚠️  No captions generated (expected for non-existent file)")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()
    print("📋 Expected behavior:")
    print("   - If Whisper installed: 'CAPTIONS: generated 15 files...'")
    print("   - If Whisper missing: 'EN subtitles not generated... Skipping translations.'")
    print("   - Result: Empty list [] for non-existent files")

if __name__ == "__main__":
    test_captions_system()
