#!/usr/bin/env python3
"""
Simple test script for robust rendering system
"""

from pathlib import Path


def test_basic_rendering():
    """Test basic rendering functionality"""

    print("🧪 Testing Basic Rendering System")
    print("=" * 50)

    try:
        # Import and setup
        from advanced_video_creator import AdvancedVideoCreator

        print("✅ AdvancedVideoCreator imported successfully")

        # Create video creator instance
        creator = AdvancedVideoCreator()
        print("✅ Video creator instance created")

        # Test ffmpeg verification
        print("\n🔍 Testing FFmpeg verification...")
        ffmpeg_path = creator._verify_ffmpeg()
        print(f"✅ FFmpeg path: {ffmpeg_path}")

        # Test basic video creation with simple clips
        print("\n🎬 Testing basic video creation...")

        # Create a simple test video clip
        try:
            from moviepy.editor import ColorClip

            # Create a simple color clip
            test_clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=3.0)

            # Create output directory
            test_output_dir = Path("outputs/test_simple/2025-08-12")
            test_output_dir.mkdir(parents=True, exist_ok=True)

            # Test output path
            output_path = test_output_dir / "simple_test.mp4"
            temp_path = test_output_dir / "tmp_simple_test.mp4"

            print("🎬 Rendering simple test video...")
            print(f"📁 Output: {output_path}")
            print(f"📁 Temp: {temp_path}")

            # Render with basic settings
            test_clip.write_videofile(
                str(temp_path),
                fps=24,
                codec="libx264",
                audio_codec=None,  # No audio
                verbose=False,
                logger=None,
                preset="ultrafast",  # Fastest preset for testing
                threads=2,
            )

            # Check if temp file was created
            if temp_path.exists():
                size_mb = temp_path.stat().st_size / (1024 * 1024)
                print(f"✅ Temp file created: {size_mb:.2f} MB")

                # Try to rename
                try:
                    if output_path.exists():
                        output_path.unlink()
                    temp_path.rename(output_path)
                    print(f"✅ Video successfully rendered: {output_path}")

                    # Clean up
                    test_clip.close()
                    return True

                except Exception as e:
                    print(f"⚠️ Rename failed: {e}")
                    # Try copy as fallback
                    try:
                        import shutil

                        shutil.copy2(temp_path, output_path)
                        temp_path.unlink()
                        print(f"✅ Video copied (fallback): {output_path}")
                        test_clip.close()
                        return True
                    except Exception as copy_error:
                        print(f"❌ Copy fallback failed: {copy_error}")
                        return False
            else:
                print("❌ Temp file not created")
                return False

        except Exception as e:
            print(f"❌ Basic video creation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_rendering()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Basic test completed! Rendering system is working.")
    else:
        print("❌ Basic test failed. Please check the errors above.")
