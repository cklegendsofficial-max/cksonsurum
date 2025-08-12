#!/usr/bin/env python3
"""
Shorts Maker - Enhanced Short Video Generator with ffmpeg Control
Creates 4 horizontal + 3 vertical short videos from long-form content
"""

import logging
import os
import random
from typing import List, Optional, Tuple


# Try to import video processing libraries
try:
    from moviepy.editor import (
        ColorClip,
        CompositeVideoClip,
        TextClip,
        VideoClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.fx import resize

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available")

try:
    import imageio_ffmpeg

    IMAGEIO_FFMPEG_AVAILABLE = True
except ImportError:
    IMAGEIO_FFMPEG_AVAILABLE = False
    print("⚠️ imageio-ffmpeg not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShortsMaker")


class ShortsMaker:
    """Enhanced Short Video Generator with ffmpeg control"""

    def __init__(self):
        self.ffmpeg_path = self._verify_ffmpeg()
        self.horizontal_durations = [15, 30, 45, 60]  # 4 horizontal shorts
        self.vertical_durations = [15, 30, 60]  # 3 vertical shorts

        if not self.ffmpeg_path:
            logger.warning("ffmpeg not found - shorts generation will be skipped")
        else:
            logger.info(f"✅ ffmpeg found: {self.ffmpeg_path}")

    def _verify_ffmpeg(self) -> Optional[str]:
        """Verify ffmpeg availability with multiple fallbacks"""
        try:
            # 1. Try imageio-ffmpeg first (most reliable)
            if IMAGEIO_FFMPEG_AVAILABLE:
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                if ffmpeg_path and os.path.exists(ffmpeg_path):
                    logger.info(f"Using imageio-ffmpeg: {ffmpeg_path}")
                    return ffmpeg_path

            # 2. Try system ffmpeg
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("Using system ffmpeg")
                return "ffmpeg"

            # 3. Try common Windows paths
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            ]

            for path in common_paths:
                if os.path.exists(path):
                    logger.info(f"Using Windows ffmpeg: {path}")
                    return path

            return None

        except Exception as e:
            logger.warning(f"ffmpeg verification failed: {e}")
            return None

    def create_shorts(
        self, long_form_video_path: str, output_folder: str
    ) -> Tuple[List[str], List[str]]:
        """
        Create horizontal and vertical short videos

        Args:
            long_form_video_path: Path to the long-form video
            output_folder: Output directory for shorts

        Returns:
            Tuple of (horizontal_shorts, vertical_shorts) file paths
        """
        if not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy not available - skipping shorts generation")
            return [], []

        if not self.ffmpeg_path:
            logger.warning("ffmpeg not found - skipping shorts generation")
            return [], []

        if not os.path.exists(long_form_video_path):
            logger.error(f"Long form video not found: {long_form_video_path}")
            return [], []

        # Create shorts subdirectory
        shorts_dir = os.path.join(output_folder, "shorts")
        os.makedirs(shorts_dir, exist_ok=True)

        try:
            # Load video
            video = VideoFileClip(long_form_video_path)
            logger.info(f"✅ Loaded video: {video.duration:.1f}s")

            # Create horizontal shorts
            horizontal_shorts = self._create_horizontal_shorts(video, shorts_dir)

            # Create vertical shorts
            vertical_shorts = self._create_vertical_shorts(video, shorts_dir)

            video.close()

            logger.info("✅ Shorts generation completed:")
            logger.info(f"  - Horizontal: {len(horizontal_shorts)}")
            logger.info(f"  - Vertical: {len(vertical_shorts)}")

            return horizontal_shorts, vertical_shorts

        except Exception as e:
            logger.error(f"❌ Shorts generation failed: {e}")
            return [], []

    def _create_horizontal_shorts(
        self, video: VideoFileClip, output_dir: str
    ) -> List[str]:
        """Create 4 horizontal short videos"""
        shorts = []

        for i, duration in enumerate(self.horizontal_durations):
            try:
                # Extract random segment
                start_time = random.uniform(0, max(0, video.duration - duration))
                end_time = start_time + duration

                # Create short clip
                short_clip = video.subclip(start_time, end_time)

                # Add hook at beginning
                hook_clip = self._create_hook_clip(
                    f"Hook {i+1}: The Mystery Deepens...", 3
                )
                final_short = concatenate_videoclips(
                    [hook_clip, short_clip], method="compose"
                )

                # Save with robust rendering
                output_path = os.path.join(output_dir, f"short_{i+1}.mp4")
                temp_path = os.path.join(output_dir, f"tmp_short_{i+1}.mp4")

                self._render_video_robust(
                    final_short, temp_path, output_path, "horizontal"
                )
                shorts.append(output_path)

                logger.info(f"✅ Horizontal short {i+1} created: {duration}s")

            except Exception as e:
                logger.warning(f"⚠️ Horizontal short {i+1} failed: {e}")
                continue

        return shorts

    def _create_vertical_shorts(
        self, video: VideoFileClip, output_dir: str
    ) -> List[str]:
        """Create 3 vertical short videos (9:16 aspect ratio)"""
        shorts = []

        for i, duration in enumerate(self.vertical_durations):
            try:
                # Extract random segment
                start_time = random.uniform(0, max(0, video.duration - duration))
                end_time = start_time + duration

                # Create short clip
                short_clip = video.subclip(start_time, end_time)

                # Convert to vertical (9:16 aspect ratio)
                vertical_clip = self._convert_to_vertical(short_clip)

                # Add hook at beginning
                hook_clip = self._create_hook_clip(
                    f"VHook {i+1}: The Mystery Deepens...", 3
                )
                hook_clip = self._convert_to_vertical(hook_clip)

                final_short = concatenate_videoclips(
                    [hook_clip, vertical_clip], method="compose"
                )

                # Save with robust rendering
                output_path = os.path.join(output_dir, f"vshort_{i+1}.mp4")
                temp_path = os.path.join(output_dir, f"tmp_vshort_{i+1}.mp4")

                self._render_video_robust(
                    final_short, temp_path, output_path, "vertical"
                )
                shorts.append(output_path)

                logger.info(f"✅ Vertical short {i+1} created: {duration}s")

            except Exception as e:
                logger.warning(f"⚠️ Vertical short {i+1} failed: {e}")
                continue

        return shorts

    def _convert_to_vertical(self, clip: VideoFileClip) -> VideoFileClip:
        """Convert horizontal video to vertical (9:16 aspect ratio)"""
        try:
            # Target dimensions for vertical (9:16 aspect ratio)
            target_width = 1080
            target_height = 1920

            # Resize and crop to fit vertical format
            resized = clip.resize(width=target_width)

            # Center crop to target height
            if resized.h > target_height:
                y_center = resized.h // 2
                y_start = max(0, y_center - target_height // 2)
                y_end = min(resized.h, y_start + target_height)
                cropped = resized.crop(y1=y_start, y2=y_end)
            else:
                cropped = resized

            return cropped

        except Exception as e:
            logger.warning(f"Vertical conversion failed: {e}")
            return clip

    def _create_hook_clip(self, text: str, duration: float) -> VideoClip:
        """Create a hook clip with text overlay"""
        try:
            # Create text clip
            text_clip = TextClip(
                text,
                fontsize=70,
                color="white",
                font="Arial-Bold",
                stroke_color="black",
                stroke_width=3,
            ).set_duration(duration)

            # Create background
            bg_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0)).set_duration(
                duration
            )

            # Composite text over background
            hook_clip = CompositeVideoClip([bg_clip, text_clip.set_position("center")])

            return hook_clip

        except Exception as e:
            logger.warning(f"Hook clip creation failed: {e}")
            # Return a simple black clip as fallback
            return ColorClip(size=(1920, 1080), color=(0, 0, 0)).set_duration(duration)

    def _render_video_robust(
        self, clip: VideoClip, temp_path: str, output_path: str, video_type: str
    ):
        """Robust video rendering with error handling"""
        try:
            # Video settings based on type
            if video_type == "vertical":
                fps = 30
                preset = "fast"
                crf = "28"  # Lower quality for vertical
            else:  # horizontal
                fps = 30
                preset = "fast"
                crf = "25"

            # Write to temp file first
            clip.write_videofile(
                temp_path,
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                bitrate="2M",
                verbose=False,
                logger=None,
                preset=preset,
                threads=2,
                ffmpeg_params=[
                    "-movflags",
                    "+faststart",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    crf,
                ],
            )

            # Atomic rename
            if os.path.exists(temp_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)

                # Clean up temp file if it still exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def create_shorts_from_video(
    video_path: str, output_folder: str
) -> Tuple[List[str], List[str]]:
    """Convenience function for creating shorts"""
    maker = ShortsMaker()
    return maker.create_shorts(video_path, output_folder)


# Test function
if __name__ == "__main__":
    print("🧪 Testing Shorts Maker...")

    # Test ffmpeg detection
    maker = ShortsMaker()

    if maker.ffmpeg_path:
        print(f"✅ ffmpeg found: {maker.ffmpeg_path}")

        # Test with a dummy video path
        test_video = "test_video.mp4"
        test_output = "test_output"

        if os.path.exists(test_video):
            print("🎬 Creating test shorts...")
            horizontal, vertical = maker.create_shorts(test_video, test_output)
            print(
                f"✅ Created {len(horizontal)} horizontal + {len(vertical)} vertical shorts"
            )
        else:
            print("⚠️ Test video not found, skipping generation test")
    else:
        print("❌ ffmpeg not found - shorts generation disabled")
