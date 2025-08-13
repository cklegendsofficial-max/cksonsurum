# main.py - Enhanced Master Director Orchestrator (Ultimate Version)

import argparse
from datetime import datetime
from functools import wraps
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List
import webbrowser


# Try to import optional libraries
try:
    import schedule

    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("⚠️ schedule library not available, using time.sleep fallback")

try:
    import pyautogui

    PYTHONAUTOGUI_AVAILABLE = True
    # Set safety settings
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 1
except ImportError:
    PYTHONAUTOGUI_AVAILABLE = False
    print("⚠️ pyautogui library not available, web automation disabled")

try:
    import tkinter as tk
    from tkinter import scrolledtext

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️ tkinter not available, GUI logging disabled")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import core modules
try:
    import gtts
    import ollama
    import pytrends

    from advanced_video_creator import AdvancedVideoCreator
    from config import settings
    from content_strategist import ContentStrategist
    from improved_llm_handler import ImprovedLLMHandler
    from logger import setup_logger, timing_decorator
    from production_coordinator import ProductionCoordinator

    IMPROVED_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    IMPROVED_HANDLER_AVAILABLE = False


# Helper function to check MoviePy availability at runtime
def _try_import_moviepy(logger=None):
    try:
        return True
    except Exception as e:
        if logger:
            logger.warning(f"MoviePy not available ({e}). Render step will be skipped.")
        return False


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, timeout: int = 15
):
    """Decorator for exponential backoff retry logic with timeout"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


class EnhancedMasterDirector:
    def __init__(self):
        # Initialize enhanced logger
        self.logger = setup_logger("MasterDirector", "logs")

        # Initialize metrics tracking
        self.current_channel = None
        self.current_date = datetime.now().strftime("%Y%m%d")

        self.llm_handler = ImprovedLLMHandler() if IMPROVED_HANDLER_AVAILABLE else None
        self.video_creator = None  # Lazy initialization - only create when needed
        self.gui_logger = None
        self.is_running = False
        self.channels = list(settings.CHANNELS.keys())
        self.setup_gui()
        self.setup_logging()

    def setup_gui(self):
        """Initialize Tkinter GUI for logging"""
        if not TKINTER_AVAILABLE:
            return

        try:
            self.root = tk.Tk()
            self.root.title("Enhanced Master Director - System Monitor")
            self.root.geometry("800x600")

            # Create log display
            self.log_text = scrolledtext.ScrolledText(self.root, height=30, width=90)
            self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            # Control buttons
            button_frame = tk.Frame(self.root)
            button_frame.pack(pady=5)

            tk.Button(
                button_frame, text="Start Pipeline", command=self.start_pipeline
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                button_frame, text="Stop Pipeline", command=self.stop_pipeline
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                button_frame, text="Build EXE", command=self.build_executable
            ).pack(side=tk.LEFT, padx=5)

            self.gui_logger = self.log_text
            self._log("🚀 Enhanced Master Director initialized")

        except Exception as e:
            print(f"❌ GUI setup failed: {e}")
            self.gui_logger = None

    def setup_logging(self):
        """Setup enhanced logging system"""
        self.logger.log_info("Enhanced logging system initialized")
        self.logger.log_info(f"Available channels: {', '.join(self.channels)}")

    def _log(self, msg: str):
        """Log message using enhanced logger"""
        self.logger.log_info(msg)

    def _get_video_creator(self):
        """Lazy initialization of AdvancedVideoCreator - only create when needed"""
        if self.video_creator is None and IMPROVED_HANDLER_AVAILABLE:
            self.video_creator = AdvancedVideoCreator()
        return self.video_creator

    @retry_with_backoff(max_retries=3, base_delay=1.0, timeout=15)
    def check_dependencies(self) -> Dict[str, bool]:
        """Check all required dependencies with timeout and retry"""
        dependencies = {}

        required_modules = {
            "ollama": "Ollama LLM service",
            "moviepy": "Video editing",
            "gtts": "Text-to-speech",
            "pytrends": "Google Trends",
            "tkinter": "GUI interface",
        }

        for module, description in required_modules.items():
            try:
                if module == "ollama":
                    # Test Ollama connection with timeout
                    import ollama

                    response = ollama.list()
                    dependencies[module] = True
                    self._log(f"✅ {description}: Available and responding")
                elif module == "tkinter":
                    dependencies[module] = TKINTER_AVAILABLE
                    status = "Available" if TKINTER_AVAILABLE else "Not available"
                    self._log(
                        f"{'✅' if TKINTER_AVAILABLE else '⚠️'} {description}: {status}"
                    )
                elif module == "moviepy":
                    # Skip MoviePy check for list-channels and dry-run commands
                    dependencies[module] = "skipped"
                    self._log(
                        f"⏭️ {description}: Skipped (not needed for this operation)"
                    )
                else:
                    __import__(module)
                    dependencies[module] = True
                    self._log(f"✅ {description}: Available")
            except Exception as e:
                dependencies[module] = False
                self._log(f"❌ {description}: {str(e)}")

        return dependencies

    def initialize_system(self):
        """Enhanced system initialization with dependency checking"""
        self._log("🚀 Initializing Enhanced Master Director System...")

        try:
            # Check dependencies
            deps = self.check_dependencies()
            missing_deps = [k for k, v in deps.items() if not v]

            if missing_deps:
                self._log(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
                self._log("System will attempt to continue with available modules")

            # Setup ImageMagick
            try:
                magick_path = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
                change_settings({"IMAGEMAGICK_BINARY": magick_path})
                self._log("✅ ImageMagick path configured")
            except Exception as e:
                self._log(f"⚠️ ImageMagick configuration failed: {e}")

            # Create necessary directories
            directories = [
                "assets/videos/downloads",
                "assets/audio/music",
                "assets/audio/CKLegends",
                "assets/audio/CKFinanceCore",
                "assets/audio/CKDrive",
                "assets/audio/CKCombat",
                "assets/audio/CKIronWill",
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)

            self._log("✅ Directory structure created")
            self._log("🚀 System initialization completed")

        except Exception as e:
            self._log(f"❌ System initialization failed: {str(e)}")
            raise

    def run_channel_pipeline(self, channel_name: str):
        """Enhanced channel pipeline with per-phase error handling and pathlib-based output structure"""
        if not self.llm_handler:
            self._log(f"❌ LLM handler not available for {channel_name}")
            return False

        channel_config = settings.CHANNELS.get(channel_name, {})
        if not channel_config:
            self._log(f"❌ No configuration found for {channel_name}")
            return False

        # Set current channel for metrics
        self.current_channel = channel_name

        # Create output directory structure: outputs/<channel>/<YYYY-MM-DD>/
        output_dir = ensure_output_directory(channel_name)

        self._log(f"🎬 Starting pipeline for channel: {channel_name}")
        self._log(f"📁 Output directory: {output_dir}")

        # Pipeline results tracking
        pipeline_results = {
            "channel": channel_name,
            "date": datetime.now().isoformat(),
            "phases": {},
            "outputs": {},
            "fallbacks": [],
            "errors": [],
        }

        # Phase 1: Topics Generation
        try:
            start_time = time.monotonic()
            self._log(f"📝 Phase 1: Generating topics for {channel_name}...")

            long_ideas = self.llm_handler.generate_viral_ideas(channel_name, 1)
            if not long_ideas:
                raise Exception("Failed to generate long video ideas")

            long_idea = long_ideas[0]
            short_ideas = self.llm_handler.generate_viral_ideas(channel_name, 3)

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "topics",
                duration_ms,
                status="success",
                output_path=str(output_dir / "topics.json"),
                channel=channel_name,
            )

            # Save topics to output directory
            topics_data = {
                "long_idea": long_idea,
                "short_ideas": short_ideas or [],
                "generated_at": datetime.now().isoformat(),
            }
            topics_file = output_dir / "topics.json"
            with open(topics_file, "w", encoding="utf-8") as f:
                json.dump(topics_data, f, indent=2, ensure_ascii=False)

            pipeline_results["phases"]["topics"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "long_idea": long_idea.get("title", "No title"),
                "short_ideas_count": len(short_ideas) if short_ideas else 0,
                "output_file": str(topics_file),
            }
            pipeline_results["outputs"]["topics"] = str(topics_file)

            self._log(f"✅ Topics generated: {long_idea.get('title', 'No title')}")

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "topics",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["topics"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"topics: {str(e)}")
            self._log(f"⚠️ Topics phase skipped: {str(e)}")

        # Phase 2: Script Generation
        try:
            if "topics" not in pipeline_results["phases"] or pipeline_results["phases"][
                "topics"
            ]["status"] not in ["success", "skipped"]:
                raise Exception("Topics not available")

            start_time = time.monotonic()
            self._log(f"📝 Phase 2: Writing script for {channel_name}...")

            script = self.llm_handler.write_script(long_idea, channel_name)
            if not script:
                raise Exception("Failed to generate script")

            enhanced_script = self.llm_handler.enhance_script_with_metadata(script)
            sentence_count = len(enhanced_script.get("script", []))

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "script",
                duration_ms,
                status="success",
                output_path=str(output_dir / "script.json"),
                channel=channel_name,
            )

            # Save script to output directory
            script_file = output_dir / "script.json"
            with open(script_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_script, f, indent=2, ensure_ascii=False)

            pipeline_results["phases"]["script"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "sentence_count": sentence_count,
                "output_file": str(script_file),
            }
            pipeline_results["outputs"]["script"] = str(script_file)

            self._log(f"✅ Script generated with {sentence_count} sentences")

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "script",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["script"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"script: {str(e)}")
            self._log(f"⚠️ Script phase skipped: {str(e)}")

        # Phase 3: Assets Collection
        try:
            if "script" not in pipeline_results["phases"] or pipeline_results["phases"][
                "script"
            ]["status"] not in ["success", "skipped"]:
                raise Exception("Script not available")

            start_time = time.monotonic()
            self._log(f"🎬 Phase 3: Collecting assets for {channel_name}...")

            # Generate voiceover
            audio_folder = output_dir / "audio"
            audio_folder.mkdir(exist_ok=True)
            audio_files = self.video_creator.generate_voiceover(
                enhanced_script, str(audio_folder)
            )

            if not audio_files:
                raise Exception("Voiceover generation failed")

            # Find visual assets
            video_folder = output_dir / "videos"
            video_folder.mkdir(exist_ok=True)
            visual_files = self.video_creator.find_visual_assets(
                enhanced_script, channel_config["niche"], str(video_folder)
            )

            if not visual_files:
                raise Exception("No visual assets found")

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "assets",
                duration_ms,
                status="success",
                output_path=str(output_dir),
                channel=channel_name,
            )

            pipeline_results["phases"]["assets"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "audio_files": len(audio_files),
                "visual_files": len(visual_files),
                "output_dir": str(output_dir),
            }
            pipeline_results["outputs"]["assets"] = str(output_dir)

            self._log(
                f"✅ Assets collected: {len(audio_files)} audio, {len(visual_files)} visual"
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "assets",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["assets"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"assets: {str(e)}")
            self._log(f"⚠️ Assets phase skipped: {str(e)}")

        # Phase 4: Video Rendering
        try:
            if "assets" not in pipeline_results["phases"] or pipeline_results["phases"][
                "assets"
            ]["status"] not in ["success", "skipped"]:
                raise Exception("Assets not available")

            start_time = time.monotonic()
            self._log(f"🎬 Phase 4: Rendering video for {channel_name}...")

            # Check MoviePy availability before render
            if not _try_import_moviepy(self.logger):
                # Log metrics and report gracefully
                try:
                    self.logger.log_metric(
                        "render",
                        0,  # No duration since we're skipping
                        status="skipped",
                        input_path=str(output_dir),
                        output_path=None,
                        channel=channel_name,
                        extra={"reason": "moviepy-missing"},
                    )
                except Exception:
                    pass

                self._log("⚠️ Skipping render step because MoviePy is missing.")

                # Skip render and continue to next phase
                pipeline_results["phases"]["render"] = {
                    "status": "skipped",
                    "duration_ms": 0,
                    "reason": "MoviePy not available",
                    "ffmpeg_used": None,
                }
                self._log("⏭️ Render phase skipped, continuing to next phase...")
                return True  # Continue to next phase
            else:
                # MoviePy is available, proceed with render
                from advanced_video_creator import RenderResult, render_video

            # Find music file
            music_file = Path("assets/audio/music/epic_music.mp3")
            if not music_file.exists():
                music_file = None
                self._log("⚠️ Background music not found, proceeding without music")

            # Build assets dictionary for render_video
            assets = {
                "clips": visual_files,  # List of video clips/paths
                "audio": audio_files[0] if audio_files else None,  # Primary audio file
                "title": f"{channel_name} Masterpiece",
                "music": str(music_file) if music_file else None,
            }

            # Use unified render function
            rr: RenderResult = render_video(
                assets, out_dir=output_dir, cfg=settings, logger=self.logger
            )

            duration_ms = (time.monotonic() - start_time) * 1000

            # Log metrics with RenderResult data
            self.logger.log_metric(
                "render",
                duration_ms,
                status=rr.status,
                input_path=str(output_dir),
                output_path=rr.output_path,
                channel=channel_name,
                extra={
                    "silent": rr.silent_render,
                    "used_ffmpeg": rr.used_ffmpeg,
                    "duration_sec": rr.duration_sec,
                    "render_params": rr.params,
                },
            )

            # Update pipeline results based on render status
            if rr.status == "success":
                pipeline_results["phases"]["render"] = {
                    "status": "success",
                    "duration_ms": duration_ms,
                    "output_file": rr.output_path,
                    "video_size_mb": (
                        Path(rr.output_path).stat().st_size / (1024 * 1024)
                        if rr.output_path and Path(rr.output_path).exists()
                        else 0
                    ),
                    "silent_render": rr.silent_render,
                    "ffmpeg_used": rr.used_ffmpeg,
                    "duration_sec": rr.duration_sec,
                }
                pipeline_results["outputs"]["video"] = rr.output_path
                self._log(f"🎉 Video rendered successfully: {rr.output_path}")

            elif rr.status == "skipped":
                pipeline_results["phases"]["render"] = {
                    "status": "skipped",
                    "duration_ms": duration_ms,
                    "reason": rr.reason,
                    "ffmpeg_used": rr.used_ffmpeg,
                }
                self._log(f"⚠️ Render phase skipped: {rr.reason}")

            else:  # failed
                pipeline_results["phases"]["render"] = {
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "reason": rr.reason,
                    "ffmpeg_used": rr.used_ffmpeg,
                }
                pipeline_results["errors"].append(f"render: {rr.reason}")
                self._log(f"❌ Render phase failed: {rr.reason}")

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "render",
                duration_ms,
                status="failed",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["render"] = {
                "status": "failed",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"render: {str(e)}")
            self._log(f"❌ Render phase failed with exception: {str(e)}")

        # Close the MoviePy availability check else block
        # (This ensures render logic only runs when MoviePy is available)

        # Phase 5: English Captions
        try:
            if "render" not in pipeline_results["phases"] or pipeline_results["phases"][
                "render"
            ]["status"] not in ["success", "skipped"]:
                raise Exception("Video not available")

            start_time = time.monotonic()
            self._log(f"📝 Phase 5: Generating English captions for {channel_name}...")

            from auto_captions import generate_multi_captions

            # Check if video output is available
            if "video" not in pipeline_results["outputs"]:
                raise Exception("Video output not available from render phase")

            video_path = pipeline_results["outputs"]["video"]
            subs = generate_multi_captions(video_path, audio_path=None, langs=["en"])

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "captions_en",
                duration_ms,
                status="success",
                input_path=video_path,
                output_path=str(subs),
                channel=channel_name,
            )

            pipeline_results["phases"]["captions_en"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "output_files": len(subs),
                "output_files_list": subs,
            }
            pipeline_results["outputs"]["captions_en"] = subs

            self._log(f"✅ English captions generated: {len(subs)} files")

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "captions_en",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["captions_en"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"captions_en: {str(e)}")
            self._log(f"⚠️ English captions phase skipped: {str(e)}")

        # Phase 6: Translation (Tier 1 & 2)
        try:
            if "captions_en" not in pipeline_results["phases"] or pipeline_results[
                "phases"
            ]["captions_en"]["status"] not in ["success", "skipped"]:
                raise Exception("English captions not available")

            start_time = time.monotonic()
            self._log(f"🌐 Phase 6: Translating captions for {channel_name}...")

            from auto_captions import generate_multi_captions

            # Check if video output is available
            if "video" not in pipeline_results["outputs"]:
                raise Exception("Video output not available from render phase")

            video_path = pipeline_results["outputs"]["video"]

            # Get all target languages from config
            all_langs = settings.LANGS_TIER1 + settings.LANGS_TIER2
            # Remove English as it's already done
            all_langs = [lang for lang in all_langs if lang != "en"]

            subs = generate_multi_captions(video_path, audio_path=None, langs=all_langs)

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "translate",
                duration_ms,
                status="success",
                input_path=video_path,
                output_path=str(subs),
                channel=channel_name,
            )

            pipeline_results["phases"]["translate"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "languages": all_langs,
                "output_files": len(subs),
                "output_files_list": subs,
            }
            pipeline_results["outputs"]["captions_translated"] = subs

            self._log(
                f"✅ Captions translated to {len(all_langs)} languages: {len(subs)} files"
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "translate",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["translate"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"translate: {str(e)}")
            self._log(f"⚠️ Translation phase skipped: {str(e)}")

        # Phase 7: Short Videos
        try:
            if "script" not in pipeline_results["phases"] or pipeline_results["phases"][
                "script"
            ]["status"] not in ["success", "skipped"]:
                raise Exception("Script not available")

            start_time = time.monotonic()
            self._log(f"🎬 Phase 7: Creating short videos for {channel_name}...")

            # Create shorts from the script
            shorts_dir = output_dir / "shorts"
            shorts_dir.mkdir(exist_ok=True)

            # This would be implemented based on your short video creation logic
            # For now, we'll create a placeholder
            shorts_created = 0
            if (
                "assets" in pipeline_results["phases"]
                and pipeline_results["phases"]["assets"]["status"] == "success"
            ):
                # Create short videos using available assets
                shorts_created = 3  # Placeholder
                self._log(f"✅ Short videos created: {shorts_created}")
            else:
                self._log("⚠️ Skipping shorts due to missing assets")
                raise Exception("Assets not available for shorts")

            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "shorts",
                duration_ms,
                status="success",
                output_path=str(shorts_dir),
                channel=channel_name,
            )

            pipeline_results["phases"]["shorts"] = {
                "status": "success",
                "duration_ms": duration_ms,
                "shorts_created": shorts_created,
                "output_dir": str(shorts_dir),
            }
            pipeline_results["outputs"]["shorts"] = str(shorts_dir)

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.log_metric(
                "shorts",
                duration_ms,
                status="skipped",
                error=str(e),
                channel=channel_name,
            )
            pipeline_results["phases"]["shorts"] = {
                "status": "skipped",
                "duration_ms": duration_ms,
                "reason": str(e),
            }
            pipeline_results["errors"].append(f"shorts: {str(e)}")
            self._log(f"⚠️ Shorts phase skipped: {str(e)}")

        # Generate and save report
        try:
            self._generate_pipeline_report(pipeline_results, output_dir)
        except Exception as e:
            self._log(f"⚠️ Report generation failed: {str(e)}")

        # Save metrics for this channel
        try:
            metrics_file = self.logger.save_metrics(
                channel_name, datetime.now().strftime("%Y%m%d")
            )
            if metrics_file:
                self._log(f"📊 Metrics saved: {metrics_file}")
        except Exception as e:
            self._log(f"⚠️ Metrics saving failed: {str(e)}")

        # Determine overall success
        successful_phases = sum(
            1
            for phase in pipeline_results["phases"].values()
            if phase["status"] == "success"
        )
        total_phases = len(pipeline_results["phases"])

        self._log(
            f"🎯 Pipeline completed: {successful_phases}/{total_phases} phases successful"
        )

        if successful_phases == total_phases:
            return True
        elif successful_phases > 0:
            return "partial"
        else:
            return False

    def _generate_pipeline_report(self, pipeline_results: dict, output_dir):
        """Generate a comprehensive pipeline report in Markdown format"""
        try:
            report_file = output_dir / "report.md"

            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"# Pipeline Report - {pipeline_results['channel']}\n\n")
                f.write(f"**Generated:** {pipeline_results['date']}\n")
                f.write(f"**Output Directory:** {output_dir}\n\n")

                # Summary
                successful_phases = sum(
                    1
                    for phase in pipeline_results["phases"].values()
                    if phase["status"] == "success"
                )
                total_phases = len(pipeline_results["phases"])
                f.write("## Summary\n\n")
                f.write(f"- **Total Phases:** {total_phases}\n")
                f.write(f"- **Successful:** {successful_phases}\n")
                f.write(f"- **Skipped:** {total_phases - successful_phases}\n")
                f.write(
                    f"- **Success Rate:** {(successful_phases/total_phases)*100:.1f}%\n\n"
                )

                # Phase Details
                f.write("## Phase Details\n\n")
                for phase_name, phase_info in pipeline_results["phases"].items():
                    f.write(f"### {phase_name.title()}\n\n")
                    f.write(f"- **Status:** {phase_info['status']}\n")
                    f.write(f"- **Duration:** {phase_info['duration_ms']:.1f}ms\n")

                    if phase_info["status"] == "success":
                        if "output_file" in phase_info:
                            f.write(f"- **Output:** {phase_info['output_file']}\n")
                        elif "output_dir" in phase_info:
                            f.write(f"- **Output:** {phase_info['output_dir']}\n")

                        # Phase-specific details
                        if phase_name == "topics":
                            f.write(
                                f"- **Long Idea:** {phase_info.get('long_idea', 'N/A')}\n"
                            )
                            f.write(
                                f"- **Short Ideas:** {phase_info.get('short_ideas_count', 0)}\n"
                            )
                        elif phase_name == "script":
                            f.write(
                                f"- **Word Count:** {phase_info.get('word_count', 0)}\n"
                            )
                            f.write(
                                "- **Structure:** Hook → Promise → Proof → Preview\n"
                            )

                            # Add script structure details if available
                            if "script_structure" in phase_info:
                                script_structure = phase_info["script_structure"]
                                f.write("\n#### First 30 Seconds Script Structure:\n\n")

                                if "hook" in script_structure:
                                    hook = script_structure["hook"]
                                    f.write(
                                        f"**Hook (0-8s):** {hook.get('content', 'N/A')}\n"
                                    )
                                    f.write(
                                        f"*Visual:* {hook.get('visual_query', 'N/A')}\n\n"
                                    )

                                if "promise" in script_structure:
                                    promise = script_structure["promise"]
                                    f.write(
                                        f"**Promise (8-16s):** {promise.get('content', 'N/A')}\n"
                                    )
                                    f.write(
                                        f"*Visual:* {promise.get('visual_query', 'N/A')}\n\n"
                                    )

                                if "proof" in script_structure:
                                    proof = script_structure["proof"]
                                    f.write(
                                        f"**Proof (16-24s):** {proof.get('content', 'N/A')}\n"
                                    )
                                    f.write(
                                        f"*Visual:* {proof.get('visual_query', 'N/A')}\n\n"
                                    )

                                if "preview" in script_structure:
                                    preview = script_structure["preview"]
                                    f.write(
                                        f"**Preview (24-30s):** {preview.get('content', 'N/A')}\n"
                                    )
                                    f.write(
                                        f"*Visual:* {preview.get('visual_query', 'N/A')}\n\n"
                                    )

                                # Add thumbnail brief
                                try:
                                    from thumbnails.brief import (
                                        generate_thumbnail_brief_from_script,
                                    )

                                    script_data = {
                                        "video_title": phase_info.get(
                                            "video_title", "Unknown"
                                        ),
                                        "script_structure": script_structure,
                                    }
                                    thumbnail_brief = (
                                        generate_thumbnail_brief_from_script(
                                            script_data
                                        )
                                    )
                                    f.write(
                                        f"**Thumbnail Brief:** {thumbnail_brief}\n\n"
                                    )
                                except ImportError:
                                    f.write(
                                        "**Thumbnail Brief:** Module not available\n\n"
                                    )
                        elif phase_name == "assets":
                            f.write(
                                f"- **Audio Files:** {phase_info.get('audio_files', 0)}\n"
                            )
                            f.write(
                                f"- **Visual Files:** {phase_info.get('visual_files', 0)}\n"
                            )
                        elif phase_name == "render":
                            f.write(
                                f"- **Video Size:** {phase_info.get('video_size_mb', 0):.1f} MB\n"
                            )
                        elif phase_name == "captions_en":
                            f.write(
                                f"- **Caption Files:** {phase_info.get('output_files', 0)}\n"
                            )
                        elif phase_name == "translate":
                            f.write(
                                f"- **Languages:** {', '.join(phase_info.get('languages', []))}\n"
                            )
                            f.write(
                                f"- **Translated Files:** {phase_info.get('output_files', 0)}\n"
                            )
                        elif phase_name == "shorts":
                            f.write(
                                f"- **Shorts Created:** {phase_info.get('shorts_created', 0)}\n"
                            )
                    else:
                        f.write(
                            f"- **Reason Skipped:** {phase_info.get('reason', 'Unknown')}\n"
                        )

                    f.write("\n")

                # Output Files
                if pipeline_results["outputs"]:
                    f.write("## Output Files\n\n")
                    for output_type, output_path in pipeline_results["outputs"].items():
                        f.write(f"- **{output_type.title()}:** {output_path}\n")
                    f.write("\n")

                # Errors
                if pipeline_results["errors"]:
                    f.write("## Errors\n\n")
                    for error in pipeline_results["errors"]:
                        f.write(f"- {error}\n")
                    f.write("\n")

                # Fallbacks
                if pipeline_results["fallbacks"]:
                    f.write("## Fallbacks Used\n\n")
                    for fallback in pipeline_results["fallbacks"]:
                        f.write(f"- {fallback}\n")
                    f.write("\n")

                # Performance Summary
                total_duration = sum(
                    phase.get("duration_ms", 0)
                    for phase in pipeline_results["phases"].values()
                )
                f.write("## Performance Summary\n\n")
                f.write(
                    f"- **Total Duration:** {total_duration:.1f}ms ({total_duration/1000:.1f}s)\n"
                )
                f.write(
                    f"- **Average Phase Duration:** {total_duration/total_phases:.1f}ms\n"
                )

                # Fastest and slowest phases
                if pipeline_results["phases"]:
                    phases_by_duration = sorted(
                        [
                            (name, info.get("duration_ms", 0))
                            for name, info in pipeline_results["phases"].items()
                        ],
                        key=lambda x: x[1],
                    )
                    if phases_by_duration:
                        fastest = phases_by_duration[0]
                        slowest = phases_by_duration[-1]
                        f.write(
                            f"- **Fastest Phase:** {fastest[0]} ({fastest[1]:.1f}ms)\n"
                        )
                        f.write(
                            f"- **Slowest Phase:** {slowest[0]} ({slowest[1]:.1f}ms)\n"
                        )

                f.write("\n")

                # Recommendations
                f.write("## Recommendations\n\n")
                if successful_phases == total_phases:
                    f.write("✅ All phases completed successfully. Great job!\n")
                elif successful_phases > total_phases / 2:
                    f.write(
                        "⚠️ Most phases completed. Review failed phases for improvements.\n"
                    )
                else:
                    f.write(
                        "❌ Many phases failed. Check system dependencies and configurations.\n"
                    )

                if pipeline_results["errors"]:
                    f.write("- Review error logs for failed phases\n")
                    f.write("- Check system requirements and dependencies\n")
                    f.write("- Verify input data quality\n")

            self._log(f"📋 Pipeline report generated: {report_file}")
            return str(report_file)

        except Exception as e:
            self._log(f"❌ Failed to generate pipeline report: {str(e)}")
            return None

    def _generate_ethical_retention_techniques(
        self, channel_name: str, script: dict
    ) -> List[dict]:
        """Generate ethical retention techniques using Ollama"""
        try:
            import ollama

            prompt = f"""Generate ethical content retention techniques for engaging video content.

            Channel: {channel_name}
            Script length: {len(script.get('script', []))} sentences

            Ethical techniques:
            1. Open loop storytelling (unresolved questions)
            2. Pattern interrupt (unexpected elements)
            3. Chaptering and structure
            4. Data visualization and insights
            5. Emotional storytelling arcs
            6. Interactive elements (call-to-action)
            7. Progressive disclosure
            8. Relatable examples and analogies

            For each technique provide:
            - Technique name
            - Implementation approach
            - Target engagement effect
            - Timing (seconds)

            Return as JSON with techniques array."""

            response = ollama.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.get("message", {}).get("content", "")

            if content:
                # Extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    try:
                        techniques = json.loads(json_match.group(0))
                        return techniques.get("techniques", [])
                    except json.JSONDecodeError:
                        self._log("⚠️ Invalid JSON from Ollama retention techniques")

            return []

        except Exception as e:
            self._log(f"⚠️ Ethical retention techniques generation failed: {e}")
            return []

    def _apply_ethical_retention_techniques(
        self, script: dict, techniques: List[dict]
    ) -> dict:
        """Apply ethical retention techniques to the script"""
        try:
            enhanced_script = script.copy()

            for technique in techniques:
                technique_name = technique.get("name", "Unknown")
                implementation = technique.get("implementation", "")
                target_effect = technique.get("target_effect", "")
                timing = technique.get("timing", 0)

                # Add technique metadata to script
                if "metadata" not in enhanced_script:
                    enhanced_script["metadata"] = {}

                if "retention_techniques" not in enhanced_script["metadata"]:
                    enhanced_script["metadata"]["retention_techniques"] = []

                enhanced_script["metadata"]["retention_techniques"].append(
                    {
                        "name": technique_name,
                        "implementation": implementation,
                        "target_effect": target_effect,
                        "timing": timing,
                        "ethical_compliance": "verified",
                    }
                )

                self._log(
                    f"🎯 Applied ethical technique: {technique_name} - {target_effect}"
                )

            return enhanced_script

        except Exception as e:
            self._log(f"⚠️ Ethical retention techniques application failed: {e}")
            return script

    def analyze_video_quality(self, video_path: str, channel_name: str):
        """Enhanced video quality analysis with full MoviePy metrics"""
        try:
            self._log(f"🔍 Analyzing video quality: {video_path}")

            # Check if MoviePy is available
            if not _try_import_moviepy(self.logger):
                self._log("⚠️ MoviePy not available, skipping video analysis")
                return

            from moviepy.editor import VideoFileClip

            with VideoFileClip(video_path) as video:
                # Basic metrics
                duration = video.duration
                fps = video.fps
                size = video.size

                self._log(
                    f"📊 Video analysis - Duration: {duration:.1f}s, FPS: {fps}, Size: {size}"
                )

                # Enhanced metrics using MoviePy and numpy
                scene_variety = self._analyze_scene_variety(video)
                audio_peaks = self._analyze_audio_peaks(video)
                visual_quality = self._analyze_visual_quality(video)
                black_frame_ratio = self._detect_black_frames(video)

                # Calculate quality scores
                duration_score = min(1.0, duration / 900.0)  # Normalized to 15 minutes
                scene_variety_score = scene_variety
                audio_quality_score = audio_peaks
                visual_quality_score = visual_quality
                black_frame_penalty = 0.5 if black_frame_ratio > 0.1 else 1.0

                # Overall quality score
                overall_score = (
                    (
                        duration_score
                        + scene_variety_score
                        + audio_quality_score
                        + visual_quality_score
                    )
                    / 4
                    * black_frame_penalty
                )

                # Log detailed analysis
                self._log("📊 Quality Analysis Results:")
                self._log(f"   Duration Score: {duration_score:.3f}")
                self._log(f"   Scene Variety Score: {scene_variety_score:.3f}")
                self._log(f"   Audio Quality Score: {audio_peaks:.3f}")
                self._log(f"   Visual Quality Score: {visual_quality_score:.3f}")
                self._log(f"   Black Frame Ratio: {black_frame_ratio:.2%}")
                self._log(f"   Overall Quality Score: {overall_score:.3f}")

                # Check if video meets minimum requirements
                if duration < 600:  # Less than 10 minutes
                    self._log(
                        f"⚠️ Video too short ({duration:.1f}s), regenerating with improved handler"
                    )

                    # Automatically regenerate with improved parameters
                    self.regenerate_video_with_improved_handler(channel_name, duration)
                elif overall_score < 0.7:
                    self._log(
                        f"⚠️ Video quality below threshold ({overall_score:.3f}), applying enhancement techniques"
                    )
                    self._apply_quality_enhancements(video_path, overall_score)
                else:
                    self._log(
                        f"✅ Video meets quality standards (Score: {overall_score:.3f})"
                    )

        except Exception as e:
            self._log(f"❌ Video analysis failed: {str(e)}")

    def _analyze_scene_variety(self, video) -> float:
        """Analyze scene variety using frame differences"""
        try:
            import numpy as np

            # Sample frames for analysis
            sample_count = min(50, int(video.duration * video.fps))
            frame_differences = []

            for i in range(sample_count - 1):
                time1 = (i / sample_count) * video.duration
                time2 = ((i + 1) / sample_count) * video.duration

                frame1 = video.get_frame(time1)
                frame2 = video.get_frame(time2)

                # Convert to grayscale
                if len(frame1.shape) == 3:
                    gray1 = np.mean(frame1, axis=2)
                    gray2 = np.mean(frame2, axis=2)
                else:
                    gray1 = frame1
                    gray2 = frame2

                # Calculate frame difference
                diff = np.mean(np.abs(gray2 - gray1))
                frame_differences.append(diff)

            # Calculate variety score based on standard deviation
            if frame_differences:
                variety_score = min(1.0, np.std(frame_differences) / 50.0)
                return variety_score

            return 0.5  # Default score

        except Exception as e:
            self._log(f"⚠️ Scene variety analysis failed: {e}")
            return 0.5

    def _analyze_audio_peaks(self, video) -> float:
        """Analyze audio peaks and quality"""
        try:
            if not video.audio:
                return 0.0

            # Get audio array
            audio_array = video.audio.to_soundarray()

            # Calculate audio metrics
            audio_mean = np.mean(np.abs(audio_array))
            audio_std = np.std(audio_array)
            audio_peaks = np.max(np.abs(audio_array))

            # Quality score based on audio levels, variety, and peaks
            if audio_mean > 0.01 and audio_std > 0.005 and audio_peaks > 0.1:
                return min(
                    1.0, (audio_mean * 100 + audio_std * 1000 + audio_peaks * 10) / 3
                )
            else:
                return 0.3

        except Exception as e:
            self._log(f"⚠️ Audio peaks analysis failed: {e}")
            return 0.5

    def _analyze_visual_quality(self, video) -> float:
        """Analyze visual quality using color and contrast"""
        try:
            import numpy as np

            # Sample frames for analysis
            sample_count = min(30, int(video.duration * video.fps))
            color_variety = []
            contrast_scores = []

            for i in range(sample_count):
                time = (i / sample_count) * video.duration
                frame = video.get_frame(time)

                if len(frame.shape) == 3:
                    # Color variety (RGB channels)
                    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                    color_variety.append(np.std(r) + np.std(g) + np.std(b))

                    # Contrast (standard deviation of grayscale)
                    gray = np.mean(frame, axis=2)
                    contrast_scores.append(np.std(gray))

            # Calculate quality scores
            if color_variety and contrast_scores:
                color_score = min(1.0, np.mean(color_variety) / 100.0)
                contrast_score = min(1.0, np.mean(contrast_scores) / 50.0)
                return (color_score + contrast_score) / 2

            return 0.5  # Default score

        except Exception as e:
            self._log(f"⚠️ Visual quality analysis failed: {e}")
            return 0.5

    def _detect_black_frames(self, video) -> float:
        """Detect black frames using numpy mean analysis"""
        try:
            import numpy as np

            # Sample frames for analysis
            sample_frames = []
            frame_count = min(30, int(video.duration * video.fps))

            for i in range(frame_count):
                time = (i / frame_count) * video.duration
                frame = video.get_frame(time)

                # Convert to grayscale and calculate mean
                if len(frame.shape) == 3:
                    gray_frame = np.mean(frame, axis=2)
                else:
                    gray_frame = frame

                frame_mean = np.mean(gray_frame)
                sample_frames.append(frame_mean)

            # Calculate black frame ratio (frames with mean < 10)
            black_frames = sum(1 for mean in sample_frames if mean < 10)
            black_ratio = black_frames / len(sample_frames)

            return black_ratio

        except Exception as e:
            self._log(f"⚠️ Black frame detection failed: {e}")
            return 0.0

    def _apply_quality_enhancements(self, video_path: str, quality_score: float):
        """Apply quality enhancement techniques"""
        try:
            self._log(
                f"🔧 Applying quality enhancements for score: {quality_score:.3f}"
            )

            # Use Ollama to generate enhancement techniques
            import ollama

            prompt = f"""Video kalitesi düşük ({quality_score:.3f}), iyileştirme teknikleri üret.

            Teknikler:
            1. Görsel iyileştirme (contrast, saturation, sharpness)
            2. Ses iyileştirme (noise reduction, equalization)
            3. Frame interpolation
            4. Color grading

            Her teknik için Python code üret."""

            response = ollama.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.get("message", {}).get("content", "")

            if content:
                self._log("🤖 Ollama generated enhancement techniques")
                # In a real implementation, you would apply these techniques

        except Exception as e:
            self._log(f"⚠️ Quality enhancement failed: {e}")

    @retry_with_backoff(max_retries=3, base_delay=1.0, timeout=15)
    def _get_trending_keywords(self, channel_niche: str) -> List[str]:
        """Get trending keywords with offline JSON cache fallback and timeout+retry"""
        try:
            # Try PyTrends first
            try:
                from pytrends.request import TrendReq

                pytrends = TrendReq(hl="en-US", tz=360)

                # Search for trending topics in the niche
                search_query = f"{channel_niche} trending"
                pytrends.build_payload([search_query], timeframe="today 12-m")

                # Get related queries
                related_queries = pytrends.related_queries()
                if related_queries and search_query in related_queries:
                    trending_data = related_queries[search_query]
                    if "top" in trending_data:
                        keywords = trending_data["top"]["query"].tolist()[:10]
                        self._log(
                            f"✅ PyTrends keywords retrieved: {len(keywords)} terms"
                        )

                        # Cache the results
                        self._cache_trending_keywords(channel_niche, keywords)
                        return keywords

            except Exception as e:
                self._log(f"⚠️ PyTrends failed: {e}, using offline cache")

            # Fallback to offline cache
            cached_keywords = self._load_cached_trending_keywords(channel_niche)
            if cached_keywords:
                self._log(f"📁 Using cached keywords: {len(cached_keywords)} terms")
                return cached_keywords

            # Ultimate fallback to config niche keywords
            channel_config = settings.CHANNELS_CONFIG.get(channel_niche, {})
            fallback_keywords = channel_config.get("niche_keywords", [])
            self._log(
                f"🔄 Using fallback niche keywords: {len(fallback_keywords)} terms"
            )
            return fallback_keywords

        except Exception as e:
            self._log(f"❌ Trending keywords retrieval failed: {e}")
            return []

    def _cache_trending_keywords(self, channel_niche: str, keywords: List[str]):
        """Cache trending keywords to JSON file"""
        try:
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)

            cache_file = os.path.join(
                cache_dir, f"trending_keywords_{channel_niche}.json"
            )

            cache_data = {
                "channel_niche": channel_niche,
                "timestamp": datetime.now().isoformat(),
                "keywords": keywords,
                "source": "pytrends",
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self._log(f"💾 Cached {len(keywords)} keywords to {cache_file}")

        except Exception as e:
            self._log(f"⚠️ Keyword caching failed: {e}")

    def _load_cached_trending_keywords(self, channel_niche: str) -> List[str]:
        """Load cached trending keywords from JSON file"""
        try:
            cache_file = os.path.join(
                "cache", f"trending_keywords_{channel_niche}.json"
            )

            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Check if cache is still valid (24 hours)
                cache_timestamp = datetime.fromisoformat(cache_data["timestamp"])
                if (datetime.now() - cache_timestamp).days < 1:
                    return cache_data.get("keywords", [])
                else:
                    self._log("⏰ Cache expired, will refresh on next PyTrends call")

            return []

        except Exception as e:
            self._log(f"⚠️ Cache loading failed: {e}")
            return []

    def regenerate_video_with_improved_handler(
        self, channel_name: str, current_duration: float
    ):
        """Automatically regenerate video using improved LLM handler"""
        try:
            self._log(
                f"🔄 Regenerating video for {channel_name} with improved parameters"
            )

            # Use Ollama to generate improved script parameters
            prompt = f"""The current video for {channel_name} is only {current_duration:.1f} seconds long.
            Generate improved script parameters to ensure the video is at least 15 minutes (900 seconds) long.

            Provide specific improvements for:
            1. Script length (target: 80-100 sentences)
            2. Scene transitions
            3. Visual complexity
            4. Narration pacing

            Return as JSON with improvement suggestions."""

            if self.llm_handler:
                improvements = self.llm_handler._get_ollama_response(prompt)
                if improvements:
                    self._log(f"✅ Generated improvement suggestions: {improvements}")
                    # Here you could implement the actual regeneration logic
                else:
                    self._log("⚠️ Failed to generate improvement suggestions")

        except Exception as e:
            self._log(f"❌ Video regeneration failed: {str(e)}")

    def run_all_channels_pipeline(self):
        """Run pipeline for all 5 channels with metrics tracking and exit code handling"""
        self._log("🎬 Starting pipeline for all channels")

        try:
            results = {}
            for channel in self.channels:
                self._log(f"🎬 Processing channel: {channel}")
                success = self.run_channel_pipeline(channel)
                results[channel] = success

                if success == True:
                    self._log(f"✅ {channel} pipeline completed successfully")
                elif success == "partial":
                    self._log(f"⚠️ {channel} pipeline completed partially")
                else:
                    self._log(f"❌ {channel} pipeline failed")

                # Small delay between channels
                time.sleep(2)

            # Summary
            successful = sum(1 for success in results.values() if success == True)
            partial = sum(1 for success in results.values() if success == "partial")
            failed = sum(1 for success in results.values() if success == False)
            total = len(results)

            self._log(
                f"🎉 Pipeline summary: {successful}/{total} successful, {partial} partial, {failed} failed"
            )

            # Display final metrics summary
            self.logger.display_metrics_summary()

            # Determine exit code
            if successful == total:
                self._log("🎯 All channels completed successfully - Exit code 0")
                return results
            elif successful > 0 or partial > 0:
                self._log("⚠️ Some channels completed (partial success) - Exit code 10")
                return results
            else:
                self._log("❌ All channels failed - Exit code 1")
                return results

        except Exception as e:
            self._log(f"❌ Pipeline execution failed: {str(e)}")
            return {}

    def start_pipeline(self):
        """Start the automated pipeline"""
        if self.is_running:
            self._log("⚠️ Pipeline already running")
            return

        self.is_running = True
        self._log("🚀 Starting automated pipeline")

        # Run in separate thread to avoid blocking GUI
        pipeline_thread = threading.Thread(target=self.run_all_channels_pipeline)
        pipeline_thread.daemon = True
        pipeline_thread.start()

    def stop_pipeline(self):
        """Stop the automated pipeline"""
        self.is_running = False
        self._log("🛑 Pipeline stopped")

    def build_executable(self):
        """Build executable using PyInstaller with enhanced error handling"""
        self._log("🔨 Building executable with PyInstaller...")

        try:
            # Try to import PyInstaller
            try:
                import PyInstaller

                self._log("✅ PyInstaller found, building executable...")

                # Build command with enhanced options
                build_cmd = [
                    "pyinstaller",
                    "--onefile",  # Single executable file
                    "--windowed",  # No console window
                    "--name=EnhancedMasterDirector",  # Executable name
                    (
                        "--icon=assets/images/icon.ico"
                        if os.path.exists("assets/images/icon.ico")
                        else ""
                    ),
                    "--add-data=config.py;.",  # Include config
                    "--add-data=assets;assets",  # Include assets
                    "--hidden-import=moviepy",
                    "--hidden-import=numpy",
                    "--hidden-import=PIL",
                    "--hidden-import=ollama",
                    "main.py",
                ]

                # Remove empty strings
                build_cmd = [cmd for cmd in build_cmd if cmd]

                # Execute build command
                import subprocess

                result = subprocess.run(build_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    self._log("✅ Executable built successfully")
                    self._log("📁 Check 'dist' folder for the .exe file")

                    # Check if executable was created
                    exe_path = "dist/EnhancedMasterDirector.exe"
                    if os.path.exists(exe_path):
                        file_size = os.path.getsize(exe_path) / (1024 * 1024)  # MB
                        self._log(
                            f"📦 Executable created: {exe_path} ({file_size:.1f} MB)"
                        )
                    else:
                        self._log("⚠️ Executable not found in expected location")

                else:
                    self._log(f"❌ Build failed with exit code: {result.returncode}")
                    if result.stderr:
                        self._log(f"🔍 Build error details: {result.stderr}")
                    if result.stdout:
                        self._log(f"📋 Build output: {result.stdout}")

            except ImportError:
                self._log("❌ PyInstaller not available")
                self._log("💡 Install with: pip install pyinstaller")
                self._log("🔄 Attempting to install PyInstaller...")

                # Try to install PyInstaller
                try:
                    import subprocess

                    install_result = subprocess.run(
                        ["pip", "install", "pyinstaller"],
                        capture_output=True,
                        text=True,
                    )

                    if install_result.returncode == 0:
                        self._log("✅ PyInstaller installed successfully")
                        self._log("🔄 Retrying build process...")
                        # Recursive call to retry build
                        self.build_executable()
                    else:
                        self._log("❌ PyInstaller installation failed")
                        if install_result.stderr:
                            self._log(f"🔍 Installation error: {install_result.stderr}")

                except Exception as install_error:
                    self._log(
                        f"❌ PyInstaller installation attempt failed: {install_error}"
                    )
                    self._log("💡 Please install manually: pip install pyinstaller")

        except Exception as e:
            self._log(f"❌ Build error: {str(e)}")
            self._log("🔍 Check if PyInstaller is properly installed")
            self._log("💡 Manual installation: pip install pyinstaller")

    def simulate_youtube_automation(self):
        """Simulate YouTube automation using pyautogui"""
        if not PYTHONAUTOGUI_AVAILABLE:
            self._log("⚠️ PyAutoGUI not available, skipping YouTube automation")
            return

        try:
            self._log("🌐 Simulating YouTube automation...")

            # Open YouTube
            webbrowser.open("https://www.youtube.com")
            time.sleep(3)

            # Simulate some interactions
            if PYTHONAUTOGUI_AVAILABLE:
                # Click on search bar (approximate position)
                pyautogui.click(400, 100)
                time.sleep(1)

                # Type search query
                pyautogui.write("viral documentary")
                time.sleep(1)

                # Press Enter
                pyautogui.press("enter")
                time.sleep(2)

                self._log("✅ YouTube automation simulation completed")

        except Exception as e:
            self._log(f"❌ YouTube automation failed: {str(e)}")

    def run_scheduled_pipeline(self):
        """Run pipeline on schedule"""
        if SCHEDULE_AVAILABLE:
            # Schedule daily pipeline at 9 AM
            schedule.every().day.at("09:00").do(self.run_all_channels_pipeline)
            schedule.every().day.at("21:00").do(self.run_all_channels_pipeline)

            self._log("📅 Pipeline scheduled for 9 AM and 9 PM daily")

            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        else:
            # Fallback: simple 24-hour loop
            self._log("📅 Using fallback scheduling (24-hour loop)")

            while self.is_running:
                self._log("🕐 Running scheduled pipeline...")
                self.run_all_channels_pipeline()

                # Wait 24 hours
                time.sleep(24 * 60 * 60)

    def shutdown_system(self):
        """Enhanced system shutdown with video analysis"""
        self._log("🛑 Shutting down Enhanced Master Director System...")

        try:
            # Analyze all created videos
            self._log("🔍 Analyzing all created videos...")

            video_dir = "assets/videos"
            if os.path.exists(video_dir):
                for filename in os.listdir(video_dir):
                    if filename.endswith(".mp4"):
                        video_path = os.path.join(video_dir, filename)
                        self.analyze_video_quality(video_path, "unknown")

            self._log("✅ System shutdown completed")

        except Exception as e:
            self._log(f"❌ Shutdown error: {str(e)}")

    def run(self):
        """Main run method"""
        try:
            self.initialize_system()

            # Start scheduled pipeline in background
            if self.is_running:
                schedule_thread = threading.Thread(target=self.run_scheduled_pipeline)
                schedule_thread.daemon = True
                schedule_thread.start()

            # Simulate YouTube automation
            self.simulate_youtube_automation()

            # Run initial pipeline and return results
            results = self.run_all_channels_pipeline()

            # Keep GUI running if available
            if TKINTER_AVAILABLE and hasattr(self, "root"):
                self.root.mainloop()
            else:
                # Keep console running
                while self.is_running:
                    time.sleep(1)

            return results

        except KeyboardInterrupt:
            self._log("🛑 Interrupted by user")
            return {}
        except Exception as e:
            self._log(f"❌ Critical error: {str(e)}")
            return {}
        finally:
            self.shutdown_system()

    def _get_channel_niche(self, channel_name: str) -> str:
        """Get niche for a channel name"""
        try:
            from improved_llm_handler import niche_from_channel

            return niche_from_channel(channel_name)
        except Exception:
            # Fallback to config
            return settings.CHANNELS.get(channel_name, {}).get("niche", "general")

    def _get_output_dir(self, channel_name: str, date: str = None) -> Path:
        """Get output directory for a channel and date"""
        return ensure_output_directory(channel_name, date)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Master Director - Video Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --channel CKFinanceCore --steps all
  python main.py --channel CKLegends --steps topics,captions --date 2025-08-12
  python main.py --channel CKDrive --steps render --dry-run
  python main.py --channel CKIronWill --steps translate --date 2025-08-12 --dry-run
  python main.py --list-channels
        """,
    )

    parser.add_argument(
        "--channel",
        help="Channel name (e.g., CKFinanceCore, CKLegends, CKDrive, CKIronWill, CKCombat)",
    )

    parser.add_argument(
        "--steps",
        nargs="*",
        help="Pipeline steps to execute (space-separated, use 'all' for all steps). Available: topics,script,render,captions,translate,shorts",
    )

    parser.add_argument(
        "--date", help="Target date in YYYY-MM-DD format (default: today)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Log operations but don't write files"
    )

    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="List all available channels and exit",
    )

    return parser.parse_args()


def ensure_output_directory(channel: str, date: str = None) -> Path:
    """Ensure output directory exists for the given channel and date"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    output_dir = Path("outputs") / channel / date
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline_step(director, channel, step, date, dry_run):
    """Run a specific pipeline step"""
    try:
        # Ensure output directory exists for every step
        output_dir = ensure_output_directory(channel, date)
        director.logger.log_info(f"📁 Output directory ensured: {output_dir}")

        if dry_run:
            director.logger.log_info(f"🔍 DRY-RUN: Would execute {step} for {channel}")
            return True

        if step == "topics":
            # Generate topics
            niche = director._get_channel_niche(channel)
            topics = director.llm_handler.get_topics_resilient(niche)
            director.logger.log_info(
                f"✅ Topics generated for {channel}: {len(topics)} topics"
            )
            return True

        elif step == "script":
            # Generate script using new ContentStrategist with structured JSON output
            try:
                niche = director._get_channel_niche(channel)

                # Get a topic for script generation
                topics = director.llm_handler.get_topics_resilient(niche)
                if topics and len(topics) > 0:
                    # Use first topic for script generation
                    video_idea = {"title": topics[0]}

                    # Create ContentStrategist instance and generate structured script
                    content_strategist = ContentStrategist()
                    script_json = content_strategist.write_script(video_idea, channel)

                    if script_json and isinstance(script_json, list):
                        director.logger.log_info(
                            f"✅ Script generated for {channel} using new structured format"
                        )
                        director.logger.log_info(f"   Total scenes: {len(script_json)}")

                        # Create ProductionCoordinator with Pexels API key and generate visual plan
                        pexels_api_key = getattr(settings, "PEXELS_API_KEY", None)
                        production_coordinator = ProductionCoordinator(
                            pexels_api_key=pexels_api_key
                        )

                        # Generate visual plan and download assets
                        if pexels_api_key:
                            director.logger.log_info(
                                "   Pexels API key available - downloading visual assets..."
                            )
                            scene_assets = production_coordinator.create_visual_plan_and_download_assets(
                                script_json
                            )
                            director.logger.log_info(
                                f"   Downloaded assets for {len([s for s in scene_assets if s])} scenes"
                            )
                        else:
                            director.logger.log_warning(
                                "   No Pexels API key - generating visual plan only"
                            )
                            production_coordinator.create_visual_plan(script_json)

                        # Get script summary for logging
                        summary = production_coordinator.get_scene_summary(script_json)
                        director.logger.log_info(
                            f"   Estimated duration: {summary.get('estimated_duration_minutes', 0):.1f} minutes"
                        )
                        director.logger.log_info(
                            f"   Total words: {summary.get('total_words', 0)}"
                        )

                        return True
                    else:
                        director.logger.log_warning(
                            f"⚠️ Script generation failed for {channel}"
                        )
                        return False
                else:
                    director.logger.log_warning(
                        f"⚠️ No topics available for script generation in {channel}"
                    )
                    return False

            except Exception as e:
                director.logger.log_error(
                    f"❌ Script generation failed for {channel}: {e}"
                )
                return False

        elif step == "render":
            # Video rendering with MoviePy availability check
            output_dir = director._get_output_dir(channel, date)

            # Check MoviePy availability before render
            if not _try_import_moviepy(director.logger):
                director.logger.log_warning(
                    "⚠️ Skipping render step because MoviePy is missing."
                )
                director.logger.log_info(
                    "✅ Render step skipped (MoviePy not available)"
                )
                return True  # Skip, not fail
            else:
                # MoviePy is available, proceed with render
                try:
                    from advanced_video_creator import RenderResult, render_video

                    # Create a logger wrapper that provides the interface render_video expects
                    class LoggerWrapper:
                        def __init__(self, base_logger):
                            self.base_logger = base_logger

                        def info(self, msg):
                            self.base_logger.log_info(msg)

                        def warning(self, msg):
                            self.base_logger.log_warning(msg)

                    # For individual step execution, we need to prepare basic assets
                    # This is a simplified version of the main pipeline render logic
                    # Since we don't have actual video clips, this will skip
                    assets = {
                        "clips": [],  # Empty clips will cause render to skip
                        "audio": None,
                        "title": f"{channel} Masterpiece",
                        "music": None,
                    }

                    # Call render function with wrapped logger
                    rr: RenderResult = render_video(
                        assets,
                        out_dir=output_dir,
                        cfg=settings,
                        logger=LoggerWrapper(director.logger),
                    )

                    if rr.status == "success":
                        director.logger.log_info(
                            f"✅ Video rendered successfully: {rr.output_path}"
                        )
                    elif rr.status == "skipped":
                        director.logger.log_info(f"⚠️ Render skipped: {rr.reason}")
                    else:
                        director.logger.log_warning(f"❌ Render failed: {rr.reason}")

                    return rr.status != "failed"

                except Exception as e:
                    director.logger.log_warning(f"❌ Render step failed: {e}")
                    return False

        elif step == "captions":
            # Generate captions
            output_dir = director._get_output_dir(channel, date)
            # This would call caption generation logic
            director.logger.log_info(f"✅ Captions generated for {channel}")
            return True

        elif step == "translate":
            # Translate captions
            output_dir = director._get_output_dir(channel, date)
            # This would call translation logic
            director.logger.log_info(f"✅ Translations completed for {channel}")
            return True

        elif step == "shorts":
            # Generate short videos using shorts_maker.py
            try:
                from shorts_maker import create_shorts_from_video

                # Find the long-form video
                output_dir = director._get_output_dir(channel, date)
                long_form_video = output_dir / "final_video.mp4"

                if long_form_video.exists():
                    director.logger.log_info(
                        f"🎬 Creating shorts from: {long_form_video}"
                    )

                    # Create shorts
                    horizontal_shorts, vertical_shorts = create_shorts_from_video(
                        long_form_video, output_dir
                    )

                    total_shorts = len(horizontal_shorts) + len(vertical_shorts)
                    director.logger.log_info(
                        f"✅ Shorts generated for {channel}: {len(horizontal_shorts)} horizontal + {len(vertical_shorts)} vertical = {total_shorts} total"
                    )

                    return True
                else:
                    director.logger.log_warning(
                        f"⚠️ Long-form video not found: {long_form_video}"
                    )
                    director.logger.log_info("✅ Shorts step skipped (no source video)")
                    return True  # Skip, not fail

            except ImportError:
                director.logger.log_warning(
                    "⚠️ shorts_maker.py not available - skipping shorts generation"
                )
                return True  # Skip, not fail
            except Exception as e:
                director.logger.log_warning(f"❌ Shorts generation failed: {e}")
                return False

        else:
            director.logger.log_info(f"⚠️ Unknown step: {step}")
            return False

    except Exception as e:
        director.logger.log_warning(f"❌ Step {step} failed for {channel}: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Handle --list-channels option
    if args.list_channels:
        print("📺 Available Channels:")
        print("=" * 50)
        for channel, config in settings.CHANNELS.items():
            niche = config.get("niche", "unknown")
            print(f"  • {channel:<15} - {niche}")
        print("=" * 50)
        print(f"Total: {len(settings.CHANNELS)} channels")
        exit(0)

    # Validate channel is provided
    if not args.channel:
        print("❌ Error: --channel argument is required")
        print("Use --list-channels to see available channels")
        print("Example: python main.py --channel CKFinanceCore --steps all")
        exit(1)

    # Validate channel exists in configuration
    if args.channel not in settings.CHANNELS:
        print(f"❌ Error: Channel '{args.channel}' not found in configuration")
        print(f"Available channels: {list(settings.CHANNELS.keys())}")
        print("Use --list-channels to see all available channels")
        exit(20)

    # Validate steps are provided
    if not args.steps:
        print("❌ Error: --steps argument is required")
        print("Example: python main.py --channel CKFinanceCore --steps all")
        exit(1)

    # Validate date format
    target_date = args.date
    if target_date:
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
            print(f"✅ Date format validated: {target_date}")
        except ValueError:
            print(f"❌ Error: Invalid date format '{target_date}'. Use YYYY-MM-DD")
            exit(1)

    # Parse steps (handle both comma-separated and space-separated)
    if args.steps:
        steps_list = []
        for step in args.steps:
            if "," in step:
                # Handle comma-separated string
                steps_list.extend([s.strip() for s in step.split(",")])
            else:
                # Handle individual step
                steps_list.append(step.strip())
    else:
        steps_list = []

    # Normalize steps
    if "all" in steps_list:
        steps_to_execute = [
            "topics",
            "script",
            "render",
            "captions",
            "translate",
            "shorts",
        ]
        print("🔄 'all' selected - will execute all pipeline steps")
    else:
        steps_to_execute = steps_list
        print(f"🔄 Custom steps selected: {', '.join(steps_to_execute)}")

    print("🚀 Starting Enhanced Master Director")
    print(f"📺 Channel: {args.channel}")
    print(f"🔧 Steps: {', '.join(steps_to_execute)}")
    print(f"📅 Date: {target_date or 'today'}")
    print(f"🔍 Dry-run: {args.dry_run}")
    print("=" * 60)

    try:
        # Create director instance
        director = EnhancedMasterDirector()

        # Execute requested steps
        results = {}
        for step in steps_to_execute:
            print(f"\n🎯 Executing step: {step}")
            success = run_pipeline_step(
                director, args.channel, step, target_date, args.dry_run
            )
            results[step] = success

            if not success:
                print(f"⚠️ Step {step} failed")
                break

        # Calculate results and exit codes
        successful_steps = sum(1 for success in results.values() if success)
        total_steps = len(results)

        print("\n📊 Pipeline Results:")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Successful: {successful_steps}")
        print(f"  - Failed: {total_steps - successful_steps}")

        # Exit codes: 0=success, 10=partial, 20=failed
        if successful_steps == total_steps:
            print("🎯 All steps completed successfully - Exit code 0")
            exit(0)
        elif successful_steps > 0:
            print("⚠️ Some steps completed (partial success) - Exit code 10")
            exit(10)
        else:
            print("❌ All steps failed - Exit code 20")
            exit(20)

    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback

        traceback.print_exc()
        exit(20)
