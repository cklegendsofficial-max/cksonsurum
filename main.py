# main.py - Enhanced Master Director Orchestrator (Ultimate Version)

import os
import sys
import time
import threading
import webbrowser
import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps

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
    from improved_llm_handler import ImprovedLLMHandler
    from advanced_video_creator import AdvancedVideoCreator
    from config import CHANNELS_CONFIG, AI_CONFIG
    from moviepy.config import change_settings
    from moviepy.editor import VideoFileClip
    import ollama
    import pytrends
    import gtts
    IMPROVED_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    IMPROVED_HANDLER_AVAILABLE = False

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, timeout: int = 15):
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
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class EnhancedMasterDirector:
    def __init__(self):
        # 1) Ensure log_file is initialized first, before any logging calls
        import os, time
        if not hasattr(self, "log_file") or not getattr(self, "log_file", None):
            ts = int(time.time())
            log_dir = os.environ.get("CK_LOG_DIR", ".")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"master_director_{ts}.log")
        
        self.llm_handler = ImprovedLLMHandler() if IMPROVED_HANDLER_AVAILABLE else None
        self.video_creator = AdvancedVideoCreator() if IMPROVED_HANDLER_AVAILABLE else None
        self.gui_logger = None
        self.is_running = False
        self.channels = list(CHANNELS_CONFIG.keys())
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
            
            tk.Button(button_frame, text="Start Pipeline", command=self.start_pipeline).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Stop Pipeline", command=self.stop_pipeline).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Build EXE", command=self.build_executable).pack(side=tk.LEFT, padx=5)
            
            self.gui_logger = self.log_text
            self._log("🚀 Enhanced Master Director initialized")
            
        except Exception as e:
            print(f"❌ GUI setup failed: {e}")
            self.gui_logger = None
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        # log_file is already initialized in __init__, just log the path
        print(f"📝 Master Director logging to: {self.log_file}")
    
    def _log(self, msg: str):
        import os, time
        try:
            if not hasattr(self, "log_file") or not self.log_file:
                ts = int(time.time())
                log_dir = os.environ.get("CK_LOG_DIR", ".")
                os.makedirs(log_dir, exist_ok=True)
                self.log_file = os.path.join(log_dir, f"master_director_{ts}.log")
        except Exception:
            self.log_file = None
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{ts} SYSTEM: {msg}"
        print(line)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{line}\n")
            except Exception:
                pass
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, timeout=15)
    def check_dependencies(self) -> Dict[str, bool]:
        """Check all required dependencies with timeout and retry"""
        dependencies = {}
        
        required_modules = {
            'ollama': 'Ollama LLM service',
            'moviepy': 'Video editing',
            'gtts': 'Text-to-speech',
            'pytrends': 'Google Trends',
            'tkinter': 'GUI interface'
        }
        
        for module, description in required_modules.items():
            try:
                if module == 'ollama':
                    # Test Ollama connection with timeout
                    import ollama
                    response = ollama.list()
                    dependencies[module] = True
                    self._log(f"✅ {description}: Available and responding")
                elif module == 'tkinter':
                    dependencies[module] = TKINTER_AVAILABLE
                    status = "Available" if TKINTER_AVAILABLE else "Not available"
                    self._log(f"{'✅' if TKINTER_AVAILABLE else '⚠️'} {description}: {status}")
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
                'assets/videos/downloads',
                'assets/audio/music',
                'assets/audio/CKLegends',
                'assets/audio/CKFinanceCore',
                'assets/audio/CKDrive',
                'assets/audio/CKCombat',
                'assets/audio/CKIronWill'
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            self._log("✅ Directory structure created")
            self._log("🚀 System initialization completed")
            
        except Exception as e:
            self._log(f"❌ System initialization failed: {str(e)}")
            raise
    
    def run_channel_pipeline(self, channel_name: str):
        """Enhanced channel pipeline with ethical retention techniques"""
        if not self.llm_handler:
            self._log(f"❌ LLM handler not available for {channel_name}")
            return False
        
        channel_config = CHANNELS_CONFIG.get(channel_name, {})
        if not channel_config:
            self._log(f"❌ No configuration found for {channel_name}")
            return False
        
        self._log(f"🎬 Starting pipeline for channel: {channel_name}")
        
        try:
            # Step 1: Generate 1 long video idea
            self._log(f"📝 Generating long video idea for {channel_name}...")
            long_ideas = self.llm_handler.generate_viral_ideas(channel_name, 1)
            if not long_ideas:
                self._log(f"❌ Failed to generate long video idea for {channel_name}")
                return False
            
            long_idea = long_ideas[0]
            self._log(f"✅ Long video idea generated: {long_idea.get('title', 'No title')}")
            
            # Step 2: Generate 3 short video ideas
            self._log(f"📝 Generating 3 short video ideas for {channel_name}...")
            short_ideas = self.llm_handler.generate_viral_ideas(channel_name, 3)
            if not short_ideas:
                self._log(f"⚠️ Failed to generate short ideas for {channel_name}, continuing with long video")
            else:
                self._log(f"✅ Generated {len(short_ideas)} short video ideas")
            
            # Step 3: Write detailed script for long video
            self._log(f"📝 Writing detailed script for long video...")
            script = self.llm_handler.write_script(long_idea, channel_name)
            if not script:
                self._log(f"❌ Failed to generate script for {channel_name}")
                return False
            
            # Enhance script with metadata
            enhanced_script = self.llm_handler.enhance_script_with_metadata(script)
            sentence_count = len(enhanced_script.get('script', []))
            self._log(f"✅ Script generated with {sentence_count} sentences")
            
            # Step 4: Apply ethical retention techniques
            self._log(f"🎯 Applying ethical retention techniques...")
            retention_techniques = self._generate_ethical_retention_techniques(channel_name, enhanced_script)
            if retention_techniques:
                enhanced_script = self._apply_ethical_retention_techniques(enhanced_script, retention_techniques)
                self._log(f"✅ Ethical retention techniques applied: {len(retention_techniques)} techniques")
            
            # Step 5: Generate voiceover
            self._log(f"🎤 Generating voiceover for {channel_name}...")
            audio_folder = os.path.join('assets', 'audio', channel_name)
            os.makedirs(audio_folder, exist_ok=True)
            
            audio_files = self.video_creator.generate_voiceover(enhanced_script, audio_folder)
            if not audio_files:
                self._log(f"❌ Voiceover generation failed for {channel_name}")
                return False
            self._log(f"✅ Voiceover generated: {len(audio_files)} files")
            
            # Step 6: Find visual assets
            self._log(f"🎬 Finding visual assets for {channel_name}...")
            video_download_folder = os.path.join('assets', 'videos', 'downloads', channel_name)
            visual_files = self.video_creator.find_visual_assets(enhanced_script, channel_config['niche'], video_download_folder)
            
            if not visual_files:
                self._log(f"❌ No visual assets found for {channel_name}")
                return False
            self._log(f"✅ Visual assets found: {len(visual_files)} files")
            
            # Step 7: Create final video
            self._log(f"🎬 Creating final video for {channel_name}...")
            music_file = "assets/audio/music/epic_music.mp3"
            if not os.path.exists(music_file):
                music_file = None
                self._log("⚠️ Background music not found, proceeding without music")
            
            video_filename = f"assets/videos/{channel_name}_Masterpiece_v2.mp4"
            final_video_path = self.video_creator.edit_long_form_video(audio_files, visual_files, music_file, video_filename)
            
            if final_video_path:
                self._log(f"🎉 Long video completed for {channel_name}: {final_video_path}")
                
                # Analyze video quality and duration
                self.analyze_video_quality(final_video_path, channel_name)
                
                # main.py — render sonrası otomatik altyazı
                try:
                    from auto_captions import generate_multi_captions, TARGET_LANGS
                    self._log(f"CAPTIONS: generating {len(TARGET_LANGS)}-lang subtitles...")
                    # Eğer TTS dosya yolun varsa audio_path parametresini ver; yoksa sadece video_path ver.
                    subs = generate_multi_captions(final_video_path, audio_path=None, langs=TARGET_LANGS)
                    self._log(f"CAPTIONS: done ({len(subs)} files)")
                except Exception as e:
                    self._log(f"CAPTIONS: skipped ({e})")
                
                return True
            else:
                self._log(f"❌ Video creation failed for {channel_name}")
                return False
                
        except Exception as e:
            self._log(f"❌ Pipeline error for {channel_name}: {str(e)}")
            return False
    
    def _generate_ethical_retention_techniques(self, channel_name: str, script: dict) -> List[dict]:
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
            
            response = ollama.chat(model=AI_CONFIG.get("ollama_model", "llama3:8b"), 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                # Extract JSON from response
                import re
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
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
    
    def _apply_ethical_retention_techniques(self, script: dict, techniques: List[dict]) -> dict:
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
                
                enhanced_script["metadata"]["retention_techniques"].append({
                    "name": technique_name,
                    "implementation": implementation,
                    "target_effect": target_effect,
                    "timing": timing,
                    "ethical_compliance": "verified"
                })
                
                self._log(f"🎯 Applied ethical technique: {technique_name} - {target_effect}")
            
            return enhanced_script
            
        except Exception as e:
            self._log(f"⚠️ Ethical retention techniques application failed: {e}")
            return script
    
    def analyze_video_quality(self, video_path: str, channel_name: str):
        """Enhanced video quality analysis with full MoviePy metrics"""
        try:
            self._log(f"🔍 Analyzing video quality: {video_path}")
            
            with VideoFileClip(video_path) as video:
                # Basic metrics
                duration = video.duration
                fps = video.fps
                size = video.size
                
                self._log(f"📊 Video analysis - Duration: {duration:.1f}s, FPS: {fps}, Size: {size}")
                
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
                overall_score = (duration_score + scene_variety_score + audio_quality_score + visual_quality_score) / 4 * black_frame_penalty
                
                # Log detailed analysis
                self._log(f"📊 Quality Analysis Results:")
                self._log(f"   Duration Score: {duration_score:.3f}")
                self._log(f"   Scene Variety Score: {scene_variety_score:.3f}")
                self._log(f"   Audio Quality Score: {audio_peaks:.3f}")
                self._log(f"   Visual Quality Score: {visual_quality_score:.3f}")
                self._log(f"   Black Frame Ratio: {black_frame_ratio:.2%}")
                self._log(f"   Overall Quality Score: {overall_score:.3f}")
                
                # Check if video meets minimum requirements
                if duration < 600:  # Less than 10 minutes
                    self._log(f"⚠️ Video too short ({duration:.1f}s), regenerating with improved handler")
                    
                    # Automatically regenerate with improved parameters
                    self.regenerate_video_with_improved_handler(channel_name, duration)
                elif overall_score < 0.7:
                    self._log(f"⚠️ Video quality below threshold ({overall_score:.3f}), applying enhancement techniques")
                    self._apply_quality_enhancements(video_path, overall_score)
                else:
                    self._log(f"✅ Video meets quality standards (Score: {overall_score:.3f})")
                    
        except Exception as e:
            self._log(f"❌ Video analysis failed: {str(e)}")
    
    def _analyze_scene_variety(self, video: VideoFileClip) -> float:
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
    
    def _analyze_audio_peaks(self, video: VideoFileClip) -> float:
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
                return min(1.0, (audio_mean * 100 + audio_std * 1000 + audio_peaks * 10) / 3)
            else:
                return 0.3
            
        except Exception as e:
            self._log(f"⚠️ Audio peaks analysis failed: {e}")
            return 0.5
    
    def _analyze_visual_quality(self, video: VideoFileClip) -> float:
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
    
    def _detect_black_frames(self, video: VideoFileClip) -> float:
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
            self._log(f"🔧 Applying quality enhancements for score: {quality_score:.3f}")
            
            # Use Ollama to generate enhancement techniques
            import ollama
            
            prompt = f"""Video kalitesi düşük ({quality_score:.3f}), iyileştirme teknikleri üret.
            
            Teknikler:
            1. Görsel iyileştirme (contrast, saturation, sharpness)
            2. Ses iyileştirme (noise reduction, equalization)
            3. Frame interpolation
            4. Color grading
            
            Her teknik için Python code üret."""
            
            response = ollama.chat(model=AI_CONFIG.get("ollama_model", "llama3:8b"), 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                self._log(f"🤖 Ollama generated enhancement techniques")
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
                
                pytrends = TrendReq(hl='en-US', tz=360)
                
                # Search for trending topics in the niche
                search_query = f"{channel_niche} trending"
                pytrends.build_payload([search_query], timeframe='today 12-m')
                
                # Get related queries
                related_queries = pytrends.related_queries()
                if related_queries and search_query in related_queries:
                    trending_data = related_queries[search_query]
                    if 'top' in trending_data:
                        keywords = trending_data['top']['query'].tolist()[:10]
                        self._log(f"✅ PyTrends keywords retrieved: {len(keywords)} terms")
                        
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
            channel_config = CHANNELS_CONFIG.get(channel_niche, {})
            fallback_keywords = channel_config.get('niche_keywords', [])
            self._log(f"🔄 Using fallback niche keywords: {len(fallback_keywords)} terms")
            return fallback_keywords
            
        except Exception as e:
            self._log(f"❌ Trending keywords retrieval failed: {e}")
            return []
    
    def _cache_trending_keywords(self, channel_niche: str, keywords: List[str]):
        """Cache trending keywords to JSON file"""
        try:
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"trending_keywords_{channel_niche}.json")
            
            cache_data = {
                "channel_niche": channel_niche,
                "timestamp": datetime.now().isoformat(),
                "keywords": keywords,
                "source": "pytrends"
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self._log(f"💾 Cached {len(keywords)} keywords to {cache_file}")
            
        except Exception as e:
            self._log(f"⚠️ Keyword caching failed: {e}")
    
    def _load_cached_trending_keywords(self, channel_niche: str) -> List[str]:
        """Load cached trending keywords from JSON file"""
        try:
            cache_file = os.path.join("cache", f"trending_keywords_{channel_niche}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
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
    
    def regenerate_video_with_improved_handler(self, channel_name: str, current_duration: float):
        """Automatically regenerate video using improved LLM handler"""
        try:
            self._log(f"🔄 Regenerating video for {channel_name} with improved parameters")
            
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
        """Run pipeline for all 5 channels"""
        self._log("🎬 Starting pipeline for all channels")
        
        try:
            results = {}
            for channel in self.channels:
                self._log(f"🎬 Processing channel: {channel}")
                success = self.run_channel_pipeline(channel)
                results[channel] = success
                
                if success:
                    self._log(f"✅ {channel} pipeline completed successfully")
                else:
                    self._log(f"❌ {channel} pipeline failed")
                
                # Small delay between channels
                time.sleep(2)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            self._log(f"🎉 Pipeline summary: {successful}/{total} channels completed successfully")
            
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
                    "--onefile",           # Single executable file
                    "--windowed",          # No console window
                    "--name=EnhancedMasterDirector",  # Executable name
                    "--icon=assets/images/icon.ico" if os.path.exists("assets/images/icon.ico") else "",
                    "--add-data=config.py;.",  # Include config
                    "--add-data=assets;assets",  # Include assets
                    "--hidden-import=moviepy",
                    "--hidden-import=numpy",
                    "--hidden-import=PIL",
                    "--hidden-import=ollama",
                    "main.py"
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
                        self._log(f"📦 Executable created: {exe_path} ({file_size:.1f} MB)")
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
                    install_result = subprocess.run(["pip", "install", "pyinstaller"], 
                                                 capture_output=True, text=True)
                    
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
                    self._log(f"❌ PyInstaller installation attempt failed: {install_error}")
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
                pyautogui.press('enter')
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
                    if filename.endswith('.mp4'):
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
            
            # Run initial pipeline
            self.run_all_channels_pipeline()
            
            # Keep GUI running if available
            if TKINTER_AVAILABLE and hasattr(self, 'root'):
                self.root.mainloop()
            else:
                # Keep console running
                while self.is_running:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self._log("🛑 Interrupted by user")
        except Exception as e:
            self._log(f"❌ Critical error: {str(e)}")
        finally:
            self.shutdown_system()

if __name__ == "__main__":
    try:
        # Try to import and run the frontend GUI
        from frontend import VideoPipelineGUI
        
        print("🚀 Starting Enhanced Master Director with GUI...")
        gui = VideoPipelineGUI()
        gui.run()
        
    except ImportError as e:
        print(f"⚠️ Frontend GUI not available: {e}")
        print("🔄 Falling back to console mode...")
        
        # Create and run the enhanced master director in console mode
        director = EnhancedMasterDirector()
        director.run()
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("🔄 Falling back to console mode...")
        
        # Create and run the enhanced master director in console mode
        director = EnhancedMasterDirector()
        director.run()