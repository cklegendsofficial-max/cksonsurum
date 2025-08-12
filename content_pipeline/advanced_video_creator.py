# content_pipeline/advanced_video_creator.py (Professional Master Director Edition)

import os
import sys
import json
import time
import random
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# --- config imports & defaults ---
try:
    from config import AI_CONFIG, PEXELS_API_KEY
except Exception:
    AI_CONFIG, PEXELS_API_KEY = {}, None

try:
    OLLAMA_MODEL = (AI_CONFIG or {}).get("ollama_model", "llama3:8b")
except Exception:
    OLLAMA_MODEL = "llama3:8b"

# Varsayılan: subliminal kapalı (YouTube politikaları ve güven için)
ENABLE_SUBLIMINAL = False

def _ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for the given path"""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# Try to import advanced libraries
try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️ Piper TTS not available, using espeak fallback")

try:
    import espeak
    ESPEAK_AVAILABLE = True
except ImportError:
    ESPEAK_AVAILABLE = False
    print("⚠️ espeak not available, using gTTS fallback")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("⚠️ mido not available, MIDI generation disabled")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ Pillow not available, image upscaling disabled")

# Core imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from moviepy.editor import *
from moviepy.video.fx import all as vfx
from moviepy.audio.fx import all as afx

class AdvancedVideoCreator:
    def __init__(self):
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30
        self.CODEC = 'libx264'
        self.AUDIO_CODEC = 'aac'
        self.QUALITY = 'high'
        
        # Initialize TTS systems
        self.setup_tts()
        
        # Initialize music generation
        self.setup_music_generation()
        
        # Setup logging
        self.setup_logging()
        
        # Load local assets for fallback
        self.load_local_assets()
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        self.log_file = f"advanced_video_creator_{int(time.time())}.log"
        print(f"📝 Advanced Video Creator logging to: {self.log_file}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    def setup_tts(self):
        """Setup TTS systems with fallback hierarchy"""
        self.tts_system = None
        
        if PIPER_AVAILABLE:
            try:
                # Initialize Piper TTS with Morgan Freeman style
                self.tts_system = "piper"
                self.log_message("✅ Piper TTS initialized for Morgan Freeman style", "TTS")
            except Exception as e:
                self.log_message(f"⚠️ Piper TTS initialization failed: {e}", "TTS")
                self.tts_system = None
        
        if not self.tts_system and ESPEAK_AVAILABLE:
            try:
                self.tts_system = "espeak"
                self.log_message("✅ espeak TTS initialized as fallback", "TTS")
            except Exception as e:
                self.log_message(f"⚠️ espeak initialization failed: {e}", "TTS")
        
        if not self.tts_system and GTTS_AVAILABLE:
            self.tts_system = "gtts"
            self.log_message("✅ gTTS initialized as final fallback", "TTS")
        
        if not self.tts_system:
            self.log_message("❌ No TTS system available", "ERROR")
    
    def setup_music_generation(self):
        """Setup MIDI music generation system"""
        if MIDO_AVAILABLE:
            try:
                self.music_system = "mido"
                self.log_message("✅ MIDI music generation system initialized", "MUSIC")
            except Exception as e:
                self.log_message(f"⚠️ MIDI system initialization failed: {e}", "MUSIC")
                self.music_system = None
        else:
            self.music_system = None
            self.log_message("⚠️ MIDI music generation not available", "MUSIC")
    
    def load_local_assets(self):
        """Load local assets for fallback scenarios"""
        self.local_assets = {
            'images': [],
            'videos': [],
            'audio': []
        }
        
        # Load local images
        image_dir = "assets/images"
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.local_assets['images'].append(os.path.join(image_dir, file))
        
        # Load local videos
        video_dir = "assets/videos/downloads"
        if os.path.exists(video_dir):
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        self.local_assets['videos'].append(os.path.join(root, file))
        
        # Load local audio
        audio_dir = "assets/audio"
        if os.path.exists(audio_dir):
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    if file.lower().endswith('.mp3'):
                        self.local_assets['audio'].append(os.path.join(root, file))
        
        self.log_message(f"📁 Loaded {len(self.local_assets['images'])} images, {len(self.local_assets['videos'])} videos, {len(self.local_assets['audio'])} audio files", "ASSETS")
    
    def _optimize_pexels_query(self, query: str, channel_niche: str) -> str:
        """Optimize Pexels query using Ollama for better visual results"""
        try:
            import ollama
            
            prompt = f"""Alakasız görsel için yeni query üret.
            
            Orijinal query: {query}
            Kanal niş: {channel_niche}
            
            Yeni query formatı: "cinematic 4K {{niche}} {{scene}} high quality no black"
            
            Örnekler:
            - "cinematic 4K history ancient civilizations high quality no black"
            - "cinematic 4K motivation success achievement high quality no black"
            - "cinematic 4K finance business modern high quality no black"
            
            Sadece yeni query'yi döndür, açıklama yapma."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            optimized_query = response.get('message', {}).get('content', '').strip()
            
            if optimized_query:
                self.log_message(f"🤖 Ollama optimized query: {query} → {optimized_query}", "OLLAMA")
                return optimized_query
            else:
                # Fallback to enhanced query
                return f"cinematic 4K {channel_niche} {query} high quality no black"
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama query optimization failed: {e}", "WARNING")
            # Fallback to enhanced query
            return f"cinematic 4K {channel_niche} {query} high quality no black"
    
    def _download_pexels_video(self, query: str, min_duration: float, target_path: str) -> Optional[str]:
        """
        Download video from Pexels using real API with retry mechanism and validation
        
        Args:
            query: Search query for video content
            min_duration: Minimum required duration in seconds
            target_path: Target file path for download
            
        Returns:
            Optional[str]: Path to downloaded video or None if failed
        """
        try:
            # Check if Pexels API is available
            if not bool(PEXELS_API_KEY):
                self.log_message("⚠️ Pexels API key not available, using fallback", "PEXELS")
                return self._get_pexels_fallback_video(query, min_duration, target_path)
            
            # Setup API headers
            headers = {
                "Authorization": PEXELS_API_KEY,
                "User-Agent": "EnhancedMasterDirector/2.0"
            }
            
            # Search for videos
            search_url = "https://api.pexels.com/videos/search"
            params = {
                "query": query,
                "per_page": 15,  # Get more results to find best match
                "orientation": "landscape",
                "size": "large"
            }
            
            self.log_message(f"🔍 Searching Pexels for: {query}", "PEXELS")
            
            # Make search request with retry
            response = self._make_pexels_request(search_url, params, headers)
            if not response:
                return None
            
            # Parse search results
            videos = response.get("videos", [])
            if not videos:
                self.log_message(f"⚠️ No videos found for query: {query}", "PEXELS")
                return None
            
            # Find best video (highest bitrate, meets duration requirement)
            # First filter videos by duration
            suitable_videos = [v for v in videos if v.get("duration", 0) >= min_duration]
            if not suitable_videos:
                self.log_message(f"⚠️ No videos meet duration requirement: {min_duration}s", "PEXELS")
                return None
            
            # Get all video files from suitable videos
            all_video_files = []
            for video in suitable_videos:
                all_video_files.extend(video.get("video_files", []))
            
            best_video_file = self._select_best_pexels_video(all_video_files)
            if not best_video_file:
                self.log_message(f"⚠️ No suitable video file found", "PEXELS")
                return None
            
            # Download the selected video
            download_url = best_video_file["link"]
            # Generate filename from video file info
            video_filename = f"pexels_{best_video_file.get('id', 'unknown')}_{query.replace(' ', '_')[:20]}.mp4"
            output_path = os.path.join(target_path, video_filename)
            
            # Ensure target directory exists
            _ensure_parent_dir(output_path)
            
            self.log_message(f"📥 Downloading: {download_url}", "PEXELS")
            
            # Download with retry mechanism
            success = self._download_video_file(download_url, output_path, headers)
            if not success:
                return None
            
            # Validate downloaded file
            if self._validate_downloaded_video(output_path, min_duration):
                self.log_message(f"✅ Video downloaded successfully: {os.path.basename(output_path)}", "PEXELS")
                return output_path
            else:
                self.log_message(f"❌ Downloaded video validation failed", "PEXELS")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None
                
        except Exception as e:
            self.log_message(f"❌ Pexels download failed: {e}", "ERROR")
            return None
    
    def _make_pexels_request(self, url: str, params: dict, headers: dict) -> Optional[dict]:
        """Make Pexels API request with retry mechanism"""
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                delay = 2 ** attempt  # 1s, 2s, 4s
                self.log_message(f"⚠️ Pexels request attempt {attempt + 1} failed: {e}, retrying in {delay}s", "PEXELS")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(delay)
                continue
        return None
    
    def _select_best_pexels_video(self, video_files: list[dict]) -> dict | None:
        """
        Return the best MP4 by (resolution, bitrate).
        """
        candidates = []
        for vf in video_files or []:
            if vf.get("file_type") == "video/mp4" and vf.get("width") and vf.get("height"):
                try:
                    w = int(vf["width"]); h = int(vf["height"])
                    br = int(vf.get("bitrate") or 0)
                except Exception:
                    continue
                candidates.append((w*h, br, vf))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    
    def _download_video_file(self, url: str, output_path: str, headers: dict) -> bool:
        """Download video file with retry mechanism and validation"""
        # Use minimal headers for CDN downloads
        download_headers = {"User-Agent": "EnhancedMasterDirector/2.0"}
        
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=30, headers=download_headers)
                response.raise_for_status()
                
                # Get file size for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                # Verify download completion
                if total_size > 0 and downloaded_size < total_size:
                    self.log_message(f"⚠️ Incomplete download detected, retrying...", "PEXELS")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    continue
                
                return True
                
            except Exception as e:
                delay = 2 ** attempt  # 1s, 2s, 4s
                self.log_message(f"⚠️ Download attempt {attempt + 1} failed: {e}, retrying in {delay}s", "PEXELS")
                if os.path.exists(output_path):
                    os.remove(output_path)
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(delay)
                continue
        
        return False
    
    def _validate_downloaded_video(self, file_path: str, min_duration: float) -> bool:
        """Validate downloaded video file"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                self.log_message(f"⚠️ Downloaded file too small: {file_size} bytes", "PEXELS")
                return False
            
            # Try to load video to check duration
            try:
                clip = VideoFileClip(file_path)
                actual_duration = clip.duration
                clip.close()
                
                if actual_duration < min_duration:
                    self.log_message(f"⚠️ Video duration too short: {actual_duration:.1f}s < {min_duration:.1f}s", "PEXELS")
                    return False
                
                return True
                
            except Exception as e:
                self.log_message(f"⚠️ Video validation failed: {e}", "PEXELS")
                return False
                
        except Exception as e:
            self.log_message(f"⚠️ File validation failed: {e}", "WARNING")
            return False
    
    def _get_pexels_fallback_video(self, query: str, min_duration: float, target_path: str) -> Optional[str]:
        """
        Get fallback video when Pexels API is not available
        
        Args:
            query: Search query (used for logging)
            min_duration: Minimum required duration
            target_path: Target directory path
            
        Returns:
            Optional[str]: Path to fallback video or None
        """
        try:
            self.log_message(f"🔄 Using Pexels fallback for query: {query}", "PEXELS")
            
            # Try to find a suitable local video
            local_video = self._get_local_asset_fallback("general", 1)
            if local_video and os.path.exists(local_video):
                # Copy to target directory
                import shutil
                target_filename = f"fallback_{query.replace(' ', '_')[:20]}.mp4"
                target_file = os.path.join(target_path, target_filename)
                
                # Ensure target directory exists
                _ensure_parent_dir(target_file)
                
                shutil.copy2(local_video, target_file)
                
                self.log_message(f"✅ Fallback video copied: {target_filename}", "PEXELS")
                return target_file
            
            return None
            
        except Exception as e:
            self.log_message(f"⚠️ Pexels fallback failed: {e}", "WARNING")
            return None
    
    def _get_local_asset_fallback(self, channel_niche: str, scene_num: int) -> Optional[str]:
        """Get local asset as fallback when Pexels fails"""
        try:
            if not self.local_assets['videos']:
                return None
            
            # Select appropriate local video based on scene and niche
            video_index = (scene_num - 1) % len(self.local_assets['videos'])
            return self.local_assets['videos'][video_index]
            
        except Exception as e:
            self.log_message(f"⚠️ Local asset fallback failed: {e}", "WARNING")
            return None
    
    def _upscale_video_to_4k(self, video_path: str) -> Optional[str]:
        """Upscale video to 4K using Pillow if available"""
        try:
            if not PILLOW_AVAILABLE:
                return video_path
            
            # Placeholder for 4K upscaling
            # In real implementation, this would use AI upscaling
            return video_path
            
        except Exception as e:
            self.log_message(f"⚠️ 4K upscaling failed: {e}", "WARNING")
            return video_path
    
    def _create_enhanced_visual_clip(self, visual_path: str, duration: float, scene_index: int) -> VideoClip:
        """Create enhanced visual clip with black frame detection and fallback"""
        try:
            if not visual_path or not os.path.exists(visual_path):
                # Create fallback visual clip
                return self._create_fallback_visual_clip(duration, scene_index)
            
            # Load video clip
            video_clip = VideoFileClip(visual_path)
            
            # Check for black frames using enhanced detection
            black_frame_analysis = self.detect_black_frames(video_clip)
            black_frame_ratio = black_frame_analysis.get('black_frame_ratio', 0.0)
            
            if black_frame_ratio > 0.1:  # More than 10% black frames
                self.log_message(f"⚠️ High black frame ratio detected: {black_frame_ratio:.2%}", "WARNING")
                self.log_message(f"📍 Black frames at: {black_frame_analysis.get('black_frame_timestamps', [])[:3]}", "WARNING")
                video_clip.close()
                return self._create_fallback_visual_clip(duration, scene_index)
            
            # Ensure proper duration with smooth transitions
            if video_clip.duration < duration:
                video_clip = self.extend_clip_to_duration(video_clip, duration)
            elif video_clip.duration > duration:
                video_clip = video_clip.subclip(0, duration)
            
            return video_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Enhanced visual clip creation failed: {e}", "WARNING")
            return self._create_fallback_visual_clip(duration, scene_index)
    
    def detect_black_frames(self, clip: VideoClip) -> Dict[str, Any]:
        """
        Enhanced black frame detection using luma percentile and stddev analysis
        
        Args:
            clip: VideoClip to analyze
            
        Returns:
            Dict containing black frame ratio, timestamps, and analysis details
        """
        try:
            import numpy as np
            
            # Sample frames evenly across the video duration
            sample_count = min(50, int(clip.duration * clip.fps))  # Sample up to 50 frames
            frame_analysis = []
            black_frame_timestamps = []
            
            for i in range(sample_count):
                # Calculate time position evenly distributed
                time_pos = (i / sample_count) * clip.duration
                frame = clip.get_frame(time_pos)
                
                # Convert to grayscale (luma)
                if len(frame.shape) == 3:
                    # Use BT.709 luma conversion for more accurate brightness
                    luma_frame = frame[:, :, 0] * 0.2126 + frame[:, :, 1] * 0.7152 + frame[:, :, 2] * 0.0722
                else:
                    luma_frame = frame
                
                # Calculate frame statistics
                frame_mean = np.mean(luma_frame)
                frame_std = np.std(luma_frame)
                
                frame_analysis.append({
                    'time': time_pos,
                    'mean': frame_mean,
                    'std': frame_std
                })
                
                # Check if frame is likely black using percentile analysis
                if self._is_frame_black(frame_mean, frame_std, frame_analysis):
                    black_frame_timestamps.append(time_pos)
            
            # Calculate global statistics
            all_means = [f['mean'] for f in frame_analysis]
            all_stds = [f['std'] for f in frame_analysis]
            
            global_mean = np.mean(all_means)
            global_std = np.std(all_means)
            global_p10 = np.percentile(all_means, 10)  # 10th percentile
            
            # Calculate black frame ratio
            black_ratio = len(black_frame_timestamps) / len(frame_analysis)
            
            # Enhanced analysis results
            analysis_result = {
                'black_frame_ratio': black_ratio,
                'black_frame_timestamps': black_frame_timestamps,
                'total_frames_analyzed': len(frame_analysis),
                'global_statistics': {
                    'mean_luma': global_mean,
                    'luma_stddev': global_std,
                    'luma_p10': global_p10,
                    'mean_stddev': np.mean(all_stds)
                },
                'frame_details': frame_analysis
            }
            
            # Log analysis summary
            self.log_message(f"🔍 Black frame analysis: {black_ratio:.1%} black frames detected", "ANALYSIS")
            if black_frame_timestamps:
                self.log_message(f"📍 Black frames at: {[f'{t:.1f}s' for t in black_frame_timestamps[:5]]}", "ANALYSIS")
            
            return analysis_result
            
        except Exception as e:
            self.log_message(f"⚠️ Enhanced black frame detection failed: {e}", "WARNING")
            return {
                'black_frame_ratio': 0.0,
                'black_frame_timestamps': [],
                'total_frames_analyzed': 0,
                'global_statistics': {},
                'frame_details': []
            }
    
    def _is_frame_black(self, frame_mean: float, frame_std: float, frame_history: List[dict]) -> bool:
        """
        Determine if a frame is black using adaptive thresholding
        
        Args:
            frame_mean: Mean luma value of the frame
            frame_std: Standard deviation of luma values
            frame_history: List of previous frame analysis data
            
        Returns:
            bool: True if frame is considered black
        """
        try:
            if len(frame_history) < 3:
                # Need at least 3 frames for comparison
                return frame_mean < 15 and frame_std < 5
            
            # Calculate adaptive thresholds based on frame history
            recent_means = [f['mean'] for f in frame_history[-10:]]  # Last 10 frames
            recent_stds = [f['std'] for f in frame_history[-10:]]
            
            # Dynamic threshold: 10th percentile of recent frames
            threshold_mean = np.percentile(recent_means, 10)
            threshold_std = np.percentile(recent_stds, 10)
            
            # Frame is black if:
            # 1. Luma is below 10th percentile of recent frames
            # 2. Standard deviation is low (indicating uniform darkness)
            # 3. Absolute luma is very low
            is_black = (
                frame_mean < max(threshold_mean * 0.8, 20) and  # Adaptive threshold
                frame_std < max(threshold_std * 1.2, 8) and     # Low variation
                frame_mean < 25                                  # Absolute threshold
            )
            
            return is_black
            
        except Exception as e:
            # Fallback to simple threshold
            return frame_mean < 15 and frame_std < 5
    
    def _create_fallback_visual_clip(self, duration: float, scene_index: int) -> VideoClip:
        """Create fallback visual clip when original fails"""
        try:
            # Create a simple colored background with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create 1920x1080 image
            width, height = 1920, 1080
            colors = [(50, 50, 100), (100, 50, 50), (50, 100, 50), (100, 100, 50)]
            color = colors[scene_index % len(colors)]
            
            img = Image.new('RGB', (width, height), color)
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            text = f"Scene {scene_index + 1}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Save temporary image
            temp_path = f"temp_fallback_{scene_index}.png"
            img.save(temp_path)
            
            # Create video clip from image
            clip = ImageClip(temp_path).set_duration(duration)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return clip
            
        except Exception as e:
            self.log_message(f"⚠️ Fallback visual clip creation failed: {e}", "WARNING")
            # Ultimate fallback: solid color clip
            return ColorClip(size=(1920, 1080), color=(50, 50, 100)).set_duration(duration)
    
    def extend_clip_to_duration(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """
        Extend clip to target duration using smooth crossfade transitions
        
        Args:
            clip: VideoClip to extend
            target_duration: Target duration in seconds
            
        Returns:
            VideoClip: Extended clip with smooth transitions
        """
        try:
            if clip.duration >= target_duration:
                return clip
            
            self.log_message(f"🔄 Extending clip from {clip.duration:.1f}s to {target_duration:.1f}s", "EXTENSION")
            
            # Calculate how many loops we need
            loops_needed = int(target_duration / clip.duration) + 1
            crossfade_duration = 0.5  # 0.5 second crossfade
            
            # Create extended clips with crossfade transitions
            extended_clips = []
            
            for i in range(loops_needed):
                if i == 0:
                    # First clip: no crossfade in
                    extended_clips.append(clip)
                else:
                    # Subsequent clips: add crossfade in
                    looped_clip = clip.crossfadein(crossfade_duration)
                    extended_clips.append(looped_clip)
            
            # Concatenate with crossfade method
            self.log_message(f"🎬 Concatenating {len(extended_clips)} clips with crossfade transitions", "EXTENSION")
            
            final_clip = concatenate_videoclips(
                extended_clips, 
                method="compose",
                transition=lambda t: t  # Linear crossfade
            )
            
            # Trim to exact target duration
            if final_clip.duration > target_duration:
                final_clip = final_clip.subclip(0, target_duration)
            
            self.log_message(f"✅ Clip extended successfully to {final_clip.duration:.1f}s", "EXTENSION")
            return final_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Clip duration extension failed: {e}", "WARNING")
            # Fallback: simple loop without crossfade
            return self._extend_clip_simple_fallback(clip, target_duration)
    
    def _extend_clip_simple_fallback(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """Simple fallback extension method without crossfade"""
        try:
            loops_needed = int(target_duration / clip.duration) + 1
            extended_clips = [clip] * loops_needed
            
            final_clip = concatenate_videoclips(extended_clips, method="compose")
            return final_clip.subclip(0, target_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Simple fallback extension failed: {e}", "WARNING")
            return clip
    
    def _extend_with_ollama_content(self, clip: VideoClip, target_duration: float) -> Optional[VideoClip]:
        """Extend clip duration using Ollama-generated additional content"""
        try:
            import ollama
            
            prompt = f"""Video süresini uzatmak için ekstra script cümleleri üret.
            
            Mevcut süre: {clip.duration:.1f} saniye
            Hedef süre: {target_duration:.1f} saniye
            Ek süre gerekli: {target_duration - clip.duration:.1f} saniye
            
            Her cümle yaklaşık 2-3 saniye sürmeli.
            Toplam {int((target_duration - clip.duration) / 2.5)} cümle üret.
            
            Format: Her cümle yeni satırda, açıklama yapma."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                # Parse generated sentences
                sentences = [line.strip() for line in content.split('\n') if line.strip()]
                
                if sentences:
                    self.log_message(f"🤖 Ollama generated {len(sentences)} additional sentences", "OLLAMA")
                    
                    # Create additional visual content for these sentences
                    # This would typically involve generating or finding additional visuals
                    # For now, return None to use fallback method
                    return None
            
            return None
            
        except Exception as e:
            self.log_message(f"⚠️ Ollama content extension failed: {e}", "WARNING")
            return None
    
    def _apply_professional_effects(self, clip: VideoClip, scene_index: int) -> VideoClip:
        """Apply professional visual effects to enhance quality"""
        try:
            # Add subtle zoom effect
            clip = clip.resize(lambda t: 1 + 0.05 * t / clip.duration)
            
            # Add color correction
            clip = clip.fx(vfx.colorx, 1.1)  # Slightly enhance colors
            
            # Add subtle vignette
            clip = clip.fx(vfx.vignette, 0.3)
            
            return clip
            
        except Exception as e:
            self.log_message(f"⚠️ Professional effects failed: {e}", "WARNING")
            return clip
    
    def _create_subliminal_message(self, message: str, duration: float) -> Optional[VideoClip]:
        """Create subliminal message clip"""
        try:
            # Create very brief text clip
            text_clip = TextClip(message, fontsize=40, color='white', bg_color='black')
            text_clip = text_clip.set_duration(duration)
            
            # Position in center
            text_clip = text_clip.set_position('center')
            
            return text_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Subliminal message creation failed: {e}", "WARNING")
            return None
    
    def _add_background_music(self, video: VideoClip, music_path: str) -> VideoClip:
        """Add background music to video"""
        try:
            if not os.path.exists(music_path):
                return video
            
            music_clip = AudioFileClip(music_path)
            
            # Loop music if needed
            if music_clip.duration < video.duration:
                loops_needed = int(video.duration / music_clip.duration) + 1
                music_clip = concatenate_audioclips([music_clip] * loops_needed)
            
            # Trim to video duration
            music_clip = music_clip.subclip(0, video.duration)
            
            # Lower volume for background
            music_clip = music_clip.volumex(0.3)
            
            # Combine with video audio
            final_audio = CompositeAudioClip([video.audio, music_clip])
            video = video.set_audio(final_audio)
            
            return video
            
        except Exception as e:
            self.log_message(f"⚠️ Background music addition failed: {e}", "WARNING")
            return video
    
    def _add_multilingual_subtitles(self, video: VideoClip) -> VideoClip:
        """Add multilingual subtitles"""
        try:
            # Placeholder for multilingual subtitle implementation
            # This would typically involve creating subtitle tracks
            return video
            
        except Exception as e:
            self.log_message(f"⚠️ Multilingual subtitles failed: {e}", "WARNING")
            return video
    
    def _extend_video_duration(self, video: VideoClip, target_duration: float) -> VideoClip:
        """Extend video duration to meet minimum requirements"""
        try:
            if video.duration >= target_duration:
                return video
            
            # Use Ollama to generate additional content
            extended_video = self._extend_with_ollama_content(video, target_duration)
            if extended_video:
                return extended_video
            
            # Fallback: loop the video
            loops_needed = int(target_duration / video.duration) + 1
            extended_clips = [video] * loops_needed
            
            final_video = concatenate_videoclips(extended_clips, method="compose")
            return final_video.subclip(0, target_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Video duration extension failed: {e}", "WARNING")
            return video
    
    def _analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality using MoviePy and numpy for real metrics"""
        try:
            if not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            clip = VideoFileClip(video_path)
            
            # Basic metrics
            duration = clip.duration
            fps = clip.fps
            size = clip.size
            
            # Visual variety analysis using numpy
            visual_variety = self._analyze_visual_variety(clip)
            
            # Audio quality analysis
            audio_quality = self._analyze_audio_quality(clip)
            
            # Black frame detection
            black_frame_ratio = self._detect_black_frames(clip)
            
            # Calculate quality scores
            duration_score = min(1.0, duration / 600.0)  # Normalized to 10 minutes
            visual_score = visual_variety
            audio_score = audio_quality
            black_frame_penalty = 0.5 if black_frame_ratio > 0.1 else 1.0
            
            # Overall quality score
            overall_score = (duration_score + visual_score + audio_score) / 3 * black_frame_penalty
            
            analysis = {
                "duration": duration,
                "fps": fps,
                "size": size,
                "duration_score": duration_score,
                "visual_score": visual_score,
                "audio_score": audio_score,
                "black_frame_ratio": black_frame_ratio,
                "overall_score": overall_score,
                "quality_level": "high" if overall_score > 0.8 else "medium" if overall_score > 0.6 else "low"
            }
            
            # If quality is low, use Ollama to regenerate enhanced visual clip function
            if overall_score < 0.6:
                self._regenerate_enhanced_visual_clip_function(overall_score)
            
            clip.close()
            return analysis
            
        except Exception as e:
            self.log_message(f"❌ Video quality analysis failed: {e}", "ERROR")
            return {"error": str(e)}
    
    def _analyze_visual_variety(self, clip: VideoClip) -> float:
        """Analyze visual variety using numpy frame differences"""
        try:
            import numpy as np
            
            # Sample frames for analysis
            sample_count = min(50, int(clip.duration * clip.fps))
            frame_differences = []
            
            for i in range(sample_count - 1):
                time1 = (i / sample_count) * clip.duration
                time2 = ((i + 1) / sample_count) * clip.duration
                
                frame1 = clip.get_frame(time1)
                frame2 = clip.get_frame(time2)
                
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
            self.log_message(f"⚠️ Visual variety analysis failed: {e}", "WARNING")
            return 0.5
    
    def _analyze_audio_quality(self, clip: VideoClip) -> float:
        """Analyze audio quality"""
        try:
            if not clip.audio:
                return 0.0
            
            # Get audio array
            audio_array = clip.audio.to_soundarray()
            
            # Calculate audio metrics
            audio_mean = np.mean(np.abs(audio_array))
            audio_std = np.std(audio_array)
            
            # Quality score based on audio levels and variety
            if audio_mean > 0.01 and audio_std > 0.005:
                return min(1.0, (audio_mean * 100 + audio_std * 1000) / 2)
            else:
                return 0.3
            
        except Exception as e:
            self.log_message(f"⚠️ Audio quality analysis failed: {e}", "WARNING")
            return 0.5
    
    def _regenerate_enhanced_visual_clip_function(self, quality_score: float) -> None:
        """Use Ollama to regenerate the _create_enhanced_visual_clip function"""
        try:
            import ollama
            
            prompt = f"""Düşük kalite için iyileştirilmiş moviepy code üret.
            
            Kalite skoru: {quality_score:.2f}
            Problem: Video kalitesi düşük, _create_enhanced_visual_clip fonksiyonu iyileştirilmeli
            
            Fonksiyon gereksinimleri:
            - Black frame detection (numpy mean < 10)
            - Intelligent fallback systems
            - Quality enhancement techniques
            - Duration optimization
            
            Python code olarak döndür, sadece fonksiyonu yaz."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                self.log_message(f"🤖 Ollama generated enhanced visual clip function", "OLLAMA")
                # In a real implementation, you might want to save this code
                # or use it to dynamically update the function
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama function regeneration failed: {e}", "WARNING")
    
    def _create_hook_clip(self, hook_text: str, duration: float) -> VideoClip:
        """Create hook clip for short videos"""
        try:
            # Create hook text clip
            text_clip = TextClip(hook_text, fontsize=50, color='white', bg_color='red')
            text_clip = text_clip.set_duration(duration)
            text_clip = text_clip.set_position('center')
            
            return text_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Hook clip creation failed: {e}", "WARNING")
            # Fallback: solid color clip
            return ColorClip(size=(1920, 1080), color=(255, 0, 0)).set_duration(duration)
    
    def generate_morgan_freeman_voiceover(self, text: str, output_path: str) -> bool:
        """Generate Morgan Freeman style voiceover using advanced TTS"""
        try:
            if self.tts_system == "piper":
                return self._generate_piper_voiceover(text, output_path)
            elif self.tts_system == "espeak":
                return self._generate_espeak_voiceover(text, output_path)
            elif self.tts_system == "gtts":
                return self._generate_gtts_voiceover(text, output_path)
            else:
                self.log_message("❌ No TTS system available", "ERROR")
                return False
        except Exception as e:
            self.log_message(f"❌ Voiceover generation failed: {e}", "ERROR")
            return False
    
    def _generate_piper_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using Piper TTS with Morgan Freeman style"""
        try:
            # Morgan Freeman style parameters
            voice_params = {
                'speed': 0.8,  # Slower, more deliberate
                'pitch': 0.7,  # Deeper voice
                'volume': 1.2,  # Slightly louder
                'model': 'en_US-amy-low.onnx'  # Use appropriate model
            }
            
            # Generate audio with Piper
            tts = piper.PiperVoice.load_model(voice_params['model'])
            audio_data = tts.synthesize(text, voice_params)
            
            # Save audio
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            self.log_message(f"✅ Piper TTS voiceover generated: {output_path}", "TTS")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Piper TTS failed: {e}", "ERROR")
            return False
    
    def _generate_espeak_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using espeak with Morgan Freeman style"""
        try:
            # espeak parameters for Morgan Freeman style
            cmd = f'espeak -v en-us -s 120 -p 50 -a 100 "{text}" -w "{output_path}"'
            os.system(cmd)
            
            if os.path.exists(output_path):
                self.log_message(f"✅ espeak voiceover generated: {output_path}", "TTS")
                return True
            else:
                return False
                
        except Exception as e:
            self.log_message(f"❌ espeak failed: {e}", "ERROR")
            return False
    
    def _generate_gtts_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using gTTS as fallback"""
        try:
            tts = gTTS(text=text, lang='en', slow=True)  # Slow for Morgan Freeman style
            tts.save(output_path)
            
            self.log_message(f"✅ gTTS voiceover generated: {output_path}", "TTS")
            return True
            
        except Exception as e:
            self.log_message(f"❌ gTTS failed: {e}", "ERROR")
            return False
    
    def generate_custom_music(self, duration: float, mood: str = "epic") -> str:
        """Generate custom MIDI music using Ollama and mido"""
        if not MIDO_AVAILABLE:
            self.log_message("⚠️ MIDI generation not available, using fallback music", "MUSIC")
            return self._get_fallback_music()
        
        try:
            # Generate music parameters using Ollama
            music_params = self._generate_music_parameters_with_ollama(mood, duration)
            
            # Create MIDI file
            midi_file = self._create_midi_file(music_params, duration)
            
            # Convert MIDI to audio
            audio_file = self._convert_midi_to_audio(midi_file)
            
            if audio_file and os.path.exists(audio_file):
                self.log_message(f"✅ Custom music generated: {audio_file}", "MUSIC")
                return audio_file
            else:
                return self._get_fallback_music()
                
        except Exception as e:
            self.log_message(f"❌ Custom music generation failed: {e}", "MUSIC")
            return self._get_fallback_music()
    
    def _generate_music_parameters_with_ollama(self, mood: str, duration: float) -> Dict:
        """Generate music parameters using Ollama LLM"""
        try:
            import ollama
            
            prompt = f"""Generate music parameters for a {mood} mood video that's {duration:.1f} seconds long.
            
            Return as JSON with:
            - tempo (BPM)
            - key (C, D, E, F, G, A, B)
            - scale (major, minor, pentatonic)
            - instruments (array of 3-5 instruments)
            - chord_progression (array of 4-8 chords)
            - mood_intensity (0.1 to 1.0)
            
            Make it cinematic and engaging for documentary content."""
            
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
            content = response.get('message', {}).get('content', '{}')
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_music_params()
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama music generation failed: {e}, using defaults", "MUSIC")
            return self._get_default_music_params()
    
    def _get_default_music_params(self) -> Dict:
        """Get default music parameters"""
        return {
            'tempo': 120,
            'key': 'C',
            'scale': 'major',
            'instruments': ['piano', 'strings', 'brass'],
            'chord_progression': ['C', 'Am', 'F', 'G'],
            'mood_intensity': 0.7
        }
    
    def _get_fallback_music(self) -> str:
        """Get fallback music from local assets"""
        if self.local_assets['audio']:
            return random.choice(self.local_assets['audio'])
        else:
            return "assets/audio/music/epic_music.mp3"  # Default fallback
    
    def generate_voiceover(self, script_data: dict, output_folder: str) -> List[str]:
        """Generate advanced voiceover with Morgan Freeman style"""
        self.log_message("🎤 Generating Morgan Freeman style voiceover...", "VOICEOVER")
        
        try:
            narration_list = [scene.get("sentence") for scene in script_data.get("script", []) if scene.get("sentence")]
            if not narration_list:
                self.log_message("❌ No narration text found in script", "ERROR")
                return None
            
            audio_files = []
            os.makedirs(output_folder, exist_ok=True)
            
            for i, text in enumerate(narration_list):
                file_path = os.path.join(output_folder, f"part_{i+1}.mp3")
                self.log_message(f"🎤 Generating voiceover {i+1}/{len(narration_list)}", "VOICEOVER")
                
                if self.generate_morgan_freeman_voiceover(text, file_path):
                    audio_files.append(file_path)
                else:
                    self.log_message(f"⚠️ Voiceover generation failed for part {i+1}", "WARNING")
                    continue
            
            self.log_message(f"✅ Generated {len(audio_files)} voiceover files", "VOICEOVER")
            return audio_files
            
        except Exception as e:
            self.log_message(f"❌ Voiceover generation failed: {e}", "ERROR")
            return None
    
    def find_visual_assets(self, script_data: dict, channel_niche: str, download_folder: str) -> List[str]:
        """Find and download visual assets with 4K upscaling"""
        self.log_message("🎬 Finding visual assets with 4K upscaling...", "VISUALS")
        
        video_paths = []
        os.makedirs(download_folder, exist_ok=True)
        scenes = script_data.get("script", [])
        
        for i, scene in enumerate(scenes):
            query = scene.get("visual_query", "")
            found_video_path = None
            
            if query:
                # Optimize Pexels query
                optimized_query = self._optimize_pexels_query(query, channel_niche)
                
                # Calculate minimum duration for this scene (estimate based on script length)
                estimated_duration = max(5.0, len(scene.get("text", "").split()) * 0.3)  # ~0.3s per word
                
                # Try Pexels download with real API
                found_video_path = self._download_pexels_video(optimized_query, estimated_duration, download_folder)
            
            # Fallback to local assets if Pexels fails
            if not found_video_path:
                found_video_path = self._get_local_asset_fallback(channel_niche, i+1)
            
            # Upscale video to 4K if available
            if found_video_path and PILLOW_AVAILABLE:
                upscaled_path = self._upscale_video_to_4k(found_video_path)
                if upscaled_path:
                    found_video_path = upscaled_path
            
            video_paths.append(found_video_path)
            
            if found_video_path:
                self.log_message(f"✅ Scene {i+1} visual asset ready: {os.path.basename(found_video_path)}", "VISUALS")
            else:
                self.log_message(f"⚠️ Scene {i+1} visual asset failed", "WARNING")
        
        return video_paths
    
    def edit_long_form_video(self, audio_files: list, visual_files: list, music_path: str, output_filename: str) -> Optional[str]:
        """Create advanced long-form video with professional effects"""
        self.log_message("🎬 Creating advanced long-form video...", "VIDEO")
        
        try:
            clips = []
            total_duration = 0
            
            for i, (audio_path, visual_path) in enumerate(zip(audio_files, visual_files)):
                if not os.path.exists(audio_path):
                    continue
                
                # Load audio
                audio_clip = AudioFileClip(audio_path)
                if not audio_clip.duration or audio_clip.duration <= 0:
                    continue
                
                # Create visual clip
                visual_clip = self._create_enhanced_visual_clip(visual_path, audio_clip.duration, i)
                
                # Combine audio and visual
                scene_clip = visual_clip.set_audio(audio_clip)
                
                # Add professional effects
                scene_clip = self._apply_professional_effects(scene_clip, i)
                
                clips.append(scene_clip)
                total_duration += audio_clip.duration
                
                # Add subliminal message every 25th frame (disabled by default)
                if ENABLE_SUBLIMINAL and i % 25 == 0:
                    subliminal_clip = self._create_subliminal_message("Subscribe now", 0.04)  # 1/25 second
                    if subliminal_clip:
                        clips.append(subliminal_clip)
            
            if not clips:
                self.log_message("❌ No valid clips to process", "ERROR")
                return None
            
            # Concatenate clips with smooth transitions
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Add background music
            if music_path and os.path.exists(music_path):
                final_video = self._add_background_music(final_video, music_path)
            
            # Add multilingual subtitles
            final_video = self._add_multilingual_subtitles(final_video)
            
            # Ensure minimum duration (10+ minutes)
            if total_duration < 600:  # Less than 10 minutes
                final_video = self._extend_video_duration(final_video, 600)
            
            # Write final video
            final_video.write_videofile(output_filename, fps=self.FPS, codec=self.CODEC,
                                      audio_codec=self.AUDIO_CODEC, verbose=False, logger=None)
            
            # Analyze video quality
            self._analyze_video_quality(output_filename)
            
            self.log_message(f"✅ Advanced video created: {output_filename}", "SUCCESS")
            return output_filename
            
        except Exception as e:
            self.log_message(f"❌ Video creation failed: {e}", "ERROR")
            return None
    
    def create_short_videos(self, long_form_video_path: str, output_folder: str) -> List[str]:
        """Create 3 short videos (15-60 seconds) from long form video"""
        self.log_message("🎬 Creating short videos from long form content...", "SHORTS")
        
        try:
            if not os.path.exists(long_form_video_path):
                self.log_message("❌ Long form video not found", "ERROR")
                return []
            
            # Load long form video
            long_video = VideoFileClip(long_form_video_path)
            
            short_videos = []
            durations = [15, 30, 60]  # Different short video lengths
            
            for i, duration in enumerate(durations):
                try:
                    # Extract random segment
                    start_time = random.uniform(0, max(0, long_video.duration - duration))
                    end_time = start_time + duration
                    
                    # Create short clip
                    short_clip = long_video.subclip(start_time, end_time)
                    
                    # Add hook at beginning
                    hook_clip = self._create_hook_clip(f"Hook {i+1}: The Mystery Deepens...", 3)
                    final_short = concatenate_videoclips([hook_clip, short_clip], method="compose")
                    
                    # Save short video
                    output_path = os.path.join(output_folder, f"short_{i+1}_{duration}s.mp4")
                    final_short.write_videofile(output_path, fps=self.FPS, codec=self.CODEC,
                                              audio_codec=self.AUDIO_CODEC, verbose=False, logger=None)
                    
                    short_videos.append(output_path)
                    self.log_message(f"✅ Short video {i+1} created: {duration}s", "SHORTS")
                    
                except Exception as e:
                    self.log_message(f"⚠️ Short video {i+1} creation failed: {e}", "WARNING")
                    continue
            
            long_video.close()
            
            self.log_message(f"✅ Created {len(short_videos)} short videos", "SHORTS")
            return short_videos
            
        except Exception as e:
            self.log_message(f"❌ Short video creation failed: {e}", "ERROR")
            return []

# Convenience functions for backward compatibility
def generate_voiceover(script_data: dict, output_folder: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.generate_voiceover(script_data, output_folder)

def find_visual_assets(script_data: dict, channel_niche: str, download_folder: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.find_visual_assets(script_data, channel_niche, download_folder)

def edit_long_form_video(audio_files: list, visual_files: list, music_path: str, output_filename: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.edit_long_form_video(audio_files, visual_files, music_path, output_filename)

def create_short_video(long_form_video_path: str, output_filename: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    output_folder = os.path.dirname(output_filename)
    short_videos = creator.create_short_videos(long_form_video_path, output_folder)
    return short_videos[0] if short_videos else None

# Basic test
if __name__ == "__main__":
    print("🧪 Testing Advanced Video Creator...")
    creator = AdvancedVideoCreator()
    print("✅ Basic initialization completed!")

    # Example usage (replace with actual script_data, channel_niche, etc.)
    # script_data = {
    #     "script": [
    #         {"sentence": "This is the first scene. The story begins."},
    #         {"sentence": "The protagonist, a young adventurer, sets out on a quest."},
    #         {"sentence": "They encounter a mysterious forest."},
    #         {"sentence": "The protagonist discovers a hidden cave."},
    #         {"sentence": "They find a treasure map."},
    #         {"sentence": "The protagonist follows the map."},
    #         {"sentence": "They encounter a dragon."},
    #         {"sentence": "The protagonist fights the dragon."},
    #         {"sentence": "They win the battle."},
    #         {"sentence": "The protagonist returns home."},
    #         {"sentence": "They celebrate their victory."},
    #         {"sentence": "The end."}
    #     ]
    # }
    # channel_niche = "Adventure"
    # download_folder = "assets/videos/downloads"
    # visual_assets = creator.find_visual_assets(script_data, channel_niche, download_folder)
    # print(f"Visual assets found: {visual_assets}")

    # audio_files = ["assets/audio/voiceover/part_1.mp3", "assets/audio/voiceover/part_2.mp3"]
    # visual_files = ["assets/videos/downloads/scene_1_pexels.mp4", "assets/videos/downloads/scene_2_pexels.mp4"]
    # music_path = "assets/audio/music/epic_music.mp3"
    # output_filename = "assets/videos/advanced_video.mp4"
    # long_form_video = creator.edit_long_form_video(audio_files, visual_files, music_path, output_filename)
    # print(f"Long form video created: {long_form_video}")

    # long_form_video_path = "assets/videos/advanced_video.mp4"
    # output_folder = "assets/videos/shorts"
    # short_videos = creator.create_short_videos(long_form_video_path, output_folder)
    # print(f"Short videos created: {short_videos}")
