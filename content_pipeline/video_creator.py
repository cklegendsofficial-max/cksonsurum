# content_pipeline/video_creator.py (Usta Yönetmen Sürümü - Hassas Kurgu)

import os
import requests
from gtts import gTTS
from moviepy.editor import *

# Import config variables
try:
    from config import PEXELS_API_KEY
except ImportError:
    PEXELS_API_KEY = None

def generate_voiceover(script_data: dict, output_folder: str):
    print("  - Standart seslendirme (gTTS) işlemi başlatıldı...")
    try:
        # Yeni, cümle bazlı senaryo yapısına göre metinleri al
        narration_list = [scene.get("sentence") for scene in script_data.get("script", []) if scene.get("sentence")]
        if not narration_list: 
            print("  - HATA: Senaryoda seslendirilecek metin bulunamadı.")
            return None
        
        audio_files = []
        for i, text in enumerate(narration_list):
            file_path = os.path.join(output_folder, f"part_{i+1}.mp3")
            print(f"    - Ses dosyası oluşturuluyor: part_{i+1}.mp3")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(file_path)
            audio_files.append(file_path)
        return audio_files
    except Exception as e:
        print(f"  - HATA: Seslendirme sırasında (gTTS): {e}")
        return None

def find_visual_assets(script_data: dict, channel_niche: str, download_folder: str):
    print("\n[*] Adım 3.5: Görsel Varlıklar (Cümle Bazlı ve Yedek Planlı) Aranıyor...")
    headers = {"Authorization": PEXELS_API_KEY} if PEXELS_API_KEY else {}
    video_paths = []
    os.makedirs(download_folder, exist_ok=True)
    scenes = script_data.get("script", [])
    last_successful_video_path = None

    for i, scene in enumerate(scenes):
        query = scene.get("visual_query")
        found_video_path = None
        if query:
            try:
                print(f"  - Cümle {i+1} için video aranıyor: '{query}'")
                api_url = f"https://api.pexels.com/videos/search?query={query}&per_page=1&orientation=landscape"
                response = requests.get(api_url, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                if data.get("videos"):
                    video_url = data['videos'][0]['video_files'][0]['link']
                    video_path = os.path.join(download_folder, f"scene_{i+1}.mp4")
                    print(f"    - Video indiriliyor...")
                    video_response = requests.get(video_url, stream=True, timeout=30)
                    with open(video_path, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    found_video_path = video_path
            except Exception as e:
                print(f"    - UYARI (Pexels): Arama başarısız. Yedek plana geçiliyor. Hata: {e}")

        if not found_video_path and last_successful_video_path:
            print(f"    - Yedek plan: Bir önceki sahnenin videosu kullanılıyor.")
            found_video_path = last_successful_video_path
        
        video_paths.append(found_video_path)
        if found_video_path:
            last_successful_video_path = found_video_path
            
    return video_paths

def edit_long_form_video(audio_files: list, visual_files: list, music_path: str, output_filename: str):
    WIDTH, HEIGHT = 1280, 720
    clips = []
    print(f"\n[*] Adım 4: Video Kurgulama (Hassas Senkronizasyon) Başlatılıyor...")
    for i, (audio_path, visual_path) in enumerate(zip(audio_files, visual_files)):
        try:
            audio_clip = AudioFileClip(audio_path)
            if not audio_clip.duration or audio_clip.duration <= 0: continue

            background_clip = None
            if visual_path and os.path.exists(visual_path) and os.path.getsize(visual_path) > 0:
                raw_clip = VideoFileClip(visual_path)
                if raw_clip.duration < audio_clip.duration:
                    background_clip = raw_clip.fx(vfx.loop, duration=audio_clip.duration)
                else:
                    background_clip = raw_clip.subclip(0, audio_clip.duration)
                background_clip = background_clip.resize(height=HEIGHT).crop(x_center=background_clip.w/2, width=WIDTH, height=HEIGHT)
            else:
                print(f"  - UYARI: Sahne {i+1} için görsel bulunamadı, siyah ekran kullanılıyor.")
                background_clip = ColorClip(size=(WIDTH, HEIGHT), color=(0, 0, 0), duration=audio_clip.duration)
            
            scene_clip = background_clip.set_audio(audio_clip)
            if scene_clip.duration is not None:
                clips.append(scene_clip)
        except Exception as e:
            print(f"  - UYARI: Sahne {i+1} kurgulanırken hata: {e}.")
            continue
    
    if not clips: return None
    
    final_video = concatenate_videoclips(clips, transition=vfx.fadeout, duration=0.5)

    if music_path and os.path.exists(music_path):
        music_clip = AudioFileClip(music_path).fx(afx.audio_loop, duration=final_video.duration).volumex(0.15)
        final_video.audio = CompositeAudioClip([final_video.audio, music_clip])

    try:
        final_video.write_videofile(output_filename, fps=24, codec='libx264', audio_codec='aac')
        return output_filename
    except Exception as e:
        print(f"  - HATA: Video yazma sırasında hata: {e}")
        return None

def create_short_video(long_form_video_path: str, output_filename: str):
    # Bu fonksiyon şimdilik aynı kalıyor
    pass