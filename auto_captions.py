# auto_captions.py
import logging
import os
import pathlib
import shutil
import subprocess
from typing import Any, List, Optional


log = logging.getLogger("captions")

# ============ Tier 1 + Tier 2 hedef diller ============
try:
    from config import settings

    TARGET_LANGS = settings.LANGS_TIER1 + settings.LANGS_TIER2
except Exception:
    TARGET_LANGS = [
        # Tier 1
        "en",
        "es",
        "pt",
        "fr",
        "de",
        "ja",
        # Tier 2
        "tr",
        "ar",
        "hi",
        "ru",
        "it",
        "nl",
        "ko",
        "zh",
    ]


# ---------------- helpers ----------------
def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _ensure_ext(path: str, new_ext: str) -> str:
    p = pathlib.Path(path)
    return str(p.with_suffix(new_ext))


def _extract_audio(video_path: str) -> Optional[str]:
    """Videodan 16kHz mono wav çıkarır (ffmpeg gerekli)."""
    if not _has("ffmpeg"):
        log.warning("ffmpeg not found; cannot extract audio.")
        return None
    out = _ensure_ext(video_path, ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out
    except Exception as e:
        log.warning(f"audio extract failed: {e}")
        return None


# -------------- EN transcribe (Whisper) --------------
def transcribe_to_srt(
    audio_or_video_path: str, out_dir: Optional[str] = None, lang: str = "en"
) -> Optional[str]:
    """
    Dayanıklı EN SRT üretimi: EN SRT varsa atla, yoksa Whisper ile üret.
    SRT normalizasyonu: sıralı index, timecode fix, max 42 char satır sarma.
    """
    try:
        # Output directory belirle
        if out_dir is None:
            out_dir = os.path.dirname(audio_or_video_path)

        # EN SRT zaten varsa atla
        base_name = os.path.splitext(os.path.basename(audio_or_video_path))[0]
        en_srt_path = os.path.join(out_dir, f"{base_name}.en.srt")

        if os.path.exists(en_srt_path):
            log.info(f"EN SRT already exists, skipping: {en_srt_path}")
            return en_srt_path

        # Audio/video path kontrolü
        path = audio_or_video_path
        if not os.path.exists(path):
            log.warning(f"Path not found: {path}")
            return None

        # Video ise ses çıkar
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".wav", ".mp3", ".m4a", ".flac"}:
            extracted = _extract_audio(path)
            if not extracted:
                log.warning(f"Audio extraction failed for: {path}")
                return None
            path = extracted
            log.info(f"Audio extracted: {extracted}")

        # Whisper import kontrolü
        try:
            import whisper
        except ImportError:
            log.warning("Whisper not available; skipping subtitles.")
            return None

        # Whisper model yükle
        try:
            from config import settings

            model_name = settings.WHISPER_MODEL
        except Exception:
            model_name = os.getenv("WHISPER_MODEL", "base")

        log.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        # Transcribe
        log.info(f"Transcribing audio: {path}")
        result = model.transcribe(path, language=lang)

        if not result or "segments" not in result:
            log.warning("Whisper returned no segments")
            return None

        # SRT normalizasyonu
        normalized_srt = _normalize_srt_content(result["segments"])

        # SRT dosyasına yaz
        try:
            with open(en_srt_path, "w", encoding="utf-8") as f:
                f.write(normalized_srt)

            log.info(f"EN SRT generated successfully: {en_srt_path}")
            return en_srt_path

        except Exception as e:
            log.error(f"SRT write failed: {e}")
            return None

    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return None


def _normalize_srt_content(segments: list[dict[str, Any]]) -> str:
    """SRT içeriğini normalizasyon: index, timecode, text wrapping"""
    normalized_lines = []

    for i, segment in enumerate(segments, 1):
        # Index
        normalized_lines.append(str(i))

        # Timecode fix
        start_time = _format_timestamp(segment.get("start", 0))
        end_time = _format_timestamp(segment.get("end", 0))
        normalized_lines.append(f"{start_time} --> {end_time}")

        # Text normalization ve wrapping
        text = segment.get("text", "").strip()
        if text:
            wrapped_text = _wrap_text(text, max_length=42)
            normalized_lines.extend(wrapped_text)

        # Boş satır
        normalized_lines.append("")

    return "\n".join(normalized_lines)


def _format_timestamp(seconds: float) -> str:
    """Saniyeyi SRT timecode formatına çevir"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def _wrap_text(text: str, max_length: int = 42) -> list[str]:
    """Metni max_length karakterde sar"""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_length:
            current_line += (" " + word) if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# -------------- Çeviri (Transformers → LLM → kopya) --------------
def _transformers_translate(lines: List[str], target_lang: str) -> Optional[List[str]]:
    try:
        from transformers import MarianMTModel, MarianTokenizer

        model_map = {
            "es": "Helsinki-NLP/opus-mt-en-es",
            "pt": "Helsinki-NLP/opus-mt-en-pt",
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "de": "Helsinki-NLP/opus-mt-en-de",
            "ja": "Helsinki-NLP/opus-mt-en-jap",
            "tr": "Helsinki-NLP/opus-mt-en-tr",
            "ar": "Helsinki-NLP/opus-mt-en-ar",
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "ru": "Helsinki-NLP/opus-mt-en-ru",
            "it": "Helsinki-NLP/opus-mt-en-it",
            "nl": "Helsinki-NLP/opus-mt-en-nl",
            "ko": "Helsinki-NLP/opus-mt-en-ko",
            "zh": "Helsinki-NLP/opus-mt-en-zh",
        }
        model_name = model_map.get(target_lang)
        if not model_name:
            return None
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name)
        out = []
        chunk = []
        for ln in lines:
            chunk.append(ln)
            if len(chunk) >= 64:
                batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
                gen = mdl.generate(**batch, max_length=256)
                out += tok.batch_decode(gen, skip_special_tokens=True)
                chunk = []
        if chunk:
            batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
            gen = mdl.generate(**batch, max_length=256)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out
    except Exception:
        return None


def _llm_translate(lines: List[str], target_lang: str) -> Optional[List[str]]:
    """Ollama mevcutsa çok hafif bir JSON çeviri. Yoksa None."""
    try:
        from improved_llm_handler import ImprovedLLMHandler

        h = ImprovedLLMHandler()
        prompt = (
            f"Translate each English line to {target_lang}. "
            "Return ONLY a JSON array of strings. Keep line breaks minimal; no numbering.\n"
            f"Lines: {lines[:64]}"
        )
        res = h._get_ollama_response(prompt)  # Using existing method
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            return res
    except Exception:
        return None
    return None


def translate_srt(in_path: str, lang: str) -> Optional[str]:
    """
    Dayanıklı SRT çevirisi: 1) MarianMT (varsa) 2) Ollama LLM 3) EN kopya (fallback)
    Zaman kodlarına dokunmaz, sadece metni çevirir.
    """
    try:
        if not os.path.exists(in_path):
            log.warning(f"SRT file not found: {in_path}")
            return None

        # Output path belirle
        base_path = os.path.splitext(in_path)[0]
        if base_path.endswith(".en"):
            base_path = base_path[:-3]  # .en.srt -> .srt
        out_path = f"{base_path}.{lang}.srt"

        # Hedef dil EN ise kopya
        if lang == "en":
            try:
                import shutil

                shutil.copy2(in_path, out_path)
                log.info(f"EN SRT copied: {out_path}")
                return out_path
            except Exception as e:
                log.warning(f"EN SRT copy failed: {e}")
                return None

        # SRT dosyasını oku ve parse et
        try:
            with open(in_path, encoding="utf-8") as f:
                raw_lines = f.read().splitlines()
        except Exception as e:
            log.warning(f"SRT read failed: {e}")
            return None

        # SRT bloklarını parse et
        parsed_blocks = _parse_srt_blocks(raw_lines)
        if not parsed_blocks:
            log.warning("Failed to parse SRT blocks")
            return None

        # Metin çevirisi
        translated_blocks = _translate_srt_blocks(parsed_blocks, lang)
        if not translated_blocks:
            log.warning(f"Translation failed for {lang}, copying EN version")
            # Fallback: EN kopya
            try:
                import shutil

                shutil.copy2(in_path, out_path)
                log.info(f"EN SRT copied as fallback: {out_path}")
                return out_path
            except Exception as e:
                log.error(f"Fallback copy failed: {e}")
                return None

        # Çevrilmiş SRT'yi yaz
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(_rebuild_srt_content(translated_blocks))

            log.info(f"Translation completed: {lang} -> {out_path}")
            return out_path

        except Exception as e:
            log.error(f"SRT write failed: {e}")
            return None

    except Exception as e:
        log.error(f"Translation failed for {lang}: {e}")
        return None


def _parse_srt_blocks(raw_lines: list) -> list:
    """SRT dosyasını bloklara parse et"""
    blocks = []
    current_block = {}
    i = 0

    while i < len(raw_lines):
        line = raw_lines[i].strip()

        if not line:  # Boş satır - blok sonu
            if current_block:
                blocks.append(current_block)
                current_block = {}
            i += 1
            continue

        # Index satırı
        if line.isdigit():
            current_block = {"index": line}
            i += 1
            continue

        # Timecode satırı
        if "-->" in line:
            current_block["timecode"] = line
            i += 1
            continue

        # Text satırları
        if "index" in current_block and "timecode" in current_block:
            if "text" not in current_block:
                current_block["text"] = []
            current_block["text"].append(line)

        i += 1

    # Son bloku ekle
    if current_block:
        blocks.append(current_block)

    return blocks


def _translate_srt_blocks(blocks: list, target_lang: str) -> list:
    """SRT bloklarını çevir"""
    try:
        # 1) MarianMT ile çeviri
        translated_blocks = _translate_with_marianmt(blocks, target_lang)
        if translated_blocks:
            log.info(f"MarianMT translation successful for {target_lang}")
            return translated_blocks

        # 2) Ollama LLM ile çeviri
        translated_blocks = _translate_with_ollama(blocks, target_lang)
        if translated_blocks:
            log.info(f"Ollama translation successful for {target_lang}")
            return translated_blocks

        # 3) Başarısız
        log.warning(f"All translation methods failed for {target_lang}")
        return None

    except Exception as e:
        log.error(f"Translation process failed: {e}")
        return None


def _translate_with_marianmt(blocks: list, target_lang: str) -> Optional[list]:
    """MarianMT ile çeviri"""
    try:
        from transformers import MarianMTModel, MarianTokenizer

        model_map = {
            "es": "Helsinki-NLP/opus-mt-en-es",
            "pt": "Helsinki-NLP/opus-mt-en-pt",
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "de": "Helsinki-NLP/opus-mt-en-de",
            "ja": "Helsinki-NLP/opus-mt-en-jap",
            "tr": "Helsinki-NLP/opus-mt-en-tr",
            "ar": "Helsinki-NLP/opus-mt-en-ar",
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "ru": "Helsinki-NLP/opus-mt-en-ru",
            "it": "Helsinki-NLP/opus-mt-en-it",
            "nl": "Helsinki-NLP/opus-mt-en-nl",
            "ko": "Helsinki-NLP/opus-mt-en-ko",
            "zh": "Helsinki-NLP/opus-mt-en-zh",
        }

        model_name = model_map.get(target_lang)
        if not model_name:
            return None

        # Model ve tokenizer yükle
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        translated_blocks = []

        for block in blocks:
            translated_block = block.copy()

            if "text" in block and block["text"]:
                # Metni çevir
                text_lines = block["text"]
                translated_text = []

                # Batch çeviri (64 satırda bir)
                batch_size = 64
                for i in range(0, len(text_lines), batch_size):
                    batch = text_lines[i : i + batch_size]

                    # Tokenize
                    inputs = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    )

                    # Generate
                    outputs = model.generate(**inputs, max_length=256)

                    # Decode
                    translated_batch = tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    translated_text.extend(translated_batch)

                translated_block["text"] = translated_text

            translated_blocks.append(translated_block)

        return translated_blocks

    except Exception as e:
        log.warning(f"MarianMT translation failed for {target_lang}: {e}")
        return None


def _translate_with_ollama(blocks: list, target_lang: str) -> Optional[list]:
    """Ollama LLM ile çeviri"""
    try:
        from improved_llm_handler import ImprovedLLMHandler

        handler = ImprovedLLMHandler()
        translated_blocks = []

        for block in blocks:
            translated_block = block.copy()

            if "text" in block and block["text"]:
                # Metni çevir
                text_lines = block["text"]

                # Ollama prompt
                prompt = (
                    f"Translate these English lines to {target_lang}. "
                    "Return ONLY a JSON array of strings. Keep line breaks minimal; no numbering.\n"
                    f"Lines: {text_lines[:64]}"  # Max 64 satır
                )

                # Çeviri al
                response = handler._get_ollama_response(prompt)

                if isinstance(response, list) and all(
                    isinstance(x, str) for x in response
                ):
                    translated_block["text"] = response
                else:
                    # Response geçersizse orijinal metni kullan
                    translated_block["text"] = text_lines

            translated_blocks.append(translated_block)

        return translated_blocks

    except Exception as e:
        log.warning(f"Ollama translation failed for {target_lang}: {e}")
        return None


def _rebuild_srt_content(blocks: list) -> str:
    """Çevrilmiş bloklardan SRT içeriğini yeniden oluştur"""
    lines = []

    for block in blocks:
        lines.append(block["index"])
        lines.append(block["timecode"])

        if "text" in block and block["text"]:
            lines.extend(block["text"])

        lines.append("")  # Boş satır

    return "\n".join(lines)


def generate_multi_captions(video: str) -> List[str]:
    """
    Dayanıklı çoklu dil caption üretimi.
    EN SRT varsa atla, yoksa Whisper ile üret.
    Tüm diller için çeviri: MarianMT → Ollama → EN kopya (fallback).
    Hata başına tek satır uyarı, diğer diller devam eder.
    """
    try:
        # Output directory belirle
        video_dir = os.path.dirname(video)
        captions_dir = os.path.join(video_dir, "captions")
        os.makedirs(captions_dir, exist_ok=True)

        # Video base name
        base_name = os.path.splitext(os.path.basename(video))[0]

        # 1) EN SRT üretimi (varsa atla)
        en_srt_path = os.path.join(captions_dir, f"{base_name}.en.srt")

        if not os.path.exists(en_srt_path):
            log.info("EN SRT not found, generating with Whisper...")
            en_srt_path = transcribe_to_srt(video, captions_dir, "en")
            if not en_srt_path:
                log.error("Failed to generate EN SRT, aborting caption generation")
                return []
        else:
            log.info("EN SRT already exists, skipping generation")

        created_files = [en_srt_path]

        # 2) Çoklu dil çevirisi
        target_langs = TARGET_LANGS.copy()
        if "en" in target_langs:
            target_langs.remove("en")  # EN zaten var

        log.info(f"Starting translations for {len(target_langs)} languages...")

        for lang in target_langs:
            try:
                log.info(f"Translating to {lang}...")
                translated_path = translate_srt(en_srt_path, lang)

                if translated_path and os.path.exists(translated_path):
                    created_files.append(translated_path)
                    log.info(f"✅ {lang} translation completed: {translated_path}")
                else:
                    log.warning(f"⚠️ {lang} translation failed")

            except Exception as e:
                log.warning(
                    f"⚠️ {lang} translation error: {str(e)[:100]}"
                )  # Hata başına tek satır
                continue

        # 3) Sonuç raporu
        successful_translations = len(created_files) - 1  # EN hariç
        total_targets = len(target_langs)

        log.info("Caption generation completed:")
        log.info(f"  - EN SRT: {en_srt_path}")
        log.info(
            f"  - Translations: {successful_translations}/{total_targets} successful"
        )
        log.info(f"  - Total files: {len(created_files)}")

        # 4) Captions alt klasörüne taşı (eğer farklı yerdeyse)
        final_captions_dir = os.path.join(video_dir, "captions")
        if captions_dir != final_captions_dir:
            os.makedirs(final_captions_dir, exist_ok=True)
            for file_path in created_files:
                if os.path.dirname(file_path) != final_captions_dir:
                    filename = os.path.basename(file_path)
                    new_path = os.path.join(final_captions_dir, filename)
                    try:
                        import shutil

                        shutil.move(file_path, new_path)
                        log.info(f"Moved caption file: {filename}")
                    except Exception as e:
                        log.warning(f"Failed to move {filename}: {e}")

        return created_files

    except Exception as e:
        log.error(f"Multi-caption generation failed: {e}")
        return []
