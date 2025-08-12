# auto_captions.py
import os, shutil, subprocess, json, pathlib, logging
from typing import Optional, List

log = logging.getLogger("captions")

# ============ Tier 1 + Tier 2 hedef diller ============
TARGET_LANGS = [
    # Tier 1
    "en","es","pt","fr","de","ja",
    # Tier 2
    "tr","ar","hi","ru","it","nl","ko","zh"
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
            ["ffmpeg","-y","-i", video_path, "-vn", "-ac","1","-ar","16000", out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return out
    except Exception as e:
        log.warning(f"audio extract failed: {e}")
        return None

# -------------- EN transcribe (Whisper) --------------
def transcribe_to_srt(audio_or_video_path: str, lang: str = "en") -> Optional[str]:
    """
    Whisper varsa EN SRT üretir. Girdi video ise sesi otomatik çıkarır.
    Whisper yoksa None döner (pipeline devam eder).
    """
    path = audio_or_video_path
    if not os.path.exists(path):
        log.warning(f"path not found: {path}")
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".wav",".mp3",".m4a",".flac"}:
        extracted = _extract_audio(path)
        if not extracted:
            return None
        path = extracted

    try:
        import whisper
    except Exception:
        log.warning("whisper not available; skipping subtitles.")
        return None

    model_name = os.getenv("WHISPER_MODEL","base")
    try:
        model = whisper.load_model(model_name)
        r = model.transcribe(path, language=lang)
    except Exception as e:
        log.warning(f"whisper failed: {e}")
        return None

    srt_path = _ensure_ext(path, ".srt")
    try:
        with open(srt_path,"w",encoding="utf-8") as f:
            for i, seg in enumerate(r.get("segments",[]), 1):
                def ts(x):
                    h=int(x//3600); m=int((x%3600)//60); s=(x%60)
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".",",")
                text = (seg.get("text") or "").strip()
                f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{text}\n\n")
    except Exception as e:
        log.warning(f"SRT write failed: {e}")
        return None

    return srt_path

# -------------- Çeviri (Transformers → LLM → kopya) --------------
def _transformers_translate(lines: List[str], target_lang: str) -> Optional[List[str]]:
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_map = {
            "es":"Helsinki-NLP/opus-mt-en-es",
            "pt":"Helsinki-NLP/opus-mt-en-pt",
            "fr":"Helsinki-NLP/opus-mt-en-fr",
            "de":"Helsinki-NLP/opus-mt-en-de",
            "ja":"Helsinki-NLP/opus-mt-en-jap",
            "tr":"Helsinki-NLP/opus-mt-en-tr",
            "ar":"Helsinki-NLP/opus-mt-en-ar",
            "hi":"Helsinki-NLP/opus-mt-en-hi",
            "ru":"Helsinki-NLP/opus-mt-en-ru",
            "it":"Helsinki-NLP/opus-mt-en-it",
            "nl":"Helsinki-NLP/opus-mt-en-nl",
            "ko":"Helsinki-NLP/opus-mt-en-ko",
            "zh":"Helsinki-NLP/opus-mt-en-zh"
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
        if isinstance(res, list) and all(isinstance(x,str) for x in res):
            return res
    except Exception:
        return None
    return None

def translate_srt(srt_path: str, target_lang: str) -> Optional[str]:
    """EN SRT'yi hedef dile çevirir. Zaman kodlarını korur, sadece metni değiştirir."""
    if not os.path.exists(srt_path):
        return None
    try:
        raw = open(srt_path,"r",encoding="utf-8").read().splitlines()
    except Exception:
        return None

    # SRT bloklarını ayır
    indices, times, texts = [], [], []
    buf = []
    i = 0
    while i < len(raw):
        if raw[i].strip().isdigit() and (i+1)<len(raw) and "-->" in raw[i+1]:
            if buf:
                texts.append("\n".join(buf)); buf=[]
            indices.append(raw[i].strip()); times.append(raw[i+1].strip()); i+=2
        else:
            if raw[i].strip()=="" and buf:
                texts.append("\n".join(buf)); buf=[]
            elif raw[i].strip()!="":
                buf.append(raw[i])
            i+=1
    if buf: texts.append("\n".join(buf))

    # satır satır çeviri
    flat = []
    splits = []
    for block in texts:
        lines = block.split("\n")
        splits.append(len(lines))
        flat.extend(lines)

    translated = _transformers_translate(flat, target_lang)
    if translated is None:
        translated = _llm_translate(flat, target_lang)
    if translated is None:
        translated = flat  # son çare: kopya

    # yeniden birleştir
    it = iter(translated)
    rebuilt = []
    for idx, tm, cnt in zip(indices, times, splits):
        rebuilt.append(idx); rebuilt.append(tm)
        blk = []
        for _ in range(cnt):
            blk.append(next(it, ""))
        rebuilt.append("\n".join(blk)); rebuilt.append("")
    out_path = srt_path.replace(".srt", f".{target_lang}.srt")
    try:
        with open(out_path,"w",encoding="utf-8") as f:
            f.write("\n".join(rebuilt).strip()+"\n")
        return out_path
    except Exception:
        return None

def generate_multi_captions(video_path: str, audio_path: Optional[str] = None, langs: Optional[List[str]] = None) -> List[str]:
    """Tam akış: EN SRT (whisper) → diğer diller (transformers/LLM)."""
    langs = langs or TARGET_LANGS
    created = []

    # 1) EN transcribe
    src = audio_path if audio_path else video_path
    en_srt = transcribe_to_srt(src, lang="en")
    if not en_srt:
        log.warning("EN subtitles not generated (whisper missing?). Skipping translations.")
        return created
    created.append(en_srt)

    # 2) Çeviriler
    for lg in langs:
        if lg == "en": 
            continue
        out = translate_srt(en_srt, lg)
        if out: created.append(out)

    log.info(f"CAPTIONS: generated {len(created)} files for langs={langs}")
    return created
