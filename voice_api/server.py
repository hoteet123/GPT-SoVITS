import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (  # noqa: E402
    get_method_names,
)


DATA_DIR = Path(os.getenv("GSV_VOICE_API_DATA", ROOT_DIR / "voice_api_data")).resolve()
WEB_DIR = ROOT_DIR / "voice_api" / "web"
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
MAX_TTS_CHARS_PER_CHUNK = int(os.getenv("GSV_MAX_TTS_CHARS_PER_CHUNK", "70"))
DEFAULT_SPEED_FACTOR = float(os.getenv("GSV_DEFAULT_SPEED_FACTOR", "1.05"))
DEFAULT_TOP_K = int(os.getenv("GSV_DEFAULT_TOP_K", "12"))
DEFAULT_TOP_P = float(os.getenv("GSV_DEFAULT_TOP_P", "0.88"))
DEFAULT_TEMPERATURE = float(os.getenv("GSV_DEFAULT_TEMPERATURE", "0.78"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("GSV_DEFAULT_REPETITION_PENALTY", "1.16"))
DEFAULT_FRAGMENT_INTERVAL = float(os.getenv("GSV_DEFAULT_FRAGMENT_INTERVAL", "0.06"))
KOREAN_MAX_TTS_CHARS_PER_CHUNK = int(os.getenv("GSV_KO_MAX_TTS_CHARS_PER_CHUNK", "56"))
KOREAN_MIN_TTS_CHARS_PER_CHUNK = int(os.getenv("GSV_KO_MIN_TTS_CHARS_PER_CHUNK", "10"))
KOREAN_INITIAL_SILENCE_SECONDS = float(os.getenv("GSV_KO_INITIAL_SILENCE_SECONDS", "0.35"))
SEGMENT_PRE_PAD_SECONDS = float(os.getenv("GSV_SEGMENT_PRE_PAD_SECONDS", "0.04"))
SEGMENT_POST_PAD_SECONDS = float(os.getenv("GSV_SEGMENT_POST_PAD_SECONDS", "0.04"))
SEGMENT_TRAILING_SILENCE_SECONDS = float(os.getenv("GSV_SEGMENT_TRAILING_SILENCE_SECONDS", "0.05"))
KOREAN_WARMUP_TEXT = os.getenv("GSV_KO_WARMUP_TEXT", "준비.")
KOREAN_MODEL_PREROLL = os.getenv("GSV_KO_MODEL_PREROLL", "。")
KOREAN_SACRIFICE_PREFIX = os.getenv("GSV_KO_SACRIFICE_PREFIX", "음.")
KOREAN_DUPLICATE_MAX_CHARS = int(os.getenv("GSV_KO_DUPLICATE_MAX_CHARS", "18"))
KOREAN_CHAR_RE = re.compile(r"[가-힣]")
TERMINAL_PUNCT_RE = re.compile(r"[.!?。！？…]+$")

for directory in (VOICES_DIR, OUTPUTS_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: List[str] = Field(default_factory=list)
    prompt_text: Optional[str] = None
    prompt_lang: Optional[str] = None
    text_lang: str = "ko"
    top_k: int = Field(DEFAULT_TOP_K, ge=1)
    top_p: float = Field(DEFAULT_TOP_P, ge=0.0, le=1.0)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0)
    text_split_method: str = "cut5"
    batch_size: int = Field(1, ge=1)
    batch_threshold: float = Field(0.75, ge=0.0)
    split_bucket: bool = False
    speed_factor: float = Field(DEFAULT_SPEED_FACTOR, gt=0.0)
    fragment_interval: float = Field(DEFAULT_FRAGMENT_INTERVAL, ge=0.0)
    seed: int = -1
    parallel_infer: bool = False
    repetition_penalty: float = Field(DEFAULT_REPETITION_PENALTY, ge=0.0)
    sample_steps: int = Field(32, ge=1)
    super_sampling: bool = False
    best_of: int = Field(4, ge=1, le=8)
    retry_bad_segments: bool = True


class SaveTTSRequest(TTSRequest):
    filename: Optional[str] = None


class WeightsRequest(BaseModel):
    gpt_weights_path: Optional[str] = None
    sovits_weights_path: Optional[str] = None


class VoiceEngine:
    def __init__(self, config_path: str):
        self.config = TTS_Config(config_path)
        self._assert_model_paths_exist()
        self.pipeline = TTS(self.config)
        self.cut_methods = set(get_method_names())
        self.lock = threading.Lock()

    def _assert_model_paths_exist(self) -> None:
        required_paths = {
            "GPT weights": self.config.t2s_weights_path,
            "SoVITS weights": self.config.vits_weights_path,
            "BERT model": self.config.bert_base_path,
            "CN-HuBERT model": self.config.cnhuhbert_base_path,
        }
        missing = [f"{label}: {path}" for label, path in required_paths.items() if not Path(path).exists()]
        if missing:
            joined = "\n".join(missing)
            raise RuntimeError(
                "GPT-SoVITS pretrained model files are missing.\n"
                f"{joined}\n\n"
                "Download the model files first:\n"
                "  powershell -ExecutionPolicy Bypass -File .\\download_voice_models.ps1 -Source HF\n"
                "If Python or PyTorch packages are also missing, run setup_voice_api.ps1 with the matching Device."
            )


engine: Optional[VoiceEngine] = None
APP = FastAPI(
    title="GPT-SoVITS Voice Clone TTS API",
    description="A small API wrapper for storing reference voices and generating WAV TTS.",
    version="1.0.0",
)
cors_origins = [origin.strip() for origin in os.getenv("GSV_CORS_ORIGINS", "*").split(",") if origin.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
APP.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_engine() -> VoiceEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="TTS engine is not loaded")
    return engine


@APP.get("/")
def web_app() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def normalize_language(language: Optional[str], field_name: str) -> str:
    value = (language or "").strip().lower()
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")

    active_engine = ensure_engine()
    if value not in active_engine.config.languages:
        supported = ", ".join(active_engine.config.languages)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} '{value}' is not supported. Supported: {supported}",
        )
    return value


def validate_voice_id(voice_id: str) -> str:
    value = (voice_id or "").strip()
    if not VOICE_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail="voice_id must be 1-64 chars: letters, numbers, underscore, or hyphen",
        )
    return value


def voice_dir_for(voice_id: str) -> Path:
    safe_id = validate_voice_id(voice_id)
    path = (VOICES_DIR / safe_id).resolve()
    if VOICES_DIR not in path.parents and path != VOICES_DIR:
        raise HTTPException(status_code=400, detail="invalid voice_id")
    return path


def metadata_path_for(voice_id: str) -> Path:
    return voice_dir_for(voice_id) / "voice.json"


def load_voice(voice_id: str) -> Dict[str, Any]:
    path = metadata_path_for(voice_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' was not found")
    return json.loads(path.read_text(encoding="utf-8"))


def list_voice_metadata() -> List[Dict[str, Any]]:
    voices = []
    for metadata_path in sorted(VOICES_DIR.glob("*/voice.json")):
        try:
            voices.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return voices


def check_audio_extension(filename: Optional[str]) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"audio file extension must be one of: {allowed}")
    return extension


def save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(upload.file, output)


def convert_audio_to_wav(source: Path, destination: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg_path:
        raise HTTPException(status_code=500, detail="ffmpeg was not found; cannot convert reference audio")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "32000",
        "-af",
        "loudnorm=I=-18:TP=-1.5:LRA=11",
        "-f",
        "wav",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or "unknown ffmpeg error"
        raise HTTPException(status_code=400, detail=f"failed to convert reference audio to wav: {message}")


def output_filename(filename: Optional[str]) -> str:
    if filename:
        candidate = filename.strip()
        if not candidate.lower().endswith(".wav"):
            candidate += ".wav"
        if not OUTPUT_NAME_RE.match(candidate):
            raise HTTPException(status_code=400, detail="filename contains unsupported characters")
        return candidate
    return f"tts-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"


def audio_to_float(audio_data: Any) -> np.ndarray:
    audio_data = np.asarray(audio_data)
    if np.issubdtype(audio_data.dtype, np.integer):
        return audio_data.astype(np.float32) / float(np.iinfo(audio_data.dtype).max)
    return audio_data.astype(np.float32)


def wav_bytes(sample_rate: int, audio_data: Any) -> bytes:
    audio_data = audio_to_float(audio_data)
    peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
    if 0.0001 < peak < 0.88:
        audio_data = audio_data * (0.88 / peak)
    audio_data = np.clip(audio_data, -0.98, 0.98)
    buffer = BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    return buffer.getvalue()


def has_korean(text: str) -> bool:
    return bool(KOREAN_CHAR_RE.search(text))


def normalize_tts_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = text.replace("~", ".").replace("…", ".")
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=[^\s,.!?;:])", r"\1 ", text)
    return text.strip()


def speech_char_count(text: str) -> int:
    return len(re.sub(r"[\s,.!?;:，、。！？…\"'()\[\]{}<>]", "", text))


def ensure_terminal_punctuation(text: str) -> str:
    text = text.strip()
    if text and not TERMINAL_PUNCT_RE.search(text):
        return f"{text}."
    return text


def compact_korean_pause_punctuation(text: str) -> str:
    if not has_korean(text):
        return text
    text = re.sub(r"\s*[,，、;:]\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentence_parts(text: str) -> List[str]:
    text = normalize_tts_text(text)
    if not text:
        return []
    return [item.strip() for item in re.findall(r".+?(?:[.!?。！？…]+|$)", text) if item.strip()]


def korean_first_sentence_split(text: str) -> tuple[Optional[str], str]:
    text = normalize_tts_text(text)
    if not has_korean(text):
        return None, text
    parts = split_sentence_parts(text)
    if len(parts) < 2:
        return None, text
    first_sentence = ensure_terminal_punctuation(parts[0])
    if speech_char_count(first_sentence) > 14:
        return None, text
    remainder = " ".join(parts[1:]).strip()
    return first_sentence, remainder


def split_long_text_part(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    tokens = re.findall(r"\S+", text)
    for token in tokens:
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = token
        elif len(token) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for index in range(0, len(token), max_chars):
                chunks.append(token[index : index + max_chars].strip())
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_part_for_tts(part: str, max_chars: int) -> List[str]:
    if len(part) <= max_chars:
        return [part]

    chunks: List[str] = []
    clauses = [item.strip() for item in re.split(r"(?<=[,;:，、])\s*", part) if item.strip()]
    current = ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip()
        if current and len(candidate) > max_chars:
            chunks.extend(split_long_text_part(current, max_chars))
            current = clause
        else:
            current = candidate
    if current:
        chunks.extend(split_long_text_part(current, max_chars))
    return chunks


def pop_first_word(text: str) -> tuple[str, str]:
    words = text.split()
    if not words:
        return "", ""
    return words[0], " ".join(words[1:]).strip()


def pop_last_word(text: str) -> tuple[str, str]:
    words = text.split()
    if not words:
        return "", ""
    return " ".join(words[:-1]).strip(), words[-1]


def rebalance_short_tts_chunks(chunks: List[str], min_chars: int, max_chars: int) -> List[str]:
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if len(chunks) <= 1:
        return chunks

    max_with_borrow = max_chars + 4
    for index in range(len(chunks) - 1):
        while speech_char_count(chunks[index]) < min_chars and chunks[index + 1]:
            word, remainder = pop_first_word(chunks[index + 1])
            if not word:
                break
            candidate = f"{chunks[index]} {word}".strip()
            if len(candidate) > max_with_borrow:
                break
            chunks[index] = candidate
            chunks[index + 1] = remainder

    for index in range(len(chunks) - 1, 0, -1):
        while speech_char_count(chunks[index]) < min_chars and chunks[index - 1]:
            remainder, word = pop_last_word(chunks[index - 1])
            if not word:
                break
            candidate = f"{word} {chunks[index]}".strip()
            if len(candidate) > max_with_borrow or speech_char_count(remainder) < min_chars:
                break
            chunks[index - 1] = remainder
            chunks[index] = candidate

    cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
    merged: List[str] = []
    for chunk in cleaned:
        if merged and speech_char_count(chunk) < 8:
            candidate = f"{merged[-1]} {chunk}".strip()
            if len(candidate) <= max_with_borrow:
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return merged


def split_text_for_tts(text: str, max_chars: int = MAX_TTS_CHARS_PER_CHUNK) -> List[str]:
    text = normalize_tts_text(text)
    if not text:
        return []

    korean = has_korean(text)
    if korean:
        max_chars = min(max_chars, KOREAN_MAX_TTS_CHARS_PER_CHUNK)

    parts = split_sentence_parts(text)
    if not parts:
        parts = [text]

    if korean and len(parts) > 1 and speech_char_count(parts[0]) <= 12:
        first_pair = f"{parts[0]} {parts[1]}".strip()
        if len(first_pair) <= max_chars + 16:
            parts = [first_pair, *parts[2:]]

    chunks: List[str] = []
    for part in parts:
        chunks.extend(split_part_for_tts(part, max_chars))

    if korean:
        chunks = rebalance_short_tts_chunks(chunks, KOREAN_MIN_TTS_CHARS_PER_CHUNK, max_chars)

    return [ensure_terminal_punctuation(chunk) for chunk in chunks if chunk.strip()]


def estimate_min_duration(text: str) -> float:
    visible_chars = speech_char_count(text)
    if has_korean(text):
        return max(1.0, min(visible_chars / 6.7, 18.0))
    return max(0.35, min(visible_chars / 11.0, 10.0))


def estimate_max_duration(text: str) -> float:
    min_duration = estimate_min_duration(text)
    if has_korean(text):
        return max(min_duration + 3.0, min_duration * 2.6)
    return max(min_duration + 1.5, min_duration * 2.3)


def audio_quality(audio: np.ndarray, sample_rate: int, text: str) -> Dict[str, float]:
    audio = audio_to_float(audio)
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    active_ratio = float(np.mean(np.abs(audio) > 0.003)) if audio.size else 0.0
    min_duration = estimate_min_duration(text)
    max_duration = estimate_max_duration(text)
    korean = has_korean(text)
    rms_floor = 0.010 if korean else 0.0015
    peak_floor = 0.035 if korean else 0.006
    duration_floor_ratio = 0.78 if korean else 0.55

    duration_score = min(duration / min_duration, 1.6) if min_duration else 0.0
    loudness_score = min(rms * 35.0, 1.6)
    peak_score = min(peak * 2.0, 1.0)
    short_penalty = max((min_duration * duration_floor_ratio - duration) / max(min_duration, 0.1), 0.0) * 4.0
    quiet_penalty = max((rms_floor - rms) / max(rms_floor, 0.0001), 0.0) * 3.0
    peak_penalty = max((peak_floor - peak) / max(peak_floor, 0.0001), 0.0) * 1.5
    silence_penalty = 3.0 if peak < peak_floor * 0.2 or rms < rms_floor * 0.2 else 0.0
    long_penalty = max((duration - max_duration) / max(max_duration, 0.1), 0.0) * 6.0
    frame_size = max(1, int(sample_rate * 0.05)) if sample_rate else 1
    rms_values = frame_rms(audio, frame_size)
    active_rms = rms_values[rms_values > max(rms_floor * 0.5, 0.003)] if rms_values.size else np.array([])
    if active_rms.size >= 3 and float(np.mean(active_rms)) > 0:
        prosody_variation = float(np.std(active_rms) / max(float(np.mean(active_rms)), 0.0001))
    else:
        prosody_variation = 0.0
    variation_score = min(max(prosody_variation - 0.08, 0.0) * (2.0 if korean else 1.0), 0.85)
    flat_penalty = max((0.11 - prosody_variation) / 0.11, 0.0) * (0.45 if korean else 0.15)

    score = (
        duration_score
        + loudness_score
        + peak_score
        + active_ratio
        + variation_score
        - short_penalty
        - quiet_penalty
        - peak_penalty
        - silence_penalty
        - long_penalty
        - flat_penalty
    )
    is_bad = bool(
        peak < peak_floor
        or rms < rms_floor
        or duration < min_duration * duration_floor_ratio
        or duration > max_duration * 1.35
    )
    return {
        "score": score,
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "active_ratio": active_ratio,
        "prosody_variation": prosody_variation,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "is_bad": float(is_bad),
    }


def normalize_segment_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio_to_float(audio)
    if not audio.size:
        return audio

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    if peak <= 0.0001 or rms <= 0.0001:
        return audio

    target_rms = 0.070
    target_peak = 0.88
    scale = 1.0
    if rms < target_rms:
        scale = min(target_rms / rms, target_peak / peak, 80.0)
    if scale > 1.0:
        audio = np.clip(audio * scale, -0.98, 0.98)
    return audio


def pad_segment_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = audio_to_float(audio)
    if not audio.size or sample_rate <= 0:
        return audio
    pre = np.zeros(int(sample_rate * SEGMENT_PRE_PAD_SECONDS), dtype=np.float32)
    post = np.zeros(int(sample_rate * SEGMENT_POST_PAD_SECONDS), dtype=np.float32)
    return np.concatenate([pre, audio, post])


def trim_trailing_silence(audio: np.ndarray, sample_rate: int, keep_seconds: float = SEGMENT_TRAILING_SILENCE_SECONDS) -> np.ndarray:
    audio = audio_to_float(audio)
    if not audio.size or sample_rate <= 0:
        return audio

    frame_size = max(1, int(sample_rate * 0.02))
    rms_values = frame_rms(audio, frame_size)
    if not rms_values.size:
        return audio

    threshold = max(float(np.max(rms_values)) * 0.04, 0.003)
    active_indices = np.flatnonzero(rms_values > threshold)
    if not active_indices.size:
        return audio

    last_active_frame = int(active_indices[-1])
    keep_samples = int(sample_rate * max(0.0, keep_seconds))
    end_sample = min(len(audio), (last_active_frame + 1) * frame_size + keep_samples)
    return audio[:end_sample]


def frame_rms(audio: np.ndarray, frame_size: int) -> np.ndarray:
    if len(audio) < frame_size:
        return np.array([], dtype=np.float32)
    usable = audio[: len(audio) - (len(audio) % frame_size)]
    if not usable.size:
        return np.array([], dtype=np.float32)
    frames = usable.reshape(-1, frame_size)
    return np.sqrt(np.mean(frames * frames, axis=1))


def trim_sacrificial_prefix(
    audio: np.ndarray,
    sample_rate: int,
    max_cut_seconds: float = 2.4,
    merge_gap_seconds: float = 0.18,
) -> np.ndarray:
    audio = audio_to_float(audio)
    if not audio.size or sample_rate <= 0:
        return audio

    frame_size = max(1, int(sample_rate * 0.02))
    rms_values = frame_rms(audio, frame_size)
    if rms_values.size < 8:
        return audio

    threshold = max(float(np.max(rms_values)) * 0.10, 0.006)
    active = rms_values > threshold
    raw_regions: List[tuple[int, int]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and active[index]:
            index += 1
        end = index
        if end - start >= 2:
            raw_regions.append((start, end))

    regions: List[tuple[int, int]] = []
    max_word_gap_frames = max(2, int(merge_gap_seconds / 0.02))
    for start, end in raw_regions:
        if regions and start - regions[-1][1] <= max_word_gap_frames:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    if len(regions) < 2:
        return audio

    second_start_frame = regions[1][0]
    cut_frame = max(0, second_start_frame - 2)
    cut_sample = min(len(audio), cut_frame * frame_size)
    if cut_sample <= 0 or cut_sample > int(sample_rate * max_cut_seconds):
        return audio
    return audio[cut_sample:]


def first_sentence_candidate_ok(audio: np.ndarray, sample_rate: int, first_sentence: str) -> bool:
    if sample_rate <= 0:
        return False
    duration = len(audio_to_float(audio)) / sample_rate
    speech_chars = max(1, speech_char_count(first_sentence))
    estimated_duration = estimate_min_duration(first_sentence)
    min_duration = max(0.70, estimated_duration * 0.72)
    max_duration = max(2.50, estimated_duration * 2.80, speech_chars * 0.45)
    return min_duration <= duration <= max_duration


def duplicate_segment_text(text_chunk: str, chunk_index: int) -> str:
    duplicated_text = f"{text_chunk} {text_chunk}".strip()
    if chunk_index != 0 and KOREAN_MODEL_PREROLL:
        duplicated_text = f"{KOREAN_MODEL_PREROLL} {duplicated_text}".strip()
    return duplicated_text


def seed_for_attempt(base_seed: Any, chunk_index: int, attempt: int) -> int:
    try:
        seed = int(base_seed)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return -1
    return seed + chunk_index * 1009 + attempt


def apply_attempt_sampling(candidate_inputs: Dict[str, Any], attempt: int, korean: bool) -> None:
    if not korean:
        return

    candidate_inputs["split_bucket"] = False
    candidate_inputs["batch_size"] = 1
    candidate_inputs["parallel_infer"] = False
    base_speed = min(max(float(candidate_inputs.get("speed_factor", DEFAULT_SPEED_FACTOR)), 0.85), 1.30)
    base_top_k = int(candidate_inputs.get("top_k", DEFAULT_TOP_K))
    base_top_p = float(candidate_inputs.get("top_p", DEFAULT_TOP_P))
    base_temperature = float(candidate_inputs.get("temperature", DEFAULT_TEMPERATURE))
    base_repetition = float(candidate_inputs.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    candidate_inputs["speed_factor"] = base_speed
    variant = attempt % 5
    if variant == 1:
        candidate_inputs["top_k"] = max(8, min(base_top_k, 12))
        candidate_inputs["top_p"] = max(0.82, min(base_top_p, 0.86))
        candidate_inputs["temperature"] = max(0.68, min(base_temperature, 0.74))
        candidate_inputs["repetition_penalty"] = max(base_repetition, 1.18)
    elif variant == 2:
        candidate_inputs["top_k"] = max(14, base_top_k)
        candidate_inputs["top_p"] = max(base_top_p, 0.90)
        candidate_inputs["temperature"] = max(base_temperature, 0.82)
        candidate_inputs["speed_factor"] = min(base_speed + 0.02, 1.30)
        candidate_inputs["repetition_penalty"] = min(base_repetition, 1.16)
    elif variant == 3:
        candidate_inputs["top_k"] = max(10, min(base_top_k, 14))
        candidate_inputs["top_p"] = max(base_top_p, 0.88)
        candidate_inputs["temperature"] = max(base_temperature, 0.78)
        candidate_inputs["speed_factor"] = max(base_speed - 0.02, 0.85)
        candidate_inputs["repetition_penalty"] = min(base_repetition, 1.15)
    elif variant == 4:
        candidate_inputs["top_k"] = max(16, base_top_k)
        candidate_inputs["top_p"] = max(base_top_p, 0.92)
        candidate_inputs["temperature"] = max(base_temperature, 0.86)
        candidate_inputs["speed_factor"] = min(base_speed + 0.01, 1.30)
        candidate_inputs["repetition_penalty"] = min(base_repetition, 1.12)


def collect_tts_audio(active_engine: VoiceEngine, inputs: Dict[str, Any]) -> tuple[int, np.ndarray]:
    generator = active_engine.pipeline.run(inputs)
    sample_rate: Optional[int] = None
    parts: List[np.ndarray] = []
    for current_sample_rate, audio_data in generator:
        if sample_rate is None:
            sample_rate = current_sample_rate
        elif sample_rate != current_sample_rate:
            raise RuntimeError(f"sample rate changed during synthesis: {sample_rate} -> {current_sample_rate}")
        parts.append(audio_to_float(audio_data))

    if sample_rate is None or not parts:
        raise RuntimeError("tts generator returned no audio")
    return sample_rate, np.concatenate(parts)


def model_text_for_chunk(text_chunk: str, chunk_index: int) -> str:
    text_chunk = compact_korean_pause_punctuation(text_chunk)
    if chunk_index == 0 and has_korean(text_chunk) and KOREAN_MODEL_PREROLL:
        return f"{KOREAN_MODEL_PREROLL} {text_chunk}".strip()
    return text_chunk


def warm_up_korean_generation(active_engine: VoiceEngine, inputs: Dict[str, Any]) -> None:
    if not has_korean(inputs.get("text", "")) or not KOREAN_WARMUP_TEXT.strip():
        return

    warmup_inputs = dict(inputs)
    warmup_inputs["text"] = f"{KOREAN_MODEL_PREROLL} {KOREAN_WARMUP_TEXT}".strip()
    warmup_inputs["text_lang"] = "ko"
    warmup_inputs["text_split_method"] = "cut0"
    warmup_inputs["seed"] = -1
    warmup_inputs["split_bucket"] = False
    warmup_inputs["parallel_infer"] = False
    warmup_inputs["batch_size"] = 1
    warmup_inputs["speed_factor"] = min(max(float(warmup_inputs.get("speed_factor", DEFAULT_SPEED_FACTOR)), 0.85), 1.30)
    warmup_inputs["top_k"] = min(int(warmup_inputs.get("top_k", DEFAULT_TOP_K)), 12)
    warmup_inputs["top_p"] = min(float(warmup_inputs.get("top_p", DEFAULT_TOP_P)), 0.88)
    warmup_inputs["temperature"] = min(float(warmup_inputs.get("temperature", DEFAULT_TEMPERATURE)), 0.78)
    warmup_inputs["repetition_penalty"] = max(float(warmup_inputs.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)), 1.16)

    try:
        collect_tts_audio(active_engine, warmup_inputs)
        print("korean_warmup_done", json.dumps({"text": warmup_inputs["text"]}, ensure_ascii=False))
    except Exception as exc:
        print("korean_warmup_failed", str(exc))


def synthesize_best_segment(
    active_engine: VoiceEngine,
    inputs: Dict[str, Any],
    text_chunk: str,
    chunk_index: int,
) -> tuple[int, np.ndarray]:
    best_of = max(1, min(int(inputs.get("best_of", 1)), 10))
    retry_bad_segments = bool(inputs.get("retry_bad_segments", True))
    if not retry_bad_segments:
        best_of = 1
    if retry_bad_segments and has_korean(text_chunk):
        best_of = max(best_of, 4)

    best_sample_rate: Optional[int] = None
    best_audio: Optional[np.ndarray] = None
    best_metrics: Optional[Dict[str, float]] = None
    errors: List[str] = []
    korean = has_korean(text_chunk)

    for attempt in range(best_of):
        candidate_inputs = dict(inputs)
        candidate_inputs["text"] = model_text_for_chunk(text_chunk, chunk_index)
        candidate_inputs["text_split_method"] = "cut0"
        candidate_inputs["seed"] = seed_for_attempt(inputs.get("seed", -1), chunk_index, attempt)
        apply_attempt_sampling(candidate_inputs, attempt, korean)

        try:
            sample_rate, audio = collect_tts_audio(active_engine, candidate_inputs)
            metrics = audio_quality(audio, sample_rate, text_chunk)
        except Exception as exc:
            errors.append(str(exc))
            continue

        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_sample_rate = sample_rate
            best_audio = audio
            best_metrics = metrics

    if best_sample_rate is None or best_audio is None or best_metrics is None:
        joined_errors = "; ".join(errors) if errors else "no candidates"
        raise RuntimeError(f"tts failed for segment '{text_chunk}': {joined_errors}")

    if best_metrics["peak"] <= 0.0001:
        raise RuntimeError(f"tts returned silence for segment '{text_chunk}'")

    best_audio = pad_segment_edges(
        trim_trailing_silence(normalize_segment_audio(best_audio), best_sample_rate),
        best_sample_rate,
    )
    final_metrics = audio_quality(best_audio, best_sample_rate, text_chunk)
    print(
        "segment_quality",
        json.dumps(
            {
                "chunk": chunk_index,
                "text": text_chunk,
                "best_of": best_of,
                "score": round(best_metrics["score"], 4),
                "duration": round(best_metrics["duration"], 3),
                "rms": round(best_metrics["rms"], 5),
                "prosody": round(best_metrics["prosody_variation"], 4),
                "final_rms": round(final_metrics["rms"], 5),
                "final_peak": round(final_metrics["peak"], 4),
                "bad": bool(best_metrics["is_bad"]),
            },
            ensure_ascii=False,
        ),
    )
    return best_sample_rate, best_audio


def synthesize_korean_duplicate_segment(
    active_engine: VoiceEngine,
    inputs: Dict[str, Any],
    text_chunk: str,
    chunk_index: int,
) -> tuple[int, np.ndarray, bool]:
    duplicate_inputs = dict(inputs)
    duplicate_inputs["best_of"] = max(int(duplicate_inputs.get("best_of", 4)), 6)

    duplicated_text = duplicate_segment_text(text_chunk, chunk_index)
    sample_rate, duplicated_audio = synthesize_best_segment(
        active_engine,
        duplicate_inputs,
        duplicated_text,
        chunk_index,
    )
    duplicated_trimmed = trim_sacrificial_prefix(
        duplicated_audio,
        sample_rate,
        max_cut_seconds=5.0,
        merge_gap_seconds=0.42,
    )
    duplicated_cut_seconds = (len(duplicated_audio) - len(duplicated_trimmed)) / sample_rate if sample_rate else 0.0
    duplicated_trimmed = pad_segment_edges(
        trim_trailing_silence(normalize_segment_audio(duplicated_trimmed), sample_rate),
        sample_rate,
    )
    expected_cut_seconds = max(0.75, estimate_min_duration(text_chunk) * 0.80)
    duplicated_ok = duplicated_cut_seconds >= expected_cut_seconds and first_sentence_candidate_ok(
        duplicated_trimmed,
        sample_rate,
        text_chunk,
    )
    full_single_ok = duplicated_cut_seconds < 0.25 and first_sentence_candidate_ok(
        duplicated_audio,
        sample_rate,
        text_chunk,
    )
    print(
        "korean_duplicate_segment",
        json.dumps(
            {
                "chunk": chunk_index,
                "duplicated_text": duplicated_text,
                "original_duration": round(len(duplicated_audio) / sample_rate, 3) if sample_rate else 0,
                "trimmed_duration": round(len(duplicated_trimmed) / sample_rate, 3) if sample_rate else 0,
                "cut_seconds": round(duplicated_cut_seconds, 3),
                "expected_cut_seconds": round(expected_cut_seconds, 3),
                "mode": "trimmed" if duplicated_ok else ("single" if full_single_ok else "unused"),
            },
            ensure_ascii=False,
        ),
    )
    if duplicated_ok:
        return sample_rate, duplicated_trimmed, True
    if full_single_ok:
        return sample_rate, pad_segment_edges(
            trim_trailing_silence(normalize_segment_audio(duplicated_audio), sample_rate),
            sample_rate,
        ), True
    return sample_rate, duplicated_trimmed, False


def synthesize_first_sentence_with_sacrifice(
    active_engine: VoiceEngine,
    inputs: Dict[str, Any],
    first_sentence: str,
) -> tuple[int, np.ndarray]:
    sample_rate, duplicated_audio, duplicated_used = synthesize_korean_duplicate_segment(
        active_engine,
        inputs,
        first_sentence,
        0,
    )
    print(
        "first_sentence_duplicate",
        json.dumps(
            {
                "text": first_sentence,
                "used": duplicated_used,
            },
            ensure_ascii=False,
        ),
    )
    if duplicated_used:
        return sample_rate, duplicated_audio

    first_sentence_inputs = dict(inputs)
    first_sentence_inputs["best_of"] = max(int(first_sentence_inputs.get("best_of", 4)), 6)

    sacrifice_text = f"{KOREAN_SACRIFICE_PREFIX} {first_sentence}".strip()
    sample_rate, audio = synthesize_best_segment(active_engine, first_sentence_inputs, sacrifice_text, 0)
    trimmed_audio = trim_sacrificial_prefix(audio, sample_rate)
    trimmed_audio = pad_segment_edges(
        trim_trailing_silence(normalize_segment_audio(trimmed_audio), sample_rate),
        sample_rate,
    )
    print(
        "first_sentence_sacrifice",
        json.dumps(
            {
                "sacrifice_text": sacrifice_text,
                "original_duration": round(len(audio) / sample_rate, 3) if sample_rate else 0,
                "trimmed_duration": round(len(trimmed_audio) / sample_rate, 3) if sample_rate else 0,
            },
            ensure_ascii=False,
        ),
    )
    return sample_rate, trimmed_audio


def build_tts_inputs(request: TTSRequest) -> Dict[str, Any]:
    active_engine = ensure_engine()
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    profile: Optional[Dict[str, Any]] = None
    if request.voice_id:
        profile = load_voice(request.voice_id)

    ref_audio_path = request.ref_audio_path or (profile or {}).get("reference_audio")
    prompt_text = request.prompt_text
    if prompt_text is None:
        prompt_text = (profile or {}).get("prompt_text")
    prompt_lang = request.prompt_lang or (profile or {}).get("prompt_lang")

    if not ref_audio_path:
        raise HTTPException(status_code=400, detail="voice_id or ref_audio_path is required")
    if not Path(ref_audio_path).exists():
        raise HTTPException(status_code=400, detail=f"reference audio was not found: {ref_audio_path}")
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required")

    text_lang = normalize_language(request.text_lang, "text_lang")
    prompt_lang = normalize_language(prompt_lang, "prompt_lang")

    if request.text_split_method not in active_engine.cut_methods:
        supported = ", ".join(sorted(active_engine.cut_methods))
        raise HTTPException(
            status_code=400,
            detail=f"text_split_method '{request.text_split_method}' is not supported. Supported: {supported}",
        )

    missing_aux = [path for path in request.aux_ref_audio_paths if not Path(path).exists()]
    if missing_aux:
        raise HTTPException(status_code=400, detail=f"aux_ref_audio_paths not found: {missing_aux}")

    return {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": str(ref_audio_path),
        "aux_ref_audio_paths": request.aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "temperature": request.temperature,
        "text_split_method": request.text_split_method,
        "batch_size": request.batch_size,
        "batch_threshold": request.batch_threshold,
        "split_bucket": False if has_korean(text) else request.split_bucket,
        "speed_factor": request.speed_factor,
        "fragment_interval": request.fragment_interval,
        "seed": request.seed,
        "media_type": "wav",
        "streaming_mode": False,
        "return_fragment": False,
        "fixed_length_chunk": False,
        "parallel_infer": request.parallel_infer,
        "repetition_penalty": request.repetition_penalty,
        "sample_steps": request.sample_steps,
        "super_sampling": request.super_sampling,
        "best_of": request.best_of,
        "retry_bad_segments": request.retry_bad_segments,
        "overlap_length": 2,
        "min_chunk_length": 16,
    }


def synthesize_wav(request: TTSRequest) -> bytes:
    active_engine = ensure_engine()
    inputs = build_tts_inputs(request)
    try:
        audio_parts: List[np.ndarray] = []
        sample_rate: Optional[int] = None
        first_sentence, remainder_text = korean_first_sentence_split(inputs["text"])
        text_chunks = split_text_for_tts(remainder_text if first_sentence else inputs["text"])
        if not text_chunks:
            if first_sentence:
                text_chunks = []
            else:
                raise HTTPException(status_code=400, detail="text is required")

        with active_engine.lock:
            warm_up_korean_generation(active_engine, inputs)
            if first_sentence:
                current_sample_rate, audio_data = synthesize_first_sentence_with_sacrifice(
                    active_engine,
                    inputs,
                    first_sentence,
                )
                sample_rate = current_sample_rate
                if KOREAN_INITIAL_SILENCE_SECONDS > 0:
                    audio_parts.append(
                        np.zeros(int(current_sample_rate * KOREAN_INITIAL_SILENCE_SECONDS), dtype=np.float32)
                    )
                audio_parts.append(audio_data)
                if text_chunks:
                    audio_parts.append(
                        np.zeros(
                            int(current_sample_rate * max(0.0, float(inputs.get("fragment_interval", 0.0)))),
                            dtype=np.float32,
                        )
                    )

            for chunk_index, text_chunk in enumerate(text_chunks):
                effective_chunk_index = chunk_index + (1 if first_sentence else 0)
                if has_korean(text_chunk) and speech_char_count(text_chunk) <= KOREAN_DUPLICATE_MAX_CHARS:
                    current_sample_rate, audio_data, duplicate_used = synthesize_korean_duplicate_segment(
                        active_engine,
                        inputs,
                        text_chunk,
                        effective_chunk_index,
                    )
                    if not duplicate_used:
                        current_sample_rate, audio_data = synthesize_best_segment(
                            active_engine,
                            inputs,
                            text_chunk,
                            effective_chunk_index,
                        )
                else:
                    current_sample_rate, audio_data = synthesize_best_segment(
                        active_engine,
                        inputs,
                        text_chunk,
                        effective_chunk_index,
                    )
                if sample_rate is None:
                    sample_rate = current_sample_rate
                    if has_korean(inputs["text"]) and KOREAN_INITIAL_SILENCE_SECONDS > 0:
                        audio_parts.append(
                            np.zeros(int(current_sample_rate * KOREAN_INITIAL_SILENCE_SECONDS), dtype=np.float32)
                        )
                elif sample_rate != current_sample_rate:
                    raise RuntimeError(f"sample rate changed during synthesis: {sample_rate} -> {current_sample_rate}")
                audio_parts.append(audio_data)
                pause_seconds = max(0.0, float(inputs.get("fragment_interval", 0.0)))
                if pause_seconds and chunk_index < len(text_chunks) - 1:
                    audio_parts.append(np.zeros(int(current_sample_rate * pause_seconds), dtype=np.float32))

        if sample_rate is None or not audio_parts:
            raise RuntimeError("tts returned no audio")

        combined_audio = np.concatenate(audio_parts)
        peak = float(np.max(np.abs(combined_audio))) if combined_audio.size else 0.0
        if peak <= 0.0001:
            raise RuntimeError("tts returned silence; try a clearer 5-10 second reference voice")

        return wav_bytes(sample_rate, combined_audio)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tts failed: {exc}") from exc


def response_wav(audio: bytes, filename: str = "tts.wav") -> Response:
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@APP.get("/health")
def health() -> Dict[str, Any]:
    active_engine = ensure_engine()
    return {
        "ok": True,
        "version": active_engine.config.version,
        "device": str(active_engine.config.device),
        "is_half": bool(active_engine.config.is_half),
        "languages": active_engine.config.languages,
        "cut_methods": sorted(active_engine.cut_methods),
        "voices": [voice["voice_id"] for voice in list_voice_metadata()],
    }


@APP.get("/voices")
def list_voices() -> Dict[str, Any]:
    return {"voices": list_voice_metadata()}


@APP.get("/voices/{voice_id}")
def get_voice(voice_id: str) -> Dict[str, Any]:
    return load_voice(voice_id)


@APP.post("/voices")
async def create_voice(
    voice_id: str = Form(...),
    prompt_text: str = Form(...),
    prompt_lang: str = Form("all_ko"),
    consent_confirmed: bool = Form(...),
    overwrite: bool = Form(False),
    reference_audio: UploadFile = File(...),
) -> Dict[str, Any]:
    if not consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="consent_confirmed must be true for any voice profile you register",
        )

    safe_id = validate_voice_id(voice_id)
    normalized_prompt_lang = normalize_language(prompt_lang, "prompt_lang")
    if not prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text is required")

    extension = check_audio_extension(reference_audio.filename)
    voice_dir = voice_dir_for(safe_id)
    metadata_path = voice_dir / "voice.json"
    if metadata_path.exists() and not overwrite:
        raise HTTPException(status_code=409, detail=f"voice '{safe_id}' already exists")

    if voice_dir.exists() and overwrite:
        shutil.rmtree(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    source_audio_path = voice_dir / f"source{extension}"
    reference_audio_path = voice_dir / "reference.wav"
    save_upload(reference_audio, source_audio_path)
    convert_audio_to_wav(source_audio_path, reference_audio_path)

    metadata = {
        "voice_id": safe_id,
        "reference_audio": str(reference_audio_path),
        "source_audio": str(source_audio_path),
        "prompt_text": prompt_text.strip(),
        "prompt_lang": normalized_prompt_lang,
        "created_at": now_iso(),
        "consent_confirmed": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


@APP.delete("/voices/{voice_id}")
def delete_voice(voice_id: str) -> Dict[str, Any]:
    voice_dir = voice_dir_for(voice_id)
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' was not found")
    shutil.rmtree(voice_dir)
    return {"deleted": validate_voice_id(voice_id)}


@APP.post("/tts")
def tts(request: TTSRequest) -> Response:
    return response_wav(synthesize_wav(request))


@APP.post("/tts/save")
def tts_save(request: SaveTTSRequest) -> Dict[str, Any]:
    audio = synthesize_wav(request)
    filename = output_filename(request.filename)
    output_path = OUTPUTS_DIR / filename
    output_path.write_bytes(audio)
    return {
        "filename": filename,
        "path": str(output_path),
        "url": f"/outputs/{filename}",
        "content_type": "audio/wav",
    }


@APP.get("/outputs/{filename}")
def output_file(filename: str) -> FileResponse:
    safe_name = output_filename(filename)
    path = (OUTPUTS_DIR / safe_name).resolve()
    if OUTPUTS_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="invalid output filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="output file was not found")
    return FileResponse(path, media_type="audio/wav", filename=safe_name)


@APP.post("/clone-tts")
async def clone_tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    text_lang: str = Form("all_ko"),
    prompt_lang: str = Form("all_ko"),
    consent_confirmed: bool = Form(...),
    top_k: int = Form(DEFAULT_TOP_K),
    top_p: float = Form(DEFAULT_TOP_P),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    speed_factor: float = Form(DEFAULT_SPEED_FACTOR),
    seed: int = Form(-1),
    best_of: int = Form(4),
    retry_bad_segments: bool = Form(True),
    reference_audio: UploadFile = File(...),
) -> Response:
    if not consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="consent_confirmed must be true when using a reference voice",
        )

    extension = check_audio_extension(reference_audio.filename)
    temp_id = uuid.uuid4().hex
    temp_path = TEMP_DIR / f"{temp_id}{extension}"
    temp_wav_path = TEMP_DIR / f"{temp_id}.wav"
    try:
        save_upload(reference_audio, temp_path)
        convert_audio_to_wav(temp_path, temp_wav_path)
        request = TTSRequest(
            text=text,
            ref_audio_path=str(temp_wav_path),
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            text_lang=text_lang,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed_factor=speed_factor,
            seed=seed,
            best_of=best_of,
            retry_bad_segments=retry_bad_segments,
        )
        return response_wav(synthesize_wav(request))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
            temp_wav_path.unlink(missing_ok=True)
        except OSError:
            pass


@APP.post("/weights")
def set_weights(request: WeightsRequest) -> Dict[str, Any]:
    active_engine = ensure_engine()
    changed = []
    with active_engine.lock:
        if request.gpt_weights_path:
            if not Path(request.gpt_weights_path).exists():
                raise HTTPException(status_code=400, detail="gpt_weights_path was not found")
            active_engine.pipeline.init_t2s_weights(request.gpt_weights_path)
            changed.append("gpt")
        if request.sovits_weights_path:
            if not Path(request.sovits_weights_path).exists():
                raise HTTPException(status_code=400, detail="sovits_weights_path was not found")
            active_engine.pipeline.init_vits_weights(request.sovits_weights_path)
            changed.append("sovits")
    return {"changed": changed, "version": active_engine.config.version}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Program-friendly GPT-SoVITS TTS API")
    parser.add_argument("--host", default=os.getenv("GSV_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GSV_API_PORT", "9881")))
    parser.add_argument(
        "--config",
        default=os.getenv("GSV_TTS_CONFIG", "GPT_SoVITS/configs/tts_infer.yaml"),
        help="Path to GPT-SoVITS tts_infer.yaml",
    )
    return parser.parse_args()


def main() -> None:
    global engine
    args = parse_args()
    engine = VoiceEngine(args.config)
    uvicorn.run(APP, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
