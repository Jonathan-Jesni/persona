"""
voice.py — fully-offline voice for Persona
==========================================
- Text-to-speech via Piper (per-character voices loaded from ./voices/*.onnx)
- Speech-to-text via faster-whisper (model auto-downloads once, then offline)

Both engines are optional: if a package or model is missing, the corresponding
`*_available()` returns False and the API/UI degrade gracefully — the app still
runs. Models live under ./voices (Piper) and the faster-whisper cache.
"""

import io
import os
import wave
import logging

logger = logging.getLogger(__name__)

VOICES_DIR = os.getenv("VOICES_DIR", "voices")
WHISPER_MODEL = os.getenv("PERSONA_WHISPER", "base")

# Lazy singletons
_piper_voices: dict = {}     # voice_id -> loaded PiperVoice
_whisper_model = None


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #
def list_voices() -> list[dict]:
    """Available Piper voices = every *.onnx under VOICES_DIR."""
    out = []
    if os.path.isdir(VOICES_DIR):
        for fn in sorted(os.listdir(VOICES_DIR)):
            if fn.lower().endswith(".onnx"):
                vid = fn[:-5]  # strip .onnx
                # Pretty label: "en_US-amy-medium" -> "Amy (en US, medium)"
                parts = vid.split("-")
                label = vid
                if len(parts) >= 2:
                    name = parts[1].capitalize()
                    extra = ", ".join([parts[0].replace("_", " ")] + parts[2:])
                    label = f"{name} ({extra})" if extra else name
                out.append({"id": vid, "label": label})
    return out


def tts_available() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return len(list_voices()) > 0


def stt_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# Text-to-speech (Piper)                                                      #
# --------------------------------------------------------------------------- #
def _load_voice(voice_id: str):
    if voice_id in _piper_voices:
        return _piper_voices[voice_id]
    from piper import PiperVoice
    onnx = os.path.join(VOICES_DIR, voice_id + ".onnx")
    if not os.path.exists(onnx):
        raise FileNotFoundError(f"Voice model not found: {onnx}")
    config = onnx + ".json"
    voice = PiperVoice.load(onnx, config_path=config if os.path.exists(config) else None)
    _piper_voices[voice_id] = voice
    return voice


def tts_synth(text: str, voice_id: str | None = None) -> bytes:
    """Synthesize `text` into WAV bytes using the given (or first available) voice."""
    voices = list_voices()
    if not voices:
        raise RuntimeError("No Piper voices installed (add *.onnx to ./voices)")
    if not voice_id or voice_id not in {v["id"] for v in voices}:
        voice_id = voices[0]["id"]
    voice = _load_voice(voice_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Speech-to-text (faster-whisper)                                             #
# --------------------------------------------------------------------------- #
def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model '%s' (first run downloads it)…", WHISPER_MODEL)
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def stt_transcribe(audio_bytes: bytes) -> str:
    """Transcribe an audio clip (any container PyAV can decode) to text."""
    model = _get_whisper()
    segments, _info = model.transcribe(io.BytesIO(audio_bytes), beam_size=1)
    return " ".join(seg.text for seg in segments).strip()
