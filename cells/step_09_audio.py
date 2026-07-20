"""Cell 09 — Audio pipeline (Kokoro-82M TTS + faster-whisper-small ASR).

Implements docs/modules/audio.md: TTS and ASR engines exposed at the env
boundary. Training never imports this module (docs/modules/audio.md §6.3).
Heavy deps (``kokoro``, ``faster_whisper``, ``torchaudio``, ``soundfile``)
are loaded lazily inside ``_load_*`` helpers so this cell imports cleanly
in environments where those optional packages are absent, and so tests can
monkeypatch the loaders to return fakes without ever touching the network.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import struct
import threading
import time
import unicodedata
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

import numpy as np
from cachetools import LRUCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public literal types (audio.md §2.1, §2.2)
# ---------------------------------------------------------------------------

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]
VoicePack = Literal[
    "hi_female_1",
    "hi_male_1",
    "ta_female_1",
    "kn_male_1",
    "en_indian_female_1",
]

_LANGUAGE_CODES: frozenset[str] = frozenset({"hi", "ta", "kn", "en", "hinglish"})
_VOICE_PACKS_SET: frozenset[str] = frozenset(
    {
        "hi_female_1",
        "hi_male_1",
        "ta_female_1",
        "kn_male_1",
        "en_indian_female_1",
    }
)


# ---------------------------------------------------------------------------
# Errors (audio.md §2.3)
# ---------------------------------------------------------------------------


class AudioError(Exception):
    """Base class for all audio-module errors."""


class ModelLoadError(AudioError):
    """Raised when Kokoro or faster-whisper cannot be instantiated."""


class UnsupportedLanguageError(AudioError):
    """Raised when a non-registered language code is passed to synthesize()."""


class UnsupportedVoicePackError(AudioError):
    """Raised when a voice pack is not in VOICE_PACKS[lang].allowed."""


class AudioDecodeError(AudioError):
    """Raised when transcribe() cannot decode the input bytes."""


class AudioTooLongError(AudioError):
    """Raised when transcribe() receives audio longer than max_duration_s in strict mode."""


class TTSOutOfMemoryError(AudioError):
    """Raised when TTS synthesis exhausts memory mid-call."""


# ---------------------------------------------------------------------------
# Data records (audio.md §2.1, §2.2, §2.2a, §4.1, §4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoicePackMapping:
    """Per-language default + allowed voice packs. audio.md §4.3."""

    language: LanguageCode
    default: VoicePack
    allowed: tuple[VoicePack, ...]


VOICE_PACKS: dict[LanguageCode, VoicePackMapping] = {
    "hi": VoicePackMapping(
        language="hi",
        default="hi_female_1",
        allowed=("hi_female_1", "hi_male_1"),
    ),
    "ta": VoicePackMapping(
        language="ta",
        default="ta_female_1",
        allowed=("ta_female_1",),
    ),
    "kn": VoicePackMapping(
        language="kn",
        default="kn_male_1",
        allowed=("kn_male_1",),
    ),
    "en": VoicePackMapping(
        language="en",
        default="en_indian_female_1",
        allowed=("en_indian_female_1",),
    ),
    "hinglish": VoicePackMapping(
        language="hinglish",
        default="en_indian_female_1",
        allowed=("en_indian_female_1", "hi_female_1"),
    ),
}


@dataclass(frozen=True)
class TranscriptResult:
    """ASR output surfaced to the env observation builder. audio.md §4.1."""

    text: str
    language_detected: LanguageCode | Literal["unknown"]
    confidence: float
    duration_s: float


@dataclass(frozen=True)
class AudioTrace:
    """Per-call diagnostic record emitted via the configured trace sink.

    audio.md §2.2a, §3.8.
    """

    op: Literal["synthesize", "transcribe"]
    input_hash: str
    language: str
    duration_s: float
    latency_ms: int
    confidence: float | None
    cache_hit: bool
    degraded: bool
    ts_ist: str


TraceSink = Callable[[AudioTrace], None]


# ---------------------------------------------------------------------------
# Lazy dep loaders — patched by tests to inject fakes.
# ---------------------------------------------------------------------------


def _load_kokoro() -> Any:
    """Return the ``kokoro`` module. Patched in tests."""

    import kokoro

    return kokoro


def _load_faster_whisper() -> Any:
    """Return the ``faster_whisper`` module. Patched in tests."""

    import faster_whisper

    return faster_whisper


def _load_torchaudio_functional() -> Any:
    """Return ``torchaudio.functional``. Patched in tests."""

    import torchaudio.functional as F

    return F


def _load_torchaudio() -> Any:
    """Return the top-level ``torchaudio`` module. Patched in tests."""

    import torchaudio

    return torchaudio


def _load_soundfile() -> Any:
    """Return the ``soundfile`` module. Patched in tests."""

    import soundfile

    return soundfile


def _load_torch() -> Any:
    """Return the ``torch`` module. Patched in tests."""

    import torch

    return torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_IST_TZ = timezone(timedelta(hours=5, minutes=30))


def _ts_ist_now() -> str:
    return datetime.now(tz=_IST_TZ).isoformat(timespec="milliseconds")


def _input_hash(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _logprob_to_confidence(avg_logprob: float) -> float:
    """Map faster-whisper ``avg_logprob`` into [0, 1] per audio.md §3.5."""

    clamped = max(-1.5, min(0.0, float(avg_logprob)))
    return round(math.exp(clamped), 3)


def _riff_header_sample_rate(audio_bytes: bytes) -> int | None:
    """Return the sample-rate field from a RIFF header, or None if not RIFF."""

    if len(audio_bytes) < 28:
        return None
    if audio_bytes[0:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return None
    return int(struct.unpack_from("<I", audio_bytes, 24)[0])


def _pcm16_silence_wav(duration_s: float, sample_rate_hz: int = 16000) -> bytes:
    """Build a 16-bit mono PCM WAV of pure silence for warmup / fallback."""

    n_samples = max(1, int(duration_s * sample_rate_hz))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _np_to_wav_bytes(pcm: np.ndarray, sample_rate_hz: int) -> bytes:
    """Encode a float32 mono numpy array as 16-bit PCM RIFF WAV bytes.

    Used when torchaudio is unavailable or mocked — the fallback path
    produces the same byte-level contract (RIFF header + 16 kHz mono 16-bit).
    """

    if pcm.dtype != np.int16:
        clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
        pcm_i16 = (clipped * 32767.0).astype(np.int16)
    else:
        pcm_i16 = pcm
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


_TTS_CACHE_MAX_BYTES: int = 64 * 1024 * 1024
_TTS_CACHE_MAX_ENTRIES: int = 256


def _available_voice_packs(kokoro_module: Any) -> set[str]:
    """Probe the installed Kokoro bundle for shipped voice-pack names.

    Looks for ``AVAILABLE_VOICES``, ``list_voices()``, or ``VOICES``. A fresh
    install typically exposes at least one of these. If none is present we
    fall back to the full canonical set (best-effort; runtime per-call
    fallback in ``_resolve_voice_pack`` still protects against missing packs).
    """

    candidates: set[str] = set()
    for attr in ("AVAILABLE_VOICES", "VOICES"):
        value = getattr(kokoro_module, attr, None)
        if isinstance(value, (list, tuple, set, frozenset)):
            candidates.update(str(v) for v in value)
    list_voices = getattr(kokoro_module, "list_voices", None)
    if callable(list_voices):
        try:
            value = list_voices()
            if isinstance(value, (list, tuple, set, frozenset)):
                candidates.update(str(v) for v in value)
        except Exception:  # pragma: no cover — defensive
            pass
    if not candidates:
        return set(_VOICE_PACKS_SET)
    return candidates


_FALLBACK_CHAIN: dict[str, str] = {
    "ta_female_1": "hi_female_1",
    "kn_male_1": "hi_female_1",
    "hi_male_1": "hi_female_1",
    "hi_female_1": "en_indian_female_1",
}


class TTSEngine:
    """Kokoro-82M wrapper. Constructed via ``get_tts_engine()``.

    One instance per process. All heavy deps are imported lazily.
    """

    def __init__(
        self,
        *,
        model_id: str = "hexgrad/Kokoro-82M",
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._model_id = model_id
        self._trace_sink = trace_sink
        self._lock = threading.Lock()
        self._cache: LRUCache[tuple[Any, ...], bytes] = LRUCache(
            maxsize=_TTS_CACHE_MAX_BYTES, getsizeof=len
        )
        self._numpy_cache: LRUCache[tuple[Any, ...], np.ndarray] = LRUCache(
            maxsize=_TTS_CACHE_MAX_BYTES, getsizeof=lambda a: int(a.nbytes)
        )
        self._fallback_used: dict[str, str] = {}
        try:
            kokoro = _load_kokoro()
        except Exception as exc:  # network / disk / import failure
            raise ModelLoadError(f"failed to load kokoro: {exc}") from exc
        self._kokoro = kokoro
        try:
            pipeline_cls = getattr(kokoro, "KPipeline", None)
            if pipeline_cls is None:
                raise AttributeError("kokoro.KPipeline missing")
            self._pipeline = pipeline_cls(model_id=model_id)
        except Exception as exc:
            raise ModelLoadError(f"failed to construct KPipeline: {exc}") from exc
        self._available_packs = _available_voice_packs(kokoro)
        self._verify_critical_packs()

    def _verify_critical_packs(self) -> None:
        if (
            "en_indian_female_1" not in self._available_packs
            and "hi_female_1" not in self._available_packs
        ):
            raise ModelLoadError("no usable voice pack for hi or en")

    def _resolve_voice_pack(self, requested: VoicePack) -> tuple[VoicePack, bool, str | None]:
        """Walk the fallback chain until an available pack is found.

        Returns ``(resolved_pack, degraded, fallback_from)``.
        """

        current = requested
        original = requested
        degraded = False
        fallback_from: str | None = None
        visited: set[str] = set()
        while current not in self._available_packs:
            if current in visited:
                break
            visited.add(current)
            successor = _FALLBACK_CHAIN.get(current)
            if successor is None:
                raise ModelLoadError(
                    f"no usable voice pack; chain exhausted from {original!r}"
                )
            fallback_from = original
            current = cast("VoicePack", successor)
            degraded = True
        if degraded:
            self._fallback_used[original] = current
        return current, degraded, fallback_from

    def _emit_trace(self, trace: AudioTrace) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(trace)
        except Exception:  # telemetry must never break production
            logger.debug("trace sink raised; swallowed", exc_info=True)

    def _render_pcm(self, text: str, voice_pack: VoicePack, seed: int) -> np.ndarray:
        """Invoke Kokoro inside a forked RNG context and return 24 kHz float32 PCM."""

        torch = _load_torch()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            try:
                result = self._pipeline(text, voice=voice_pack)
            except MemoryError as exc:
                raise TTSOutOfMemoryError(f"TTS OOM: {exc}") from exc
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "alloc" in msg:
                    raise TTSOutOfMemoryError(f"TTS OOM: {exc}") from exc
                raise
        return _coerce_to_float32_mono(result)

    def _resample_to_16k(self, pcm_24k: np.ndarray) -> np.ndarray:
        """Downsample 24 kHz → 16 kHz via torchaudio.functional.resample."""

        try:
            F = _load_torchaudio_functional()
        except Exception as exc:  # pragma: no cover — hard runtime failure
            raise ModelLoadError(f"torchaudio.functional missing: {exc}") from exc
        torch = _load_torch()
        tensor = torch.from_numpy(pcm_24k.astype(np.float32)).unsqueeze(0)
        resampled = F.resample(
            tensor, orig_freq=24000, new_freq=16000, lowpass_filter_width=64
        )
        out = resampled.squeeze(0).cpu().numpy().astype(np.float32)
        return cast("np.ndarray", out)

    def _encode_wav(self, pcm_16k: np.ndarray, sample_rate_hz: int) -> bytes:
        """Encode the 16 kHz float32 PCM into 16-bit mono RIFF WAV bytes."""

        try:
            torchaudio = _load_torchaudio()
            torch = _load_torch()
            tensor = torch.from_numpy(pcm_16k.astype(np.float32)).unsqueeze(0)
            buf = io.BytesIO()
            torchaudio.save(
                buf,
                tensor,
                sample_rate=sample_rate_hz,
                bits_per_sample=16,
                format="wav",
                encoding="PCM_S",
            )
            return buf.getvalue()
        except Exception:
            # Fall back to stdlib wave encoder so the byte contract still holds
            # even when torchaudio is unavailable.
            return _np_to_wav_bytes(pcm_16k, sample_rate_hz)

    def synthesize(
        self,
        text: str,
        language_code: LanguageCode,
        voice_pack: VoicePack | None = None,
        *,
        seed: int = 0,
        sample_rate_hz: int = 16000,
    ) -> bytes:
        """Return 16-bit PCM mono WAV bytes. audio.md §2.1, §4.4."""

        if sample_rate_hz != 16000:
            raise UnsupportedLanguageError(
                f"sample_rate_hz={sample_rate_hz} unsupported; only 16000 allowed in v1"
            )
        if language_code not in _LANGUAGE_CODES:
            raise UnsupportedLanguageError(f"language_code={language_code!r} unsupported")
        mapping = VOICE_PACKS[language_code]
        if voice_pack is None:
            voice_pack = mapping.default
        if voice_pack not in mapping.allowed:
            raise UnsupportedVoicePackError(
                f"voice_pack={voice_pack!r} not allowed for language={language_code!r}"
            )
        text_hash = _input_hash(text.encode("utf-8"))
        cache_key = (text_hash, voice_pack, seed, sample_rate_hz, "bytes")
        start = time.perf_counter()
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            latency_ms = int((time.perf_counter() - start) * 1000)
            duration_s = _wav_duration_s(cached)
            self._emit_trace(
                AudioTrace(
                    op="synthesize",
                    input_hash=text_hash,
                    language=language_code,
                    duration_s=duration_s,
                    latency_ms=latency_ms,
                    confidence=None,
                    cache_hit=True,
                    degraded=False,
                    ts_ist=_ts_ist_now(),
                )
            )
            return cached
        resolved_pack, degraded, _ = self._resolve_voice_pack(voice_pack)
        pcm_24k = self._render_pcm(text, resolved_pack, seed)
        pcm_16k = self._resample_to_16k(pcm_24k)
        wav_bytes = self._encode_wav(pcm_16k, sample_rate_hz)
        with self._lock:
            self._cache[cache_key] = wav_bytes
        latency_ms = int((time.perf_counter() - start) * 1000)
        duration_s = _wav_duration_s(wav_bytes)
        self._emit_trace(
            AudioTrace(
                op="synthesize",
                input_hash=text_hash,
                language=language_code,
                duration_s=duration_s,
                latency_ms=latency_ms,
                confidence=None,
                cache_hit=False,
                degraded=degraded,
                ts_ist=_ts_ist_now(),
            )
        )
        return wav_bytes

    def synthesize_to_gradio(
        self,
        text: str,
        language_hint: LanguageCode,
        voice_pack: VoicePack | None = None,
        *,
        seed: int = 0,
    ) -> tuple[int, np.ndarray]:
        """Return ``(sample_rate, float32 mono ndarray)``. audio.md §2.1."""

        if language_hint not in _LANGUAGE_CODES:
            raise UnsupportedLanguageError(f"language_hint={language_hint!r} unsupported")
        mapping = VOICE_PACKS[language_hint]
        if voice_pack is None:
            voice_pack = mapping.default
        if voice_pack not in mapping.allowed:
            raise UnsupportedVoicePackError(
                f"voice_pack={voice_pack!r} not allowed for language={language_hint!r}"
            )
        text_hash = _input_hash(text.encode("utf-8"))
        sample_rate_hz = 16000
        cache_key = (text_hash, voice_pack, seed, sample_rate_hz, "numpy")
        start = time.perf_counter()
        with self._lock:
            cached = self._numpy_cache.get(cache_key)
        if cached is not None:
            self._emit_trace(
                AudioTrace(
                    op="synthesize",
                    input_hash=text_hash,
                    language=language_hint,
                    duration_s=float(len(cached)) / sample_rate_hz,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    confidence=None,
                    cache_hit=True,
                    degraded=False,
                    ts_ist=_ts_ist_now(),
                )
            )
            return sample_rate_hz, cached.copy()
        resolved_pack, degraded, _ = self._resolve_voice_pack(voice_pack)
        pcm_24k = self._render_pcm(text, resolved_pack, seed)
        pcm_16k = self._resample_to_16k(pcm_24k)
        with self._lock:
            self._numpy_cache[cache_key] = pcm_16k
        self._emit_trace(
            AudioTrace(
                op="synthesize",
                input_hash=text_hash,
                language=language_hint,
                duration_s=float(len(pcm_16k)) / sample_rate_hz,
                latency_ms=int((time.perf_counter() - start) * 1000),
                confidence=None,
                cache_hit=False,
                degraded=degraded,
                ts_ist=_ts_ist_now(),
            )
        )
        return sample_rate_hz, pcm_16k.copy()

    def warmup(self) -> None:
        """Probe each voice pack; log WARN on missing Indic packs. audio.md §4.3.1."""

        for lang, mapping in VOICE_PACKS.items():
            for pack in mapping.allowed:
                if pack not in self._available_packs:
                    logger.warning(
                        "voice pack %r missing from bundle (language=%s); will fall back at synth time",
                        pack,
                        lang,
                    )
        try:
            self.synthesize("warmup", "en")
        except Exception:  # pragma: no cover — warmup best-effort
            logger.debug("warmup synthesize failed; continuing", exc_info=True)


def _coerce_to_float32_mono(result: Any) -> np.ndarray:
    """Turn whatever Kokoro returned into a 1-D float32 numpy array."""

    torch = _load_torch()
    if hasattr(result, "cpu") and hasattr(result, "numpy"):
        arr = result.detach().cpu().numpy()
    elif isinstance(result, tuple):
        audio_like = result[0]
        if hasattr(audio_like, "cpu") and hasattr(audio_like, "numpy"):
            arr = audio_like.detach().cpu().numpy()
        else:
            arr = np.asarray(audio_like)
    elif isinstance(result, np.ndarray):
        arr = result
    else:
        try:
            tensor = torch.as_tensor(result)
            arr = tensor.detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover — defensive
            raise TTSOutOfMemoryError(f"unexpected Kokoro return type: {type(result)!r}: {exc}") from exc
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr


def _wav_duration_s(wav_bytes: bytes) -> float:
    """Return the duration in seconds for a RIFF WAV payload (best-effort)."""

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return round(frames / rate, 3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def _map_language(code: str | None) -> LanguageCode | Literal["unknown"]:
    if code in _LANGUAGE_CODES:
        return cast("LanguageCode", code)
    return "unknown"


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


class ASREngine:
    """faster-whisper-small wrapper. Constructed via ``get_asr_engine()``.

    audio.md §2.2. Heavy deps loaded lazily.
    """

    def __init__(
        self,
        *,
        model_id: str = "Systran/faster-whisper-small",
        compute_type: Literal["int8", "int8_float16"] = "int8",
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._model_id = model_id
        self._compute_type = compute_type
        self._trace_sink = trace_sink
        self._lock = threading.Lock()
        try:
            fw = _load_faster_whisper()
        except Exception as exc:
            raise ModelLoadError(f"failed to load faster_whisper: {exc}") from exc
        model_cls = getattr(fw, "WhisperModel", None)
        if model_cls is None:
            raise ModelLoadError("faster_whisper.WhisperModel missing")
        try:
            self._model = model_cls(model_id, compute_type=compute_type, device="cpu")
        except Exception as exc:
            raise ModelLoadError(f"failed to construct WhisperModel: {exc}") from exc

    def _emit_trace(self, trace: AudioTrace) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(trace)
        except Exception:
            logger.debug("trace sink raised; swallowed", exc_info=True)

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: LanguageCode | None,
        *,
        beam_size: int = 1,
        vad_filter: bool = True,
        max_duration_s: float = 30.0,
    ) -> TranscriptResult:
        """Decode WAV/PCM bytes. audio.md §2.2, §3.5, §4.4."""

        start = time.perf_counter()
        pcm, clip_duration = self._decode_input(audio_bytes)
        if clip_duration > max_duration_s:
            pcm = pcm[: int(max_duration_s * 16000)]
            clip_duration = max_duration_s
        language_for_whisper: str | None
        if language_hint == "hinglish":
            language_for_whisper = "hi"
        elif language_hint is None:
            language_for_whisper = None
        else:
            language_for_whisper = language_hint
        segments, info = self._run_whisper(
            pcm,
            language=language_for_whisper,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        segments_list = list(segments)
        detected_code = _map_language(getattr(info, "language", None))
        vad_dropped_all = getattr(info, "vad_dropped_all_segments", None)
        if vad_dropped_all is None:
            vad_dropped_all = len(segments_list) == 0 and vad_filter
        combined_text = _nfc("".join(getattr(s, "text", "") for s in segments_list))
        duration_s = round(min(float(clip_duration), float(max_duration_s)), 3)
        degraded = False
        if combined_text == "":
            confidence = 0.0
            if vad_dropped_all:
                detected: LanguageCode | Literal["unknown"] = "unknown"
            else:
                detected = detected_code
                degraded = True
        else:
            confidence = _duration_weighted_confidence(segments_list)
            detected = _infer_hinglish(detected_code, combined_text, language_hint)
        result = TranscriptResult(
            text=combined_text,
            language_detected=detected,
            confidence=confidence,
            duration_s=duration_s,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._emit_trace(
            AudioTrace(
                op="transcribe",
                input_hash=_input_hash(audio_bytes),
                language=language_hint or "unknown",
                duration_s=duration_s,
                latency_ms=latency_ms,
                confidence=confidence,
                cache_hit=False,
                degraded=degraded,
                ts_ist=_ts_ist_now(),
            )
        )
        return result

    def _decode_input(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        """Return (float32 mono @ 16 kHz, duration_s); raise AudioDecodeError on mismatch."""

        if len(audio_bytes) >= 3 and audio_bytes[:3] == b"ID3":
            raise AudioDecodeError("MP3 / ID3-tagged inputs are not supported (no ffmpeg in image)")
        rate = _riff_header_sample_rate(audio_bytes)
        if rate is not None:
            if rate != 16000:
                raise AudioDecodeError("input must be 16 kHz mono; caller must pre-resample")
            try:
                sf = _load_soundfile()
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            except Exception as exc:
                raise AudioDecodeError(f"soundfile failed to decode RIFF WAV: {exc}") from exc
            if sr != 16000:
                raise AudioDecodeError("input must be 16 kHz mono; caller must pre-resample")
            arr = np.asarray(data, dtype=np.float32).reshape(-1)
            duration = float(len(arr)) / 16000.0
            return arr, duration
        # Raw float32 PCM path (demo mic input). 16 kHz assumed. We only accept
        # payloads that look like plausible audio — ≥ 0.25 s of float32 samples
        # (4000 × 4 = 16000 bytes) whose magnitudes fit inside the normalized
        # [-1, 1] range that Gradio emits. Short / out-of-range payloads are
        # rejected so arbitrary random bytes do not slip through.
        min_raw_pcm_bytes = 4000 * 4
        if len(audio_bytes) >= min_raw_pcm_bytes and len(audio_bytes) % 4 == 0:
            pcm = np.frombuffer(audio_bytes, dtype=np.float32).copy()
            if pcm.size and np.all(np.isfinite(pcm)) and np.max(np.abs(pcm)) <= 2.0:
                duration = float(pcm.size) / 16000.0
                return pcm, duration
        raise AudioDecodeError("input is not a valid 16 kHz RIFF WAV or float32 PCM payload")

    def _run_whisper(
        self,
        pcm: np.ndarray,
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> tuple[Any, Any]:
        try:
            segments, info = self._model.transcribe(
                pcm,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
        except Exception as exc:
            raise AudioDecodeError(f"whisper decode failed: {exc}") from exc
        return segments, info

    def warmup(self) -> None:
        """Run one transcribe() on 0.5 s of silence to force load. audio.md §2.2."""

        silence = _pcm16_silence_wav(0.5)
        try:
            self.transcribe(silence, "en")
        except Exception:  # pragma: no cover — warmup best-effort
            logger.debug("warmup transcribe failed; continuing", exc_info=True)


def _duration_weighted_confidence(segments: list[Any]) -> float:
    if not segments:
        return 0.0
    total_dur = 0.0
    weighted = 0.0
    for seg in segments:
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        dur = max(0.0, end - start)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        confidence = _logprob_to_confidence(avg_logprob)
        if dur == 0.0:
            total_dur += 1.0
            weighted += confidence
        else:
            total_dur += dur
            weighted += confidence * dur
    if total_dur == 0.0:
        return 0.0
    return round(weighted / total_dur, 3)


def _infer_hinglish(
    detected: LanguageCode | Literal["unknown"],
    text: str,
    hint: LanguageCode | None,
) -> LanguageCode | Literal["unknown"]:
    """Downgrade ``hi`` to ``hinglish`` when the decoded text is code-mixed.

    Heuristic per audio.md §3.6: ≥ 2 ASCII words intermixed with Devanagari.
    """

    if hint != "hinglish":
        return detected
    if detected != "hi":
        return detected
    ascii_words = [tok for tok in text.split() if tok.isascii() and tok.isalpha()]
    has_devanagari = any("ऀ" <= ch <= "ॿ" for ch in text)
    if len(ascii_words) >= 2 and has_devanagari:
        return "hinglish"
    return detected


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


_tts_engine: TTSEngine | None = None
_asr_engine: ASREngine | None = None
_tts_lock = threading.Lock()
_asr_lock = threading.Lock()


def get_tts_engine(
    *, trace_sink: TraceSink | None = None, model_id: str = "hexgrad/Kokoro-82M"
) -> TTSEngine:
    """Return the process-wide TTSEngine singleton. audio.md §3.2, §3.8."""

    global _tts_engine
    with _tts_lock:
        if _tts_engine is None:
            _tts_engine = TTSEngine(model_id=model_id, trace_sink=trace_sink)
        elif trace_sink is not None and trace_sink is not _tts_engine._trace_sink:
            logger.warning("get_tts_engine: different sink passed after construction; ignoring")
        return _tts_engine


def get_asr_engine(
    *,
    trace_sink: TraceSink | None = None,
    model_id: str = "Systran/faster-whisper-small",
    compute_type: Literal["int8", "int8_float16"] = "int8",
) -> ASREngine:
    """Return the process-wide ASREngine singleton. audio.md §3.2, §3.8."""

    global _asr_engine
    with _asr_lock:
        if _asr_engine is None:
            _asr_engine = ASREngine(
                model_id=model_id, compute_type=compute_type, trace_sink=trace_sink
            )
        elif trace_sink is not None and trace_sink is not _asr_engine._trace_sink:
            logger.warning("get_asr_engine: different sink passed after construction; ignoring")
        return _asr_engine


def _reset_singletons_for_tests() -> None:
    """Tear down singletons. Tests only. audio.md §3.2 "Unload. Never." exemption."""

    global _tts_engine, _asr_engine
    with _tts_lock:
        _tts_engine = None
    with _asr_lock:
        _asr_engine = None


__all__ = [
    "AudioDecodeError",
    "AudioError",
    "AudioTooLongError",
    "AudioTrace",
    "ASREngine",
    "LanguageCode",
    "ModelLoadError",
    "TTSEngine",
    "TTSOutOfMemoryError",
    "TranscriptResult",
    "TraceSink",
    "UnsupportedLanguageError",
    "UnsupportedVoicePackError",
    "VOICE_PACKS",
    "VoicePack",
    "VoicePackMapping",
    "get_asr_engine",
    "get_tts_engine",
]
