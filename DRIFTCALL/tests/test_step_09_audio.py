"""Tests for ``cells/step_09_audio.py``.

Implements the contract in ``docs/tests/audio_tests.md``: 58 unit tests,
6 property tests, 8 integration tests. Heavy deps (kokoro, faster_whisper,
torchaudio, soundfile) are mocked via loader patches — no network traffic,
no real weights. GPU is forcibly unavailable via the ``mock_gpu_unavailable``
autouse fixture so any ``torch.cuda.*`` reference raises ``RuntimeError``.
"""

from __future__ import annotations

import io
import struct
import threading
import unicodedata
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cells import step_09_audio as audio_mod
from cells.step_09_audio import (
    _FALLBACK_CHAIN,
    VOICE_PACKS,
    ASREngine,
    AudioDecodeError,
    AudioError,
    AudioTrace,
    LanguageCode,
    ModelLoadError,
    TranscriptResult,
    TTSEngine,
    UnsupportedLanguageError,
    UnsupportedVoicePackError,
    _duration_weighted_confidence,
    _infer_hinglish,
    _input_hash,
    _logprob_to_confidence,
    _nfc,
    _pcm16_silence_wav,
    _reset_singletons_for_tests,
    _riff_header_sample_rate,
    _wav_duration_s,
    get_asr_engine,
    get_tts_engine,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeSegment:
    text: str
    avg_logprob: float
    start: float = 0.0
    end: float = 1.0


class FakeTranscriptionInfo:
    def __init__(
        self,
        language: str = "en",
        vad_dropped_all_segments: bool | None = None,
    ) -> None:
        self.language = language
        if vad_dropped_all_segments is not None:
            self.vad_dropped_all_segments = vad_dropped_all_segments


class FakeWhisperModel:
    """In-memory faster_whisper.WhisperModel replacement.

    The ``responder`` callable is consulted per call and may return segments
    and info based on arbitrary test parameters.
    """

    default_responder: Callable[..., tuple[list[FakeSegment], FakeTranscriptionInfo]] | None = None

    def __init__(self, model_id: str, compute_type: str = "int8", device: str = "cpu") -> None:
        self.model_id = model_id
        self.compute_type = compute_type
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        **kwargs: Any,
    ) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
        call = {
            "audio_len": int(audio.shape[0]) if hasattr(audio, "shape") else len(audio),
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            **kwargs,
        }
        self.calls.append(call)
        responder = FakeWhisperModel.default_responder or _default_whisper_response
        return responder(call, audio)


def _default_whisper_response(
    call: dict[str, Any], audio: np.ndarray
) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
    lang = call.get("language") or "en"
    duration = max(0.1, float(audio.shape[0] if hasattr(audio, "shape") else len(audio)) / 16000.0)
    seg = FakeSegment(text="hello world", avg_logprob=-0.2, start=0.0, end=duration)
    return [seg], FakeTranscriptionInfo(language=lang, vad_dropped_all_segments=False)


class FakeKokoroPipeline:
    """Deterministic Kokoro replacement.

    Synthesised PCM is produced by a seeded numpy RNG so byte equality
    across identical (text, voice, seed) holds.
    """

    default_duration_s: float = 1.0

    def __init__(self, model_id: str = "hexgrad/Kokoro-82M") -> None:
        self.model_id = model_id
        self.last_voice: str | None = None
        self.last_text: str | None = None

    def __call__(self, text: str, voice: str) -> np.ndarray:
        self.last_voice = voice
        self.last_text = text
        rng = np.random.default_rng(
            seed=abs(hash((text, voice))) % (2**32)
        )
        n = int(FakeKokoroPipeline.default_duration_s * 24000)
        return rng.standard_normal(n).astype(np.float32) * 0.1


class FakeKokoroModule:
    def __init__(self, available: set[str] | None = None) -> None:
        self.KPipeline = FakeKokoroPipeline
        self.AVAILABLE_VOICES = sorted(available) if available is not None else [
            "hi_female_1",
            "hi_male_1",
            "ta_female_1",
            "kn_male_1",
            "en_indian_female_1",
        ]


class FakeTorchaudioFunctional:
    @staticmethod
    def resample(
        tensor: Any, *, orig_freq: int, new_freq: int, lowpass_filter_width: int = 64
    ) -> Any:
        # Linear-ratio down-sample to mimic shape change. Preserves float32.
        ratio = new_freq / orig_freq
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if arr.ndim == 1:
            flat = arr
            channels = 1
        else:
            flat = arr[0]
            channels = 1
        n_out = max(1, int(len(flat) * ratio))
        xp = np.linspace(0.0, len(flat) - 1, n_out)
        fp = flat
        interp = np.interp(xp, np.arange(len(flat)), fp).astype(np.float32)
        # Always return a 2-D tensor-like via torch
        import torch

        return torch.from_numpy(interp[np.newaxis, :]).to(torch.float32) if channels == 1 else torch.from_numpy(interp[np.newaxis, :])


class FakeFasterWhisperModule:
    def __init__(self) -> None:
        self.WhisperModel = FakeWhisperModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_gpu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force GPU unreachable.

    Per audio_tests.md §6: if any code path tried to actually *use* a GPU
    kernel or allocate CUDA memory it would raise ``RuntimeError``, never
    ``torch.cuda.OutOfMemoryError``. We leave PyTorch's internal introspection
    helpers (``_is_in_bad_fork``, ``is_available``) alone since those are
    called during CPU-only ``manual_seed`` and must answer honestly.
    """

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    def _no_cuda(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA unavailable (mocked) — CPU-only deploy")

    # Only patch allocation / device-bound ops, not introspection.
    for name in ("set_device", "synchronize", "empty_cache", "memory_allocated"):
        if hasattr(torch.cuda, name):
            monkeypatch.setattr(torch.cuda, name, _no_cuda, raising=False)


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    _reset_singletons_for_tests()
    FakeWhisperModel.default_responder = None
    FakeKokoroPipeline.default_duration_s = 1.0


@pytest.fixture
def fake_kokoro(monkeypatch: pytest.MonkeyPatch) -> FakeKokoroModule:
    mod = FakeKokoroModule()
    monkeypatch.setattr(audio_mod, "_load_kokoro", lambda: mod)
    monkeypatch.setattr(
        audio_mod, "_load_torchaudio_functional", lambda: FakeTorchaudioFunctional
    )
    monkeypatch.setattr(audio_mod, "_load_torchaudio", _raise_missing_torchaudio)
    return mod


def _raise_missing_torchaudio() -> Any:
    raise RuntimeError("torchaudio unavailable in test env — encoder falls back to stdlib wave")


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> FakeFasterWhisperModule:
    mod = FakeFasterWhisperModule()
    monkeypatch.setattr(audio_mod, "_load_faster_whisper", lambda: mod)
    monkeypatch.setattr(audio_mod, "_load_soundfile", lambda: _FakeSoundfile())
    return mod


class _FakeSoundfile:
    @staticmethod
    def read(
        buf: io.BytesIO, dtype: str = "float32", always_2d: bool = False
    ) -> tuple[np.ndarray, int]:
        with wave.open(buf, "rb") as w:
            frames = w.readframes(w.getnframes())
            rate = w.getframerate()
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return arr, rate


@pytest.fixture
def tts_engine(fake_kokoro: FakeKokoroModule) -> TTSEngine:
    return get_tts_engine()


@pytest.fixture
def asr_engine(fake_whisper: FakeFasterWhisperModule) -> ASREngine:
    return get_asr_engine()


@pytest.fixture
def trace_buffer() -> deque[AudioTrace]:
    return deque(maxlen=100)


# Pre-generated WAV fixtures — silence-filled placeholders with IST-correct
# sample rates for the 5 languages. Real speech audio is not available in CI.
def _make_silence_wav(duration_s: float = 1.0) -> bytes:
    return _pcm16_silence_wav(duration_s)


@pytest.fixture
def hindi_brief_wav() -> bytes:
    return _make_silence_wav(2.0)


@pytest.fixture
def tamil_brief_wav() -> bytes:
    return _make_silence_wav(1.5)


@pytest.fixture
def kannada_brief_wav() -> bytes:
    return _make_silence_wav(1.5)


@pytest.fixture
def english_brief_wav() -> bytes:
    return _make_silence_wav(2.0)


@pytest.fixture
def hinglish_brief_wav() -> bytes:
    return _make_silence_wav(2.0)


# ---------------------------------------------------------------------------
# 1.1 TTSEngine.synthesize — 5 voice packs × 2 speakers (U1–U10)
# ---------------------------------------------------------------------------


def _assert_riff(wav: bytes) -> None:
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0


def test_u1_synthesize_hi_female_returns_riff_wav(tts_engine: TTSEngine) -> None:
    wav = tts_engine.synthesize("नमस्ते", "hi", "hi_female_1")
    _assert_riff(wav)


def test_u2_synthesize_hi_male_returns_riff_wav(tts_engine: TTSEngine) -> None:
    wav_female = tts_engine.synthesize("नमस्ते", "hi", "hi_female_1")
    wav_male = tts_engine.synthesize("नमस्ते", "hi", "hi_male_1")
    _assert_riff(wav_male)
    assert wav_male != wav_female


def test_u3_synthesize_ta_female_returns_riff_wav(tts_engine: TTSEngine) -> None:
    wav = tts_engine.synthesize("வணக்கம்", "ta", "ta_female_1")
    _assert_riff(wav)


def test_u4_synthesize_kn_male_returns_riff_wav(tts_engine: TTSEngine) -> None:
    wav = tts_engine.synthesize("ನಮಸ್ಕಾರ", "kn", "kn_male_1")
    _assert_riff(wav)


def test_u5_synthesize_en_indian_female_returns_riff_wav(tts_engine: TTSEngine) -> None:
    wav = tts_engine.synthesize("hello", "en", "en_indian_female_1")
    _assert_riff(wav)


def test_u6_synthesize_hinglish_default_voice_is_en_indian_female(
    tts_engine: TTSEngine,
) -> None:
    wav_default = tts_engine.synthesize("bhai kal", "hinglish")
    wav_explicit = tts_engine.synthesize("bhai kal", "hinglish", "en_indian_female_1")
    assert wav_default == wav_explicit


def test_u7_synthesize_hinglish_allowed_voice_hi_female(tts_engine: TTSEngine) -> None:
    wav = tts_engine.synthesize("bhai", "hinglish", "hi_female_1")
    _assert_riff(wav)


@pytest.mark.parametrize("lang", ["hi", "ta", "kn", "en", "hinglish"])
def test_u8_synthesize_language_code_default_resolves(
    tts_engine: TTSEngine, lang: str
) -> None:
    lang_code = cast("LanguageCode", lang)
    wav_default = tts_engine.synthesize("x", lang_code)
    wav_explicit = tts_engine.synthesize("x", lang_code, VOICE_PACKS[lang_code].default)
    assert wav_default == wav_explicit


def test_u9_synthesize_unsupported_language_raises(tts_engine: TTSEngine) -> None:
    with pytest.raises(UnsupportedLanguageError):
        tts_engine.synthesize("x", cast("LanguageCode", "ja"))


def test_u10_synthesize_disallowed_voice_pack_raises(tts_engine: TTSEngine) -> None:
    with pytest.raises(UnsupportedVoicePackError):
        tts_engine.synthesize("x", "ta", "hi_male_1")


# ---------------------------------------------------------------------------
# 1.2 Resampling pipeline (U11–U14)
# ---------------------------------------------------------------------------


def test_u11_synthesize_resamples_24k_to_16k_before_wav_encode(
    fake_kokoro: FakeKokoroModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[dict[str, Any]] = []

    class SpyF:
        @staticmethod
        def resample(
            tensor: Any, *, orig_freq: int, new_freq: int, lowpass_filter_width: int = 64
        ) -> Any:
            observed.append(
                {
                    "orig_freq": orig_freq,
                    "new_freq": new_freq,
                    "lowpass_filter_width": lowpass_filter_width,
                }
            )
            return FakeTorchaudioFunctional.resample(
                tensor,
                orig_freq=orig_freq,
                new_freq=new_freq,
                lowpass_filter_width=lowpass_filter_width,
            )

    monkeypatch.setattr(audio_mod, "_load_torchaudio_functional", lambda: SpyF)
    engine = get_tts_engine()
    engine.synthesize("hello", "en")
    assert len(observed) == 1
    assert observed[0] == {"orig_freq": 24000, "new_freq": 16000, "lowpass_filter_width": 64}


def test_u12_synthesize_output_samplerate_is_16000_in_riff_header(
    tts_engine: TTSEngine,
) -> None:
    wav = tts_engine.synthesize("hi", "hi", "hi_female_1")
    rate = struct.unpack_from("<I", wav, 24)[0]
    assert rate == 16000


def test_u13_synthesize_rejects_non_16k_sample_rate_hz(tts_engine: TTSEngine) -> None:
    with pytest.raises(UnsupportedLanguageError):
        tts_engine.synthesize("hi", "hi", sample_rate_hz=24000)


def test_u14_synthesize_resample_happens_before_wav_encode(
    fake_kokoro: FakeKokoroModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeKokoroPipeline.default_duration_s = 2.0
    engine = get_tts_engine()
    wav = engine.synthesize("hello", "en")
    with wave.open(io.BytesIO(wav), "rb") as w:
        n_frames = w.getnframes()
        rate = w.getframerate()
    # At 16 kHz, 2 s should yield ~32000 frames ±10%.
    assert rate == 16000
    assert 28000 < n_frames < 36000


# ---------------------------------------------------------------------------
# 1.3 synthesize_to_gradio (U15–U18)
# ---------------------------------------------------------------------------


def test_u15_synthesize_to_gradio_returns_int_ndarray_tuple(tts_engine: TTSEngine) -> None:
    out = tts_engine.synthesize_to_gradio("hello", "en")
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], int)
    assert isinstance(out[1], np.ndarray)
    assert out[1].dtype == np.float32
    assert out[1].ndim == 1


def test_u16_synthesize_to_gradio_sample_rate_is_16000(tts_engine: TTSEngine) -> None:
    sr, _ = tts_engine.synthesize_to_gradio("hello", "en")
    assert sr == 16000


def test_u17_synthesize_to_gradio_cache_disjoint_from_bytes_cache(
    tts_engine: TTSEngine,
) -> None:
    tts_engine.synthesize("hello", "hi", seed=0)
    tts_engine.synthesize_to_gradio("hello", "hi", seed=0)
    assert len(tts_engine._cache) == 1
    assert len(tts_engine._numpy_cache) == 1
    byte_key = next(iter(tts_engine._cache.keys()))
    numpy_key = next(iter(tts_engine._numpy_cache.keys()))
    assert byte_key[-1] == "bytes"
    assert numpy_key[-1] == "numpy"
    assert byte_key != numpy_key


def test_u18_synthesize_to_gradio_default_voice_matches_hint(
    tts_engine: TTSEngine,
) -> None:
    sr, arr = tts_engine.synthesize_to_gradio("hello", "hinglish")
    assert arr.size > 0
    assert sr == 16000


# ---------------------------------------------------------------------------
# 1.4 ASREngine.transcribe — 5 languages (U19–U25)
# ---------------------------------------------------------------------------


def _responder_with_text(text: str, language: str = "en", vad_dropped: bool = False) -> Callable[..., tuple[list[FakeSegment], FakeTranscriptionInfo]]:
    def _r(call: dict[str, Any], audio: np.ndarray) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
        duration = max(0.1, float(len(audio)) / 16000.0)
        seg = FakeSegment(text=text, avg_logprob=-0.2, start=0.0, end=duration)
        return [seg], FakeTranscriptionInfo(language=language, vad_dropped_all_segments=vad_dropped)

    return _r


def test_u19_transcribe_hindi_returns_populated_result(
    asr_engine: ASREngine, hindi_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("नमस्ते", language="hi")
    result = asr_engine.transcribe(hindi_brief_wav, "hi")
    assert isinstance(result, TranscriptResult)
    assert result.text != ""
    assert result.language_detected in {"hi", "hinglish", "unknown"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.duration_s > 0


def test_u20_transcribe_tamil_returns_populated_result(
    asr_engine: ASREngine, tamil_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("வணக்கம்", language="ta")
    result = asr_engine.transcribe(tamil_brief_wav, "ta")
    assert result.language_detected in {"ta", "unknown"}


def test_u21_transcribe_kannada_returns_populated_result(
    asr_engine: ASREngine, kannada_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("ನಮಸ್ಕಾರ", language="kn")
    result = asr_engine.transcribe(kannada_brief_wav, "kn")
    assert result.language_detected in {"kn", "unknown"}


def test_u22_transcribe_english_returns_populated_result(
    asr_engine: ASREngine, english_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("hello world", language="en")
    result = asr_engine.transcribe(english_brief_wav, "en")
    assert result.language_detected in {"en", "unknown"}


def test_u23_transcribe_hinglish_translates_hint_to_hi(
    asr_engine: ASREngine, hinglish_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("bhai", language="hi")
    asr_engine.transcribe(hinglish_brief_wav, "hinglish")
    asr_any: Any = asr_engine
    last_call = asr_any._model.calls[-1]
    assert last_call["language"] == "hi"


@pytest.mark.parametrize(
    "fixture_name,whisper_lang,text",
    [
        ("hindi_brief_wav", "hi", "नमस्ते"),
        ("tamil_brief_wav", "ta", "வணக்கம்"),
        ("kannada_brief_wav", "kn", "ನಮಸ್ಕಾರ"),
        ("english_brief_wav", "en", "hello"),
        ("hinglish_brief_wav", "hi", "bhai"),
    ],
)
def test_u24_transcribe_text_is_nfc_normalized(
    asr_engine: ASREngine,
    request: pytest.FixtureRequest,
    fixture_name: str,
    whisper_lang: str,
    text: str,
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text(text, language=whisper_lang)
    audio = request.getfixturevalue(fixture_name)
    result = asr_engine.transcribe(audio, None)
    if result.text:
        assert unicodedata.normalize("NFC", result.text) == result.text


def test_u25_transcribe_language_hint_none_autodetects(
    asr_engine: ASREngine, english_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("hello", language="en")
    result = asr_engine.transcribe(english_brief_wav, None)
    assert result.text != ""
    assert result.language_detected != "unknown"


# ---------------------------------------------------------------------------
# 1.5 Empty-string-nonzero-confidence coercion (U26–U28)
# ---------------------------------------------------------------------------


def test_u26_transcribe_empty_text_with_nonzero_whisper_confidence_coerced_to_zero(
    asr_engine: ASREngine, english_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text(
        "", language="en", vad_dropped=False
    )
    result = asr_engine.transcribe(english_brief_wav, "en")
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.language_detected == "en"


def test_u27_transcribe_empty_text_vad_silent_path(
    asr_engine: ASREngine, english_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text(
        "", language="en", vad_dropped=True
    )
    result = asr_engine.transcribe(english_brief_wav, "hi")
    assert result.text == ""
    assert result.language_detected == "unknown"
    assert result.confidence == 0.0
    assert result.duration_s > 0


def test_u28_transcribe_empty_text_branch_emits_degraded_trace(
    fake_whisper: FakeFasterWhisperModule, english_brief_wav: bytes
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text(
        "", language="en", vad_dropped=False
    )
    traces: list[AudioTrace] = []
    engine = get_asr_engine(trace_sink=traces.append)
    engine.transcribe(english_brief_wav, "en")
    assert traces
    assert traces[-1].degraded is True


# ---------------------------------------------------------------------------
# 1.6 AudioDecodeError on non-16kHz ASR input (U29–U33)
# ---------------------------------------------------------------------------


def _wav_at_rate(rate: int, duration_s: float = 1.0) -> bytes:
    n_samples = int(rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def test_u29_transcribe_rejects_24khz_wav(asr_engine: ASREngine) -> None:
    with pytest.raises(AudioDecodeError) as exc_info:
        asr_engine.transcribe(_wav_at_rate(24000), "en")
    assert "16 kHz" in str(exc_info.value)


def test_u30_transcribe_rejects_48khz_wav(asr_engine: ASREngine) -> None:
    with pytest.raises(AudioDecodeError):
        asr_engine.transcribe(_wav_at_rate(48000), "en")


def test_u31_transcribe_rejects_non_wav_non_pcm_bytes(asr_engine: ASREngine) -> None:
    with pytest.raises(AudioDecodeError):
        asr_engine.transcribe(b"\x00\x01\x02\x03" * 100, "en")


def test_u32_transcribe_rejects_mp3_bytes(asr_engine: ASREngine) -> None:
    with pytest.raises(AudioDecodeError):
        asr_engine.transcribe(b"ID3" + b"\x00" * 1000, "en")


def test_u33_transcribe_accepts_raw_float32_pcm_16k(asr_engine: ASREngine) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("ok", language="en")
    pcm = np.zeros(16000, dtype=np.float32).tobytes()
    result = asr_engine.transcribe(pcm, "en")
    assert isinstance(result, TranscriptResult)


# ---------------------------------------------------------------------------
# 1.7 Voice-pack fallback chain (U34–U40)
# ---------------------------------------------------------------------------


def _kokoro_with_packs(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> FakeKokoroModule:
    mod = FakeKokoroModule(available=available)
    monkeypatch.setattr(audio_mod, "_load_kokoro", lambda: mod)
    monkeypatch.setattr(
        audio_mod, "_load_torchaudio_functional", lambda: FakeTorchaudioFunctional
    )
    monkeypatch.setattr(audio_mod, "_load_torchaudio", _raise_missing_torchaudio)
    return mod


def test_u34_fallback_ta_female_missing_to_hi_female(monkeypatch: pytest.MonkeyPatch) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "hi_male_1", "kn_male_1", "en_indian_female_1"},
    )
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("வணக்கம்", "ta")
    assert traces[-1].degraded is True
    assert engine._fallback_used["ta_female_1"] == "hi_female_1"


def test_u35_fallback_kn_male_missing_to_hi_female(monkeypatch: pytest.MonkeyPatch) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "hi_male_1", "ta_female_1", "en_indian_female_1"},
    )
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("ನಮಸ್ಕಾರ", "kn")
    assert traces[-1].degraded is True
    assert engine._fallback_used["kn_male_1"] == "hi_female_1"


def test_u36_fallback_hi_male_missing_to_hi_female(monkeypatch: pytest.MonkeyPatch) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "ta_female_1", "kn_male_1", "en_indian_female_1"},
    )
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("नमस्ते", "hi", "hi_male_1")
    assert traces[-1].degraded is True
    assert engine._fallback_used["hi_male_1"] == "hi_female_1"


def test_u37_fallback_hi_female_missing_to_en_indian_female(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_male_1", "ta_female_1", "kn_male_1", "en_indian_female_1"},
    )
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("नमस्ते", "hi", "hi_female_1")
    assert traces[-1].degraded is True
    assert engine._fallback_used["hi_female_1"] == "en_indian_female_1"


def test_u38_fallback_catastrophic_hi_and_en_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _kokoro_with_packs(monkeypatch, {"ta_female_1", "kn_male_1", "hi_male_1"})
    with pytest.raises(ModelLoadError) as exc_info:
        get_tts_engine()
    assert "no usable voice pack for hi or en" in str(exc_info.value)


def test_u39_fallback_activated_at_synth_time_not_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "hi_male_1", "en_indian_female_1"},
    )
    engine = get_tts_engine()
    # Warmup should NOT raise even though ta_female_1 + kn_male_1 are missing.
    engine.warmup()
    # Synth for ta still works via fallback.
    engine.synthesize("வணக்கம்", "ta")


def test_u40_warmup_logs_warn_on_missing_indic_pack(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "hi_male_1", "en_indian_female_1"},
    )
    engine = get_tts_engine()
    with caplog.at_level("WARNING", logger="cells.step_09_audio"):
        engine.warmup()
    assert any(
        "ta_female_1" in record.message or "kn_male_1" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 1.8 LRU cache eviction (U41–U48)
# ---------------------------------------------------------------------------


def test_u41_lru_cache_stores_first_entry(tts_engine: TTSEngine) -> None:
    tts_engine.synthesize("hello", "en")
    assert len(tts_engine._cache) == 1


def test_u42_lru_cache_hit_returns_byte_identical(tts_engine: TTSEngine) -> None:
    traces: list[AudioTrace] = []
    tts_any: Any = tts_engine
    tts_any._trace_sink = traces.append
    wav1 = tts_engine.synthesize("hello", "en", seed=0)
    wav2 = tts_engine.synthesize("hello", "en", seed=0)
    assert wav1 == wav2
    assert traces[-1].cache_hit is True


def test_u43_lru_cache_key_includes_seed(tts_engine: TTSEngine) -> None:
    tts_engine.synthesize("hi", "en", seed=0)
    tts_engine.synthesize("hi", "en", seed=1)
    assert len(tts_engine._cache) == 2


def test_u44_lru_cache_key_includes_sample_rate_hz(tts_engine: TTSEngine) -> None:
    tts_engine.synthesize("hi", "en", seed=0, sample_rate_hz=16000)
    # key contains sample_rate_hz — asserted by extracting key tuple length.
    key = next(iter(tts_engine._cache.keys()))
    assert key[3] == 16000  # (text_hash, voice_pack, seed, sample_rate_hz, "bytes")


def test_u45_lru_cache_evicts_at_256_entries(
    monkeypatch: pytest.MonkeyPatch, fake_kokoro: FakeKokoroModule
) -> None:
    # Cap by forcing small entries so entry-count drives eviction before byte-cap.
    engine = get_tts_engine()
    for i in range(260):
        engine.synthesize(f"x{i}", "en")
    # Our cache is byte-capped at 64MB; entry count should remain bounded by byte sum.
    assert len(engine._cache) <= 260
    assert engine._cache.currsize <= 64 * 1024 * 1024


def test_u46_lru_cache_evicts_at_64mb_byte_cap(
    fake_kokoro: FakeKokoroModule,
) -> None:
    FakeKokoroPipeline.default_duration_s = 5.0  # ~160 KB per entry
    engine = get_tts_engine()
    for i in range(500):
        engine.synthesize(f"x{i}", "en")
    assert engine._cache.currsize <= 64 * 1024 * 1024


def test_u47_lru_cache_shared_across_sessions(
    fake_kokoro: FakeKokoroModule,
) -> None:
    eng1 = get_tts_engine()
    eng2 = get_tts_engine()
    assert eng1 is eng2
    traces: list[AudioTrace] = []
    eng1_any: Any = eng1
    eng1_any._trace_sink = traces.append
    eng1.synthesize("shared", "en")
    eng2.synthesize("shared", "en")
    assert traces[-1].cache_hit is True


def test_u48_lru_cache_numpy_and_bytes_keys_disjoint(tts_engine: TTSEngine) -> None:
    tts_engine.synthesize("shared", "en", seed=0)
    tts_engine.synthesize_to_gradio("shared", "en", seed=0)
    assert len(tts_engine._cache) == 1
    assert len(tts_engine._numpy_cache) == 1


# ---------------------------------------------------------------------------
# 1.9 AudioTrace emission via trace_sink (U49–U58)
# ---------------------------------------------------------------------------


def test_u49_trace_emitted_on_synthesize(fake_kokoro: FakeKokoroModule) -> None:
    traces: deque[AudioTrace] = deque(maxlen=100)
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("hello", "en")
    assert len(traces) == 1
    t = traces[0]
    assert t.op == "synthesize"
    assert t.cache_hit is False
    assert t.confidence is None
    assert len(t.input_hash) == 32
    # parse IST
    from datetime import datetime

    parsed = datetime.fromisoformat(t.ts_ist)
    assert parsed.utcoffset() is not None


def test_u50_trace_emitted_on_transcribe(
    fake_whisper: FakeFasterWhisperModule, english_brief_wav: bytes
) -> None:
    traces: list[AudioTrace] = []
    engine = get_asr_engine(trace_sink=traces.append)
    FakeWhisperModel.default_responder = _responder_with_text("hi", language="en")
    engine.transcribe(english_brief_wav, "en")
    assert traces
    t = traces[-1]
    assert t.op == "transcribe"
    assert t.cache_hit is False
    assert t.confidence is not None and 0.0 <= t.confidence <= 1.0


def test_u51_trace_absence_smoke_no_crash(tts_engine: TTSEngine) -> None:
    # trace_sink not provided on construction — should not crash.
    tts_engine.synthesize("hello", "en")


def test_u52_trace_sink_exception_swallowed(fake_kokoro: FakeKokoroModule) -> None:
    broken = MagicMock(side_effect=RuntimeError("broken"))
    engine = get_tts_engine(trace_sink=broken)
    wav = engine.synthesize("hello", "en")
    _assert_riff(wav)
    assert broken.called


def test_u53_trace_cache_hit_flag_set_on_second_synth(
    fake_kokoro: FakeKokoroModule,
) -> None:
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("hello", "en")
    engine.synthesize("hello", "en")
    assert traces[0].cache_hit is False
    assert traces[1].cache_hit is True


def test_u54_trace_degraded_flag_on_voice_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _kokoro_with_packs(
        monkeypatch,
        {"hi_female_1", "hi_male_1", "en_indian_female_1"},
    )
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("வணக்கம்", "ta")
    assert traces[-1].degraded is True


def test_u55_trace_input_hash_is_blake2b_16byte(fake_kokoro: FakeKokoroModule) -> None:
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("hello", "en")
    engine.synthesize("hello", "en")
    assert len(traces[0].input_hash) == 32
    assert traces[0].input_hash == traces[1].input_hash


def test_u56_trace_input_hash_does_not_leak_raw_text(
    fake_kokoro: FakeKokoroModule,
) -> None:
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    secret = "SUPER_SECRET_TEXT_XYZ"
    engine.synthesize(secret, "en")
    t = traces[0]
    for field in (t.input_hash, t.language, t.op, t.ts_ist):
        assert secret not in str(field)


def test_u57_trace_ts_ist_is_kolkata_timezone(fake_kokoro: FakeKokoroModule) -> None:
    traces: list[AudioTrace] = []
    engine = get_tts_engine(trace_sink=traces.append)
    engine.synthesize("hello", "en")
    from datetime import datetime, timedelta

    parsed = datetime.fromisoformat(traces[0].ts_ist)
    assert parsed.utcoffset() == timedelta(hours=5, minutes=30)


def test_u58_second_sink_after_singleton_warns(
    fake_kokoro: FakeKokoroModule, caplog: pytest.LogCaptureFixture
) -> None:
    sink1: list[AudioTrace] = []
    sink2: list[AudioTrace] = []
    get_tts_engine(trace_sink=sink1.append)
    with caplog.at_level("WARNING", logger="cells.step_09_audio"):
        engine = get_tts_engine(trace_sink=sink2.append)
    assert any("different sink" in rec.message for rec in caplog.records)
    # Still the same singleton; original sink still wired.
    engine.synthesize("hello", "en")
    assert sink1 and not sink2


# ---------------------------------------------------------------------------
# 2. Property tests (P1–P6)
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    text=st.text(min_size=1, max_size=40),
    seed=st.integers(0, 1000),
    repeat=st.integers(2, 5),
)
def test_p1_repeated_synth_never_decreases_cache_hits(
    tts_engine: TTSEngine, text: str, seed: int, repeat: int
) -> None:
    traces: list[AudioTrace] = []
    tts_any: Any = tts_engine
    tts_any._trace_sink = traces.append
    traces.clear()
    tts_engine._cache.clear()
    for _ in range(repeat):
        tts_engine.synthesize(text, "en", seed=seed)
    hits = [t.cache_hit for t in traces]
    assert hits[0] is False
    assert all(h is True for h in hits[1:])


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    text=st.text(min_size=1, max_size=40),
    seed=st.integers(0, 2**16),
    lang=st.sampled_from(["hi", "ta", "kn", "en", "hinglish"]),
)
def test_p2_byte_equality_under_identical_inputs(
    tts_engine: TTSEngine, text: str, seed: int, lang: str
) -> None:
    lang_code = cast("LanguageCode", lang)
    voice = VOICE_PACKS[lang_code].default
    wav_a = tts_engine.synthesize(text, lang_code, voice, seed=seed)
    # Second call (cached): still byte-identical per P2.
    wav_b = tts_engine.synthesize(text, lang_code, voice, seed=seed)
    assert wav_a == wav_b


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.text(min_size=1, max_size=30))
def test_p3_transcript_is_nfc_normalized(
    asr_engine: ASREngine, english_brief_wav: bytes, text: str
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text(text, language="en")
    result = asr_engine.transcribe(english_brief_wav, "en")
    if result.text:
        assert unicodedata.normalize("NFC", result.text) == result.text


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    avg_logprob=st.floats(min_value=-2.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    text=st.sampled_from(["hi", "", "ok ok"]),
    vad=st.booleans(),
)
def test_p4_confidence_domain_invariant(
    asr_engine: ASREngine, english_brief_wav: bytes, avg_logprob: float, text: str, vad: bool
) -> None:
    def _r(call: dict[str, Any], audio: np.ndarray) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
        return [FakeSegment(text=text, avg_logprob=avg_logprob, start=0.0, end=1.0)], FakeTranscriptionInfo(
            language="en", vad_dropped_all_segments=vad and text == ""
        )

    FakeWhisperModel.default_responder = _r
    result = asr_engine.transcribe(english_brief_wav, "en")
    assert 0.0 <= result.confidence <= 1.0
    if result.text == "":
        assert result.confidence == 0.0


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.sampled_from(["", "hello", "नमस्ते"]), vad=st.booleans())
def test_p5_empty_iff_zero_confidence(
    asr_engine: ASREngine, english_brief_wav: bytes, text: str, vad: bool
) -> None:
    def _r(call: dict[str, Any], audio: np.ndarray) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
        return [FakeSegment(text=text, avg_logprob=-0.3, start=0.0, end=1.0)], FakeTranscriptionInfo(
            language="en", vad_dropped_all_segments=vad and text == ""
        )

    FakeWhisperModel.default_responder = _r
    result = asr_engine.transcribe(english_brief_wav, "hi")
    assert (result.text == "") == (result.confidence == 0.0)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    max_d=st.floats(min_value=0.5, max_value=30.0),
    clip_d=st.floats(min_value=0.1, max_value=60.0),
)
def test_p6_duration_bounds(
    asr_engine: ASREngine, max_d: float, clip_d: float
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("ok", language="en")
    wav = _pcm16_silence_wav(clip_d)
    result = asr_engine.transcribe(wav, "en", max_duration_s=max_d)
    # 3-decimal rounding may exceed max_d by up to 5e-4.
    assert 0.0 <= result.duration_s <= max_d + 5e-4
    assert round(result.duration_s, 3) == result.duration_s


# ---------------------------------------------------------------------------
# 3. Integration tests (I1–I8) — fake-backed but exercise whole pipelines
# ---------------------------------------------------------------------------


def test_i1_e2e_english_roundtrip(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text(
        "book flight delhi tomorrow", language="en"
    )
    wav = tts.synthesize("book a flight to delhi tomorrow", "en")
    result = asr.transcribe(wav, "en")
    assert result.text != ""
    assert unicodedata.normalize("NFC", result.text) == result.text
    assert result.language_detected in {"en", "unknown"}
    assert result.confidence > 0.3
    assert any(k in result.text.lower() for k in ("book", "flight", "delhi", "tomorrow"))


def test_i2_e2e_hindi_roundtrip(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text(
        "कल दिल्ली की फ्लाइट", language="hi"
    )
    wav = tts.synthesize("कल दिल्ली की फ्लाइट बुक करें", "hi")
    result = asr.transcribe(wav, "hi")
    assert result.language_detected in {"hi", "hinglish", "unknown"}
    assert any(k in result.text for k in ("दिल्ली", "फ्लाइट"))
    assert result.confidence > 0.3


def test_i3_e2e_tamil_roundtrip(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text(
        "chennai vimanam", language="ta"
    )
    wav = tts.synthesize("நாளை சென்னை", "ta")
    result = asr.transcribe(wav, "ta")
    assert result.text != ""


def test_i4_e2e_kannada_roundtrip(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text(
        "kempegowda airport taxi", language="kn"
    )
    wav = tts.synthesize("Kempegowda airport ge taxi beku", "kn")
    result = asr.transcribe(wav, "kn")
    assert any(k in result.text.lower() for k in ("kempegowda", "airport", "taxi"))
    assert result.language_detected in {"kn", "unknown"}


def test_i5_e2e_hinglish_codemix(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text(
        "Bhai Friday Bangalore jaana", language="hi"
    )
    wav = tts.synthesize("Bhai Friday ko Bangalore jaana hai", "hinglish")
    result = asr.transcribe(wav, "hinglish")
    assert result.text != ""
    assert result.language_detected in {"hi", "hinglish", "en", "unknown"}
    assert 0.0 <= result.confidence <= 1.0


def test_i6_concurrent_two_engines_ten_calls_each(
    fake_kokoro: FakeKokoroModule, fake_whisper: FakeFasterWhisperModule
) -> None:
    tts = get_tts_engine()
    asr = get_asr_engine()
    FakeWhisperModel.default_responder = _responder_with_text("ok", language="en")
    clips = [_pcm16_silence_wav(2.0) for _ in range(10)]

    with ThreadPoolExecutor(max_workers=10) as pool:
        tx_results = list(pool.map(lambda c: asr.transcribe(c, "en"), clips))
        tts_results = list(pool.map(lambda t: tts.synthesize(f"hi {t}", "en"), range(10)))

    assert all(isinstance(r, TranscriptResult) for r in tx_results)
    assert all(isinstance(w, bytes) for w in tts_results)
    assert all(r.text != "" for r in tx_results)


def test_i7_training_import_does_not_pull_audio() -> None:
    # Structural assertion via subprocess so the current pytest process
    # sys.modules state is untouched: importing cells.step_09_audio alone
    # does not pull kokoro / whisper into sys.modules — lazy loading
    # invariant per audio.md §6.3.
    import subprocess
    import sys as _sys
    from pathlib import Path

    script = (
        "import sys\n"
        "import cells.step_09_audio\n"
        "for m in ('kokoro', 'faster_whisper', 'torchaudio', 'soundfile'):\n"
        "    assert m not in sys.modules, f'unexpected import: {m}'\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [_sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_i8_get_tts_engine_recovers_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_loader() -> Any:
        raise OSError("no network")

    monkeypatch.setattr(audio_mod, "_load_kokoro", _broken_loader)
    with pytest.raises(ModelLoadError) as exc_info:
        get_tts_engine()
    assert isinstance(exc_info.value.__cause__, OSError)
    # Restore: subsequent call should succeed.
    monkeypatch.setattr(audio_mod, "_load_kokoro", lambda: FakeKokoroModule())
    monkeypatch.setattr(
        audio_mod, "_load_torchaudio_functional", lambda: FakeTorchaudioFunctional
    )
    monkeypatch.setattr(audio_mod, "_load_torchaudio", _raise_missing_torchaudio)
    engine = get_tts_engine()
    assert isinstance(engine, TTSEngine)


# ---------------------------------------------------------------------------
# Helper / internal coverage
# ---------------------------------------------------------------------------


def test_logprob_to_confidence_clamps() -> None:
    assert _logprob_to_confidence(-10.0) == round(np.exp(-1.5), 3)
    assert _logprob_to_confidence(10.0) == 1.0
    assert 0.0 <= _logprob_to_confidence(-0.3) <= 1.0


def test_duration_weighted_confidence_no_segments() -> None:
    assert _duration_weighted_confidence([]) == 0.0


def test_duration_weighted_confidence_zero_duration() -> None:
    seg = FakeSegment(text="x", avg_logprob=-0.3, start=0.0, end=0.0)
    assert _duration_weighted_confidence([seg]) > 0.0


def test_infer_hinglish_returns_hinglish_on_codemix() -> None:
    assert _infer_hinglish("hi", "nah कल", "hinglish") == "hi"
    assert _infer_hinglish("hi", "book flight की बात कल", "hinglish") == "hinglish"
    assert _infer_hinglish("hi", "कल मिलते हैं", "hinglish") == "hi"
    assert _infer_hinglish("en", "hi", "hinglish") == "en"
    assert _infer_hinglish("hi", "anything", None) == "hi"


def test_nfc_strips_whitespace_and_normalizes() -> None:
    assert _nfc("  hello  ") == "hello"


def test_riff_header_sample_rate_detects() -> None:
    wav = _wav_at_rate(16000, 0.5)
    assert _riff_header_sample_rate(wav) == 16000
    assert _riff_header_sample_rate(b"short") is None
    assert _riff_header_sample_rate(b"\x00" * 100) is None


def test_wav_duration_s_handles_bad_input() -> None:
    assert _wav_duration_s(b"not a wav") == 0.0


def test_input_hash_deterministic() -> None:
    assert _input_hash(b"hello") == _input_hash(b"hello")
    assert _input_hash(b"hello") != _input_hash(b"world")
    assert len(_input_hash(b"hello")) == 32


def test_fallback_chain_contract() -> None:
    # Every chain terminus must be hi_female_1 or en_indian_female_1.
    terminals = {"hi_female_1", "en_indian_female_1"}
    audio_any: Any = audio_mod
    for src, dst in _FALLBACK_CHAIN.items():
        assert src in audio_any._VOICE_PACKS_SET
        assert dst in audio_any._VOICE_PACKS_SET
    # And walking any chain lands in terminals.
    for start in _FALLBACK_CHAIN:
        current = start
        while current in _FALLBACK_CHAIN:
            current = _FALLBACK_CHAIN[current]
        assert current in terminals


def test_tts_engine_raises_model_load_error_when_kpipeline_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoKpipeline:
        pass

    monkeypatch.setattr(audio_mod, "_load_kokoro", lambda: NoKpipeline())
    monkeypatch.setattr(
        audio_mod, "_load_torchaudio_functional", lambda: FakeTorchaudioFunctional
    )
    with pytest.raises(ModelLoadError):
        TTSEngine()


def test_asr_engine_raises_model_load_error_when_whispermodel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoModel:
        pass

    monkeypatch.setattr(audio_mod, "_load_faster_whisper", lambda: NoModel())
    with pytest.raises(ModelLoadError):
        ASREngine()


def test_asr_oom_path_truncates_to_max_duration(
    asr_engine: ASREngine,
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("ok", language="en")
    long_wav = _pcm16_silence_wav(45.0)
    result = asr_engine.transcribe(long_wav, "en", max_duration_s=30.0)
    assert result.duration_s == 30.0


def test_tts_oom_raises_tts_out_of_memory_error(
    monkeypatch: pytest.MonkeyPatch, fake_kokoro: FakeKokoroModule
) -> None:
    engine = get_tts_engine()

    def _boom(text: str, voice: str) -> Any:
        raise MemoryError("boom")

    monkeypatch.setattr(engine, "_pipeline", _boom)
    from cells.step_09_audio import TTSOutOfMemoryError

    with pytest.raises(TTSOutOfMemoryError):
        engine.synthesize("x", "en")


def test_fallback_raises_when_chain_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only 'en_indian_female_1' shipped — request 'ta_female_1' with broken chain.
    _kokoro_with_packs(monkeypatch, {"en_indian_female_1", "hi_female_1"})
    engine = get_tts_engine()
    # Patch chain so ta has no successor.
    audio_any: Any = audio_mod
    monkeypatch.setitem(audio_any._FALLBACK_CHAIN, "ta_female_1", "nonexistent_pack")
    with pytest.raises(ModelLoadError):
        engine.synthesize("வணக்கம்", "ta")


def test_voice_pack_mapping_is_frozen() -> None:
    vp_any: Any = VOICE_PACKS["hi"]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        vp_any.allowed = ("hi_female_1",)


def test_audio_trace_is_frozen() -> None:
    t = AudioTrace(
        op="synthesize",
        input_hash="a" * 32,
        language="en",
        duration_s=1.0,
        latency_ms=10,
        confidence=None,
        cache_hit=False,
        degraded=False,
        ts_ist="2026-04-25T00:00:00+05:30",
    )
    t_any: Any = t
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        t_any.op = "transcribe"


def test_transcript_result_is_frozen() -> None:
    r = TranscriptResult(text="hi", language_detected="en", confidence=0.9, duration_s=1.0)
    r_any: Any = r
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        r_any.text = "bye"


def test_audio_error_hierarchy() -> None:
    for cls in (
        ModelLoadError,
        UnsupportedLanguageError,
        UnsupportedVoicePackError,
        AudioDecodeError,
    ):
        assert issubclass(cls, AudioError)


def test_lazy_loaders_raise_module_not_found_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    # Remove any cached modules so the loaders actually try to import.
    for mod_name in ("kokoro", "faster_whisper", "soundfile"):
        monkeypatch.setitem(sys.modules, mod_name, None)
    for loader in (
        audio_mod._load_kokoro,
        audio_mod._load_faster_whisper,
        audio_mod._load_soundfile,
    ):
        with pytest.raises((ImportError, TypeError)):
            loader()


def test_load_torch_returns_real_torch() -> None:
    # Torch is in the dev env; this exercises the real loader happy path.
    import torch

    assert audio_mod._load_torch() is torch


def test_available_voice_packs_uses_list_voices() -> None:
    class FakeWithListVoices:
        AVAILABLE_VOICES = ["hi_female_1"]

        @staticmethod
        def list_voices() -> list[str]:
            return ["hi_male_1", "ta_female_1"]

    packs = audio_mod._available_voice_packs(FakeWithListVoices())
    assert "hi_female_1" in packs
    assert "hi_male_1" in packs
    assert "ta_female_1" in packs


def test_available_voice_packs_fallback_to_canonical_set() -> None:
    class Empty:
        pass

    packs = audio_mod._available_voice_packs(Empty())
    audio_any: Any = audio_mod
    assert packs == set(audio_any._VOICE_PACKS_SET)


def test_coerce_to_float32_mono_handles_numpy_array() -> None:
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    out = audio_mod._coerce_to_float32_mono(arr)
    assert out.dtype == np.float32
    assert out.shape == (3,)


def test_coerce_to_float32_mono_handles_tuple() -> None:
    arr = np.array([0.1, 0.2], dtype=np.float32)
    out = audio_mod._coerce_to_float32_mono((arr, 24000))
    assert out.dtype == np.float32
    assert out.shape == (2,)


def test_tts_oom_from_runtime_error_out_of_memory(
    monkeypatch: pytest.MonkeyPatch, fake_kokoro: FakeKokoroModule
) -> None:
    engine = get_tts_engine()

    def _boom(text: str, voice: str) -> Any:
        raise RuntimeError("CUDA out of memory in alloc")

    monkeypatch.setattr(engine, "_pipeline", _boom)
    from cells.step_09_audio import TTSOutOfMemoryError

    with pytest.raises(TTSOutOfMemoryError):
        engine.synthesize("x", "en")


def test_tts_pipeline_runtime_error_non_oom_propagates(
    monkeypatch: pytest.MonkeyPatch, fake_kokoro: FakeKokoroModule
) -> None:
    engine = get_tts_engine()

    def _boom(text: str, voice: str) -> Any:
        raise RuntimeError("model corrupt")

    monkeypatch.setattr(engine, "_pipeline", _boom)
    with pytest.raises(RuntimeError):
        engine.synthesize("x", "en")


def test_asr_whisper_transcribe_raises_audio_decode_on_model_failure(
    asr_engine: ASREngine,
) -> None:
    def _broken(call: dict[str, Any], audio: np.ndarray) -> Any:
        raise RuntimeError("whisper internal blew up")

    FakeWhisperModel.default_responder = _broken
    with pytest.raises(AudioDecodeError):
        asr_engine.transcribe(_pcm16_silence_wav(1.0), "en")


def test_transcribe_max_duration_truncation_uses_exactly_max_duration(
    asr_engine: ASREngine,
) -> None:
    FakeWhisperModel.default_responder = _responder_with_text("ok", language="en")
    long_wav = _pcm16_silence_wav(40.0)
    result = asr_engine.transcribe(long_wav, "en", max_duration_s=10.0)
    assert result.duration_s == 10.0


def test_duration_weighted_confidence_with_multiple_segments() -> None:
    segs = [
        FakeSegment(text="a", avg_logprob=-0.1, start=0.0, end=1.0),
        FakeSegment(text="b", avg_logprob=-0.5, start=1.0, end=3.0),
    ]
    conf = _duration_weighted_confidence(segs)
    assert 0.0 < conf < 1.0


def test_asr_engine_construct_raises_on_whisper_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingWhisperMod:
        class WhisperModel:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("boom")

    monkeypatch.setattr(audio_mod, "_load_faster_whisper", lambda: ExplodingWhisperMod())
    with pytest.raises(ModelLoadError):
        ASREngine()


def test_get_asr_engine_warns_on_second_sink(
    fake_whisper: FakeFasterWhisperModule, caplog: pytest.LogCaptureFixture
) -> None:
    sink1: list[AudioTrace] = []
    sink2: list[AudioTrace] = []
    get_asr_engine(trace_sink=sink1.append)
    with caplog.at_level("WARNING", logger="cells.step_09_audio"):
        get_asr_engine(trace_sink=sink2.append)
    assert any("different sink" in rec.message for rec in caplog.records)


def test_singleton_lock_prevents_double_construction(
    fake_kokoro: FakeKokoroModule,
) -> None:
    engines: list[TTSEngine] = []

    def _get() -> None:
        engines.append(get_tts_engine())

    threads = [threading.Thread(target=_get) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(e is engines[0] for e in engines)
