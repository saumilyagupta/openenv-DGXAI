# audio.md — DriftCall Audio Pipeline (Kokoro-82M TTS + faster-whisper-small ASR)

**Owner:** Person C (Training & Data), secondary: Person A (integration glue)
**Implements:** DESIGN.md §9 (Audio Pipeline, 9.1–9.4), §3.3 (Deployed Env Topology), §3.4 (Demo Topology)
**Status:** DRAFT — pending ≥ 2 fresh critic rounds

---

## 1. Purpose

`driftcall/audio/` houses the two model wrappers that convert between text and speech at the **env boundary**: `tts_kokoro.py` (text → 16 kHz mono WAV bytes) and `asr_whisper.py` (WAV/PCM bytes → transcript + detected language + confidence). They exist so the deployed env and demo Space can honestly claim "voice-first" while the training loop stays text-in/text-out for throughput.

This module is the **single place** where audio-model state lives. Both engines are heavy (~82M and ~244M params respectively) and slow to initialize on CPU, so each exposes a module-level singleton constructed lazily on first call and reused across all sessions in the process. The FastAPI env (`app.py`) calls the factory once at startup; the Gradio demo (`demo/app_gradio.py`) does the same. The training loop (`training/train_grpo.py`) **never imports these modules** — not even the factory — because `import kokoro` / `import faster_whisper` pulls in torchaudio and a 50 MB tokenizer per process, and we do not want that weight in the GRPO rollout worker.

The guiding constraints from DESIGN.md §9:

1. **CPU-only.** Both models must run on the free-tier HF Space (basic CPU). No `cuda` fall-through, no `torch.compile`, no GPU-dependent kernels. Kokoro-82M is 3–11× real-time on CPU; faster-whisper-small (int8) is ~1× real-time. Both fit in <1.2 GB RAM each.
2. **Deterministic where possible.** TTS takes a `seed: int = 0` argument forwarded to torch's generator so synthesized clips are byte-reproducible given the same (text, voice, seed) triple. ASR uses `beam_size=1` (greedy) for reproducibility; with `vad_filter=True`, outputs are stable across runs on the same input.
3. **Latency budgets.** TTS < 500 ms for a 1-sentence utterance. ASR ≈ 1× real-time (a 4-second clip decodes in ≈ 4 seconds on CPU basic). Env `/step` endpoint budgets 2 seconds total per turn — the audio path must not dominate.
4. **Indic support.** Hindi, Tamil, Kannada, English, and Hinglish (code-mixed). Voice-pack selection per language is defined in §4.3; ASR language hint is passed per-episode from `GoalSpec.language`.

The module is **not** called on every training rollout — DESIGN.md §9.4 is emphatic about this, and §3 ("Behavior spec") documents the runtime split.

### 1.1 Whisper size trade-off + migration path

`faster-whisper-small` (~244M params, ~120 MB int8) was chosen to hit the ~1× real-time decode budget on free-tier CPU Space. We explicitly acknowledge this comes at a cost: `small` has **measurably degraded Word Error Rate on Hindi / Tamil / Kannada** compared to `large-v3` — published faster-whisper benchmarks show roughly a 5–10 percentage point WER gap on Indic audio depending on noise and code-mix. `large-v3` is not a free-tier option: ~3 GB weights on disk, >3 GB resident RAM during decode, and ≥ 3× real-time on CPU basic — it would bust both memory (16 GB tier shared across app, sim-caller, TTS, observation builder) and the §3 latency budget.

**Migration path (explicit, not aspirational):**

1. If Batch C2 baseline R1 on Hindi episodes is **< 0.4**, bump to `Systran/faster-whisper-medium` (~700 MB int8). This is a one-line `model_id=` change; all behaviour in this doc still holds. Move the **env Space only** (not demo) to HF CPU Pro (+$5/mo, fits in the ≤ $30/mo deployment budget per DESIGN.md §13).
2. If `medium` is still insufficient on Hindi/Tamil/Kannada (R1 < 0.4 after Stage-1 training), escalate to `large-v3` **on the demo Space only** (ZeroGPU), keeping the env Space on `small`/`medium` on CPU. This means the demo plays the more impressive transcript while the env used for reward grading stays on the deterministic CPU config — an acceptable asymmetry because demo ASR is never used for reward attribution (see §6.3 — rewards do not re-transcribe).

The chosen default for hackathon ship is `small` + int8 on CPU. Any escalation above requires orchestrator approval and a DESIGN.md §9.2 update.

---

## 2. Interface

Every declaration below is the *exact* target signature. `env.py` / `app.py` / `demo/app_gradio.py` depend on these signatures; no addition or rename is allowed without a DESIGN.md update first.

### 2.1 `driftcall/audio/tts_kokoro.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]
VoicePack = Literal[
    "hi_female_1",
    "hi_male_1",
    "ta_female_1",
    "kn_male_1",
    "en_indian_female_1",
]


@dataclass(frozen=True)
class VoicePackMapping:
    """Per-language default + allowed voice packs for Kokoro.

    DESIGN.md §9.1 lists the five packs. The mapping is frozen at module
    load and never mutated.
    """

    language: LanguageCode
    default: VoicePack
    allowed: tuple[VoicePack, ...]


# Module-level constant. Frozen at import time; see §4.3 for the authoritative
# per-row rationale. The literal below IS the full contents — five entries, one
# per LanguageCode. No runtime mutation.
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


class TTSEngine:
    """Kokoro-82M wrapper. One instance per process.

    Constructed via `get_tts_engine()`; do NOT instantiate directly in
    consumer code — the singleton guarantees the model is loaded once.
    """

    def __init__(
        self,
        *,
        model_id: str = "hexgrad/Kokoro-82M",
        trace_sink: "Callable[[AudioTrace], None] | None" = None,
    ) -> None: ...

    def synthesize(
        self,
        text: str,
        language_code: LanguageCode,
        voice_pack: VoicePack | None = None,
        *,
        seed: int = 0,
        sample_rate_hz: int = 16000,
    ) -> bytes:
        """Return 16-bit PCM mono WAV bytes.

        - `voice_pack=None` → use `VOICE_PACKS[language_code].default`.
        - `voice_pack` outside `VOICE_PACKS[language_code].allowed` → `UnsupportedVoicePackError`.
        - Deterministic given (text, voice_pack, seed, sample_rate_hz).
        - Cached in LRU (see §3.4).
        - Returns the full WAV (RIFF header + PCM), ready to write to disk
          or send as `Response(content=..., media_type="audio/wav")`.
        """

    def synthesize_to_gradio(
        self,
        text: str,
        language_hint: LanguageCode,
        voice_pack: VoicePack | None = None,
        *,
        seed: int = 0,
    ) -> tuple[int, "np.ndarray"]:
        """Gradio-friendly sibling of `synthesize`.

        Returns `(sample_rate, float32 np.ndarray)` with shape `(n_samples,)`
        (mono). This matches Gradio's `gr.Audio(type="numpy")` expected output.
        Internally calls the same Kokoro path as `synthesize()`, skipping the
        WAV encoding step and returning the float32 tensor-as-numpy directly.
        The LRU cache from §3.4 is NOT shared — Gradio-path outputs are
        cached separately under a key that includes a `fmt="numpy"` discriminator,
        so byte-cache and numpy-cache never collide.

        - `voice_pack=None` → use `VOICE_PACKS[language_hint].default`.
        - Sample rate is fixed at 16000 to match the `synthesize()` contract.
        - Deterministic given (text, voice_pack, seed).
        """

    def warmup(self) -> None:
        """Run one synthesize() with a canonical string to force model load.
        Called by `app.py` startup hook so the first real request is fast.
        """


def get_tts_engine() -> TTSEngine:
    """Return the process-wide TTSEngine singleton (lazy-constructed)."""
```

**Which caller uses which helper (binding contract):**

| Caller | Helper | Return type | Framing |
|---|---|---|---|
| FastAPI `/synthesize` endpoint in `app.py` | `TTSEngine.synthesize` | `bytes` (RIFF WAV) | `Response(content=wav_bytes, media_type="audio/wav")` |
| FastAPI `/step` audio field in `app.py` | `TTSEngine.synthesize` | `bytes` | Embedded as base64 inside the JSON step response. |
| Gradio demo in `demo/app_gradio.py` | `TTSEngine.synthesize_to_gradio` | `tuple[int, np.ndarray]` | Direct return to `gr.Audio(type="numpy")` output component. |
| Tests | Either, per-case | — | WAV-bytes tests use `synthesize`; spectral / numpy-domain tests use `synthesize_to_gradio`. |

Rationale for two helpers rather than `synthesize` + a numpy-wrapper: re-decoding WAV bytes back into float32 numpy inside the Gradio path wastes ~3 ms and doubles the memory briefly (encoded bytes + re-decoded tensor). Keeping a numpy-native return avoids that round-trip for the demo-critical path.

### 2.2 `driftcall/audio/asr_whisper.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]


@dataclass(frozen=True)
class TranscriptResult:
    """ASR output surfaced to the env observation builder.

    - `text` is NFC-normalized Unicode; empty string on silence.
    - `language_detected` is the Whisper-reported language code; may disagree
      with the hint (e.g., hint="hi", detected="en" for code-mixed utterances).
    - `confidence` is the mean token log-prob mapped to [0.0, 1.0] via
      exp-normalize (see §3.5). 1.0 = perfect, 0.0 = pathological.
    - `duration_s` is the decoded clip length in seconds (float, rounded to 3dp).
    """

    text: str
    language_detected: LanguageCode | Literal["unknown"]
    confidence: float
    duration_s: float


class ASREngine:
    """faster-whisper-small (int8) wrapper. One instance per process.

    Constructed via `get_asr_engine()`.
    """

    def __init__(
        self,
        *,
        model_id: str = "Systran/faster-whisper-small",
        compute_type: Literal["int8", "int8_float16"] = "int8",
        trace_sink: "Callable[[AudioTrace], None] | None" = None,
    ) -> None: ...

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: LanguageCode | None,
        *,
        beam_size: int = 1,
        vad_filter: bool = True,
        max_duration_s: float = 30.0,
    ) -> TranscriptResult:
        """Decode a WAV/PCM clip into a TranscriptResult.

        - `audio_bytes` must be a RIFF WAV with mono 16-bit PCM at 16 kHz OR
          raw float32 PCM at 16 kHz (detected by magic bytes). Other formats
          → `AudioDecodeError`.
        - `language_hint="hinglish"` is translated to `language="hi"` at the
          Whisper call site (Whisper has no Hinglish code); detected language
          may come back as "hi" or "en".
        - `language_hint=None` → autodetect (slower on first pass).
        - Truncates to `max_duration_s` silently and sets
          `result.duration_s = max_duration_s` (see edge case §7.3).
        - Returns a populated `TranscriptResult`; never raises on a merely
          low-confidence decode — that is a policy decision for the caller.
        """

    def warmup(self) -> None:
        """Run one transcribe() on 0.5s of silence to load weights + VAD."""


def get_asr_engine() -> ASREngine:
    """Return the process-wide ASREngine singleton (lazy-constructed)."""
```

### 2.2a `driftcall/audio/trace.py` (shared between TTS + ASR)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class AudioTrace:
    """Per-call diagnostic record for synthesize() and transcribe().

    Emitted via the `trace_sink` callback passed to each engine's __init__.
    Consumed by the `/audio/trace` FastAPI endpoint and the demo UI live overlay.
    Never mutated after construction (frozen).
    """

    op: Literal["synthesize", "transcribe"]
    input_hash: str       # blake2b hex digest of text (for TTS) or audio bytes (for ASR)
    language: str         # requested language code or "unknown"
    duration_s: float     # clip duration in seconds (output for TTS, input for ASR)
    latency_ms: int       # wall-clock call latency
    confidence: float | None   # ASR: TranscriptResult.confidence; TTS: None
    cache_hit: bool       # TTS: LRU hit? ASR: always False
    degraded: bool        # True on voice-pack fallback (TTS) OR coerced-empty (ASR)
    ts_ist: str           # ISO-8601 timestamp in Asia/Kolkata tz


TraceSink = Callable[[AudioTrace], None]
```

### 2.3 Custom exceptions

Defined in `driftcall/audio/errors.py` (tiny module, shared):

```python
class AudioError(Exception): ...
class ModelLoadError(AudioError): ...
class UnsupportedLanguageError(AudioError): ...
class UnsupportedVoicePackError(AudioError): ...
class AudioDecodeError(AudioError): ...
class AudioTooLongError(AudioError): ...   # only raised if caller passes strict=True
class TTSOutOfMemoryError(AudioError): ...
```

`env.py` catches `AudioError` at the boundary and either degrades (see §5) or 500s the HTTP response.

### 2.4 `__all__`

```python
# tts_kokoro.py
__all__ = [
    "LanguageCode",
    "VoicePack",
    "VoicePackMapping",
    "VOICE_PACKS",
    "TTSEngine",
    "get_tts_engine",
]

# asr_whisper.py
__all__ = [
    "LanguageCode",
    "TranscriptResult",
    "ASREngine",
    "get_asr_engine",
]

# trace.py
__all__ = [
    "AudioTrace",
    "TraceSink",
]
```

---

## 3. Behavior spec

### 3.1 Training-vs-deploy split (DESIGN.md §9.4 — load-bearing)

| Runtime | Imports `driftcall.audio`? | TTS in loop? | ASR in loop? | Why |
|---|---|---|---|---|
| **Training** (`training/train_grpo.py`, local V100) | **No.** Explicit negative contract. | No | No | Speed. Pre-authored text transcripts go straight into `DriftCallObservation.last_transcript`. `last_confidence=1.0` (treated as perfect ASR). ~10× faster rollouts. |
| **Deployed env** (HF Space CPU basic, `app.py`) | Yes, via `get_tts_engine()` + `get_asr_engine()` at startup | Yes (on `SPEAK` actions) | Yes (on every inbound `/step` that carries audio bytes) | DESIGN.md §9.4: "env is genuinely voice-driven for realism". Sim-caller in §3.1 synthesizes user utterances; ASR at env boundary transcribes before embedding into observation. |
| **Demo Space** (Gradio, ZeroGPU / A10G) | Yes | Yes | Yes + live mic input | Judge interaction. |

`env.py` toggles between modes via a single flag: `DriftCallEnv(audio_boundary_enabled: bool = False)`. Default `False` means the training path; `True` is set only inside `app.py` (FastAPI) and `demo/app_gradio.py`. The flag is checked once in `__init__`; it does not change per-step. **Tests:** `tests/test_env.py` must verify that `DriftCallEnv(audio_boundary_enabled=False)` does not import `driftcall.audio.*` at all (use `sys.modules` assertion before/after reset).

### 3.2 Model load lifecycle

- **Lazy singleton.** `get_tts_engine()` and `get_asr_engine()` wrap a `_tts: TTSEngine | None = None` / `_asr: ASREngine | None = None` module-global. First call constructs and caches; subsequent calls return the cache. Thread-safe via `threading.Lock` (not asyncio — FastAPI workers are thread-per-request under the default gunicorn/uvicorn sync path, and even on async workers the lock is uncontended after warmup).
- **Download.** Kokoro-82M and faster-whisper-small are pulled from HF Hub on first load. The Dockerfile for the env Space (`deploy_env_space.md`) pre-pulls both into `/root/.cache/huggingface/` at image-build time so cold start on Space does not re-download (multi-gigabyte pull would exceed the free-tier timeout).
- **Warmup.** `app.py` lifespan hook calls `get_tts_engine().warmup()` and `get_asr_engine().warmup()` serially before the server binds its port. This burns ~8 seconds but ensures the first user request does not face a 5+ second first-inference penalty. The demo Space does the same in its Gradio `demo.load` event.
- **Unload.** Never. The engines live for the process lifetime. Sessions come and go; the models stay hot. This is safe because both are stateless between calls (no session-private buffers).

### 3.3 Determinism

- **TTS.** Kokoro exposes a `torch.Generator` seed. `synthesize(..., seed=N)` forwards `torch.manual_seed(N)` inside a `torch.random.fork_rng()` context so the global RNG is unaffected (critical — do not pollute the trainer's RNG). Given identical `(text, voice_pack, seed, sample_rate_hz)`, byte-for-byte output. Floating-point non-determinism across CPU architectures is theoretical but not observed on x86_64 AVX2, which is the only target (Docker image pins to `python:3.11-slim` on amd64).
- **ASR.** `beam_size=1` disables beam search (greedy decoding, deterministic given weights + input). `vad_filter=True` uses a deterministic silero-VAD pass that is stable across runs. `temperature=0.0` is the faster-whisper default — we do not override.
- **Training implication.** Neither engine is called in training, so RNG safety there is moot, but `fork_rng()` is kept for hygiene in case future eval scripts run TTS after a seeded rollout.

### 3.4 LRU caching for TTS

- **Key.** `(text_hash, voice_pack, seed, sample_rate_hz)` where `text_hash = blake2b(text.encode("utf-8"), digest_size=16).hexdigest()`. Using the hash (not the raw string) bounds key size and keeps the LRU memory footprint predictable. **Key-extension rationale:** `seed` and `sample_rate_hz` are in the key because `synthesize` accepts them as arguments and they change output bytes; omitting them would cause silent cache-hit corruption when a caller changes either parameter. This is why the key is richer than the DESIGN.md-level sketch `(text, voice_pack)`.
- **Value.** The WAV bytes (typically 30–80 KB for a 1-sentence Hindi utterance at 16 kHz; up to ~180 KB for a 4–6 s Hindi utterance).
- **Capacity.** 256 entries (`functools.lru_cache(maxsize=256)` is NOT used because it doesn't handle the hash-first indirection cleanly — we use `cachetools.LRUCache(maxsize=256, getsizeof=len)` with an explicit lock and an optional byte-budget cap of **64 MB** via `cachetools.LRUCache`'s `getsizeof` + `maxsize` byte-limit mode). Implementation note: cachetools treats `maxsize` as either entry-count or total `getsizeof` sum depending on constructor form; we use the byte-sum form so worst-case memory is bounded by the byte cap, not the entry count.
- **Memory envelope (worst-case vs typical).**
  - Typical: 256 × ~60 KB ≈ **15 MB** (old number — still correct for average 1-sentence utterances).
  - Worst-case: 256 × ~180 KB = **46 MB** (4–6 s Hindi utterances at 16 kHz 16-bit post-resample).
  - Upper-bounded by the byte cap at **64 MB**. Above 64 MB, oldest entries evict by LRU order regardless of the 256 entry count.
  - Pre-resample (24 kHz, Kokoro native) bytes would be ~69 MB worst-case if we cached pre-resample; we do NOT — the resample in §4.4 happens inside `synthesize()` before WAV encoding, so the cache stores 16 kHz bytes only. This is why the cache key includes `sample_rate_hz`: if a future caller ever requests 24 kHz output, it will cache under a separate key rather than colliding with 16 kHz entries.
- **Cache scope.** **Process-wide singleton, GLOBAL — not per-session.** All concurrent sessions (up to 10 per DESIGN.md §3.3) share ONE cache. This is intentional: the TTS output for `(text, voice_pack, seed, sample_rate_hz)` is deterministic and carries no session-private data, so sharing is safe and maximises hit rate (sim-caller re-synthesizing the same `seed_utterance` across sessions benefits from the shared cache).
- **Invalidation.** None — (text, voice, seed, sample_rate_hz) tuples deterministically produce the same bytes, so cache entries are eternal. Model change invalidates everything by process restart.
- **Why cache.** Demo Space replays the same goal utterance across multiple toggle switches (base ⇄ trained LoRA), and the env's sim-caller re-synthesizes the same `seed_utterance` each time the user re-runs an episode. Hit rate is >90% in the demo setting, turning a 300 ms synth into a 1 ms memcpy.
- **No ASR caching.** ASR inputs are already-variable WAV bytes; repeat rate is low, and keying on audio-byte hashes is O(audio length). Not worth it.

### 3.5 Confidence mapping (ASR)

faster-whisper exposes per-segment `avg_logprob` (mean log-probability over tokens, in the range roughly `[-1.5, 0.0]`). We map it to a [0, 1] confidence via:

```python
def _logprob_to_confidence(avg_logprob: float) -> float:
    # avg_logprob ∈ [-1.5, 0.0] approx. Clamp then exp-normalize.
    clamped = max(-1.5, min(0.0, avg_logprob))
    return round(math.exp(clamped), 3)
```

This matches the DESIGN.md §4.1 `DriftCallObservation.last_confidence` semantics (`0.0 ≤ c ≤ 1.0`, 1.0 in training). When the clip has multiple segments, we take the duration-weighted mean of per-segment confidences.

**Empty-text-with-nonzero-confidence branching.** faster-whisper can decode to `text=""` while still reporting `avg_logprob > -1.5` (i.e., `confidence > 0`) on short non-silent clips where the acoustic model produces only whitespace / punctuation tokens that get stripped in post-processing. This is distinct from the VAD-silent case (§7.4) where VAD drops every segment before decode. Branching logic inside `transcribe()`:

```
if text == "":
    if vad_dropped_all_segments:
        # §7.4 silent-audio path
        return TranscriptResult(text="", language_detected="unknown",
                                confidence=0.0, duration_s=clip_duration)
    else:
        # Decoded to empty but audio was not VAD-silent.
        # Coerce confidence to 0.0 (we cannot trust a confident empty decode)
        # and flag as low-confidence decode so callers can treat it like the
        # silent path without losing the language hint that whisper provided.
        return TranscriptResult(
            text="",
            language_detected=<whisper-reported language, mapped>,
            confidence=0.0,
            duration_s=clip_duration,
            # degraded=True via trace sink (§3.8); no exception raised
        )
```

The env treats `text == ""` as "no intelligible speech" regardless of which branch produced it. This matches DESIGN.md §4.1's implicit contract: `last_transcript=""` means the agent should `CLARIFY` rather than assume intent.

### 3.6 Language detection & Hinglish handling

- `language_hint="hinglish"` is translated to Whisper's `language="hi"` at the call site. Whisper has no Hinglish token, but Hindi decoding on code-mixed audio produces readable transliteration + English words in Latin script roughly 85% of the time. Noise is expected and documented as Risk 3 in DESIGN.md §14.
- `TranscriptResult.language_detected` reports what Whisper says, not the hint. If hint is `"hinglish"` and Whisper reports `"hi"`, we downgrade to `"hinglish"` only when the decoded text contains ≥ 2 ASCII-letter words intermixed with Devanagari (heuristic; documented in tests).
- If Whisper returns a language code not in our 5-value Literal (e.g., `"ur"` for Urdu, `"mr"` for Marathi), `language_detected="unknown"` is surfaced; `env.py` logs a warning and falls back to `language_hint` for R4 reward attribution.

### 3.7 Concurrency

- Both engines are CPU-bound Python calls into C extensions (Kokoro via torch, faster-whisper via CTranslate2). They **release the GIL** during inference, so threaded FastAPI workers can process N concurrent transcribes at a small RAM cost. Max concurrency is governed by the env-space session cap (10 concurrent sessions per DESIGN.md §3.3). RAM usage: 10 concurrent transcribes × ~150 MB peak = 1.5 GB — fits the free CPU tier's 16 GB with margin.
- No per-session model state means two sessions can share an engine instance without lock contention beyond what CTranslate2 internally serializes.

### 3.8 Diagnostic tracing hook

Both engines accept an optional `trace_sink: Callable[[AudioTrace], None] | None = None` kwarg in `__init__`. When provided, **every call** to `synthesize()`, `synthesize_to_gradio()`, or `transcribe()` emits exactly one `AudioTrace` record (schema in §2.2a) to the sink **after** the core work completes but **before** the return statement. Emissions are wrapped in `try/except Exception: pass` so a broken sink never crashes the audio path — telemetry must never break production.

**Default.** `trace_sink=None` means no emission, zero overhead.

**Wiring in `app.py`.** The FastAPI startup hook constructs a module-global ring buffer of the most recent **100** traces (`collections.deque(maxlen=100)`) and passes its `.append` method as the sink to both engines at `get_tts_engine()` / `get_asr_engine()` construction:

```
_trace_buffer: deque[AudioTrace] = deque(maxlen=100)
tts = get_tts_engine(trace_sink=_trace_buffer.append)
asr = get_asr_engine(trace_sink=_trace_buffer.append)
```

(Note: `get_tts_engine` / `get_asr_engine` are updated to accept and forward `trace_sink` through to the first-call `__init__`; subsequent calls after the singleton is constructed ignore the kwarg — warn in logs if a different sink is passed after construction.)

**Endpoint.** `GET /audio/trace` returns `{"traces": [AudioTrace.asdict(), ...]}` with the most recent 100 records, newest-first. No auth (demo-only; the env Space is behind judge tokens anyway per DESIGN.md §3.3). This endpoint is defined in `app.py`, not here.

**Demo UI.** `demo/app_gradio.py` polls `/audio/trace` every 2 s and overlays a sparkline of `latency_ms` per op and a counter of `degraded=True` events. This is how judges see the trace health live.

**Privacy.** `input_hash` is a blake2b digest — raw text and raw audio bytes never leave the process via the trace. This is a hard invariant.

---

## 4. Data structures

### 4.1 `TranscriptResult`

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `text` | `str` | Decoded transcript, NFC-normalized Unicode | Non-None; may be empty on silence; no trailing whitespace | `ASREngine.transcribe` |
| `language_detected` | `LanguageCode \| "unknown"` | Whisper-reported language, mapped to our 5 codes or `"unknown"` | One of `{"hi","ta","kn","en","hinglish","unknown"}` | `ASREngine.transcribe` |
| `confidence` | `float` | Duration-weighted exp-normalized mean log-prob | `0.0 ≤ c ≤ 1.0`; `0.0` whenever `text == ""` (both VAD-silent per §7.4 AND decoded-empty-despite-audio per §3.5 — the latter coerces any nonzero whisper-reported confidence to `0.0`); `1.0` only by convention in training when ASR is bypassed entirely (see §3.1) | `ASREngine.transcribe` |
| `duration_s` | `float` | Clip length in seconds | `0.0 ≤ d ≤ max_duration_s`; rounded to 3dp | `ASREngine.transcribe` |

Frozen dataclass; immutable by project convention (CLAUDE.md §4.2).

### 4.2 `VoicePackMapping`

Frozen dataclass. Five instances live in the `VOICE_PACKS` module-level dict — one per `LanguageCode`. Never re-assigned after module load.

### 4.2a `AudioTrace`

Frozen dataclass, defined in `driftcall/audio/trace.py` (schema in §2.2a, emission semantics in §3.8). Fields: `op`, `input_hash`, `language`, `duration_s`, `latency_ms`, `confidence`, `cache_hit`, `degraded`, `ts_ist`. All fields are immutable; `AudioTrace` instances are produced at the tail of each synth/transcribe call and fed to the configured `trace_sink`. Consumed by `app.py`'s `/audio/trace` endpoint and the demo UI live overlay. Never serialized to disk by this module (app-level concern).

### 4.3 Voice pack table (DESIGN.md §9.1)

| `language` | `default` | `allowed` | Notes |
|---|---|---|---|
| `"hi"` | `"hi_female_1"` | `("hi_female_1", "hi_male_1")` | Kokoro Hindi voices. Female default matches most task-brief personas. |
| `"ta"` | `"ta_female_1"` | `("ta_female_1",)` | Only one Tamil pack available at Kokoro-82M size. |
| `"kn"` | `"kn_male_1"` | `("kn_male_1",)` | Only one Kannada pack. |
| `"en"` | `"en_indian_female_1"` | `("en_indian_female_1",)` | Indian-accented English per DESIGN.md §9.1. |
| `"hinglish"` | `"en_indian_female_1"` | `("en_indian_female_1", "hi_female_1")` | Hinglish utterances transliterate English lexis into Devanagari poorly, and Hindi-voice delivery of Latin script is poorer still. `en_indian_female_1` delivers code-mixed ASCII text most naturally; `hi_female_1` is retained as an A/B fallback for utterances that are ≥ 80% Devanagari. **Choice documented here per task brief.** |

Total: **5 language codes mapped, 5 distinct voice packs used across the table.**

#### 4.3.1 Shipped voice packs at pinned version

At the `kokoro>=0.3,<0.4` pin (DESIGN.md §9.1, this doc §6.1), Kokoro-82M's **actually-shipped** voice packs at the time of pinning are: `hi_female_1`, `hi_male_1`, `en_indian_female_1`, and a best-effort set of Indic packs (`ta_female_1`, `kn_male_1`) whose bundling with the HF-distributed weights is **not guaranteed** across minor releases. The Kokoro project ships voice packs as separate `.pt` files inside the model repo; some Indic packs have been reshuffled between `0.3.x` minor versions. This module must behave sanely when an Indic pack is missing from the installed bundle.

**Missing-voice-pack fallback chain (evaluated at `synthesize()` call time, not at warmup, so fallbacks can be per-call telemetry rather than fatal startup errors):**

| Requested pack | If missing from bundle, fall back to | Emitted metadata |
|---|---|---|
| `ta_female_1` | `hi_female_1` | `degraded=True`, `fallback_from="ta_female_1"` in the audio trace (§3.8) |
| `kn_male_1` | `hi_female_1` | `degraded=True`, `fallback_from="kn_male_1"` |
| `hi_male_1` | `hi_female_1` | `degraded=True`, `fallback_from="hi_male_1"` |
| `hi_female_1` | `en_indian_female_1` (last resort for Hindi text) | `degraded=True`, `fallback_from="hi_female_1"` |
| `en_indian_female_1` | — (catastrophic if also missing; see below) | — |

**Warmup policy.** `TTSEngine.warmup()` probes each pack in `VOICE_PACKS` values by attempting a 1-word synthesis. Missing Indic packs (`ta_female_1`, `kn_male_1`, `hi_male_1`) are logged at `WARN` and the fallback chain is activated for subsequent calls — **warmup does not abort the Space**. The ONE condition that DOES abort the Space at warmup is: **both `en_indian_female_1` AND `hi_female_1` missing** — this is catastrophic because there is no voice at all for Hindi or English, which are the ≥ 95% traffic languages. In that case `ModelLoadError("no usable voice pack for hi or en")` is raised and the Space fails to bind its port.

**Downstream visibility.** Whenever a fallback is used, the `degraded=True` flag travels with the response. For TTS, this lives in the `AudioTrace` (§3.8) attached to the ring buffer; for ASR, there is an analogous mechanism in §3.5's empty-string edge case. `env.py` surfaces `degraded=True` into `DriftCallObservation` via a future `last_audio_degraded: bool` field if the rewards/models doc adds it; until then the flag is telemetry-only and does not influence reward.

### 4.4 Audio byte format (WAV contract)

- **TTS output:** RIFF WAV, mono, 16-bit PCM, 16 kHz. Produced via `torchaudio.save(..., format="wav", bits_per_sample=16, sample_rate=16000)` into an in-memory `io.BytesIO`, then `.getvalue()`. Header + data.
- **Resampling call site (canonical).** Kokoro-82M synthesizes at **24 kHz** natively. Resampling to the env's 16 kHz target happens **inside `TTSEngine.synthesize`, BEFORE WAV encoding**, via:
  ```python
  import torchaudio.functional as F
  pcm_16k = F.resample(pcm_24k, orig_freq=24000, new_freq=16000, lowpass_filter_width=64)
  # ...then torchaudio.save(buf, pcm_16k, sample_rate=16000, bits_per_sample=16, format="wav")
  ```
  `torchaudio.save(..., sample_rate=16000)` is called **after** the resample — it is an encoder, not a resampler. The `sample_rate` kwarg on `save` only writes the RIFF header value; it does not change the tensor's sample rate. Consequence: LRU-cached bytes are always 16 kHz (see §3.4). The `sample_rate_hz` synth-argument is validated at the top of `synthesize()` — only `16000` is supported in the v1 contract; any other value raises `UnsupportedLanguageError`-style error (future work: allow 24 kHz path per Open Question historically 9.4, now resolved — see §9).
- **ASR input resampling policy.** ASR does **NOT** auto-resample. If `transcribe()` receives audio whose header sample-rate is not 16 kHz (detected via `soundfile.info` before full read, or via the RIFF `nSamplesPerSec` field at bytes 24–27), it raises `AudioDecodeError("input must be 16 kHz mono; caller must pre-resample")`. Rationale: silently resampling at the ASR boundary hides caller bugs and costs 20–40 ms per call; since TTS already produces 16 kHz and the Gradio mic component is configured to deliver 16 kHz, any non-16kHz input indicates a mis-wired caller that must be fixed, not papered over.
- **ASR input:** same format required. The `transcribe` method sniffs magic bytes (`RIFF....WAVE` at offset 0–11) and dispatches to `soundfile.read(BytesIO(audio_bytes))`; raw float32 PCM at 16 kHz is accepted as a second path for the demo mic input pipeline (which delivers float32 by default from Gradio's `type="numpy"` component, configured with `sample_rate=16000` in §6.2 wiring). Other formats (mp3, ogg, flac) are **rejected** with `AudioDecodeError` — we do not ship ffmpeg in the CPU Space image.

---

## 5. Error modes

| Situation | Exception | Handled by |
|---|---|---|
| Kokoro-82M weights cannot be pulled from HF Hub (network / rate-limit / disk full) at `get_tts_engine()` first-call | `ModelLoadError` wrapping original `huggingface_hub` / `OSError` | `app.py` startup hook fails fast → HF Space log shows error; server does not bind. Retried on next container boot. |
| faster-whisper-small weights cannot be pulled at `get_asr_engine()` first-call | `ModelLoadError` | Same as above. |
| `synthesize(..., language_code=X)` with `X` not in `VOICE_PACKS` keys | `UnsupportedLanguageError` | `env.py` catches → logs, falls back to `en_indian_female_1` at `en`, and sets R4 penalty flag for language mismatch (enforced by rewards, not here). |
| `synthesize(..., voice_pack=X)` where X not in `VOICE_PACKS[language_code].allowed` | `UnsupportedVoicePackError` | Caller error — 400 at HTTP boundary. |
| `transcribe()` receives bytes with no valid WAV header and no float32-PCM magic | `AudioDecodeError` | `env.py` returns an `UNKNOWN_AUDIO` status in observation; `last_transcript=""`, `last_confidence=0.0`. |
| `transcribe()` low-confidence decode (`confidence < 0.3`) | **Not** an exception. Returned normally. | Caller (`env.py`) sets `DriftCallObservation.last_confidence` honestly; downstream the agent may `CLARIFY` to re-prompt. R4 does not penalize low ASR confidence — it is a natural observation feature. |
| `transcribe()` returns `text=""` with whisper-reported `confidence > 0` (decoded-empty-despite-audio, not VAD-silent — see §3.5) | **Not** an exception. `confidence` is coerced to `0.0`, `degraded=True` in trace, result returned with whisper-reported `language_detected`. | Env treats identically to the silent case: "no intelligible speech"; agent should `CLARIFY`. |
| Audio duration > `max_duration_s` (default 30 s) | Truncated silently. NOT raised. Unless caller passes `strict=True` (not in default signature) — then `AudioTooLongError`. | Documented in §7.3. `env.py` always uses the default (silent truncation). |
| TTS OOM mid-synthesis on a pathologically long string (> 4 KB of text) | `TTSOutOfMemoryError` wrapping the originating `MemoryError` or `RuntimeError` (CPU-only deployment per §1, §3.1, §6.1 — CUDA OOM cannot occur; torch on CPU raises `RuntimeError` or Python's built-in `MemoryError` on large tensor allocation failure) | `env.py` catches → agent's `SPEAK` is dropped with a warning; the turn still counts. R4 penalty for format non-compliance does not apply (env-side failure, not agent fault). |
| Indic voice pack (`ta_female_1`, `kn_male_1`, `hi_male_1`) missing from Kokoro bundle | **No exception** — fallback chain per §4.3.1 activated. `degraded=True` attached to trace. | Warmup logs WARN. Startup continues. |
| BOTH `en_indian_female_1` AND `hi_female_1` missing from Kokoro bundle (catastrophic — no Hindi/English voice) | `ModelLoadError("no usable voice pack for hi or en")` | `app.py` warmup catches and aborts startup — the Space will not bind its port. Operator must re-pull weights or downgrade `kokoro` pin. |
| `voice_pack` argument not in `VOICE_PACKS[language_code].allowed` | `UnsupportedVoicePackError` (caller bug, distinct from bundle missing) | Caller error — 400 at HTTP boundary. |
| `language_hint=None` with silent/empty audio | Returns `TranscriptResult(text="", language_detected="unknown", confidence=0.0, duration_s=<duration>)`. No exception. | Normal flow. |
| Concurrent `warmup()` calls from two threads | Second call is a no-op (singleton guard); first blocks until ready. | Tested. |

**Partial-result policy:** ASR never returns a partial `TranscriptResult`. Either the decode completes (even if `text=""`) or an `AudioError` subclass propagates. No `None` fields.

---

## 6. Dependencies

### 6.1 Upstream (what this module imports)

| Dependency | Version pin | License | Why |
|---|---|---|---|
| `kokoro` (Kokoro-82M official SDK wrapping the HF model `hexgrad/Kokoro-82M`) | `>=0.3, <0.4` | Apache 2.0 | TTS synthesis. Pure-CPU path. |
| `faster-whisper` | `>=1.0, <2.0` | MIT | ASR via CTranslate2 int8 runtime. |
| `ctranslate2` | (transitive of faster-whisper) | MIT | CTranslate2 runtime, CPU-only wheel. |
| `torchaudio` | `>=2.1, <3.0` | BSD-3 | WAV encoding from raw Kokoro PCM tensors. Pulled in by Kokoro anyway. |
| `soundfile` | `>=0.12` | BSD-3 | WAV decoding for ASR input; works without ffmpeg. |
| `cachetools` | `>=5.3` | MIT | `LRUCache` for TTS bytes. |
| Python stdlib | — | — | `math`, `io`, `hashlib`, `threading`, `dataclasses`, `enum`, `typing`. |

Not depended on: `ffmpeg-python`, `librosa`, `pydub`, `gradio` (demo-only), `fastapi` (app-only).

### 6.2 Downstream (who imports `driftcall/audio/`)

| Consumer | Imports |
|---|---|
| `app.py` (FastAPI env entrypoint) | `get_tts_engine`, `get_asr_engine`, `TranscriptResult`, all exceptions. Uses `TTSEngine.synthesize` (WAV bytes path) for HTTP responses via `Response(content=..., media_type="audio/wav")` or base64-embedded inside `/step`. Called at startup hook + on every `/step` that carries audio. |
| `demo/app_gradio.py` | `get_tts_engine`, `get_asr_engine`, `TranscriptResult`, all exceptions. Uses `TTSEngine.synthesize_to_gradio` (`tuple[int, np.ndarray]`) as the direct return value for `gr.Audio(type="numpy")` output components. Mic component feeds `transcribe()` via `audio_bytes` obtained from Gradio's float32 PCM at 16 kHz (configured on the component, not converted at the audio-module layer). Never calls `synthesize` (bytes) — that is only for FastAPI. |
| `driftcall/env.py` | **Only when `audio_boundary_enabled=True`.** Calls `get_tts_engine()` / `get_asr_engine()` lazily inside `_maybe_synthesize()` / `_maybe_transcribe()` helpers. Never imports at module top. |
| `tests/test_audio.py` | All public symbols for unit tests. |
| `tests/test_e2e.py` | `TranscriptResult` for constructing deploy-mode integration fixtures. |

### 6.3 Explicit non-consumers (load-bearing)

- `training/train_grpo.py` — **MUST NOT** import any symbol from `driftcall.audio`. Enforced by a linter rule in `pyproject.toml`:
  ```toml
  [tool.ruff.lint.flake8-tidy-imports.banned-api]
  "driftcall.audio".msg = "Training loop is text-only (DESIGN.md §9.4). Do not import audio in training/."
  ```
  The rule is scoped to `training/**/*.py` via a `per-file-ignores` override pattern.
- `training/eval_baseline.py`, `training/eval_final.py` — same rule. Eval runs on text transcripts; if live-audio eval is needed later, it becomes a separate `eval_audio.py` script.
- `rewards.py` — does not import audio. Rewards read `DriftCallObservation.last_confidence` (a float) and `last_lang` (a string) which the env boundary has already set. Rewards do not re-transcribe.

### 6.4 Model assets

Licenses below cover **model weights** (distinct from the Python package licenses in §6.1).

| Model repo | Params / size | License (weights) | Notes |
|---|---|---|---|
| `hexgrad/Kokoro-82M` | 82M params, ~330 MB fp32 (~160 MB int8, unused) | Apache 2.0 | Kokoro fp32 is fast enough on CPU; int8 path not exercised. |
| `Systran/faster-whisper-small` | ~244M params, ~470 MB fp32 / ~120 MB int8 | Apache 2.0 | We use int8 on CPU. See §1.1 for WER trade-off vs `medium` / `large-v3` and the migration path. |

Total cache-on-disk footprint: ~450 MB. Dockerfile pre-pulls both into `/root/.cache/huggingface/`; image size budget per DESIGN.md Risk 10: < 2 GB total. Audio weights take ~25% of that. If §1.1's migration is triggered and we swap to `Systran/faster-whisper-medium` (~700 MB int8), total weights rise to ~1 GB and the image size budget still holds.

---

## 7. Edge cases

Eight cases that the test plan (`docs/tests/audio_tests.md`) must cover. Each case is the minimum test that would catch regressions.

### 7.1 Hinglish code-mix Whisper noise

`transcribe(wav_of("Bhai Friday ko Bangalore jaana hai"), language_hint="hinglish")` — Whisper-Hindi decoding on code-mixed audio returns mixed Devanagari+Latin output. Test asserts: (a) `text` is non-empty, (b) `confidence` is finite in [0, 1], (c) `language_detected` is one of `{"hi", "hinglish"}`. Text-equality is NOT asserted (Risk 3, semantic match downstream). If this test becomes flaky on a new faster-whisper release, we pin the version tighter — do not loosen the assertions.

### 7.2 Kannada voice pack quality

`synthesize("Namaskara, saha haridu", language_code="kn")` — Kokoro's Kannada pack is known to produce occasional glitches on loanwords. Test asserts: (a) returns non-empty WAV bytes, (b) the WAV parses with `soundfile.read` and has `>= 1.5 s` duration for this phrase, (c) duration is within 30% of expected (2.0 s). Audio-quality assertions beyond this are out of scope — DESIGN.md Risk 8 accepts "pre-generate demo audio with careful voice-pack selection" as mitigation.

### 7.3 Long utterance truncation

`transcribe()` receives a 45-second WAV when `max_duration_s=30.0`. Default path: silent truncation; `result.duration_s == 30.0`; `text` contains only the first 30 s of content. Test: feed a synthesized 45-s clip of counted numbers 1–45, assert the decoded text does NOT contain "40" or "45". No exception raised.

### 7.4 Silent audio

`transcribe(wav_of_silence(duration_s=3.0), language_hint="hi")` — VAD filter drops all segments. `TranscriptResult(text="", language_detected="unknown", confidence=0.0, duration_s=3.0)` is returned. Explicitly NO exception. `env.py` interprets `text==""` as "user did not speak" and the agent observation reflects that.

### 7.5 Wrong-language hint

`transcribe(wav_of("The flight leaves at six"), language_hint="ta")` — Whisper is forced into Tamil decoding on English audio. Result typically garbled. Test asserts: (a) no exception, (b) `language_detected` may disagree with hint, (c) `confidence` is likely low (< 0.5 expected, not strictly asserted to avoid flakes). `env.py` logs a WARN but does not retry with autodetect — retry is the agent's job (via `CLARIFY`).

### 7.6 Concurrent sessions sharing engine

Spawn 5 threads each calling `transcribe()` on distinct 2-second clips simultaneously. Assert: (a) all 5 return `TranscriptResult`, (b) wall-clock is less than 5× sequential (thanks to GIL release in CTranslate2), (c) no exceptions. Same test for TTS, but parallelism benefit is smaller (torch on CPU serializes heavily).

### 7.7 TTS LRU hit

Call `synthesize(text="नमस्ते", language_code="hi", seed=0)` twice back-to-back. First call p50 ≈ 250 ms, second call p50 < 5 ms (LRU hit). Assert second call returns byte-identical WAV and is ≥ 10× faster. This guards against accidental cache-key drift.

### 7.8 TTS seed determinism

`synthesize(text="कल मिलते हैं", language_code="hi", seed=7)` called from two separate fresh processes (subprocess fixture) produces byte-identical WAV. Guards against RNG leak from outer training/eval code. Uses `fork_rng` internally; test validates by calling `random.random()` before and after to confirm global RNG is undisturbed.

### 7.9 Training-loop import firewall

Import `training.train_grpo` in a subprocess. After import, assert `"driftcall.audio.tts_kokoro" not in sys.modules` and `"driftcall.audio.asr_whisper" not in sys.modules`. This guards DESIGN.md §9.4 at the structural level. The ruff banned-api rule should fire in CI; this test belts-and-braces it.

### 7.10 Model-load failure at startup

Monkeypatch `kokoro.KPipeline` to raise `OSError("no network")`. Call `get_tts_engine()`. Assert `ModelLoadError` is raised with the original `OSError` in `__cause__`. Second call re-attempts load (singleton state did NOT cache the failure) — this is intentional so a transient HF Hub outage does not permanently break the process. Test both on an ASR mock too.

---

## 8. Examples

### 8.1 Hindi TTS round-trip (deployed env sim-caller path)

```python
from __future__ import annotations

from driftcall.audio.tts_kokoro import get_tts_engine

tts = get_tts_engine()
wav_bytes = tts.synthesize(
    text="नमस्ते, कल दिल्ली की फ्लाइट बुक करनी है, सात हज़ार के अंदर।",
    language_code="hi",
    voice_pack="hi_female_1",
    seed=0,
)

# Assertions typical of the test and the sim-caller:
assert isinstance(wav_bytes, bytes)
assert wav_bytes[:4] == b"RIFF"
assert wav_bytes[8:12] == b"WAVE"
# Write to disk for debugging:
# pathlib.Path("goal_hi.wav").write_bytes(wav_bytes)

# File size for a ~4 s clip at 16 kHz 16-bit mono ≈ 128 KB.
assert 60_000 < len(wav_bytes) < 180_000

# Duration can be extracted cheaply via soundfile:
import io, soundfile
info = soundfile.info(io.BytesIO(wav_bytes))
assert 3.0 < info.duration < 6.0
assert info.samplerate == 16_000
assert info.channels == 1
```

### 8.2 Hinglish ASR (env boundary transcribing user audio)

```python
from __future__ import annotations

from pathlib import Path

from driftcall.audio.asr_whisper import get_asr_engine, TranscriptResult

asr = get_asr_engine()
wav_bytes = Path("user_hinglish_bangalore.wav").read_bytes()

result: TranscriptResult = asr.transcribe(
    audio_bytes=wav_bytes,
    language_hint="hinglish",
    beam_size=1,
    vad_filter=True,
)

# Expected shape:
assert isinstance(result, TranscriptResult)
assert "bangalore" in result.text.lower() or "बैंगलोर" in result.text
assert result.language_detected in {"hi", "hinglish"}
assert 0.0 <= result.confidence <= 1.0
assert result.duration_s > 0.0

# Embedding into env observation (happens in env.py, not here):
# obs = replace(obs,
#     last_transcript=result.text,
#     last_lang=result.language_detected if result.language_detected != "unknown" else goal.language,
#     last_confidence=result.confidence,
# )
```

### 8.3 Kannada round-trip TTS → ASR (demo Space self-test)

```python
from __future__ import annotations

from driftcall.audio.tts_kokoro import get_tts_engine
from driftcall.audio.asr_whisper import get_asr_engine

tts = get_tts_engine()
asr = get_asr_engine()

original_text = "Kempegowda airport ge taxi beku"
wav = tts.synthesize(text=original_text, language_code="kn", seed=42)
result = asr.transcribe(audio_bytes=wav, language_hint="kn")

# Round-trip fidelity (semantic, not exact — Kannada ASR has noise):
assert result.text != ""
assert result.language_detected in {"kn", "unknown"}
# Soft assertion: at least one keyword survives the round-trip.
assert any(tok in result.text.lower() for tok in ("kempegowda", "airport", "taxi"))
# Confidence floor for demo playback:
assert result.confidence > 0.3, f"Kannada round-trip confidence too low: {result.confidence}"
```

Additional integration-level flow (not a unit test, for orientation):

```
┌─ sim-caller ─┐   TTS      ┌─ env boundary ─┐   ASR      ┌─ env core ─┐
│ goal text    │──────────▶│ 16 kHz WAV bytes │──────────▶│ observation │
│ (GoalSpec    │           │ (bytes over HTTP │           │ (text,lang, │
│  .seed_utt)  │           │  in /step body)  │           │  conf)      │
└──────────────┘            └──────────────────┘           └─────────────┘
```

---

## 9. Open questions

1. **VAD filter confidence on Hinglish code-mix.** `vad_filter=True` uses silero-VAD trained primarily on European languages. Early smoke tests suggest it sometimes clips Hindi laterals ("ल", "न"). If this materially hurts R1 on Hinglish episodes during Phase C baseline eval, we may flip to `vad_filter=False` at the cost of ~10% slower decoding. Escalate to orchestrator after baseline runs in Batch C2.

2. **Kokoro voice pack A/B for Hinglish.** §4.3 documents `en_indian_female_1` as default and `hi_female_1` as fallback. We have no empirical data yet on which produces better judge perception in the demo. Decision deferred to demo rehearsal in Batch C5 — Person D to record both variants and pick by ear.

3. **Should the env return raw WAV bytes to the agent, or just the transcript?** Current design: transcript only (via `DriftCallObservation.last_transcript`). An argument for also returning WAV: the agent could self-re-transcribe with a different model. Counter: we want to lock the ASR-as-oracle contract for reward reproducibility. **Recommendation:** keep transcript-only. If overturned in review, `DriftCallObservation` gets a new optional `last_audio_b64: str | None` field and this doc + `models.md` both update.

4. ~~**Sample rate upgrade path.** 16 kHz is the minimum for Whisper-small; 24 kHz would sound better for TTS playback in the demo. Kokoro natively produces 24 kHz; we currently resample down. If Space CPU budget permits, we may expose 24 kHz for TTS output while ASR continues at 16 kHz — this costs 50% more bandwidth over HTTP. Deferred; do not implement until demo-polish sprint.~~ **RESOLVED (see §4.4).** v1 contract pins TTS output to 16 kHz and resamples inside `synthesize()` before WAV encoding via `torchaudio.functional.resample(tensor, orig_freq=24000, new_freq=16000, lowpass_filter_width=64)`. ASR never auto-resamples; non-16 kHz input raises `AudioDecodeError`. 24 kHz playback path is out of scope for hackathon ship and will not be added without a DESIGN.md §9 update.
