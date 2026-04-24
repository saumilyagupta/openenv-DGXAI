# audio_tests.md — Test Plan for `driftcall/audio/*.py`

**Owner:** Person B (Rewards & Tests), secondary: Person C (Training & Data — authored module)
**Target module:** `DRIFTCALL/docs/modules/audio.md` (sealed)
**Implements coverage for:** DESIGN.md §9 (Audio Pipeline, 9.1–9.4), §3.3 (Deployed Env Topology), §3.4 (Demo Topology), §4.1 (`DriftCallObservation.last_confidence`)
**Frameworks:** `pytest`, `hypothesis`, `soundfile`, `numpy`
**Status:** DRAFT — pending ≥ 1 fresh critic round (test-plan gate is lighter per CLAUDE.md §3.2 Batch D4)

---

## 0. Scope & Non-goals

`driftcall/audio/` contains two heavy model wrappers (`TTSEngine` over Kokoro-82M and `ASREngine` over faster-whisper-small), a frozen dataclass `TranscriptResult`, a frozen `AudioTrace` diagnostic record, and a shared exceptions module. This plan targets:

1. **Contractual behavior** — every signature, return shape, and invariant declared in `audio.md §2` through `§5`.
2. **Determinism** — byte-reproducibility of TTS under fixed `(text, voice_pack, seed, sample_rate_hz)`; greedy-decode stability for ASR.
3. **Error taxonomy** — every `AudioError` subclass in `audio.md §2.3` is exercised by at least one test that raises it.
4. **LRU behavior** — eviction, byte-cap, key-extension (seed, sample_rate_hz), cross-format discriminator (`fmt="numpy"` vs bytes).
5. **Voice-pack fallback chain** — `ta→hi`, `kn→hi`, `hi_male→hi_female`, `hi_female→en_indian_female`; catastrophic `hi+en` missing raises `ModelLoadError`.
6. **Trace emission** — `AudioTrace` round-trip through `trace_sink`, `degraded=True` propagation, absence-of-sink smoke.
7. **Language coverage** — Hindi, Tamil, Kannada, English, Hinglish end-to-end.
8. **Concurrency safety** — two engines × 10 parallel calls; GIL-release via CTranslate2.
9. **Import firewall** — `training.train_grpo` import does NOT pull `driftcall.audio.*` into `sys.modules`.
10. **NFC normalization** — every decoded `TranscriptResult.text` is asserted `unicodedata.normalize("NFC", text) == text`.

**Out of scope:** GPU kernel tests (CPU-only deploy per `audio.md §1`; GPU calls are mocked to raise `RuntimeError`, never CUDA OOM). Whisper WER thresholds on real Indic audio (that's `evaluation.md`'s turf). Cross-architecture FP non-determinism (pinned to x86_64 AVX2 per `audio.md §3.3`).

Every test below cites the clause in `docs/modules/audio.md` it covers via `audio.md §X.Y` in the docstring.

---

## 1. Unit tests

All unit tests live in `DRIFTCALL/tests/test_audio.py`. Fixtures (§5) are imported from `tests/conftest.py`. Import line under test:

```python
from driftcall.audio.tts_kokoro import (
    TTSEngine,
    VoicePack,
    VoicePackMapping,
    VOICE_PACKS,
    get_tts_engine,
)
from driftcall.audio.asr_whisper import (
    ASREngine,
    TranscriptResult,
    get_asr_engine,
)
from driftcall.audio.trace import AudioTrace, TraceSink
from driftcall.audio.errors import (
    AudioError,
    ModelLoadError,
    UnsupportedLanguageError,
    UnsupportedVoicePackError,
    AudioDecodeError,
    AudioTooLongError,
    TTSOutOfMemoryError,
)
```

### 1.1 `TTSEngine.synthesize` — 5 voice packs × 2 speakers

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U1 | `test_synthesize_hi_female_returns_riff_wav` | `wav[:4] == b"RIFF"`, `wav[8:12] == b"WAVE"`; `soundfile.info` yields `samplerate==16000`, `channels==1`; `duration > 0.5`. | audio.md §2.1, §4.4 |
| U2 | `test_synthesize_hi_male_returns_riff_wav` | Same as U1 with `voice_pack="hi_male_1"`; non-empty bytes; distinct bytes from U1 (different voice). | audio.md §4.3 |
| U3 | `test_synthesize_ta_female_returns_riff_wav` | Tamil phrase; RIFF/WAVE headers; 16 kHz mono; duration ≥ 0.5 s. | audio.md §4.3 ta row |
| U4 | `test_synthesize_kn_male_returns_riff_wav` | Kannada phrase; RIFF/WAVE headers; 16 kHz mono. | audio.md §4.3 kn row |
| U5 | `test_synthesize_en_indian_female_returns_riff_wav` | English phrase; RIFF/WAVE headers; 16 kHz mono. | audio.md §4.3 en row |
| U6 | `test_synthesize_hinglish_default_voice_is_en_indian_female` | Call with `language_code="hinglish"`, `voice_pack=None`; engine must pick `en_indian_female_1` per §4.3 default; same synth byte-equal to explicit call with that pack. | audio.md §4.3 hinglish row, §2.1 default rule |
| U7 | `test_synthesize_hinglish_allowed_voice_hi_female` | `voice_pack="hi_female_1"` accepted for hinglish; produces WAV. | audio.md §4.3 |
| U8 | `test_synthesize_language_code_default_resolves` | `voice_pack=None` for each of 5 `LanguageCode` values resolves to `VOICE_PACKS[code].default`. Parametrized over 5 codes. | audio.md §2.1 default rule |
| U9 | `test_synthesize_unsupported_language_raises` | `synthesize(text="x", language_code="ja")` (type-ignored cast) raises `UnsupportedLanguageError`. | audio.md §5 row 3 |
| U10 | `test_synthesize_disallowed_voice_pack_raises` | `synthesize(text="x", language_code="ta", voice_pack="hi_male_1")` raises `UnsupportedVoicePackError`. | audio.md §5 row 4, §2.1 voice_pack rule |

### 1.2 Resampling pipeline (24 kHz → 16 kHz)

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U11 | `test_synthesize_resamples_24k_to_16k_before_wav_encode` | Monkeypatch `torchaudio.functional.resample` with a spy. Call `synthesize(text="hello", language_code="en")`. Spy observed exactly once with kwargs `orig_freq=24000`, `new_freq=16000`, `lowpass_filter_width=64`. | audio.md §4.4 |
| U12 | `test_synthesize_output_samplerate_is_16000_in_riff_header` | Parse WAV bytes; RIFF header `nSamplesPerSec` (bytes 24–27, little-endian uint32) == 16000. | audio.md §4.4 |
| U13 | `test_synthesize_rejects_non_16k_sample_rate_hz` | `synthesize(..., sample_rate_hz=24000)` raises an error; v1 contract supports only 16 kHz. | audio.md §4.4 last para |
| U14 | `test_synthesize_resample_happens_before_wav_encode` | Patch `torchaudio.save`. Assert `save` is called with a tensor whose length implies 16 kHz (length_seconds × 16000 ± small tolerance), not 24 kHz. | audio.md §4.4 |

### 1.3 `TTSEngine.synthesize_to_gradio`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U15 | `test_synthesize_to_gradio_returns_int_ndarray_tuple` | Return value is `tuple` of length 2; `isinstance(result[0], int)`; `isinstance(result[1], np.ndarray)`; `result[1].dtype == np.float32`; `result[1].ndim == 1`. | audio.md §2.1 `synthesize_to_gradio` |
| U16 | `test_synthesize_to_gradio_sample_rate_is_16000` | Tuple first element equals 16000 exactly. | audio.md §2.1 |
| U17 | `test_synthesize_to_gradio_cache_disjoint_from_bytes_cache` | Call `synthesize(text=T, language_code="hi", seed=0)` then `synthesize_to_gradio(text=T, language_hint="hi", seed=0)`. Inspect cache internals: two distinct entries exist (byte-cache key and numpy-cache key with `fmt="numpy"` discriminator). | audio.md §2.1 "LRU cache … NOT shared" |
| U18 | `test_synthesize_to_gradio_default_voice_matches_hint` | `voice_pack=None` resolves to `VOICE_PACKS[language_hint].default`; numpy output non-empty. | audio.md §2.1 |

### 1.4 `ASREngine.transcribe` — 5 languages

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U19 | `test_transcribe_hindi_returns_populated_result` | Feed `hindi_brief_wav`; result is `TranscriptResult`, `text != ""`, `language_detected in {"hi", "hinglish", "unknown"}`, `0.0 <= confidence <= 1.0`, `duration_s > 0`. | audio.md §2.2, §3.6 |
| U20 | `test_transcribe_tamil_returns_populated_result` | Feed `tamil_brief_wav`; same assertions; `language_detected in {"ta", "unknown"}`. | audio.md §3.6 |
| U21 | `test_transcribe_kannada_returns_populated_result` | Feed `kannada_brief_wav`; same assertions; `language_detected in {"kn", "unknown"}`. | audio.md §3.6 |
| U22 | `test_transcribe_english_returns_populated_result` | Feed `english_brief_wav`; `language_detected in {"en", "unknown"}`. | audio.md §3.6 |
| U23 | `test_transcribe_hinglish_translates_hint_to_hi` | Monkeypatch faster-whisper model call; assert the kwarg passed is `language="hi"` (not `"hinglish"`) per `audio.md §3.6`. | audio.md §3.6 first bullet |
| U24 | `test_transcribe_text_is_nfc_normalized` | Decode any non-empty transcript; assert `unicodedata.normalize("NFC", result.text) == result.text`. Parametrized over all 5 language fixtures. | audio.md §4.1 row 1 |
| U25 | `test_transcribe_language_hint_none_autodetects` | `language_hint=None` on `english_brief_wav` returns non-empty text, `language_detected != "unknown"` on a clean clip. | audio.md §2.2 |

### 1.5 Empty-string-nonzero-confidence coercion

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U26 | `test_transcribe_empty_text_with_nonzero_whisper_confidence_coerced_to_zero` | Monkeypatch faster-whisper to return `segments=[Segment(text="", avg_logprob=-0.3)]`, `vad_dropped_all_segments=False`. Call `transcribe`. Assert `result.text == ""`, `result.confidence == 0.0`, `result.language_detected` is the whisper-reported value (not `"unknown"`). | audio.md §3.5 branching |
| U27 | `test_transcribe_empty_text_vad_silent_path` | Monkeypatch to simulate `vad_dropped_all_segments=True`. Assert `result == TranscriptResult(text="", language_detected="unknown", confidence=0.0, duration_s=<clip>)`. | audio.md §7.4, §3.5 |
| U28 | `test_transcribe_empty_text_branch_emits_degraded_trace` | With `trace_sink` bound, non-VAD empty-with-confidence branch emits `AudioTrace(degraded=True)`. | audio.md §3.5, §3.8 |

### 1.6 `AudioDecodeError` on non-16kHz ASR input

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U29 | `test_transcribe_rejects_24khz_wav` | Build a RIFF WAV with `nSamplesPerSec=24000` (minimal mono 16-bit stream). `transcribe(...)` raises `AudioDecodeError` with message containing `"must be 16 kHz"`. | audio.md §4.4 ASR resampling policy |
| U30 | `test_transcribe_rejects_48khz_wav` | Same as U29 with 48000. | audio.md §4.4 |
| U31 | `test_transcribe_rejects_non_wav_non_pcm_bytes` | Feed `b"\x00\x01\x02\x03" * 100` (no RIFF, no float32-PCM magic). Raises `AudioDecodeError`. | audio.md §4.4, §5 row 5 |
| U32 | `test_transcribe_rejects_mp3_bytes` | Feed bytes starting with `b"ID3"` (MP3 tag). Raises `AudioDecodeError` — no ffmpeg in image. | audio.md §4.4 |
| U33 | `test_transcribe_accepts_raw_float32_pcm_16k` | Pre-generate float32 mono PCM at 16 kHz (no RIFF); transcribe accepts (demo mic path). | audio.md §4.4 second paragraph |

### 1.7 Voice-pack fallback chain

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U34 | `test_fallback_ta_female_missing_to_hi_female` | Monkeypatch `_voice_pack_available("ta_female_1")` to return False. Call `synthesize(text="வணக்கம்", language_code="ta")`. Engine uses `hi_female_1` under the hood; `AudioTrace.degraded == True`, `fallback_from == "ta_female_1"`. | audio.md §4.3.1 row 1 |
| U35 | `test_fallback_kn_male_missing_to_hi_female` | Analogous with `kn_male_1` → `hi_female_1`; trace `degraded=True, fallback_from="kn_male_1"`. | audio.md §4.3.1 row 2 |
| U36 | `test_fallback_hi_male_missing_to_hi_female` | `hi_male_1` missing → `hi_female_1`; trace flagged. | audio.md §4.3.1 row 3 |
| U37 | `test_fallback_hi_female_missing_to_en_indian_female` | `hi_female_1` missing → `en_indian_female_1`; trace flagged. | audio.md §4.3.1 row 4 |
| U38 | `test_fallback_catastrophic_hi_and_en_missing_raises` | Both `en_indian_female_1` and `hi_female_1` absent at warmup → `get_tts_engine()` raises `ModelLoadError` with message containing `"no usable voice pack for hi or en"`. | audio.md §4.3.1, §5 catastrophic row |
| U39 | `test_fallback_activated_at_synth_time_not_warmup` | Warmup succeeds (logs WARN) when only Indic pack missing; fallback evaluated on `synthesize()` call, not at `warmup()`. | audio.md §4.3.1 "evaluated at synthesize() call time" |
| U40 | `test_warmup_logs_warn_on_missing_indic_pack` | `caplog` captures WARN record mentioning missing pack name; warmup returns normally. | audio.md §4.3.1 warmup policy |

### 1.8 LRU cache eviction at 256 entries

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U41 | `test_lru_cache_stores_first_entry` | First `synthesize(...)` populates cache; direct inspection shows 1 entry. | audio.md §3.4 |
| U42 | `test_lru_cache_hit_returns_byte_identical` | Call twice with identical `(text, voice_pack, seed, sample_rate_hz)`; bytes are `==`; second call latency-p50 under 10 ms using monkeypatched clock OR via `AudioTrace.cache_hit == True`. | audio.md §3.4, §7.7 |
| U43 | `test_lru_cache_key_includes_seed` | `synthesize(text="hi", seed=0)` and `synthesize(text="hi", seed=1)` produce two distinct cache entries (no collision). | audio.md §3.4 key-extension rationale |
| U44 | `test_lru_cache_key_includes_sample_rate_hz` | Attempt two syntheses differing only in `sample_rate_hz` (once v2 supports 24k; for v1 this is asserted via cache-key extraction logic). | audio.md §3.4 |
| U45 | `test_lru_cache_evicts_at_256_entries` | Force-insert 257 distinct `(text_i)` syntheses (mock the engine to return tiny bytes). After the 257th insertion, cache size `<= 256`; the oldest key is gone. | audio.md §3.4 "Capacity. 256 entries" |
| U46 | `test_lru_cache_evicts_at_64mb_byte_cap` | Insert entries whose total `getsizeof` sum would exceed 64 MB; assert cache total byte size `<= 64 * 1024 * 1024` after final insert. | audio.md §3.4 byte-budget cap |
| U47 | `test_lru_cache_shared_across_sessions` | Two engine *consumers* (simulating two sessions) both hit the same singleton cache; second consumer sees `cache_hit=True`. | audio.md §3.4 "Process-wide singleton, GLOBAL" |
| U48 | `test_lru_cache_numpy_and_bytes_keys_disjoint` | `synthesize` and `synthesize_to_gradio` for identical (text, voice, seed) produce two cache entries (one keyed with `fmt="numpy"` discriminator). | audio.md §2.1 synthesize_to_gradio |

### 1.9 `AudioTrace` emission via `trace_sink`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U49 | `test_trace_emitted_on_synthesize` | Engine constructed with `trace_sink=deque.append`; call `synthesize`. Deque length == 1; record is `AudioTrace` with `op="synthesize"`, `cache_hit=False` (first call), `confidence is None`, non-empty `input_hash`, `ts_ist` parseable as ISO-8601. | audio.md §3.8 |
| U50 | `test_trace_emitted_on_transcribe` | ASR engine with `trace_sink`; call `transcribe`. Trace has `op="transcribe"`, `cache_hit=False` always, `confidence` ∈ [0,1], `degraded` bool. | audio.md §3.8, §2.2a |
| U51 | `test_trace_absence_smoke_no_crash` | `TTSEngine(trace_sink=None)`; `synthesize()` returns normally; no attribute errors, no deque references, zero-overhead path. | audio.md §3.8 "Default. trace_sink=None" |
| U52 | `test_trace_sink_exception_swallowed` | Sink raises `RuntimeError("broken")`. `synthesize()` still returns valid bytes. No exception propagates. | audio.md §3.8 "try/except Exception: pass" |
| U53 | `test_trace_cache_hit_flag_set_on_second_synth` | First synth: `cache_hit=False`. Second identical synth: `cache_hit=True`. | audio.md §2.2a, §3.4 |
| U54 | `test_trace_degraded_flag_on_voice_fallback` | Force fallback (U34); captured trace has `degraded=True`. | audio.md §4.3.1 |
| U55 | `test_trace_input_hash_is_blake2b_16byte` | Hash string length == 32 (hex of 16-byte digest); deterministic across calls with same input. | audio.md §3.8 Privacy, §2.2a |
| U56 | `test_trace_input_hash_does_not_leak_raw_text` | Trace record fields inspected; no substring of the input text appears anywhere (privacy invariant). | audio.md §3.8 Privacy |
| U57 | `test_trace_ts_ist_is_kolkata_timezone` | `ts_ist` parses to a tz-aware datetime in `Asia/Kolkata`. | audio.md §2.2a |
| U58 | `test_second_sink_after_singleton_warns` | Construct singleton via `get_tts_engine(trace_sink=S1)`. Second call `get_tts_engine(trace_sink=S2)` logs WARN "different sink" and returns original engine (S1 still wired). | audio.md §3.8 note |

**Unit test total: 58 cases.** (target ≥ 30 satisfied)

---

## 2. Property tests

Properties live in `DRIFTCALL/tests/test_audio_properties.py` using `hypothesis`. Strategies are registered in `tests/conftest.py` where shared.

### 2.1 Cache hit-rate invariants

**P1 — Repeated synth never decreases cache hit count.**
```
@given(text=st.text(min_size=1, max_size=200),
       voice=st.sampled_from(list(allowed_for("hi"))),
       seed=st.integers(0, 1000),
       repeat=st.integers(2, 10))
```
Call `synthesize` `repeat` times with identical arguments. Assert: after call k (k ≥ 2), the cache-hit trace count is exactly `k-1`. Monotonic; never regresses. Maps to `audio.md §3.4` determinism + cache semantics.

### 2.2 TTS determinism per `(text, voice_pack, seed, sample_rate_hz)`

**P2 — Byte-equality under identical inputs.**
```
@given(text=st.text(min_size=1, max_size=100),
       voice=st.sampled_from(list(all_voices)),
       lang=st.sampled_from(["hi", "ta", "kn", "en", "hinglish"]),
       seed=st.integers(0, 2**31 - 1))
```
Only valid `(lang, voice)` combos are materialised (`assume(voice in VOICE_PACKS[lang].allowed)`). Call `synthesize` twice in the same process with a cache-bypass flag. Assert `wav_a == wav_b` (byte-for-byte). Maps to `audio.md §3.3`, §7.8.

### 2.3 Transcript NFC-normalized

**P3 — Every decoded transcript is NFC-normalized.**
```
@given(fixture=st.sampled_from(["hindi_brief_wav", "tamil_brief_wav",
                                 "kannada_brief_wav", "english_brief_wav",
                                 "hinglish_brief_wav"]))
```
For each fixture bytes, `transcribe(...)` → `result`. Assert:
```python
import unicodedata
assert unicodedata.normalize("NFC", result.text) == result.text
```
Maps to `audio.md §4.1` text constraint, rules note.

### 2.4 Confidence ∈ [0, 1] across all inputs

**P4 — ASR confidence domain invariant.**
```
@given(audio=st.one_of(st.sampled_from(all_language_fixtures),
                       st.binary(min_size=44, max_size=200_000)
                         .map(_wrap_as_16k_wav_or_silence)))
```
For every bytes input that does not raise `AudioDecodeError`, `0.0 <= result.confidence <= 1.0`, and `result.confidence == 0.0` whenever `result.text == ""` (both VAD-silent and empty-with-coercion branches). Maps to `audio.md §4.1` confidence row, §3.5.

### 2.5 Empty-text ⇒ zero-confidence invariant (stronger than P4)

**P5 — Text-empty implies confidence-zero (bi-conditional on empty).**
```
@given(audio_bytes=_valid_16k_wav_strategy())
```
```python
r = asr.transcribe(audio_bytes, language_hint="hi")
assert (r.text == "") == (r.confidence == 0.0)
```
Bidirectional: empty ⇔ zero-confidence. Catches any regression where whisper's mean logprob leaks through on an empty decode. Maps to `audio.md §3.5`, §4.1 confidence row.

### 2.6 Duration bounds

**P6 — `duration_s` always clamped to `[0, max_duration_s]`.**
```
@given(max_d=st.floats(0.5, 30.0),
       clip_d=st.floats(0.1, 60.0))
```
Synthesize a silence WAV of length `clip_d`, transcribe with `max_duration_s=max_d`. Assert `0.0 <= r.duration_s <= max_d` and `r.duration_s == round(r.duration_s, 3)` (three-decimal contract). Maps to `audio.md §4.1` duration row, §7.3.

**Property test total: 6 properties.** (target ≥ 5 satisfied)

---

## 3. Integration tests

Integration tests live in `DRIFTCALL/tests/test_audio_integration.py`. They exercise real Kokoro and real faster-whisper; they are tagged `@pytest.mark.integration` and gated behind the `DRIFTCALL_RUN_INTEGRATION=1` env var in CI for slow runs, but default-enabled locally for `person-C` trusted dev. Fixtures are session-scoped so model load happens once per pytest process.

### 3.1 Full TTS → WAV bytes → Whisper → transcript round trip

**I1 — `test_e2e_english_roundtrip`**
Pipeline: `tts.synthesize(text="book a flight to delhi tomorrow", language_code="en") -> wav_bytes -> asr.transcribe(wav_bytes, language_hint="en") -> result`.
Assertions:
- `result.text != ""`
- `result.text` NFC-normalized
- `result.language_detected in {"en", "unknown"}`
- `result.confidence > 0.3`
- Keyword overlap: at least one of `{"book", "flight", "delhi", "tomorrow"}` in `result.text.lower()`.

Maps to `audio.md §8.1`, §8.3 round-trip example.

### 3.2 Hindi round-trip

**I2 — `test_e2e_hindi_roundtrip`**
Synthesize Hindi brief (`"कल दिल्ली की फ्लाइट बुक करें"`), transcribe. Assert keyword overlap on Devanagari tokens (at least 1 of `{"दिल्ली", "फ्लाइट"}`). `language_detected in {"hi", "hinglish", "unknown"}`. Confidence > 0.3.

### 3.3 Tamil round-trip

**I3 — `test_e2e_tamil_roundtrip`**
Synthesize Tamil brief; transcribe. Soft assertion per `audio.md §7.2` (Kannada-style: one keyword survives); Tamil TTS pack is `ta_female_1` (or fallback-chain to Hindi, which is acceptable as long as transcribe picks up recognizable phonemes).

### 3.4 Kannada round-trip

**I4 — `test_e2e_kannada_roundtrip`**
Per `audio.md §8.3` example: `"Kempegowda airport ge taxi beku"`. Assert at least one of `{"kempegowda", "airport", "taxi"}` survives in `result.text.lower()`. `language_detected in {"kn", "unknown"}`. `confidence > 0.3`.

### 3.5 Hinglish code-mix handling

**I5 — `test_e2e_hinglish_codemix`**
Synthesize `"Bhai Friday ko Bangalore jaana hai"` with `language_code="hinglish"`. Transcribe with `language_hint="hinglish"` (translated to `language="hi"` at whisper call site per `audio.md §3.6`). Assert:
- `result.text != ""`
- `result.language_detected in {"hi", "hinglish", "en", "unknown"}`
- `0.0 <= result.confidence <= 1.0`
- NO text-equality assertion (Risk 3 per `audio.md §7.1`).

Maps to `audio.md §7.1`, §3.6.

### 3.6 Concurrent session safety: 2 engines × 10 parallel calls

**I6 — `test_concurrent_two_engines_ten_calls_each`**
Construct two independent engine singletons (by tearing down and re-acquiring via a test-only module-reload fixture). From `concurrent.futures.ThreadPoolExecutor(max_workers=10)`, dispatch 10 `transcribe` calls on distinct 2-second fixture clips against each engine, then 10 `synthesize` calls against each. Assert:
- All 40 calls complete without exception.
- Wall-clock time for 10 parallel transcribes < 5 × single-call latency (GIL release via CTranslate2 per `audio.md §3.7`).
- No `TranscriptResult` field is `None`.
- No cache corruption: subsequent serial `synthesize` with a seen key hits the cache.

Maps to `audio.md §3.7`, §7.6.

### 3.7 Training-loop import firewall (structural)

**I7 — `test_training_import_does_not_pull_audio`**
Subprocess: `python -c "import training.train_grpo; import sys; assert 'driftcall.audio.tts_kokoro' not in sys.modules; assert 'driftcall.audio.asr_whisper' not in sys.modules"`. Subprocess exit code 0.

Maps to `audio.md §7.9`, §6.3.

### 3.8 Model-load failure recovery

**I8 — `test_get_tts_engine_recovers_after_transient_failure`**
First call: monkeypatch `kokoro.KPipeline` to raise `OSError("no network")` → `get_tts_engine()` raises `ModelLoadError` with original `OSError` as `__cause__`. Restore the patch. Second call succeeds and returns a working engine. Singleton did NOT cache the failure.

Maps to `audio.md §7.10`, §5 row 1.

---

## 4. Coverage target

**Target:** **100% line coverage** and **≥ 95% branch coverage** on:
- `driftcall/audio/tts_kokoro.py`
- `driftcall/audio/asr_whisper.py`
- `driftcall/audio/trace.py`
- `driftcall/audio/errors.py`
- `driftcall/audio/__init__.py`

Measured via:
```
python3 -m pytest tests/ \
  --cov=driftcall/audio \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=100
```

**Exclusions (via `# pragma: no cover`, each justified in the code next to the pragma):**
- GPU branches — the module is CPU-only per `audio.md §1`, §3.1, §6.1. Any `if torch.cuda.is_available():` guard carrying a CUDA-only fallback path is marked `# pragma: no cover — CPU-only deploy`. Mock-GPU tests ensure the guard raises `RuntimeError`, not `torch.cuda.OutOfMemoryError`, if inadvertently entered.
- `if __name__ == "__main__":` debug shims (none expected, but reserved).
- Platform-specific branches for non-x86_64 (`audio.md §3.3`: "Docker image pins to python:3.11-slim on amd64").

**What the pragma budget must NOT exempt:**
- Voice-pack fallback chain (§4.3.1) — every row covered.
- Empty-string-confidence branch (§3.5) — both sub-branches covered.
- ASR non-16kHz rejection (§4.4) — covered.
- `trace_sink` both `None` and `Callable` variants — covered by U51 (absence) and U49/U50 (presence).

**Branch budget:**

| Branch family | Count | Must be covered |
|---|---|---|
| Voice-pack fallback rungs | 5 | U34–U38 |
| Empty-text paths (VAD-silent vs empty-with-confidence) | 2 | U26, U27 |
| `trace_sink=None` vs set | 2 | U51 vs U49 |
| Language-hint hinglish translation branch | 1 | U23 |
| Sample-rate validation (16 kHz accept, else reject) | 2 | U12, U13 |
| Cache hit vs miss | 2 | U41, U42 |
| Fallback WARN vs catastrophic `ModelLoadError` | 2 | U40 vs U38 |

Total ~16 dominant branches; all hit. Remaining 5% allowance covers defensive `except Exception: pass` arms in trace emission (hit by U52) and mypy-narrowing `assert` statements.

---

## 5. Fixtures

All fixtures are defined in `DRIFTCALL/tests/conftest.py` and **shared with** `DRIFTCALL/docs/tests/deploy_env_space_tests.md` (same fixture names, same byte contents — generated once, cached to disk at `tests/fixtures/audio/`).

### 5.1 Pre-generated WAV fixtures (5 languages)

Generated by `tests/fixtures/generate_audio_fixtures.py` (run-once, checked into Git-LFS), producing 16 kHz mono 16-bit PCM WAV under `tests/fixtures/audio/`.

| Fixture | File | Content | Notes |
|---|---|---|---|
| `hindi_brief_wav` | `hindi_brief.wav` | `"कल दिल्ली की फ्लाइट बुक करें, सात हज़ार के अंदर"` synthesized via Kokoro `hi_female_1`, seed=0 | ~4.5 s duration; size ~144 KB |
| `tamil_brief_wav` | `tamil_brief.wav` | `"நாளை சென்னை விமானத்தை பதிவு செய்யவும்"` via `ta_female_1` (or fallback), seed=0 | ~3.5 s |
| `kannada_brief_wav` | `kannada_brief.wav` | `"Kempegowda airport ge taxi beku"` via `kn_male_1` (or fallback), seed=0 | ~3 s |
| `english_brief_wav` | `english_brief.wav` | `"Book a flight to Delhi tomorrow under seven thousand rupees"` via `en_indian_female_1`, seed=0 | ~4 s |
| `hinglish_brief_wav` | `hinglish_brief.wav` | `"Bhai Friday ko Bangalore jaana hai, budget dus hazaar"` via `en_indian_female_1`, seed=0 | ~4.5 s |

Fixture scope: **session**, read once from disk, returned as `bytes`. Example:
```python
@pytest.fixture(scope="session")
def hindi_brief_wav() -> bytes:
    return (FIXTURES_DIR / "hindi_brief.wav").read_bytes()
```

### 5.2 Engine singletons

| Fixture | Scope | Construction |
|---|---|---|
| `tts_engine` | `session` | `get_tts_engine(trace_sink=_test_ring_buffer.append)` once per pytest process |
| `asr_engine` | `session` | `get_asr_engine(trace_sink=_test_ring_buffer.append)` once per pytest process |

Teardown: engines persist for the pytest process; `_test_ring_buffer` is cleared between tests via an autouse `function`-scoped `clear_trace_ring` fixture.

### 5.3 Trace-sink fixtures

| Fixture | Scope | Value |
|---|---|---|
| `null_trace_sink` | `function` | `None` — used for absence-of-sink smoke tests (U51) |
| `ring_trace_sink` | `function` | `collections.deque(maxlen=100)`; fixture yields `deque.append`; tests assert on the deque directly (available via `request.getfixturevalue("ring_trace_sink_deque")`) |
| `ring_trace_sink_deque` | `function` | The underlying deque (for inspection) |
| `broken_trace_sink` | `function` | `Mock(side_effect=RuntimeError("broken"))` — used for U52 |

### 5.4 Helper builders

| Fixture / helper | Use |
|---|---|
| `build_wav_bytes(samplerate: int, duration_s: float, fill=0.0)` | Build arbitrary-samplerate silence WAV for U29 / U30 negative tests |
| `build_float32_pcm_16k(duration_s: float)` | Build raw float32 PCM at 16 kHz (no RIFF) for U33 |
| `fresh_module_reload()` | Context manager that reloads `driftcall.audio.tts_kokoro` & `asr_whisper` to tear down singletons for I6 concurrency tests |
| `mock_kokoro_missing_pack(pack_name: str)` | Monkeypatch `_voice_pack_available(pack_name)` → False for U34–U38 |

### 5.5 Shared-with-deploy-tests contract

The fixtures named in §5.1 and §5.2 **MUST** match exactly (by name and by bytes) the ones referenced in `docs/tests/deploy_env_space_tests.md`. Test-author for the deploy plan imports them from the same `conftest.py`. If the deploy plan requires additional WAV fixtures, they live alongside these under the same directory and follow the same naming convention (`<language>_<purpose>_wav`).

---

## 6. Rules compliance (CLAUDE.md + task brief)

- **pytest** — all tests use pytest (no unittest.TestCase).
- **CPU-only** — GPU code paths are excluded via `# pragma: no cover — CPU-only deploy`; any mock of a GPU call raises `RuntimeError("CUDA unavailable (mocked)")`, NEVER a `torch.cuda.OutOfMemoryError` — enforced in `tests/conftest.py`'s `mock_gpu_unavailable` autouse fixture.
- **NFC assertion** — U24 + P3 assert `unicodedata.normalize("NFC", t) == t` on every non-empty transcript; the integration tests I1–I5 also assert NFC via a shared `assert_nfc(result)` helper.
- **No audio bytes in logs** — test helpers scrub any `caplog` messages for base64-encoded WAV or raw PCM hex-dumps; a `no_audio_bytes_in_logs` autouse fixture greps every `caplog.records[i].message` for `RIFF`, `WAVE`, or base64-like blobs > 64 chars and fails the test if found.

---

## 7. Open questions

1. **ta_female_1 availability.** Per `audio.md §4.3.1`, the Tamil pack may be absent from the installed Kokoro bundle. The Tamil fixture (`tamil_brief_wav`) is generated via the fallback chain if the pack is missing, which means I3 tests the fallback path rather than the native Tamil pack. This is acceptable but logged. If a future Kokoro release ships Tamil natively, regenerate the fixture and rerun I3 for the native-pack sanity check.
2. **Silero-VAD determinism across hypothesis-generated clips.** P4 and P5 assume VAD behavior is deterministic. Silero is deterministic per `audio.md §3.3`, but shrinking in hypothesis may produce pathological clips. If flakes appear in CI, add `@settings(max_examples=50, deadline=None)` and pin a seed.
3. **Coverage of the `second_sink_after_singleton` WARN path (U58)** depends on how `get_tts_engine` internally logs. If the warning is via `warnings.warn`, switch from `caplog` to `pytest.warns`. Decision: confirm at implementation time in Batch C3.

---

**End of test plan. Author B sign-off pending; critic round to follow per CLAUDE.md §3.4.**
