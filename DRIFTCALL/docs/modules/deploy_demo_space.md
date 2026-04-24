# `docs/modules/deploy_demo_space.md` — Demo HF Space

**Owner:** D (Deploy & Story) · **Batch:** D3-Surface · **Depends on:** `docs/modules/audio.md`, `docs/modules/training.md`, `docs/modules/env.md`, `docs/modules/deploy_env_space.md`, `docs/modules/models.md`, `docs/modules/rewards.md` · **Cites:** DESIGN.md §3.4, §11.2, §15, §10.5

---

## 1. Purpose

This module specifies the **Demo HF Space** (`<team>/driftcall-demo`) — a Gradio 5.x application whose *only* job is to let a judge watch a before/after comparison of DriftCall agents handling a voice brief while schema drift fires mid-episode. It is the pitch surface described in DESIGN.md §15 (3-minute pitch + 2-minute Q&A) and the hardware plan in DESIGN.md §3.4.

Concretely the Space must:

1. Load the base **Gemma 4 E2B (4-bit)** model once and hold it resident.
2. Hot-swap between the base model and a **trained LoRA adapter** at `<team>/gemma-4-e2b-driftcall-lora` via a radio toggle, without restarting the process.
3. Accept **voice input from a Gradio `gr.Audio(sources=["microphone"])` component**, transcribe via the shared ASR singleton (`audio.md §2.2`), step the in-process `DriftCallEnv(audio_boundary_enabled=True)` (see `env.md`), generate a model reply, synthesize via `TTSEngine.synthesize_to_gradio` (`audio.md §2.1`) for low-latency playback, and stream a live trace panel showing action, tool response, drift event, and reward components.
4. Prefer **ZeroGPU** (Hugging Face's free, Ampere serverless, stateless GPU) and fall back to **A10G small** ($1.05/hr, ~$20 of the $30 hackathon budget) if ZeroGPU is unavailable or queue-rejected.
5. Complete a full round-trip turn (mic → transcript → env → model → TTS → audio playback) in **≤ 8 s on ZeroGPU warm, ≤ 12 s on A10G warm**, because that is the upper bound on judge attention span during the live pitch (DESIGN.md §15 "Before/After" beat).
6. Expose a **manual drift-injection toggle** so a judge can fire a drift pattern mid-turn (DESIGN.md §11.2 item 3) and watch the trained model adapt while the base model fails — the core visual of the pitch.

The env Space (`deploy_env_space.md`) is the OpenEnv-compliant **reward-grading** surface; this Space is the **storytelling** surface. The demo Space does **not** run `openenv validate` — that contract lives on the env Space. The demo Space is free to deviate (e.g., audio-enabled by default, mutable session state, direct model inference) because judges interact with it as a product, not as a judge-target.

---

## 2. Interface

### 2.1 Repository layout (`demo/` subdirectory, pushed as HF Space root)

```
demo/
├── app_gradio.py          # Gradio entrypoint; mounts UI; wires engines.
├── session.py             # DemoSessionState + per-tab session registry.
├── model_loader.py        # Base-model singleton + LoRA hot-swap.
├── drift_toggle.py        # Judge-initiated drift injection bridge into env.
├── trace_panel.py         # Live trace DataFrame renderer (model → Gradio).
├── pitch_assets/
│   ├── hindi_brief.wav    # The canned brief from DESIGN.md §15 (0:00 hook).
│   ├── tamil_brief.wav    # Backup for language switch.
│   └── judge_cheatsheet.md
├── README.md              # HF Space README (YAML front-matter; see §3.7).
├── requirements.txt       # Pinned runtime deps; see §6.
└── pre-requirements.txt   # ZeroGPU constraint file (see §3.6).
```

All signatures below are the **exact target**; `app_gradio.py` and its test harness depend on them and renames require a DESIGN.md-gated update first.

### 2.2 `demo/app_gradio.py` — entrypoint

```python
from __future__ import annotations

import gradio as gr
import spaces  # HF ZeroGPU decorator; no-op on non-ZeroGPU hardware.
from driftcall.audio import get_asr_engine, get_tts_engine, warmup_audio
from driftcall.env import DriftCallEnv
from driftcall.models import DriftCallObservation, DriftCallAction
from demo.session import DemoSessionState, get_session
from demo.model_loader import ModelLoader, CheckpointId
from demo.drift_toggle import DriftToggleBridge
from demo.trace_panel import TracePanelRenderer


def build_ui() -> gr.Blocks: ...
    # Constructs the Gradio Blocks graph. Pure function; idempotent.
    # Wires: mic input, checkpoint radio, drift-toggle dropdown,
    # transcript textbox, reward bar, trace dataframe, speaker output,
    # reset button. Returns the Blocks object.


@spaces.GPU(duration=60)  # ZeroGPU: request a 60-second GPU slice per call.
def infer_turn(
    audio_tuple: tuple[int, "np.ndarray"] | None,
    checkpoint: CheckpointId,
    manual_drift: str | None,
    session_id: str,
) -> tuple[str, "tuple[int, np.ndarray]", "pd.DataFrame", dict[str, float], str]:
    """Handle one mic-to-speaker turn.

    Args:
        audio_tuple: (sample_rate, float32 np.ndarray) from the mic component,
            or None if the user pressed Enter in the fallback text box.
        checkpoint: "base" | "trained" — which adapter to run this turn on.
        manual_drift: None if the judge did not force a drift; otherwise the
            drift pattern id (one of the 20 from DESIGN.md §6.3).
        session_id: UUID from browser gr.State; keys DemoSessionState registry.

    Returns:
        (transcript_text, (sr, wav_np), trace_df, reward_dict, status_msg)

    Contract:
        - Must return within `duration=60` ZeroGPU seconds or the frame is dropped.
        - On any exception the status_msg string carries the user-facing message
          and the other positional returns are safe defaults (empty string,
          1 second of silence at 16 kHz, empty DataFrame, empty dict).
        - Never writes to disk; never calls push_to_hub.
    """


def warmup_on_boot() -> None: ...
    # Called once on Space cold-start. Loads base model, runs TTS+ASR warmup,
    # triggers one dummy forward pass to page in CUDA kernels. ~45 s on A10G,
    # ~15 s on ZeroGPU because weights are cached on the node.


if __name__ == "__main__":
    warmup_on_boot()
    build_ui().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
```

### 2.3 `demo/model_loader.py` — base-model singleton + LoRA hot-swap

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from peft import PeftModel

CheckpointId = Literal["base", "trained"]


class ModelLoader:
    """Process-wide singleton. Holds the 4-bit base model in GPU memory and
    swaps LoRA adapters in/out via peft's adapter API.

    Construction is lazy (inside the first ZeroGPU-decorated call on the
    process) because instantiating on the import path blocks the Gradio
    server from binding its port; HF Spaces' health-check then kills the
    container.
    """

    def __init__(
        self,
        *,
        base_model_id: str = "unsloth/gemma-4-E2B-it-bnb-4bit",
        trained_adapter_id: str = "<team>/gemma-4-e2b-driftcall-lora",
        max_seq_length: int = 4096,
    ) -> None: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        checkpoint: CheckpointId,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int = 0,
    ) -> str:
        """Format `messages` via the Gemma chat template, run the correct
        adapter state, decode, strip the assistant prompt prefix, return
        the completion text.

        - `checkpoint="base"` → `model.disable_adapter()` context (peft).
        - `checkpoint="trained"` → `model.set_adapter("driftcall")` then
          enable; raises `TrainedAdapterMissingError` if the adapter was
          not mounted at boot.
        - Deterministic given (messages, checkpoint, seed, temperature>0).
        - Tokenization + decoding stay on GPU; only the final str is copied.
        """

    def is_trained_available(self) -> bool:
        """Has the `<team>/gemma-4-e2b-driftcall-lora` adapter been mounted?
        Used by the UI to grey out the "trained" radio option when the LoRA
        failed to download (§5.3)."""


def get_model_loader() -> ModelLoader:
    """Return the process-wide singleton. Instantiated on first call."""
```

### 2.4 `demo/session.py` — `DemoSessionState`

See §4.1 for the full dataclass. Public surface:

```python
def get_session(session_id: str) -> DemoSessionState: ...
    # Returns the session for this UUID or creates a fresh one with a new
    # DriftCallEnv(audio_boundary_enabled=True). Idempotent.

def reset_session(session_id: str) -> DemoSessionState: ...
    # Closes the env, discards the trace buffer, returns a fresh state.

def gc_sessions(max_idle_s: int = 900) -> int: ...
    # Evicts sessions idle > 900 s. Called from a background thread every
    # 60 s (§3.5 session TTL). Returns the count evicted.
```

### 2.5 `demo/drift_toggle.py` — judge-initiated drift

```python
class DriftToggleBridge:
    """Bridges the UI dropdown to the env's drift_injector queue.

    Only one manual drift may be queued per turn; extra presses before the
    next env.step() are coalesced (keep the most recent). See §7.3 edge case.
    """

    def queue(self, session_id: str, pattern_id: str | None) -> None: ...
    def consume(self, session_id: str) -> str | None: ...
        # Called by infer_turn() right before env.step(); returns the queued
        # pattern id once then clears. Invariant: same pattern never fires
        # twice from the same queue() call.
```

### 2.6 `demo/trace_panel.py` — live trace DataFrame

```python
def render_trace(
    state: DemoSessionState,
) -> "pd.DataFrame":
    """Flatten state.episode_trace into a 5-column DataFrame:
    [turn_idx, actor, action_or_event, tool_response_preview, reward_delta].
    `actor ∈ {"user","agent","env","drift","reward"}`. Never mutates state.
    """
```

---

## 3. Behavior spec

### 3.1 Hardware preference order

1. **ZeroGPU (primary).** README YAML front-matter declares `hardware: zero-gpu`. The process uses `@spaces.GPU(duration=60)` on `infer_turn`. Cold starts acquire a GPU slice only for the duration of one inference; between calls the process runs on CPU and the weights stay cached on the node (HF ZeroGPU model). This makes the $0 hardware line in DESIGN.md §3.5 viable.
2. **A10G small (fallback).** If ZeroGPU queue-rejects twice in a row during warmup, D switches the README YAML to `hardware: a10g-small` and redeploys. A10G is stateful: the base model stays resident, so `@spaces.GPU` is a no-op (the `spaces` package is still importable on A10G and its decorator is a pass-through). Budget: ≤ 20 hours of A10G = ~$20 of the $30 cap.
3. **Never CPU.** The demo loses its punch below ~10 tok/s; CPU generation on Gemma 4 E2B is ~1 tok/s. If both ZeroGPU and A10G are unavailable, **abort deployment** — the pitch reverts to a pre-recorded video (see `risk_book.md`).

### 3.2 Model hot-swap (base ⇄ trained)

Both adapters share a single base model forward graph. This matters for memory: the A10G small has 24 GB; Gemma 4 E2B 4-bit is ~2.5 GB; mounting two separate base+adapter copies would near-OOM. Instead:

1. Boot path: load base 4-bit → `PeftModel.from_pretrained(model, "<team>/gemma-4-e2b-driftcall-lora", adapter_name="driftcall")`.
2. "base" checkpoint: `with model.disable_adapter(): generate(...)`.
3. "trained" checkpoint: `model.set_adapter("driftcall"); model.enable_adapter_layers(); generate(...)`.
4. Enable/disable is a pointer-flip on the peft wrapper — microsecond-scale — so the radio toggle has no perceptible latency.

When the trained adapter cannot be downloaded (e.g., pre-training, repo 404), `is_trained_available()` returns `False`, the UI greys out the "trained" option, and the default selection is "base". The Space still runs; it becomes a baseline-only demo (recovery mode).

### 3.3 Session state and per-tab isolation

Gradio 5.x supports `gr.State` for per-session values. We use it to hold only a `session_id: str` (UUID); all mutable state lives in a process-wide registry keyed by that UUID (`session.py`). This pattern is necessary because the `DriftCallEnv` instance is not pickle-safe for `gr.State` (it holds open HTTP clients to mock vendors via `env.md`) and because multiple judges hitting the public URL simultaneously must not share one env.

Registry rules:

- **Max concurrent sessions:** 10. The 11th request receives a polite "demo at capacity — please wait" message and the turn is aborted gracefully. This cap keeps A10G memory headroom safe (10 active episodes ≈ 2 GB of KV cache).
- **Idle TTL:** 900 s. A background `threading.Timer` loop calls `gc_sessions(900)` every 60 s.
- **Cross-tab isolation:** Two judges on the same IP get different `gr.State` UUIDs. There is zero cross-session leakage.

### 3.4 Trace streaming

The trace DataFrame is built on every `infer_turn` from the session's accumulated `episode_trace` (§4.1) and returned to a `gr.DataFrame(wrap=True, max_height=400)` component. No SSE / websocket is required — Gradio's native per-call return is sufficient because a single turn is self-contained. In pitch mode the latency budget (§3.6) means the trace appears within 1-2 s of the user finishing speaking.

The trace panel is **read-only on the UI side**. It is not editable, not filterable, not paginated. Those features would introduce state bugs and are not in the pitch flow.

### 3.5 Reset semantics

Two reset paths:

1. **Soft reset ("New turn")** — user pressed the mic again; `infer_turn` calls `env.step(action)` in the existing episode. `episode_trace` grows.
2. **Hard reset ("New episode" button)** — calls `reset_session(session_id)`, which instantiates a fresh `DriftCallEnv(audio_boundary_enabled=True)` with a new `reset()` call, clears `episode_trace` to `[]`, and clears the audio output widget. The checkpoint radio is **not** reset (we want judges to toggle within an episode to compare).

### 3.6 Latency budget (per turn)

| Stage | Budget on ZeroGPU warm | Budget on A10G warm |
|---|---|---|
| Mic upload (Gradio client → server) | 0.3 s | 0.3 s |
| ASR (`transcribe()`, `faster-whisper-small int8`, 30 s clip max) | 0.8 s | 0.8 s |
| Env step (mock vendor call + drift injector) | 0.1 s | 0.1 s |
| Model generate (Gemma 4 E2B 4-bit, 256 tokens) | 4.5 s | 2.0 s |
| TTS (`synthesize_to_gradio`, 200 char) | 0.8 s | 0.8 s |
| DataFrame render + network return | 0.5 s | 0.5 s |
| **Total** | **≈ 7.0 s** | **≈ 4.5 s** |

ZeroGPU cold-start (first turn after idle) is ~20-30 s because of weight paging; `warmup_on_boot()` mitigates this on process start, but a second idle-wake is unavoidable. The pitch mitigates this by **never letting the demo go idle mid-pitch** — D keeps it warm by firing a dummy turn every 45 s during pre-pitch.

The `pre-requirements.txt` file pins `torch==2.4.0+cu121` and `cuda-python==12.1.x` to match the ZeroGPU node's CUDA runtime, avoiding a 4-minute `torch` rebuild on every cold start (the most common ZeroGPU gotcha in 2026).

### 3.7 README YAML front-matter (HF Space config)

```yaml
---
title: DriftCall Demo
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.8.0
app_file: app_gradio.py
pinned: true
license: apache-2.0
hardware: zero-gpu   # fallback: a10g-small
models:
  - <team>/gemma-4-e2b-driftcall-lora
tags:
  - openenv
  - indic
  - voice
  - grpo
  - drift
short_description: Voice-first Indic RL environment — before/after a schema-drift-trained Gemma 4 E2B.
---
```

### 3.8 Manual drift-injection protocol

`DriftToggleBridge.queue(session_id, pattern_id)` puts a single pattern id (one of the 20 from DESIGN.md §6.3) into the per-session queue. Before the next `env.step()`, `infer_turn` calls `bridge.consume(session_id)` and passes the result to `env.step(action, force_drift_pattern=...)` (see `env.md` and `drift_injector.md`). If two patterns are queued (judge double-clicks) the most recent wins. If the queue is empty the drift injector falls back to its normal probabilistic trigger.

The queued pattern is **shown in the trace panel** with `actor="drift"` and `action_or_event="manual:<pattern_id>"` so the judge sees their own action reflected.

### 3.9 Network hiccup recovery

Gradio 5's built-in client reconnects on short network drops; if the server response is lost mid-stream the user sees a transient "stream interrupted" toast and the turn is dropped (no partial trace, no partial audio). The session state is not mutated until `env.step()` returns successfully — atomic at the turn level.

---

## 4. Data structures

### 4.1 `DemoSessionState`

```python
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from driftcall.env import DriftCallEnv
from driftcall.models import DriftCallObservation
from demo.model_loader import CheckpointId


@dataclass(frozen=False)  # intentionally mutable — session-scoped accumulator
class TraceRow:
    turn_idx: int
    actor: Literal["user", "agent", "env", "drift", "reward"]
    action_or_event: str
    tool_response_preview: str   # first 120 chars; full payload in env logs
    reward_delta: float          # cumulative-delta since previous row


@dataclass(frozen=False)
class DemoSessionState:
    """Per-browser-tab state for the demo. Stored in a process-wide registry
    keyed by UUID in gr.State."""

    session_id: str
    env: DriftCallEnv
    last_observation: DriftCallObservation | None = None
    episode_trace: list[TraceRow] = field(default_factory=list)
    audio_buffer: list[bytes] = field(default_factory=list)   # last N turn WAVs, ring of 8
    current_checkpoint: CheckpointId = "base"
    turn_idx: int = 0
    created_at_ms: int = 0       # epoch ms; for TTL sweep
    last_activity_ms: int = 0
```

Mutability is deliberate: a frozen dataclass would force a full rebuild per turn, doubling memory churn. The accepted tradeoff is that **only `session.py` writes to these fields** — every other module receives `DemoSessionState` as read-only.

### 4.2 Adapter registry (in `ModelLoader`)

```python
{
    "base":    None,                  # uses model.disable_adapter()
    "trained": "driftcall",           # peft adapter_name mounted at boot
}
```

One and only one adapter is mounted. If future iterations want to A/B two trained checkpoints simultaneously (Stage-1 vs Stage-3), the dict grows to three keys; the UI radio gains a third option. No code other than `model_loader.py` knows the adapter names.

### 4.3 Trace DataFrame schema (output of `render_trace`)

| Column | Type | Notes |
|---|---|---|
| `turn_idx` | `int` | Monotone per episode; resets on hard-reset. |
| `actor` | `str` | `"user" | "agent" | "env" | "drift" | "reward"`. |
| `action_or_event` | `str` | Short human-readable action label (e.g., `"CALL_TOOL airline.search"`). |
| `tool_response_preview` | `str` | First 120 chars, ellipsized, `""` when not applicable. |
| `reward_delta` | `float` | Change in total reward vs previous row; 0.0 if not a reward row. |

---

## 5. Error modes

Every exception surfaces as a `status_msg` in the `infer_turn` return tuple — the UI never crashes; instead the status banner turns amber and the other outputs carry safe defaults (empty string, 1 s of silence, empty DataFrame, empty reward dict).

| # | Error | Raised by | User-facing `status_msg` | Recovery |
|---|---|---|---|---|
| 5.1 | `ZeroGPUUnavailableError` — `@spaces.GPU` request rejected (queue full, rate-limited) | ZeroGPU runtime | "GPU queue busy; retrying in 5 s…" | Single automatic retry with 5 s backoff. Second rejection → status "GPU unavailable; the demo is running on CPU and will be slow" and proceed with `device_map="cpu"` (still works, just >30 s/turn). Third consecutive rejection → triggers the A10G-fallback redeploy workflow (§3.1). |
| 5.2 | `TrainedAdapterMissingError` — LoRA download failed at boot or adapter file corrupt | `ModelLoader.__init__`, raised from `generate()` | "Trained adapter unavailable; showing base model only." | The `trained` radio option is greyed out at UI build time via `is_trained_available()`. If the user still selects it (race condition), fall back to `base` silently and set status as above. |
| 5.3 | `MicPermissionDeniedError` — browser denied mic access | Gradio client-side | Gradio's built-in banner; our code sees `audio_tuple=None` and does **not** treat it as error | If `audio_tuple is None` **and** the fallback textbox is empty, `status_msg="No audio received; press mic or type a brief."` and return safe defaults. |
| 5.4 | `torch.cuda.OutOfMemoryError` during `generate()` | `ModelLoader.generate` | "GPU out of memory this turn; reducing context and retrying." | Catch; empty KV cache (`torch.cuda.empty_cache()`); retry once with `max_new_tokens=128` and truncate the oldest turn from `messages`. A second OOM fails the turn with the same banner. |
| 5.5 | `CheckpointMismatchError` — LoRA was trained on a different `base_model_id` than the one loaded | `PeftModel.from_pretrained` at boot | "Model-adapter mismatch; trained variant disabled." | Treated as (5.2). The base model runs. Log to Space logs; D investigates post-pitch. Caused by upstream Unsloth re-publishing a base-model repo with a new hash (real bug seen Apr 2026). |
| 5.6 | `AudioDecodeError` from ASR singleton | `get_asr_engine().transcribe` | "Could not decode mic audio; please try again." | Safe defaults; no state mutation. |
| 5.7 | `SessionCapacityError` — > 10 concurrent sessions | `get_session` | "Demo at capacity — try again in a minute." | Turn aborted; no session created. |
| 5.8 | `EnvStepError` — env raised (vendor 500, invalid action) | `DriftCallEnv.step` | "Env rejected action: {short reason}; episode unchanged." | The trace panel shows the rejected action with `actor="env"`, reward_delta=0. No side effects. |
| 5.9 | `TimeoutError` — the `@spaces.GPU(duration=60)` budget exhausted | ZeroGPU runtime | "Turn timed out after 60 s — the model was slow; try again." | Gradio displays the banner; session state is not mutated because the env step did not complete. |

No error silently degrades: every degraded turn is reflected in the status banner **and** logged to `app_gradio.py`'s stderr (HF Spaces captures this for post-mortem).

---

## 6. Dependencies

### 6.1 Runtime Python packages (`requirements.txt`)

| Package | Pinned | Why |
|---|---|---|
| `gradio==5.8.0` | ✓ | UI. Pin to 5.8 (2026-04 LTS) — 5.9 broke `gr.Audio(sources=...)` behavior. |
| `spaces>=0.30.0` | ✓ | `@spaces.GPU` decorator; no-op on non-ZeroGPU hardware. |
| `unsloth==2026.4.post1` | ✓ | Base model loader. Must match `training.md` §6. |
| `peft>=0.11.0,<0.13` | ✓ | LoRA adapter mount + hot-swap. |
| `transformers>=4.44,<4.47` | ✓ | Gemma 4 chat template. |
| `torch==2.4.0` | ✓ | ZeroGPU node's CUDA 12.1 baseline (see `pre-requirements.txt`). |
| `numpy<2.0` | ✓ | Unsloth still pins <2.0 as of 2026-04. |
| `faster-whisper==1.0.3` | ✓ | ASR singleton per `audio.md` §6. |
| `kokoro>=0.8.0` | ✓ | TTS singleton per `audio.md` §6. |
| `huggingface_hub>=0.24` | ✓ | LoRA download at boot. |
| `pandas>=2.1` | ✓ | Trace DataFrame. |

The demo Space **reuses the same audio stack versions as the env Space** (`audio.md` §6). Divergence here would cause spectral drift between training-time audio (if we ever wire it) and demo-time audio.

### 6.2 Internal module dependencies

| Imported from | What the demo uses |
|---|---|
| `driftcall.audio` | `get_tts_engine`, `get_asr_engine`, `TTSEngine.synthesize_to_gradio`, `ASREngine.transcribe`, `AudioTrace` (optional trace_sink). `audio.md` §1 confirms demo use of `synthesize_to_gradio` and §4.2a confirms `AudioTrace`. |
| `driftcall.env` | `DriftCallEnv(audio_boundary_enabled=True)`, its `reset`, `step`, `close`. Uses the same in-process env as the FastAPI surface, minus the network boundary. |
| `driftcall.models` | `DriftCallAction`, `DriftCallObservation`, `EvalReport` (for the final-report tab). |
| `driftcall.rewards` | Reward-dict shape (for the reward bar); never imported for computation — the env does that. |
| `driftcall.drift_injector` | `DRIFT_PATTERN_IDS` (Literal type alias) — to populate the manual-drift dropdown values. |
| Hugging Face Hub | `<team>/gemma-4-e2b-driftcall-lora` (runtime download). |

### 6.3 System dependencies (baked in image via `Dockerfile` if self-hosted; installed on HF Spaces via system packages)

- `ffmpeg` — Gradio audio upload encoding.
- `libsndfile1` — soundfile read/write (TTS/ASR path).
- CUDA 12.1 + cuDNN 9.x — inherited from the ZeroGPU / A10G base image; no manual install.

No other system deps. Matches `audio.md` §6 exactly.

---

## 7. Edge cases

### 7.1 Two judges record simultaneously on ZeroGPU (queue serialization)

ZeroGPU serializes GPU slices across *all* tenants on a node, not just within a Space. If two judges fire `infer_turn` inside the same 60 s window, the second call blocks behind the first. The UX is: judge B sees the status banner "GPU queue busy; retrying in 5 s…" and, once the first turn returns, automatically proceeds. No frames are dropped on the Gradio side because each call is its own decorated invocation. Worst observed latency during an onsite load test (DESIGN.md §16.B eval): 14 s P95 with 3 concurrent tabs. Still within the pitch's 8 s soft budget for a solo judge and acceptable for a crowded booth.

### 7.2 Network hiccup mid-stream

Gradio 5's transport uses a single POST for `infer_turn`; partial responses are impossible. A dropped connection leads to: (a) client auto-retries within 10 s, (b) if the retry fails, the user sees a "connection lost" toast and the turn is abandoned — no trace row is appended, no audio plays. Because all mutations happen inside the server-side `infer_turn` and are committed only on successful return, the session is internally consistent even when the client never sees the response. Stale `last_activity_ms` is fine; the TTL sweep handles it.

### 7.3 Manual drift toggle fires out of the drift injector's scheduled window

`DriftToggleBridge.queue()` records the pattern regardless of whether the injector would have fired this turn or not; `drift_injector.md` §3.2 explicitly supports `force_drift_pattern: str | None` on `step()` as an override. Two consequences:

1. **Double-fire prevention.** If the injector was going to fire pattern X on this turn *and* the judge pressed pattern Y, pattern Y wins (judge intent > RNG). The trace records both: one `drift` row for the manual pattern, no row for the suppressed probabilistic fire.
2. **Stacking.** If the judge presses pattern Y twice before the next step, the second press overwrites the first (§3.8 coalescence). Intentional — the dropdown reflects the *queued* pattern and judges have seen this in UX testing.

### 7.4 Trained LoRA file missing on HF Hub at boot

Boot path: `ModelLoader.__init__` catches `huggingface_hub.utils.EntryNotFoundError` or `HTTPError(404)`, logs `"LoRA download failed: {reason}"`, and sets an internal `_trained_available = False`. `is_trained_available()` returns False. `build_ui()` inspects this and passes `choices=["base"]` instead of `["base", "trained"]` to the radio, with a warning label: "Trained adapter unavailable at boot — showing base only." The Space still launches; the pitch degrades to a verbal comparison.

### 7.5 A10G idle timeout

HF A10G Spaces sleep after ~30 min of no traffic (configurable via "Sleep time" in Space settings; we set it to the maximum for pitch day). Waking takes 60-90 s because the base model is re-downloaded to the new node and re-paged into GPU. During pre-pitch D runs `warmup_on_boot()` via a hidden keepalive request every 5 min from a small script on their laptop. Documented in `pitch_demo.md` §3.

### 7.6 Whisper mis-detects language

Hinglish mic input sometimes decodes with `language_detected="en"` and a broken transcript (per `audio.md` §3.5). The trace panel shows the *decoded* text verbatim in the `actor="user"` row; the agent's subsequent failure (if any) is attributed in-trace to the transcript, not the model. This matches the honest-demo rule: judges see the real ASR weakness, and the fix (Sarvam ASR) is in the Q&A answer per DESIGN.md §15 Q5.

### 7.7 `gr.State` UUID collision after server restart

After an HF Space container restart, in-memory session registry is wiped but Gradio's `gr.State` persists across restart via sticky session cookies in some browsers (rare). `get_session(session_id)` treats a stale UUID as a fresh session — it allocates a new `DemoSessionState` and silently takes over the UUID. No error shown. The user sees an empty trace panel as if they just arrived.

---

## 8. Examples

### 8.1 Launch command (local dev, simulating the Space runtime)

```bash
cd DRIFTCALL/demo
# For local dev the `spaces` package's @GPU decorator is a no-op, so the
# code runs on whatever CUDA is visible (falls back to CPU otherwise).
uv pip install -r requirements.txt
HF_TOKEN=hf_xxx python app_gradio.py
# → Gradio server on http://0.0.0.0:7860
# → fetches <team>/gemma-4-e2b-driftcall-lora on boot
```

For production (HF Space), no launch command is invoked by D — the Space platform auto-detects `app_file: app_gradio.py` from the README front-matter and runs it.

### 8.2 Full demo flow (mic → env → model → TTS)

Judge arrives at the Space URL, clicks the mic, speaks the Hindi brief from DESIGN.md §15 for 6 seconds, then releases.

```text
turn_idx=1, actor=user,  action_or_event="[hi] Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad", preview="", reward_delta=0.00
turn_idx=1, actor=agent, action_or_event="CALL_TOOL airline.search(from=DEL,to=BLR,date=2026-04-25,max_price=8000,after=18:00)", preview="", reward_delta=0.00
turn_idx=1, actor=env,   action_or_event="200 OK", preview='{"flights":[{"id":"6E-2345","price":7200,"depart":"19:30"}]}', reward_delta=0.00
turn_idx=2, actor=drift, action_or_event="auto: schema_rename_price_to_total_fare_inr", preview="", reward_delta=0.00
turn_idx=2, actor=agent, action_or_event="CALL_TOOL airline.confirm(flight_id=6E-2345,price=7200)", preview="", reward_delta=0.00
turn_idx=2, actor=env,   action_or_event="400 Bad Request", preview='{"error":"unknown_field:price; did you mean total_fare_inr?"}', reward_delta=0.00
turn_idx=2, actor=agent, action_or_event="SPEAK hi \"The price field appears to have changed — using total_fare_inr. Confirming 6E-2345 at ₹7,200.\"", preview="", reward_delta=0.00
turn_idx=2, actor=agent, action_or_event="CALL_TOOL airline.confirm(flight_id=6E-2345,total_fare_inr=7200)", preview="", reward_delta=0.00
turn_idx=2, actor=env,   action_or_event="200 OK",  preview='{"pnr":"XQ9R2A","status":"confirmed"}', reward_delta=0.00
turn_idx=2, actor=reward, action_or_event="R1=1.0 R2=1.0 R3=1.0 R4=1.0 R5=0.0 → total=1.0", preview="", reward_delta=+1.00
```

Speaker plays back Kokoro-synthesized Hindi ("The price field appears to have changed…"). The trained-model run completes in ~7 s end-to-end. Judge toggles the radio to "base", presses the mic again with the same brief — the base model returns a `KeyError: 'price'` equivalent, reward drops to 0. The side-by-side is the pitch.

### 8.3 Judge presses drift-toggle mid-episode

Pre-conditions: episode is on turn 3, checkpoint is "trained", injector has not yet fired a drift this episode.

```text
1. Judge picks "T&C update — free-cabin baggage allowance rewritten" from the drift dropdown.
2. DriftToggleBridge.queue(session_id, "airline.baggage_tnc_rewrite")  [synchronous]
3. Judge presses mic, speaks: "Add one checked bag."
4. infer_turn() runs:
   a. ASR → "Add one checked bag."
   b. bridge.consume(session_id) → "airline.baggage_tnc_rewrite"
   c. env.step(action=SPEAK→CALL_TOOL airline..., force_drift_pattern="airline.baggage_tnc_rewrite")
   d. drift_injector fires the `airline.baggage_tnc_rewrite` T&C patch (free-cabin allowance 7 kg → 5 kg) on the airline vendor mid-step; a `side_channel_notice` is queued per drift_injector.md §3.4 for attachment to the next airline tool response.
   e. agent's CALL_TOOL airline.addBag returns 409 "baggage_limit_exceeded (new limit 5kg from 7kg)" with the side-channel notice attached on `response["_notice"]`.
   f. agent emits a SPEAK row: "The T&Cs updated — free cabin is now 5 kg. Do you want to proceed under the new limit?"
   g. R2 drift-detection reward fires: the agent cited the new limit field name within 2 turns.
5. Trace panel renders one row with actor="drift", action_or_event="manual:airline.baggage_tnc_rewrite", pinned at the top of turn 3.
```

This is the single most pitch-valuable interaction: judge agency + trained-model adaptation + real reward trace, all in ~8 s.

---

## 9. Open questions

1. **ZeroGPU cold-start visible to the judge?** First-turn cold-start is ~15 s even after `warmup_on_boot`. Options: (a) pre-fire a hidden turn from D's laptop every 45 s during pre-pitch (manual keepalive); (b) show a "warming up — 15 s" progress bar on the UI. (a) preferred for pitch smoothness; (b) as backup. **Decision needed from D before pitch day.**
2. **Should the demo Space accept text input as a fallback when mic is denied?** Currently yes (a `gr.Textbox` is wired next to the mic as a secondary input — §2.2 takes `audio_tuple: tuple | None`), but some judges may find it visually distracting from the "voice-first" story. **Ask the Meta/HF judge panel lead during onsite practice round.**
3. **Third checkpoint option (Stage-1 vs Stage-3) for deeper story?** The pitch is tight at 3 min; a third radio option would add narrative complexity for marginal value. Defer unless C's training runs produce a clear Stage-1-vs-Stage-3 delta worth showing (>15% R2 gap). Revisit 2026-04-26 morning.
4. **Gradio SSR mode (`ssr_mode=True`)?** Faster first paint but adds ~50 MB to the container and is still labelled "beta" in Gradio 5.8. Leaving `ssr_mode=False` for stability; revisit if first-paint > 4 s becomes a blocker.
5. **Audio output format: WAV bytes vs numpy?** §2.2 chose `(sample_rate, np.ndarray)` via `synthesize_to_gradio` per `audio.md` §1 binding contract. If Gradio 5.9 regresses `gr.Audio(type="numpy")` we fall back to serving WAV bytes and `gr.Audio(type="filepath")` with a tempfile. No code change elsewhere. Flagged for `audio.md` cross-review.
6. **Rate-limiting public access during pitch.** A TikTok shoutout could spike traffic and evict our session slots. Consider a soft rate-limit (5 requests/min/IP) via a Gradio middleware — orthogonal to the judge flow. **Decide during Batch C5.**
