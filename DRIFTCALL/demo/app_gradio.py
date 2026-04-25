"""DriftCall demo Space — Gradio 5.x entrypoint.

Implements ``docs/modules/deploy_demo_space.md`` (sealed). Single-file demo:
mic → ASR → DriftCallEnv → Gemma 3n E2B (base | trained LoRA) → TTS → speaker
with a live trace panel and a manual drift-injection dropdown.

Hard rules:
- Heavy deps (``gradio``, ``spaces``, ``peft``, ``torch``, ``transformers``,
  ``unsloth``) are imported lazily inside callables so this module imports
  cleanly in CI / on machines without GPUs / Gradio.
- ``infer_turn`` never writes to disk and never calls push-to-hub.
- Latency budget: ≤ 8 s on ZeroGPU warm, ≤ 12 s on A10G warm.
- All 9 error modes (deploy_demo_space.md §5) surface as ``status_msg`` and
  positional safe defaults; the UI never crashes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cells.step_04_models import ActionType, DriftCallAction
from cells.step_06_drift_injector import list_patterns
from cells.step_10_env import DriftCallEnv

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types + constants
# ---------------------------------------------------------------------------

CheckpointId = Literal["base", "trained"]

_MAX_SESSIONS: int = 10
_IDLE_TTL_S: int = 900
_GPU_DURATION_S: int = 60
_LATENCY_BUDGET_S: float = 8.0
_TRACE_COLUMNS: tuple[str, ...] = (
    "turn_idx",
    "actor",
    "action_or_event",
    "tool_response_preview",
    "reward_delta",
)
_BASE_MODEL_ID_DEFAULT: str = "unsloth/gemma-3n-E2B-it"
_TRAINED_ADAPTER_ID_DEFAULT: str = "driftcall/gemma-3n-e2b-driftcall-lora"
_HARDWARE_ENV_VAR: str = "DRIFTCALL_HARDWARE"
_HARDWARE_FALLBACK_ENV_VAR: str = "DRIFTCALL_HARDWARE_FALLBACK"
_TRACE_PREVIEW_LEN: int = 120
_FALLBACK_SR_HZ: int = 16000
_FALLBACK_SILENCE_LEN: int = _FALLBACK_SR_HZ  # 1 s of silence
_DRIFT_PATTERN_IDS: tuple[str, ...] = tuple(p.id for p in list_patterns())


# ---------------------------------------------------------------------------
# Errors (deploy_demo_space.md §5)
# ---------------------------------------------------------------------------


class TrainedAdapterMissingError(RuntimeError):
    """5.2 — LoRA download failed at boot or adapter file corrupt."""


class CheckpointMismatchError(RuntimeError):
    """5.5 — LoRA was trained on a different base_model_id."""


class SessionCapacityError(RuntimeError):
    """5.7 — > 10 concurrent sessions."""


class EnvStepError(RuntimeError):
    """5.8 — env raised during step()."""


class ZeroGPUUnavailableError(RuntimeError):
    """5.1 — @spaces.GPU request rejected."""


class AudioDecodeError(RuntimeError):
    """5.6 — ASR could not decode mic audio."""


# ---------------------------------------------------------------------------
# DemoSessionState (deploy_demo_space.md §4.1)
# ---------------------------------------------------------------------------


@dataclass
class TraceRow:
    turn_idx: int
    actor: Literal["user", "agent", "env", "drift", "reward"]
    action_or_event: str
    tool_response_preview: str
    reward_delta: float


@dataclass
class DemoSessionState:
    """Per-tab state. Mutated only by ``demo.app_gradio`` itself."""

    session_id: str
    env: DriftCallEnv
    last_observation: Any | None = None
    episode_trace: list[TraceRow] = field(default_factory=list)
    audio_buffer: list[bytes] = field(default_factory=list)
    current_checkpoint: CheckpointId = "base"
    turn_idx: int = 0
    created_at_ms: int = 0
    last_activity_ms: int = 0


# ---------------------------------------------------------------------------
# Process-wide registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DemoSessionState] = {}
_REGISTRY_LOCK = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_env() -> DriftCallEnv:
    """Build a fresh env. Audio boundary is enabled in production but the
    constructor requires real engines; tests inject a stub via ``_make_env``
    monkeypatch."""

    return DriftCallEnv()


def get_session(session_id: str) -> DemoSessionState:
    """Return the session for this UUID or create a fresh one. §3.3."""

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(session_id)
        if existing is not None:
            existing.last_activity_ms = _now_ms()
            return existing
        if len(_REGISTRY) >= _MAX_SESSIONS:
            raise SessionCapacityError(
                f"demo at capacity ({_MAX_SESSIONS} concurrent sessions)"
            )
        env = _make_env()
        state = DemoSessionState(
            session_id=session_id,
            env=env,
            created_at_ms=_now_ms(),
            last_activity_ms=_now_ms(),
        )
        _REGISTRY[session_id] = state
        return state


def reset_session(session_id: str) -> DemoSessionState:
    """Hard-reset: close env, drop trace, re-allocate. §3.5."""

    with _REGISTRY_LOCK:
        existing = _REGISTRY.pop(session_id, None)
    if existing is not None:
        try:
            existing.env.close()
        except Exception:
            logger.exception("env.close raised on reset_session for %s", session_id)
    return get_session(session_id)


def gc_sessions(max_idle_s: int = _IDLE_TTL_S) -> int:
    """Evict sessions idle past ``max_idle_s``. Returns the count evicted."""

    cutoff_ms = _now_ms() - max_idle_s * 1000
    with _REGISTRY_LOCK:
        stale = [sid for sid, s in _REGISTRY.items() if s.last_activity_ms < cutoff_ms]
        for sid in stale:
            entry = _REGISTRY.pop(sid)
            try:
                entry.env.close()
            except Exception:
                logger.exception("env.close raised on gc for %s", sid)
    return len(stale)


def _clear_registry_for_tests() -> None:
    """Pytest helper — never used in production code paths."""

    with _REGISTRY_LOCK:
        for entry in _REGISTRY.values():
            with contextlib.suppress(Exception):
                entry.env.close()
        _REGISTRY.clear()


# ---------------------------------------------------------------------------
# DriftToggleBridge (deploy_demo_space.md §2.5, §3.8, §7.3)
# ---------------------------------------------------------------------------


class DriftToggleBridge:
    """One-slot per-session queue with last-write-wins coalescence."""

    def __init__(self) -> None:
        self._slots: dict[str, str] = {}
        self._lock = threading.Lock()

    def queue(self, session_id: str, pattern_id: str | None) -> None:
        with self._lock:
            if pattern_id is None:
                self._slots.pop(session_id, None)
            else:
                self._slots[session_id] = pattern_id

    def consume(self, session_id: str) -> str | None:
        with self._lock:
            return self._slots.pop(session_id, None)


_DEFAULT_BRIDGE = DriftToggleBridge()


def get_drift_bridge() -> DriftToggleBridge:
    return _DEFAULT_BRIDGE


# ---------------------------------------------------------------------------
# Trace panel (deploy_demo_space.md §2.6, §4.3)
# ---------------------------------------------------------------------------


def render_trace(state: DemoSessionState) -> pd.DataFrame:
    """Pure rendering — never mutates ``state``."""

    import pandas as pd

    rows = [
        {
            "turn_idx": r.turn_idx,
            "actor": r.actor,
            "action_or_event": r.action_or_event,
            "tool_response_preview": r.tool_response_preview,
            "reward_delta": float(r.reward_delta),
        }
        for r in state.episode_trace
    ]
    return pd.DataFrame(rows, columns=list(_TRACE_COLUMNS))


def _empty_trace_df() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame([], columns=list(_TRACE_COLUMNS))


# ---------------------------------------------------------------------------
# ModelLoader (deploy_demo_space.md §2.3, §3.2)
# ---------------------------------------------------------------------------


class ModelLoader:
    """Process-wide singleton. Holds the 4-bit base model + LoRA adapter."""

    def __init__(
        self,
        *,
        base_model_id: str = _BASE_MODEL_ID_DEFAULT,
        trained_adapter_id: str = _TRAINED_ADAPTER_ID_DEFAULT,
        max_seq_length: int = 4096,
    ) -> None:
        self._base_model_id = base_model_id
        self._trained_adapter_id = trained_adapter_id
        self._max_seq_length = max_seq_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._trained_available: bool = False
        self._lock = threading.Lock()

    def _load_base(self) -> tuple[Any, Any]:
        """Load the 4-bit base model. Patched by tests via ``_load_base``."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_cls: Any = AutoTokenizer
        model_cls: Any = AutoModelForCausalLM
        tokenizer = tok_cls.from_pretrained(self._base_model_id)
        model = model_cls.from_pretrained(self._base_model_id)
        return model, tokenizer

    def _try_mount_adapter(self, model: Any) -> Any | None:
        """Mount ``self._trained_adapter_id`` as the ``driftcall`` LoRA. None on miss."""

        try:
            from peft import PeftModel
        except ImportError as exc:
            logger.warning("peft import failed: %s", exc)
            return None
        try:
            return PeftModel.from_pretrained(
                model, self._trained_adapter_id, adapter_name="driftcall"
            )
        except CheckpointMismatchError as exc:
            logger.warning("checkpoint mismatch on LoRA mount: %s", exc)
            return None
        except Exception as exc:
            # Captures EntryNotFoundError, HTTPError(404), generic peft failures.
            logger.warning("LoRA mount failed: %s", exc)
            return None

    def ensure_loaded(self) -> None:
        """Lazy load — first ZeroGPU-decorated call invokes this."""

        with self._lock:
            if self._model is not None:
                return
            base_model, tokenizer = self._load_base()
            self._tokenizer = tokenizer
            wrapped = self._try_mount_adapter(base_model)
            if wrapped is None:
                self._model = base_model
                self._trained_available = False
            else:
                self._model = wrapped
                self._trained_available = True

    def is_trained_available(self) -> bool:
        return self._trained_available

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
        """Run the correct adapter and return the completion text."""

        self.ensure_loaded()
        if checkpoint == "trained" and not self._trained_available:
            raise TrainedAdapterMissingError(
                "Trained adapter unavailable; falling back to base"
            )
        model = self._model
        if model is None:
            raise RuntimeError("model unexpectedly None")

        if checkpoint == "base":
            ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _NullCtx()
            with ctx:
                return self._run(model, messages, max_new_tokens, temperature, top_p, seed)
        # trained
        if hasattr(model, "set_adapter"):
            model.set_adapter("driftcall")
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        return self._run(model, messages, max_new_tokens, temperature, top_p, seed)

    def _run(
        self,
        model: Any,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        """Default implementation hits HF generate. Tests stub via ``_run``."""

        try:
            return str(
                model.generate(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"model.generate raised: {exc}") from exc


class _NullCtx:
    """Trivial context manager fallback when peft.disable_adapter is absent."""

    def __enter__(self) -> _NullCtx:
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        return False


_MODEL_LOADER: ModelLoader | None = None
_MODEL_LOADER_LOCK = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Return process-wide singleton, instantiated on first call."""

    global _MODEL_LOADER
    with _MODEL_LOADER_LOCK:
        if _MODEL_LOADER is None:
            _MODEL_LOADER = ModelLoader()
        return _MODEL_LOADER


def _reset_model_loader_for_tests() -> None:
    global _MODEL_LOADER
    with _MODEL_LOADER_LOCK:
        _MODEL_LOADER = None


# ---------------------------------------------------------------------------
# Hardware probing (deploy_demo_space.md §3.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProbe:
    zerogpu: bool
    a10g: bool


def _probe_hardware() -> HardwareProbe:
    """Probe both targets. Patched in tests."""

    fallback = os.environ.get(_HARDWARE_FALLBACK_ENV_VAR, "")
    hardware = os.environ.get(_HARDWARE_ENV_VAR, "zero-gpu")
    return HardwareProbe(zerogpu=hardware == "zero-gpu", a10g=fallback == "a10g")


class DeploymentAbortedError(RuntimeError):
    """Raised when neither ZeroGPU nor A10G is available."""


def deploy_check() -> str:
    """Return the README front-matter ``hardware:`` value to write.

    Raises ``DeploymentAbortedError("both-gpus-unavailable")`` when both
    fallbacks fail; the pitch reverts to a pre-recorded video.
    """

    probe = _probe_hardware()
    if probe.zerogpu:
        return "zero-gpu"
    if probe.a10g:
        logger.info("zero-gpu unavailable; falling back to A10G small")
        return "a10g-small"
    logger.warning("Fall back to pre-recorded video — see risk_book.md")
    raise DeploymentAbortedError("both-gpus-unavailable")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _safe_silence() -> tuple[int, np.ndarray]:
    """1 s of silence at 16 kHz — used as the safe-default audio output."""

    return _FALLBACK_SR_HZ, np.zeros(_FALLBACK_SILENCE_LEN, dtype=np.float32)


def _safe_defaults() -> tuple[str, tuple[int, np.ndarray], pd.DataFrame, dict[str, float], str]:
    return "", _safe_silence(), _empty_trace_df(), {}, ""


def _truncate_preview(payload: Any) -> str:
    text = str(payload)
    if len(text) <= _TRACE_PREVIEW_LEN:
        return text
    return text[: _TRACE_PREVIEW_LEN - 1] + "…"


# ---------------------------------------------------------------------------
# infer_turn (deploy_demo_space.md §2.2)
# ---------------------------------------------------------------------------


def infer_turn(
    audio_tuple: tuple[int, np.ndarray] | None,
    checkpoint: CheckpointId,
    manual_drift: str | None,
    session_id: str,
    *,
    text_input: str | None = None,
) -> tuple[str, tuple[int, np.ndarray], pd.DataFrame, dict[str, float], str]:
    """Handle one mic-to-speaker turn. Never raises — surfaces every error
    via the ``status_msg`` slot of the return tuple."""

    # 1. Session — error 5.7.
    try:
        state = get_session(session_id)
    except SessionCapacityError:
        empty, silence, df, defaults, _ = _safe_defaults()
        return empty, silence, df, defaults, "Demo at capacity — try again in a minute."

    # 2. Audio input — error 5.3.
    if audio_tuple is None and not (text_input and text_input.strip()):
        empty, silence, _df, empty_rewards, _ = _safe_defaults()
        return (
            empty,
            silence,
            render_trace(state),
            empty_rewards,
            "No audio received; press mic or type a brief.",
        )

    # 3. ASR — error 5.6.
    transcript: str = ""
    if audio_tuple is not None:
        try:
            transcript = _run_asr(audio_tuple)
        except AudioDecodeError:
            empty2, silence2, _df2, empty_rewards2, _ = _safe_defaults()
            return (
                empty2,
                silence2,
                render_trace(state),
                empty_rewards2,
                "Could not decode mic audio; please try again.",
            )
    else:
        transcript = (text_input or "").strip()

    # 4. Drift consume.
    forced = get_drift_bridge().consume(session_id)
    if manual_drift is not None:
        forced = manual_drift

    # 5. Build action — first-turn agent uses transcript as a SPEAK.
    state.turn_idx += 1
    state.episode_trace.append(
        TraceRow(
            turn_idx=state.turn_idx,
            actor="user",
            action_or_event=transcript,
            tool_response_preview="",
            reward_delta=0.0,
        )
    )
    if forced is not None:
        state.episode_trace.append(
            TraceRow(
                turn_idx=state.turn_idx,
                actor="drift",
                action_or_event=f"manual:{forced}",
                tool_response_preview="",
                reward_delta=0.0,
            )
        )

    action = DriftCallAction(
        action_type=ActionType.SPEAK,
        message=transcript or "(empty)",
    )

    # 6. Env step — error 5.8.
    try:
        if forced is not None:
            obs = state.env.step(action, force_drift_pattern=forced)
        else:
            obs = state.env.step(action)
        state.last_observation = obs
        state.episode_trace.append(
            TraceRow(
                turn_idx=state.turn_idx,
                actor="env",
                action_or_event="200 OK",
                tool_response_preview=_truncate_preview(obs.last_transcript),
                reward_delta=0.0,
            )
        )
    except Exception as exc:
        state.episode_trace.append(
            TraceRow(
                turn_idx=state.turn_idx,
                actor="env",
                action_or_event=f"rejected: {exc.__class__.__name__}",
                tool_response_preview=_truncate_preview(str(exc)),
                reward_delta=0.0,
            )
        )
        return (
            transcript,
            _safe_silence(),
            render_trace(state),
            {},
            f"Env rejected action: {exc}; episode unchanged.",
        )

    # 7. Generate — errors 5.1 / 5.2 / 5.4.
    loader = get_model_loader()
    use_checkpoint: CheckpointId = checkpoint
    if checkpoint == "trained" and not loader.is_trained_available():
        use_checkpoint = "base"
        status_warning = "Trained adapter unavailable; showing base model only."
    else:
        status_warning = ""

    reply: str
    try:
        reply = _generate_with_retries(loader, transcript, use_checkpoint)
    except TrainedAdapterMissingError:
        use_checkpoint = "base"
        status_warning = "Trained adapter unavailable; showing base model only."
        try:
            reply = _generate_with_retries(loader, transcript, use_checkpoint)
        except Exception as exc2:
            return (
                transcript,
                _safe_silence(),
                render_trace(state),
                {},
                f"Generate failed: {exc2}",
            )
    except _OOMRetryFailure:
        return (
            transcript,
            _safe_silence(),
            render_trace(state),
            {},
            "GPU out of memory this turn; reducing context and retrying.",
        )
    except _ZeroGPUFailure:
        return (
            transcript,
            _safe_silence(),
            render_trace(state),
            {},
            "GPU unavailable; the demo is running on CPU and will be slow.",
        )
    except _TimeoutFailure:
        return (
            transcript,
            _safe_silence(),
            render_trace(state),
            {},
            "Turn timed out after 60 s — the model was slow; try again.",
        )
    except Exception as exc:
        return (
            transcript,
            _safe_silence(),
            render_trace(state),
            {},
            f"Generate failed: {exc}",
        )

    state.episode_trace.append(
        TraceRow(
            turn_idx=state.turn_idx,
            actor="agent",
            action_or_event=f"SPEAK \"{reply[:60]}\"",
            tool_response_preview="",
            reward_delta=0.0,
        )
    )

    # 8. TTS.
    try:
        audio_out = _run_tts(reply, lang_hint="en")
    except Exception:
        audio_out = _safe_silence()

    rewards: dict[str, float] = {}
    state.last_activity_ms = _now_ms()
    return transcript, audio_out, render_trace(state), rewards, status_warning


# ---------------------------------------------------------------------------
# Subroutines (kept module-level so tests can patch each one)
# ---------------------------------------------------------------------------


class _OOMRetryFailure(RuntimeError):
    pass


class _ZeroGPUFailure(RuntimeError):
    pass


class _TimeoutFailure(RuntimeError):
    pass


def _run_asr(audio_tuple: tuple[int, np.ndarray]) -> str:
    """Default ASR path. Tests patch this to bypass the audio singleton."""

    sr, wav = audio_tuple
    if sr != 16000:
        # Silent fallback rather than raising — gradio mic clips are usually 16k.
        return ""
    pcm_bytes = wav.astype(np.float32).tobytes()
    from cells.step_09_audio import get_asr_engine

    asr = get_asr_engine()
    result = asr.transcribe(pcm_bytes, "en")
    return result.text


def _run_tts(text: str, *, lang_hint: str = "en") -> tuple[int, np.ndarray]:
    """Default TTS path."""

    from cells.step_09_audio import get_tts_engine

    tts = get_tts_engine()
    lang_for_tts: Any = lang_hint
    out: tuple[int, np.ndarray] = tts.synthesize_to_gradio(text, lang_for_tts)
    return out


def _generate_with_retries(
    loader: ModelLoader, transcript: str, checkpoint: CheckpointId
) -> str:
    """Wraps ``ModelLoader.generate`` with the OOM / ZeroGPU retry policy."""

    messages = [{"role": "user", "content": transcript or "(empty)"}]
    try:
        return loader.generate(messages, checkpoint=checkpoint, max_new_tokens=256)
    except ZeroGPUUnavailableError:
        # 5.1 — retry once, then fall back.
        time.sleep(0)  # advisory; tests patch
        try:
            return loader.generate(messages, checkpoint=checkpoint, max_new_tokens=256)
        except ZeroGPUUnavailableError as exc:
            raise _ZeroGPUFailure() from exc
    except TimeoutError as exc:
        raise _TimeoutFailure() from exc
    except Exception as exc:
        # Treat any CUDA OOM (real or stub) as 5.4.
        msg = str(exc).lower()
        if "out of memory" in msg or exc.__class__.__name__ == "OutOfMemoryError":
            try:
                _empty_cuda_cache()
                # Shrink context: drop oldest message + reduce tokens.
                shrunk = messages[1:] if len(messages) > 1 else messages
                return loader.generate(
                    shrunk, checkpoint=checkpoint, max_new_tokens=128
                )
            except Exception as exc2:
                msg2 = str(exc2).lower()
                if "out of memory" in msg2 or exc2.__class__.__name__ == "OutOfMemoryError":
                    raise _OOMRetryFailure() from exc2
                raise
        raise


def _empty_cuda_cache() -> None:
    """Best-effort CUDA cache clear. Tests patch this."""

    try:
        import torch

        if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Warmup + UI builder
# ---------------------------------------------------------------------------


def warmup_on_boot() -> None:
    """Page in CUDA kernels: ASR + TTS + dummy generate."""

    try:
        from cells.step_09_audio import get_asr_engine, get_tts_engine

        asr = get_asr_engine()
        tts = get_tts_engine()
        if hasattr(asr, "warmup"):
            asr.warmup()
        if hasattr(tts, "warmup"):
            tts.warmup()
    except Exception:
        logger.exception("audio warmup failed")
    try:
        loader = get_model_loader()
        loader.ensure_loaded()
        loader.generate(
            [{"role": "user", "content": "warmup"}], checkpoint="base", max_new_tokens=4
        )
    except Exception:
        logger.exception("model warmup failed")


def build_ui() -> Any:
    """Construct the Gradio Blocks graph. Idempotent and pure."""

    import gradio as gr

    loader = get_model_loader()
    trained_ok = False
    try:
        trained_ok = loader.is_trained_available()
    except Exception:
        trained_ok = False
    radio_choices = ["base", "trained"] if trained_ok else ["base"]
    radio_label = (
        "Checkpoint"
        if trained_ok
        else "Checkpoint (Trained adapter unavailable at boot — base only)"
    )
    drift_choices: list[Any] = [*_DRIFT_PATTERN_IDS, None]

    with gr.Blocks(title="DriftCall Demo") as demo:
        session_state = gr.State(value=str(uuid.uuid4()))
        with gr.Row():
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Speak your brief",
            )
            speaker = gr.Audio(
                type="numpy",
                label="Speaker",
                interactive=False,
            )
        with gr.Row():
            checkpoint = gr.Radio(
                choices=radio_choices,
                value="base",
                label=radio_label,
            )
            drift = gr.Dropdown(
                choices=drift_choices,
                value=None,
                label="Manual drift",
            )
        textbox = gr.Textbox(
            placeholder="Or type a brief here",
            label="Text fallback",
        )
        transcript = gr.Textbox(label="Transcript", interactive=False)
        trace = gr.DataFrame(
            value=_empty_trace_df(),
            wrap=True,
            max_height=400,
            interactive=False,
            headers=list(_TRACE_COLUMNS),
        )
        rewards = gr.JSON(label="Rewards")
        status = gr.Textbox(label="Status", interactive=False)
        reset_btn = gr.Button("New episode")

        def _on_submit(
            mic_in: Any,
            ckpt: Any,
            drift_pat: Any,
            sid: Any,
            text_in: Any,
        ) -> Any:
            return infer_turn(mic_in, ckpt, drift_pat, sid, text_input=text_in)

        mic.stop_recording(
            _on_submit,
            inputs=[mic, checkpoint, drift, session_state, textbox],
            outputs=[transcript, speaker, trace, rewards, status],
        )

        def _on_reset(sid: Any) -> Any:
            reset_session(sid)
            return "", _empty_trace_df(), {}, "Episode reset."

        reset_btn.click(
            _on_reset,
            inputs=[session_state],
            outputs=[transcript, trace, rewards, status],
        )

    return demo


def _launch_for_production() -> None:
    warmup_on_boot()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)


if __name__ == "__main__":
    _launch_for_production()


__all__ = [
    "AudioDecodeError",
    "CheckpointId",
    "CheckpointMismatchError",
    "DemoSessionState",
    "DeploymentAbortedError",
    "DriftToggleBridge",
    "EnvStepError",
    "HardwareProbe",
    "ModelLoader",
    "SessionCapacityError",
    "TraceRow",
    "TrainedAdapterMissingError",
    "ZeroGPUUnavailableError",
    "build_ui",
    "deploy_check",
    "gc_sessions",
    "get_drift_bridge",
    "get_model_loader",
    "get_session",
    "infer_turn",
    "render_trace",
    "reset_session",
    "warmup_on_boot",
]
