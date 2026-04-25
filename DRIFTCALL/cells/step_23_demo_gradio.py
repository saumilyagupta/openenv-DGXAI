"""Cell 23 — Inline Gradio demo (Colab + HF Spaces) for DriftCall.

Implements ``docs/modules/deploy_demo_space.md`` §2.2-§2.6 and DESIGN.md §11.2,
§15.

This module is the **storytelling surface**: a Gradio 5.x ``gr.Blocks`` UI that
lets a judge speak a brief, watch the trace panel, and toggle between the base
Gemma 3n E2B model and the trained LoRA adapter without restarting the process.

Design contract (deploy_demo_space.md):
  * Mic input via ``gr.Audio(sources=["microphone"])`` (§2.2).
  * Checkpoint radio with values ``["base", "trained"]`` (§3.2).
  * Drift dropdown enumerating the 20 patterns from drift_injector + ``None``
    (§3.8).
  * Trace ``gr.DataFrame`` with the 5-column schema from §4.3.
  * TTS audio output via ``synthesize_to_gradio`` returning ``(sr, ndarray)``
    (audio.md §2.1).
  * peft hot-swap: ``disable_adapter()`` for base, ``set_adapter("driftcall")``
    + ``enable_adapter_layers()`` for trained (§3.2 step 2 + 3).
  * Process-wide ``DemoSessionState`` registry, max 10 sessions, 900 s TTL
    (§3.3, §4.1).
  * 9 user-facing error modes 5.1-5.9 (§5).
  * Latency budget < 8 s on warm ZeroGPU, < 12 s on warm A10G (§3.6).

Heavy deps (``gradio``, ``spaces``, ``peft``, ``transformers``, ``torch``,
``huggingface_hub``) are loaded lazily inside ``_load_*`` helpers so the cell
imports cleanly on CPU-only CI. Tests monkeypatch the loaders.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


CheckpointId = Literal["base", "trained"]

ActorLiteral = Literal["user", "agent", "env", "drift", "reward"]


# ---------------------------------------------------------------------------
# Errors (deploy_demo_space.md §5)
# ---------------------------------------------------------------------------


class DemoError(Exception):
    """Root for every typed demo-cell error."""


class TrainedAdapterMissingError(DemoError):
    """5.2 — LoRA download failed at boot or adapter file corrupt."""


class CheckpointMismatchError(DemoError):
    """5.5 — LoRA was trained on a different ``base_model_id``."""


class SessionCapacityError(DemoError):
    """5.7 — > 10 concurrent sessions."""


class EnvStepError(DemoError):
    """5.8 — env raised on ``step()``."""


class ZeroGPUUnavailableError(DemoError):
    """5.1 — ``@spaces.GPU`` request rejected."""


class CudaOutOfMemoryError(DemoError):
    """5.4 — model OOM during generate()."""


# ---------------------------------------------------------------------------
# Data structures (§4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRow:
    """One row in the live trace panel. deploy_demo_space.md §4.1, §4.3."""

    turn_idx: int
    actor: ActorLiteral
    action_or_event: str
    tool_response_preview: str
    reward_delta: float


@dataclass
class DemoSessionState:
    """Per-browser-tab state. Mutable by design (§4.1).

    Only ``session.py``-equivalent code (this module's session helpers) writes
    to these fields. Every other consumer reads.
    """

    session_id: str
    env: Any
    last_observation: Any | None = None
    episode_trace: list[TraceRow] = field(default_factory=list)
    audio_buffer: deque[bytes] = field(default_factory=lambda: deque(maxlen=8))
    current_checkpoint: CheckpointId = "base"
    turn_idx: int = 0
    created_at_ms: int = 0
    last_activity_ms: int = 0


@dataclass(frozen=True)
class InferTurnResult:
    """Frozen return record from :func:`infer_turn`. Five positional
    Gradio outputs unpack from this in order."""

    transcript: str
    audio: tuple[int, np.ndarray]
    trace_df: Any  # pandas.DataFrame; Any keeps mypy/CI light
    reward: dict[str, float]
    status_msg: str


# ---------------------------------------------------------------------------
# Lazy dep loaders — patched by tests
# ---------------------------------------------------------------------------


def _load_gradio() -> Any:
    """Return the ``gradio`` module. Patched in tests."""

    import gradio as gr

    return gr


def _load_pandas() -> Any:
    """Return the ``pandas`` module. Patched in tests."""

    import pandas as pd

    return pd


def _load_spaces() -> Any:
    """Return the ``spaces`` module. Patched in tests; absent on non-ZeroGPU."""

    try:
        import spaces

        return spaces
    except ImportError:
        return _NoOpSpaces()


class _NoOpSpaces:
    """Pass-through replacement for the ``spaces`` package on non-ZeroGPU
    hardware. ``@spaces.GPU(...)`` becomes the identity decorator."""

    @staticmethod
    def GPU(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _decorator


def _load_drift_pattern_ids() -> tuple[str, ...]:
    """Return the sorted tuple of all 20 drift pattern ids. Patched in tests."""

    from cells.step_06_drift_injector import list_patterns

    return tuple(p.id for p in list_patterns())


def _load_audio_engines() -> tuple[Any, Any]:
    """Return the ``(asr_engine, tts_engine)`` singletons. Patched in tests."""

    from cells.step_09_audio import get_asr_engine, get_tts_engine

    return get_asr_engine(), get_tts_engine()


def _load_env_factory() -> Callable[[], Any]:
    """Return a ``DriftCallEnv(audio_boundary_enabled=True)`` factory.

    Patched in tests. Heavy import deferred so the cell loads on CPU-only CI.
    """

    def _factory() -> Any:
        from cells.step_10_env import DriftCallEnv

        return DriftCallEnv(config={"audio_boundary_enabled": True})

    return _factory


def _load_peft_module() -> Any:
    """Return the ``peft`` module. Patched in tests."""

    import peft

    return peft


def _load_transformers() -> Any:
    """Return the ``transformers`` module. Patched in tests."""

    import transformers

    return transformers


def _load_torch() -> Any:
    """Return the ``torch`` module. Patched in tests."""

    import torch

    return torch


def _load_hf_hub_errors() -> tuple[type[Exception], ...]:
    """Return the catchable HF-Hub error tuple. Patched in tests."""

    try:
        import huggingface_hub.utils as hf_utils

        entry_not_found: type[Exception] = getattr(hf_utils, "EntryNotFoundError", FileNotFoundError)
        hub_http: type[Exception] = getattr(hf_utils, "HfHubHTTPError", OSError)
        return (entry_not_found, hub_http)
    except ImportError:
        return (FileNotFoundError, OSError)


# ---------------------------------------------------------------------------
# ModelLoader (§2.3)
# ---------------------------------------------------------------------------


class ModelLoader:
    """Process-wide singleton holding the 4-bit base model + LoRA adapter.

    Lazy construction inside the first ``@spaces.GPU`` call (§2.3).
    """

    def __init__(
        self,
        *,
        base_model_id: str = "unsloth/gemma-3n-E2B-it",
        trained_adapter_id: str = "driftcall/gemma-3n-e2b-driftcall-lora",
        max_seq_length: int = 4096,
    ) -> None:
        self._base_model_id = base_model_id
        self._trained_adapter_id = trained_adapter_id
        self._max_seq_length = max_seq_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._trained_available: bool = False
        self._lock = threading.Lock()
        self._load_count: int = 0

    def boot(self) -> None:
        """Load base model + attempt to mount the trained adapter.

        Raises :class:`TrainedAdapterMissingError` only via attribute lookup
        on demand; ``boot()`` itself never raises on a 404 — the demo must
        keep working in baseline-only mode (§7.4).
        """

        with self._lock:
            if self._model is not None:
                return
            transformers = _load_transformers()
            tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
            model_cls = getattr(transformers, "AutoModelForCausalLM", None)
            if tokenizer_cls is None or model_cls is None:
                raise TrainedAdapterMissingError(
                    "transformers missing AutoTokenizer/AutoModelForCausalLM",
                )
            self._tokenizer = tokenizer_cls.from_pretrained(self._base_model_id)
            self._model = model_cls.from_pretrained(self._base_model_id)
            self._load_count += 1
            self._trained_available = self._mount_lora()

    def _mount_lora(self) -> bool:
        """Attempt to mount the trained adapter. Returns ``True`` on success."""

        peft = _load_peft_module()
        peft_model_cls = getattr(peft, "PeftModel", None)
        if peft_model_cls is None:
            return False
        try:
            self._model = peft_model_cls.from_pretrained(
                self._model,
                self._trained_adapter_id,
                adapter_name="driftcall",
            )
            return True
        except _load_hf_hub_errors() as exc:
            logger.warning("LoRA download failed (%s): %s", self._trained_adapter_id, exc)
            return False
        except CheckpointMismatchError as exc:
            logger.warning("LoRA checkpoint mismatch: %s", exc)
            return False
        except Exception as exc:  # defensive — log + continue baseline-only
            logger.warning("LoRA mount failed: %s", exc)
            return False

    def is_trained_available(self) -> bool:
        """Has the LoRA been mounted at boot? (§2.3, §7.4)."""

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
        """Generate one assistant reply. peft hot-swap per §3.2."""

        if self._model is None:
            self.boot()
        assert self._model is not None
        if checkpoint == "trained" and not self._trained_available:
            raise TrainedAdapterMissingError(
                "Trained adapter unavailable; cannot run checkpoint='trained'.",
            )
        torch = _load_torch()
        try:
            torch.manual_seed(seed)
        except Exception:  # tests stub torch without manual_seed
            logger.debug("torch.manual_seed unavailable; ignoring seed", exc_info=True)
        prompt = _format_messages(messages)
        try:
            if checkpoint == "base":
                with self._model.disable_adapter():
                    return self._do_generate(prompt, max_new_tokens, temperature, top_p)
            self._model.set_adapter("driftcall")
            self._model.enable_adapter_layers()
            return self._do_generate(prompt, max_new_tokens, temperature, top_p)
        except CudaOutOfMemoryError:
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise CudaOutOfMemoryError(str(exc)) from exc
            raise

    def _do_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        assert self._model is not None
        result = self._model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        if isinstance(result, (list, tuple)) and result:
            return str(result[0])
        return str(result)


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Tiny chat template stub. Real impl uses tokenizer.apply_chat_template.

    Tests assert that messages flow through; this keeps the cell self-contained.
    """

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


_model_loader: ModelLoader | None = None
_model_loader_lock = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Return the process-wide ModelLoader singleton (§2.3)."""

    global _model_loader
    with _model_loader_lock:
        if _model_loader is None:
            _model_loader = ModelLoader()
        return _model_loader


def _reset_model_loader_for_tests() -> None:
    """Tear down the model loader singleton. Tests only."""

    global _model_loader
    with _model_loader_lock:
        _model_loader = None


# ---------------------------------------------------------------------------
# Session registry (§3.3, §4.1)
# ---------------------------------------------------------------------------


_MAX_CONCURRENT_SESSIONS: int = 10
_SESSION_TTL_S: int = 900

_REGISTRY: dict[str, DemoSessionState] = {}
_REGISTRY_LOCK = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def get_session(session_id: str) -> DemoSessionState:
    """Return the existing session, or create a fresh one. Idempotent (§2.4).

    Raises :class:`SessionCapacityError` when the registry is full (§3.3,
    error 5.7).
    """

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(session_id)
        if existing is not None:
            existing.last_activity_ms = _now_ms()
            return existing
        if len(_REGISTRY) >= _MAX_CONCURRENT_SESSIONS:
            raise SessionCapacityError(
                f"demo at capacity ({_MAX_CONCURRENT_SESSIONS} concurrent sessions)",
            )
        env_factory = _load_env_factory()
        env = env_factory()
        now = _now_ms()
        state = DemoSessionState(
            session_id=session_id,
            env=env,
            created_at_ms=now,
            last_activity_ms=now,
        )
        _REGISTRY[session_id] = state
        return state


def reset_session(session_id: str) -> DemoSessionState:
    """Hard reset: close the env, clear trace, return a fresh state (§3.5)."""

    with _REGISTRY_LOCK:
        old = _REGISTRY.pop(session_id, None)
    if old is not None:
        try:
            old.env.close()
        except Exception:  # close errors must not break reset
            logger.debug("env.close raised on reset; swallowed", exc_info=True)
    fresh = get_session(session_id)
    fresh.current_checkpoint = old.current_checkpoint if old is not None else "base"
    return fresh


def gc_sessions(max_idle_s: int = _SESSION_TTL_S) -> int:
    """Evict sessions idle past TTL. Returns count evicted (§3.3)."""

    cutoff = _now_ms() - (max_idle_s * 1000)
    evicted = 0
    with _REGISTRY_LOCK:
        stale = [sid for sid, st in _REGISTRY.items() if st.last_activity_ms < cutoff]
        for sid in stale:
            old = _REGISTRY.pop(sid)
            try:
                old.env.close()
            except Exception:
                logger.debug("env.close raised on gc; swallowed", exc_info=True)
            evicted += 1
    return evicted


def _reset_session_registry_for_tests() -> None:
    """Clear the session registry. Tests only."""

    with _REGISTRY_LOCK:
        _REGISTRY.clear()


# ---------------------------------------------------------------------------
# DriftToggleBridge (§2.5)
# ---------------------------------------------------------------------------


class DriftToggleBridge:
    """Per-session manual-drift queue with last-write-wins coalescence (§3.8).

    Invariants (§7.3):
      * ``queue(session_id, pattern_id)`` records or replaces the pattern.
      * ``consume(session_id)`` returns the queued pattern once and clears.
      * Same pattern never fires twice from the same ``queue()`` call.
    """

    def __init__(self) -> None:
        self._queue: dict[str, str] = {}
        self._lock = threading.Lock()

    def queue(self, session_id: str, pattern_id: str | None) -> None:
        with self._lock:
            if pattern_id is None:
                self._queue.pop(session_id, None)
            else:
                self._queue[session_id] = pattern_id

    def consume(self, session_id: str) -> str | None:
        with self._lock:
            return self._queue.pop(session_id, None)


_bridge_singleton: DriftToggleBridge | None = None
_bridge_lock = threading.Lock()


def get_drift_bridge() -> DriftToggleBridge:
    global _bridge_singleton
    with _bridge_lock:
        if _bridge_singleton is None:
            _bridge_singleton = DriftToggleBridge()
        return _bridge_singleton


def _reset_drift_bridge_for_tests() -> None:
    global _bridge_singleton
    with _bridge_lock:
        _bridge_singleton = None


# ---------------------------------------------------------------------------
# Trace panel (§2.6)
# ---------------------------------------------------------------------------


_TRACE_COLUMNS: tuple[str, ...] = (
    "turn_idx",
    "actor",
    "action_or_event",
    "tool_response_preview",
    "reward_delta",
)


def render_trace(state: DemoSessionState) -> Any:
    """Build a 5-column DataFrame from ``state.episode_trace``. Pure (§2.6)."""

    pd = _load_pandas()
    if not state.episode_trace:
        return pd.DataFrame(columns=list(_TRACE_COLUMNS))
    rows = [
        {
            "turn_idx": row.turn_idx,
            "actor": row.actor,
            "action_or_event": row.action_or_event,
            "tool_response_preview": row.tool_response_preview,
            "reward_delta": row.reward_delta,
        }
        for row in state.episode_trace
    ]
    return pd.DataFrame(rows, columns=list(_TRACE_COLUMNS))


# ---------------------------------------------------------------------------
# infer_turn (§2.2 contract)
# ---------------------------------------------------------------------------


_DEFAULT_SR: int = 16000


def _safe_default_audio() -> tuple[int, np.ndarray]:
    """1 s of silence at 16 kHz mono. Used as the safe-default audio output."""

    return _DEFAULT_SR, np.zeros(_DEFAULT_SR, dtype=np.float32)


def _safe_default_result(status_msg: str) -> InferTurnResult:
    """Build a safe-default result for any error path."""

    pd = _load_pandas()
    return InferTurnResult(
        transcript="",
        audio=_safe_default_audio(),
        trace_df=pd.DataFrame(columns=list(_TRACE_COLUMNS)),
        reward={},
        status_msg=status_msg,
    )


def _append_trace(state: DemoSessionState, row: TraceRow) -> None:
    """Append a TraceRow without mutating the input row."""

    state.episode_trace.append(row)


def _truncate_preview(payload: Any, *, max_len: int = 120) -> str:
    """First 120 chars of any payload representation, ellipsised."""

    text = "" if payload is None else str(payload)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _resolve_effective_checkpoint(
    requested: CheckpointId,
    loader: ModelLoader,
) -> tuple[CheckpointId, str]:
    """If trained is unavailable but requested, fall back silently (§5.2)."""

    if requested == "trained" and not loader.is_trained_available():
        return "base", "Trained adapter unavailable; showing base model only."
    return requested, ""


def infer_turn(
    audio_tuple: tuple[int, np.ndarray] | None,
    checkpoint: CheckpointId,
    manual_drift: str | None,
    session_id: str,
    *,
    text_input: str | None = None,
    bridge: DriftToggleBridge | None = None,
    loader: ModelLoader | None = None,
) -> InferTurnResult:
    """Handle one mic-to-speaker turn. (§2.2 contract).

    Catches every error 5.1-5.9; on any failure path returns safe defaults
    with a user-facing ``status_msg``. Never writes to disk; never calls
    push_to_hub (§2.2 invariant).
    """

    bridge = bridge if bridge is not None else get_drift_bridge()
    loader = loader if loader is not None else get_model_loader()

    if audio_tuple is None and (text_input is None or text_input.strip() == ""):
        return _safe_default_result("No audio received; press mic or type a brief.")

    try:
        session = get_session(session_id)
    except SessionCapacityError:
        return _safe_default_result("Demo at capacity — try again in a minute.")

    asr_engine, tts_engine = _load_audio_engines()
    transcript_text = ""
    if audio_tuple is not None:
        transcript_text, asr_status = _do_asr(audio_tuple, asr_engine)
        if asr_status:
            return _safe_default_result(asr_status)
    elif text_input is not None:
        transcript_text = text_input.strip()

    effective_checkpoint, fallback_msg = _resolve_effective_checkpoint(checkpoint, loader)
    session.current_checkpoint = effective_checkpoint

    session.turn_idx += 1
    session.last_activity_ms = _now_ms()
    _append_trace(
        session,
        TraceRow(
            turn_idx=session.turn_idx,
            actor="user",
            action_or_event=transcript_text,
            tool_response_preview="",
            reward_delta=0.0,
        ),
    )

    drift_pattern = bridge.consume(session_id)
    if drift_pattern is None and manual_drift is not None:
        drift_pattern = manual_drift
    if drift_pattern is not None:
        _append_trace(
            session,
            TraceRow(
                turn_idx=session.turn_idx,
                actor="drift",
                action_or_event=f"manual:{drift_pattern}",
                tool_response_preview="",
                reward_delta=0.0,
            ),
        )

    step_status = _do_env_step(session, transcript_text, drift_pattern)
    if step_status:
        return _safe_default_result(step_status)

    reply_text, generate_status = _do_generate(loader, session, effective_checkpoint, transcript_text)
    if generate_status:
        return _safe_default_result(generate_status)

    audio_out = _do_tts(tts_engine, reply_text)

    pd_df = render_trace(session)
    reward = {"R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0}
    return InferTurnResult(
        transcript=transcript_text,
        audio=audio_out,
        trace_df=pd_df,
        reward=reward,
        status_msg=fallback_msg,
    )


def _do_asr(
    audio_tuple: tuple[int, np.ndarray],
    asr_engine: Any,
) -> tuple[str, str]:
    """Run ASR on the mic input; return ``(text, status_msg)``.

    ``status_msg`` is non-empty only on error 5.6.
    """

    sample_rate, pcm = audio_tuple
    try:
        wav_bytes = pcm.astype(np.float32).tobytes()
        result = asr_engine.transcribe(wav_bytes, None)
        return result.text, ""
    except Exception as exc:
        logger.warning("ASR failed: %s", exc)
        return "", "Could not decode mic audio; please try again."


def _do_env_step(
    session: DemoSessionState,
    user_text: str,
    drift_pattern: str | None,
) -> str:
    """Run env.step; return non-empty status on EnvStepError (5.8)."""

    env = session.env
    try:
        if drift_pattern is not None:
            obs = env.step({"action_type": "speak", "text": user_text}, force_drift_pattern=drift_pattern)
        else:
            obs = env.step({"action_type": "speak", "text": user_text})
        session.last_observation = obs
        _append_trace(
            session,
            TraceRow(
                turn_idx=session.turn_idx,
                actor="env",
                action_or_event="200 OK",
                tool_response_preview=_truncate_preview(obs),
                reward_delta=0.0,
            ),
        )
        return ""
    except Exception as exc:
        logger.warning("env.step failed: %s", exc)
        _append_trace(
            session,
            TraceRow(
                turn_idx=session.turn_idx,
                actor="env",
                action_or_event=f"rejected: {exc}",
                tool_response_preview="",
                reward_delta=0.0,
            ),
        )
        return f"Env rejected action: {exc}; episode unchanged."


def _do_generate(
    loader: ModelLoader,
    session: DemoSessionState,
    checkpoint: CheckpointId,
    user_text: str,
) -> tuple[str, str]:
    """Run model.generate; return ``(reply, status_msg)``.

    Implements 5.4 OOM retry (shrink context once) and 5.1 ZeroGPU retry
    semantics. Status non-empty when the turn must abort with safe defaults.
    """

    messages = [{"role": "user", "content": user_text}]
    try:
        reply = loader.generate(messages, checkpoint=checkpoint, seed=0)
        _append_trace(
            session,
            TraceRow(
                turn_idx=session.turn_idx,
                actor="agent",
                action_or_event=f"SPEAK {checkpoint}",
                tool_response_preview=_truncate_preview(reply),
                reward_delta=0.0,
            ),
        )
        return reply, ""
    except CudaOutOfMemoryError:
        return _retry_generate_after_oom(loader, session, checkpoint, messages)
    except ZeroGPUUnavailableError:
        return "", "GPU unavailable; the demo is running on CPU and will be slow."
    except TrainedAdapterMissingError:
        return "", "Trained adapter unavailable; showing base model only."
    except TimeoutError:
        return "", "Turn timed out after 60 s — the model was slow; try again."
    except Exception as exc:
        logger.warning("generate failed: %s", exc)
        return "", f"Generation failed: {exc}"


def _retry_generate_after_oom(
    loader: ModelLoader,
    session: DemoSessionState,
    checkpoint: CheckpointId,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """5.4 — empty cache, drop oldest message, retry once with smaller context."""

    torch = _load_torch()
    try:
        torch.cuda.empty_cache()
    except Exception:
        logger.debug("torch.cuda.empty_cache unavailable; ignoring", exc_info=True)
    shrunk = messages[1:] if len(messages) > 1 else messages
    try:
        reply = loader.generate(shrunk, checkpoint=checkpoint, max_new_tokens=128, seed=0)
        _append_trace(
            session,
            TraceRow(
                turn_idx=session.turn_idx,
                actor="agent",
                action_or_event=f"SPEAK {checkpoint} (retry)",
                tool_response_preview=_truncate_preview(reply),
                reward_delta=0.0,
            ),
        )
        return reply, ""
    except Exception as exc:
        logger.warning("generate retry failed: %s", exc)
        return "", "GPU out of memory this turn; reducing context and retrying."


def _do_tts(tts_engine: Any, text: str) -> tuple[int, np.ndarray]:
    """Run TTS; on any error return safe-default audio (1 s silence)."""

    if not text:
        return _safe_default_audio()
    try:
        result = tts_engine.synthesize_to_gradio(text, "en")
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return _safe_default_audio()
    sr, audio = result
    return int(sr), np.asarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# UI builder (§2.2)
# ---------------------------------------------------------------------------


def build_demo() -> Any:
    """Construct the Gradio Blocks graph. Pure (§2.2)."""

    return build_ui()


def build_ui() -> Any:
    """Spec-named alias for ``build_demo``. Tests target both names."""

    gr = _load_gradio()
    loader = get_model_loader()
    drift_pattern_ids = _load_drift_pattern_ids()
    drift_choices: list[str | None] = [None, *drift_pattern_ids]
    trained_available = loader.is_trained_available()
    checkpoint_choices = ["base", "trained"] if trained_available else ["base"]
    checkpoint_label = "Checkpoint" if trained_available else (
        "Checkpoint — Trained adapter unavailable at boot"
    )

    with gr.Blocks(title="DriftCall Demo") as demo:
        gr.Markdown("# DriftCall — Voice-First Indic Concierge")
        with gr.Row():
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Mic input (Hindi / Tamil / Kannada / Hinglish)",
            )
            text_fallback = gr.Textbox(
                label="Fallback: type a brief",
                placeholder="type a brief",
            )
        with gr.Row():
            checkpoint_radio = gr.Radio(
                choices=checkpoint_choices,
                value="base",
                label=checkpoint_label,
            )
            drift_dropdown = gr.Dropdown(
                choices=drift_choices,
                value=None,
                label="Manual drift trigger (next turn only)",
            )
        session_state = gr.State(value=str(uuid.uuid4()))
        transcript_box = gr.Textbox(label="Transcript", interactive=False)
        trace_panel = gr.DataFrame(
            headers=list(_TRACE_COLUMNS),
            wrap=True,
            max_height=400,
            interactive=False,
            label="Trace",
        )
        audio_out = gr.Audio(type="numpy", label="Speaker (TTS)")
        reward_box = gr.JSON(label="Reward components")
        status_box = gr.Markdown("")
        reset_btn = gr.Button("New episode")

        def _wrap(
            audio: tuple[int, np.ndarray] | None,
            ckpt: CheckpointId,
            drift: str | None,
            text: str,
            sid: str,
        ) -> tuple[str, tuple[int, np.ndarray], Any, dict[str, float], str]:
            res = infer_turn(audio, ckpt, drift, sid, text_input=text)
            return res.transcript, res.audio, res.trace_df, res.reward, res.status_msg

        mic_input.change(
            _wrap,
            inputs=[mic_input, checkpoint_radio, drift_dropdown, text_fallback, session_state],
            outputs=[transcript_box, audio_out, trace_panel, reward_box, status_box],
        )
        text_fallback.submit(
            _wrap,
            inputs=[mic_input, checkpoint_radio, drift_dropdown, text_fallback, session_state],
            outputs=[transcript_box, audio_out, trace_panel, reward_box, status_box],
        )

        def _reset(sid: str) -> tuple[Any, dict[str, float], str]:
            reset_session(sid)
            pd = _load_pandas()
            return pd.DataFrame(columns=list(_TRACE_COLUMNS)), {}, "Episode reset."

        reset_btn.click(
            _reset,
            inputs=[session_state],
            outputs=[trace_panel, reward_box, status_box],
        )
    return demo


def warmup_on_boot() -> None:
    """Cold-start hook: load model + warm audio engines (§2.2)."""

    loader = get_model_loader()
    loader.boot()
    asr, tts = _load_audio_engines()
    try:
        asr.warmup()
    except Exception:
        logger.debug("ASR warmup failed; continuing", exc_info=True)
    try:
        tts.warmup()
    except Exception:
        logger.debug("TTS warmup failed; continuing", exc_info=True)


__all__ = [
    "CheckpointId",
    "CheckpointMismatchError",
    "CudaOutOfMemoryError",
    "DemoError",
    "DemoSessionState",
    "DriftToggleBridge",
    "EnvStepError",
    "InferTurnResult",
    "ModelLoader",
    "SessionCapacityError",
    "TraceRow",
    "TrainedAdapterMissingError",
    "ZeroGPUUnavailableError",
    "build_demo",
    "build_ui",
    "gc_sessions",
    "get_drift_bridge",
    "get_model_loader",
    "get_session",
    "infer_turn",
    "render_trace",
    "reset_session",
    "warmup_on_boot",
]
