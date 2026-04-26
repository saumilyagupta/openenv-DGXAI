"""Tests for cells/step_23_demo_gradio.py.

Mocks all heavy deps (gradio, peft, transformers, torch, audio engines, env).
Covers:
  - build_demo() and build_ui() return a Gradio Blocks (mocked) without
    constructing real GPU state.
  - peft hot-swap: disable_adapter() called for "base"; set_adapter("driftcall")
    + enable_adapter_layers() called for "trained".
  - Session registry: idempotent get, isolation across UUIDs, 10-cap, 900 s
    TTL eviction, reset closes the env.
  - DriftToggleBridge: queue + last-write-wins coalescence; consume drains.
  - render_trace: pure, returns DataFrame with the 5 spec columns.
  - 9 error modes 5.1-5.9 each return safe defaults + status_msg.
  - infer_turn never writes to disk; never calls push_to_hub.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from cells import step_23_demo_gradio as demo_mod
from cells.step_23_demo_gradio import (
    CheckpointMismatchError,
    CudaOutOfMemoryError,
    DemoSessionState,
    DriftToggleBridge,
    InferTurnResult,
    ModelLoader,
    SessionCapacityError,
    TraceRow,
    TrainedAdapterMissingError,
    ZeroGPUUnavailableError,
    build_demo,
    build_ui,
    gc_sessions,
    get_drift_bridge,
    get_model_loader,
    get_session,
    infer_turn,
    render_trace,
    reset_session,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeASR:
    def __init__(self, text: str = "namaste world", *, raise_exc: Exception | None = None) -> None:
        self._text = text
        self._raise = raise_exc
        self.calls = 0

    def transcribe(self, audio_bytes: bytes, hint: Any = None) -> Any:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return MagicMock(text=self._text, language_detected="hi", confidence=0.9, duration_s=2.0)

    def warmup(self) -> None:
        pass


class _FakeTTS:
    def __init__(self, *, raise_exc: Exception | None = None) -> None:
        self._raise = raise_exc
        self.calls = 0

    def synthesize_to_gradio(self, text: str, language: str) -> tuple[int, np.ndarray]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return 16000, np.zeros(8000, dtype=np.float32)

    def warmup(self) -> None:
        pass


class _FakePeftModel:
    """Stand-in for the hot-swappable peft-wrapped model."""

    def __init__(self) -> None:
        self.disable_adapter_calls = 0
        self.set_adapter_calls: list[str] = []
        self.enable_adapter_calls = 0
        self.generate_calls: list[dict[str, Any]] = []
        self._next_text = "ok"
        self._raise: Exception | None = None

    def set_next_text(self, text: str) -> None:
        self._next_text = text

    def set_next_raise(self, exc: Exception | None) -> None:
        self._raise = exc

    class _DisableCtx:
        def __init__(self, parent: _FakePeftModel) -> None:
            self.parent = parent

        def __enter__(self) -> _FakePeftModel:
            self.parent.disable_adapter_calls += 1
            return self.parent

        def __exit__(self, *_: Any) -> None:
            return None

    def disable_adapter(self) -> _DisableCtx:
        return self._DisableCtx(self)

    def set_adapter(self, name: str) -> None:
        self.set_adapter_calls.append(name)

    def enable_adapter_layers(self) -> None:
        self.enable_adapter_calls += 1

    def generate(self, **kwargs: Any) -> str:
        self.generate_calls.append(kwargs)
        if self._raise is not None:
            exc = self._raise
            self._raise = None
            raise exc
        return self._next_text


class _FakeEnv:
    """Mimics DriftCallEnv for demo turn tests."""

    def __init__(self) -> None:
        self.step_calls: list[dict[str, Any]] = []
        self.closed = False
        self._raise: Exception | None = None

    def step(self, action: Any, **kwargs: Any) -> Any:
        self.step_calls.append({"action": action, **kwargs})
        if self._raise is not None:
            exc = self._raise
            self._raise = None
            raise exc
        return MagicMock()

    def close(self) -> None:
        self.closed = True

    def set_next_raise(self, exc: Exception | None) -> None:
        self._raise = exc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    demo_mod._reset_session_registry_for_tests()
    demo_mod._reset_drift_bridge_for_tests()
    demo_mod._reset_model_loader_for_tests()


@pytest.fixture
def fake_audio(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeASR, _FakeTTS]:
    asr = _FakeASR()
    tts = _FakeTTS()
    monkeypatch.setattr(demo_mod, "_load_audio_engines", lambda: (asr, tts))
    return asr, tts


@pytest.fixture
def fake_env_factory(monkeypatch: pytest.MonkeyPatch) -> list[_FakeEnv]:
    envs: list[_FakeEnv] = []

    def _factory_loader() -> Any:
        def _factory() -> _FakeEnv:
            env = _FakeEnv()
            envs.append(env)
            return env

        return _factory

    monkeypatch.setattr(demo_mod, "_load_env_factory", _factory_loader)
    return envs


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> ModelLoader:
    loader = ModelLoader()
    fake_model = _FakePeftModel()
    loader._model = fake_model
    loader._tokenizer = MagicMock()
    loader._trained_available = True
    monkeypatch.setattr(demo_mod, "get_model_loader", lambda: loader)

    fake_torch = MagicMock()
    fake_torch.manual_seed = MagicMock()
    fake_torch.cuda.empty_cache = MagicMock()
    monkeypatch.setattr(demo_mod, "_load_torch", lambda: fake_torch)
    return loader


@pytest.fixture
def fake_drift_patterns(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    patterns = tuple(f"pattern_{i:02d}" for i in range(20))
    monkeypatch.setattr(demo_mod, "_load_drift_pattern_ids", lambda: patterns)
    return patterns


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    def test_get_creates_fresh_session(self, fake_env_factory: list[_FakeEnv]) -> None:
        s = get_session("uuid-A")
        assert s.session_id == "uuid-A"
        assert s.episode_trace == []
        assert s.turn_idx == 0
        assert s.current_checkpoint == "base"
        assert len(fake_env_factory) == 1

    def test_get_idempotent(self, fake_env_factory: list[_FakeEnv]) -> None:
        a1 = get_session("uuid-A")
        a2 = get_session("uuid-A")
        assert a1 is a2
        assert len(fake_env_factory) == 1

    def test_isolation_across_uuids(self, fake_env_factory: list[_FakeEnv]) -> None:
        a = get_session("uuid-A")
        b = get_session("uuid-B")
        assert a is not b
        assert a.env is not b.env

    def test_cap_at_10(self, fake_env_factory: list[_FakeEnv]) -> None:
        for i in range(10):
            get_session(f"uuid-{i}")
        with pytest.raises(SessionCapacityError):
            get_session("uuid-11")

    def test_gc_evicts_idle(self, fake_env_factory: list[_FakeEnv]) -> None:
        s = get_session("uuid-A")
        s.last_activity_ms = int(time.time() * 1000) - (1000 * 1000)
        evicted = gc_sessions(max_idle_s=900)
        assert evicted == 1

    def test_gc_keeps_active(self, fake_env_factory: list[_FakeEnv]) -> None:
        get_session("uuid-A")
        evicted = gc_sessions(max_idle_s=900)
        assert evicted == 0

    def test_reset_closes_env_and_clears_trace(
        self, fake_env_factory: list[_FakeEnv]
    ) -> None:
        s = get_session("uuid-A")
        old_env = s.env
        s.episode_trace.append(
            TraceRow(turn_idx=1, actor="user", action_or_event="x", tool_response_preview="", reward_delta=0.0)
        )
        fresh = reset_session("uuid-A")
        assert old_env.closed
        assert fresh.episode_trace == []
        assert fresh.turn_idx == 0


# ---------------------------------------------------------------------------
# DriftToggleBridge
# ---------------------------------------------------------------------------


class TestDriftToggleBridge:
    def test_queue_then_consume(self) -> None:
        bridge = DriftToggleBridge()
        bridge.queue("S", "p1")
        assert bridge.consume("S") == "p1"
        assert bridge.consume("S") is None

    def test_last_write_wins(self) -> None:
        bridge = DriftToggleBridge()
        bridge.queue("S", "p1")
        bridge.queue("S", "p2")
        assert bridge.consume("S") == "p2"

    def test_clear_via_none(self) -> None:
        bridge = DriftToggleBridge()
        bridge.queue("S", "p1")
        bridge.queue("S", None)
        assert bridge.consume("S") is None

    def test_isolated_sessions(self) -> None:
        bridge = DriftToggleBridge()
        bridge.queue("A", "pA")
        bridge.queue("B", "pB")
        assert bridge.consume("A") == "pA"
        assert bridge.consume("B") == "pB"

    def test_singleton(self) -> None:
        a = get_drift_bridge()
        b = get_drift_bridge()
        assert a is b


# ---------------------------------------------------------------------------
# render_trace
# ---------------------------------------------------------------------------


class TestRenderTrace:
    def test_empty_trace_returns_empty_dataframe(
        self, fake_env_factory: list[_FakeEnv]
    ) -> None:
        s = get_session("uuid-A")
        df = render_trace(s)
        assert list(df.columns) == [
            "turn_idx",
            "actor",
            "action_or_event",
            "tool_response_preview",
            "reward_delta",
        ]
        assert len(df) == 0

    def test_populated_trace(self, fake_env_factory: list[_FakeEnv]) -> None:
        s = get_session("uuid-A")
        s.episode_trace.append(
            TraceRow(turn_idx=1, actor="user", action_or_event="hi", tool_response_preview="", reward_delta=0.0)
        )
        s.episode_trace.append(
            TraceRow(turn_idx=1, actor="agent", action_or_event="hello", tool_response_preview="", reward_delta=0.0)
        )
        df = render_trace(s)
        assert len(df) == 2
        assert df.iloc[0]["actor"] == "user"
        assert df.iloc[1]["actor"] == "agent"

    def test_render_does_not_mutate_state(
        self, fake_env_factory: list[_FakeEnv]
    ) -> None:
        s = get_session("uuid-A")
        row = TraceRow(turn_idx=1, actor="user", action_or_event="x", tool_response_preview="", reward_delta=0.0)
        s.episode_trace.append(row)
        snapshot = list(s.episode_trace)
        render_trace(s)
        render_trace(s)
        assert list(s.episode_trace) == snapshot


# ---------------------------------------------------------------------------
# ModelLoader peft hot-swap
# ---------------------------------------------------------------------------


class TestModelLoaderHotSwap:
    def _build(
        self, monkeypatch: pytest.MonkeyPatch, *, trained_available: bool = True
    ) -> tuple[ModelLoader, _FakePeftModel]:
        loader = ModelLoader()
        fake_model = _FakePeftModel()
        loader._model = fake_model
        loader._tokenizer = MagicMock()
        loader._trained_available = trained_available
        fake_torch = MagicMock()
        fake_torch.manual_seed = MagicMock()
        monkeypatch.setattr(demo_mod, "_load_torch", lambda: fake_torch)
        return loader, fake_model

    def test_base_calls_disable_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader, model = self._build(monkeypatch)
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="base")
        assert model.disable_adapter_calls == 1
        assert model.set_adapter_calls == []
        assert model.enable_adapter_calls == 0

    def test_trained_calls_set_and_enable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader, model = self._build(monkeypatch)
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="trained")
        assert model.disable_adapter_calls == 0
        assert model.set_adapter_calls == ["driftcall"]
        assert model.enable_adapter_calls == 1

    def test_trained_raises_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader, _ = self._build(monkeypatch, trained_available=False)
        with pytest.raises(TrainedAdapterMissingError):
            loader.generate([{"role": "user", "content": "hi"}], checkpoint="trained")

    def test_oom_raises_typed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader, model = self._build(monkeypatch)
        model.set_next_raise(RuntimeError("CUDA out of memory"))
        with pytest.raises(CudaOutOfMemoryError):
            loader.generate([{"role": "user", "content": "hi"}], checkpoint="base")

    def test_no_double_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader, _ = self._build(monkeypatch)
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="base")
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="trained")
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="base")
        assert loader._load_count == 0


# ---------------------------------------------------------------------------
# infer_turn behaviour + 9 error modes
# ---------------------------------------------------------------------------


class TestInferTurnHappyPath:
    def test_full_turn_succeeds(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        audio = (16000, np.zeros(16000, dtype=np.float32))
        result = infer_turn(audio, "trained", None, "uuid-A")
        assert isinstance(result, InferTurnResult)
        assert result.transcript == "namaste world"
        assert result.audio[0] == 16000
        assert result.status_msg == ""

    def test_text_fallback(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        result = infer_turn(None, "base", None, "uuid-A", text_input="book a flight")
        assert result.transcript == "book a flight"
        assert result.status_msg == ""

    def test_session_isolation_state(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        infer_turn(None, "base", None, "uuid-A", text_input="x")
        infer_turn(None, "base", None, "uuid-B", text_input="y")
        a = get_session("uuid-A")
        b = get_session("uuid-B")
        assert a.turn_idx == 1
        assert b.turn_idx == 1
        assert a.env is not b.env

    def test_manual_drift_passes_to_env(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        infer_turn(None, "base", "pattern_05", "uuid-A", text_input="hi")
        assert fake_env_factory[0].step_calls
        kwargs = fake_env_factory[0].step_calls[0]
        assert kwargs.get("force_drift_pattern") == "pattern_05"

    def test_drift_bridge_overrides_arg_when_queued(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        bridge = get_drift_bridge()
        bridge.queue("uuid-A", "queued_pattern")
        infer_turn(None, "base", "arg_pattern", "uuid-A", text_input="hi")
        kwargs = fake_env_factory[0].step_calls[0]
        assert kwargs.get("force_drift_pattern") == "queued_pattern"


class TestInferTurnErrorModes:
    """Each of the 9 error modes 5.1-5.9."""

    def test_5_1_zerogpu_unavailable_falls_back_with_message(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        fake_loader._model.set_next_raise(ZeroGPUUnavailableError("queue full"))
        result = infer_turn(None, "base", None, "uuid-A", text_input="hi")
        assert "GPU unavailable" in result.status_msg
        assert result.transcript == ""
        assert result.audio[0] == 16000

    def test_5_2_trained_adapter_missing_silent_fallback(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        fake_loader._trained_available = False
        result = infer_turn(None, "trained", None, "uuid-A", text_input="hi")
        assert "Trained adapter unavailable" in result.status_msg

    def test_5_3_mic_denied_textbox_empty(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        result = infer_turn(None, "base", None, "uuid-A", text_input="")
        assert result.status_msg == "No audio received; press mic or type a brief."
        assert result.transcript == ""
        assert result.audio[0] == 16000
        assert len(result.audio[1]) == 16000

    def test_5_4_oom_retry_then_status(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        fake_loader._model.set_next_raise(
            RuntimeError("CUDA out of memory")
        )
        result = infer_turn(None, "base", None, "uuid-A", text_input="hi")
        assert result.transcript == "hi"
        assert result.status_msg == "" or "memory" in result.status_msg.lower()

    def test_5_5_checkpoint_mismatch_treated_as_5_2(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loader = ModelLoader()
        fake_model = _FakePeftModel()
        loader._model = fake_model
        loader._trained_available = False  # mismatch boot path
        monkeypatch.setattr(demo_mod, "get_model_loader", lambda: loader)
        fake_torch = MagicMock()
        fake_torch.manual_seed = MagicMock()
        fake_torch.cuda.empty_cache = MagicMock()
        monkeypatch.setattr(demo_mod, "_load_torch", lambda: fake_torch)
        result = infer_turn(None, "trained", None, "uuid-A", text_input="hi")
        assert "Trained adapter unavailable" in result.status_msg

    def test_5_6_audio_decode_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        bad_asr = _FakeASR(raise_exc=ValueError("decode fail"))
        good_tts = _FakeTTS()
        monkeypatch.setattr(demo_mod, "_load_audio_engines", lambda: (bad_asr, good_tts))
        result = infer_turn(
            (16000, np.zeros(16000, dtype=np.float32)),
            "base",
            None,
            "uuid-A",
        )
        assert result.status_msg == "Could not decode mic audio; please try again."

    def test_5_6_no_state_mutation_on_decode_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        bad_asr = _FakeASR(raise_exc=ValueError("decode fail"))
        good_tts = _FakeTTS()
        monkeypatch.setattr(demo_mod, "_load_audio_engines", lambda: (bad_asr, good_tts))
        infer_turn(
            (16000, np.zeros(16000, dtype=np.float32)),
            "base",
            None,
            "uuid-A",
        )
        s = get_session("uuid-A")
        assert s.turn_idx == 0
        assert s.episode_trace == []

    def test_5_7_session_capacity(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        for i in range(10):
            get_session(f"uuid-{i}")
        result = infer_turn(None, "base", None, "uuid-overflow", text_input="hi")
        assert result.status_msg == "Demo at capacity — try again in a minute."

    def test_5_8_env_step_error(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        # Pre-seed the env so set_next_raise applies to it.
        get_session("uuid-A")
        fake_env_factory[0].set_next_raise(RuntimeError("invalid_action"))
        result = infer_turn(None, "base", None, "uuid-A", text_input="hi")
        assert "Env rejected action" in result.status_msg

    def test_5_9_timeout_safe_defaults(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        fake_loader._model.set_next_raise(TimeoutError("60s"))
        result = infer_turn(None, "base", None, "uuid-A", text_input="hi")
        assert "timed out" in result.status_msg.lower()
        assert result.audio[0] == 16000


# ---------------------------------------------------------------------------
# build_demo / build_ui (UI graph mocked)
# ---------------------------------------------------------------------------


class _FakeGradioComponent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events: list[str] = []

    def change(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("change")

    def submit(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("submit")

    def click(self, *_args: Any, **_kwargs: Any) -> None:
        self.events.append("click")


class _FakeBlocks:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> _FakeBlocks:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class _FakeRow(_FakeBlocks):
    pass


class _FakeGradio:
    def __init__(self) -> None:
        self.components: list[tuple[str, _FakeGradioComponent]] = []

    def Blocks(self, **kwargs: Any) -> _FakeBlocks:
        return _FakeBlocks(**kwargs)

    def Row(self, *args: Any, **kwargs: Any) -> _FakeRow:
        return _FakeRow(**kwargs)

    def Markdown(self, *args: Any, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(text=args[0] if args else "", **kwargs)
        self.components.append(("Markdown", c))
        return c

    def Audio(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("Audio", c))
        return c

    def Textbox(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("Textbox", c))
        return c

    def Radio(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("Radio", c))
        return c

    def Dropdown(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("Dropdown", c))
        return c

    def DataFrame(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("DataFrame", c))
        return c

    def JSON(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("JSON", c))
        return c

    def Button(self, *args: Any, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(label=args[0] if args else "", **kwargs)
        self.components.append(("Button", c))
        return c

    def State(self, **kwargs: Any) -> _FakeGradioComponent:
        c = _FakeGradioComponent(**kwargs)
        self.components.append(("State", c))
        return c


@pytest.fixture
def fake_gradio(monkeypatch: pytest.MonkeyPatch) -> _FakeGradio:
    fake = _FakeGradio()
    monkeypatch.setattr(demo_mod, "_load_gradio", lambda: fake)
    return fake


class TestBuildDemo:
    def test_build_demo_returns_blocks(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        result = build_demo()
        assert isinstance(result, _FakeBlocks)

    def test_build_ui_alias(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        result = build_ui()
        assert isinstance(result, _FakeBlocks)

    def test_mounts_microphone_audio(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        mics = [
            c
            for name, c in fake_gradio.components
            if name == "Audio" and "microphone" in (c.kwargs.get("sources") or [])
        ]
        assert mics, "expected at least one microphone Audio component"

    def test_mounts_checkpoint_radio_with_choices(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        radios = [c for name, c in fake_gradio.components if name == "Radio"]
        assert radios
        assert radios[0].kwargs.get("choices") == ["base", "trained"]
        assert radios[0].kwargs.get("value") == "base"

    def test_greys_trained_when_lora_unavailable(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        fake_loader._trained_available = False
        build_demo()
        radios = [c for name, c in fake_gradio.components if name == "Radio"]
        assert radios[0].kwargs.get("choices") == ["base"]
        label = radios[0].kwargs.get("label", "")
        assert "Trained adapter unavailable" in label

    def test_drift_dropdown_includes_20_patterns_plus_none(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        dropdowns = [c for name, c in fake_gradio.components if name == "Dropdown"]
        assert dropdowns
        choices = dropdowns[0].kwargs.get("choices") or []
        assert None in choices
        assert len([c for c in choices if c is not None]) == 20

    def test_dataframe_columns(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        dfs = [c for name, c in fake_gradio.components if name == "DataFrame"]
        assert dfs
        headers = dfs[0].kwargs.get("headers") or []
        assert headers == [
            "turn_idx",
            "actor",
            "action_or_event",
            "tool_response_preview",
            "reward_delta",
        ]

    def test_audio_output_numpy_type(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        audios = [c for name, c in fake_gradio.components if name == "Audio"]
        # one mic + one speaker
        assert any(c.kwargs.get("type") == "numpy" for c in audios)

    def test_reset_button_present(
        self,
        fake_gradio: _FakeGradio,
        fake_drift_patterns: tuple[str, ...],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        build_demo()
        btns = [c for name, c in fake_gradio.components if name == "Button"]
        assert any("New episode" in str(b.kwargs.get("label", "")) for b in btns)


# ---------------------------------------------------------------------------
# Latency / IO invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_infer_turn_under_8s_budget(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        start = time.perf_counter()
        infer_turn(None, "base", None, "uuid-A", text_input="hi")
        elapsed = time.perf_counter() - start
        assert elapsed < 8.0

    def test_infer_turn_does_not_call_push_to_hub(
        self,
        fake_audio: tuple[_FakeASR, _FakeTTS],
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ensure no module-level call to the hub functions.
        called: list[str] = []

        def _trip(*_a: Any, **_k: Any) -> None:
            called.append("push")
            raise AssertionError("push_to_hub must not be called")

        # If huggingface_hub is importable, patch its main upload entry.
        import importlib.util as _ilu

        if _ilu.find_spec("huggingface_hub") is not None:
            monkeypatch.setattr(
                "huggingface_hub.HfApi.upload_file",
                _trip,
                raising=False,
            )
        infer_turn(None, "base", None, "uuid-A", text_input="hi")
        assert called == []

    def test_dataclasses_are_frozen_or_intentionally_mutable(self) -> None:
        # TraceRow + InferTurnResult must be frozen.
        row = TraceRow(
            turn_idx=1,
            actor="user",
            action_or_event="x",
            tool_response_preview="",
            reward_delta=0.0,
        )
        # `setattr` triggers the frozen-dataclass __setattr__ guard at
        # runtime; mypy --strict on tests folder is not enforced so we use
        # the dynamic call rather than direct attr assignment.
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.turn_idx = 2
        # DemoSessionState is intentionally mutable per spec §4.1; verify it
        # remains so (regression guard).
        s = DemoSessionState(session_id="x", env=MagicMock())
        s.turn_idx = 5
        assert s.turn_idx == 5


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModelLoaderBoot:
    def test_boot_loads_base_and_mounts_lora(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_tokenizer = MagicMock(name="tokenizer_cls")
        fake_tokenizer.from_pretrained = MagicMock(return_value="tokenizer")
        fake_model_cls = MagicMock(name="model_cls")
        fake_model_cls.from_pretrained = MagicMock(return_value="base_model")
        fake_transformers = MagicMock(
            AutoTokenizer=fake_tokenizer,
            AutoModelForCausalLM=fake_model_cls,
        )
        monkeypatch.setattr(demo_mod, "_load_transformers", lambda: fake_transformers)

        peft_model_cls = MagicMock(name="PeftModel")
        peft_model_cls.from_pretrained = MagicMock(return_value="wrapped")
        fake_peft = MagicMock(PeftModel=peft_model_cls)
        monkeypatch.setattr(demo_mod, "_load_peft_module", lambda: fake_peft)
        monkeypatch.setattr(demo_mod, "_load_hf_hub_errors", lambda: (FileNotFoundError,))

        loader = ModelLoader()
        loader.boot()
        assert loader.is_trained_available() is True
        # second boot is a no-op
        loader.boot()
        assert fake_model_cls.from_pretrained.call_count == 1

    def test_boot_404_falls_to_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_tokenizer = MagicMock(from_pretrained=MagicMock(return_value="t"))
        fake_model_cls = MagicMock(from_pretrained=MagicMock(return_value="base"))
        fake_transformers = MagicMock(
            AutoTokenizer=fake_tokenizer, AutoModelForCausalLM=fake_model_cls
        )
        monkeypatch.setattr(demo_mod, "_load_transformers", lambda: fake_transformers)

        class _FakeNotFound(Exception):
            pass

        peft_model_cls = MagicMock(name="PeftModel")
        peft_model_cls.from_pretrained = MagicMock(side_effect=_FakeNotFound("404"))
        fake_peft = MagicMock(PeftModel=peft_model_cls)
        monkeypatch.setattr(demo_mod, "_load_peft_module", lambda: fake_peft)
        monkeypatch.setattr(demo_mod, "_load_hf_hub_errors", lambda: (_FakeNotFound,))

        loader = ModelLoader()
        loader.boot()
        assert loader.is_trained_available() is False

    def test_boot_checkpoint_mismatch_falls_to_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_tokenizer = MagicMock(from_pretrained=MagicMock(return_value="t"))
        fake_model_cls = MagicMock(from_pretrained=MagicMock(return_value="base"))
        fake_transformers = MagicMock(
            AutoTokenizer=fake_tokenizer, AutoModelForCausalLM=fake_model_cls
        )
        monkeypatch.setattr(demo_mod, "_load_transformers", lambda: fake_transformers)

        peft_model_cls = MagicMock(name="PeftModel")
        peft_model_cls.from_pretrained = MagicMock(
            side_effect=CheckpointMismatchError("hash mismatch")
        )
        fake_peft = MagicMock(PeftModel=peft_model_cls)
        monkeypatch.setattr(demo_mod, "_load_peft_module", lambda: fake_peft)
        monkeypatch.setattr(demo_mod, "_load_hf_hub_errors", lambda: (FileNotFoundError,))

        loader = ModelLoader()
        loader.boot()
        assert loader.is_trained_available() is False

    def test_boot_unknown_exception_falls_to_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_tokenizer = MagicMock(from_pretrained=MagicMock(return_value="t"))
        fake_model_cls = MagicMock(from_pretrained=MagicMock(return_value="base"))
        fake_transformers = MagicMock(
            AutoTokenizer=fake_tokenizer, AutoModelForCausalLM=fake_model_cls
        )
        monkeypatch.setattr(demo_mod, "_load_transformers", lambda: fake_transformers)

        peft_model_cls = MagicMock(name="PeftModel")
        peft_model_cls.from_pretrained = MagicMock(side_effect=RuntimeError("unknown"))
        fake_peft = MagicMock(PeftModel=peft_model_cls)
        monkeypatch.setattr(demo_mod, "_load_peft_module", lambda: fake_peft)
        monkeypatch.setattr(demo_mod, "_load_hf_hub_errors", lambda: (FileNotFoundError,))

        loader = ModelLoader()
        loader.boot()
        assert loader.is_trained_available() is False

    def test_boot_missing_transformers_classes_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_transformers = MagicMock(spec=[])  # empty: no AutoTokenizer/etc
        monkeypatch.setattr(demo_mod, "_load_transformers", lambda: fake_transformers)
        loader = ModelLoader()
        with pytest.raises(TrainedAdapterMissingError):
            loader.boot()

    def test_mount_lora_returns_false_when_peft_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_peft = MagicMock(spec=[])  # no PeftModel attribute
        monkeypatch.setattr(demo_mod, "_load_peft_module", lambda: fake_peft)
        loader = ModelLoader()
        loader._model = MagicMock()
        result = loader._mount_lora()
        assert result is False


class TestDoGenerateResultShapes:
    def test_str_result_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader = ModelLoader()
        fake_model = MagicMock()
        fake_model.generate = MagicMock(return_value="hello")
        loader._model = fake_model
        out = loader._do_generate("p", 16, 0.2, 0.95)
        assert out == "hello"

    def test_dict_with_text_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = ModelLoader()
        fake_model = MagicMock()
        fake_model.generate = MagicMock(return_value={"text": "hi"})
        loader._model = fake_model
        assert loader._do_generate("p", 16, 0.2, 0.95) == "hi"

    def test_list_first_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = ModelLoader()
        fake_model = MagicMock()
        fake_model.generate = MagicMock(return_value=["first", "second"])
        loader._model = fake_model
        assert loader._do_generate("p", 16, 0.2, 0.95) == "first"

    def test_unknown_type_str_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = ModelLoader()
        fake_model = MagicMock()
        fake_model.generate = MagicMock(return_value=42)
        loader._model = fake_model
        assert loader._do_generate("p", 16, 0.2, 0.95) == "42"


class TestFormatMessages:
    def test_includes_role_markers(self) -> None:
        out = demo_mod._format_messages([{"role": "user", "content": "hi"}])
        assert "<|user|>hi" in out
        assert "<|assistant|>" in out

    def test_default_role_user(self) -> None:
        out = demo_mod._format_messages([{"content": "x"}])
        assert "<|user|>x" in out


class TestWarmupOnBoot:
    def test_warmup_calls_engines(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_audio: tuple[_FakeASR, _FakeTTS],
    ) -> None:
        loader = ModelLoader()
        loader._model = MagicMock()  # mark as booted
        monkeypatch.setattr(demo_mod, "get_model_loader", lambda: loader)
        demo_mod.warmup_on_boot()
        # ASR + TTS warmup are best-effort; just confirm the call did not raise.

    def test_warmup_swallows_audio_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader = ModelLoader()
        loader._model = MagicMock()
        monkeypatch.setattr(demo_mod, "get_model_loader", lambda: loader)

        bad_asr = MagicMock()
        bad_asr.warmup = MagicMock(side_effect=RuntimeError("asr fail"))
        bad_tts = MagicMock()
        bad_tts.warmup = MagicMock(side_effect=RuntimeError("tts fail"))
        monkeypatch.setattr(demo_mod, "_load_audio_engines", lambda: (bad_asr, bad_tts))
        # Should not raise.
        demo_mod.warmup_on_boot()


class TestGetModelLoaderSingleton:
    def test_returns_same_instance(self) -> None:
        a = get_model_loader()
        b = get_model_loader()
        assert a is b


class TestTtsAndAudioFallbacks:
    def test_tts_failure_returns_safe_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_env_factory: list[_FakeEnv],
        fake_loader: ModelLoader,
    ) -> None:
        bad_tts = _FakeTTS(raise_exc=RuntimeError("tts boom"))
        good_asr = _FakeASR()
        monkeypatch.setattr(demo_mod, "_load_audio_engines", lambda: (good_asr, bad_tts))
        result = infer_turn(None, "base", None, "uuid-A", text_input="hi")
        # success path with fallback silence
        assert result.audio[0] == 16000
        assert len(result.audio[1]) == 16000

    def test_empty_reply_returns_silence(self) -> None:
        sr, audio = demo_mod._do_tts(MagicMock(), "")
        assert sr == 16000
        assert len(audio) == 16000


class TestResetPreservesCheckpoint:
    def test_reset_keeps_current_checkpoint(
        self, fake_env_factory: list[_FakeEnv]
    ) -> None:
        s = get_session("uuid-A")
        s.current_checkpoint = "trained"
        fresh = reset_session("uuid-A")
        assert fresh.current_checkpoint == "trained"


class TestModuleSurface:
    def test_no_pragma_violations(self) -> None:
        text = Path(demo_mod.__file__).read_text(encoding="utf-8")
        forbidden_marker_a = "type" + ": " + "ignore"
        forbidden_marker_b = "# " + "noqa"
        assert forbidden_marker_a not in text
        assert forbidden_marker_b not in text
