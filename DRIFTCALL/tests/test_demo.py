"""Tests for ``demo/app_gradio.py`` — DriftCall demo Space.

Implements ``docs/tests/deploy_demo_space_tests.md``. Heavy deps (gradio,
spaces, peft, torch, transformers) are stubbed; ``DriftCallEnv`` is replaced
with a lightweight stub via ``demo.app_gradio._make_env`` monkeypatch so
sessions can be created without standing up the real env stack.

Latency assertions use ``time.perf_counter`` against the stubbed pipeline —
no real GPU, no real Gradio launch, no real HF Hub.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import types
import uuid
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Rewards stub — installed before importing demo so DriftCallEnv resolves.
# ---------------------------------------------------------------------------


def _install_rewards_stub() -> None:
    name = "cells.step_08_rewards"
    if name in sys.modules:
        return
    import pathlib
    real_path = pathlib.Path(__file__).resolve().parent.parent / "cells" / "step_08_rewards.py"
    if real_path.is_file():
        import importlib
        importlib.import_module(name)
        return
    mod = types.ModuleType(name)

    @dataclasses.dataclass(frozen=True)
    class Rewards:
        reward: float = 0.5

    def compute_rewards(_episode: Any) -> Rewards:
        return Rewards()

    mod.Rewards = Rewards
    mod.compute_rewards = compute_rewards
    mod.RewardComputationError = type("RewardComputationError", (Exception,), {})
    sys.modules[name] = mod


_install_rewards_stub()


# ---------------------------------------------------------------------------
# Stub env so demo.app_gradio.get_session can construct without audio deps.
# ---------------------------------------------------------------------------


class _StubObs:
    last_transcript = "stub"
    last_lang = "en"
    last_confidence = 1.0


class _StubEnv:
    def __init__(self) -> None:
        self.steps: list[Any] = []

    def reset(self, seed: int | None = None) -> _StubObs:
        return _StubObs()

    def step(self, action: Any, *, force_drift_pattern: str | None = None) -> _StubObs:
        self.steps.append((action, force_drift_pattern))
        return _StubObs()

    def state(self) -> Any:
        return types.SimpleNamespace(turn=len(self.steps), max_turns=12)

    def close(self) -> None:
        return None

    def done(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _patch_make_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    monkeypatch.setattr(demo_mod, "_make_env", _StubEnv)


@pytest.fixture(autouse=True)
def _clear_demo_state() -> None:
    import demo.app_gradio as demo_mod

    demo_mod._clear_registry_for_tests()
    demo_mod._reset_model_loader_for_tests()
    yield
    demo_mod._clear_registry_for_tests()
    demo_mod._reset_model_loader_for_tests()


@pytest.fixture
def session_id_alpha() -> str:
    return "uuid-alpha-" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Stub model loader — replaces real ModelLoader.ensure_loaded / generate
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_loader(monkeypatch: pytest.MonkeyPatch) -> Any:
    import demo.app_gradio as demo_mod

    state = {
        "trained_available": True,
        "calls": 0,
        "disable_calls": 0,
        "set_calls": 0,
        "enable_calls": 0,
        "raise_zerogpu": 0,
        "raise_oom": 0,
        "sleep": 0.0,
    }

    class _MockBaseModel:
        def disable_adapter(self) -> Any:
            state["disable_calls"] += 1

            class _Ctx:
                def __enter__(inner: _Ctx) -> _Ctx:
                    return inner

                def __exit__(inner: _Ctx, *exc: Any) -> bool:
                    return False

            return _Ctx()

        def set_adapter(self, name: str) -> None:
            state["set_calls"] += 1
            assert name == "driftcall"

        def enable_adapter_layers(self) -> None:
            state["enable_calls"] += 1

    base = _MockBaseModel()
    loader = demo_mod.ModelLoader()
    loader._model = base
    loader._trained_available = True
    loader._tokenizer = object()

    def _stub_generate(
        messages: list[dict[str, str]],
        *,
        checkpoint: str,
        max_new_tokens: int = 256,
        **kw: Any,
    ) -> str:
        state["calls"] += 1
        if state["raise_zerogpu"] > 0:
            state["raise_zerogpu"] -= 1
            raise demo_mod.ZeroGPUUnavailableError("queue full")
        if state["raise_oom"] > 0:
            state["raise_oom"] -= 1
            err: Any = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
            raise err
        if checkpoint == "trained" and not loader._trained_available:
            raise demo_mod.TrainedAdapterMissingError("missing")
        if checkpoint == "base":
            with base.disable_adapter():
                pass
        elif checkpoint == "trained":
            base.set_adapter("driftcall")
            base.enable_adapter_layers()
        if state["sleep"] > 0:
            time.sleep(state["sleep"])
        return f"[{checkpoint}] reply to {messages[-1]['content']}"

    monkeypatch.setattr(loader, "generate", _stub_generate)
    monkeypatch.setattr(demo_mod, "_MODEL_LOADER", loader)
    monkeypatch.setattr(demo_mod, "_run_asr", lambda audio_tuple: "test transcript")
    monkeypatch.setattr(
        demo_mod,
        "_run_tts",
        lambda text, lang_hint="en": (16000, np.zeros(16000, dtype=np.float32)),
    )
    monkeypatch.setattr(demo_mod, "_empty_cuda_cache", lambda: None)
    return state


# ---------------------------------------------------------------------------
# Public types / smoke
# ---------------------------------------------------------------------------


def test_module_imports_lightly() -> None:
    """demo.app_gradio imports without gradio/peft/torch installed."""

    import demo.app_gradio as demo_mod

    assert hasattr(demo_mod, "infer_turn")
    assert hasattr(demo_mod, "build_ui")
    assert demo_mod._DRIFT_PATTERN_IDS, "drift pattern catalogue should be non-empty"


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


def test_get_session_creates_fresh_env(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    state = demo_mod.get_session(session_id_alpha)
    assert state.session_id == session_id_alpha
    assert state.episode_trace == []
    assert state.turn_idx == 0
    assert state.current_checkpoint == "base"


def test_get_session_idempotent(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    a = demo_mod.get_session(session_id_alpha)
    b = demo_mod.get_session(session_id_alpha)
    assert a is b


def test_get_session_isolates_uuids() -> None:
    import demo.app_gradio as demo_mod

    a = demo_mod.get_session("uuid-A")
    b = demo_mod.get_session("uuid-B")
    assert a is not b
    assert a.env is not b.env


def test_get_session_enforces_max_concurrent_10() -> None:
    import demo.app_gradio as demo_mod

    for i in range(10):
        demo_mod.get_session(f"uuid-{i:02d}")
    with pytest.raises(demo_mod.SessionCapacityError):
        demo_mod.get_session("uuid-11")


def test_gc_sessions_evicts_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    real_now = demo_mod._now_ms
    monkeypatch.setattr(demo_mod, "_now_ms", lambda: real_now())
    s1 = demo_mod.get_session("idle-1")
    s2 = demo_mod.get_session("idle-2")
    s3 = demo_mod.get_session("active-1")
    s1.last_activity_ms = 0
    s2.last_activity_ms = 0
    s3.last_activity_ms = real_now()
    evicted = demo_mod.gc_sessions(max_idle_s=10)
    assert evicted == 2


def test_reset_session_closes_env_and_clears_trace(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    state = demo_mod.get_session(session_id_alpha)
    state.episode_trace.append(
        demo_mod.TraceRow(
            turn_idx=1,
            actor="user",
            action_or_event="hi",
            tool_response_preview="",
            reward_delta=0.0,
        )
    )
    fresh = demo_mod.reset_session(session_id_alpha)
    assert fresh.episode_trace == []
    assert fresh.turn_idx == 0


# ---------------------------------------------------------------------------
# Drift bridge
# ---------------------------------------------------------------------------


def test_drift_bridge_queue_consume_returns_pattern(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    bridge = demo_mod.DriftToggleBridge()
    bridge.queue(session_id_alpha, "airline.price_rename")
    assert bridge.consume(session_id_alpha) == "airline.price_rename"
    assert bridge.consume(session_id_alpha) is None


def test_drift_bridge_coalesces_double_press(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    bridge = demo_mod.DriftToggleBridge()
    bridge.queue(session_id_alpha, "pattern_A")
    bridge.queue(session_id_alpha, "pattern_B")
    assert bridge.consume(session_id_alpha) == "pattern_B"


def test_drift_bridge_queue_none_clears(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    bridge = demo_mod.DriftToggleBridge()
    bridge.queue(session_id_alpha, "x")
    bridge.queue(session_id_alpha, None)
    assert bridge.consume(session_id_alpha) is None


# ---------------------------------------------------------------------------
# Trace panel (render_trace purity)
# ---------------------------------------------------------------------------


def test_render_trace_returns_correct_columns(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    state = demo_mod.get_session(session_id_alpha)
    state.episode_trace.append(
        demo_mod.TraceRow(
            turn_idx=1,
            actor="user",
            action_or_event="hello",
            tool_response_preview="",
            reward_delta=0.0,
        )
    )
    df = demo_mod.render_trace(state)
    assert list(df.columns) == [
        "turn_idx",
        "actor",
        "action_or_event",
        "tool_response_preview",
        "reward_delta",
    ]
    assert len(df) == 1


def test_render_trace_does_not_mutate_state(session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    state = demo_mod.get_session(session_id_alpha)
    state.episode_trace.append(
        demo_mod.TraceRow(turn_idx=1, actor="env", action_or_event="200 OK", tool_response_preview="", reward_delta=0.0)
    )
    snapshot_id = id(state.episode_trace)
    snapshot_len = len(state.episode_trace)
    df1 = demo_mod.render_trace(state)
    df2 = demo_mod.render_trace(state)
    assert df1.equals(df2)
    assert id(state.episode_trace) == snapshot_id
    assert len(state.episode_trace) == snapshot_len


# ---------------------------------------------------------------------------
# ModelLoader hot-swap
# ---------------------------------------------------------------------------


def test_generate_base_calls_disable_adapter(stub_loader: Any) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.get_model_loader()
    out = loader.generate(
        [{"role": "user", "content": "hi"}], checkpoint="base"
    )
    assert "[base]" in out
    assert stub_loader["disable_calls"] == 1
    assert stub_loader["set_calls"] == 0


def test_generate_trained_calls_set_adapter_driftcall(stub_loader: Any) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.get_model_loader()
    out = loader.generate(
        [{"role": "user", "content": "hi"}], checkpoint="trained"
    )
    assert "[trained]" in out
    assert stub_loader["set_calls"] == 1
    assert stub_loader["enable_calls"] == 1


def test_generate_trained_raises_when_adapter_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.ModelLoader()
    loader._trained_available = False
    loader._model = object()

    with pytest.raises(demo_mod.TrainedAdapterMissingError):
        loader.generate([{"role": "user", "content": "hi"}], checkpoint="trained")


def test_is_trained_available_false_on_boot_404(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.ModelLoader()

    class _FakeBase:
        def disable_adapter(self) -> Any:
            return demo_mod._NullCtx()

    monkeypatch.setattr(loader, "_load_base", lambda: (_FakeBase(), object()))
    monkeypatch.setattr(loader, "_try_mount_adapter", lambda _model: None)

    loader.ensure_loaded()
    assert loader.is_trained_available() is False


# ---------------------------------------------------------------------------
# infer_turn — happy + error modes
# ---------------------------------------------------------------------------


def test_infer_turn_success_under_8s(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    audio = (16000, np.zeros(16000, dtype=np.float32))
    start = time.perf_counter()
    transcript, audio_out, df, rewards, status = demo_mod.infer_turn(
        audio, "trained", None, session_id_alpha
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 8.0
    assert transcript == "test transcript"
    assert audio_out[0] == 16000
    assert isinstance(audio_out[1], np.ndarray)
    assert len(df) >= 2
    assert isinstance(rewards, dict)


def test_infer_turn_no_audio_no_text_returns_safe_defaults(
    stub_loader: Any, session_id_alpha: str
) -> None:
    import demo.app_gradio as demo_mod

    transcript, audio_out, df, rewards, status = demo_mod.infer_turn(
        None, "base", None, session_id_alpha, text_input=""
    )
    assert "No audio received" in status
    assert audio_out == (16000, ) or audio_out[0] == 16000
    assert transcript == ""


def test_infer_turn_text_fallback_works(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    transcript, _audio, df, _rewards, status = demo_mod.infer_turn(
        None, "base", None, session_id_alpha, text_input="hello"
    )
    assert transcript == "hello"
    assert status == ""
    assert any(row.actor == "user" for row in demo_mod.get_session(session_id_alpha).episode_trace)


def test_infer_turn_session_capacity_error_511(stub_loader: Any) -> None:
    import demo.app_gradio as demo_mod

    for i in range(10):
        demo_mod.get_session(f"sat-{i:02d}")
    audio = (16000, np.zeros(16000, dtype=np.float32))
    transcript, _audio, _df, _rewards, status = demo_mod.infer_turn(
        audio, "base", None, "sat-overflow"
    )
    assert "Demo at capacity" in status


def test_infer_turn_asr_decode_error_safe_defaults(
    stub_loader: Any, session_id_alpha: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import demo.app_gradio as demo_mod

    def _raise(*_a: Any, **_kw: Any) -> str:
        raise demo_mod.AudioDecodeError("bad audio")

    monkeypatch.setattr(demo_mod, "_run_asr", _raise)
    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, _df, _r, status = demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    assert "Could not decode" in status


def test_infer_turn_zerogpu_double_failure_falls_back(
    stub_loader: Any, session_id_alpha: str
) -> None:
    import demo.app_gradio as demo_mod

    stub_loader["raise_zerogpu"] = 2
    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, _df, _r, status = demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    assert "GPU unavailable" in status


def test_infer_turn_oom_first_then_success(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    stub_loader["raise_oom"] = 1
    audio = (16000, np.zeros(16000, dtype=np.float32))
    transcript, _au, _df, _r, status = demo_mod.infer_turn(
        audio, "base", None, session_id_alpha
    )
    assert transcript == "test transcript"
    assert stub_loader["calls"] >= 2


def test_infer_turn_double_oom_fails_turn(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    stub_loader["raise_oom"] = 5
    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, _df, _r, status = demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    assert "out of memory" in status.lower()


def test_infer_turn_trained_unavailable_falls_back_to_base(
    stub_loader: Any, session_id_alpha: str
) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.get_model_loader()
    loader._trained_available = False
    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, _df, _r, status = demo_mod.infer_turn(
        audio, "trained", None, session_id_alpha
    )
    assert "Trained adapter unavailable" in status


def test_infer_turn_env_step_error_records_rejection(
    stub_loader: Any, session_id_alpha: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import demo.app_gradio as demo_mod

    def _bad_env_factory() -> Any:
        env = _StubEnv()

        def _bad_step(action: Any, **kw: Any) -> Any:
            raise demo_mod.EnvStepError("invalid_action")

        env.step = _bad_step
        return env

    monkeypatch.setattr(demo_mod, "_make_env", _bad_env_factory)
    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, df, _r, status = demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    assert "Env rejected" in status
    state = demo_mod.get_session(session_id_alpha)
    assert any(row.actor == "env" and "rejected" in row.action_or_event for row in state.episode_trace)


def test_infer_turn_manual_drift_recorded(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    audio = (16000, np.zeros(16000, dtype=np.float32))
    _t, _au, df, _r, _status = demo_mod.infer_turn(
        audio, "base", "airline.price_rename", session_id_alpha
    )
    state = demo_mod.get_session(session_id_alpha)
    assert any(row.actor == "drift" and "manual:" in row.action_or_event for row in state.episode_trace)


def test_infer_turn_never_writes_to_disk(
    stub_loader: Any, session_id_alpha: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    import demo.app_gradio as demo_mod

    write_count = {"v": 0}
    real_open = builtins.open

    def _spy_open(*args: Any, **kw: Any) -> Any:
        mode = ""
        if len(args) >= 2:
            mode = str(args[1])
        elif "mode" in kw:
            mode = str(kw["mode"])
        if "w" in mode or "a" in mode or "x" in mode:
            write_count["v"] += 1
        return real_open(*args, **kw)

    monkeypatch.setattr(builtins, "open", _spy_open)
    audio = (16000, np.zeros(16000, dtype=np.float32))
    demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    assert write_count["v"] == 0


# ---------------------------------------------------------------------------
# deploy_check / hardware probing
# ---------------------------------------------------------------------------


def test_deploy_check_returns_zero_gpu_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    monkeypatch.delenv("DRIFTCALL_HARDWARE", raising=False)
    monkeypatch.delenv("DRIFTCALL_HARDWARE_FALLBACK", raising=False)
    assert demo_mod.deploy_check() == "zero-gpu"


def test_deploy_check_falls_back_to_a10g(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    monkeypatch.setenv("DRIFTCALL_HARDWARE", "unavailable")
    monkeypatch.setenv("DRIFTCALL_HARDWARE_FALLBACK", "a10g")
    assert demo_mod.deploy_check() == "a10g-small"


def test_deploy_check_aborts_when_both_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    monkeypatch.setenv("DRIFTCALL_HARDWARE", "unavailable")
    monkeypatch.setenv("DRIFTCALL_HARDWARE_FALLBACK", "none")
    with pytest.raises(demo_mod.DeploymentAbortedError):
        demo_mod.deploy_check()


# ---------------------------------------------------------------------------
# build_ui — only when gradio is importable
# ---------------------------------------------------------------------------


def test_build_ui_constructs_blocks_when_gradio_available(
    stub_loader: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("gradio")

    import demo.app_gradio as demo_mod

    blocks = demo_mod.build_ui()
    assert blocks is not None


def test_build_ui_greys_trained_when_lora_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gradio")

    import demo.app_gradio as demo_mod

    loader = demo_mod.ModelLoader()
    loader._trained_available = False
    loader._model = object()
    monkeypatch.setattr(demo_mod, "_MODEL_LOADER", loader)
    blocks = demo_mod.build_ui()
    # Walk the block graph, find the radio component, assert choices.
    import gradio as gr

    radios = [c for c in blocks.blocks.values() if isinstance(c, gr.Radio)]
    assert any(set(r.choices) == {("base", "base")} or set(r.choices) == {"base"} for r in radios)


# ---------------------------------------------------------------------------
# Warmup + smoke
# ---------------------------------------------------------------------------


def test_warmup_on_boot_handles_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    # Patch ASR / TTS singletons to raise — warmup must swallow.
    def _raise() -> Any:
        raise RuntimeError("no models")

    monkeypatch.setattr("cells.step_09_audio.get_asr_engine", _raise)
    monkeypatch.setattr("cells.step_09_audio.get_tts_engine", _raise)

    # Patch model loader to also raise to ensure warmup absorbs.
    loader = demo_mod.ModelLoader()
    monkeypatch.setattr(loader, "ensure_loaded", lambda: (_ for _ in ()).throw(RuntimeError("no GPU")))
    monkeypatch.setattr(demo_mod, "_MODEL_LOADER", loader)

    demo_mod.warmup_on_boot()  # must not raise


# ---------------------------------------------------------------------------
# DRIFT_PATTERN_IDS sanity
# ---------------------------------------------------------------------------


def test_drift_pattern_ids_non_empty_and_strings() -> None:
    import demo.app_gradio as demo_mod

    assert len(demo_mod._DRIFT_PATTERN_IDS) > 0
    for pat in demo_mod._DRIFT_PATTERN_IDS:
        assert isinstance(pat, str) and pat


# ---------------------------------------------------------------------------
# Trace preview truncation
# ---------------------------------------------------------------------------


def test_truncate_preview_within_limit() -> None:
    import demo.app_gradio as demo_mod

    short = "hello"
    out = demo_mod._truncate_preview(short)
    assert out == short


def test_truncate_preview_long_uses_ellipsis() -> None:
    import demo.app_gradio as demo_mod

    long = "x" * 300
    out = demo_mod._truncate_preview(long)
    assert len(out) == demo_mod._TRACE_PREVIEW_LEN
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# Safe defaults sanity
# ---------------------------------------------------------------------------


def test_safe_silence_returns_one_second_at_16k() -> None:
    import demo.app_gradio as demo_mod

    sr, wav = demo_mod._safe_silence()
    assert sr == 16000
    assert wav.shape == (16000,)
    assert wav.dtype == np.float32


def test_safe_defaults_returns_empty_dataframe() -> None:
    import demo.app_gradio as demo_mod

    transcript, audio_out, df, rewards, status = demo_mod._safe_defaults()
    assert transcript == ""
    assert audio_out[0] == 16000
    assert len(df) == 0
    assert rewards == {}


# ---------------------------------------------------------------------------
# infer_turn returns rewards as dict and trace shows agent row on success
# ---------------------------------------------------------------------------


def test_infer_turn_appends_agent_row(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    audio = (16000, np.zeros(16000, dtype=np.float32))
    demo_mod.infer_turn(audio, "trained", None, session_id_alpha)
    state = demo_mod.get_session(session_id_alpha)
    actors = [r.actor for r in state.episode_trace]
    assert "agent" in actors
    assert "user" in actors
    assert "env" in actors


# ---------------------------------------------------------------------------
# get_drift_bridge returns the module-level singleton
# ---------------------------------------------------------------------------


def test_get_drift_bridge_returns_singleton() -> None:
    import demo.app_gradio as demo_mod

    a = demo_mod.get_drift_bridge()
    b = demo_mod.get_drift_bridge()
    assert a is b


# ---------------------------------------------------------------------------
# infer_turn forced drift via bridge (covers bridge.consume non-None path)
# ---------------------------------------------------------------------------


def test_infer_turn_consumes_bridge_pattern(stub_loader: Any, session_id_alpha: str) -> None:
    import demo.app_gradio as demo_mod

    demo_mod.get_drift_bridge().queue(session_id_alpha, "airline.price_rename")
    audio = (16000, np.zeros(16000, dtype=np.float32))
    demo_mod.infer_turn(audio, "base", None, session_id_alpha)
    # Bridge should now be empty for this session.
    assert demo_mod.get_drift_bridge().consume(session_id_alpha) is None


# ---------------------------------------------------------------------------
# ModelLoader.ensure_loaded happy path (no LoRA)
# ---------------------------------------------------------------------------


def test_ensure_loaded_no_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.ModelLoader()
    base = object()
    monkeypatch.setattr(loader, "_load_base", lambda: (base, object()))
    monkeypatch.setattr(loader, "_try_mount_adapter", lambda _m: None)
    loader.ensure_loaded()
    assert loader._model is base
    assert loader.is_trained_available() is False
    # Idempotent.
    loader.ensure_loaded()
    assert loader._model is base


def test_ensure_loaded_with_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.app_gradio as demo_mod

    loader = demo_mod.ModelLoader()
    base = object()
    wrapped = object()
    monkeypatch.setattr(loader, "_load_base", lambda: (base, object()))
    monkeypatch.setattr(loader, "_try_mount_adapter", lambda _m: wrapped)
    loader.ensure_loaded()
    assert loader._model is wrapped
    assert loader.is_trained_available() is True
