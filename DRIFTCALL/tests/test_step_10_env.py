"""Tests for cells/step_10_env.py.

Implements docs/tests/env_tests.md (35 unit + 6 property + 4 integration).
The reward layer (cells/step_08_rewards) is mocked via sys.modules to keep
this test module independent of step_08's own implementation timeline.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import types
from dataclasses import FrozenInstanceError
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftEvent,
    GoalSpec,
)

# ---------------------------------------------------------------------------
# Stub rewards module (installed before importing step_10_env)
# ---------------------------------------------------------------------------


def _install_rewards_stub() -> types.ModuleType:
    """Install a minimal cells.step_08_rewards stub if the real module is absent.

    Returns the stub module so individual tests can monkeypatch
    ``compute_rewards`` directly.
    """
    name = "cells.step_08_rewards"
    if name in sys.modules:
        return sys.modules[name]

    # If the real implementation is on disk, import it directly so we don't
    # poison sys.modules for tests that run in the same pytest session and
    # need the real symbols (e.g. AVAILABLE_TOOL_REGISTRY in test_step_08).
    real_path = pathlib.Path(__file__).resolve().parent.parent / "cells" / "step_08_rewards.py"
    if real_path.is_file():
        import importlib
        return importlib.import_module(name)

    mod = types.ModuleType(name)

    @dataclasses.dataclass(frozen=True)
    class Rewards:
        r1: float = 1.0
        r2: float = 0.5
        r3: float = 1.0
        r4: float = 1.0
        r5: float = 0.0
        reward: float = 0.85

    def compute_rewards(episode: Any) -> Rewards:
        # Reflect terminated_by into r1 so tests can assert ABORT/TIMEOUT path.
        if episode.terminated_by in ("ABORT", "TIMEOUT", "ANTI_HACK"):
            return Rewards(r1=0.0, reward=0.20)
        return Rewards()

    class RewardComputationError(Exception):
        pass

    class Episode:
        pass

    mod.Rewards = Rewards
    mod.Episode = Episode
    mod.compute_rewards = compute_rewards
    mod.RewardComputationError = RewardComputationError
    sys.modules[name] = mod
    return mod


_REWARDS_STUB = _install_rewards_stub()


from cells.step_10_env import (  # noqa: E402
    AudioPipelineError,
    ConcurrentStepError,
    DriftCallEnv,
    DriftCallEnvError,
    DriftInjectionError,
    EnvClosedError,
    EnvNotReadyError,
    EpisodeAlreadyTerminalError,
    EpisodeNotTerminalError,
    InvalidActionError,
    InvalidConfigError,
    RewardComputationError,
    UnknownDomainError,
    UnknownToolError,
)

# ---------------------------------------------------------------------------
# Stub TTS / ASR engines (mirror env_tests.md §5 stubs)
# ---------------------------------------------------------------------------


class StubTTS:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.raise_on_next: bool = False

    def synthesize(
        self,
        text: str,
        language_code: str,
        voice_pack: Any | None = None,
        *,
        seed: int = 0,
        sample_rate_hz: int = 16000,
    ) -> bytes:
        if self.raise_on_next:
            raise RuntimeError("synthetic TTS failure")
        self.calls.append((text, language_code, voice_pack, seed, sample_rate_hz))
        return f"WAV[{text}:{language_code}]".encode()


@dataclasses.dataclass(frozen=True)
class StubTranscript:
    text: str
    language_detected: str
    confidence: float
    duration_s: float


class StubASR:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.raise_on_next: bool = False

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: str | None,
        *,
        beam_size: int = 1,
        vad_filter: bool = True,
        max_duration_s: float = 30.0,
    ) -> StubTranscript:
        if self.raise_on_next:
            raise RuntimeError("synthetic ASR failure")
        self.calls.append(audio_bytes)
        return StubTranscript(
            text="shaam ko, 7 baje",
            language_detected="hinglish",
            confidence=0.82,
            duration_s=1.250,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_speak(message: str = "hello") -> DriftCallAction:
    return DriftCallAction(action_type=ActionType.SPEAK, message=message)


def _valid_submit(confidence: float = 0.8, message: str = "done") -> DriftCallAction:
    return DriftCallAction(
        action_type=ActionType.SUBMIT, confidence=confidence, message=message
    )


def _empty_scheduler(stage: int, episode_seed: int, goal: GoalSpec) -> tuple:
    return ()


def _scripted_scheduler(events: tuple[DriftEvent, ...]) -> Any:
    def _sched(stage: int, episode_seed: int, goal: GoalSpec) -> tuple[DriftEvent, ...]:
        return events
    return _sched


# =====================================================================
# 1.1 __init__ + EnvConfig.from_mapping (U1–U9)
# =====================================================================


def test_U1_init_default_config_ok() -> None:
    env = DriftCallEnv()
    assert env._config.curriculum_stage == 1
    assert env._config.language_weights == {
        "en": 0.4, "hinglish": 0.4, "hi": 0.1, "ta": 0.05, "kn": 0.05
    }
    assert env._config.audio_boundary_enabled is False
    assert env._state is None


def test_U2_init_rejects_unknown_key() -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv({"curriculum_stage": 1, "frobnicate": True})
    assert "frobnicate" in str(ei.value)


@pytest.mark.parametrize("bad_stage", [0, 4, -1, "1", 1.0, None, True])
def test_U3_init_rejects_invalid_stage(bad_stage: Any) -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv({"curriculum_stage": bad_stage})
    assert "curriculum_stage" in str(ei.value)


def test_U4_init_rejects_weights_wrong_sum() -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv({"language_weights": {"en": 0.5, "hinglish": 0.4}})
    assert "sum" in str(ei.value).lower()


def test_U5_init_rejects_weights_negative() -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv(
            {"language_weights": {"en": 0.6, "hinglish": 0.5, "hi": -0.1}}
        )
    assert "negative" in str(ei.value).lower()


def test_U6_init_rejects_audio_enabled_missing_tts() -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv(
            {
                "audio_boundary_enabled": True,
                "tts_engine": None,
                "asr_engine": StubASR(),
            }
        )
    assert "tts_engine" in str(ei.value)


def test_U7_init_rejects_audio_disabled_with_tts() -> None:
    with pytest.raises(InvalidConfigError) as ei:
        DriftCallEnv(
            {"audio_boundary_enabled": False, "tts_engine": StubTTS()}
        )
    assert "tts_engine" in str(ei.value)


def test_U8_init_is_pure_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch os.urandom and builtins.open to raise — __init__ must not invoke.
    monkeypatch.setattr("os.urandom", lambda _n: (_ for _ in ()).throw(AssertionError("urandom called")))
    real_open = open
    def _fail_open(*a: Any, **kw: Any) -> Any:
        raise AssertionError("open called")
    monkeypatch.setattr("builtins.open", _fail_open)
    try:
        env = DriftCallEnv({"curriculum_stage": 2})
        assert env._config.curriculum_stage == 2
    finally:
        monkeypatch.setattr("builtins.open", real_open)


def test_U9_init_stores_frozen_config_copy() -> None:
    weights = {"en": 0.4, "hinglish": 0.4, "hi": 0.1, "ta": 0.05, "kn": 0.05}
    env = DriftCallEnv({"language_weights": weights})
    weights["en"] = 0.99
    assert env._config.language_weights["en"] == 0.4
    assert env._config.__dataclass_params__.frozen is True
    with pytest.raises(FrozenInstanceError):
        env._config.curriculum_stage = 3  # type: ignore[misc]


# Extra E1 cases referenced in §4.1 matrix.
def test_init_invalid_audio_boundary_type() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"audio_boundary_enabled": "yes"})


def test_init_invalid_max_turns_override_value() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"max_turns_override": 0})


def test_init_invalid_max_turns_override_type() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"max_turns_override": "8"})


def test_init_invalid_scheduler_not_callable() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"scheduler": 42})  # type: ignore[dict-item]


def test_init_unknown_language_in_weights() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"language_weights": {"de": 1.0}})


def test_init_audio_disabled_with_asr_only() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv(
            {"audio_boundary_enabled": False, "asr_engine": StubASR()}
        )


def test_init_audio_enabled_missing_asr() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv(
            {
                "audio_boundary_enabled": True,
                "tts_engine": StubTTS(),
                "asr_engine": None,
            }
        )


def test_init_invalid_weight_type() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"language_weights": {"en": "0.4", "hinglish": "0.6"}})


def test_init_empty_weights() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv({"language_weights": {}})


def test_init_config_not_dict() -> None:
    with pytest.raises(InvalidConfigError):
        DriftCallEnv("nope")  # type: ignore[arg-type]


# =====================================================================
# 1.2 reset() — trajectory setup (U10–U17)
# =====================================================================


def test_U10_reset_stage1_sets_max_turns_8() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    obs = env.reset(seed=1)
    assert env._state is not None
    assert env._state.max_turns == 8
    assert obs.budget_remaining == 8
    assert obs.turn == 0


def test_U11_reset_stage2_sets_max_turns_12() -> None:
    env = DriftCallEnv({"curriculum_stage": 2, "scheduler": _empty_scheduler})
    obs = env.reset(seed=1)
    assert env._state is not None
    assert env._state.max_turns == 12
    assert obs.budget_remaining == 12


def test_U12_reset_stage3_sets_max_turns_16() -> None:
    env = DriftCallEnv({"curriculum_stage": 3, "scheduler": _empty_scheduler})
    obs = env.reset(seed=1)
    assert env._state is not None
    assert env._state.max_turns == 16
    assert obs.budget_remaining == 16


def test_U13_reset_populates_curriculum_stage_on_state() -> None:
    env = DriftCallEnv({"curriculum_stage": 2, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    # Stage is bound on config (the env folds it into Episode at termination).
    assert env._config.curriculum_stage == 2


def test_U14_reset_passes_language_weights_to_task_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    real_generate = __import__("cells.step_07_task_generator", fromlist=["generate"]).generate
    def _wrap(seed: int, stage: int, weights: Any) -> GoalSpec:
        captured["seed"] = seed
        captured["stage"] = stage
        captured["weights"] = weights
        return real_generate(seed, stage, weights)
    monkeypatch.setattr("cells.step_10_env.task_generate", _wrap)

    custom = {"en": 0.5, "hinglish": 0.5}
    env = DriftCallEnv(
        {"curriculum_stage": 1, "language_weights": custom, "scheduler": _empty_scheduler}
    )
    env.reset(seed=7)
    assert captured["weights"] == env._config.language_weights


def test_U15_reset_same_seed_same_goal_and_schedule() -> None:
    env_a = DriftCallEnv({"curriculum_stage": 2, "scheduler": _empty_scheduler})
    env_b = DriftCallEnv({"curriculum_stage": 2, "scheduler": _empty_scheduler})
    obs_a = env_a.reset(seed=42)
    obs_b = env_b.reset(seed=42)
    assert obs_a.goal == obs_b.goal
    assert env_a._state is not None and env_b._state is not None
    assert env_a._state.drift_schedule == env_b._state.drift_schedule
    assert env_a._state.vendor_states == env_b._state.vendor_states


def test_U16_reset_none_seed_populates_from_urandom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub task generation so seed-dependent template glitches don't matter —
    # we only care that env._seed is populated from os.urandom and varies.
    sentinel_goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000},
        language="en",
        seed_utterance="x",
    )
    monkeypatch.setattr("cells.step_10_env.task_generate", lambda *a, **kw: sentinel_goal)
    seeds_seen: set[int] = set()
    for _ in range(3):
        env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
        env.reset(seed=None)
        assert env._seed is not None
        seeds_seen.add(env._seed)
    assert len(seeds_seen) >= 2


def test_U17_reset_audio_boundary_enabled_invokes_tts_synthesize() -> None:
    tts = StubTTS()
    asr = StubASR()
    env = DriftCallEnv(
        {
            "curriculum_stage": 1,
            "audio_boundary_enabled": True,
            "tts_engine": tts,
            "asr_engine": asr,
            "scheduler": _empty_scheduler,
        }
    )
    obs = env.reset(seed=11)
    assert len(tts.calls) == 1
    assert tts.calls[0][0] == obs.goal.seed_utterance
    assert tts.calls[0][1] == obs.goal.language
    assert obs.last_transcript == obs.goal.seed_utterance


# =====================================================================
# 1.3 step() — pipeline (U18–U24)
# =====================================================================


def test_U18_step_validates_before_any_mutation() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    # A SPEAK is benign; verify state changes via dataclasses.replace.
    prev = env._state
    obs = env.step(_valid_speak("hi"))
    assert env._state is not prev
    assert obs.turn == 1


def test_U19_step_increments_turn_after_validate() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    obs = env.step(_valid_speak())
    assert obs.turn == 1
    obs = env.step(_valid_speak())
    assert obs.turn == 2


def test_U20_step_fires_drifts_before_dispatch() -> None:
    drift = DriftEvent(
        turn=1,
        drift_type="schema",
        domain="airline",
        description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.price_rename",
    )
    env = DriftCallEnv(
        {
            "curriculum_stage": 2,
            "scheduler": _scripted_scheduler((drift,)),
        }
    )
    env.reset(seed=42)
    obs = env.step(_valid_speak("note"))
    assert any(d.pattern_id == "airline.price_rename" for d in obs.drift_log)
    assert env._state is not None
    assert env._state.schema_versions["airline"] == "v2"


def test_U21_step_records_action_via_dataclasses_replace() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    prev_state = env._state
    a = _valid_speak("alpha")
    env.step(a)
    next_state = env._state
    assert prev_state is not None and next_state is not None
    assert prev_state is not next_state
    assert id(prev_state.actions) != id(next_state.actions)
    assert next_state.actions == prev_state.actions + (a,)


def test_U22_step_checks_terminal_after_record_timeout() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    for _ in range(8):
        env.step(_valid_speak("ok"))
    assert env.done() is True
    assert env.episode().terminated_by == "TIMEOUT"
    assert env._state is not None and env._state.turn == 8


def test_U23_step_submit_calls_compute_rewards_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewards_calls: list[Any] = []
    sentinel = object()

    def _recorder(episode: Any) -> Any:
        rewards_calls.append(episode)
        return sentinel

    monkeypatch.setattr(_REWARDS_STUB, "compute_rewards", _recorder)
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    env.step(_valid_submit())
    assert len(rewards_calls) == 1
    assert env.rewards() is sentinel
    # Memoized identity.
    assert env.rewards() is env.rewards()


def test_U24_step_abort_forces_r1_zero() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    env.step(DriftCallAction(action_type=ActionType.ABORT))
    assert env.episode().terminated_by == "ABORT"
    assert env.rewards().r1 == 0.0


# =====================================================================
# 1.4 _validate_action purity (U25–U28)
# =====================================================================


def test_U25_invalid_action_raises_no_state_mutation() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    prev_state = env._state
    bad = DriftCallAction(
        action_type=ActionType.TOOL_CALL, tool_name="airline.search", tool_args=None
    )
    with pytest.raises(InvalidActionError):
        env.step(bad)
    assert env._state is prev_state
    assert env._rewards is None


def test_U26_env_valid_after_invalid_action() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    bad = DriftCallAction(action_type=ActionType.SPEAK, message="")
    with pytest.raises(InvalidActionError):
        env.step(bad)
    obs = env.step(_valid_speak("ok"))
    assert obs.turn == 1


def test_U27_invalid_action_no_drift_fired_no_terminal_marker() -> None:
    drift = DriftEvent(
        turn=1, drift_type="schema", domain="airline",
        description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        from_version="v1", to_version="v2", pattern_id="airline.price_rename",
    )
    env = DriftCallEnv(
        {"curriculum_stage": 2, "scheduler": _scripted_scheduler((drift,))}
    )
    env.reset(seed=42)
    with pytest.raises(InvalidActionError):
        env.step(DriftCallAction(action_type=ActionType.SPEAK, message=""))
    assert env._state is not None
    assert env._state.drift_fired == ()
    assert env.done() is False


def test_U28_oversize_rationale_raises_invalid_action() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    bad = DriftCallAction(
        action_type=ActionType.SUBMIT, confidence=0.5, rationale="x" * 201
    )
    prev_state = env._state
    with pytest.raises(InvalidActionError):
        env.step(bad)
    assert env._state is prev_state


# =====================================================================
# 1.5 state() — frozen reference (U29–U30)
# =====================================================================


def test_U29_state_returns_frozen_reference() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    s = env.state()
    assert s is env._state
    assert s.__dataclass_params__.frozen is True
    with pytest.raises(FrozenInstanceError):
        s.turn = 99  # type: ignore[misc]


def test_U30_state_unready_raises_e2() -> None:
    env = DriftCallEnv()
    with pytest.raises(EnvNotReadyError):
        env.state()
    assert env.done() is False


# =====================================================================
# 1.6 close() — idempotency (U31–U32)
# =====================================================================


def test_U31_close_idempotent() -> None:
    env = DriftCallEnv()
    env.close()
    env.close()
    env.close()
    assert env._closed is True


def test_U32_close_does_not_free_shared_audio_engines() -> None:
    tts = StubTTS()
    asr = StubASR()
    env = DriftCallEnv(
        {
            "audio_boundary_enabled": True,
            "tts_engine": tts,
            "asr_engine": asr,
            "scheduler": _empty_scheduler,
        }
    )
    env.reset(seed=11)
    env.close()
    assert env._closed is True
    assert not hasattr(tts, "close")
    assert not hasattr(asr, "close")


# =====================================================================
# 1.7 Terminal-only accessors + error taxonomy (U33–U35)
# =====================================================================


def test_U33_episode_before_terminal_raises_e6() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    with pytest.raises(EpisodeNotTerminalError):
        env.episode()
    with pytest.raises(EpisodeNotTerminalError):
        env.rewards()
    assert env.done() is False


def test_U34_double_submit_raises_e5() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=42)
    env.step(_valid_submit())
    rewards_obj = env.rewards()
    with pytest.raises(EpisodeAlreadyTerminalError):
        env.step(_valid_submit())
    assert env.done() is True
    assert env.rewards() is rewards_obj


def test_U35_all_12_errors_derive_from_driftcallenverror() -> None:
    classes = {
        InvalidConfigError,
        EnvNotReadyError,
        EnvClosedError,
        InvalidActionError,
        EpisodeAlreadyTerminalError,
        EpisodeNotTerminalError,
        ConcurrentStepError,
        UnknownDomainError,
        UnknownToolError,
        DriftInjectionError,
        RewardComputationError,
        AudioPipelineError,
    }
    assert len(classes) == 12
    for cls in classes:
        assert issubclass(cls, DriftCallEnvError)
        assert issubclass(cls, Exception)


# =====================================================================
# Extra error-mode coverage (env_tests.md §4.1 matrix)
# =====================================================================


def test_step_before_reset_raises_e2() -> None:
    env = DriftCallEnv()
    with pytest.raises(EnvNotReadyError):
        env.step(_valid_speak())


def test_episode_before_reset_raises_e6() -> None:
    env = DriftCallEnv()
    with pytest.raises(EpisodeNotTerminalError):
        env.episode()


def test_step_after_close_raises_e3() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    env.close()
    with pytest.raises(EnvClosedError):
        env.step(_valid_speak())


def test_reset_after_close_raises_e3() -> None:
    env = DriftCallEnv()
    env.close()
    with pytest.raises(EnvClosedError):
        env.reset(seed=1)


def test_probe_schema_unknown_domain_raises_e8() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    bad = DriftCallAction(action_type=ActionType.PROBE_SCHEMA, tool_name="spaceship")
    with pytest.raises(UnknownDomainError):
        env.step(bad)


def test_tool_call_unknown_tool_raises_e9() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    bad = DriftCallAction(
        action_type=ActionType.TOOL_CALL,
        tool_name="airline.teleport",
        tool_args={},
    )
    with pytest.raises(UnknownToolError):
        env.step(bad)


def test_drift_fold_error_propagates_e10() -> None:
    bad_event = DriftEvent(
        turn=1,
        drift_type="schema",
        domain="airline",
        description="bogus",
        from_version="v1",
        to_version="v2",
        pattern_id="not.a.real.pattern",
    )
    env = DriftCallEnv(
        {"curriculum_stage": 2, "scheduler": _scripted_scheduler((bad_event,))}
    )
    env.reset(seed=1)
    with pytest.raises(DriftInjectionError):
        env.step(_valid_speak())


def test_reward_compute_error_propagates_e11(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_episode: Any) -> Any:
        raise RuntimeError("rewards exploded")
    monkeypatch.setattr(_REWARDS_STUB, "compute_rewards", _boom)
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(RewardComputationError):
        env.step(_valid_submit())


def test_audio_pipeline_error_on_reset_is_e12_class() -> None:
    tts = StubTTS()
    asr = StubASR()
    tts.raise_on_next = True
    env = DriftCallEnv(
        {
            "audio_boundary_enabled": True,
            "tts_engine": tts,
            "asr_engine": asr,
            "scheduler": _empty_scheduler,
        }
    )
    with pytest.raises(AudioPipelineError):
        env.reset(seed=1)
    # Env remains unready post-failure.
    assert env._state is None


def test_reset_scripted_bad_schedule_raises_e1() -> None:
    def _bad_sched(stage: int, episode_seed: int, goal: GoalSpec) -> tuple:
        from cells.step_06_drift_injector import DriftScheduleConflictError
        raise DriftScheduleConflictError("forced failure")
    env = DriftCallEnv({"curriculum_stage": 2, "scheduler": _bad_sched})
    with pytest.raises(InvalidConfigError):
        env.reset(seed=1)


def test_reentrant_step_raises_e7() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    # Manually flip the guard to simulate re-entrancy.
    env._step_in_progress = True
    with pytest.raises(ConcurrentStepError):
        env.step(_valid_speak())


def test_invalid_action_unknown_action_type() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)

    class FakeAction:
        action_type = "nope"

    with pytest.raises(InvalidActionError):
        env.step(FakeAction())  # type: ignore[arg-type]


def test_speak_with_nul_byte_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(DriftCallAction(action_type=ActionType.SPEAK, message="a\x00b"))


def test_speak_too_long_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(
            DriftCallAction(action_type=ActionType.SPEAK, message="x" * 2001)
        )


def test_submit_out_of_range_confidence_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(
            DriftCallAction(action_type=ActionType.SUBMIT, confidence=1.5)
        )


def test_submit_missing_confidence_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(DriftCallAction(action_type=ActionType.SUBMIT))


def test_abort_forbids_tool_name_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(
            DriftCallAction(action_type=ActionType.ABORT, tool_name="airline.search")
        )


def test_force_drift_pattern_unknown_raises_e4() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(_valid_speak(), force_drift_pattern="not.a.pattern")


def test_force_drift_pattern_overrides_schedule() -> None:
    drift_other = DriftEvent(
        turn=1, drift_type="schema", domain="airline",
        description="x",
        from_version="v1", to_version="v2",
        pattern_id="airline.price_rename",
    )
    env = DriftCallEnv(
        {"curriculum_stage": 2, "scheduler": _scripted_scheduler((drift_other,))}
    )
    env.reset(seed=1)
    obs = env.step(_valid_speak(), force_drift_pattern="hotel.cancel_window_shrink")
    pattern_ids = {d.pattern_id for d in obs.drift_log}
    assert "hotel.cancel_window_shrink" in pattern_ids
    assert "airline.price_rename" not in pattern_ids


def test_probe_schema_known_domain_returns_tool_result() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    obs = env.step(
        DriftCallAction(action_type=ActionType.PROBE_SCHEMA, tool_name="airline")
    )
    assert obs.tool_results
    assert obs.tool_results[-1].tool_name == "probe:airline"
    assert obs.tool_results[-1].status == "ok"
    assert obs.tool_results[-1].latency_ms == 0


def test_clarify_action_records_no_tool_result() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    obs = env.step(
        DriftCallAction(action_type=ActionType.CLARIFY, message="when?")
    )
    assert obs.tool_results == ()


def test_tool_call_search_returns_ok() -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    obs = env.reset(seed=42)
    # Pick a tool we know exists; airline.search is broadly available.
    obs = env.step(
        DriftCallAction(
            action_type=ActionType.TOOL_CALL,
            tool_name="airline.search",
            tool_args={"from_": "HYD", "to": "BLR", "date": "2026-04-25"},
        )
    )
    assert len(obs.tool_results) == 1
    assert obs.tool_results[0].tool_name == "airline.search"


# =====================================================================
# 2. Property tests (P1–P6)
# =====================================================================


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=10**4))
def test_P1_step_is_pure_per_call(seed: int) -> None:
    # Pin the goal so seed-dependent template glitches in the YAML library
    # don't poison the property test (orthogonal to env determinism).
    sentinel_goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000},
        language="en",
        seed_utterance="x",
    )
    with mock.patch(
        "cells.step_10_env.task_generate", return_value=sentinel_goal
    ):
        a = _valid_speak("alpha")
        e1 = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
        e2 = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
        e1.reset(seed=seed)
        e2.reset(seed=seed)
        o1 = e1.step(a)
        o2 = e2.step(a)
        # Per env.md §9 Q5, episode_id is uuid4 (non-deterministic by design);
        # everything else is byte-identical.
        assert o1 == o2
        assert e1._state is not None and e2._state is not None
        assert dataclasses.replace(e1._state, episode_id="X") == dataclasses.replace(
            e2._state, episode_id="X"
        )


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=10**4))
def test_P2_validation_failure_preserves_pre_step_state(seed: int) -> None:
    sentinel_goal = GoalSpec(
        domain="airline", intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000}, language="en", seed_utterance="x",
    )
    with mock.patch("cells.step_10_env.task_generate", return_value=sentinel_goal):
        env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
        env.reset(seed=seed)
        prev = env._state
        bad = DriftCallAction(action_type=ActionType.SPEAK, message="")
        with pytest.raises(InvalidActionError):
            env.step(bad)
        assert env._state is prev


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(n=st.integers(min_value=1, max_value=7))
def test_P3_turn_counter_monotone_non_decreasing(n: int) -> None:
    sentinel_goal = GoalSpec(
        domain="airline", intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000}, language="en", seed_utterance="x",
    )
    with mock.patch("cells.step_10_env.task_generate", return_value=sentinel_goal):
        env = DriftCallEnv({"curriculum_stage": 3, "scheduler": _empty_scheduler})
        env.reset(seed=1)
        assert env._state is not None
        last = env._state.turn
        for _ in range(n):
            if env.done():
                break
            env.step(_valid_speak("x"))
            assert env._state is not None
            assert env._state.turn == last + 1
            last = env._state.turn


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=10**4))
def test_P4_frozen_state_identity_changes_on_transition(seed: int) -> None:
    sentinel_goal = GoalSpec(
        domain="airline", intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000}, language="en", seed_utterance="x",
    )
    with mock.patch("cells.step_10_env.task_generate", return_value=sentinel_goal):
        env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
        env.reset(seed=seed)
        prev = env._state
        env.step(_valid_speak("alpha"))
        nxt = env._state
        assert prev is not None and nxt is not None
        assert prev is not nxt
        assert id(prev.actions) != id(nxt.actions)


@pytest.mark.parametrize("path", ["SUBMIT", "ABORT", "TIMEOUT"])
def test_P5_rewards_memoized_identity(path: str) -> None:
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    env.reset(seed=1)
    if path == "SUBMIT":
        env.step(_valid_submit())
    elif path == "ABORT":
        env.step(DriftCallAction(action_type=ActionType.ABORT))
    else:
        for _ in range(8):
            env.step(_valid_speak("x"))
    a = env.rewards()
    for _ in range(10):
        assert env.rewards() is a
    assert env.episode() is env.episode()


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=10**4))
def test_P6_available_tools_fixed_for_episode(seed: int) -> None:
    sentinel_goal = GoalSpec(
        domain="airline", intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000}, language="en", seed_utterance="x",
    )
    with mock.patch("cells.step_10_env.task_generate", return_value=sentinel_goal):
        env = DriftCallEnv({"curriculum_stage": 2, "scheduler": _empty_scheduler})
        obs = env.reset(seed=seed)
        base = obs.available_tools
        obs2 = env.step(_valid_speak("a"))
        assert obs2.available_tools == base


# =====================================================================
# 3. Integration tests (I1–I4)
# =====================================================================


def test_I1_episode_stage1_airline_happy_submit() -> None:
    """env.md §8.1 — Stage-1 happy path."""
    env = DriftCallEnv({"curriculum_stage": 1, "scheduler": _empty_scheduler})
    obs = env.reset(seed=42)
    assert obs.drift_log == ()
    # Sequence of valid actions then SUBMIT.
    env.step(_valid_speak("plan"))
    env.step(_valid_speak("search"))
    env.step(_valid_speak("pick"))
    env.step(_valid_speak("confirm"))
    env.step(_valid_submit(confidence=0.9, message="done"))
    assert env.done() is True
    assert env.episode().terminated_by == "SUBMIT"
    assert env.episode().turns_used == 5
    # Stub rewards returns r1==1.0, r2==0.5 on success.
    assert env.rewards().r1 == 1.0


def test_I2_episode_stage2_drift_detect_adapt() -> None:
    """env.md §8.2 — Stage-2 with airline.price_rename at turn 3."""
    drift = DriftEvent(
        turn=3,
        drift_type="schema",
        domain="airline",
        description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.price_rename",
    )
    env = DriftCallEnv(
        {"curriculum_stage": 2, "scheduler": _scripted_scheduler((drift,))}
    )
    env.reset(seed=7)
    env.step(_valid_speak("turn1"))
    env.step(_valid_speak("turn2"))
    obs = env.step(_valid_speak("turn3"))  # drift fires here
    assert any(d.pattern_id == "airline.price_rename" for d in obs.drift_log)
    env.step(
        DriftCallAction(
            action_type=ActionType.SPEAK,
            message="Note: total_fare_inr replaced price.",
        )
    )
    env.step(_valid_speak("adapt"))
    env.step(_valid_submit(confidence=0.8, message="booked"))
    assert env.done() is True
    assert env.episode().terminated_by == "SUBMIT"
    # Drift event captured in episode.
    assert any(d.pattern_id == "airline.price_rename" for d in env.episode().drift_log)


def test_I3_episode_stage3_compound_drift_timeout() -> None:
    """env.md §8.3 — Stage-3 with two drifts; force a TIMEOUT."""
    drift_a = DriftEvent(
        turn=3,
        drift_type="schema",
        domain="airline",
        description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.price_rename",
    )
    drift_b = DriftEvent(
        turn=9,
        drift_type="auth",
        domain="payment",
        description="token_v1 401s; token_v2 with scope=payments:write:v2 required",
        from_version="v1",
        to_version="v2",
        pattern_id="payment.auth_scope_upgrade",
    )
    env = DriftCallEnv(
        {
            "curriculum_stage": 3,
            "scheduler": _scripted_scheduler((drift_a, drift_b)),
        }
    )
    env.reset(seed=2026)
    for _ in range(16):
        env.step(_valid_speak("x"))
    assert env.done() is True
    ep = env.episode()
    assert ep.terminated_by == "TIMEOUT"
    assert ep.turns_used == 16
    # Both drifts fired.
    fired_ids = {d.pattern_id for d in ep.drift_log}
    assert "airline.price_rename" in fired_ids
    assert "payment.auth_scope_upgrade" in fired_ids
    assert env.rewards().r1 == 0.0


def test_I4_episode_audio_boundary_enabled_stubs() -> None:
    """env.md §8.4 — audio boundary enabled, stubs only."""
    tts = StubTTS()
    asr = StubASR()
    env = DriftCallEnv(
        {
            "curriculum_stage": 1,
            "audio_boundary_enabled": True,
            "tts_engine": tts,
            "asr_engine": asr,
            "scheduler": _empty_scheduler,
        }
    )
    env.reset(seed=11)
    env.step(
        DriftCallAction(action_type=ActionType.CLARIFY, message="when?")
    )
    env.step(_valid_speak("alright"))
    env.step(_valid_submit())
    assert env.done() is True
    # TTS was invoked at least once on reset.
    assert len(tts.calls) >= 1
    # No bytes objects in reward inputs (textual only).
    ep = env.episode()
    for tr in ep.tool_results:
        assert not isinstance(tr.response, bytes)
    for a in ep.actions:
        assert not isinstance(a.message, bytes)
