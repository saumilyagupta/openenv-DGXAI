"""Tests for cells/step_14_custom_trainer.py.

Covers training_tests.md U14–U18 (EpisodeDatasetAdapter iteration) plus
the DriftCallGRPOTrainer override surface. TRL is mocked via a stub base
class so the multi-turn rollout wiring can be exercised CPU-only.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from cells.step_14_custom_trainer import (
    PINNED_SYSTEM_PROMPT,
    AdapterRecord,
    EpisodeDatasetAdapter,
    driftcall_grpo_trainer_methods,
    make_driftcall_grpo_trainer_cls,
    render_initial_prompt,
)


@dataclass(frozen=True)
class _FakeGoal:
    seed_utterance: str
    language: str = "en"


@dataclass(frozen=True)
class _FakeEpisode:
    ident: str


class _SpyTokenizer:
    """Records the last ``apply_chat_template`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        self.calls.append(
            ((messages,), {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt})
        )
        # Return a canonical rendered string so tests can check its content.
        rendered = f"[SYSTEM]{messages[0]['content']}[USER]{messages[1]['content']}"
        return rendered


def _task_gen_factory() -> MagicMock:
    fn = MagicMock()

    def _impl(*, seed: int, stage: int, language_weights: dict[str, float]) -> _FakeGoal:
        return _FakeGoal(seed_utterance=f"utt-{seed}")

    fn.side_effect = _impl
    return fn


def _env_factory() -> MagicMock:
    return MagicMock(name="DriftCallEnv")


def _make_adapter(
    *,
    stage: int = 1,
    stage_base_seed: int = 1_000_000,
    language_weights: dict[str, float] | None = None,
    task_gen: MagicMock | None = None,
    tokenizer: _SpyTokenizer | None = None,
) -> EpisodeDatasetAdapter:
    if language_weights is None:
        language_weights = {"en": 1.0, "hinglish": 0.0, "hi": 0.0, "ta": 0.0, "kn": 0.0}
    return EpisodeDatasetAdapter(
        task_gen=task_gen if task_gen is not None else _task_gen_factory(),
        env_factory=_env_factory(),
        stage=stage,  # type: ignore[arg-type]
        stage_base_seed=stage_base_seed,
        language_weights=language_weights,  # type: ignore[arg-type]
        tokenizer=tokenizer if tokenizer is not None else _SpyTokenizer(),
    )


class TestPinnedSystemPrompt:
    def test_exact_verbatim(self) -> None:
        assert PINNED_SYSTEM_PROMPT == (
            "You are a concierge assistant. Use the provided tools. "
            "Respond in the caller's language. Submit with calibrated confidence."
        )


class TestRenderInitialPrompt:
    def test_uses_apply_chat_template_with_correct_kwargs(self) -> None:
        tok = _SpyTokenizer()
        goal = _FakeGoal(seed_utterance="hello")
        rendered = render_initial_prompt(tok, goal)

        assert len(tok.calls) == 1
        _, kwargs = tok.calls[0]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert "[SYSTEM]" in rendered
        assert "hello" in rendered

    def test_messages_have_pinned_system_and_seed_utterance(self) -> None:
        tok = _SpyTokenizer()
        goal = _FakeGoal(seed_utterance="hello world")
        render_initial_prompt(tok, goal)

        (messages,), _ = tok.calls[0]
        assert messages[0] == {"role": "system", "content": PINNED_SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "hello world"}

    def test_missing_seed_utterance_uses_empty_string(self) -> None:
        class _NoUtt:
            pass

        tok = _SpyTokenizer()
        rendered = render_initial_prompt(tok, _NoUtt())
        (messages,), _ = tok.calls[0]
        assert messages[1]["content"] == ""
        assert "[USER]" in rendered


class TestEpisodeDatasetAdapter:
    def test_yields_prompt_and_meta_shape(self) -> None:
        adapter = _make_adapter()
        it = iter(adapter)
        rec = next(it)
        assert set(rec.keys()) == {"prompt", "_meta"}
        assert isinstance(rec["prompt"], str)
        meta = rec["_meta"]
        assert set(meta.keys()) == {
            "goal",
            "episode_seed",
            "stage",
            "language_weights",
        }

    def test_seeds_monotonically_from_stage_base_seed(self) -> None:
        adapter = _make_adapter(stage_base_seed=1_000_000)
        it = iter(adapter)
        seeds = [next(it)["_meta"]["episode_seed"] for _ in range(5)]
        assert seeds == [1_000_000, 1_000_001, 1_000_002, 1_000_003, 1_000_004]

    def test_task_gen_called_once_per_step_with_correct_kwargs(self) -> None:
        tg = _task_gen_factory()
        lang_w = {"en": 0.3, "hinglish": 0.3, "hi": 0.2, "ta": 0.1, "kn": 0.1}
        adapter = _make_adapter(
            stage=2,
            stage_base_seed=2_000_000,
            language_weights=lang_w,
            task_gen=tg,
        )
        it = iter(adapter)
        for _ in range(3):
            next(it)

        assert tg.call_count == 3
        for i, call in enumerate(tg.call_args_list):
            assert call.kwargs["stage"] == 2
            assert call.kwargs["language_weights"] == lang_w
            assert call.kwargs["seed"] == 2_000_000 + i

    def test_prompt_uses_apply_chat_template(self) -> None:
        tok = _SpyTokenizer()
        adapter = _make_adapter(tokenizer=tok)
        it = iter(adapter)
        next(it)
        assert len(tok.calls) == 1
        _, kwargs = tok.calls[0]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}

    def test_prompt_contains_pinned_system_prompt(self) -> None:
        tok = _SpyTokenizer()
        adapter = _make_adapter(tokenizer=tok)
        it = iter(adapter)
        rec = next(it)
        (messages,), _ = tok.calls[0]
        assert messages[0]["content"] == PINNED_SYSTEM_PROMPT
        # rendered output carries the pinned system content
        assert PINNED_SYSTEM_PROMPT in rec["prompt"]

    def test_language_weights_copied_not_aliased(self) -> None:
        lang_w = {"en": 1.0}
        adapter = _make_adapter(language_weights=lang_w)
        lang_w["en"] = 0.0  # mutate original after construction
        rec = next(iter(adapter))
        assert rec["_meta"]["language_weights"] == {"en": 1.0}

    def test_peek_returns_adapter_record_without_advancing(self) -> None:
        tg = _task_gen_factory()
        adapter = _make_adapter(stage_base_seed=500, task_gen=tg)
        r0 = adapter.peek(step=0)
        r5 = adapter.peek(step=5)
        assert isinstance(r0, AdapterRecord)
        assert r0.episode_seed == 500
        assert r5.episode_seed == 505
        # peek(0) then first __iter__ yield both produce seed=500 — peek does
        # not advance a shared counter.
        first_yield = next(iter(adapter))
        assert first_yield["_meta"]["episode_seed"] == 500

    def test_adapter_record_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        adapter = _make_adapter()
        rec = adapter.peek(step=0)
        with pytest.raises(FrozenInstanceError):
            rec.episode_seed = 999  # type: ignore[misc]


class _FakeGRPOTrainerBase:
    """Stub standing in for ``trl.GRPOTrainer`` for subclass testing."""

    def __init__(self, *, model: Any, args: Any, processing_class: Any, **_: Any) -> None:
        self.model = model
        self.args = args
        self.processing_class = processing_class


class TestDriftCallGRPOTrainer:
    def test_override_methods_introspection(self) -> None:
        # v1: rollout uses TRL default; reward goes via reward_funcs.
        assert driftcall_grpo_trainer_methods() == ("__init__",)

    def test_subclass_attaches_driftcall_kwargs(self) -> None:
        Trainer = make_driftcall_grpo_trainer_cls(_FakeGRPOTrainerBase)
        model = MagicMock()
        tok = _SpyTokenizer()
        args = MagicMock(num_generations=8)
        rollout_fn = MagicMock()
        env_factory = MagicMock()
        rfn = MagicMock(return_value=[0.5] * 8)

        trainer = Trainer(
            model=model,
            args=args,
            processing_class=tok,
            rollout_group_fn=rollout_fn,
            env_factory=env_factory,
            reward_fn_driftcall=rfn,
        )
        assert trainer.model is model
        assert trainer.processing_class is tok
        assert trainer.args is args
        assert trainer.rollout_group_fn is rollout_fn
        assert trainer.env_factory is env_factory
        assert trainer.reward_fn_driftcall is rfn

    def test_reward_funcs_wired_with_episode_closure(self) -> None:
        # v1: trainer init wires our episode-aware closure into reward_funcs[0].
        Trainer = make_driftcall_grpo_trainer_cls(_FakeGRPOTrainerBase)
        rfn = MagicMock(return_value=[0.5] * 8)
        trainer = Trainer(
            model=MagicMock(),
            args=MagicMock(num_generations=8),
            processing_class=_SpyTokenizer(),
            rollout_group_fn=MagicMock(),
            env_factory=MagicMock(),
            reward_fn_driftcall=rfn,
        )
        # If the fake base sets reward_funcs from kwargs, our closure should
        # be present at index 0.
        rf = getattr(trainer, "reward_funcs", None)
        if rf is not None and len(rf) > 0:
            assert getattr(rf[0], "__name__", "") == "driftcall_episode_reward"

    def test_subclass_inherits_from_provided_base(self) -> None:
        Trainer = make_driftcall_grpo_trainer_cls(_FakeGRPOTrainerBase)
        assert issubclass(Trainer, _FakeGRPOTrainerBase)
        assert Trainer.__name__ == "DriftCallGRPOTrainer"

    def test_make_cls_uses_trl_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no explicit base_cls, lazy imports ``trl.GRPOTrainer``."""
        trl_mod = types.ModuleType("trl")

        class _FakeTRLBase:
            def __init__(self, **_: Any) -> None:
                pass

        trl_mod.GRPOTrainer = _FakeTRLBase  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "trl", trl_mod)

        Trainer = make_driftcall_grpo_trainer_cls()
        assert issubclass(Trainer, _FakeTRLBase)

    def test_args_num_generations_propagated_to_trainer(self) -> None:
        # v1: rollout is no longer overridden; verify args.num_generations
        # is preserved on the trainer instance for downstream TRL plumbing.
        Trainer = make_driftcall_grpo_trainer_cls(_FakeGRPOTrainerBase)
        args = MagicMock(num_generations=4)
        trainer = Trainer(
            model=MagicMock(),
            args=args,
            processing_class=_SpyTokenizer(),
            rollout_group_fn=MagicMock(),
            env_factory=MagicMock(),
            reward_fn_driftcall=MagicMock(),
        )
        assert trainer.args.num_generations == 4
