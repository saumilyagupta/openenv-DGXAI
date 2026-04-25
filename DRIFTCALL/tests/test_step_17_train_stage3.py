"""Tests for cells/step_17_train_stage3.py.

CUDA + heavy deps mocked: ``unsloth``, ``trl``, ``wandb``, ``peft`` are
injected via ``sys.modules``. Covers:

- Constructor accepts stage param (Stage-3 fixed at 3).
- num_steps default == 150 (DESIGN.md §10.3 Stage-3 row).
- resume_from required (None rejected) — must be a ``pathlib.Path``.
- Language mix identical to Stage 2 — 30/30/20/10/10.
- warmup_ratio == 0.0 in the resolved plan (no double warmup).
- Language-weight floor (0.05) enforced for non-English cohorts.
- WandB offline fallback path covered.
- BF16 assertion fires on dtype mismatch (V100 safety).
- Trainer resumes via ``resume_from_checkpoint=str(resume_from)``.
- Checkpoint save uses ``safe_serialization=True``.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from cells import step_17_train_stage3 as stage3
from cells.step_12_gemma_boot import BF16SlippageError
from cells.step_17_train_stage3 import (
    COHORT_MIN_WEIGHT_AT_STAGE_GE_2,
    CSV_COLUMNS,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    LANGUAGE_WEIGHTS,
    NON_ENGLISH_LANGUAGES,
    STAGE,
    STAGE_BASE_SEED,
    WARMUP_RATIO,
    StageRunPlan,
    WandBStartupError,
    build_run_plan,
    save_checkpoint,
    train,
    write_local_csv_row,
)

# ---------------------------------------------------------------------------
# Shared fakes / fixtures
# ---------------------------------------------------------------------------


def _fake_model_with_dtype(dtype: torch.dtype) -> MagicMock:
    param = torch.zeros(1, dtype=dtype)
    model = MagicMock(name="model")
    model.parameters = MagicMock(side_effect=lambda: iter([param]))
    return model


def _install_fake_unsloth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_dtype: torch.dtype = torch.float16,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    base_model = _fake_model_with_dtype(base_dtype)
    peft_init_model = MagicMock(name="peft_init_model")
    tokenizer = MagicMock(name="tokenizer")

    FastModel = MagicMock(name="FastModel")
    FastModel.from_pretrained = MagicMock(return_value=(base_model, tokenizer))
    FastModel.get_peft_model = MagicMock(return_value=peft_init_model)

    unsloth_mod: Any = types.ModuleType("unsloth")
    unsloth_mod.FastVisionModel = FastModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return base_model, peft_init_model, tokenizer


def _install_fake_peft(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    resumed_model = MagicMock(name="resumed_peft_model")
    PeftModel = MagicMock(name="PeftModel")
    PeftModel.from_pretrained = MagicMock(return_value=resumed_model)

    peft_mod: Any = types.ModuleType("peft")
    peft_mod.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_mod)
    return resumed_model


_LAST_FAKE_TRAINER: list[_FakeGRPOTrainerBase] = []


class _FakeGRPOTrainerBase:
    """Real class standing in for ``trl.GRPOTrainer`` (see stage-1 tests)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = kwargs.get("model")
        self.args = kwargs.get("args")
        self.processing_class = kwargs.get("processing_class")
        self.train_dataset = kwargs.get("train_dataset")
        self.train_calls: list[dict[str, Any]] = []
        _LAST_FAKE_TRAINER.append(self)

    def train(self, **kwargs: Any) -> None:
        self.train_calls.append(dict(kwargs))


def _install_fake_trl(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGRPOTrainerBase]:
    class _FakeGRPOConfig:
        def __init__(self, **kwargs: Any) -> None:
            defaults = {"bf16": False}
            defaults.update(kwargs)
            for k, v in defaults.items():
                setattr(self, k, v)

    _LAST_FAKE_TRAINER.clear()
    trl_mod: Any = types.ModuleType("trl")
    trl_mod.GRPOConfig = _FakeGRPOConfig
    trl_mod.GRPOTrainer = _FakeGRPOTrainerBase
    monkeypatch.setitem(sys.modules, "trl", trl_mod)
    return _FakeGRPOTrainerBase


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch, *, init_raises: bool = False
) -> MagicMock:
    wandb_mod: Any = types.ModuleType("wandb")
    init = MagicMock(name="wandb.init")
    if init_raises:
        init.side_effect = RuntimeError("network failure")
    wandb_mod.init = init
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    return init


def _install_fake_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    mod: Any = types.ModuleType("cells.step_08_rewards")

    class _R:
        reward = 0.5

    mod.compute_rewards = MagicMock(return_value=_R())
    monkeypatch.setitem(sys.modules, "cells.step_08_rewards", mod)


def _full_train_setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_dtype: torch.dtype = torch.float16,
) -> tuple[MagicMock, MagicMock]:
    _, _, tokenizer = _install_fake_unsloth(monkeypatch, base_dtype=base_dtype)
    resumed_model = _install_fake_peft(monkeypatch)
    _install_fake_trl(monkeypatch)
    _install_fake_wandb(monkeypatch)
    _install_fake_rewards(monkeypatch)
    return resumed_model, tokenizer


def _last_trainer() -> _FakeGRPOTrainerBase:
    assert _LAST_FAKE_TRAINER, "no trainer was constructed"
    return _LAST_FAKE_TRAINER[-1]


def _callables() -> tuple[MagicMock, MagicMock, MagicMock]:
    return MagicMock(name="task_gen"), MagicMock(name="env_factory"), MagicMock(
        name="rollout_group_fn"
    )


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------


class TestStage3Constants:
    def test_stage_constant_is_3(self) -> None:
        assert STAGE == 3

    def test_default_num_steps_is_150(self) -> None:
        assert DEFAULT_NUM_STEPS == 150

    def test_warmup_ratio_is_zero(self) -> None:
        assert WARMUP_RATIO == 0.0

    def test_default_output_dir(self) -> None:
        assert Path("checkpoints/stage3_final") == DEFAULT_OUTPUT_DIR

    def test_stage_base_seed_is_three_million(self) -> None:
        assert STAGE_BASE_SEED == 3_000_000

    def test_cohort_floor_is_0_05(self) -> None:
        assert COHORT_MIN_WEIGHT_AT_STAGE_GE_2 == 0.05

    def test_non_english_languages_set(self) -> None:
        assert set(NON_ENGLISH_LANGUAGES) == {"hi", "ta", "kn", "hinglish"}


class TestStage3LanguageMix:
    def test_language_mix_identical_to_stage_2(self) -> None:
        # Per DESIGN.md §10.3 Stage 3 reuses Stage 2 mix.
        assert math.isclose(LANGUAGE_WEIGHTS["en"], 0.30, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["hinglish"], 0.30, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["hi"], 0.20, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["ta"], 0.10, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["kn"], 0.10, abs_tol=1e-9)

    def test_language_weights_sum_to_one(self) -> None:
        assert math.isclose(sum(LANGUAGE_WEIGHTS.values()), 1.0, abs_tol=1e-9)

    def test_csv_schema_has_20_columns(self) -> None:
        assert len(CSV_COLUMNS) == 20
        assert CSV_COLUMNS[0] == "step"


# ---------------------------------------------------------------------------
# build_run_plan
# ---------------------------------------------------------------------------


class TestBuildRunPlan:
    def test_returns_frozen_plan(self) -> None:
        plan = build_run_plan(resume_from=Path("checkpoints/stage2_final"))
        assert isinstance(plan, StageRunPlan)
        # Runtime attribute name keeps the frozen-dataclass assertion
        # opaque to static linters while still hitting FrozenInstanceError.
        mutable_field = "num_steps"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(plan, mutable_field, 999)

    def test_default_num_steps_150(self) -> None:
        plan = build_run_plan(resume_from=Path("checkpoints/stage2_final"))
        assert plan.num_steps == 150

    def test_warmup_ratio_zero(self) -> None:
        plan = build_run_plan(resume_from=Path("checkpoints/stage2_final"))
        assert plan.warmup_ratio == 0.0

    def test_resume_from_required(self) -> None:
        with pytest.raises(ValueError, match="requires resume_from"):
            build_run_plan(resume_from=None)

    def test_resume_from_must_be_path(self) -> None:
        with pytest.raises(TypeError, match="must be a pathlib.Path"):
            build_run_plan(resume_from=cast("Any", "checkpoints/stage2_final"))

    def test_num_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_steps must be"):
            build_run_plan(resume_from=Path("x"), num_steps=0)

    def test_language_weights_validated(self) -> None:
        bad = {"en": 0.95, "hinglish": 0.0, "hi": 0.0, "ta": 0.0, "kn": 0.05}
        with pytest.raises(ValueError, match=r"weight >= 0\.05"):
            build_run_plan(resume_from=Path("x"), language_weights=bad)

    def test_default_language_weights_pass(self) -> None:
        plan = build_run_plan(resume_from=Path("x"))
        assert plan.language_weights == LANGUAGE_WEIGHTS

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        plan = build_run_plan(
            resume_from=Path("x"),
            output_dir=tmp_path / "custom",
        )
        assert plan.output_dir == tmp_path / "custom"

    def test_stage_pinned_to_3(self) -> None:
        plan = build_run_plan(resume_from=Path("x"))
        assert plan.stage == 3

    def test_resume_from_preserved(self, tmp_path: Path) -> None:
        plan = build_run_plan(resume_from=tmp_path / "stage2")
        assert plan.resume_from == tmp_path / "stage2"


# ---------------------------------------------------------------------------
# write_local_csv_row + save_checkpoint
# ---------------------------------------------------------------------------


class TestLocalCsvCallback:
    def test_header_then_row(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        write_local_csv_row(csv_path=csv_path, logs={"step": 9, "train/reward_mean": 0.61})
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",")[0] == "step"
        assert lines[1].split(",")[0] == "9"

    def test_nan_encoded_literal_string(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        write_local_csv_row(
            csv_path=csv_path, logs={"step": 1, "train/reward_mean": float("nan")}
        )
        text = csv_path.read_text(encoding="utf-8")
        assert "nan" in text.splitlines()[1]


class TestSaveCheckpoint:
    def test_uses_safe_serialization_true(self, tmp_path: Path) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        out = save_checkpoint(model=model, tokenizer=tokenizer, output_dir=tmp_path / "ckpt")
        assert out == tmp_path / "ckpt"
        model.save_pretrained.assert_called_once_with(
            str(tmp_path / "ckpt"), safe_serialization=True
        )

    def test_no_naive_merge_called(self, tmp_path: Path) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        save_checkpoint(model=model, tokenizer=tokenizer, output_dir=tmp_path / "ckpt")
        assert not model.merge_and_unload.called
        assert not model.save_pretrained_merged.called


# ---------------------------------------------------------------------------
# BF16 + WandB + train()
# ---------------------------------------------------------------------------


class TestBF16Assertion:
    def test_bf16_slippage_raises_at_train_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_unsloth(monkeypatch, base_dtype=torch.bfloat16)
        _install_fake_peft(monkeypatch)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(BF16SlippageError, match="BF16 slipped through"):
            train(
                num_steps=1,
                resume_from=tmp_path / "stage2",
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_fp16_passes_assertion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        resumed_model, _ = _full_train_setup(monkeypatch, base_dtype=torch.float16)
        task_gen, env_factory, rollout_group_fn = _callables()
        out = train(
            num_steps=1,
            resume_from=tmp_path / "stage2",
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        assert out == tmp_path / "ckpt"
        resumed_model.save_pretrained.assert_called_once()


class TestWandBOfflineFallback:
    def test_offline_mode_skips_startup_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WANDB_MODE", "offline")
        _install_fake_unsloth(monkeypatch, base_dtype=torch.float16)
        _install_fake_peft(monkeypatch)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch, init_raises=True)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        out = train(
            num_steps=1,
            resume_from=tmp_path / "stage2",
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        assert out == tmp_path / "ckpt"

    def test_online_mode_raises_startup_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("WANDB_MODE", raising=False)
        _install_fake_unsloth(monkeypatch, base_dtype=torch.float16)
        _install_fake_peft(monkeypatch)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch, init_raises=True)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(WandBStartupError):
            train(
                num_steps=1,
                resume_from=tmp_path / "stage2",
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )


class TestTrainEntry:
    def test_train_requires_resume_from(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(ValueError, match="requires resume_from"):
            train(
                num_steps=1,
                resume_from=None,
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_train_rejects_wrong_stage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(ValueError, match="stage must be 3"):
            train(
                stage=cast("Any", 2),
                num_steps=1,
                resume_from=tmp_path / "stage2",
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_train_resumes_from_stage2_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        train(
            num_steps=1,
            resume_from=tmp_path / "stage2",
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        trainer = _last_trainer()
        assert len(trainer.train_calls) == 1
        assert trainer.train_calls[0] == {
            "resume_from_checkpoint": str(tmp_path / "stage2")
        }

    def test_train_attaches_stage2_adapters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        peft_mod = sys.modules["peft"]
        task_gen, env_factory, rollout_group_fn = _callables()
        train(
            num_steps=1,
            resume_from=tmp_path / "stage2",
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        peft_mod.PeftModel.from_pretrained.assert_called_once()
        kwargs = peft_mod.PeftModel.from_pretrained.call_args
        assert kwargs.args[1] == str(tmp_path / "stage2")
        assert kwargs.kwargs.get("is_trainable") is True

    def test_train_requires_callables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        with pytest.raises(ValueError, match="task_gen, env_factory"):
            train(
                num_steps=1,
                resume_from=tmp_path / "stage2",
                output_dir=tmp_path / "ckpt",
            )


class TestModuleSurface:
    def test_train_callable(self) -> None:
        assert callable(stage3.train)

    def test_module_no_pragma_violations(self) -> None:
        text = Path(stage3.__file__).read_text(encoding="utf-8")
        assert "type: ignore" not in text
        assert "noqa" not in text
