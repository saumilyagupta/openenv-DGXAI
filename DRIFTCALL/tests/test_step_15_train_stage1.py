"""Tests for cells/step_15_train_stage1.py.

CUDA + heavy deps mocked: ``unsloth``, ``trl``, ``wandb``, ``peft`` are
injected via ``sys.modules`` so the suite runs CPU-only. Covers:

- Constructor accepts stage param (Stage-1 fixed at 1).
- num_steps default == 150 (DESIGN.md §10.3 Stage-1 row).
- resume_from rejected for stage 1 (must be None).
- Language mix exactly 50/30/20 (en/hinglish/hi); ta=kn=0.
- warmup_ratio == 0.1 in the resolved plan.
- WandB offline fallback path covered (WANDB_MODE=offline never raises).
- BF16 assertion fires on dtype mismatch (V100 safety; training.md §3.1).
- Checkpoint save uses ``safe_serialization=True`` (DESIGN.md §10.5).
- ``write_local_csv_row`` uses the stable 20-column schema and encodes NaN.
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

from cells import step_15_train_stage1 as stage1
from cells.step_12_gemma_boot import BF16SlippageError
from cells.step_15_train_stage1 import (
    CSV_COLUMNS,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    LANGUAGE_WEIGHTS,
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
) -> tuple[MagicMock, MagicMock]:
    base_model = _fake_model_with_dtype(base_dtype)
    peft_model = MagicMock(name="peft_model")
    tokenizer = MagicMock(name="tokenizer")

    FastModel = MagicMock(name="FastModel")
    FastModel.from_pretrained = MagicMock(return_value=(base_model, tokenizer))
    FastModel.get_peft_model = MagicMock(return_value=peft_model)

    unsloth_mod: Any = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = FastModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return peft_model, tokenizer


_LAST_FAKE_TRAINER: list[_FakeGRPOTrainerBase] = []


class _FakeGRPOTrainerBase:
    """Real class standing in for ``trl.GRPOTrainer``.

    A real class is required because ``make_driftcall_grpo_trainer_cls``
    builds the subclass via ``type(name, (base,), ...)`` — that operation
    does a metaclass check which a ``MagicMock`` instance cannot satisfy.
    """

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
    """Install fakes covering both ``trl.GRPOConfig`` and ``trl.GRPOTrainer``."""

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
    peft_model, tokenizer = _install_fake_unsloth(monkeypatch, base_dtype=base_dtype)
    _install_fake_trl(monkeypatch)
    _install_fake_wandb(monkeypatch)
    _install_fake_rewards(monkeypatch)
    return peft_model, tokenizer


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


class TestStage1Constants:
    def test_stage_constant_is_1(self) -> None:
        assert STAGE == 1

    def test_default_num_steps_is_150(self) -> None:
        assert DEFAULT_NUM_STEPS == 150

    def test_warmup_ratio_is_0_1(self) -> None:
        assert math.isclose(WARMUP_RATIO, 0.1, abs_tol=1e-9)

    def test_default_output_dir(self) -> None:
        assert Path("checkpoints/stage1_final") == DEFAULT_OUTPUT_DIR

    def test_stage_base_seed_is_one_million(self) -> None:
        assert STAGE_BASE_SEED == 1_000_000


class TestStage1LanguageMix:
    def test_language_mix_50_30_20(self) -> None:
        assert math.isclose(LANGUAGE_WEIGHTS["en"], 0.50, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["hinglish"], 0.30, abs_tol=1e-9)
        assert math.isclose(LANGUAGE_WEIGHTS["hi"], 0.20, abs_tol=1e-9)

    def test_no_tamil_kannada_in_stage_1(self) -> None:
        assert LANGUAGE_WEIGHTS["ta"] == 0.0
        assert LANGUAGE_WEIGHTS["kn"] == 0.0

    def test_language_weights_sum_to_one(self) -> None:
        assert math.isclose(sum(LANGUAGE_WEIGHTS.values()), 1.0, abs_tol=1e-9)


class TestStage1CsvSchema:
    def test_csv_header_first_column_is_step(self) -> None:
        assert CSV_COLUMNS[0] == "step"

    def test_csv_has_exactly_20_columns(self) -> None:
        assert len(CSV_COLUMNS) == 20

    def test_csv_includes_per_language_block(self) -> None:
        for col in (
            "train/reward_hi",
            "train/reward_ta",
            "train/reward_kn",
            "train/reward_en",
        ):
            assert col in CSV_COLUMNS


# ---------------------------------------------------------------------------
# build_run_plan
# ---------------------------------------------------------------------------


class TestBuildRunPlan:
    def test_returns_frozen_stage_run_plan(self) -> None:
        plan = build_run_plan()
        assert isinstance(plan, StageRunPlan)
        # Use a runtime attribute name so neither ruff nor mypy can reduce
        # the call to a static read-only property assignment; the frozen
        # dataclass still raises FrozenInstanceError at runtime.
        mutable_field = "num_steps"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(plan, mutable_field, 999)

    def test_default_num_steps_150(self) -> None:
        plan = build_run_plan()
        assert plan.num_steps == 150

    def test_default_warmup_ratio_0_1(self) -> None:
        plan = build_run_plan()
        assert math.isclose(plan.warmup_ratio, 0.1, abs_tol=1e-9)

    def test_default_resume_from_is_none(self) -> None:
        plan = build_run_plan()
        assert plan.resume_from is None

    def test_resume_from_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not receive resume_from"):
            build_run_plan(resume_from=Path("checkpoints/stage0"))

    def test_num_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_steps must be"):
            build_run_plan(num_steps=0)

    def test_language_weights_copied_in_plan(self) -> None:
        plan = build_run_plan()
        assert plan.language_weights == LANGUAGE_WEIGHTS
        # Defensive copy — mutating the plan dict should not affect the module constant.
        plan.language_weights["en"] = 0.0
        assert LANGUAGE_WEIGHTS["en"] == 0.50

    def test_custom_output_dir_respected(self, tmp_path: Path) -> None:
        plan = build_run_plan(output_dir=tmp_path / "custom")
        assert plan.output_dir == tmp_path / "custom"

    def test_default_output_dir_used(self) -> None:
        plan = build_run_plan()
        assert plan.output_dir == DEFAULT_OUTPUT_DIR

    def test_stage_field_pinned_to_1(self) -> None:
        plan = build_run_plan()
        assert plan.stage == 1


# ---------------------------------------------------------------------------
# write_local_csv_row
# ---------------------------------------------------------------------------


class TestLocalCsvCallback:
    def test_header_written_on_first_call(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        write_local_csv_row(csv_path=csv_path, logs={"step": 1, "train/reward_mean": 0.5})
        content = csv_path.read_text(encoding="utf-8").splitlines()
        assert content[0].split(",")[0] == "step"
        assert len(content[0].split(",")) == 20
        assert len(content) == 2  # header + 1 row

    def test_subsequent_rows_dont_rewrite_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        write_local_csv_row(csv_path=csv_path, logs={"step": 1})
        write_local_csv_row(csv_path=csv_path, logs={"step": 2})
        content = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(content) == 3  # header + 2 rows
        assert content[1].split(",")[0] == "1"
        assert content[2].split(",")[0] == "2"

    def test_nan_encoded_as_string(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        write_local_csv_row(
            csv_path=csv_path,
            logs={"step": 1, "train/reward_mean": float("nan")},
        )
        text = csv_path.read_text(encoding="utf-8")
        assert "nan" in text.splitlines()[1]


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_save_uses_safe_serialization_true(self, tmp_path: Path) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        out = save_checkpoint(model=model, tokenizer=tokenizer, output_dir=tmp_path / "ckpt")
        assert out == tmp_path / "ckpt"
        model.save_pretrained.assert_called_once_with(
            str(tmp_path / "ckpt"), safe_serialization=True
        )
        tokenizer.save_pretrained.assert_called_once_with(str(tmp_path / "ckpt"))

    def test_save_creates_output_dir(self, tmp_path: Path) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        target = tmp_path / "fresh"
        save_checkpoint(model=model, tokenizer=tokenizer, output_dir=target)
        assert target.exists()

    def test_no_merge_and_unload_called(self, tmp_path: Path) -> None:
        model = MagicMock()
        tokenizer = MagicMock()
        save_checkpoint(model=model, tokenizer=tokenizer, output_dir=tmp_path / "ckpt")
        assert not model.merge_and_unload.called
        assert not model.save_pretrained_merged.called


# ---------------------------------------------------------------------------
# train(): BF16 slippage and WandB offline fallback
# ---------------------------------------------------------------------------


class TestBF16Assertion:
    def test_bf16_slippage_raises_at_train_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_unsloth(monkeypatch, base_dtype=torch.bfloat16)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(BF16SlippageError, match="BF16 slipped through"):
            train(
                num_steps=1,
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_fp16_model_passes_assertion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        peft_model, _ = _full_train_setup(monkeypatch, base_dtype=torch.float16)
        task_gen, env_factory, rollout_group_fn = _callables()
        out = train(
            num_steps=1,
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        assert out == tmp_path / "ckpt"
        peft_model.save_pretrained.assert_called_once()


class TestWandBOfflineFallback:
    def test_offline_mode_skips_startup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("WANDB_MODE", "offline")
        _install_fake_unsloth(monkeypatch, base_dtype=torch.float16)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch, init_raises=True)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        # Must NOT raise WandBStartupError when offline.
        out = train(
            num_steps=1,
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        assert out == tmp_path / "ckpt"

    def test_online_mode_raises_startup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("WANDB_MODE", raising=False)
        _install_fake_unsloth(monkeypatch, base_dtype=torch.float16)
        _install_fake_trl(monkeypatch)
        _install_fake_wandb(monkeypatch, init_raises=True)
        _install_fake_rewards(monkeypatch)

        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(WandBStartupError):
            train(
                num_steps=1,
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )


class TestTrainEntryValidation:
    def test_train_rejects_resume_from_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(ValueError, match="must not receive resume_from"):
            train(
                num_steps=1,
                resume_from=tmp_path / "stage0",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_train_rejects_wrong_stage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        with pytest.raises(ValueError, match="stage must be 1"):
            train(
                stage=cast("Any", 2),
                num_steps=1,
                output_dir=tmp_path / "ckpt",
                task_gen=task_gen,
                env_factory=env_factory,
                rollout_group_fn=rollout_group_fn,
            )

    def test_train_requires_callables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        with pytest.raises(ValueError, match="task_gen, env_factory"):
            train(num_steps=1, output_dir=tmp_path / "ckpt")

    def test_train_invokes_trainer_train_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _full_train_setup(monkeypatch)
        task_gen, env_factory, rollout_group_fn = _callables()
        train(
            num_steps=1,
            output_dir=tmp_path / "ckpt",
            task_gen=task_gen,
            env_factory=env_factory,
            rollout_group_fn=rollout_group_fn,
        )
        # Stage 1 calls trainer.train() with NO resume_from_checkpoint kwarg.
        trainer = _last_trainer()
        assert len(trainer.train_calls) == 1
        assert trainer.train_calls[0] == {}


# Module surface sanity (introspection-only — keeps the public contract honest)
class TestModuleSurface:
    def test_train_callable_exposed(self) -> None:
        assert callable(stage1.train)

    def test_module_has_no_type_ignore_or_noqa(self) -> None:
        src_path = Path(stage1.__file__)
        text = src_path.read_text(encoding="utf-8")
        assert "type: ignore" not in text
        assert "noqa" not in text
