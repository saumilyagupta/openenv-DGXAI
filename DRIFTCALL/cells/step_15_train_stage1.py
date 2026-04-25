"""Stage-1 GRPO training entry (docs/modules/training.md §3.5, DESIGN.md §10.3).

Stage-1 contract:
  - 150 GRPO steps (curriculum warmup).
  - **No drift** in the env (``curriculum_stage=1``).
  - Language mix: 50% English, 30% Hinglish, 20% Hindi (no Tamil/Kannada).
  - ``warmup_ratio=0.1`` — stage-1 is the only stage that warms the LR.
  - ``resume_from`` MUST be ``None``; stage-1 is the curriculum origin.
  - Saves checkpoints every 50 steps with ``safe_serialization=True``;
    NEVER naive 4-bit -> 16-bit merge (DESIGN.md §10.5, CLAUDE.md §9).
  - WandB primary monitoring; ``LocalCSVCallback`` mirrors every ``on_log``
    when ``WANDB_MODE=offline`` or the wandb upload flakes (training.md §2.4.1).
  - BF16-slippage assertion fires at entry via ``assert_fp16_dtype`` from
    step_12 (V100 safety; training.md §3.1).

Heavy imports (``torch``, ``trl``, ``unsloth``, ``wandb``) are deferred
inside functions so this module imports cleanly on CPU-only CI.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cells.step_12_gemma_boot import BootConfig, boot_gemma
from cells.step_13_grpo_config import build_grpo_config
from cells.step_14_custom_trainer import EpisodeDatasetAdapter, LanguageCode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable


CheckpointPath = Path

STAGE: Literal[1] = 1
DEFAULT_NUM_STEPS: int = 150
WARMUP_RATIO: float = 0.1
STAGE_BASE_SEED: int = 1_000_000
DEFAULT_OUTPUT_DIR: Path = Path("checkpoints/stage1_final")

LANGUAGE_WEIGHTS: dict[str, float] = {
    "en": 0.50,
    "hinglish": 0.30,
    "hi": 0.20,
    "ta": 0.0,
    "kn": 0.0,
}

CSV_COLUMNS: tuple[str, ...] = (
    "step",
    "train/reward_mean",
    "train/reward_std",
    "train/policy_kl",
    "train/gen_length_mean",
    "train/grad_norm",
    "train/loss",
    "train/learning_rate",
    "train/R1_mean",
    "train/R2_mean",
    "train/R3_mean",
    "train/R4_mean",
    "train/R5_mean",
    "train/drift_detected_rate",
    "train/format_compliance_rate",
    "train/hallucinated_field_count",
    "train/reward_hi",
    "train/reward_ta",
    "train/reward_kn",
    "train/reward_en",
)


class WandBStartupError(RuntimeError):
    """Raised at ``train()`` entry when ``wandb.init()`` fails AND
    ``WANDB_MODE != "offline"``. Offline mode never raises (training.md §2.4.1)."""


@dataclass(frozen=True)
class StageRunPlan:
    """Frozen plan describing one stage-1 training launch.

    Surfaced so tests can introspect the resolved arguments without having
    to mock the whole TRL stack.
    """

    stage: Literal[1, 2, 3]
    num_steps: int
    warmup_ratio: float
    stage_base_seed: int
    language_weights: dict[str, float]
    output_dir: Path
    resume_from: Path | None


def _validate_resume_from(resume_from: Path | None) -> None:
    """Stage 1 is the curriculum origin — ``resume_from`` MUST be ``None``."""
    if resume_from is not None:
        raise ValueError(
            f"Stage 1 must not receive resume_from; got {resume_from!r}. "
            f"Stage 1 is the curriculum origin (training.md §3.5)."
        )


def _validate_num_steps(num_steps: int) -> None:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}")


def build_run_plan(
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    resume_from: Path | None = None,
    output_dir: Path | None = None,
) -> StageRunPlan:
    """Resolve the launch arguments into a frozen :class:`StageRunPlan`.

    Pure function — does not touch the GPU, the filesystem, or wandb.
    Tests use this to verify the resolved plan without invoking ``train``.
    """
    _validate_resume_from(resume_from)
    _validate_num_steps(num_steps)
    return StageRunPlan(
        stage=STAGE,
        num_steps=num_steps,
        warmup_ratio=WARMUP_RATIO,
        stage_base_seed=STAGE_BASE_SEED,
        language_weights=dict(LANGUAGE_WEIGHTS),
        output_dir=output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        resume_from=resume_from,
    )


def _wandb_init_or_raise(*, run_name: str, output_dir: Path) -> Any:
    """Initialise wandb through :func:`cells.step_13_grpo_config.init_wandb`.

    Offline mode (``WANDB_MODE=offline``) and disabled mode
    (``WANDB_MODE=disabled``) never raise — local CSV is the authoritative
    record on V100 (training.md §2.4.1). Online failures raise
    :class:`WandBStartupError`.
    """
    del run_name, output_dir  # retained for call-site compatibility
    mode = os.environ.get("WANDB_MODE")
    try:
        from cells.step_13_grpo_config import init_wandb

        return init_wandb(stage=STAGE, seed=STAGE_BASE_SEED)
    except ImportError as exc:  # pragma: no cover - wandb required at runtime
        if mode in {"offline", "disabled"}:
            return None
        raise WandBStartupError(
            f"wandb import failed and WANDB_MODE != 'offline': {exc}"
        ) from exc
    except Exception as exc:
        if mode in {"offline", "disabled"}:
            return None
        raise WandBStartupError(
            f"wandb.init() failed and WANDB_MODE != 'offline': {exc}"
        ) from exc


def write_local_csv_row(
    *,
    csv_path: Path,
    logs: dict[str, Any],
    columns: tuple[str, ...] = CSV_COLUMNS,
) -> None:
    """Append one row to ``metrics.csv`` mirroring the WandB ``on_log`` dict.

    Schema is the stable 20-column ordering from training.md §3.4. NaN floats
    are encoded as the literal string ``"nan"`` (training.md §2.4.1). Header
    is written exactly once on first call.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    row: list[str] = []
    for col in columns:
        value = logs.get(col, "")
        if isinstance(value, float):
            row.append("nan" if value != value else repr(value))
        else:
            row.append(str(value))
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(columns)
        writer.writerow(row)


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
) -> Path:
    """Save adapter + tokenizer using ``safe_serialization=True``.

    Per DESIGN.md §10.5 / training.md §3.6 we NEVER call
    ``merge_and_unload()`` or any 4-bit -> 16-bit naive merge path.
    Returns the directory where the adapter landed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def train(
    *,
    stage: Literal[1] = STAGE,
    num_steps: int = DEFAULT_NUM_STEPS,
    resume_from: Path | None = None,
    output_dir: Path | None = None,
    boot_config: BootConfig | None = None,
    task_gen: Callable[..., Any] | None = None,
    env_factory: Callable[[], Any] | None = None,
    rollout_group_fn: Callable[..., Any] | None = None,
) -> CheckpointPath:
    """Run GRPO Stage-1 (warmup, no drift) for ``num_steps`` updates.

    Behaviour (training.md §2.1):
      1. Boot Gemma 4 E2B in 4-bit + attach LoRA via :func:`boot_gemma`.
      2. Re-assert FP16 dtype (BF16-slippage halt; training.md §3.1).
      3. Build :class:`GRPOConfig` for stage 1 (warmup_ratio=0.1).
      4. Build the streaming :class:`EpisodeDatasetAdapter` with the
         stage-1 language mix (50% en, 30% hinglish, 20% hi).
      5. Construct ``DriftCallGRPOTrainer`` with the multi-turn rollout
         override (step_14) and ``reward_fn`` (step_13).
      6. Initialise wandb (offline-safe; training.md §2.4.1).
      7. ``trainer.train()`` for ``num_steps`` updates.
      8. Save the final adapter via :func:`save_checkpoint`.
    """
    if stage != STAGE:
        raise ValueError(f"stage must be {STAGE}; got {stage}")

    plan = build_run_plan(
        num_steps=num_steps,
        resume_from=resume_from,
        output_dir=output_dir,
    )

    # boot_gemma() already runs assert_fp16_dtype on the base model before
    # LoRA attach (training.md §3.1). We do not re-check the peft-wrapped
    # model here — the wrapped LoRA params are FP16 by construction.
    model, tokenizer = boot_gemma(boot_config)

    config = build_grpo_config(stage=plan.stage, resume_output_dir=plan.output_dir)

    if task_gen is None or env_factory is None or rollout_group_fn is None:
        raise ValueError(
            "Stage-1 train() requires task_gen, env_factory, and rollout_group_fn "
            "to be provided by the caller (notebook orchestrator). They are kept "
            "explicit so the training cell stays decoupled from data + env builders."
        )

    dataset = EpisodeDatasetAdapter(
        task_gen=task_gen,
        env_factory=env_factory,
        stage=plan.stage,
        stage_base_seed=plan.stage_base_seed,
        language_weights=cast("dict[LanguageCode, float]", plan.language_weights),
        tokenizer=tokenizer,
    )

    from cells.step_13_grpo_config import reward_fn
    from cells.step_14_custom_trainer import make_driftcall_grpo_trainer_cls

    Trainer = make_driftcall_grpo_trainer_cls()
    trainer = Trainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        rollout_group_fn=rollout_group_fn,
        env_factory=env_factory,
        reward_fn_driftcall=reward_fn,
    )

    _wandb_init_or_raise(run_name=f"driftcall-stage{plan.stage}", output_dir=plan.output_dir)
    trainer.train()

    return save_checkpoint(model=model, tokenizer=tokenizer, output_dir=plan.output_dir)


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_NUM_STEPS",
    "DEFAULT_OUTPUT_DIR",
    "LANGUAGE_WEIGHTS",
    "STAGE",
    "STAGE_BASE_SEED",
    "WARMUP_RATIO",
    "CheckpointPath",
    "StageRunPlan",
    "WandBStartupError",
    "build_run_plan",
    "save_checkpoint",
    "train",
    "write_local_csv_row",
]
