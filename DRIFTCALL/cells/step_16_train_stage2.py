"""Stage-2 GRPO training entry (docs/modules/training.md §3.5, DESIGN.md §10.3).

Stage-2 contract:
  - 200 GRPO steps (single-drift curriculum).
  - **One drift per episode** in the env (``curriculum_stage=2``).
  - Language mix: 30% English, 30% Hinglish, 20% Hindi, 10% Tamil, 10% Kannada.
  - ``warmup_ratio=0.0`` — never re-warm the LR mid-curriculum
    (training.md §3.5; one continuous cosine across all 500 steps).
  - ``resume_from`` is REQUIRED — must point at the Stage-1 final
    checkpoint directory. None is rejected.
  - Validates ``language_weights`` per training.md §7f: every non-English
    cohort must carry weight >= 0.05 at stage >= 2.
  - Saves checkpoints every 50 steps with ``safe_serialization=True``;
    NEVER naive 4-bit -> 16-bit merge (DESIGN.md §10.5, CLAUDE.md §9).
  - WandB primary monitoring; ``LocalCSVCallback`` mirrors every ``on_log``
    when ``WANDB_MODE=offline`` or the wandb upload flakes (training.md §2.4.1).
  - BF16-slippage assertion fires at entry via ``assert_fp16_dtype`` from
    step_12 (V100 safety; training.md §3.1).

Heavy imports (``torch``, ``trl``, ``unsloth``, ``wandb``, ``peft``) are
deferred inside functions so this module imports cleanly on CPU-only CI.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cells.step_12_gemma_boot import BootConfig, assert_fp16_dtype
from cells.step_13_grpo_config import build_grpo_config
from cells.step_14_custom_trainer import EpisodeDatasetAdapter, LanguageCode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable


CheckpointPath = Path

STAGE: Literal[2] = 2
DEFAULT_NUM_STEPS: int = 200
WARMUP_RATIO: float = 0.0
STAGE_BASE_SEED: int = 2_000_000
DEFAULT_OUTPUT_DIR: Path = Path("checkpoints/stage2_final")
COHORT_MIN_WEIGHT_AT_STAGE_GE_2: float = 0.05
NON_ENGLISH_LANGUAGES: tuple[str, ...] = ("hi", "ta", "kn", "hinglish")

LANGUAGE_WEIGHTS: dict[str, float] = {
    "en": 0.30,
    "hinglish": 0.30,
    "hi": 0.20,
    "ta": 0.10,
    "kn": 0.10,
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
    """Frozen plan describing one stage-2 training launch."""

    stage: Literal[1, 2, 3]
    num_steps: int
    warmup_ratio: float
    stage_base_seed: int
    language_weights: dict[str, float]
    output_dir: Path
    resume_from: Path


def _validate_resume_from(resume_from: Path | None) -> Path:
    """Stage 2 REQUIRES a stage-1 checkpoint to resume from."""
    if resume_from is None:
        raise ValueError(
            "Stage 2 requires resume_from (path to Stage-1 final checkpoint); "
            "got None (training.md §3.5 stage transitions)."
        )
    if not isinstance(resume_from, Path):
        raise TypeError(
            f"resume_from must be a pathlib.Path; got {type(resume_from).__name__}"
        )
    return resume_from


def _validate_num_steps(num_steps: int) -> None:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}")


def _validate_language_weights(language_weights: dict[str, float]) -> None:
    """Every non-English cohort must carry weight >= 0.05 at stage 2/3.

    Prevents :class:`LanguageCohortCollapseError` upstream
    (training.md §7f).
    """
    for lang in NON_ENGLISH_LANGUAGES:
        weight = language_weights.get(lang, 0.0)
        if weight < COHORT_MIN_WEIGHT_AT_STAGE_GE_2:
            raise ValueError(
                f"language_weights['{lang}'] = {weight} < "
                f"{COHORT_MIN_WEIGHT_AT_STAGE_GE_2}; weight >= 0.05 for "
                f"non-English at stage >= 2 (training.md §7f)."
            )


def build_run_plan(
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    resume_from: Path | None = None,
    output_dir: Path | None = None,
    language_weights: dict[str, float] | None = None,
) -> StageRunPlan:
    """Resolve the launch arguments into a frozen :class:`StageRunPlan`.

    Pure function — does not touch the GPU, the filesystem, or wandb.
    """
    resolved_resume = _validate_resume_from(resume_from)
    _validate_num_steps(num_steps)
    weights = dict(language_weights) if language_weights is not None else dict(LANGUAGE_WEIGHTS)
    _validate_language_weights(weights)
    return StageRunPlan(
        stage=STAGE,
        num_steps=num_steps,
        warmup_ratio=WARMUP_RATIO,
        stage_base_seed=STAGE_BASE_SEED,
        language_weights=weights,
        output_dir=output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        resume_from=resolved_resume,
    )


def _wandb_init_or_raise(*, run_name: str, output_dir: Path) -> Any:
    """Initialise wandb through :func:`cells.step_13_grpo_config.init_wandb`.

    Offline mode (``WANDB_MODE=offline``) and disabled mode
    (``WANDB_MODE=disabled``) never raise; online failures raise
    :class:`WandBStartupError` (training.md §2.4.1, §3.3.3).
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
    """Append one row to ``metrics.csv`` mirroring the WandB ``on_log`` dict."""
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
    """Save adapter + tokenizer using ``safe_serialization=True``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _load_base_model(boot_config: BootConfig | None) -> tuple[Any, Any]:
    """Load the 4-bit Gemma base model (no LoRA attach) and verify FP16.

    Stage 2 must NOT call :func:`cells.step_12_gemma_boot.boot_gemma`
    because that helper attaches a *fresh* LoRA via ``get_peft_model``;
    we instead load the base only, then wrap with the saved Stage-1
    adapters via :func:`_load_stage1_adapters` (training.md §3.1, §3.6).
    """
    cfg = boot_config if boot_config is not None else BootConfig()

    import torch
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
    )
    assert_fp16_dtype(model)
    return model, tokenizer


def _load_stage1_adapters(model: Any, resume_from: Path) -> Any:
    """Attach the Stage-1 LoRA adapters to the freshly-booted base model.

    Returns the wrapped :class:`PeftModel`. Heavy import deferred so the
    cell loads on CPU-only CI without ``peft`` installed.
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)


def train(
    *,
    stage: Literal[2] = STAGE,
    num_steps: int = DEFAULT_NUM_STEPS,
    resume_from: Path | None = None,
    output_dir: Path | None = None,
    boot_config: BootConfig | None = None,
    task_gen: Callable[..., Any] | None = None,
    env_factory: Callable[[], Any] | None = None,
    rollout_group_fn: Callable[..., Any] | None = None,
) -> CheckpointPath:
    """Run GRPO Stage-2 (single drift) for ``num_steps`` updates.

    Behaviour (training.md §3.5 stage transitions):
      1. Load Gemma 4 E2B base in 4-bit (FP16-pinned) — no fresh LoRA.
      2. Assert FP16 dtype on the base (BF16-slippage halt).
      3. Attach Stage-1 LoRA adapters via ``PeftModel.from_pretrained``.
      4. Build :class:`GRPOConfig` for stage 2 (warmup_ratio=0.0).
      5. Build the streaming :class:`EpisodeDatasetAdapter` with the
         stage-2 language mix.
      6. Construct ``DriftCallGRPOTrainer`` with the multi-turn rollout
         override and ``reward_fn``.
      7. Initialise wandb (offline-safe).
      8. ``trainer.train(resume_from_checkpoint=str(resume_from))`` —
         restores optimizer/scheduler state + TRL-internal RNG.
      9. Save the final adapter via :func:`save_checkpoint`.
    """
    if stage != STAGE:
        raise ValueError(f"stage must be {STAGE}; got {stage}")

    plan = build_run_plan(
        num_steps=num_steps,
        resume_from=resume_from,
        output_dir=output_dir,
    )

    base_model, tokenizer = _load_base_model(boot_config)
    model = _load_stage1_adapters(base_model, plan.resume_from)

    config = build_grpo_config(stage=plan.stage, resume_output_dir=plan.output_dir)
    config.max_steps = plan.num_steps

    if task_gen is None or env_factory is None or rollout_group_fn is None:
        raise ValueError(
            "Stage-2 train() requires task_gen, env_factory, and rollout_group_fn "
            "to be provided by the caller (notebook orchestrator)."
        )

    dataset = EpisodeDatasetAdapter(
        task_gen=task_gen,
        env_factory=env_factory,
        stage=plan.stage,
        stage_base_seed=plan.stage_base_seed,
        language_weights=cast("dict[LanguageCode, float]", plan.language_weights),
        tokenizer=tokenizer,
        num_steps=plan.num_steps,
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
    trainer.train(resume_from_checkpoint=str(plan.resume_from))

    return save_checkpoint(model=model, tokenizer=tokenizer, output_dir=plan.output_dir)


__all__ = [
    "COHORT_MIN_WEIGHT_AT_STAGE_GE_2",
    "CSV_COLUMNS",
    "DEFAULT_NUM_STEPS",
    "DEFAULT_OUTPUT_DIR",
    "LANGUAGE_WEIGHTS",
    "NON_ENGLISH_LANGUAGES",
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
