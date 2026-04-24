"""GRPOConfig builder + reward_fn wiring (docs/modules/training.md §2.4, §2.3).

Two public entry points:

- :func:`build_grpo_config(stage, *, num_generations=8, resume_output_dir=None)`
  returns a TRL ``GRPOConfig`` whose fields match training.md §2.4 verbatim.
  Invariants (asserted post-construction): ``use_bias_correction_kl is True``,
  ``fp16 is True``, ``gradient_checkpointing is True``,
  ``per_device_train_batch_size == 1``, ``num_generations in {4, 8}``,
  ``num_generations * gradient_accumulation_steps == 32``, ``beta == 0.04``,
  ``max_prompt_length == 1024``, ``max_completion_length == 2048``,
  ``warmup_ratio == (0.1 if stage == 1 else 0.0)``.

- :func:`reward_fn(prompts, completions, *, _meta, episodes, **kwargs)` is the
  TRL-0.23 reward contract used by ``DriftCallGRPOTrainer``. It is a pure
  delegating wrapper over ``cells.step_08_rewards.compute_rewards`` (see
  docs/modules/rewards.md §3.1 purity contract). No pre-normalization,
  no RNG, no I/O.

TRL is imported lazily inside ``build_grpo_config`` so this cell loads on
CPU-only CI. ``compute_rewards`` is imported lazily so step_08 landing after
step_13 does not cascade-break the import graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

StageT = Literal[1, 2, 3]


LEARNING_RATE: float = 5e-6
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.99
WEIGHT_DECAY: float = 0.01
LR_SCHEDULER_TYPE: str = "cosine"
OPTIM: str = "paged_adamw_8bit"

PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
EFFECTIVE_ROLLOUTS_PER_UPDATE: int = 32

DEFAULT_NUM_GENERATIONS: int = 8
ALLOWED_NUM_GENERATIONS: tuple[int, ...] = (4, 8)

MAX_PROMPT_LENGTH: int = 1024
MAX_COMPLETION_LENGTH: int = 2048

BETA_KL: float = 0.04

SAMPLING_TEMPERATURE: float = 0.9
SAMPLING_TOP_P: float = 0.95

LOGGING_STEPS: int = 5
SAVE_STEPS: int = 50
SAVE_TOTAL_LIMIT: int = 10

REPORT_TO: str = "wandb"

WARMUP_RATIO_STAGE1: float = 0.1
WARMUP_RATIO_STAGE2_3: float = 0.0


@dataclass(frozen=True)
class _ConfigInvariants:
    """Invariant bundle returned by :func:`assert_config_invariants`.

    Used by tests to verify exact field values without re-parsing the
    ``GRPOConfig`` object.
    """

    stage: StageT
    num_generations: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    beta: float
    max_prompt_length: int
    max_completion_length: int
    per_device_train_batch_size: int
    use_bias_correction_kl: bool
    fp16: bool
    gradient_checkpointing: bool
    report_to: str
    run_name: str
    output_dir: str


def _derive_grad_accum(num_generations: int) -> int:
    """Return grad_accum so that G*grad_accum == 32 (training.md §7b)."""
    return 8 if num_generations == 4 else 4


def _warmup_ratio_for_stage(stage: StageT) -> float:
    """One continuous cosine schedule across 500 steps — only stage-1 warms."""
    return WARMUP_RATIO_STAGE1 if stage == 1 else WARMUP_RATIO_STAGE2_3


def _validate_num_generations(num_generations: int) -> None:
    if num_generations not in ALLOWED_NUM_GENERATIONS:
        raise AssertionError(
            f"num_generations in {{4, 8}} required; got {num_generations}"
        )


def _validate_stage(stage: int) -> None:
    if stage not in (1, 2, 3):
        raise AssertionError(f"stage in {{1, 2, 3}} required; got {stage}")


def build_grpo_config(
    stage: StageT,
    *,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    resume_output_dir: Path | None = None,
) -> Any:
    """Build a TRL ``GRPOConfig`` matching training.md §2.4 exactly.

    Validates ``num_generations in {4, 8}`` before import so CPU-only
    tests can trigger the assertion without TRL installed.
    """
    _validate_stage(stage)
    _validate_num_generations(num_generations)

    warmup_ratio = _warmup_ratio_for_stage(stage)
    grad_accum = _derive_grad_accum(num_generations)
    output_dir = str(resume_output_dir) if resume_output_dir is not None else f"checkpoints/stage{stage}"
    run_name = f"driftcall-stage{stage}"

    from trl import GRPOConfig

    config = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA_KL,
        use_bias_correction_kl=True,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        output_dir=output_dir,
        report_to=REPORT_TO,
        run_name=run_name,
    )

    assert_config_invariants(config, stage=stage, num_generations=num_generations)
    return config


def assert_config_invariants(
    config: Any,
    *,
    stage: StageT,
    num_generations: int,
) -> _ConfigInvariants:
    """Post-construction field checks — training.md §2.4 invariants.

    Returns a frozen :class:`_ConfigInvariants` snapshot so callers (tests)
    can introspect without re-reading the mutable TRL config object.
    """
    if getattr(config, "use_bias_correction_kl", None) is not True:
        raise AssertionError(
            "use_bias_correction_kl must be True (TRL issue #4637; training.md §3.3)"
        )
    if getattr(config, "fp16", None) is not True:
        raise AssertionError("fp16 must be True on V100 (training.md §3.1)")
    if getattr(config, "bf16", False) is True:
        raise AssertionError("bf16 must be False on V100 (training.md §3.1)")
    if getattr(config, "gradient_checkpointing", None) is not True:
        raise AssertionError("gradient_checkpointing must be True")
    if config.per_device_train_batch_size != PER_DEVICE_TRAIN_BATCH_SIZE:
        raise AssertionError(
            f"per_device_train_batch_size must be {PER_DEVICE_TRAIN_BATCH_SIZE}"
        )
    if config.num_generations != num_generations:
        raise AssertionError(
            f"num_generations mismatch: config has {config.num_generations}, expected {num_generations}"
        )
    expected_grad_accum = _derive_grad_accum(num_generations)
    if config.gradient_accumulation_steps != expected_grad_accum:
        raise AssertionError(
            f"gradient_accumulation_steps must be {expected_grad_accum} when "
            f"num_generations == {num_generations}"
        )
    product = config.num_generations * config.gradient_accumulation_steps
    if product != EFFECTIVE_ROLLOUTS_PER_UPDATE:
        raise AssertionError(
            f"num_generations * gradient_accumulation_steps must be "
            f"{EFFECTIVE_ROLLOUTS_PER_UPDATE}; got {product}"
        )
    expected_warmup = _warmup_ratio_for_stage(stage)
    if config.warmup_ratio != expected_warmup:
        raise AssertionError(
            f"warmup_ratio must be {expected_warmup} for stage {stage}; "
            f"got {config.warmup_ratio}"
        )
    if config.beta != BETA_KL:
        raise AssertionError(f"beta must be {BETA_KL}; got {config.beta}")
    if config.max_prompt_length != MAX_PROMPT_LENGTH:
        raise AssertionError(f"max_prompt_length must be {MAX_PROMPT_LENGTH}")
    if config.max_completion_length != MAX_COMPLETION_LENGTH:
        raise AssertionError(
            f"max_completion_length must be {MAX_COMPLETION_LENGTH}"
        )
    if config.report_to != REPORT_TO:
        raise AssertionError(f"report_to must be {REPORT_TO!r}")
    expected_run_name = f"driftcall-stage{stage}"
    if config.run_name != expected_run_name:
        raise AssertionError(
            f"run_name must be {expected_run_name!r}; got {config.run_name!r}"
        )

    return _ConfigInvariants(
        stage=stage,
        num_generations=config.num_generations,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        beta=config.beta,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        use_bias_correction_kl=config.use_bias_correction_kl,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        run_name=config.run_name,
        output_dir=config.output_dir,
    )


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def reward_fn(
    prompts: list[str],
    completions: list[str],
    *,
    _meta: list[dict[str, Any]],
    episodes: list[Any],
    **kwargs: Any,
) -> list[float]:
    """TRL-0.23-compatible reward function (training.md §2.3).

    Contract:
      - ``prompts``, ``completions``, ``_meta``, ``episodes`` all have the
        same length G (num_generations).
      - Delegates to ``compute_rewards`` per-episode; returns
        ``[r.reward for r in rewards_list]`` with each value clamped to
        ``[0, 1]`` and rounded to 3 decimals.
      - No reward normalization pre-GRPO — group-relative advantage is
        applied inside TRL (training.md §3.2, DESIGN.md §7.4).
      - No RNG, no clock, no I/O (rewards.md §3.1).
    """
    if len(episodes) != len(prompts) or len(episodes) != len(completions):
        raise ValueError(
            f"prompts/completions/episodes length mismatch: "
            f"{len(prompts)}, {len(completions)}, {len(episodes)}"
        )
    if len(_meta) != len(episodes):
        raise ValueError(
            f"_meta length {len(_meta)} != episodes length {len(episodes)}"
        )

    from cells.step_08_rewards import compute_rewards

    out: list[float] = []
    for ep in episodes:
        rewards = compute_rewards(ep)
        out.append(round(_clamp_unit(float(rewards.reward)), 3))
    return out


__all__ = [
    "ALLOWED_NUM_GENERATIONS",
    "BETA_KL",
    "DEFAULT_NUM_GENERATIONS",
    "EFFECTIVE_ROLLOUTS_PER_UPDATE",
    "LEARNING_RATE",
    "MAX_COMPLETION_LENGTH",
    "MAX_PROMPT_LENGTH",
    "PER_DEVICE_TRAIN_BATCH_SIZE",
    "REPORT_TO",
    "StageT",
    "WARMUP_RATIO_STAGE1",
    "WARMUP_RATIO_STAGE2_3",
    "assert_config_invariants",
    "build_grpo_config",
    "reward_fn",
]
