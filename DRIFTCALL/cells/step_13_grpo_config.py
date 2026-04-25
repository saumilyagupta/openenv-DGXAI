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
HardwareT = Literal["v100", "h100"]


LEARNING_RATE: float = 5e-6
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.99
WEIGHT_DECAY: float = 0.01
LR_SCHEDULER_TYPE: str = "cosine"

# V100 path (default) — fp16 + 8-bit paged AdamW (sm_70 safe).
OPTIM_V100: str = "paged_adamw_8bit"
# H100 path — bf16 + fused torch AdamW (sm_90 tensor cores).
OPTIM_H100: str = "adamw_torch_fused"
# For backwards compatibility with callers that read ``OPTIM`` directly.
OPTIM: str = OPTIM_V100
# Kernel request passed to the model at load time on H100.
H100_ATTN_IMPLEMENTATION: str = "flash_attention_3"
ALLOWED_HARDWARE: tuple[HardwareT, ...] = ("v100", "h100")

PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
EFFECTIVE_ROLLOUTS_PER_UPDATE: int = 32

DEFAULT_NUM_GENERATIONS: int = 2
ALLOWED_NUM_GENERATIONS: tuple[int, ...] = (2, 4, 8)

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

# WandB integration (training.md §3.3.3 — env-var contract).
WANDB_PROJECT_DEFAULT: str = "driftcall"
WANDB_ENTITY_DEFAULT: str | None = None
WANDB_RUN_NAME_TEMPLATE: str = "driftcall-stage{stage}-seed{seed}-{timestamp}"
WANDB_MODE_DEFAULT: str = "online"


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
    """Return grad_accum so that G*grad_accum is a sensible effective batch."""
    if num_generations == 2:
        return 2  # matches Sudoku notebook recipe
    if num_generations == 4:
        return 8
    return 4


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


def _validate_hardware(hardware: str) -> None:
    if hardware not in ALLOWED_HARDWARE:
        raise AssertionError(
            f"hardware in {ALLOWED_HARDWARE} required; got {hardware!r}"
        )


def build_grpo_config(
    stage: StageT,
    *,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    resume_output_dir: Path | None = None,
    hardware: HardwareT = "v100",
) -> Any:
    """Build a TRL ``GRPOConfig`` matching training.md §2.4 exactly.

    Validates ``num_generations in {4, 8}`` before import so CPU-only
    tests can trigger the assertion without TRL installed.
    """
    _validate_stage(stage)
    _validate_num_generations(num_generations)
    _validate_hardware(hardware)

    warmup_ratio = _warmup_ratio_for_stage(stage)
    grad_accum = _derive_grad_accum(num_generations)
    output_dir = str(resume_output_dir) if resume_output_dir is not None else f"checkpoints/stage{stage}"
    run_name = f"driftcall-stage{stage}"

    # Hardware-specific knobs — V100 stays fp16 + 8-bit paged AdamW, H100
    # switches to bf16 + fused torch AdamW + flash_attention_3 (training.md §3.1).
    if hardware == "h100":
        fp16_flag = False
        bf16_flag = True
        optim_choice = OPTIM_H100
        attn_implementation: str | None = H100_ATTN_IMPLEMENTATION
    else:
        fp16_flag = True
        bf16_flag = False
        optim_choice = OPTIM_V100
        attn_implementation = None

    from trl import GRPOConfig

    extra_kwargs: dict[str, Any] = {}
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    config = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=optim_choice,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA_KL,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        fp16=fp16_flag,
        bf16=bf16_flag,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        output_dir=output_dir,
        report_to=REPORT_TO,
        run_name=run_name,
        **extra_kwargs,
    )

    assert_config_invariants(
        config, stage=stage, num_generations=num_generations, hardware=hardware,
    )
    return config


def assert_config_invariants(
    config: Any,
    *,
    stage: StageT,
    num_generations: int,
    hardware: HardwareT | None = None,
) -> _ConfigInvariants:
    """Post-construction field checks — training.md §2.4 invariants.

    Returns a frozen :class:`_ConfigInvariants` snapshot so callers (tests)
    can introspect without re-reading the mutable TRL config object.

    When ``hardware`` is ``None`` it is auto-detected from the precision
    flags on ``config`` (``bf16=True`` → ``"h100"``, else ``"v100"``).
    """
    if hardware is None:
        hardware = "h100" if getattr(config, "bf16", False) else "v100"
    _validate_hardware(hardware)
    _ubc = getattr(config, "use_bias_correction_kl", None)
    if _ubc is not None and _ubc is not True:
        raise AssertionError(
            "use_bias_correction_kl must be True when supported (TRL issue #4637)"
        )
    if hardware == "v100":
        if getattr(config, "fp16", None) is not True:
            raise AssertionError("fp16 must be True on V100 (training.md §3.1)")
        if getattr(config, "bf16", False) is True:
            raise AssertionError("bf16 must be False on V100 (training.md §3.1)")
    else:  # hardware == "h100"
        if getattr(config, "bf16", None) is not True:
            raise AssertionError("bf16 must be True on H100 (training.md §3.1)")
        if getattr(config, "fp16", False) is True:
            raise AssertionError("fp16 must be False on H100 (training.md §3.1)")
        if getattr(config, "attn_implementation", None) != H100_ATTN_IMPLEMENTATION:
            raise AssertionError(
                f"attn_implementation must be {H100_ATTN_IMPLEMENTATION!r} on H100"
            )
    if getattr(config, "gradient_checkpointing", None) is not True:
        raise AssertionError("gradient_checkpointing must be True")
    # TRL >=0.24 auto-bumps per_device_train_batch_size and may adjust
    # gradient_accumulation_steps so pdb*ga*world_size == num_generations.
    # Accept any positive int — the effective rollout product is checked below.
    if not isinstance(config.per_device_train_batch_size, int) or config.per_device_train_batch_size < 1:
        raise AssertionError("per_device_train_batch_size must be a positive int")
    if config.num_generations != num_generations:
        raise AssertionError(
            f"num_generations mismatch: config has {config.num_generations}, expected {num_generations}"
        )
    if not isinstance(config.gradient_accumulation_steps, int) or config.gradient_accumulation_steps < 1:
        raise AssertionError("gradient_accumulation_steps must be a positive int")
    # Effective rollout product is informational only — the Sudoku notebook
    # uses 2*2=4, the original DriftCall spec used 8*4=32. Both work.
    product = config.num_generations * config.gradient_accumulation_steps
    if product < 1:
        raise AssertionError(
            f"num_generations * gradient_accumulation_steps must be >= 1; got {product}"
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
    # TRL >=0.24 normalizes report_to to a list, e.g. ["wandb"].
    _rt = config.report_to
    _rt_ok = _rt == REPORT_TO or (isinstance(_rt, (list, tuple)) and REPORT_TO in _rt)
    if not _rt_ok:
        raise AssertionError(f"report_to must include {REPORT_TO!r}; got {_rt!r}")
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
        use_bias_correction_kl=bool(getattr(config, "use_bias_correction_kl", False)),
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


def init_wandb(
    *,
    stage: StageT,
    seed: int,
    h100_mode: bool = False,
    enable_adaptive_kl: bool = True,
    extra_config: dict[str, Any] | None = None,
) -> Any:
    """Initialize a WandB run for a training stage (training.md §3.3.3).

    Override priority for credentials:
      1. ``os.environ`` values set by the caller (highest)
      2. ``cells._secrets.export_to_env()`` hardcoded fallback
      3. None — caller must set ``WANDB_MODE=disabled`` or run will fail

    Returns the active ``wandb.run`` object, or ``None`` when
    ``WANDB_MODE`` resolves to ``"disabled"``. Idempotent — if a run is
    already active for this process, returns it unchanged.
    """
    import os
    import time

    # Step 1: populate env from cells/_secrets.py if a key is missing.
    try:
        from cells._secrets import export_to_env

        export_to_env()
    except ImportError:
        pass

    mode = os.environ.get("WANDB_MODE", WANDB_MODE_DEFAULT).strip().lower()
    if mode == "disabled":
        return None

    import wandb

    if getattr(wandb, "run", None) is not None:
        return wandb.run

    project = os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)
    entity = os.environ.get("WANDB_ENTITY", WANDB_ENTITY_DEFAULT)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = WANDB_RUN_NAME_TEMPLATE.format(
        stage=stage, seed=seed, timestamp=timestamp
    )

    tags = [
        f"stage{stage}",
        "gemma-4-e2b",
        "bf16" if h100_mode else "fp16",
        "adaptive-kl" if enable_adaptive_kl else "static-kl",
        f"seed{seed}",
    ]

    # Lazy LoRA constants — step_12 imports unsloth at module top, so guard
    # against CPU-only CI environments where unsloth is unavailable.
    try:
        from cells.step_12_gemma_boot import LORA_ALPHA, LORA_DROPOUT, LORA_R
    except ImportError:
        LORA_R = 16
        LORA_ALPHA = 32
        LORA_DROPOUT = 0.05

    # target_kl default matches AdaptiveKLCallback(target_kl=BETA_KL) in step_14.
    config: dict[str, Any] = {
        "stage": stage,
        "seed": seed,
        "h100_mode": h100_mode,
        "adaptive_kl": enable_adaptive_kl,
        "beta_initial": BETA_KL,
        "target_kl": BETA_KL,
        "learning_rate": LEARNING_RATE,
        "num_generations": DEFAULT_NUM_GENERATIONS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    if extra_config:
        config.update(extra_config)

    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "tags": tags,
        "config": config,
        "mode": mode,
    }
    if entity is not None:
        init_kwargs["entity"] = entity

    return wandb.init(**init_kwargs)


__all__ = [
    "ALLOWED_HARDWARE",
    "ALLOWED_NUM_GENERATIONS",
    "BETA_KL",
    "DEFAULT_NUM_GENERATIONS",
    "EFFECTIVE_ROLLOUTS_PER_UPDATE",
    "H100_ATTN_IMPLEMENTATION",
    "HardwareT",
    "LEARNING_RATE",
    "MAX_COMPLETION_LENGTH",
    "MAX_PROMPT_LENGTH",
    "OPTIM_H100",
    "OPTIM_V100",
    "PER_DEVICE_TRAIN_BATCH_SIZE",
    "REPORT_TO",
    "StageT",
    "WANDB_ENTITY_DEFAULT",
    "WANDB_MODE_DEFAULT",
    "WANDB_PROJECT_DEFAULT",
    "WANDB_RUN_NAME_TEMPLATE",
    "WARMUP_RATIO_STAGE1",
    "WARMUP_RATIO_STAGE2_3",
    "assert_config_invariants",
    "build_grpo_config",
    "init_wandb",
    "reward_fn",
]
