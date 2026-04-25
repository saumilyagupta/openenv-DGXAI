"""Gemma 4 E2B boot via Unsloth FastModel (docs/modules/training.md §3.1).

Contract:
  - Base model: ``unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`` (4-bit NF4).
  - Precision: explicit FP16 (V100-safe; explicit ``dtype=torch.float16``
    at load prevents Unsloth auto-picking BF16).
  - LoRA: r=16, α=32, 7 target modules (q/k/v/o + gate/up/down), Unsloth
    gradient checkpointing, ``random_state=3407``.
  - Hard halt: ``next(model.parameters()).dtype`` MUST be ``torch.float16``;
    any BF16 parameter triggers ``BF16SlippageError`` before optimizer build.

Heavy imports (``unsloth``, ``torch``) are deferred inside functions so this
cell loads on CPU-only CI runners where Unsloth is not installed. Tests mock
``FastModel.from_pretrained`` and ``FastModel.get_peft_model``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# V100 + recent-torch + Unsloth chunked log-softmax has a known dynamo
# shape-tracing bug. Disable torch.compile/dynamo before any Unsloth import.
# Both env-var and programmatic disable are belt-and-suspenders.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

BASE_MODEL_ID: str = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH: int = 4096
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_RANDOM_STATE: int = 3407
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class BF16SlippageError(AssertionError):
    """Raised at boot entry when the loaded model has any BF16 parameter.

    V100 (sm_70) lacks BF16 tensor cores. Silent BF16 via software emulation
    causes ~10x slowdown plus the numerical-instability patterns in
    ``docs/modules/training.md §7a``. Halt before the optimizer is built.
    """


@dataclass(frozen=True)
class BootConfig:
    """Arguments to :func:`boot_gemma`. Frozen per DriftCall immutability rule."""

    base_model_id: str = BASE_MODEL_ID
    max_seq_length: int = MAX_SEQ_LENGTH
    load_in_4bit: bool = True
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    lora_random_state: int = LORA_RANDOM_STATE
    use_gradient_checkpointing: str = "unsloth"


def assert_fp16_dtype(model: Any) -> None:
    """Assert the first trainable parameter is torch.float16 (V100 safety).

    Raises :class:`BF16SlippageError` with the halt message from
    ``docs/modules/training.md §3.1``. Called once at ``boot_gemma`` entry,
    before any LoRA attach or optimizer build.
    """
    import torch

    params_iter = model.parameters()
    try:
        first_param = next(params_iter)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise BF16SlippageError(
            "Model has no parameters; cannot verify dtype."
        ) from exc

    dtype = first_param.dtype
    if dtype != torch.float16:
        raise BF16SlippageError(
            f"BF16 slipped through: V100 unsafe. "
            f"next(model.parameters()).dtype == {dtype}, expected torch.float16. "
            f"Root cause: Unsloth auto-picked BF16 despite dtype=torch.float16 kwarg. "
            f"Halt training; do NOT proceed on V100."
        )


def boot_gemma(config: BootConfig | None = None) -> tuple[Any, Any]:
    """Load Gemma 4 E2B in 4-bit + attach LoRA; return (model, tokenizer).

    Per the official Unsloth Gemma 4 GRPO guide
    (https://unsloth.ai/docs/models/gemma-4/train#reinforcement-learning-rl):

        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-4-E2B-it",
            fast_inference=False,
        )

    ``FastLanguageModel`` (the language-only loader) routes through
    ``unsloth/models/llama.py`` which has all the GRPO-specific patches
    baked in (UNSLOTH_RETURN_HIDDEN_STATES honoring at L1509, KV-shared-no-
    cache fix, etc.). ``FastModel`` (the multimodal loader) routes through
    ``unsloth/models/vision.py`` which lacks those patches and crashes
    during the GRPO log-softmax / multinomial-sampling steps.

    The audio + vision towers are still loaded as part of the checkpoint —
    they're just not exercised during text-only RL training. Inference time
    on the deployed env can use the audio capability separately via the
    standard processor path.

    Steps:
      1. ``FastLanguageModel.from_pretrained(model_name=..., load_in_4bit=True,
         dtype=torch.float16, fast_inference=False)``.
      2. ``assert_fp16_dtype(model)`` — raises :class:`BF16SlippageError`
         if any BF16 slipped through.
      3. ``FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
         target_modules=(...), use_gradient_checkpointing="unsloth",
         random_state=3407)``.
      4. Return ``(peft_model, tokenizer)``.

    All heavy imports are lazy so the module is importable on CPU-only CI.
    """
    cfg = config if config is not None else BootConfig()

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
        fast_inference=False,  # disables vLLM; uses Unsloth inference (RL guide)
    )

    assert_fp16_dtype(model)

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.lora_random_state,
    )

    return peft_model, tokenizer


__all__ = [
    "BASE_MODEL_ID",
    "BF16SlippageError",
    "BootConfig",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "LORA_RANDOM_STATE",
    "LORA_TARGET_MODULES",
    "MAX_SEQ_LENGTH",
    "assert_fp16_dtype",
    "boot_gemma",
]
