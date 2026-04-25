"""Gemma 3n E2B boot via Unsloth FastModel (docs/modules/training.md §3.1).

Contract:
  - Base model: ``unsloth/gemma-3n-E2B-it`` (4-bit Dynamic
    NF4 quantization).
  - Precision: hardware-aware.
      V100 (sm_70) — explicit FP16 (``dtype=torch.float16``); Gemma 3n is
      BF16-native, so we force FP16 on V100 to avoid BF16 software-emulation
      slowdown / numerical instability.
      H100 (sm_90) — BF16 (``dtype=torch.bfloat16``); uses native tensor cores.
  - LoRA: r=16, α=32, dropout=0.05, vision towers frozen, language + attention
    + MLP trainable via Unsloth's multimodal API (``finetune_vision_layers=False,
    finetune_language_layers=True, finetune_attention_modules=True,
    finetune_mlp_modules=True``), Unsloth gradient checkpointing,
    ``random_state=3407``.
  - V100 halt: ``next(model.parameters()).dtype`` MUST be ``torch.float16``
    after FP16 load; any BF16 parameter triggers :class:`BF16SlippageError`
    before optimizer build.
  - H100 halt: ``next(model.parameters()).dtype`` MUST be ``torch.bfloat16``
    after BF16 load; any FP16 parameter triggers :class:`FP16SlippageError`
    before optimizer build.

Heavy imports (``unsloth``, ``torch``) are deferred inside functions so this
cell loads on CPU-only CI runners where Unsloth is not installed. Tests mock
``FastModel.from_pretrained`` and ``FastModel.get_peft_model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BASE_MODEL_ID: str = "unsloth/gemma-3n-E2B-it"
MAX_SEQ_LENGTH: int = 4096
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_RANDOM_STATE: int = 3407

# Gemma 3n multimodal LoRA flags — vision/audio towers stay frozen so GRPO
# trains only the language stack (Unsloth Gemma 3N notebook §fine-tune).
FINETUNE_VISION_LAYERS: bool = False
FINETUNE_LANGUAGE_LAYERS: bool = True
FINETUNE_ATTENTION_MODULES: bool = True
FINETUNE_MLP_MODULES: bool = True

HardwareT = Literal["v100", "h100"]
ALLOWED_HARDWARE: tuple[HardwareT, ...] = ("v100", "h100")


class BF16SlippageError(AssertionError):
    """Raised when the loaded model has any BF16 parameter on V100.

    V100 (sm_70) lacks BF16 tensor cores. Silent BF16 via software emulation
    causes ~10x slowdown plus numerical-instability patterns in
    ``docs/modules/training.md §7a``. Halt before the optimizer is built.
    """


class FP16SlippageError(AssertionError):
    """Raised when the loaded model has any FP16 parameter on H100.

    H100 (sm_90) has native BF16 tensor cores. Running FP16 on H100 means
    leaving native hardware capability unused and may cause gradient underflow
    at large batch sizes. Halt before the optimizer is built.
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
    lora_random_state: int = LORA_RANDOM_STATE
    finetune_vision_layers: bool = FINETUNE_VISION_LAYERS
    finetune_language_layers: bool = FINETUNE_LANGUAGE_LAYERS
    finetune_attention_modules: bool = FINETUNE_ATTENTION_MODULES
    finetune_mlp_modules: bool = FINETUNE_MLP_MODULES
    use_gradient_checkpointing: str = "unsloth"
    hardware: HardwareT = "v100"


def assert_dtype_for_hardware(model: Any, hardware: HardwareT) -> None:
    """Assert the first parameter dtype matches the expected precision for hardware.

    V100 must be ``torch.float16``; raises :class:`BF16SlippageError` otherwise.
    H100 must be ``torch.bfloat16``; raises :class:`FP16SlippageError` otherwise.
    Called once at ``boot_gemma`` entry, before any LoRA attach or optimizer build.
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
    if hardware == "v100":
        if dtype != torch.float16:
            raise BF16SlippageError(
                f"BF16 slipped through: V100 unsafe. "
                f"next(model.parameters()).dtype == {dtype}, expected torch.float16. "
                f"Root cause: Unsloth auto-picked BF16 despite dtype=torch.float16 kwarg. "
                f"Halt training; do NOT proceed on V100."
            )
    else:  # h100
        if dtype != torch.bfloat16:
            raise FP16SlippageError(
                f"FP16 slipped through: H100 should use BF16. "
                f"next(model.parameters()).dtype == {dtype}, expected torch.bfloat16. "
                f"Root cause: dtype kwarg may have forced FP16 on H100. "
                f"Halt training; do NOT proceed on H100 with FP16."
            )


def assert_fp16_dtype(model: Any) -> None:
    """Assert the first trainable parameter is torch.float16 (V100 safety).

    Thin wrapper around :func:`assert_dtype_for_hardware` for backwards
    compatibility with call sites that predate the hardware-aware API.
    Raises :class:`BF16SlippageError` with the halt message from
    ``docs/modules/training.md §3.1``.
    """
    assert_dtype_for_hardware(model, "v100")


def boot_gemma(config: BootConfig | None = None) -> tuple[Any, Any]:
    """Load Gemma 3n E2B in 4-bit + attach LoRA; return (model, tokenizer).

    Steps (training.md §3.1):
      1. ``FastModel.from_pretrained(base_model_id, max_seq_length=...,
         load_in_4bit=True, dtype=torch.float16)`` on V100
         or ``dtype=torch.bfloat16`` on H100.
      2. ``assert_dtype_for_hardware(model, hardware)`` — raises
         :class:`BF16SlippageError` or :class:`FP16SlippageError` if the dtype
         does not match the hardware.
      3. ``FastModel.get_peft_model(model, r=16, lora_alpha=32,
         finetune_vision_layers=False, finetune_language_layers=True,
         finetune_attention_modules=True, finetune_mlp_modules=True,
         use_gradient_checkpointing="unsloth", random_state=3407)``.
      4. Return ``(peft_model, tokenizer)``.

    All heavy imports are lazy so the module is importable on CPU-only CI.
    """
    cfg = config if config is not None else BootConfig()

    import torch
    from unsloth import FastModel

    dtype = torch.float16 if cfg.hardware == "v100" else torch.bfloat16

    model, tokenizer = FastModel.from_pretrained(
        cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=dtype,
    )

    assert_dtype_for_hardware(model, cfg.hardware)

    peft_model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.lora_random_state,
    )

    return peft_model, tokenizer


__all__ = [
    "ALLOWED_HARDWARE",
    "BASE_MODEL_ID",
    "BF16SlippageError",
    "BootConfig",
    "FINETUNE_ATTENTION_MODULES",
    "FINETUNE_LANGUAGE_LAYERS",
    "FINETUNE_MLP_MODULES",
    "FINETUNE_VISION_LAYERS",
    "FP16SlippageError",
    "HardwareT",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "LORA_RANDOM_STATE",
    "MAX_SEQ_LENGTH",
    "assert_dtype_for_hardware",
    "assert_fp16_dtype",
    "boot_gemma",
]
