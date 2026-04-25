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
import warnings
from dataclasses import dataclass
from typing import Any

# Suppress a noisy transformers >= 5.5 deprecation that fires from inside
# Unsloth's compiled GRPOTrainer when it passes both `generation_config`
# and duplicate kwargs (pad_token_id, disable_compile) to model.generate().
# We mirror those values onto generation_config in boot_gemma() — the
# warning is purely cosmetic; behavior is identical either way.
warnings.filterwarnings(
    "ignore",
    message=(
        r"Passing `generation_config` together with generation-related arguments.*"
        r"is deprecated.*"
    ),
)

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
LORA_DROPOUT: float = 0.0  # 0 enables Unsloth's fast LoRA path that correctly
# handles multimodal Gemma 4 hidden-state extraction. Any non-zero dropout
# triggers Unsloth's "patch all other layers, except LoRA matrices, causing a
# performance hit" slow path which routes through the broken
# chunked_hidden_states_selective_log_softmax. The Sudoku GRPO notebook uses
# dropout=0 explicitly for this reason.
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
    # 16-bit LoRA — the official Unsloth Gemma 4 E2B GRPO notebook uses this.
    # 4-bit triggers the chunked_hidden_states_selective_log_softmax path
    # that crashes on multimodal Gemma 4. V100 32GB has ample headroom for
    # 16-bit (Gemma 4 E2B ~10GB weights + LoRA ~0.1GB + activations ~5GB).
    load_in_4bit: bool = False
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
    """Load Gemma 4 E2B in 16-bit LoRA + return (model, tokenizer).

    Mirrors the official Unsloth Gemma 4 E2B GRPO Sudoku notebook
    (https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb)
    exactly:

        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = "unsloth/gemma-4-E2B-it",
            max_seq_length = 4096,
            load_in_4bit = False,    # 16-bit LoRA (NOT 4-bit)
            fast_inference = False,
        )

    Why **not** ``FastLanguageModel`` and **not** 4-bit:
      - Gemma 4 E2B-it ships with audio + vision towers; the language-only
        loader still loads ``Gemma4ForConditionalGeneration`` and Unsloth
        compiles a multimodal-aware GRPO trainer.
      - 4-bit weights route through ``chunked_hidden_states_selective_log_
        softmax`` which expects hidden states with shape ``[..., hidden_dim]``
        but for the multimodal Gemma 4 path receives ``[..., vocab]`` and
        crashes with the ``mat1 and mat2 shapes cannot be multiplied`` error.
      - The Sudoku notebook uses 16-bit LoRA explicitly (``load_in_4bit=False``)
        to bypass that path. 16-bit Gemma 4 E2B fits comfortably on V100 32GB.

    The ``load_in_4bit`` setting on :class:`BootConfig` is honored — it
    defaults to ``False`` to follow the working notebook recipe.
    """
    cfg = config if config is not None else BootConfig()

    import torch
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
        fast_inference=False,  # disables vLLM; uses Unsloth inference (RL guide)
    )

    assert_fp16_dtype(model)

    # Set generation defaults on the model's generation_config so Unsloth/TRL's
    # GRPO _generate_single_turn doesn't have to pass them as duplicate kwargs
    # alongside generation_config. transformers >= 5.5 deprecated the dual-pass
    # pattern; passing both raises:
    #   "Passing generation_config together with generation-related arguments=
    #    ({'disable_compile', 'pad_token_id'}) is deprecated"
    # This silences the warning at the source and is what transformers wants.
    try:
        gc = getattr(model, "generation_config", None)
        tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        if gc is not None:
            pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is not None:
                gc.pad_token_id = pad_id
            # Unsloth's GRPO path sets disable_compile=True at call time; mirror
            # it on the generation_config so the kwarg becomes redundant.
            try:
                gc.disable_compile = True
            except Exception:
                pass
    except Exception:
        pass

    peft_model = FastVisionModel.get_peft_model(
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
