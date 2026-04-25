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

BASE_MODEL_ID: str = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
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


def _patch_gemma4_for_unsloth_return_hidden_states() -> None:
    """Make Gemma 4 forward methods honor ``UNSLOTH_RETURN_HIDDEN_STATES=1``.

    Root cause of the chunked log-softmax shape mismatch we hit before:
    ``unsloth/models/llama.py`` line 1509-1521 patches the LM-head forward to
    short-circuit and return ``hidden_states`` (stuffed into the ``logits``
    field of ``CausalLMOutputWithPast``) whenever the env var is set. This
    is what TRL/Unsloth's GRPOTrainer relies on to compute per-token logps.

    Gemma 4 (incl. the multimodal E2B/E4B variants) doesn't get this patch
    upstream, so the model returns ACTUAL logits in ``logits``. Then the
    chunked log-softmax tries ``logits @ lm_head.T`` and the matmul fails:

        RuntimeError: mat1 and mat2 shapes cannot be multiplied
                      (BS, vocab=262144) and (hidden=1536, vocab=262144).

    Apply the same return-hidden-states branch to both
    ``Gemma4ForCausalLM.forward`` (text-only variant) and
    ``Gemma4ForConditionalGeneration.forward`` (multimodal — used by the
    ``-it`` 4-bit checkpoint with audio + vision towers). Idempotent.
    """
    import os as _os
    import torch  # noqa: F401  (kept for type clarity)

    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.gemma4 import modeling_gemma4 as _g4
    except Exception:
        return

    _Causal = getattr(_g4, "Gemma4ForCausalLM", None)
    _CondGen = getattr(_g4, "Gemma4ForConditionalGeneration", None)

    def _wrap_forward(target_cls, output_cls):
        if target_cls is None or getattr(target_cls, "_DRIFTCALL_RHS_PATCHED", False):
            return
        original = target_cls.forward

        def patched(self, *args, **kwargs):
            if _os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
                return original(self, *args, **kwargs)

            # Run the inner model body to get hidden states without the
            # lm_head projection. Different Gemma 4 classes expose the
            # backbone under different attribute names.
            inner = getattr(self, "model", None) or getattr(self, "language_model", None)
            if inner is None:
                # Unknown layout — fall back to the original forward (will
                # produce logits, but at least won't crash).
                return original(self, *args, **kwargs)

            # Strip kwargs the backbone doesn't accept.
            backbone_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {
                    "input_ids", "attention_mask", "position_ids",
                    "past_key_values", "inputs_embeds", "use_cache",
                    "output_attentions", "output_hidden_states",
                    "return_dict", "cache_position", "labels",
                    "logits_to_keep",
                }
            }
            outputs = inner(*args, **backbone_kwargs)
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                # Unknown output type — fall back to original.
                return original(self, *args, **kwargs)

            logits_to_keep = kwargs.get("logits_to_keep", 0)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]

            return output_cls(
                loss=None,
                logits=hidden_states,  # ← key trick: hidden states in logits
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

        target_cls.forward = patched
        target_cls._DRIFTCALL_RHS_PATCHED = True

    _wrap_forward(_Causal, CausalLMOutputWithPast)
    # Multimodal output type may differ; reuse CausalLMOutputWithPast which
    # GRPOTrainer reads via ``.logits``. Fields not set are ignored downstream.
    _wrap_forward(_CondGen, CausalLMOutputWithPast)

    # Second root-cause patch: Gemma 4's processor emits multimodal-only
    # kwargs (mm_token_type_ids, pixel_values_lengths, image_sizes) that
    # generate()'s _validate_model_kwargs rejects with a ValueError because
    # they aren't in the forward() signature. For text-only training those
    # kwargs are either all-zero or absent semantically — drop them before
    # validation. Audio/vision codepaths still emit input_features /
    # pixel_values which ARE in the forward signature and pass through.
    _MM_KWARGS_TO_DROP = (
        "mm_token_type_ids",
        "pixel_values_lengths",
        "image_sizes",
    )

    def _wrap_validate(target_cls):
        if target_cls is None or getattr(target_cls, "_DRIFTCALL_VAL_PATCHED", False):
            return
        original = target_cls._validate_model_kwargs

        def patched_validate(self, model_kwargs):
            for k in _MM_KWARGS_TO_DROP:
                model_kwargs.pop(k, None)
            return original(self, model_kwargs)

        target_cls._validate_model_kwargs = patched_validate
        target_cls._DRIFTCALL_VAL_PATCHED = True

    _wrap_validate(_Causal)
    _wrap_validate(_CondGen)


def boot_gemma(config: BootConfig | None = None) -> tuple[Any, Any]:
    """Load Gemma 4 E2B in 4-bit + attach LoRA; return (model, tokenizer).

    Steps (training.md §3.1):
      1. ``FastModel.from_pretrained(base_model_id, max_seq_length=...,
         load_in_4bit=True, dtype=torch.float16)``.
      2. ``assert_fp16_dtype(model)`` — raises :class:`BF16SlippageError`
         if any BF16 slipped through.
      3. ``FastModel.get_peft_model(model, r=16, lora_alpha=32,
         target_modules=(...), use_gradient_checkpointing="unsloth",
         random_state=3407)``.
      4. Return ``(peft_model, tokenizer)``.

    All heavy imports are lazy so the module is importable on CPU-only CI.
    """
    cfg = config if config is not None else BootConfig()

    import torch
    # Programmatic dynamo disable in addition to the env vars set at module
    # import. Required for V100 to avoid tracing failures inside Unsloth's
    # chunked log-softmax path.
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.disable = True
    except Exception:
        pass
    from unsloth import FastModel

    # ROOT CAUSE FIX: Gemma 4 forward methods don't honor
    # UNSLOTH_RETURN_HIDDEN_STATES=1 (the env var Unsloth/TRL's GRPOTrainer
    # uses to ask for hidden states without an lm_head projection). Without
    # this patch, GRPO's chunked log-softmax tries to do `logits @ lm_head.T`
    # and crashes with a matmul shape mismatch.
    _patch_gemma4_for_unsloth_return_hidden_states()

    model, tokenizer = FastModel.from_pretrained(
        cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
    )

    assert_fp16_dtype(model)

    peft_model = FastModel.get_peft_model(
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
