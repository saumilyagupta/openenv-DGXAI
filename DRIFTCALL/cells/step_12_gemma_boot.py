"""Gemma 3 4B boot via Unsloth FastLanguageModel (docs/modules/training.md §3.1).

v1 contract — text-only Gemma 3 to ship before deadline:

  - Base model: ``unsloth/gemma-3-4b-it`` (text-only; no audio/vision).
  - Loader: ``FastLanguageModel`` (routes through the well-tested llama.py
    GRPO patches; avoids the unsloth_base_fast_generate vision wrapper that
    breaks GRPO sampling on multimodal Gemma 4 + V100 + transformers 5.5).
  - Precision: explicit FP16 (V100-safe).
  - LoRA: r=16, α=32, 7 target modules, dropout=0 (Unsloth fast LoRA path),
    Unsloth gradient checkpointing, ``random_state=3407``.
  - Hard halt: ``next(model.parameters()).dtype`` MUST be ``torch.float16``;
    any BF16 parameter triggers ``BF16SlippageError`` before optimizer build.

Phase 2 (post-deadline): switch back to ``unsloth/gemma-4-E2B-it`` once
Unsloth ships a Gemma 4 multimodal GRPO fix (see GitHub issue tracker).
Audio capability is held aside until that lands.

Heavy imports (``unsloth``, ``torch``) are deferred inside functions so this
cell loads on CPU-only CI runners where Unsloth is not installed.
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
# Disable Unsloth's "static cache + compile_config" generation path. On
# transformers >= 5.5 with multimodal Gemma 4, the static cache produces
# malformed logit shapes during _sample, manifesting as
# `RuntimeError: prob_dist must be 1 or 2 dim` in torch.multinomial.
# Falls back to the dynamic-cache path which is correct on V100.
os.environ.setdefault("UNSLOTH_DISABLE_STATIC_GENERATION", "1")
# Explicitly start with hidden-states trick OFF; only Unsloth's GRPO
# _get_per_token_logps_and_entropies block sets this to "1" temporarily.
# Defensive reset prevents leaks across re-imports / test runs.
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

BASE_MODEL_ID: str = "unsloth/gemma-3-4b-it"
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


def _patch_gemma4_return_hidden_states() -> None:
    """Make Gemma 4 forward honor ``UNSLOTH_RETURN_HIDDEN_STATES=1``.

    ROOT CAUSE for the chunked-log-softmax shape mismatch:
    Unsloth's GRPOTrainer uses a env-var trick to ask the model to return
    the residual hidden states (shape ``[B, S, hidden_dim=1536]``) instead
    of LM-head logits (``[B, S, vocab=262144]``). The Llama/Mistral forward
    methods are patched upstream (``unsloth/models/llama.py`` L1509-1521)
    to honor this env var. Gemma 4's forward methods are NOT — both
    ``Gemma4ForCausalLM`` and ``Gemma4ForConditionalGeneration`` always run
    ``logits = self.lm_head(hidden_states)``. The trainer then mistakes
    those vocab-shaped logits for hidden states and crashes:

        RuntimeError: mat1 and mat2 shapes cannot be multiplied
                      (BS, 262144) and (1536, 262144)

    Mirror the Llama pattern on both Gemma 4 forward methods. Idempotent
    via ``_DRIFTCALL_RHS_PATCHED`` sentinel. Branch ONLY fires when the
    env var is explicitly "1" (set by GRPO's _get_per_token_logps_and_
    entropies block); inference and standard training pass through the
    original forward unchanged, so audio + vision capability is preserved.
    """
    import os as _os

    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.gemma4 import modeling_gemma4 as _g4
    except Exception:
        return

    _Causal = getattr(_g4, "Gemma4ForCausalLM", None)
    _CondGen = getattr(_g4, "Gemma4ForConditionalGeneration", None)

    def _wrap(target_cls):
        if target_cls is None or getattr(target_cls, "_DRIFTCALL_RHS_PATCHED", False):
            return
        original = target_cls.forward

        def patched(self, *args, **kwargs):
            if _os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
                return original(self, *args, **kwargs)

            inner = getattr(self, "model", None) or getattr(self, "language_model", None)
            if inner is None:
                return original(self, *args, **kwargs)

            backbone_kwargs = {
                k: v for k, v in kwargs.items()
                if k in {
                    "input_ids", "attention_mask", "position_ids",
                    "past_key_values", "inputs_embeds", "use_cache",
                    "output_attentions", "output_hidden_states",
                    "return_dict", "cache_position",
                    # Multimodal kwargs the backbone accepts:
                    "pixel_values", "pixel_values_videos", "input_features",
                    "input_features_mask", "image_position_ids",
                    "video_position_ids", "mm_token_type_ids",
                }
            }
            outputs = inner(*args, **backbone_kwargs)
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                return original(self, *args, **kwargs)

            logits_to_keep = kwargs.get("logits_to_keep", 0)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]

            return CausalLMOutputWithPast(
                loss=None,
                logits=hidden_states,  # ← key: hidden states stuffed into logits
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

        target_cls.forward = patched
        target_cls._DRIFTCALL_RHS_PATCHED = True

    _wrap(_Causal)
    _wrap(_CondGen)

    # Second patch: drop multimodal kwargs that transformers' validator
    # rejects with "model_kwargs are not used by the model" because its
    # introspection doesn't honor Unpack[TransformersKwargs]. The model's
    # forward accepts these via **kwargs but the validator can't see that.
    _MM_DROP = ("mm_token_type_ids", "pixel_values_lengths", "image_sizes")

    def _wrap_validate(target_cls):
        if target_cls is None or getattr(target_cls, "_DRIFTCALL_VAL_PATCHED", False):
            return
        original = target_cls._validate_model_kwargs

        def patched_validate(self, model_kwargs):
            for k in _MM_DROP:
                model_kwargs.pop(k, None)
            return original(self, model_kwargs)

        target_cls._validate_model_kwargs = patched_validate
        target_cls._DRIFTCALL_VAL_PATCHED = True

    _wrap_validate(_Causal)
    _wrap_validate(_CondGen)


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
    from unsloth import FastLanguageModel

    # FastLanguageModel returns (model, tokenizer) directly — no processor
    # unwrapping needed (Gemma 3 is text-only).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
        fast_inference=False,
    )

    assert_fp16_dtype(model)

    # Set generation defaults on the model's generation_config so Unsloth/TRL's
    # GRPO doesn't pass duplicate kwargs alongside generation_config (silences
    # the transformers >= 5.5 deprecation warning).
    try:
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None:
                gc.pad_token_id = pad_id
    except Exception:
        pass

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
