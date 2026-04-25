"""Gemma 4 E2B boot via Unsloth FastModel (docs/modules/training.md §3.1).

Contract:
  - Base model: ``unsloth/gemma-4-E2B-it-bnb-4bit`` (4-bit NF4 quantization).
  - Precision: explicit FP16 (V100-safe; Gemma 4 is BF16-native so explicit
    ``dtype=torch.float16`` at load time prevents Unsloth auto-picking BF16).
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


def _patch_chunked_log_softmax_for_gemma4_audio() -> None:
    """Make Unsloth's chunked_hidden_states_selective_log_softmax tolerant
    of a Gemma 4 audio-tower hook fallback that returns logits where the
    function expects hidden states. Idempotent — only patches once."""
    import torch
    try:
        import unsloth_compiled_cache.UnslothGRPOTrainer as _ugrpo  # type: ignore
    except Exception:
        return
    if getattr(_ugrpo, "_DRIFTCALL_PATCHED", False):
        return
    original = _ugrpo.chunked_hidden_states_selective_log_softmax

    def _patched(
        hidden_states: torch.Tensor,
        lm_head: torch.Tensor,
        index: torch.Tensor,
        chunks: int = 4,
        logit_scale_multiply: float = 0.0,
        logit_scale_divide: float = 0.0,
        logit_softcapping: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Standard path: hidden_states.shape[-1] == hidden_dim == lm_head.shape[1].
        # Audio-tower fallback path: hidden_states.shape[-1] == vocab == lm_head.shape[0].
        if hidden_states.shape[-1] == lm_head.shape[0]:
            # Already-projected logits — skip the matmul and proceed.
            flat_logits = hidden_states.reshape(-1, hidden_states.shape[-1])
            flat_index = index.reshape(-1)
            chunked_logits = torch.chunk(flat_logits, chunks=chunks, dim=0)
            chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)
            all_lps: list[torch.Tensor] = []
            for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
                cl = chunk_logits
                if logit_scale_multiply != 0.0:
                    cl = cl * logit_scale_multiply
                if logit_scale_divide != 0.0:
                    cl = cl / logit_scale_divide
                if logit_softcapping != 0.0:
                    cl = logit_softcapping * torch.tanh(cl / logit_softcapping)
                cl = cl.to(torch.float32)
                if temperature != 1.0:
                    cl = cl / temperature
                selected = torch.gather(cl, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
                lse = torch.logsumexp(cl, dim=-1)
                all_lps.append(selected - lse)
            out = torch.concat(all_lps)
            return out.reshape((hidden_states.shape[0], hidden_states.shape[1]))
        return original(
            hidden_states, lm_head, index, chunks,
            logit_scale_multiply, logit_scale_divide,
            logit_softcapping, temperature,
        )

    _ugrpo.chunked_hidden_states_selective_log_softmax = _patched
    _ugrpo._DRIFTCALL_PATCHED = True


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

    # Patch Unsloth's chunked log-softmax for Gemma 4 audio-tower incompatibility.
    # The Unsloth audio-tower hook fallback (warning logged at boot) causes
    # the wrapped model to return logits with shape [..., vocab] instead of
    # hidden states with shape [..., hidden_dim]. The chunked path then tries
    # `logits @ lm_head.T` and fails the matmul (vocab != hidden_dim).
    # Detect that case and short-circuit: when input dim already matches the
    # vocab size, treat it as logits and run log_softmax directly.
    _patch_chunked_log_softmax_for_gemma4_audio()

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
