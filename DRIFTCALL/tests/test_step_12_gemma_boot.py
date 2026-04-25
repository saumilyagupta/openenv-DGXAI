"""Tests for cells/step_12_gemma_boot.py.

Mocks ``unsloth.FastModel`` so the suite runs CPU-only with no heavy
weights downloaded. Covers:

- BF16-slippage halt (training_tests.md U30)
- FP16 model passes assertion (U31)
- LoRA attach with correct kwargs (training.md §3.1)
- Model/tokenizer returned as a tuple
- Invariants: r=16, alpha=32, 7 target modules, random_state=3407
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from cells.step_12_gemma_boot import (
    BASE_MODEL_ID,
    LORA_ALPHA,
    LORA_R,
    LORA_RANDOM_STATE,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    BF16SlippageError,
    BootConfig,
    assert_fp16_dtype,
    boot_gemma,
)


def _fake_model_with_dtype(dtype: torch.dtype) -> MagicMock:
    """Build a fake model whose first parameter has the given dtype."""
    param = torch.zeros(1, dtype=dtype)
    model = MagicMock()
    model.parameters = MagicMock(return_value=iter([param]))
    return model


def _install_fake_unsloth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_dtype: torch.dtype = torch.float16,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Inject a stub ``unsloth`` module and return (FastModel, model, peft)."""
    fake_tokenizer = MagicMock(name="tokenizer")
    peft_model = MagicMock(name="peft_model")

    base_model = _fake_model_with_dtype(base_dtype)

    FastModel = MagicMock(name="FastModel")
    FastModel.from_pretrained = MagicMock(return_value=(base_model, fake_tokenizer))
    FastModel.get_peft_model = MagicMock(return_value=peft_model)

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = FastModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return FastModel, base_model, peft_model


class TestConstants:
    def test_base_model_id_is_pinned(self) -> None:
        assert BASE_MODEL_ID == "unsloth/gemma-3-4b-it"

    def test_max_seq_length_4096(self) -> None:
        assert MAX_SEQ_LENGTH == 4096

    def test_lora_r_16_alpha_32(self) -> None:
        assert LORA_R == 16
        assert LORA_ALPHA == 32

    def test_lora_random_state_3407(self) -> None:
        assert LORA_RANDOM_STATE == 3407

    def test_lora_target_modules_exactly_seven(self) -> None:
        assert LORA_TARGET_MODULES == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    def test_boot_config_defaults_match_constants(self) -> None:
        cfg = BootConfig()
        assert cfg.base_model_id == BASE_MODEL_ID
        assert cfg.max_seq_length == MAX_SEQ_LENGTH
        # 16-bit LoRA per Unsloth Gemma 4 E2B GRPO Sudoku notebook.
        assert cfg.load_in_4bit is False
        assert cfg.lora_r == LORA_R
        assert cfg.lora_alpha == LORA_ALPHA
        assert cfg.lora_target_modules == LORA_TARGET_MODULES
        assert cfg.lora_random_state == LORA_RANDOM_STATE
        assert cfg.use_gradient_checkpointing == "unsloth"

    def test_boot_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = BootConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.lora_r = 99  # type: ignore[misc]


class TestAssertFp16Dtype:
    def test_fp16_model_passes(self) -> None:
        model = _fake_model_with_dtype(torch.float16)
        assert_fp16_dtype(model)  # no raise

    def test_bf16_model_raises(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        with pytest.raises(BF16SlippageError) as exc:
            assert_fp16_dtype(model)
        assert "BF16 slipped through" in str(exc.value)
        assert "V100 unsafe" in str(exc.value)

    def test_fp32_model_also_raises(self) -> None:
        model = _fake_model_with_dtype(torch.float32)
        with pytest.raises(BF16SlippageError):
            assert_fp16_dtype(model)

    def test_bf16_slippage_is_assertion_error_subclass(self) -> None:
        assert issubclass(BF16SlippageError, AssertionError)


class TestBootGemma:
    def test_returns_peft_model_and_tokenizer_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, peft = _install_fake_unsloth(monkeypatch)
        fake_tokenizer = FastModel.from_pretrained.return_value[1]

        result = boot_gemma()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is peft
        assert result[1] is fake_tokenizer

    def test_from_pretrained_called_with_pinned_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma()

        call = FastModel.from_pretrained.call_args
        # FastVisionModel uses model_name= kwarg per Unsloth Gemma 4 E2B
        # GRPO Sudoku notebook (the official RL recipe).
        assert call.kwargs["model_name"] == BASE_MODEL_ID
        assert call.kwargs["max_seq_length"] == MAX_SEQ_LENGTH
        # 16-bit LoRA — 4-bit triggers chunked-log-softmax shape bug on
        # multimodal Gemma 4.
        assert call.kwargs["load_in_4bit"] is False
        assert call.kwargs["dtype"] is torch.float16
        # fast_inference=False required for GRPO (Unsloth RL guide).
        assert call.kwargs.get("fast_inference") is False

    def test_get_peft_model_called_with_lora_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma()

        call = FastModel.get_peft_model.call_args
        assert call.args == (base,)
        assert call.kwargs["r"] == LORA_R
        assert call.kwargs["lora_alpha"] == LORA_ALPHA
        assert call.kwargs["target_modules"] == list(LORA_TARGET_MODULES)
        assert call.kwargs["use_gradient_checkpointing"] == "unsloth"
        assert call.kwargs["random_state"] == LORA_RANDOM_STATE

    def test_bf16_base_halts_before_peft_attach(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(
            monkeypatch, base_dtype=torch.bfloat16
        )

        with pytest.raises(BF16SlippageError):
            boot_gemma()

        # get_peft_model must NOT be called when BF16 slippage is detected.
        assert FastModel.get_peft_model.call_count == 0

    def test_custom_config_overrides_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        cfg = BootConfig(
            base_model_id="custom/model",
            max_seq_length=2048,
            load_in_4bit=False,
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=("q_proj", "v_proj"),
            lora_random_state=1234,
            use_gradient_checkpointing="false",
        )
        boot_gemma(cfg)

        call_from = FastModel.from_pretrained.call_args
        assert call_from.kwargs["model_name"] == "custom/model"
        assert call_from.kwargs["max_seq_length"] == 2048
        assert call_from.kwargs["load_in_4bit"] is False

        call_peft = FastModel.get_peft_model.call_args
        assert call_peft.kwargs["r"] == 8
        assert call_peft.kwargs["lora_alpha"] == 16
        assert call_peft.kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert call_peft.kwargs["random_state"] == 1234
        assert call_peft.kwargs["use_gradient_checkpointing"] == "false"

    def test_assert_fp16_halt_message_includes_expected_dtype(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        with pytest.raises(BF16SlippageError) as exc:
            assert_fp16_dtype(model)
        msg = str(exc.value)
        assert "torch.bfloat16" in msg
        assert "torch.float16" in msg
        assert "Halt training" in msg

    def test_boot_gemma_uses_none_config_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing ``config=None`` picks the default BootConfig."""
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma(None)

        call_from = FastModel.from_pretrained.call_args
        assert call_from.kwargs["model_name"] == BASE_MODEL_ID

    def test_module_importable_without_unsloth(self) -> None:
        """Reimport the module without unsloth in sys.modules — must not raise."""
        import importlib

        # simulate: importing the cell on a CPU-only runner
        if "unsloth" in sys.modules:
            del sys.modules["unsloth"]
        mod: Any = importlib.import_module("cells.step_12_gemma_boot")
        assert mod.BASE_MODEL_ID == BASE_MODEL_ID


class TestAssertFp16DtypeEmptyModel:
    def test_empty_parameters_raises(self) -> None:
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([]))
        with pytest.raises(BF16SlippageError):
            assert_fp16_dtype(model)
