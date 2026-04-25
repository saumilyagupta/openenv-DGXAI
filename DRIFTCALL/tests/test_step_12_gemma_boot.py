"""Tests for cells/step_12_gemma_boot.py.

Mocks ``unsloth.FastModel`` so the suite runs CPU-only with no heavy
weights downloaded. Covers:

- BF16-slippage halt on V100 (training_tests.md U30)
- FP16-slippage halt on H100
- FP16 model passes V100 assertion (U31)
- BF16 model passes H100 assertion
- LoRA attach with correct Gemma 3n flags (finetune_vision_layers=False, etc.)
- Model/tokenizer returned as a tuple
- Invariants: r=16, alpha=32, random_state=3407
- Hardware-aware dtype selection
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
    FINETUNE_ATTENTION_MODULES,
    FINETUNE_LANGUAGE_LAYERS,
    FINETUNE_MLP_MODULES,
    FINETUNE_VISION_LAYERS,
    LORA_ALPHA,
    LORA_R,
    LORA_RANDOM_STATE,
    MAX_SEQ_LENGTH,
    BF16SlippageError,
    BootConfig,
    FP16SlippageError,
    assert_dtype_for_hardware,
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
    unsloth_mod.FastModel = FastModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return FastModel, base_model, peft_model


class TestConstants:
    def test_base_model_id_is_pinned(self) -> None:
        assert BASE_MODEL_ID == "unsloth/gemma-3n-E2B-it"

    def test_max_seq_length_4096(self) -> None:
        assert MAX_SEQ_LENGTH == 4096

    def test_lora_r_16_alpha_32(self) -> None:
        assert LORA_R == 16
        assert LORA_ALPHA == 32

    def test_lora_random_state_3407(self) -> None:
        assert LORA_RANDOM_STATE == 3407

    def test_finetune_flags(self) -> None:
        assert FINETUNE_VISION_LAYERS is False
        assert FINETUNE_LANGUAGE_LAYERS is True
        assert FINETUNE_ATTENTION_MODULES is True
        assert FINETUNE_MLP_MODULES is True

    def test_boot_config_defaults_match_constants(self) -> None:
        cfg = BootConfig()
        assert cfg.base_model_id == BASE_MODEL_ID
        assert cfg.max_seq_length == MAX_SEQ_LENGTH
        assert cfg.load_in_4bit is True
        assert cfg.lora_r == LORA_R
        assert cfg.lora_alpha == LORA_ALPHA
        assert cfg.lora_random_state == LORA_RANDOM_STATE
        assert cfg.use_gradient_checkpointing == "unsloth"
        assert cfg.hardware == "v100"
        assert cfg.finetune_vision_layers is False
        assert cfg.finetune_language_layers is True
        assert cfg.finetune_attention_modules is True
        assert cfg.finetune_mlp_modules is True

    def test_boot_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = BootConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.lora_r = 99  # type: ignore[misc]


class TestAssertDtypeForHardware:
    def test_fp16_model_passes_v100(self) -> None:
        model = _fake_model_with_dtype(torch.float16)
        assert_dtype_for_hardware(model, "v100")  # no raise

    def test_bf16_model_raises_on_v100(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        with pytest.raises(BF16SlippageError) as exc:
            assert_dtype_for_hardware(model, "v100")
        assert "BF16 slipped through" in str(exc.value)
        assert "V100 unsafe" in str(exc.value)

    def test_fp32_model_raises_on_v100(self) -> None:
        model = _fake_model_with_dtype(torch.float32)
        with pytest.raises(BF16SlippageError):
            assert_dtype_for_hardware(model, "v100")

    def test_bf16_model_passes_h100(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        assert_dtype_for_hardware(model, "h100")  # no raise

    def test_fp16_model_raises_on_h100(self) -> None:
        model = _fake_model_with_dtype(torch.float16)
        with pytest.raises(FP16SlippageError) as exc:
            assert_dtype_for_hardware(model, "h100")
        assert "FP16 slipped through" in str(exc.value)
        assert "H100" in str(exc.value)

    def test_bf16_slippage_is_assertion_error_subclass(self) -> None:
        assert issubclass(BF16SlippageError, AssertionError)

    def test_fp16_slippage_is_assertion_error_subclass(self) -> None:
        assert issubclass(FP16SlippageError, AssertionError)


class TestAssertFp16DtypeBackcompat:
    """assert_fp16_dtype is kept for backwards compatibility; delegates to v100 path."""

    def test_fp16_model_passes(self) -> None:
        model = _fake_model_with_dtype(torch.float16)
        assert_fp16_dtype(model)  # no raise

    def test_bf16_model_raises(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        with pytest.raises(BF16SlippageError) as exc:
            assert_fp16_dtype(model)
        assert "BF16 slipped through" in str(exc.value)
        assert "V100 unsafe" in str(exc.value)


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

    def test_from_pretrained_called_with_pinned_kwargs_v100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma()

        call = FastModel.from_pretrained.call_args
        assert call.args == (BASE_MODEL_ID,)
        assert call.kwargs["max_seq_length"] == MAX_SEQ_LENGTH
        assert call.kwargs["load_in_4bit"] is True
        assert call.kwargs["dtype"] is torch.float16

    def test_from_pretrained_uses_bf16_on_h100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(
            monkeypatch, base_dtype=torch.bfloat16
        )
        cfg = BootConfig(hardware="h100")
        boot_gemma(cfg)

        call = FastModel.from_pretrained.call_args
        assert call.kwargs["dtype"] is torch.bfloat16

    def test_get_peft_model_called_with_gemma3n_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma()

        call = FastModel.get_peft_model.call_args
        assert call.args == (base,)
        assert call.kwargs["r"] == LORA_R
        assert call.kwargs["lora_alpha"] == LORA_ALPHA
        assert call.kwargs["finetune_vision_layers"] is False
        assert call.kwargs["finetune_language_layers"] is True
        assert call.kwargs["finetune_attention_modules"] is True
        assert call.kwargs["finetune_mlp_modules"] is True
        assert call.kwargs["use_gradient_checkpointing"] == "unsloth"
        assert call.kwargs["random_state"] == LORA_RANDOM_STATE

    def test_no_target_modules_kwarg_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma()

        call = FastModel.get_peft_model.call_args
        assert "target_modules" not in call.kwargs

    def test_bf16_base_halts_before_peft_attach_on_v100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(
            monkeypatch, base_dtype=torch.bfloat16
        )

        with pytest.raises(BF16SlippageError):
            boot_gemma()

        assert FastModel.get_peft_model.call_count == 0

    def test_fp16_base_halts_before_peft_attach_on_h100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(
            monkeypatch, base_dtype=torch.float16
        )
        cfg = BootConfig(hardware="h100")
        with pytest.raises(FP16SlippageError):
            boot_gemma(cfg)

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
            lora_random_state=1234,
            use_gradient_checkpointing="false",
            finetune_vision_layers=True,
            finetune_language_layers=False,
        )
        boot_gemma(cfg)

        call_from = FastModel.from_pretrained.call_args
        assert call_from.args == ("custom/model",)
        assert call_from.kwargs["max_seq_length"] == 2048
        assert call_from.kwargs["load_in_4bit"] is False

        call_peft = FastModel.get_peft_model.call_args
        assert call_peft.kwargs["r"] == 8
        assert call_peft.kwargs["lora_alpha"] == 16
        assert call_peft.kwargs["random_state"] == 1234
        assert call_peft.kwargs["use_gradient_checkpointing"] == "false"
        assert call_peft.kwargs["finetune_vision_layers"] is True
        assert call_peft.kwargs["finetune_language_layers"] is False

    def test_assert_fp16_halt_message_includes_expected_dtype(self) -> None:
        model = _fake_model_with_dtype(torch.bfloat16)
        with pytest.raises(BF16SlippageError) as exc:
            assert_dtype_for_hardware(model, "v100")
        msg = str(exc.value)
        assert "torch.bfloat16" in msg
        assert "torch.float16" in msg
        assert "Halt training" in msg

    def test_boot_gemma_uses_none_config_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FastModel, _base, _peft = _install_fake_unsloth(monkeypatch)
        boot_gemma(None)

        call_from = FastModel.from_pretrained.call_args
        assert call_from.args == (BASE_MODEL_ID,)

    def test_module_importable_without_unsloth(self) -> None:
        """Reimport the module without unsloth in sys.modules — must not raise."""
        import importlib

        if "unsloth" in sys.modules:
            del sys.modules["unsloth"]
        mod: Any = importlib.import_module("cells.step_12_gemma_boot")
        assert mod.BASE_MODEL_ID == BASE_MODEL_ID


class TestAssertDtypeEmptyModel:
    def test_empty_parameters_raises(self) -> None:
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([]))
        with pytest.raises(BF16SlippageError):
            assert_dtype_for_hardware(model, "v100")
