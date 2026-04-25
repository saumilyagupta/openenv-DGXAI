"""Tests for the wandb init helper (task #14 — upgrade 5).

Covers:
  - ``init_wandb`` helper in ``cells.step_13_grpo_config``
  - Env-var contract (project / entity / mode / run-name template)
  - Tag resolution for stage / precision / adaptive-kl
  - Idempotence vs an existing ``wandb.run``
  - AdaptiveKLCallback pushes custom fields into ``logs`` so TRL's
    default reporter forwards them to wandb without extra glue.

All wandb calls are mocked — the suite must stay CPU-only and offline.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fake wandb module
# ---------------------------------------------------------------------------


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    *,
    init_raises: bool = False,
    active_run: Any | None = None,
) -> MagicMock:
    """Install a stub ``wandb`` module and return the init MagicMock."""
    wandb_mod: Any = types.ModuleType("wandb")
    init = MagicMock(name="wandb.init")
    if init_raises:
        init.side_effect = RuntimeError("simulated wandb failure")
    else:
        fake_run = MagicMock(name="wandb.run")
        fake_run.config = MagicMock(name="wandb.run.config")
        init.return_value = fake_run
    login = MagicMock(name="wandb.login")
    config = MagicMock(name="wandb.config")
    config.update = MagicMock(name="wandb.config.update")
    wandb_mod.init = init
    wandb_mod.login = login
    wandb_mod.config = config
    wandb_mod.run = active_run
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    return init


# ---------------------------------------------------------------------------
# init_wandb — disabled path
# ---------------------------------------------------------------------------


class TestInitWandbDisabled:
    def test_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "disabled")
        init = _install_fake_wandb(monkeypatch)
        run = init_wandb(stage=1, seed=42)
        assert run is None
        init.assert_not_called()

    def test_disabled_no_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "disabled")
        _install_fake_wandb(monkeypatch)
        import wandb  # the fake we just installed

        init_wandb(stage=1, seed=42)
        wandb.login.assert_not_called()  # type: ignore[attr-defined]

    def test_offline_still_initialises_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline mode logs to local dir — do not short-circuit like disabled."""
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "offline")
        init = _install_fake_wandb(monkeypatch)
        run = init_wandb(stage=1, seed=42)
        assert run is not None
        init.assert_called_once()


# ---------------------------------------------------------------------------
# init_wandb — env overrides
# ---------------------------------------------------------------------------


class TestInitWandbEnvOverrides:
    def test_project_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_PROJECT", "custom-project")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=2, seed=7)
        kwargs = init.call_args.kwargs
        assert kwargs["project"] == "custom-project"

    def test_project_default_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cells.step_13_grpo_config import WANDB_PROJECT_DEFAULT, init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=1, seed=1)
        assert init.call_args.kwargs["project"] == WANDB_PROJECT_DEFAULT

    def test_entity_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_ENTITY", "driftcall-team")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=1, seed=1)
        assert init.call_args.kwargs["entity"] == "driftcall-team"

    def test_no_entity_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=1, seed=1)
        # entity omitted (None) so wandb defaults to the caller's account.
        assert init.call_args.kwargs.get("entity") is None


# ---------------------------------------------------------------------------
# init_wandb — tags
# ---------------------------------------------------------------------------


class TestInitWandbTags:
    def _call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        stage: int = 1,
        seed: int = 1,
        h100_mode: bool = False,
        enable_adaptive_kl: bool = True,
    ) -> list[str]:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(
            stage=stage,
            seed=seed,
            h100_mode=h100_mode,
            enable_adaptive_kl=enable_adaptive_kl,
        )
        tags = init.call_args.kwargs["tags"]
        return list(tags)

    def test_stage_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tags = self._call(monkeypatch, stage=2)
        assert "stage2" in tags

    def test_seed_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tags = self._call(monkeypatch, seed=777)
        assert "seed777" in tags

    def test_gemma_tag_always_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tags = self._call(monkeypatch)
        assert "gemma-4-e2b" in tags

    def test_h100_tag_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tags = self._call(monkeypatch, h100_mode=True)
        assert "bf16" in tags
        assert "fp16" not in tags

    def test_v100_tag_when_h100_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tags = self._call(monkeypatch, h100_mode=False)
        assert "fp16" in tags
        assert "bf16" not in tags

    def test_adaptive_kl_tag_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tags = self._call(monkeypatch, enable_adaptive_kl=True)
        assert "adaptive-kl" in tags
        assert "static-kl" not in tags

    def test_static_kl_tag_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tags = self._call(monkeypatch, enable_adaptive_kl=False)
        assert "static-kl" in tags
        assert "adaptive-kl" not in tags


# ---------------------------------------------------------------------------
# init_wandb — config dict
# ---------------------------------------------------------------------------


class TestInitWandbConfigDict:
    def test_config_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=1, seed=42, h100_mode=False)
        cfg = init.call_args.kwargs["config"]
        expected_keys = {
            "stage",
            "seed",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "beta_initial",
            "target_kl",
            "learning_rate",
            "num_generations",
            "h100_mode",
        }
        assert expected_keys.issubset(set(cfg.keys())), (
            f"missing keys: {expected_keys - set(cfg.keys())}"
        )

    def test_config_matches_constants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cells.step_12_gemma_boot import LORA_ALPHA, LORA_DROPOUT, LORA_R
        from cells.step_13_grpo_config import (
            BETA_KL,
            DEFAULT_NUM_GENERATIONS,
            LEARNING_RATE,
            init_wandb,
        )

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=3, seed=11, h100_mode=True)
        cfg = init.call_args.kwargs["config"]
        assert cfg["stage"] == 3
        assert cfg["seed"] == 11
        assert cfg["h100_mode"] is True
        assert cfg["lora_r"] == LORA_R
        assert cfg["lora_alpha"] == LORA_ALPHA
        assert cfg["lora_dropout"] == LORA_DROPOUT
        assert cfg["beta_initial"] == BETA_KL
        assert cfg["target_kl"] == BETA_KL
        assert cfg["learning_rate"] == LEARNING_RATE
        assert cfg["num_generations"] == DEFAULT_NUM_GENERATIONS


# ---------------------------------------------------------------------------
# init_wandb — idempotence + run naming
# ---------------------------------------------------------------------------


class TestInitWandbIdempotent:
    def test_second_call_returns_existing_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        existing_run = MagicMock(name="preexisting-run")
        init = _install_fake_wandb(monkeypatch, active_run=existing_run)
        run = init_wandb(stage=1, seed=1)
        assert run is existing_run
        init.assert_not_called()

    def test_run_name_includes_stage_and_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cells.step_13_grpo_config import init_wandb

        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        init = _install_fake_wandb(monkeypatch)
        init_wandb(stage=2, seed=314)
        name = init.call_args.kwargs["name"]
        assert "stage2" in name
        assert "seed314" in name


# ---------------------------------------------------------------------------
# AdaptiveKLCallback — extra log fields pushed back into logs dict
# ---------------------------------------------------------------------------


@dataclass
class _FakeArgs:
    beta: float = 0.04


@dataclass
class _FakeState:
    global_step: int = 0


class TestAdaptiveKLLogsExtraFields:
    def test_beta_adaptive_added_to_logs(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        logs: dict[str, Any] = {"kl": 0.08}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert "train/beta_adaptive" in logs
        assert logs["train/beta_adaptive"] == pytest.approx(args.beta)

    def test_kl_measured_added(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        logs: dict[str, Any] = {"kl": 0.12}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert logs["train/kl_measured"] == pytest.approx(0.12)

    def test_kl_target_added(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.07, kp=1.0)
        args = _FakeArgs(beta=0.04)
        logs: dict[str, Any] = {"kl": 0.07}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert logs["train/kl_target"] == pytest.approx(0.07)

    def test_clamp_min_flag_fires_on_collapse(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(
            target_kl=0.04, kp=50.0, beta_min=0.001, beta_max=1.0
        )
        args = _FakeArgs(beta=0.002)
        logs: dict[str, Any] = {"kl": 0.0}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert logs["train/beta_clamped_to_min"] == 1
        assert logs["train/beta_clamped_to_max"] == 0

    def test_clamp_max_flag_fires_on_runaway(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(
            target_kl=0.04, kp=50.0, beta_min=0.001, beta_max=0.5
        )
        args = _FakeArgs(beta=0.4)
        logs: dict[str, Any] = {"kl": 5.0}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert logs["train/beta_clamped_to_max"] == 1
        assert logs["train/beta_clamped_to_min"] == 0

    def test_no_clamp_when_in_band(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        logs: dict[str, Any] = {"kl": 0.04}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert logs["train/beta_clamped_to_min"] == 0
        assert logs["train/beta_clamped_to_max"] == 0

    def test_no_fields_on_missing_kl(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        args = _FakeArgs(beta=0.04)
        logs: dict[str, Any] = {"loss": 0.5}
        cb.on_log(args, _FakeState(global_step=1), None, logs=logs)
        assert "train/beta_adaptive" not in logs
        assert "train/kl_measured" not in logs
