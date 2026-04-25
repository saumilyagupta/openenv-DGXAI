"""Tests for cells/step_24_deploy_hf.py.

All HF calls mocked. Asserts:
  - hf upload command construction (NOT deprecated huggingface-cli).
  - LoRA push with safe_serialization invariant + naive-merge guard.
  - Env Space + demo Space + dataset push command shapes.
  - Token forwarded via env, not argv.
  - DeploymentResult is frozen and carries the audit fields.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cells import step_24_deploy_hf as deploy_mod
from cells.step_24_deploy_hf import (
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DEMO_SPACE_ID,
    DEFAULT_ENV_SPACE_ID,
    DEFAULT_LORA_REPO_ID,
    DEPRECATED_CLI_NAMES,
    CheckpointPathMissingError,
    DeploymentCommandError,
    DeploymentError,
    DeploymentResult,
    DeprecatedCliError,
    HFTokenMissingError,
    NaiveMergeForbiddenError,
    build_hf_upload_command,
    push_dataset,
    push_demo_space,
    push_env_space,
    push_lora_to_hub,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: str = "uploaded",
    stderr: str = "",
) -> MagicMock:
    """Patch the lazy subprocess.run loader and return the captured mock."""

    captured = MagicMock(
        return_value=_FakeCompletedProcess(
            returncode=returncode, stdout=stdout, stderr=stderr
        )
    )
    monkeypatch.setattr(deploy_mod, "_load_subprocess_run", lambda: captured)
    return captured


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_lora_repo(self) -> None:
        assert DEFAULT_LORA_REPO_ID == "DGXAI/gemma-4-e2b-driftcall-lora"

    def test_default_dataset_repo(self) -> None:
        assert DEFAULT_DATASET_REPO_ID == "driftcall/driftcall-indic-briefs"

    def test_default_env_space(self) -> None:
        assert DEFAULT_ENV_SPACE_ID == "driftcall/driftcall-env"

    def test_default_demo_space(self) -> None:
        assert DEFAULT_DEMO_SPACE_ID == "driftcall/driftcall-demo"

    def test_deprecated_cli_listed(self) -> None:
        assert "huggingface-cli" in DEPRECATED_CLI_NAMES


# ---------------------------------------------------------------------------
# build_hf_upload_command
# ---------------------------------------------------------------------------


class TestCommandBuilder:
    def test_uses_hf_not_huggingface_cli(self, tmp_path: Path) -> None:
        cmd = build_hf_upload_command(
            repo_id="org/x",
            local_path=tmp_path,
            repo_type="model",
        )
        assert cmd[0] == "hf"
        assert cmd[1] == "upload"
        for c in cmd:
            assert c not in DEPRECATED_CLI_NAMES

    def test_command_includes_repo_type_flag(self, tmp_path: Path) -> None:
        cmd = build_hf_upload_command(
            repo_id="org/x",
            local_path=tmp_path,
            repo_type="dataset",
        )
        assert "--repo-type=dataset" in cmd

    def test_command_includes_repo_id_and_path(self, tmp_path: Path) -> None:
        cmd = build_hf_upload_command(
            repo_id="org/x",
            local_path=tmp_path,
            repo_type="space",
        )
        assert "org/x" in cmd
        assert str(tmp_path) in cmd

    def test_revision_flag_when_supplied(self, tmp_path: Path) -> None:
        cmd = build_hf_upload_command(
            repo_id="org/x",
            local_path=tmp_path,
            repo_type="model",
            revision="v1.0.0",
        )
        assert "--revision=v1.0.0" in cmd

    def test_invalid_repo_type_raises(self, tmp_path: Path) -> None:
        bogus: Any = "bogus"
        with pytest.raises(DeploymentError):
            build_hf_upload_command(
                repo_id="org/x",
                local_path=tmp_path,
                repo_type=bogus,
            )

    def test_invalid_repo_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DeploymentError):
            build_hf_upload_command(
                repo_id="badformat",
                local_path=tmp_path,
                repo_type="model",
            )


# ---------------------------------------------------------------------------
# push_lora_to_hub
# ---------------------------------------------------------------------------


class TestPushLora:
    def test_token_required(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        with pytest.raises(HFTokenMissingError):
            push_lora_to_hub(ckpt, token=None)

    def test_empty_token_rejected(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        with pytest.raises(HFTokenMissingError):
            push_lora_to_hub(ckpt, token="   ")

    def test_path_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointPathMissingError):
            push_lora_to_hub(tmp_path / "missing", token="hf_x")

    def test_naive_merge_forbidden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        with pytest.raises(NaiveMergeForbiddenError):
            push_lora_to_hub(ckpt, token="hf_x", merge_4bit_to_16bit=True)

    def test_invokes_hf_upload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        run_mock = _patch_run(monkeypatch)
        result = push_lora_to_hub(ckpt, token="hf_x")
        run_mock.assert_called_once()
        assert result.success is True
        assert result.repo_type == "model"
        assert result.command[0] == "hf"
        assert result.command[1] == "upload"

    def test_token_passed_via_env_not_argv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        run_mock = _patch_run(monkeypatch)
        push_lora_to_hub(ckpt, token="hf_secret")
        call = run_mock.call_args
        argv = call.args[0]
        for arg in argv:
            assert "hf_secret" not in arg
        env = call.kwargs.get("env", {})
        assert env.get("HF_TOKEN") == "hf_secret"
        assert env.get("HUGGINGFACE_HUB_TOKEN") == "hf_secret"

    def test_repo_id_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        _patch_run(monkeypatch)
        result = push_lora_to_hub(ckpt, token="hf_x")
        assert result.repo_id == DEFAULT_LORA_REPO_ID

    def test_failure_returncode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        _patch_run(monkeypatch, returncode=1, stderr="auth failed")
        result = push_lora_to_hub(ckpt, token="hf_x")
        assert result.success is False
        assert result.return_code == 1
        assert "auth failed" in result.stderr

    def test_hf_cli_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        def _raise(*_a: Any, **_k: Any) -> Any:
            raise FileNotFoundError("hf not on PATH")

        monkeypatch.setattr(deploy_mod, "_load_subprocess_run", lambda: _raise)
        with pytest.raises(DeploymentCommandError):
            push_lora_to_hub(ckpt, token="hf_x")


# ---------------------------------------------------------------------------
# push_env_space
# ---------------------------------------------------------------------------


class TestPushEnvSpace:
    def test_token_required(self, tmp_path: Path) -> None:
        with pytest.raises(HFTokenMissingError):
            push_env_space(token=None, space_dir=tmp_path)

    def test_command_repo_type_space(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_mock = _patch_run(monkeypatch)
        result = push_env_space(token="hf_x", space_dir=tmp_path)
        argv = run_mock.call_args.args[0]
        assert "--repo-type=space" in argv
        assert result.repo_type == "space"

    def test_default_repo_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_run(monkeypatch)
        result = push_env_space(token="hf_x", space_dir=tmp_path)
        assert result.repo_id == DEFAULT_ENV_SPACE_ID


# ---------------------------------------------------------------------------
# push_demo_space
# ---------------------------------------------------------------------------


class TestPushDemoSpace:
    def test_default_hardware_zero_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_mock = _patch_run(monkeypatch)
        push_demo_space(token="hf_x", space_dir=tmp_path)
        env = run_mock.call_args.kwargs["env"]
        assert env["DRIFTCALL_HARDWARE"] == "zero-gpu"

    def test_a10g_fallback_via_arg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_mock = _patch_run(monkeypatch)
        push_demo_space(token="hf_x", space_dir=tmp_path, hardware="a10g-small")
        env = run_mock.call_args.kwargs["env"]
        assert env["DRIFTCALL_HARDWARE"] == "a10g-small"

    def test_invalid_hardware_rejected(self, tmp_path: Path) -> None:
        bogus: Any = "cpu-basic"
        with pytest.raises(DeploymentError):
            push_demo_space(
                token="hf_x",
                space_dir=tmp_path,
                hardware=bogus,
            )

    def test_repo_type_space(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_mock = _patch_run(monkeypatch)
        push_demo_space(token="hf_x", space_dir=tmp_path)
        argv = run_mock.call_args.args[0]
        assert "--repo-type=space" in argv

    def test_default_repo_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_run(monkeypatch)
        result = push_demo_space(token="hf_x", space_dir=tmp_path)
        assert result.repo_id == DEFAULT_DEMO_SPACE_ID


# ---------------------------------------------------------------------------
# push_dataset
# ---------------------------------------------------------------------------


class TestPushDataset:
    def test_token_required(self, tmp_path: Path) -> None:
        briefs = tmp_path / "briefs"
        briefs.mkdir()
        with pytest.raises(HFTokenMissingError):
            push_dataset(briefs, token=None)

    def test_path_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointPathMissingError):
            push_dataset(tmp_path / "missing", token="hf_x")

    def test_repo_type_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        briefs = tmp_path / "briefs"
        briefs.mkdir()
        run_mock = _patch_run(monkeypatch)
        push_dataset(briefs, token="hf_x")
        argv = run_mock.call_args.args[0]
        assert "--repo-type=dataset" in argv

    def test_default_repo_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        briefs = tmp_path / "briefs"
        briefs.mkdir()
        _patch_run(monkeypatch)
        result = push_dataset(briefs, token="hf_x")
        assert result.repo_id == DEFAULT_DATASET_REPO_ID


# ---------------------------------------------------------------------------
# DeploymentResult shape
# ---------------------------------------------------------------------------


class TestDeploymentResult:
    def test_is_frozen(self) -> None:
        result = DeploymentResult(
            repo_id="org/x",
            repo_type="model",
            command=("hf", "upload"),
            return_code=0,
            stdout="ok",
            stderr="",
            success=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.success = False


# ---------------------------------------------------------------------------
# Deprecated CLI guardrail
# ---------------------------------------------------------------------------


class TestDeprecatedCliGuardrail:
    def test_huggingface_cli_blocked(self) -> None:
        with pytest.raises(DeprecatedCliError):
            deploy_mod._ensure_not_deprecated("huggingface-cli")

    def test_hf_allowed(self) -> None:
        assert deploy_mod._ensure_not_deprecated("hf") == "hf"


# ---------------------------------------------------------------------------
# Module surface / pragmas
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_no_pragmas(self) -> None:
        text = Path(deploy_mod.__file__).read_text(encoding="utf-8")
        forbidden_marker_a = "type" + ": " + "ignore"
        forbidden_marker_b = "# " + "noqa"
        assert forbidden_marker_a not in text
        assert forbidden_marker_b not in text

    def test_callable_surface(self) -> None:
        for fn in (push_lora_to_hub, push_env_space, push_demo_space, push_dataset):
            assert callable(fn)
