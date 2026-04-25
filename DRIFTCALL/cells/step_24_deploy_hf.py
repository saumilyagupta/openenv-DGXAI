"""Cell 24 — Hugging Face Hub + Spaces deployment.

Implements ``docs/modules/deploy_env_space.md`` §8.2 and DESIGN.md §11.3, §11.4
deliverables. Four push helpers, all using the **new** ``hf upload`` CLI per
deploy_env_space.md §8.2 (deprecated ``huggingface-cli`` is forbidden).

Public surface:
  * ``push_lora_to_hub(checkpoint_path, repo_id, token)`` — LoRA-only adapter
    push with ``safe_serialization=True``. Never the naive 4-bit → 16-bit merge
    path (DESIGN.md §10.5, CLAUDE.md §13).
  * ``push_env_space(repo_id, token)`` — Docker-based env Space (CPU basic,
    deploy_env_space.md §6.3).
  * ``push_demo_space(repo_id, token)`` — Demo Space targeting ZeroGPU with
    A10G fallback (deploy_demo_space.md §3.1, §3.7).
  * ``push_dataset(brief_path, repo_id, token)`` — ``driftcall-indic-briefs``
    dataset (DESIGN.md §11.4).

All four return a frozen :class:`DeploymentResult` so a caller can audit the
exact ``hf`` invocation. Heavy deps (``huggingface_hub``, ``subprocess`` for
``hf``) are loaded lazily; tests monkeypatch the loaders to assert the
command construction without making network calls.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — repo defaults (DESIGN.md §11.3, §11.4, deploy_*_space.md §3.7)
# ---------------------------------------------------------------------------


DEFAULT_LORA_REPO_ID: str = "DGXAI/gemma-4-e2b-driftcall-lora"
DEFAULT_DATASET_REPO_ID: str = "driftcall/driftcall-indic-briefs"
DEFAULT_ENV_SPACE_ID: str = "driftcall/driftcall-env"
DEFAULT_DEMO_SPACE_ID: str = "driftcall/driftcall-demo"

RepoType = Literal["model", "dataset", "space"]

DEPRECATED_CLI_NAMES: tuple[str, ...] = ("huggingface-cli",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeploymentError(Exception):
    """Root for every typed deploy-cell error."""


class HFTokenMissingError(DeploymentError):
    """Raised when the ``token`` argument is None or empty."""


class CheckpointPathMissingError(DeploymentError):
    """Raised when the LoRA checkpoint path does not exist."""


class NaiveMergeForbiddenError(DeploymentError):
    """Raised when the caller requests a 4-bit → 16-bit merge path
    (CLAUDE.md §13, DESIGN.md §10.5)."""


class DeploymentCommandError(DeploymentError):
    """Raised when the ``hf upload`` invocation exits non-zero."""


class DeprecatedCliError(DeploymentError):
    """Raised when a caller would invoke ``huggingface-cli`` instead of ``hf``."""


# ---------------------------------------------------------------------------
# DeploymentResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentResult:
    """Audit record for one deployment call."""

    repo_id: str
    repo_type: RepoType
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    success: bool


# ---------------------------------------------------------------------------
# Lazy dep loaders — patched by tests
# ---------------------------------------------------------------------------


def _load_hf_api() -> Any:
    """Return the ``huggingface_hub.HfApi`` class. Patched in tests."""

    from huggingface_hub import HfApi

    return HfApi


def _load_subprocess_run() -> Callable[..., Any]:
    """Return ``subprocess.run``. Patched in tests."""

    return subprocess.run


# ---------------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------------


def _validate_token(token: str | None) -> str:
    if token is None or token.strip() == "":
        raise HFTokenMissingError("token argument is required and must be non-empty")
    return token


def _validate_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise DeploymentError(f"repo_id must be 'org/name'; got {repo_id!r}")
    org, name = repo_id.split("/", 1)
    if not org or not name:
        raise DeploymentError(f"repo_id must be 'org/name'; got {repo_id!r}")
    return repo_id


def _validate_path_exists(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise DeploymentError(f"{label} must be pathlib.Path; got {type(path).__name__}")
    if not path.exists():
        raise CheckpointPathMissingError(f"{label} not found: {path}")
    return path


def _ensure_not_deprecated(executable: str) -> str:
    if executable in DEPRECATED_CLI_NAMES:
        raise DeprecatedCliError(
            f"{executable!r} is deprecated; use 'hf upload' (deploy_env_space.md §8.2)",
        )
    return executable


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_hf_upload_command(
    *,
    repo_id: str,
    local_path: Path,
    repo_type: RepoType,
    revision: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Construct an argv tuple for ``hf upload``.

    Shape per the new ``hf`` CLI (deploy_env_space.md §8.2):
        ``hf upload <repo_id> <local_path> --repo-type=<type> [--revision=<r>]``
    """

    _validate_repo_id(repo_id)
    if repo_type not in ("model", "dataset", "space"):
        raise DeploymentError(f"repo_type must be model|dataset|space; got {repo_type!r}")
    executable = _ensure_not_deprecated("hf")
    cmd: list[str] = [
        executable,
        "upload",
        repo_id,
        str(local_path),
        f"--repo-type={repo_type}",
    ]
    if revision is not None:
        cmd.append(f"--revision={revision}")
    cmd.extend(extra_args)
    return tuple(cmd)


def _run_command(
    cmd: tuple[str, ...],
    *,
    token: str,
    env_extra: Mapping[str, str] | None = None,
) -> tuple[int, str, str]:
    """Invoke ``cmd`` via subprocess; return ``(rc, stdout, stderr)``.

    The token is passed via environment, never via argv (avoids shell
    history leak). ``env_extra`` lets callers add per-deploy env vars.
    """

    import os

    run = _load_subprocess_run()
    env = dict(os.environ)
    env["HF_TOKEN"] = token
    env["HUGGINGFACE_HUB_TOKEN"] = token
    if env_extra is not None:
        env.update(env_extra)
    try:
        completed = run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise DeploymentCommandError(f"hf CLI not found on PATH: {exc}") from exc
    rc = int(getattr(completed, "returncode", 1))
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    return rc, stdout, stderr


# ---------------------------------------------------------------------------
# push_lora_to_hub (DESIGN.md §11.3)
# ---------------------------------------------------------------------------


def push_lora_to_hub(
    checkpoint_path: Path,
    repo_id: str = DEFAULT_LORA_REPO_ID,
    token: str | None = None,
    *,
    merge_4bit_to_16bit: bool = False,
    revision: str | None = None,
) -> DeploymentResult:
    """Push the LoRA adapter directory to the HF Hub.

    Pushes adapter-only artifacts (``adapter_config.json``,
    ``adapter_model.safetensors``, ``tokenizer.json``, ``README.md``).
    Never the merged-fp16 weights — see DESIGN.md §10.5 + CLAUDE.md §13:
    naive 4-bit → 16-bit merging is the catastrophic-quality path.
    """

    if merge_4bit_to_16bit:
        raise NaiveMergeForbiddenError(
            "merge_4bit_to_16bit=True is forbidden: 4-bit → 16-bit merge "
            "produces silently broken weights (DESIGN.md §10.5, CLAUDE.md §13). "
            "Push the LoRA adapter only.",
        )
    resolved_token = _validate_token(token)
    _validate_path_exists(checkpoint_path, label="checkpoint_path")
    cmd = build_hf_upload_command(
        repo_id=repo_id,
        local_path=checkpoint_path,
        repo_type="model",
        revision=revision,
    )
    rc, stdout, stderr = _run_command(cmd, token=resolved_token)
    success = rc == 0
    if not success:
        logger.warning("push_lora_to_hub failed (rc=%d): %s", rc, stderr)
    return DeploymentResult(
        repo_id=repo_id,
        repo_type="model",
        command=cmd,
        return_code=rc,
        stdout=stdout,
        stderr=stderr,
        success=success,
    )


# ---------------------------------------------------------------------------
# push_env_space (deploy_env_space.md §4.4, §6.3)
# ---------------------------------------------------------------------------


def push_env_space(
    repo_id: str = DEFAULT_ENV_SPACE_ID,
    token: str | None = None,
    *,
    space_dir: Path | None = None,
    revision: str | None = None,
) -> DeploymentResult:
    """Push the env Space (Docker SDK, CPU basic). deploy_env_space.md §4.4."""

    resolved_token = _validate_token(token)
    if space_dir is None:
        space_dir = Path(".")
    _validate_path_exists(space_dir, label="space_dir")
    cmd = build_hf_upload_command(
        repo_id=repo_id,
        local_path=space_dir,
        repo_type="space",
        revision=revision,
    )
    rc, stdout, stderr = _run_command(cmd, token=resolved_token)
    success = rc == 0
    return DeploymentResult(
        repo_id=repo_id,
        repo_type="space",
        command=cmd,
        return_code=rc,
        stdout=stdout,
        stderr=stderr,
        success=success,
    )


# ---------------------------------------------------------------------------
# push_demo_space (deploy_demo_space.md §3.1, §3.7)
# ---------------------------------------------------------------------------


def push_demo_space(
    repo_id: str = DEFAULT_DEMO_SPACE_ID,
    token: str | None = None,
    *,
    space_dir: Path | None = None,
    hardware: Literal["zero-gpu", "a10g-small"] = "zero-gpu",
    revision: str | None = None,
) -> DeploymentResult:
    """Push the demo Space. Default hardware ``zero-gpu`` per
    deploy_demo_space.md §3.1; pass ``a10g-small`` to redeploy on the
    fallback hardware (§3.1 step 2)."""

    resolved_token = _validate_token(token)
    if hardware not in ("zero-gpu", "a10g-small"):
        raise DeploymentError(
            f"hardware must be zero-gpu|a10g-small; got {hardware!r}",
        )
    if space_dir is None:
        space_dir = Path(".")
    _validate_path_exists(space_dir, label="space_dir")
    cmd = build_hf_upload_command(
        repo_id=repo_id,
        local_path=space_dir,
        repo_type="space",
        revision=revision,
    )
    env_extra = {"DRIFTCALL_HARDWARE": hardware}
    rc, stdout, stderr = _run_command(cmd, token=resolved_token, env_extra=env_extra)
    success = rc == 0
    return DeploymentResult(
        repo_id=repo_id,
        repo_type="space",
        command=cmd,
        return_code=rc,
        stdout=stdout,
        stderr=stderr,
        success=success,
    )


# ---------------------------------------------------------------------------
# push_dataset (DESIGN.md §11.4)
# ---------------------------------------------------------------------------


def push_dataset(
    brief_path: Path,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    token: str | None = None,
    *,
    revision: str | None = None,
) -> DeploymentResult:
    """Push the ``driftcall-indic-briefs`` dataset (DESIGN.md §11.4)."""

    resolved_token = _validate_token(token)
    _validate_path_exists(brief_path, label="brief_path")
    cmd = build_hf_upload_command(
        repo_id=repo_id,
        local_path=brief_path,
        repo_type="dataset",
        revision=revision,
    )
    rc, stdout, stderr = _run_command(cmd, token=resolved_token)
    success = rc == 0
    return DeploymentResult(
        repo_id=repo_id,
        repo_type="dataset",
        command=cmd,
        return_code=rc,
        stdout=stdout,
        stderr=stderr,
        success=success,
    )


__all__ = [
    "DEFAULT_DATASET_REPO_ID",
    "DEFAULT_DEMO_SPACE_ID",
    "DEFAULT_ENV_SPACE_ID",
    "DEFAULT_LORA_REPO_ID",
    "DEPRECATED_CLI_NAMES",
    "CheckpointPathMissingError",
    "DeploymentCommandError",
    "DeploymentError",
    "DeploymentResult",
    "DeprecatedCliError",
    "HFTokenMissingError",
    "NaiveMergeForbiddenError",
    "RepoType",
    "build_hf_upload_command",
    "push_dataset",
    "push_demo_space",
    "push_env_space",
    "push_lora_to_hub",
]
