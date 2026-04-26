"""Cell 01 — Install pinned dependencies.

Runs once at notebook boot. On Colab the notebook kernel is a bare Python 3
install, so we ``pip install`` the flat pin set from ``requirements.txt``.
Locally we skip reinstall if every pin is already importable.

Also authenticates with the Hugging Face Hub when an ``HF_TOKEN`` environment
variable is set; on interactive sessions the user can run ``hf auth login``
separately. No network calls are attempted when ``HF_TOKEN`` is absent — the
cell remains a no-op so offline unit tests pass.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILENAME = "requirements.txt"

# Packages whose import name differs from their distribution name. Only list
# the handful we actually probe with ``is_installed``; everything else uses
# the distribution name verbatim.
_IMPORT_ALIASES: dict[str, str] = {
    "faster-whisper": "faster_whisper",
    "huggingface_hub": "huggingface_hub",
    "uvicorn[standard]": "uvicorn",
    "pytest-cov": "pytest_cov",
}


def is_installed(distribution: str) -> bool:
    """Return True iff the import name behind *distribution* is available."""

    base = distribution.split("[", 1)[0].split(">", 1)[0].split("<", 1)[0]
    base = base.split("==", 1)[0].split("~=", 1)[0].strip()
    module = _IMPORT_ALIASES.get(distribution, _IMPORT_ALIASES.get(base, base))
    module = module.replace("-", "_")
    return importlib.util.find_spec(module) is not None


def _find_requirements() -> Path | None:
    """Locate ``requirements.txt`` alongside the project root (worktree-safe)."""

    candidates = [
        Path.cwd() / REQUIREMENTS_FILENAME,
        Path(__file__).resolve().parent.parent / REQUIREMENTS_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def is_colab() -> bool:
    """Detect Google Colab runtime (``google.colab`` is always importable there)."""

    return importlib.util.find_spec("google.colab") is not None


def pip_install(requirements_path: Path) -> int:
    """Invoke ``pip install -r <requirements_path>`` via the current interpreter."""

    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(requirements_path)]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def hf_login_if_token_present() -> bool:
    """Log into HF Hub using ``HF_TOKEN`` env var. Returns True on success."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        return False
    try:
        from huggingface_hub import login
    except ImportError:
        return False
    login(token=token, add_to_git_credential=False)
    return True


def install(force: bool = False) -> int:
    """Top-level cell body. Idempotent: skips reinstall when pins already import.

    :param force: Reinstall even if every dependency is importable.
    :returns: 0 when deps already satisfied or pip succeeded; non-zero on pip failure.
    """

    requirements_path = _find_requirements()
    if requirements_path is None:
        return 0

    if not force and not is_colab():
        declared = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if declared and all(is_installed(pkg) for pkg in declared):
            hf_login_if_token_present()
            return 0

    rc = pip_install(requirements_path)
    if rc == 0:
        hf_login_if_token_present()
    return rc


# Cell body: execute on import so the Colab notebook runs end-to-end.
# Skip the side effect when the cell is being imported under the pytest
# runner or when a caller opts out via ``DRIFTCALL_SKIP_INSTALL=1``.
_skip_marker = "pytest" in sys.modules or os.environ.get("DRIFTCALL_SKIP_INSTALL") == "1"
_rc = 0 if _skip_marker else install()
