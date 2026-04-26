"""Cell 01 — Install pinned dependencies + notebook bootstrap.

Runs once at notebook boot. Does three things:

1. **(Colab only) clones the repo** into ``/content/openenv-DGXAI`` if absent
   and ``chdir`` into it, so subsequent cells see ``cells/`` and ``data/``
   on disk.
2. **Adds the repo root to ``sys.path``** so cross-cell imports such as
   ``from cells.step_04_models import GoalSpec`` resolve when the notebook
   runs cells in a single global namespace.
3. **Installs the pinned ``requirements.txt``**. Idempotent — on a configured
   local machine where every pin is already importable the step is a no-op.

Also authenticates with the Hugging Face Hub when ``HF_TOKEN`` is set in the
environment. ``HF_TOKEN`` absent → no-op so offline unit tests pass.

Notebook-safe: ``__file__`` is **undefined** when this source is executed as
a Jupyter cell (rather than imported as a module), so the repo root is
discovered via ``Path.cwd()`` walk + the optional ``__file__`` fallback —
never by dereferencing ``__file__`` unconditionally.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILENAME = "requirements.txt"
REPO_URL = "https://github.com/saumilyagupta/openenv-DGXAI.git"
REPO_DIRNAME = "openenv-DGXAI"
REPO_BRANCH = "submission"

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


def is_colab() -> bool:
    """Detect Google Colab runtime (``google.colab`` is always importable there)."""

    return importlib.util.find_spec("google.colab") is not None


def _module_dir() -> Path | None:
    """Resolve this file's directory; ``None`` when running as a notebook cell."""

    file_attr = globals().get("__file__")
    if file_attr is None:
        return None
    try:
        return Path(file_attr).resolve().parent
    except (OSError, ValueError):
        return None


def _looks_like_repo_root(candidate: Path) -> bool:
    return (candidate / REQUIREMENTS_FILENAME).is_file() and (candidate / "cells").is_dir()


def find_repo_root() -> Path:
    """Locate the repo root. Walks cwd + parents, falls back to ``__file__``-derived path."""

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents[:3])
    mod = _module_dir()
    if mod is not None:
        candidates.append(mod.parent)
    candidates.append(Path("/content") / REPO_DIRNAME)

    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve() if c.exists() else c
        if c in seen:
            continue
        seen.add(c)
        if _looks_like_repo_root(c):
            return c
    return cwd


def _find_requirements() -> Path | None:
    """Locate ``requirements.txt`` — checks cwd, then parents up to depth 3,
    then the directory derived from ``__file__`` (script context only)."""

    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append(cwd / REQUIREMENTS_FILENAME)
    for parent in cwd.parents[:3]:
        candidates.append(parent / REQUIREMENTS_FILENAME)
    mod = _module_dir()
    if mod is not None:
        candidates.append(mod.parent / REQUIREMENTS_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def colab_clone_if_needed() -> Path | None:
    """On Colab, clone the repo into ``/content`` if absent and ``chdir`` into it.

    Guarded so unit tests that monkeypatch ``is_colab`` to ``True`` on a
    non-Colab host never invoke ``git clone``: real Colab always has a
    ``/content`` directory, so we additionally require that to exist.
    """

    if not is_colab():
        return None
    content_root = Path("/content")
    if not content_root.is_dir():
        return None
    target = content_root / REPO_DIRNAME
    if not target.is_dir():
        cmd = [
            "git",
            "clone",
            "--depth=1",
            "--branch",
            REPO_BRANCH,
            REPO_URL,
            str(target),
        ]
        subprocess.run(cmd, check=True)
    os.chdir(target)
    return target


def add_to_syspath(repo_root: Path) -> None:
    """Prepend ``repo_root`` to ``sys.path`` so ``from cells.step_NN_*`` resolves."""

    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


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

    colab_clone_if_needed()
    add_to_syspath(find_repo_root())

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
