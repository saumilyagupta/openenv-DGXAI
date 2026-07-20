"""Cell 02 — Consolidated imports.

Grouped re-exports of stdlib + third-party modules used across later cells.
Later cells ``from cells.step_02_imports import X`` (or import names directly);
this keeps the notebook top DRY while the individual ``.py`` files remain
standalone importable modules for the test suite and the FastAPI server.

Unused-import warnings on re-exported names are silenced via the
``[tool.ruff.lint.per-file-ignores]`` override in ``pyproject.toml`` rather
than per-line ``noqa`` pragmas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import dataclasses
import hashlib
import importlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

# ---------------------------------------------------------------------------
# Third-party — heavy deps are guarded so test collection does not explode
# when a single wheel is missing on a fresh Colab runtime.
# ---------------------------------------------------------------------------

_OPTIONAL_MODULES: tuple[str, ...] = (
    "numpy",
    "yaml",
    "fastapi",
    "uvicorn",
    "pydantic",
    "soundfile",
)

_loaded: dict[str, Any] = {}
for _name in _OPTIONAL_MODULES:
    try:
        _loaded[_name] = importlib.import_module(_name)
    except ImportError:  # pragma: no cover — exercised on fresh Colab only
        _loaded[_name] = None


def get_optional(name: str) -> Any:
    """Return an optional third-party module or ``None`` when unavailable."""

    return _loaded.get(name)


# Names re-exported for downstream cells. Everything imported above is fair
# game via ``from cells.step_02_imports import X``.
__all__ = (
    # stdlib re-exports
    "Any",
    "Callable",
    "Enum",
    "Literal",
    "Mapping",
    "Path",
    "Protocol",
    "Sequence",
    "TypeVar",
    "dataclass",
    "dataclasses",
    "field",
    "hashlib",
    "io",
    "json",
    "logging",
    "math",
    "os",
    "random",
    "re",
    "sys",
    "time",
    "uuid",
    # helpers
    "get_optional",
)
