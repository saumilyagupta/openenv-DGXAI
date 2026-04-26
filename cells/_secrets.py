"""DriftCall — hardcoded secrets for private-repo runs.

This file contains credentials. Repository is private per user direction.
Do NOT make this repository public without scrubbing this file from history:

    git filter-repo --path cells/_secrets.py --invert-paths

To rotate a key: replace the value below and the running training script
will pick it up on next launch (init_wandb reads via os.environ first;
this file is the fallback when env var is unset).
"""

from __future__ import annotations

import os

# wandb.ai API key — pasted by user 2026-04-25.
# Rotate at https://wandb.ai/authorize → Reset, then update below.
WANDB_API_KEY: str = "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"

# Default project + mode — override via env if needed.
WANDB_PROJECT: str = "driftcall"
WANDB_ENTITY: str | None = None
WANDB_MODE: str = "online"


def export_to_env() -> None:
    """Push hardcoded values into ``os.environ`` if not already set.

    Called by ``init_wandb()`` at the start of each training run. Env-var
    overrides take priority — set ``WANDB_API_KEY=...`` in the shell to bypass
    this file without editing it.
    """
    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    if WANDB_ENTITY is not None:
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
    os.environ.setdefault("WANDB_MODE", WANDB_MODE)


__all__ = [
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "export_to_env",
]
