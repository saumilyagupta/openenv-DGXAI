"""DriftCall FastAPI server — OpenEnv-expected module path.

Re-exports ``app`` from the project-root ``app.py`` so ``openenv validate``
discovers it under ``server/app.py``. The single source of truth is the
root-level ``app.py``; this file MUST NOT add routing or handler behavior.

``main()`` exists solely so ``openenv validate`` finds a callable entry
point at ``server/app.py:main`` and so ``python -m server.app`` boots the
server identically to ``uvicorn app:app``.
"""

from __future__ import annotations

import os

from app import app, create_app

__all__ = ["app", "create_app", "main"]


def main() -> None:
    """Boot the FastAPI server via uvicorn.

    Honors ``HOST`` / ``PORT`` env vars; defaults match the HF Space
    contract (``0.0.0.0:7860``). Imported lazily so module import stays
    side-effect-free.
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104 - container bind
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
