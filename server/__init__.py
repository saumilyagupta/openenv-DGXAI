"""DriftCall server package — re-exports the FastAPI app for OpenEnv compliance.

OpenEnv validates the env Space by looking for ``server/app.py``. Our actual
FastAPI implementation lives at the repository root in ``app.py`` (so
``uvicorn app:app`` works for local dev and Docker). This shim package
provides the OpenEnv-expected import path without duplicating logic.
"""

from __future__ import annotations
