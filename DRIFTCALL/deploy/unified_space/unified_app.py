"""Unified DriftCall Space — single FastAPI ASGI app combining:

- canonical OpenEnv routes at root (/reset, /step, /state, /close, /healthz)
  imported as-is from app.py so the env behaviour matches the standalone
  env Space byte-for-byte.
- the project frontend (Vite-built dist/) served as static files at /,
  with a SPA fallback so deep links work.
- a /demo redirect to the dedicated Gradio demo Space (kept separate
  because it's GPU-heavy and benefits from independent scaling).

This file lives only inside the unified Space build dir; the canonical
sources at the repo root are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Reuse the canonical OpenEnv FastAPI app — same router, same auth, same
# error envelope, same session pool. We just wrap it to add the static
# mount and the /demo redirect.
from app import app as openenv_app  # type: ignore[import-not-found]

DEMO_SPACE_URL = "https://dgxai-driftcall-demo.hf.space"
SITE_DIR = Path(__file__).parent / "site"


def build_unified_app() -> FastAPI:
    # The canonical app already has all OpenEnv routes registered. We
    # extend it rather than wrap, so route ordering + middleware all
    # apply unchanged to /reset, /step, /state, /close, /healthz.
    app: FastAPI = openenv_app

    @app.get("/demo", include_in_schema=False)
    async def demo_redirect() -> RedirectResponse:
        return RedirectResponse(url=DEMO_SPACE_URL, status_code=302)

    @app.get("/openenv.yaml", include_in_schema=False)
    async def serve_manifest() -> Any:
        manifest = Path(__file__).parent / "openenv.yaml"
        if manifest.exists():
            return FileResponse(manifest, media_type="text/yaml")
        return {"error": "openenv.yaml not found"}

    # SPA static mount — must come LAST so OpenEnv routes (/reset, /step,
    # /state, /close, /healthz) take precedence over a same-named asset.
    if SITE_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=SITE_DIR, html=True),
            name="frontend",
        )

    return app


app = build_unified_app()
