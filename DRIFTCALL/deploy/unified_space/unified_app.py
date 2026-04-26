"""Unified DriftCall Space — single FastAPI ASGI app exposing every artefact
under one hostname, all on bare slash-paths.

URL surface (all served from the same origin):

    /                  static project site (Vite-built React + Pretext)
    /assets/*          site bundle (CSS / JS / fonts)
    /healthz           OpenEnv health probe (text/plain "ok")
    /reset             POST  OpenEnv canonical reset (bearer auth)
    /step              POST  OpenEnv step
    /state             GET   OpenEnv read-only state
    /close             POST  OpenEnv close
    /openenv.yaml      OpenEnv v1.0 manifest
    /demo              voice demo (iframes the dedicated GPU Space inline)
    /env               same OpenEnv API as bare paths, but landed via a UI page
    /lora              302 → trained LoRA on HF Hub
    /source            302 → repo on GitHub
    /docs              FastAPI / Swagger UI for the OpenEnv routes

The OpenEnv routes don't collide with the static frontend because they're
verb+path specific (POST /reset, GET /state, etc) — Vite assets live under
/assets/* exclusively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import app as openenv_app  # type: ignore[import-not-found]

# External resources we link out to from the unified surface.
DEMO_SPACE_URL = "https://dgxai-driftcall-demo.hf.space"
LORA_HUB_URL = "https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora"
SOURCE_URL = "https://github.com/saumilyagupta/openenv-DGXAI"

SITE_DIR = Path(__file__).parent / "site"
MANIFEST_PATH = Path(__file__).parent / "openenv.yaml"

# ---------------------------------------------------------------------------
# Inline demo HTML — iframes the dedicated GPU Space without bouncing the
# user off our origin. Same dark editorial chrome as the project site so it
# feels native, not "redirected somewhere else".
# ---------------------------------------------------------------------------

_DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DriftCall — Live Voice Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500&family=Geist+Mono:wght@300;400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
<style>
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;padding:0;height:100%;background:#0a0a0c;color:#f0eae0;font-family:"Geist",system-ui,sans-serif}
  body{display:grid;grid-template-rows:auto 1fr;min-height:100vh}
  header{display:flex;align-items:center;justify-content:space-between;padding:1rem 1.4rem;border-bottom:1px solid #1f1f28;font-family:"Geist Mono",monospace;font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;color:#a8a29a;gap:1rem;flex-wrap:wrap}
  header .brand{font-family:"Instrument Serif",serif;font-style:italic;font-size:1.6rem;color:#f0eae0;letter-spacing:-.01em;text-transform:none}
  header .brand b{color:#ff7a17}
  header nav{display:flex;gap:.85rem}
  header nav a{color:#a8a29a;text-decoration:none}
  header nav a:hover{color:#ff7a17}
  header nav a.is-active{color:#ff7a17;border-bottom:1px solid #ff7a17;padding-bottom:.15rem}
  iframe{display:block;width:100%;height:100%;border:0;background:#0a0a0c}
  .frame-wrap{position:relative;isolation:isolate;background:#0a0a0c}
  .scanlines{position:absolute;inset:0;pointer-events:none;z-index:2;mix-blend-mode:overlay;opacity:.6;background-image:repeating-linear-gradient(180deg,transparent 0,transparent 2px,rgba(255,255,255,.012) 2px,rgba(255,255,255,.012) 3px)}
</style>
</head>
<body>
<header>
  <span class="brand">Drift<b>Call</b> · <span style="color:#a8a29a;font-style:italic;">/demo</span></span>
  <nav>
    <a href="/">site</a>
    <a href="/demo" class="is-active">demo</a>
    <a href="/env">env</a>
    <a href="/openenv.yaml">manifest</a>
    <a href="/lora">lora</a>
    <a href="/source">source</a>
  </nav>
</header>
<div class="frame-wrap">
  <iframe src="__DEMO_URL__"
    title="DriftCall Gradio demo"
    allow="microphone; clipboard-read; clipboard-write"></iframe>
  <div class="scanlines"></div>
</div>
</body>
</html>
"""


_ENV_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DriftCall — OpenEnv API</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500&family=Geist+Mono:wght@300;400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
<style>
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;background:#0a0a0c;color:#f0eae0;font-family:"Geist",system-ui,sans-serif}
  body{padding:0;min-height:100vh}
  header{display:flex;align-items:center;justify-content:space-between;padding:1rem 1.4rem;border-bottom:1px solid #1f1f28;font-family:"Geist Mono",monospace;font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;color:#a8a29a;gap:1rem;flex-wrap:wrap}
  header .brand{font-family:"Instrument Serif",serif;font-style:italic;font-size:1.6rem;color:#f0eae0;letter-spacing:-.01em;text-transform:none}
  header .brand b{color:#ff7a17}
  header nav{display:flex;gap:.85rem}
  header nav a{color:#a8a29a;text-decoration:none}
  header nav a:hover{color:#ff7a17}
  header nav a.is-active{color:#ff7a17}
  main{max-width:920px;margin:0 auto;padding:3rem 1.4rem 6rem}
  h1{font-family:"Instrument Serif",serif;font-style:italic;font-size:clamp(2rem,5vw,3.6rem);letter-spacing:-.02em;line-height:.95;margin:0 0 .6rem}
  h1 em{color:#ff7a17}
  p.lede{font-family:"Instrument Serif",serif;font-style:italic;color:#d9d3c8;font-size:1.15rem;line-height:1.5;margin:0 0 2rem;max-width:60ch}
  table{width:100%;border-collapse:collapse;margin:1.5rem 0;border:1px solid #1f1f28;background:#14141a}
  th,td{text-align:left;padding:.85rem 1rem;border-bottom:1px solid #1f1f28;vertical-align:baseline;font-size:.92rem}
  th{font-family:"Geist Mono",monospace;font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:#a8a29a;font-weight:400}
  tr:last-child th,tr:last-child td{border-bottom:0}
  td.path{font-family:"Geist Mono",monospace;color:#ff7a17}
  td.method{font-family:"Geist Mono",monospace;color:#2cb39d;letter-spacing:.05em}
  pre{background:#14141a;border:1px solid #1f1f28;padding:1rem 1.2rem;overflow:auto;font-family:"Geist Mono",monospace;font-size:.82rem;color:#d9d3c8;line-height:1.55}
  code{font-family:"Geist Mono",monospace;color:#ff7a17}
  a.inline{color:#f0eae0;text-decoration:underline;text-decoration-color:#ff7a17;text-underline-offset:3px}
  a.inline:hover{color:#ff7a17}
</style>
</head>
<body>
<header>
  <span class="brand">Drift<b>Call</b> · <span style="color:#a8a29a;font-style:italic;">/env</span></span>
  <nav>
    <a href="/">site</a>
    <a href="/demo">demo</a>
    <a href="/env" class="is-active">env</a>
    <a href="/openenv.yaml">manifest</a>
    <a href="/docs">docs</a>
    <a href="/lora">lora</a>
    <a href="/source">source</a>
  </nav>
</header>
<main>
  <h1>OpenEnv API <em>at canonical paths</em></h1>
  <p class="lede">
    Every endpoint sits at the bare path the OpenEnv v1.0 spec expects —
    no <code>/api</code> prefix, no rewriting. Auth is bearer
    (<code>DRIFTCALL_ENV_TOKEN</code>) plus <code>X-Session-Id</code> on
    every mutating call. The five reward components live in
    <a class="inline" href="/source">cells/step_08_rewards.py</a>; the
    full manifest is at <a class="inline" href="/openenv.yaml">/openenv.yaml</a>.
  </p>

  <table>
    <thead><tr><th>method</th><th>path</th><th>description</th></tr></thead>
    <tbody>
      <tr><td class="method">GET</td>  <td class="path">/healthz</td>      <td>health probe (unauthenticated, returns <code>"ok"</code>)</td></tr>
      <tr><td class="method">POST</td> <td class="path">/reset</td>        <td>create or recycle a session (seed / curriculum_stage / language_weights / audio_boundary_enabled)</td></tr>
      <tr><td class="method">POST</td> <td class="path">/step</td>         <td>advance one turn — body <code>{"action": &lt;DriftCallAction&gt;}</code></td></tr>
      <tr><td class="method">GET</td>  <td class="path">/state</td>        <td>read-only <code>DriftCallState</code> snapshot</td></tr>
      <tr><td class="method">POST</td> <td class="path">/close</td>        <td>evict the server-side session</td></tr>
      <tr><td class="method">GET</td>  <td class="path">/openenv.yaml</td> <td>OpenEnv v1.0 manifest</td></tr>
      <tr><td class="method">GET</td>  <td class="path">/docs</td>         <td>auto-generated FastAPI / Swagger UI</td></tr>
    </tbody>
  </table>

  <h2 style="font-family:'Instrument Serif',serif;font-style:italic;font-size:1.5rem;color:#f0eae0;margin-top:2.5rem">Try it</h2>
  <pre><code># 1) probe (no auth)
curl https://saumilyajj-driftcall.hf.space/healthz

# 2) reset (replace TOKEN with the one set on this Space)
curl -X POST https://saumilyajj-driftcall.hf.space/reset \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "X-Session-Id: smoke-001" \\
  -H "Content-Type: application/json" \\
  -d '{"seed": 42, "curriculum_stage": 2}'

# 3) step
curl -X POST https://saumilyajj-driftcall.hf.space/step \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "X-Session-Id: smoke-001" \\
  -H "Content-Type: application/json" \\
  -d '{"action": {"type": "end_episode"}}'</code></pre>
</main>
</body>
</html>
"""


def build_unified_app() -> FastAPI:
    # Extend the canonical app — same router, same auth, same error envelope.
    app: FastAPI = openenv_app

    @app.get("/openenv.yaml", include_in_schema=False)
    async def serve_manifest() -> Any:
        if MANIFEST_PATH.exists():
            return FileResponse(MANIFEST_PATH, media_type="text/yaml")
        return {"error": "openenv.yaml not found"}

    @app.get("/demo", include_in_schema=False)
    async def serve_demo_page() -> HTMLResponse:
        # Iframe the dedicated GPU Space inside our origin so the URL bar
        # stays at this Space — feels unified, no third-party hop.
        return HTMLResponse(_DEMO_HTML.replace("__DEMO_URL__", DEMO_SPACE_URL))

    @app.get("/env", include_in_schema=False)
    async def serve_env_page() -> HTMLResponse:
        return HTMLResponse(_ENV_HTML)

    @app.get("/lora", include_in_schema=False)
    async def lora_redirect() -> RedirectResponse:
        return RedirectResponse(url=LORA_HUB_URL, status_code=302)

    @app.get("/source", include_in_schema=False)
    async def source_redirect() -> RedirectResponse:
        return RedirectResponse(url=SOURCE_URL, status_code=302)

    @app.get("/site", include_in_schema=False)
    async def site_redirect() -> RedirectResponse:
        # Alias for users who type /site explicitly.
        return RedirectResponse(url="/", status_code=302)

    # Static frontend mount — must come LAST so OpenEnv POST routes
    # (/reset /step /close) and our HTML GET handlers above take
    # precedence over a same-named asset path.
    if SITE_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=SITE_DIR, html=True),
            name="frontend",
        )

    return app


app = build_unified_app()
