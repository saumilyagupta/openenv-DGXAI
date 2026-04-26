"""Unified DriftCall Space — every artefact at a slash-path, served locally.

URL surface:

    /                  static project site (Vite-built React + Pretext)
    /assets/*          site bundle
    /healthz           OpenEnv health probe
    /reset             POST  OpenEnv canonical reset
    /step              POST  OpenEnv step
    /state             GET   OpenEnv read-only state
    /close             POST  OpenEnv close
    /openenv.yaml      OpenEnv v1.0 manifest
    /docs              FastAPI / Swagger UI for the env routes
    /demo              voice demo Gradio app, MOUNTED LOCALLY (no iframe)
    /env               landing page documenting env routes (curl recipes)
    /lora              landing page with LoRA metadata + download link
    /source            landing page with the project file tree + repo link

Nothing under this Space depends on a 302 redirect to another origin —
the demo Gradio Blocks are mounted via gr.mount_gradio_app, the LoRA
and source pages are rendered server-side from local data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# `app.py` refuses to start without DRIFTCALL_ENV_TOKEN. For the hackathon
# demo Space we publish a known public token so anyone can hit the OpenEnv
# API end-to-end. To restrict access, override via Space Settings → Secrets.
os.environ.setdefault("DRIFTCALL_ENV_TOKEN", "driftcall-demo")

# `app.py` eager-loads Kokoro TTS + faster-whisper ASR at lifespan startup.
# On this unified Space those engines aren't needed for the OpenEnv API
# surface — they only matter inside the /demo Gradio app, which constructs
# its own engines lazily on first audio I/O. We monkey-patch the eager-load
# step BEFORE importing app.py so the lifespan handler skips it. This also
# dodges the misaki/kokoro version-drift error (`module 'misaki.en' has no
# attribute 'MutableToken'`) that was crashing startup.
import importlib

_audio = importlib.import_module("cells.step_09_audio")


def _noop_engine() -> None:
    return None


_audio.get_tts_engine = _noop_engine          # type: ignore[assignment]
_audio.get_asr_engine = _noop_engine          # type: ignore[assignment]

from app import app as openenv_app  # noqa: E402  # type: ignore[import-not-found]
from online_trainer import get_online_trainer  # noqa: E402  # type: ignore[import-not-found]

LORA_HUB_URL = "https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora"
SOURCE_URL = "https://github.com/saumilyagupta/openenv-DGXAI"

SITE_DIR = Path(__file__).parent / "site"
MANIFEST_PATH = Path(__file__).parent / "openenv.yaml"

# ---------------------------------------------------------------------------
# Shared chrome — same dark editorial brutalism as the React site.
# ---------------------------------------------------------------------------
_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500&family=Geist+Mono:wght@300;400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
<style>
  *,*::before,*::after{box-sizing:border-box}
  html,body{margin:0;background:#0a0a0c;color:#f0eae0;font-family:"Geist",system-ui,sans-serif}
  body{min-height:100vh;display:grid;grid-template-rows:auto 1fr}
  header{display:flex;align-items:center;justify-content:space-between;padding:1rem 1.4rem;border-bottom:1px solid #1f1f28;font-family:"Geist Mono",monospace;font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;color:#a8a29a;gap:1rem;flex-wrap:wrap}
  header .brand{font-family:"Instrument Serif",serif;font-style:italic;font-size:1.6rem;color:#f0eae0;letter-spacing:-.01em;text-transform:none}
  header .brand b{color:#ff7a17}
  header nav{display:flex;gap:.85rem;flex-wrap:wrap}
  header nav a{color:#a8a29a;text-decoration:none}
  header nav a:hover{color:#ff7a17}
  header nav a.is-active{color:#ff7a17;border-bottom:1px solid #ff7a17;padding-bottom:.15rem}
  main{padding:3rem 1.4rem 6rem;max-width:1080px;margin:0 auto;width:100%}
  h1{font-family:"Instrument Serif",serif;font-style:italic;font-size:clamp(2rem,5vw,3.6rem);letter-spacing:-.02em;line-height:.95;margin:0 0 .6rem}
  h1 em{color:#ff7a17}
  h2{font-family:"Instrument Serif",serif;font-style:italic;font-size:1.5rem;color:#f0eae0;margin:2.5rem 0 1rem}
  p.lede{font-family:"Instrument Serif",serif;font-style:italic;color:#d9d3c8;font-size:1.15rem;line-height:1.5;margin:0 0 2rem;max-width:60ch}
  p{line-height:1.55;color:#d9d3c8}
  table{width:100%;border-collapse:collapse;margin:1.5rem 0;border:1px solid #1f1f28;background:#14141a}
  th,td{text-align:left;padding:.85rem 1rem;border-bottom:1px solid #1f1f28;vertical-align:baseline;font-size:.92rem}
  th{font-family:"Geist Mono",monospace;font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:#a8a29a;font-weight:400}
  tr:last-child th,tr:last-child td{border-bottom:0}
  td.path{font-family:"Geist Mono",monospace;color:#ff7a17}
  td.method{font-family:"Geist Mono",monospace;color:#2cb39d;letter-spacing:.05em}
  td.k{font-family:"Geist Mono",monospace;color:#a8a29a;letter-spacing:.06em;text-transform:uppercase;font-size:.72rem;width:14rem}
  pre{background:#14141a;border:1px solid #1f1f28;padding:1rem 1.2rem;overflow:auto;font-family:"Geist Mono",monospace;font-size:.82rem;color:#d9d3c8;line-height:1.55;border-radius:0}
  code{font-family:"Geist Mono",monospace;color:#ff7a17}
  a.inline{color:#f0eae0;text-decoration:underline;text-decoration-color:#ff7a17;text-underline-offset:3px}
  a.inline:hover{color:#ff7a17}
  .btn{display:inline-flex;align-items:center;gap:.7em;padding:.85rem 1.3rem;border:1px solid #ff7a17;background:#ff7a17;color:#0a0a0c;font-family:"Geist Mono",monospace;text-decoration:none;letter-spacing:-.01em;transition:background .2s}
  .btn:hover{background:#f0eae0;border-color:#f0eae0}
  .btn-ghost{background:transparent;color:#f0eae0;border-color:#1f1f28}
  .btn-ghost:hover{border-color:#ff7a17;color:#ff7a17}
  .iframe-wrap{position:relative;isolation:isolate;background:#0a0a0c;height:100%}
  .iframe-wrap iframe{display:block;width:100%;height:100%;border:0;background:#0a0a0c}
  .scanlines{position:absolute;inset:0;pointer-events:none;z-index:2;mix-blend-mode:overlay;opacity:.6;background-image:repeating-linear-gradient(180deg,transparent 0,transparent 2px,rgba(255,255,255,.012) 2px,rgba(255,255,255,.012) 3px)}
  .file-tree{font-family:"Geist Mono",monospace;font-size:.85rem;line-height:1.7;color:#d9d3c8;background:#14141a;border:1px solid #1f1f28;padding:1.5rem;white-space:pre;overflow-x:auto}
  .file-tree .dir{color:#ff7a17}
  .file-tree .note{color:#a8a29a;font-style:italic}
</style>
</head>
<body>
<header>
  <span class="brand">Drift<b>Call</b> · <span style="color:#a8a29a;font-style:italic;">__SLUG__</span></span>
  <nav>
    __NAV__
  </nav>
</header>
<main>
__BODY__
</main>
</body>
</html>
"""

_NAV_LINKS = [
    ("/", "site"),
    ("/demo", "demo"),
    ("/env", "env"),
    ("/openenv.yaml", "manifest"),
    ("/docs", "docs"),
    ("/lora", "lora"),
    ("/source", "source"),
]


def _page(title: str, slug: str, body: str, active: str) -> str:
    nav_items: list[str] = []
    for href, label in _NAV_LINKS:
        cls = ' class="is-active"' if href == active else ""
        nav_items.append(f'<a href="{href}"{cls}>{label}</a>')
    nav = "\n    ".join(nav_items)
    return (
        _HEAD
        .replace("__TITLE__", title)
        .replace("__SLUG__", slug)
        .replace("__NAV__", nav)
        .replace("__BODY__", body)
    )


# ---------------------------------------------------------------------------
# /env — landing page with curl recipes.
# ---------------------------------------------------------------------------
_ENV_BODY = """
<h1>OpenEnv API <em>at canonical paths</em></h1>
<p class="lede">
  Every endpoint sits at the bare path the OpenEnv v1.0 spec expects —
  no <code>/api</code> prefix, no rewriting. Auth is bearer
  (<code>DRIFTCALL_ENV_TOKEN</code>) plus <code>X-Session-Id</code> on
  every mutating call.
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

<h2>Try it</h2>
<p>This Space publishes a known public bearer token so anyone can exercise
the API end-to-end:
<code style="background:#14141a;border:1px solid #1f1f28;padding:.05em .4em">driftcall-demo</code>.</p>
<pre><code># 1) probe (no auth)
curl https://saumilyajj-driftcall.hf.space/healthz

# 2) reset
curl -X POST https://saumilyajj-driftcall.hf.space/reset \\
  -H "Authorization: Bearer driftcall-demo" \\
  -H "X-Session-Id: smoke-001" \\
  -H "Content-Type: application/json" \\
  -d '{"seed": 42, "curriculum_stage": 2}'

# 3) step
curl -X POST https://saumilyajj-driftcall.hf.space/step \\
  -H "Authorization: Bearer driftcall-demo" \\
  -H "X-Session-Id: smoke-001" \\
  -H "Content-Type: application/json" \\
  -d '{"action": {"type": "end_episode"}}'</code></pre>
"""


# ---------------------------------------------------------------------------
# /lora — local landing page (no 302) with model card details.
# ---------------------------------------------------------------------------
_LORA_BODY = f"""
<h1>Trained <em>LoRA</em> adapter</h1>
<p class="lede">
  GRPO-tuned LoRA over Gemma-3n-E2B-it. Five reward components, three-stage
  curriculum (no drift → single drift → compound drift), 240 steps total
  on H100 80GB. Adapter-only — never the merged 16-bit weights, per
  DESIGN.md §10.5.
</p>

<table>
  <tbody>
    <tr><td class="k">repo</td><td class="path">DGXAI/gemma-3n-e2b-driftcall-lora</td></tr>
    <tr><td class="k">base model</td><td class="path">unsloth/gemma-3n-E2B-it</td></tr>
    <tr><td class="k">adapter type</td><td>peft / lora</td></tr>
    <tr><td class="k">r</td><td class="path">16</td></tr>
    <tr><td class="k">alpha</td><td class="path">32</td></tr>
    <tr><td class="k">dropout</td><td class="path">0.0 (Unsloth fast path)</td></tr>
    <tr><td class="k">trainer</td><td>scripts/train_driftcall_grpo.py · native PyTorch GRPO</td></tr>
    <tr><td class="k">curriculum</td><td>stage 1 (70 steps, no drift) → stage 2 (100 steps, single drift) → stage 3 (70 steps, compound)</td></tr>
    <tr><td class="k">num_generations</td><td class="path">2</td></tr>
    <tr><td class="k">size on disk</td><td>adapter_model.safetensors · 84.6 MB · plus tokenizer (33.4 MB)</td></tr>
    <tr><td class="k">precision</td><td>16-bit LoRA on bf16 base (H100 native)</td></tr>
    <tr><td class="k">license</td><td>apache-2.0</td></tr>
  </tbody>
</table>

<h2>Load it</h2>
<pre><code>from unsloth import FastModel
from peft import PeftModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    max_seq_length=4096,
    load_in_4bit=False,
    full_finetuning=False,
)
model = PeftModel.from_pretrained(model, "DGXAI/gemma-3n-e2b-driftcall-lora")
model.eval()</code></pre>

<h2>The five reward components</h2>
<table>
  <thead><tr><th>id</th><th>name</th><th>weight</th><th>description</th></tr></thead>
  <tbody>
    <tr><td class="path">R1</td><td class="path">task_completion</td>      <td>0.40</td><td>did the agent actually book the cab / settle the payment / hold the reservation</td></tr>
    <tr><td class="path">R2</td><td class="path">drift_detection</td>      <td>0.20</td><td>noticed the schema mutated mid-episode and retried with the new shape</td></tr>
    <tr><td class="path">R3</td><td class="path">constraint_adherence</td> <td>0.20</td><td>budget, time window, dietary, language — all checked deterministically</td></tr>
    <tr><td class="path">R4</td><td class="path">format_compliance</td>    <td>0.10</td><td>tool args parse against the (possibly drifted) JSON schema</td></tr>
    <tr><td class="path">R5</td><td class="path">anti_hack_penalty</td>    <td>0.10</td><td>200-episode reward-hacking probe set; pure deterministic checks, no LLM judge</td></tr>
  </tbody>
</table>

<p style="margin-top:2.5rem">
  <a class="btn" href="{LORA_HUB_URL}" target="_blank" rel="noopener">
    open on hf hub <span aria-hidden>↗</span>
  </a>
  <a class="btn btn-ghost" href="/source">project source</a>
</p>
"""


# ---------------------------------------------------------------------------
# /source — local landing page (no 302) with the project file tree.
# ---------------------------------------------------------------------------
_SOURCE_BODY = f"""
<h1>Project <em>source</em></h1>
<p class="lede">
  This Space bundles the entire DriftCall project — runtime modules,
  training scripts, the Colab-runnable notebook, design docs, the test
  suite, and the spec corpus. Browse the full file tree under the
  <a class="inline" href="https://huggingface.co/spaces/saumilyajj/driftcall/tree/main" target="_blank" rel="noopener">Files tab</a>
  of this Space, or open the canonical mirror on GitHub.
</p>

<div class="file-tree"><span class="dir">/ (HF Space root)</span>
├── <span class="dir">cells/</span>                   <span class="note"># importable runtime modules (used by the running app)</span>
│   ├── step_04_models.py            <span class="note"># DriftCallAction, DriftCallObservation, ActionType</span>
│   ├── step_05_vendors.py           <span class="note"># 5 mock vendor APIs</span>
│   ├── step_06_drift_injector.py    <span class="note"># 20-pattern drift catalogue</span>
│   ├── step_07_task_generator.py    <span class="note"># deterministic seeded task generation</span>
│   ├── step_08_rewards.py           <span class="note"># the 5 reward components + Brier + uncertain floor</span>
│   ├── step_09_audio.py             <span class="note"># Kokoro TTS + faster-whisper ASR</span>
│   ├── step_10_env.py               <span class="note"># DriftCallEnv (composes 04..09)</span>
│   ├── step_12_gemma_boot.py        <span class="note"># Unsloth model loader</span>
│   ├── step_13_grpo_config.py       <span class="note"># GRPOConfig builder</span>
│   ├── step_14_custom_trainer.py    <span class="note"># trainer + dataset adapter</span>
│   ├── step_15..17_train_stage*.py  <span class="note"># 3-stage curriculum</span>
│   ├── step_18_eval_baseline.py     <span class="note"># 50-ep baseline eval</span>
│   ├── step_19_eval_final.py        <span class="note"># 50-ep final eval (paired)</span>
│   ├── step_20_probe.py             <span class="note"># 200-ep reward-hacking probe</span>
│   ├── step_23_demo_gradio.py       <span class="note"># notebook variant of the demo</span>
│   └── step_24_deploy_hf.py         <span class="note"># push_lora_to_hub + push_*_space</span>
├── <span class="dir">data/</span>                    <span class="note"># briefs, drift patterns, API schemas (authored fixtures)</span>
├── <span class="dir">scripts/</span>                 <span class="note"># training scripts (bundled, not run on this Space)</span>
│   ├── train_driftcall_grpo.py      <span class="note"># native PyTorch GRPO loop (1300 LOC)</span>
│   ├── train_full_gemma3n.sh        <span class="note"># orchestrator: stage 1 → 2 → 3</span>
│   ├── smoke_gemma3n_boot.py        <span class="note"># one-off model boot smoke</span>
│   └── run_driftcall_train.py       <span class="note"># legacy TRL-based path (deprecated)</span>
├── <span class="dir">notebooks/</span>               <span class="note"># Colab-runnable artefact</span>
│   ├── train_driftcall.ipynb        <span class="note"># the 25-cell training notebook</span>
│   └── build_notebook.py            <span class="note"># jupytext concatenator</span>
├── <span class="dir">docs/</span>                    <span class="note"># 14 module specs + 14 test plans + critic transcripts</span>
│   ├── modules/                     <span class="note"># env.md, rewards.md, training.md, …</span>
│   ├── tests/                       <span class="note"># test plan docs</span>
│   ├── critics/                     <span class="note"># reviewer transcripts</span>
│   └── api_contracts/               <span class="note"># vendor API schema specs</span>
├── <span class="dir">tests/</span>                   <span class="note"># pytest suite (35+ files)</span>
├── <span class="dir">site/</span>                    <span class="note"># Vite-built React frontend (mounted at /)</span>
├── app.py                           <span class="note"># OpenEnv FastAPI server (786 LOC)</span>
├── demo_app.py                      <span class="note"># Gradio demo (mounted at /demo)</span>
├── unified_app.py                   <span class="note"># this Space's entrypoint</span>
├── openenv.yaml                     <span class="note"># OpenEnv v1.0 manifest</span>
├── Dockerfile · requirements.txt    <span class="note"># build configuration</span>
├── DESIGN.md                        <span class="note"># 54 KB design doc, 20 sections</span>
├── CLAUDE.md                        <span class="note"># development guide / project rules</span>
├── PROJECT_README.md                <span class="note"># top-level project README</span>
├── pyproject.toml                   <span class="note"># Python project metadata</span>
└── README.md                        <span class="note"># Space card (HF page renders this)</span></div>

<p style="margin-top:2.5rem">
  <a class="btn" href="{SOURCE_URL}" target="_blank" rel="noopener">
    open on github <span aria-hidden>↗</span>
  </a>
  <a class="btn btn-ghost" href="https://huggingface.co/spaces/saumilyajj/driftcall/tree/main" target="_blank" rel="noopener">
    browse files on hf
  </a>
  <a class="btn btn-ghost" href="/docs">openenv api docs</a>
</p>
"""


def _build_demo_blocks() -> Any:
    """Build the Gradio Blocks lazily so a missing dep at import time
    doesn't kill the whole FastAPI app — instead /demo will return 503
    with a clear message until the deps come back online."""
    try:
        from demo_app import build_ui  # type: ignore[import-not-found]
        return build_ui()
    except Exception as exc:  # noqa: BLE001
        # We log via FastAPI's normal startup logs; nothing more to do.
        import logging
        logging.getLogger("unified").exception("demo blocks build failed: %s", exc)
        return None


def build_unified_app() -> FastAPI:
    app: FastAPI = openenv_app

    @app.get("/openenv.yaml", include_in_schema=False)
    async def serve_manifest() -> Any:
        if MANIFEST_PATH.exists():
            return FileResponse(MANIFEST_PATH, media_type="text/yaml")
        return {"error": "openenv.yaml not found"}

    @app.get("/env", include_in_schema=False)
    async def serve_env_page() -> HTMLResponse:
        return HTMLResponse(_page("DriftCall — OpenEnv API", "/env", _ENV_BODY, "/env"))

    @app.get("/lora", include_in_schema=False)
    async def serve_lora_page() -> HTMLResponse:
        return HTMLResponse(_page("DriftCall — LoRA", "/lora", _LORA_BODY, "/lora"))

    @app.get("/source", include_in_schema=False)
    async def serve_source_page() -> HTMLResponse:
        return HTMLResponse(_page("DriftCall — source", "/source", _SOURCE_BODY, "/source"))

    # ── Live online RL — subprocess wrapper around scripts/train_driftcall_grpo.py.
    @app.get("/training", include_in_schema=False)
    async def training_status() -> Any:
        return get_online_trainer().status()

    @app.post("/training/start", include_in_schema=False)
    async def training_start() -> Any:
        return get_online_trainer().start()

    @app.post("/training/stop", include_in_schema=False)
    async def training_stop() -> Any:
        return get_online_trainer().stop()

    # Note: @app.on_event('startup') is a no-op here because app.py already
    # registers a custom `lifespan` context manager — the two APIs are mutually
    # exclusive. We instead kick off the trainer at module load below, after
    # `app = build_unified_app()` returns. The trainer's start() spawns a
    # subprocess and returns immediately, so it never blocks uvicorn boot.

    # Mount the Gradio voice demo at /demo — runs locally, no iframe.
    # Gradio mounts under /demo/ (with trailing slash). Register a
    # bare /demo handler that redirects to /demo/ so users typing the
    # short URL don't hit the SPA static catch-all and 404.
    @app.get("/demo", include_in_schema=False)
    async def demo_trailing_slash() -> RedirectResponse:
        return RedirectResponse(url="/demo/", status_code=308)

    blocks = _build_demo_blocks()
    if blocks is not None:
        try:
            import gradio as gr  # type: ignore[import-not-found]
            gr.mount_gradio_app(app, blocks, path="/demo")
        except Exception:
            import logging
            logging.getLogger("unified").exception("gr.mount_gradio_app failed")

    # SPA static mount — must come LAST so OpenEnv routes and our
    # explicit /env, /lora, /source, /demo handlers take precedence.
    if SITE_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=SITE_DIR, html=True),
            name="frontend",
        )

    return app


app = build_unified_app()

# Autostart the online GRPO trainer subprocess at module load.
# We can't use @app.on_event('startup') because app.py registers a custom
# `lifespan` context manager — the two APIs are mutually exclusive and the
# decorator silently no-ops. Module-level call runs once on Space boot,
# spawns a subprocess, and returns immediately so uvicorn isn't blocked.
if os.environ.get("DRIFTCALL_ONLINE_AUTOSTART", "1") == "1":
    try:
        get_online_trainer().start()
    except Exception:
        import logging
        logging.getLogger("unified").exception("trainer autostart failed")
