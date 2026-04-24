---
title: DriftCall Env
emoji: 🧭
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv — Indic voice concierge under schema drift.
license: apache-2.0
---

# DriftCall

DriftCall is a voice-first Indic consumer-concierge RL environment where
the agent must book flights, schedule rides, and file complaints across
five mock vendor APIs while those APIs undergo deterministic mid-episode
**schema drift**, **pricing drift**, **T&C drift**, **policy drift**, and
**auth drift**. It is an OpenEnv-compliant REST environment plus an
in-process Python trainer; a trained LoRA adapter for Gemma 4 E2B is
published alongside.

- **OpenEnv manifest:** [`openenv.yaml`](./openenv.yaml)
- **Design spec:** [`DESIGN.md`](./DESIGN.md)
- **Phase-C implementation plan:** [`CLAUDE.md`](./CLAUDE.md)
- **Per-module specs:** [`docs/modules/`](./docs/modules)
- **Per-module test plans:** [`docs/tests/`](./docs/tests)

## Architecture at a glance

- **Env Space (this repo):** FastAPI + OpenEnv REST on CPU-basic. Kokoro-82M
  TTS + faster-whisper-small ASR are baked into the image; no outbound
  network at runtime.
- **Trainer:** in-process GRPO (TRL 0.23+, Unsloth 2026.4.5+) on a single
  V100. Text-in / text-out — audio is an env-boundary concern.
- **Demo Space:** Gradio 5 on ZeroGPU, base Gemma 4 E2B + trained LoRA
  adapter switchable via a toggle.

## Quickstart

```bash
# 1. Install the dev toolchain.
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 2. Run the tests.
python3 -m pytest tests/ -v

# 3. Serve the env locally.
export DRIFTCALL_ENV_TOKEN=dev-local-token
uvicorn app:app --host 0.0.0.0 --port 7860

# 4. Validate against the OpenEnv schema.
openenv validate http://localhost:7860 --auth-bearer "$DRIFTCALL_ENV_TOKEN"
```

## Notebook

`notebooks/train_driftcall.ipynb` is built from the numbered cells under
[`cells/`](./cells). Rebuild with:

```bash
python3 notebooks/build_notebook.py
```

## License

Apache License 2.0. See [`LICENSE`](./LICENSE) (included at repo root when
this artifact is published as a Space).
