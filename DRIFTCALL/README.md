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

## Weights & Biases (optional)

Training runs auto-log to wandb. Configure via env vars (override priority
highest-to-lowest):

1. **Environment variables** — set on the host or in your shell:
   ```bash
   export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>
   export WANDB_PROJECT=driftcall              # default
   export WANDB_ENTITY=<your-team>             # optional
   export WANDB_MODE=online                    # online | offline | disabled
   ```
2. **`cells/_secrets.py` hardcoded fallback** — used when env vars are unset.
   Edit the constant in that file to rotate the key (private repo).
3. **None** — `init_wandb()` raises at run time if `WANDB_MODE != "disabled"`
   and no API key is reachable.

Disable for local dev / CI:
```bash
export WANDB_MODE=disabled
```

Custom metrics logged each training step (training.md §3.3.3):
- `train/beta_adaptive` — current KL coefficient (mutated by `AdaptiveKLCallback`)
- `train/kl_measured` — measured KL between policy and reference
- `train/kl_target` — target KL (default = `BETA_KL` = 0.04)
- `train/beta_clamped_to_min` — 1 if β was floored at `beta_min` this step
- `train/beta_clamped_to_max` — 1 if β was ceilinged at `beta_max` this step

Run tags (set at `wandb.init`): `stage{N}`, `gemma-4-e2b`, `bf16` or `fp16`,
`adaptive-kl` or `static-kl`, `seed{N}`.

## License

Apache License 2.0. See [`LICENSE`](./LICENSE) (included at repo root when
this artifact is published as a Space).
