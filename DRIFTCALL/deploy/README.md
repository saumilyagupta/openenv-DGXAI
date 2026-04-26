# DriftCall — Deployment Targets

This folder contains **deployment packages** for DriftCall. Each subfolder is a
self-contained build target for a Hugging Face Space (or other target).

| Target | Folder | What it deploys |
|---|---|---|
| **Env Space** (OpenEnv) | `env_space/` | FastAPI + OpenEnv REST (`/reset`, `/step`, `/state`, `/close`, `/healthz`). Judge-facing. |
| **Demo Space** (Gradio) | `../demo/` | Voice-first interactive demo (mic → ASR → env → Gemma+LoRA → TTS → speaker). |

## Why a separate folder?

The canonical sources live at the repo root (`app.py`, `cells/`, `data/`,
`Dockerfile`, `openenv.yaml`, `requirements.txt`). The `deploy/<target>/build.sh`
scripts rsync those canonical files into a `build/` directory and then `hf upload`
the result as a Space. Keeping `deploy/` separate from canonical sources means:

1. **No file duplication in git.** Only Space-specific extras (Space card
   `README.md`, build script, push hooks) live in `deploy/`. Canonical app code
   stays at the root, where the training/test stack imports it.
2. **One command to ship.** `bash deploy/env_space/build.sh --push` is the
   complete deploy pipeline — no manual `cp -r` dance.
3. **Build artifacts are gitignored.** `deploy/env_space/build/` is regenerated
   from canonical sources on every build; nothing under `build/` is checked in.

## Quick start

```bash
# Build + push the env Space
HF_TOKEN=hf_...  HF_SPACE_REPO=DGXAI/driftcall-env  \
    bash deploy/env_space/build.sh --push

# Build only (inspect what'll be uploaded)
bash deploy/env_space/build.sh

# Validate the manifest before pushing
cd deploy/env_space/build && openenv validate .
```

## OpenEnv spec compliance

The env Space follows OpenEnv v1.0 (`schema_version: "1.0"` in `openenv.yaml`).
Validate with:

```bash
openenv validate deploy/env_space/build/
```

Endpoints, error envelope, auth headers, and observation/action schemas are all
documented in `deploy/env_space/README.md` (which is also the HF Space card).
