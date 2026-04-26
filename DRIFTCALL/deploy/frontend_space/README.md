---
title: DriftCall
emoji: 🌀
colorFrom: indigo
colorTo: pink
sdk: static
pinned: true
license: apache-2.0
short_description: Voice concierge under schema drift — project site
tags:
  - rl
  - voice
  - indic
  - schema-drift
  - openenv
  - gemma-3n
---

# DriftCall — project site

Static project site for [DriftCall](https://github.com/saumilyagupta/openenv-DGXAI),
the OpenEnv-compliant voice concierge env under mid-episode schema drift.
Built with Vite + React + TypeScript, vendored `@chenglou/pretext` for hero
text measurement, single saffron accent on dark editorial brutalism.

## What this Space is

Pre-built static HTML/CSS/JS only — no server, no GPU, no model. The page
embeds the live demo Space ([`DGXAI/driftcall-demo`](https://huggingface.co/spaces/DGXAI/driftcall-demo))
in an iframe and links out to the env Space and the trained LoRA.

## Sister artefacts

- 🤗 OpenEnv env Space — [`DGXAI/driftcall-env`](https://huggingface.co/spaces/DGXAI/driftcall-env)
- 🎙️ Voice demo Space — [`DGXAI/driftcall-demo`](https://huggingface.co/spaces/DGXAI/driftcall-demo)
- 🧠 Trained LoRA      — [`DGXAI/gemma-3n-e2b-driftcall-lora`](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora)
- 📦 Source            — [github](https://github.com/saumilyagupta/openenv-DGXAI)

## Build

The site is regenerated from canonical sources at the repo root via
`deploy/frontend_space/build.sh`:

```bash
cd DRIFTCALL
HF_TOKEN=hf_...  bash deploy/frontend_space/build.sh --push
```

The script runs `npm ci && npm run build` in `frontend/`, copies the
produced `dist/` into `deploy/frontend_space/build/`, drops this README
on top, and `hf upload`s the result. The upload is just static files —
no Docker, no Python, no runtime.
