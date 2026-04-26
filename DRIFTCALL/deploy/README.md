# DriftCall — Deployment Targets

Self-contained deployment subtree. After the trained LoRA is pushed to
[`DGXAI/gemma-3n-e2b-driftcall-lora`](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora),
this folder bundles **everything else** needed to ship DriftCall as live
HF Spaces and run inference end-to-end.

## Layout

```
deploy/
├── README.md             # this file
├── build_all.sh          # builds (and optionally pushes) every target
│
├── env_space/            ◀── OpenEnv REST env Space (Docker SDK)
│   ├── build.sh          #     rsyncs canonical app.py, cells/, data/, Dockerfile,
│   │                     #     openenv.yaml, requirements.txt → build/
│   ├── README.md         #     HF Space card (Docker SDK frontmatter)
│   ├── .gitignore
│   └── build/            #     [generated, gitignored] ready to push
│
├── demo_space/           ◀── Voice-first Gradio demo Space (Gradio SDK)
│   ├── build.sh          #     rsyncs canonical demo/app_gradio.py, cells/, data/ → build/
│   ├── README.md         #     HF Space card (Gradio SDK frontmatter)
│   ├── requirements.txt  #     gradio + faster-whisper + Kokoro + unsloth + peft
│   ├── .gitignore
│   └── build/            #     [generated, gitignored] ready to push
│
└── inference/            ◀── OpenEnv gym client + Gemma+LoRA policy (CLI)
    ├── __init__.py       #     exports DriftCallGymClient + GemmaPolicy
    ├── client.py         #     thin REST client (gymnasium-style verbs)
    ├── policy.py         #     base + LoRA policy (lazy-loads on first .act())
    ├── run.py            #     CLI: python -m deploy.inference.run …
    ├── requirements.txt
    └── README.md
```

## What lives where

| Component | Source | Hosted at |
|---|---|---|
| Trained LoRA adapter | (training output, pushed by `cells/step_24_deploy_hf.py`) | `DGXAI/gemma-3n-e2b-driftcall-lora` |
| OpenEnv env Space    | `deploy/env_space/build/` (built from repo root)        | `DGXAI/driftcall-env`              |
| Demo Space           | `deploy/demo_space/build/` (built from repo root)       | `DGXAI/driftcall-demo`             |
| Inference client     | `deploy/inference/` (no Space — runs locally)           | n/a                                |

The trained LoRA is **not** baked into either Space — both pull it from
HF Hub at runtime. This keeps Space images small and lets you iterate on
training without rebuilding the Spaces.

## One-shot deploy

```bash
# from the repo root
HF_TOKEN=hf_...  bash deploy/build_all.sh --push

# Override target repos via env
HF_SPACE_ENV_REPO=DGXAI/driftcall-env  \
HF_SPACE_DEMO_REPO=DGXAI/driftcall-demo \
HF_TOKEN=hf_...  bash deploy/build_all.sh --push
```

Or build / push targets individually:

```bash
bash deploy/env_space/build.sh --push       # only env Space
bash deploy/demo_space/build.sh --push      # only demo Space
```

## Build artifacts

Every `build.sh` creates a `build/` subdir under its target. Those dirs are
gitignored and regenerated on every run — they are **not** the source of
truth. The canonical sources stay at the repo root (`app.py`, `demo/app_gradio.py`,
`cells/`, `data/`, `Dockerfile`, `openenv.yaml`, `requirements.txt`).

## End-to-end loop

```
                       trained on RunPod / GPU box
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │  HF Hub: trained LoRA       │
                  │  DGXAI/gemma-3n-e2b-…-lora  │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  ▼                             ▼
       ┌────────────────────┐         ┌────────────────────┐
       │  env_space/        │         │  demo_space/       │
       │  HF Space (Docker) │         │  HF Space (Gradio) │
       │  /reset /step …    │         │  mic→ASR→…→TTS     │
       └────────┬───────────┘         └────────────────────┘
                │
                │  HTTP + bearer
                ▼
       ┌────────────────────┐
       │  inference/        │
       │  GemmaPolicy       │
       │  + DriftCallGym    │
       └────────────────────┘
```

## Validate before pushing

```bash
# OpenEnv manifest validation
openenv validate deploy/env_space/build/

# Smoke the gym client against the live env Space (after env_space is up)
DRIFTCALL_ENV_TOKEN=...  HF_TOKEN=...  \
    python -m deploy.inference.run --num-episodes 1 --seed 42 --curriculum-stage 2
```
