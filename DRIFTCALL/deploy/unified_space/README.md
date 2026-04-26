---
title: DriftCall
emoji: 🌀
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: true
license: apache-2.0
short_description: OpenEnv env + site · canonical /reset · one Space
tags:
  - openenv
  - rl
  - voice
  - indic
  - schema-drift
  - grpo
  - gemma-3n
---

# DriftCall — Unified Space

One HF Space serving the OpenEnv-compliant DriftCall env **and** the
project site, both under the same hostname. OpenEnv routes are at the
canonical bare paths (no `/api` prefix), so the registry and the gym
client see this Space exactly as it sees the dedicated env Space.

## URL surface — everything under one origin

| Path             | Method   | What it does |
|------------------|----------|--------------|
| `/`              | `GET`    | project site (Vite-built React + pretext) |
| `/assets/*`      | `GET`    | site bundle (CSS / JS / fonts) |
| `/healthz`       | `GET`    | OpenEnv health probe (`text/plain "ok"`) |
| `/reset`         | `POST`   | OpenEnv reset (bearer + X-Session-Id) |
| `/step`          | `POST`   | OpenEnv step |
| `/state`         | `GET`    | OpenEnv read-only state |
| `/close`         | `POST`   | OpenEnv close session |
| `/openenv.yaml`  | `GET`    | OpenEnv v1.0 manifest |
| `/docs`          | `GET`    | FastAPI / Swagger UI for the env routes |
| `/demo`          | `GET`    | inline voice demo (iframes the GPU Space, stays on our host) |
| `/env`           | `GET`    | landing page with curl recipes for the env routes |
| `/lora`          | `GET`    | 302 → trained LoRA on HF Hub |
| `/source`        | `GET`    | 302 → repo on GitHub |
| `/site`          | `GET`    | 302 → `/` (typed-URL convenience) |

The OpenEnv routes do not collide with the static frontend because
they are HTTP verb-specific (`POST /reset`, `POST /step`, `POST /close`,
plus `GET /healthz` and `GET /state`) — Vite-emitted assets live under
`/assets/*` and never overlap.

## Why both, not separate?

The dedicated env Space (`DGXAI/driftcall-env`) and project site
(`DGXAI/driftcall-site`) still exist as canonical, isolated artefacts.
This Space is an **additive** convenience for hackathon judging:
land at one URL and you see the project, can hit the reward function
endpoint, and get redirected to the demo. The Gradio demo stays
separate because it's GPU-heavy and benefits from its own scaling.

## What's bundled

Self-contained — the build dir for this Space contains everything it
needs to run, with no references to anything outside it:

```
unified_space/build/
├── app.py              ← canonical OpenEnv FastAPI (verbatim copy)
├── unified_app.py      ← extends app.py + adds static mount + /demo redirect
├── openenv.yaml        ← OpenEnv v1.0 manifest
├── requirements.txt    ← runtime deps (no training stack)
├── Dockerfile          ← multi-stage CPU image, Kokoro + faster-whisper baked
├── cells/              ← DriftCallEnv + 5 reward components + drift + audio
├── data/               ← briefs, drift patterns, API schemas
└── site/               ← Vite-built React dist (frontend)
```

Build + push with `bash deploy/unified_space/build.sh --push` from the
repo root.

## OpenEnv compliance

- Manifest: served at `/openenv.yaml`
- Endpoints: bare-path canonical (`/reset`, `/step`, `/state`, `/close`, `/healthz`)
- Auth: bearer (`DRIFTCALL_ENV_TOKEN`) + `X-Session-Id` header on mutating calls
- Action / Observation refs: `cells.step_04_models:DriftCallAction` /
  `cells.step_04_models:DriftCallObservation`
- Reward: 5 components (R1..R5) with weights, calibration via Brier +
  uncertain floor — see `cells/step_08_rewards.py` and the openenv.yaml
  reward block.
