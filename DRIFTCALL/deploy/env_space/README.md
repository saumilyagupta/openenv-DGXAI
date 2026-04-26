---
title: DriftCall Env
emoji: 🛫
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: true
license: apache-2.0
short_description: OpenEnv-compliant Indic voice concierge env under schema drift
tags:
  - openenv
  - rl
  - voice
  - indic
  - schema-drift
  - grpo
---

# DriftCall — OpenEnv Env Space

OpenEnv-compliant RL environment exposing **DriftCall**, a voice-first Indic
consumer concierge env under schema / policy / pricing / auth drift.

## REST surface (OpenEnv v1.0)

| Method | Path        | Purpose |
|--------|-------------|---------|
| `GET`  | `/healthz`  | Health probe (unauthenticated). |
| `POST` | `/reset`    | Create or recycle a session. |
| `POST` | `/step`     | Advance one turn. |
| `GET`  | `/state`    | Read `DriftCallState`. |
| `POST` | `/close`    | Evict a session. |

All mutating endpoints require:

```
Authorization: Bearer <DRIFTCALL_ENV_TOKEN>
X-Session-Id:  [A-Za-z0-9_-]{1,64}
```

Error envelope:

```json
{ "error": { "code": "<slug>", "message": "<str>", "request_id": "<asgi-id>" } }
```

`Cache-Control: no-store` on every response. Only `M5 max_sessions` carries
`Retry-After: 30`. No stack traces ever leak.

## Action / observation schemas

- Action:      `cells.step_04_models:DriftCallAction`
- Observation: `cells.step_04_models:DriftCallObservation`

Reward: scalar in `[-1.0, 1.0]`, decomposed into 5 components
(see `docs/modules/rewards.md` in the source repo).

## Episode params (passed in `/reset`)

| Field | Type | Range | Required |
|---|---|---|---|
| `seed` | int | — | no |
| `curriculum_stage` | int | 1–3 | no |
| `language_weights` | object | — | no |
| `audio_boundary_enabled` | bool | — | no |

`max_turns = 16` per episode.

## Build / deploy

```bash
# from repo root
bash deploy/env_space/build.sh           # builds deploy/env_space/build/
bash deploy/env_space/build.sh --push    # builds + uploads to HF_SPACE_REPO

# env vars
HF_SPACE_REPO  default: DGXAI/driftcall-env
HF_TOKEN       required for --push
```

## Sources

This Space is built from `deploy/env_space/build.sh` which rsyncs the
canonical sources at the repo root:

- `app.py`             — FastAPI / OpenEnv server (786 LOC)
- `cells/`             — importable modules (env, drift injector, rewards, …)
- `data/`              — authored fixtures (briefs, drift patterns, schemas)
- `Dockerfile`         — multi-stage CPU image; Kokoro + faster-whisper baked in
- `openenv.yaml`       — manifest validated by `openenv validate .`
- `requirements.txt`   — runtime deps (no training stack)

The model + LoRA adapter are **not** baked into the Space — eval calls reach
out to HF Hub for the trained adapter (`DGXAI/gemma-4-e2b-driftcall-lora`).
