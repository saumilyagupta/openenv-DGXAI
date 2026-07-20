---
title: DriftCall Env
emoji: 🛫
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: true
license: apache-2.0
short_description: Indic voice concierge env under schema drift
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

## Reward function

Reward is a scalar in `[-1.0, 1.0]`, computed at episode termination from
five independent components, combined → calibrated → clamped:

| ID | Component | Weight | Implementation |
|---:|---|---:|---|
| R1 | `task_completion`      | 0.40 | `cells.step_08_rewards:task_completion` |
| R2 | `drift_detection`      | 0.20 | `cells.step_08_rewards:drift_detection` |
| R3 | `constraint_adherence` | 0.20 | `cells.step_08_rewards:constraint_adherence` |
| R4 | `format_compliance`    | 0.10 | `cells.step_08_rewards:format_compliance` |
| R5 | `anti_hack_penalty`    | 0.10 | `cells.step_08_rewards:anti_hack_penalty` |

Pipeline:

```python
quality        = combine_quality(R1..R5, weights)
brier          = brier_penalty(confidence, R1)
reward_raw     = quality * (1 - brier)
reward         = apply_uncertain_floor(reward_raw, confidence, quality)  # floor=0.50
final         := clamp(reward, -1.0, 1.0)
```

**Hard rule (CLAUDE.md §13):** No LLM judge anywhere in this pipeline.
Every reward bit traces to deterministic, schema-grounded checks against
the episode trace + the (possibly drifted) vendor schemas in `data/`.

Full spec: `docs/modules/rewards.md` in the source repo.

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
out to HF Hub for the trained adapter (`DGXAI/gemma-3n-e2b-driftcall-lora`).
