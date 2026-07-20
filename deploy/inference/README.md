# DriftCall Inference — OpenEnv Gym Client

Connects to the deployed env Space (`DGXAI/driftcall-env`) and runs inference
episodes with the trained LoRA (`DGXAI/gemma-3n-e2b-driftcall-lora`) loaded on
top of `unsloth/gemma-3n-E2B-it`.

## Layout

```
deploy/inference/
├── __init__.py        # exports DriftCallGymClient + GemmaPolicy
├── client.py          # OpenEnv REST client (gymnasium-style verbs)
├── policy.py          # base + LoRA model wrapper (lazy-loads on first .act())
├── run.py             # CLI: python -m deploy.inference.run …
├── requirements.txt   # extras on top of the env Space deps
└── README.md          # this file
```

## Quick start

```bash
# env vars
export DRIFTCALL_ENV_URL=https://dgxai-driftcall-env.hf.space
export DRIFTCALL_ENV_TOKEN=...   # the token configured on the Space
export HF_TOKEN=...              # only needed for a private adapter

# 1 episode against the deployed env, with the trained adapter
python -m deploy.inference.run \
    --num-episodes 1 --seed 42 --curriculum-stage 2 \
    --out-jsonl runs/inference.jsonl

# Baseline (no adapter)
python -m deploy.inference.run \
    --num-episodes 1 --seed 42 --curriculum-stage 2 \
    --adapter-id ""
```

## Programmatic use (gym-style)

```python
from deploy.inference import DriftCallGymClient, GemmaPolicy
from deploy.inference.policy import PolicyConfig

policy = GemmaPolicy(PolicyConfig(adapter_id="DGXAI/gemma-3n-e2b-driftcall-lora"))

with DriftCallGymClient() as env:
    obs, info = env.reset(seed=42, curriculum_stage=2)
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action).as_tuple()
        done = terminated or truncated
    print("reward:", info.get("episode_reward"))
```

## How it talks to the env Space

| Verb       | HTTP        | Behaviour |
|------------|-------------|-----------|
| `reset`    | `POST /reset` | Optional `seed`, `curriculum_stage` (1–3), `language_weights`, `audio_boundary_enabled`. |
| `step`     | `POST /step`  | Body `{"action": <DriftCallAction>}`. Returns 5-tuple. |
| `state`    | `GET /state`  | Read-only `DriftCallState` snapshot. |
| `close`    | `POST /close` | Evict server-side session. |
| `healthz`  | `GET /healthz`| Unauthenticated probe. |

Every mutating call carries `Authorization: Bearer <DRIFTCALL_ENV_TOKEN>` and
`X-Session-Id: <[A-Za-z0-9_-]{1,64}>`. Errors come back as the documented
envelope and are mapped to typed exceptions:

| Exception              | When |
|------------------------|------|
| `GymAuthError`         | M1 401 unauthorized |
| `GymSessionError`      | M2/M3/M4/M12 session issues |
| `GymCapacityError`     | M5 max_sessions / 429 |
| `GymClientError`       | everything else (network, 5xx, schema) |

## Why this lives in `deploy/`

`deploy/env_space/` is the **server side** (the Space). `deploy/inference/`
is the **client side** that actually exercises the deployed env with a
trained policy. Together they form the full OpenEnv loop:

```
deploy/inference/  ─── HTTP ───▶  deploy/env_space/  (HF Space)
   GemmaPolicy                    DriftCallEnv (FastAPI)
   + LoRA adapter                 + drift injector
   + base Gemma-3n                + reward grader
```

You can run `deploy/inference/run.py` against any OpenEnv-compliant env
that exposes the same surface — the client doesn't depend on local cells.
