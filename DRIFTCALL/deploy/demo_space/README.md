---
title: DriftCall Demo
emoji: 🎙️
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: "5.8.0"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Indic voice concierge — base vs trained Gemma-3n LoRA
tags:
  - gradio
  - voice
  - asr
  - tts
  - indic
  - rl
  - lora
  - gemma-3n
models:
  - unsloth/gemma-3n-E2B-it
  - DGXAI/gemma-3n-e2b-driftcall-lora
---

# DriftCall Demo — Voice Concierge under Schema Drift

Live Gradio demo of the **DriftCall** RL environment. Speak a query in Hindi /
Tamil / Kannada / English / Hinglish; the agent (Gemma-3n-E2B + DriftCall LoRA)
plans across cab / hotel / airline / restaurant / payment vendors while
mid-episode drift hits the schemas.

## Pipeline

```
mic ──▶ faster-whisper ASR ──▶ DriftCallEnv ──▶ Gemma-3n + LoRA ──▶ Kokoro TTS ──▶ speaker
                                    ▲                  │
                                    │                  ▼
                            drift toggle        live trace panel
```

## Models

- Base: [`unsloth/gemma-3n-E2B-it`](https://huggingface.co/unsloth/gemma-3n-E2B-it)
- Trained adapter: [`DGXAI/gemma-3n-e2b-driftcall-lora`](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora)
- Switch in the UI between **base** (untrained) and **trained** to A/B compare
  the impact of the GRPO curriculum.

## Build / deploy

```bash
# from the repo root
bash deploy/demo_space/build.sh           # build deploy/demo_space/build/
bash deploy/demo_space/build.sh --push    # build + hf upload as a Space

# env vars
HF_SPACE_REPO  default: DGXAI/driftcall-demo
HF_TOKEN       required for --push
```

## Latency budget

- ZeroGPU warm: ≤ 8 s round-trip
- A10G warm:    ≤ 12 s round-trip

If a hardware probe falls outside the budget, the UI surfaces a `status_msg`
warning instead of crashing — the 9 documented error modes (5.1–5.9 in
`docs/modules/deploy_demo_space.md`) all return positional safe defaults.

## Sources

This Space is built from `deploy/demo_space/build.sh` which rsyncs the
canonical sources at the repo root:

- `demo/app_gradio.py`  — Gradio app (renamed to `app.py` in the build)
- `cells/`              — env, models, drift injector, audio, …
- `data/`               — briefs, drift patterns, API schemas
- `requirements.txt`    — Gradio + faster-whisper + Kokoro + unsloth + peft
