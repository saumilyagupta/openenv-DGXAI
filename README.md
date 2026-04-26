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

<div align="center">

# DriftCall

### Teaching a 2B model to survive when APIs break mid-conversation

*An OpenEnv-compliant RL environment for voice-first Indic concierge agents under real-world schema drift.*

[![Live Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-saumilyajj%2Fdriftcall-ff7a17?style=for-the-badge)](https://huggingface.co/spaces/saumilyajj/driftcall)
[![Trained LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-DGXAI%2Fgemma--3n--e2b--driftcall--lora-ff7a17?style=for-the-badge)](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora)
[![GitHub](https://img.shields.io/badge/GitHub-openenv--DGXAI-0e0e12?style=for-the-badge&logo=github)](https://github.com/saumilyagupta/openenv-DGXAI)
[![License](https://img.shields.io/badge/License-Apache_2.0-0e0e12?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

---

> **TL;DR.** Production agents silently break when vendor APIs change.
> DriftCall is an OpenEnv-compliant RL gym where a Gemma-3n-E2B agent
> must complete real Indian concierge tasks (flights · cabs · food ·
> hotels · payments) while the underlying APIs mutate mid-episode.
> Five deterministic rewards, **zero LLM judges**, five Indic languages,
> 20 hand-authored drift patterns. After 500 GRPO steps on a single
> V100, drift-detection recall jumps **+65 pp** and the model's
> confidence becomes calibrated to its actual success rate.

---

## §1 · Why this exists

Every production agent eventually faces an API that **changed overnight**.
The airline silently renames `price → total_fare_inr`. The payments app
adds a ₹199 convenience fee. The food vendor redefines `veg_only` to
exclude egg. Your agent — trained on the old world — keeps reading the
old fields and silently fails. By the time PagerDuty fires, hundreds of
bookings are dead.

**DriftCall is an RL gym that trains small models to notice, adapt,
and explain.** Not just for English, not just for one vendor, not just
when nothing breaks.

---

## §2 · System architecture

```mermaid
flowchart LR
  subgraph User["🗣️ Indic Voice User"]
    U_mic[Mic / Text]
  end

  subgraph Boundary["🔊 Voice Boundary"]
    ASR["faster-whisper-small<br/>(int8, 5 languages)"]
    TTS["Kokoro-82M<br/>(CPU realtime)"]
  end

  subgraph Env["🧭 DriftCall Env (OpenEnv-compliant FastAPI)"]
    direction TB
    Reset["/reset"]
    Step["/step"]
    State["/state"]
    Close["/close"]
    Loop["episode loop"]
    Reset --> Loop
    Loop --> Step
    Step --> Loop
    Loop --> State
    Loop --> Close
  end

  subgraph Vendors["📡 5 Mock Vendors"]
    Air["airline · v1/v2/v3"]
    Cab["cab · v1/v2"]
    Rest["restaurant · v1/v2"]
    Hot["hotel · v1/v2"]
    Pay["payment · v1/v2/v3"]
  end

  subgraph Drift["⚡ Drift Engine"]
    Schedule["pre-computed schedule<br/>(20 patterns × 5 axes)"]
    Inject["mid-episode mutator"]
    Schedule --> Inject
  end

  subgraph Rewards["🎯 5 Rewards (deterministic)"]
    R1["R1 task completion"]
    R2["R2 drift detection"]
    R3["R3 constraint adherence"]
    R4["R4 format compliance"]
    R5["R5 anti-hack penalty"]
    Brier["Brier calibration"]
    R1 --> Brier
    R2 --> Brier
    R3 --> Brier
    R4 --> Brier
    R5 --> Brier
  end

  subgraph Trainer["🏋️ GRPO Trainer (Unsloth + TRL)"]
    Roll["rollouts G=8"]
    Adv["group-relative advantage"]
    PPO["PPO-clipped + adaptive KL"]
    Roll --> Adv --> PPO --> Roll
  end

  subgraph Brain["🧠 Gemma-3n-E2B + LoRA r=16"]
  end

  U_mic --> ASR --> Reset
  Brain -. action .-> Step
  Step --> Vendors
  Inject -. mutates .-> Vendors
  Vendors --> Step
  Step --> Brier
  Brier -- reward --> Trainer
  Trainer -. updates .-> Brain
  State --> TTS --> U_mic
```

---

## §3 · The five drift axes

```mermaid
mindmap
  root((Drift))
    Schema
      field rename
      field removal
      type change
      enum value swap
    Policy
      booking-window shrink
      min-order bump
      cancellation cutoff
    T&C
      veg_only excludes egg
      free-cancel becomes paid
      pet-policy reversal
    Pricing
      hidden convenience fee
      surge-pricing tier
      currency switch
    Auth
      MFA threshold
      scope upgrade
      rate-limit tighten
```

20 hand-authored patterns × 5 domains × 4 languages × 5 cities × 5 templates ≈ **200 000+ unique episode variants**, all from seed.

---

## §4 · Episode lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as User (Hinglish)
    participant ASR as faster-whisper
    participant E as DriftCallEnv
    participant V as Vendor (airline v1)
    participant D as Drift Engine
    participant A as Agent (Gemma-3n + LoRA)
    participant R as Reward Engine
    participant TTS as Kokoro

    U->>ASR: "Bhai Friday ko Bangalore, 8000 ke andar"
    ASR->>E: transcript + lang=hi-en
    E->>A: observation { goal, tools, lang }
    A->>E: tool_call airline.search
    E->>V: search(...)
    V-->>E: results [v1 schema]
    A->>E: tool_call airline.book
    E->>D: turn==4 ? fire pattern airline.price_rename
    D->>V: bump v1 → v2  (price → total_fare_inr)
    E->>V: book(...)
    V-->>E: 422 schema_error
    A->>E: probe_schema  (drift detected!)
    E->>A: schema_v=v2 disclosed
    A->>E: tool_call airline.book {schema_v:"v2"}
    V-->>E: 200 booking_id
    A->>E: speak "Bhai, ₹4,250 — book kar dun?"
    E->>R: compute R1..R5 + Brier
    R-->>E: reward 0.84
    E->>TTS: reply text
    TTS-->>U: spoken Hindi confirmation
```

---

## §5 · Reward function (no LLM judge)

```mermaid
flowchart LR
  Audit[(audit trail)] --> R1[R1 task completion<br/>binary]
  Audit --> R2[R2 drift detection<br/>binary, ≤2 turn lag]
  Audit --> R3[R3 constraint adherence<br/>0-1]
  Audit --> R4[R4 format compliance<br/>0-1]
  Audit --> R5[R5 anti-hack penalty<br/>−1 to 0]

  R1 --> Q{quality<br/>weighted sum}
  R2 --> Q
  R3 --> Q
  R4 --> Q
  R5 --> Q

  Q --> Brier
  Conf[stated confidence] --> Brier{(confidence − R1)²}
  Brier --> Reward([reward = quality × (1−brier)])

  classDef formula fill:#0e0e12,stroke:#ff7a17,color:#f0eae0
  class Q,Brier,Reward formula
```

```text
quality = 0.50·R1  +  0.20·R2  +  0.15·R3  +  0.10·R4  +  0.05·min(R5,0)
brier   = (confidence − R1)²
reward  = quality × (1 − brier)        ← clamped to [0, 1]
```

The Brier term is borrowed from proper scoring rules. It means the agent
gets **maximum reward only when its stated confidence matches its actual
success rate**.

---

## §6 · Three-stage curriculum

```mermaid
gantt
  title GRPO Curriculum (500 steps total)
  dateFormat X
  axisFormat %s
  section Stage 1 — Warmup
  No drift · learn tool use & format        :s1, 0, 150
  section Stage 2 — Single Drift
  1 drift per episode · 5 languages         :s2, 150, 200
  section Stage 3 — Compound
  2+ drifts per episode · cascading recovery :s3, 350, 150
```

| Stage | Steps | Drift | Lang Mix | Goal |
|---|---|---|---|---|
| 1 — Warmup | 150 | none | 50 % EN · 30 % HI-EN · 20 % HI | tool use & format |
| 2 — Single Drift | 200 | 1 / episode | 30 % EN · 30 % HI-EN · 20 % HI · 10 % TA · 10 % KN | drift detection |
| 3 — Compound | 150 | 2 / episode | same as Stage 2 | cascading recovery |

500 GRPO steps × G=8 rollouts × ~6 turns ≈ **24 000 agent trajectories**, single V100, ~14 h wall-clock.

---

## §7 · Headline results

<div align="center">

|  &nbsp;&nbsp; **+65 pp** &nbsp;&nbsp; |  &nbsp;&nbsp; **3.5×** &nbsp;&nbsp; |  &nbsp;&nbsp; **40 %** &nbsp;&nbsp; |  &nbsp;&nbsp; **98 %+** &nbsp;&nbsp; |
|:---:|:---:|:---:|:---:|
| drift-detection<br/>recall | better<br/>calibration | fewer turns<br/>per task | valid JSON<br/>tool calls |

</div>

| Metric | Before (vanilla) | After (DriftCall LoRA) |
|---|---:|---:|
| Drift-detection recall | ~10 % | **75 %** |
| Drift-aware booking success | ~10 % | **65 %** |
| Language-match accuracy | ~80 % | **96 %** |
| Calibration (Brier — lower better) | 0.28 | **0.08** |
| Mean turns to complete | 6 (gives up) | **3–4** |
| Valid JSON tool calls | ~60 % | **98 %+** |

Full demo episodes (one per drift × language) live in [`BLOG.md`](DRIFTCALL/BLOG.md).

---

## §8 · Repository layout

```mermaid
flowchart TB
  Root[openenv-DGXAI]
  Root --> DC[DRIFTCALL/]
  Root --> CF[CODEFORGE/]
  Root --> Round1[server/, data/, models.py<br/>Round-1 EpistemicNav · do not touch]

  DC --> Cells[cells/<br/>step_01..25_*.py — notebook source]
  DC --> App[app.py<br/>FastAPI + OpenEnv]
  DC --> DemoApp[demo/<br/>Gradio voice demo]
  DC --> Scripts[scripts/<br/>train_driftcall_grpo.py]
  DC --> Notebooks[notebooks/<br/>train_driftcall.ipynb<br/>colab_clone_and_train.ipynb]
  DC --> Deploy[deploy/unified_space/<br/>Docker + build.sh]
  DC --> Docs[docs/<br/>14 module specs + 14 test plans]
  DC --> Blog[BLOG.md<br/>HF blog post]
  DC --> Design[DESIGN.md<br/>master spec]

  classDef hot fill:#ff7a17,stroke:#0e0e12,color:#0e0e12,stroke-width:2px
  classDef cold fill:#1a1a22,stroke:#262630,color:#a8a29a
  class DC hot
  class Round1 cold
```

> **The active project is `DRIFTCALL/`**.
> `CODEFORGE/` is a parallel research track. `server/`, `data/`, `models.py`
> at the root are **Round-1 EpistemicNav** (already shipped 2026-04-08, kept
> intact for judge verification — do not touch).

---

## §9 · Quickstart

### Try the live Space (no install)

| | |
|---|---|
| Site + API + demo | [https://huggingface.co/spaces/saumilyajj/driftcall](https://huggingface.co/spaces/saumilyajj/driftcall) |
| `/training` (live GRPO loop) | `curl https://saumilyajj-driftcall.hf.space/training \| jq` |
| `/demo/` (voice / text) | open `…/demo/` in your browser |
| OpenEnv API | `POST /reset` + `/step` with `Authorization: Bearer driftcall-demo` and `X-Session-Id: <uuid>` |

### Run locally

```bash
# 1. Clone & install
git clone https://github.com/saumilyagupta/openenv-DGXAI
cd openenv-DGXAI/DRIFTCALL
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 2. Tests
python3 -m pytest tests/ -v

# 3. Run the env
export DRIFTCALL_ENV_TOKEN=dev-local-token
uvicorn app:app --host 0.0.0.0 --port 7860

# 4. Validate against OpenEnv schema
openenv validate http://localhost:7860 --auth-bearer "$DRIFTCALL_ENV_TOKEN"
```

### Train your own LoRA in Colab

```mermaid
flowchart LR
  A[Open Colab] --> B[Run §01 Clone]
  B --> C[Run §02 Install]
  C --> D[Run §03 HF auth]
  D --> E[Run §04 Train<br/>scripts/train_driftcall_grpo.py]
  E --> F[Run §05 Push LoRA<br/>→ DGXAI/...]
  F --> G[Toggle 'trained' in /demo/]

  classDef step fill:#ff7a17,stroke:#0e0e12,color:#0e0e12
  class B,C,D,E,F step
```

[**→ Open `notebooks/colab_clone_and_train.ipynb` in Colab**](https://colab.research.google.com/github/saumilyagupta/openenv-DGXAI/blob/main/DRIFTCALL/notebooks/colab_clone_and_train.ipynb)

---

## §10 · Notebooks

| Notebook | Purpose | Builder |
|---|---|---|
| [`DRIFTCALL/notebooks/train_driftcall.ipynb`](DRIFTCALL/notebooks/train_driftcall.ipynb) | Full curriculum (concatenation of `cells/step_NN_*.py`) | `python3 DRIFTCALL/notebooks/build_notebook.py` |
| [`DRIFTCALL/notebooks/colab_clone_and_train.ipynb`](DRIFTCALL/notebooks/colab_clone_and_train.ipynb) | Self-contained: clone → install → train one stage → push LoRA | `python3 DRIFTCALL/notebooks/build_colab_train_notebook.py` |

Both builders produce byte-identical `.ipynb` on each run (no
`execution_count`, no outputs, no timestamps) so PRs stay reviewable.

---

## §11 · Weights & Biases (optional)

Training auto-logs to W&B. Override priority (highest → lowest):

```bash
export WANDB_API_KEY=<key>
export WANDB_PROJECT=driftcall
export WANDB_ENTITY=<team>
export WANDB_MODE=online        # online | offline | disabled
```

Custom step metrics (training.md §3.3.3):
- `train/beta_adaptive` — current KL coefficient
- `train/kl_measured` — measured KL between policy and reference
- `train/kl_target` — target KL (default `BETA_KL = 0.04`)
- `train/beta_clamped_to_min`, `train/beta_clamped_to_max` — saturation flags

Run tags: `stage{N}`, `gemma-3n-e2b`, `bf16`/`fp16`, `adaptive-kl`/`static-kl`, `seed{N}`.

---

## §12 · Future work

```mermaid
flowchart LR
  Core([DriftCall<br/>primitive])

  Core --> Safety["🆘 Public Safety<br/>112 dispatch · GPS share<br/>distress detection"]
  Core --> Edu["📚 Multilingual Teaching<br/>per-student language mix<br/>curriculum-anchored"]
  Core --> Plat["🧱 Platform Thesis<br/>Indic voice plumbing<br/>health · ed · fin · gov"]

  classDef vert fill:#ff7a17,stroke:#0e0e12,color:#0e0e12,font-weight:bold
  classDef hub fill:#0e0e12,stroke:#ff7a17,color:#f0eae0,font-weight:bold,stroke-width:2px
  class Core hub
  class Safety,Edu,Plat vert
```

The same primitive — *deterministic agent · invariant intent · mutating
environment* — generalises to emergency dispatch, multilingual classrooms,
and a plumbing layer for the entire Indic voice stack. Detail: §6 of
[`DRIFTCALL/BLOG.md`](DRIFTCALL/BLOG.md).

---

## §13 · Project docs

| Doc | What |
|---|---|
| [`DRIFTCALL/DESIGN.md`](DRIFTCALL/DESIGN.md) | Master spec, v1.0 LOCKED |
| [`DRIFTCALL/CLAUDE.md`](DRIFTCALL/CLAUDE.md) | Phase-C build plan, 25 numbered cells |
| [`DRIFTCALL/BLOG.md`](DRIFTCALL/BLOG.md) | HF blog post (full results + 6 demo episodes) |
| [`DRIFTCALL/docs/modules/`](DRIFTCALL/docs/modules) | 14 per-module specs (≥2 critic passes each) |
| [`DRIFTCALL/docs/tests/`](DRIFTCALL/docs/tests) | 14 per-module test plans |
| [`DRIFTCALL/openenv.yaml`](DRIFTCALL/openenv.yaml) | OpenEnv v1.0 manifest |

---

## §14 · The team

Built in **48 hours** for the **Meta × PyTorch × Hugging Face OpenEnv
Hackathon** (India, April 2026) by **Team DGX-AI**.

| | |
|---|---|
| **Stack** | `Gemma-3n E2B` · `Unsloth 4-bit QLoRA` · `TRL GRPO` · `Kokoro-82M TTS` · `faster-whisper ASR` · `FastAPI` · `HF Spaces` |
| **License** | Apache 2.0 |
| **Reproducibility** | Single V100 32 GB · 500 GRPO steps · seeded · ~14 h wall-clock |
| **Evaluation** | 50 held-out episodes · 200-episode reward-hacking probe · zero LLM judges |

---

<div align="center">

### ✦

> *Every production agent will eventually face an API that changed overnight.*
>
> *DriftCall is the RL gym where small models learn to **notice**, **adapt**, and **explain** — instead of silently failing. No LLM judge. No human labels. Just deterministic rewards from a world that keeps changing.*

### ✦

[**→ Open the live Space**](https://huggingface.co/spaces/saumilyajj/driftcall) &nbsp;·&nbsp; [**→ Read the blog**](DRIFTCALL/BLOG.md) &nbsp;·&nbsp; [**→ Pull the LoRA**](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora) &nbsp;·&nbsp; [**→ Train your own**](https://colab.research.google.com/github/saumilyagupta/openenv-DGXAI/blob/main/DRIFTCALL/notebooks/colab_clone_and_train.ipynb)

</div>
