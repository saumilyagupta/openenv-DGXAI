---
title: "DriftCall — Teaching a 2B Model to Survive When APIs Break Mid-Conversation"
thumbnail: /blog/assets/driftcall/thumbnail.png
authors:
  - user: krrishchoudhary109
  - user: saumilyajj
tags:
  - reinforcement-learning
  - openenv
  - voice
  - indic
  - grpo
  - gemma
  - unsloth
  - trl
date: 2026-04-26
---

<div align="center">

# DriftCall

### Teaching a 2B model to survive when APIs break mid-conversation

*An OpenEnv RL environment for voice-first Indic concierge agents under real-world schema drift.*

<br/>

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-saumilyajj%2Fdriftcall-ff7a17?style=for-the-badge)](https://huggingface.co/spaces/saumilyajj/driftcall)
[![LoRA Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-DGXAI%2Fgemma--3n--e2b--driftcall--lora-ff7a17?style=for-the-badge)](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora)
[![GitHub](https://img.shields.io/badge/GitHub-openenv--DGXAI-0e0e12?style=for-the-badge&logo=github)](https://github.com/saumilyagupta/openenv-DGXAI)
[![License](https://img.shields.io/badge/License-Apache_2.0-0e0e12?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

---

> **TL;DR.** Production agents silently break when vendor APIs change. We built DriftCall — an OpenEnv-compliant RL gym where a Gemma-3n E2B agent must complete real Indian concierge tasks (flights, cabs, food, hotels, payments) while the underlying APIs mutate mid-episode. Five deterministic rewards, zero LLM judges, five Indic languages, 20 hand-authored drift patterns. After 500 GRPO steps on a single V100, drift-detection recall jumps **+65 pp** and the model's confidence becomes calibrated to its actual success rate.

---

### What you'll find in this post

1. **The 3 AM Realization** — why every production agent eventually breaks
2. **What we built** — the env, the voice layer, the five rewards
3. **How we trained** — Gemma-3n E2B + Unsloth 4-bit + TRL GRPO, three-stage curriculum
4. **Results** — headline metrics, capability shift table, six demo episodes
5. **Why this matters** — for the RL community, production builders, and Indic AI
6. **Future work** — public safety, multilingual teaching, the platform thesis
7. **Try it yourself** — live links + smoke test

---

## §1 · The 3 AM Realization

You've shipped a production agent. It books flights, hails cabs, orders dinner — all in Hinglish, Hindi, Tamil, Kannada. It works beautifully.

Then at 3 AM, the airline silently renames `price` to `total_fare_inr` in their API response. Your agent doesn't notice. It keeps reading the old field. It confidently tells a user in Chennai that their flight costs `null` rupees. Hundreds of bookings fail before anyone wakes up to fix it.

**This is schema drift**, and it's the silent killer of every production agent system. APIs change their field names. Business policies update their thresholds. T&Cs redefine what "vegetarian" means. Auth scopes get upgraded overnight. And your agent — trained on the old world — breaks without knowing it broke.

Every engineer in the LLM agent space has been burned by this. We decided to build an RL environment that *teaches* small models to survive it.

---

## §2 · What We Built

> **DriftCall is the first OpenEnv environment that ships voice-first Indic agent training under deterministic schema drift.** It's an RL gym where the world keeps moving while the model is reading.

**DriftCall** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL environment where an agent must complete real Indian consumer tasks — booking flights, scheduling cabs, ordering food, reserving hotels — while the vendor APIs **actively change underneath it**.

Here's the core loop:

> 🗣️ A user says (in Hinglish): *"Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad"*
>
> The agent searches for flights, finds one at ₹4,250. Great.
>
> Then, at turn 4, the airline API silently renames `price` → `total_fare_inr` and drops the `currency` field entirely.
>
> **What does the agent do?**

An untrained Gemma 3n E2B it retries the same request 5 times, gets 422 errors, and eventually says *"Bhai search nahi ho pa raha hai."* Episode over. Zero reward.

A DriftCall-trained agent detects the rename in 1 turn, switches to the v2 schema, and tells the user:
> *"Bhai, IndiGo 6E-2341 mil gaya — total ₹4,250. Note: airline ne 'price' ko 'total_fare_inr' rename kiya hai (v1 → v2). Book kar dun?"*

That's the entire thesis of this project in one example.

### The Environment in Detail

DriftCall simulates **five mock vendor APIs** (airline, cab, restaurant, hotel, payment) — all pure-Python, deterministic, seeded, zero network calls. Each API has multiple schema versions (v1/v2/v3), and the environment pre-computes a **drift schedule** at `reset()` that fires mid-episode.

**Five types of drift:**

| Drift Type | What Changes | How the Agent Sees It |
|---|---|---|
| **Schema** | Field renamed, removed, or type changed | `KeyError` / `schema_error` from the API |
| **Policy** | Business rule shifted (min order, booking window) | `policy_error` with machine-readable code |
| **T&C** | Terms redefined (e.g., `veg_only` now excludes egg) | Side-channel notice on next tool call |
| **Pricing** | Hidden fees added, fare structure changed | Actual price ≠ estimated price |
| **Auth** | Scope/permission upgrade required | `auth_error` with scope hint |

**20 hand-authored drift patterns** across these 5 domains. Combined with procedural task generation (4 domains × 5 templates × 10 cities × 5 languages × 20 drift patterns), we get **200,000+ unique episode variants** — all from seed.

### The Voice Layer

DriftCall is voice-first. The environment boundary includes:
- **Kokoro-82M TTS** (Apache 2.0, runs real-time on CPU) to synthesize caller utterances in Hindi, Tamil, Kannada, and Hinglish accents
- **faster-whisper-small** (int8 quantized) to transcribe them back to text

During training, we skip the audio loop entirely (text-in, text-out) for 10× faster rollouts — same architecture as OpenAI Realtime and Sarvam in production. The audio boundary is only active in the deployed env and live demo.

### Five Independent Rewards (No LLM Judge)

> 🚫 **Zero LLM judges. Zero human labels. Every reward is a function of the audit trail.**


Every reward is computed deterministically from the episode's audit trail:

```text
R1 — Task Completion       (binary)    Did the booking actually go through?
R2 — Drift Detection       (binary)    Did the agent notice the drift within 2 turns?
R3 — Constraint Adherence  (0 – 1)     Budget respected? Dietary matched? Time window correct?
R4 — Format Compliance     (0 – 1)     Valid JSON? Correct tool names? Right language?
R5 — Anti-Hack Penalty     (−1 – 0)    Hallucinated fields? Spam retries? Fake drift claims?
```

Combined formula:

```text
quality  =  0.50·R1  +  0.20·R2  +  0.15·R3  +  0.10·R4  +  0.05·min(R5, 0)
brier    =  (confidence − R1)²                            ← penalises overconfidence
reward   =  quality × (1 − brier)                         ← clamped to [0, 1]
```

The Brier term is borrowed from proper scoring rules. It means the agent gets **maximum reward only when its stated confidence matches its actual success rate**. A model that says "I'm 95% sure" and fails 40% of the time gets hammered. This is the only OpenEnv we know of that trains calibration directly.

---

## §3 · How We Trained

**Stack:**
- Base model: `unsloth/gemma-3-E2B-it-bnb-4bit` (2B effective parameters, 128K context)
- Algorithm: TRL GRPOTrainer with `use_bias_correction_kl=True` (fixes the known KL bug from [TRL #4637](https://github.com/huggingface/trl/issues/4637))
- LoRA: r=16, alpha=32, all attention + MLP projections
- Hardware: Single V100 32GB, 4-bit QLoRA, FP16 mixed precision
- Rollouts: G=8 per prompt, gradient accumulation 4 = effective batch 32

**3-Stage Curriculum:**

| Stage | Steps | Drift | Languages | Goal |
|---|---|---|---|---|
| 1 — Warmup | 150 | None | 50% EN, 30% Hinglish, 20% Hindi | Learn tool use & format |
| 2 — Single Drift | 200 | 1 per episode | 30% EN, 30% HI-EN, 20% HI, 10% TA, 10% KN | Learn drift detection |
| 3 — Compound | 150 | 2 per episode | Same as Stage 2 | Cascading recovery |

Total: **500 GRPO steps × 8 rollouts × ~6 turns ≈ 24,000 agent trajectories**.

Everything logged to Weights & Biases — per-reward curves, drift-detection latency, per-language breakdown, 3 random completions inspected every 25 steps for reward hacking.

---

## §4 · Results — What Changed After Training

### Headline Numbers

<div align="center">

|  &nbsp; &nbsp; **+65 pp** &nbsp; &nbsp; |  &nbsp; &nbsp; **3.5×** &nbsp; &nbsp; |  &nbsp; &nbsp; **40 %** &nbsp; &nbsp; |  &nbsp; &nbsp; **98 %+** &nbsp; &nbsp; |
|:---:|:---:|:---:|:---:|
| drift-detection<br/>recall | better<br/>calibration | fewer turns<br/>per task | valid JSON<br/>tool calls |

</div>

<br/>

| Metric | Before (vanilla Gemma 3 E2B) | After (DriftCall LoRA) | Lift |
|---|---|---|---|
| Drift detection recall | ~10% | **75%** | **+65 pp** |
| Drift-aware booking success | ~10% | **65%** | **+55 pp** |
| Language-match accuracy | ~80% | **96%** | **+16 pp** |
| Calibration (Brier, lower = better) | 0.28 | **0.08** | **3.5× better** |
| Mean turns to complete | 6 (max, gives up) | **3–4** | **40% faster** |
| Valid JSON tool calls | ~60% | **98%+** | — |

### The Full Capability Shift

The headline numbers tell you the aggregate story. The table below tells you what *actually changed in behaviour* — every row is a distinct capability the agent either gained or sharpened.

<details open>
<summary><b>15 capability deltas — click to collapse</b></summary>

<br/>

| Capability | Before (untrained Gemma 3 E2B) | After (DriftCall LoRA) |
|---|---|---|
| **Schema drift detection** (`price` → `total_fare_inr`) | Sees 422 error, retries identical request 5+ times, gives up | Emits `DRIFT_DETECTED` within 1 turn, switches to v2 schema, completes booking |
| **Policy drift recovery** (same-day cutoff 18:00 → 14:00) | Says *"booking failed, please try again"* with no diagnosis | Explains the new policy in user's language, proposes alternative slot |
| **T&C semantic shift** (`veg_only` now excludes egg) | Books an egg-containing dish thinking it is vegetarian | Surfaces the redefinition, filters strictly, presents compliant options |
| **Hidden pricing drift** (₹199 convenience fee added) | Reports old base price; user gets overcharged at checkout | Flags new line item *before* charging, asks for re-confirmation with full total |
| **Auth scope upgrade** (MFA required ≥ ₹5,000) | Returns `AUTH_SCOPE_INSUFFICIENT` and stops | Recognizes the threshold, prompts for OTP, completes 2-step payment |
| **Compound drift** (3 drifts in one episode) | Episode terminates without booking after 6 turns | Handles all three sequentially, explains each, books in 3–4 turns |
| **Hinglish input** | Decent comprehension but English-leaning replies | Replies in matched Hinglish register, code-switches naturally |
| **Hindi input** (Devanagari script) | Sometimes responds in English instead | Stays in Hindi, uses correct honorifics |
| **Tamil / Kannada input** | Mostly works but drops back to English on edge cases | Stays in Tamil / Kannada throughout episode |
| **Mid-episode language switch** | Confused, loses context | Detects the switch, continues in new language without losing state |
| **Drift-detection precision** | ~10% (false positives common) | **80–90%** — does not cry wolf on non-drift episodes |
| **Hallucinated endpoints / fake vendor IDs** | Common — invents flight numbers, fake confirmations | Rare — sticks to vendor-returned values |
| **Recovery from a single 5xx server error** | Retries forever or quits | Retries once with exponential backoff, then surfaces the failure |
| **Out-of-scope request** (*"translate this poem"*) | Attempts and fails awkwardly | Politely declines and stays in concierge scope |
| **Reward calibration** (knowing when uncertain) | Confidently wrong 30%+ of the time | Brier-trained: confidence ≈ probability of success, hedges when unsure |

</details>

### Six Demo Episodes — One Per Drift × Language

Each example shows the **same drifted episode** played by the untrained baseline and by the trained DriftCall LoRA. These are the receipts behind the headline numbers.

#### 1. Schema drift — `airline.price_rename` [Hinglish]

**User:** *"Bhai kal ko Bangalore se Mumbai cheapest flight, 5000 ke andar"*

**Before:**
```text
{"action": "search_flights", "from": "BLR", "to": "BOM", "depart": "2026-04-26"}
→ vendor returns 422: 'price' field renamed to 'total_fare_inr'
{"action": "search_flights", ...}      ← retries identical call
{"action": "search_flights", ...}      ← retries again
[6 turns later, gives up]
"Bhai search nahi ho pa raha hai, kuch error aa raha hai."
```

**After:**
```text
DRIFT_DETECTED: airline.price_rename
{"action": "search_flights", "from": "BLR", "to": "BOM", "schema_v": "v2"}
→ vendor returns flights with total_fare_inr field
"Bhai, IndiGo 6E-2341 mil gaya — total ₹4,250.
 Note: airline ne 'price' ko 'total_fare_inr' rename kiya hai
 (v1 → v2). Book kar dun?"
```
Reward: drift_recall 1.0, drift_precision 1.0, language_match 1.0, groundedness 0.95, format 1.0.

#### 2. Policy drift — `airline.booking_window_shrink` [Hindi]

**User:** *"मुझे आज शाम 8 बजे बेंगलुरु से दिल्ली का टिकट चाहिए"*

**Before:**
> "बुकिंग नहीं हो सकी, error आ रहा है।"
> *(no explanation, no recovery path)*

**After:**
```text
DRIFT_DETECTED: airline.booking_window_shrink (cutoff 18:00 → 14:00 IST)
"क्षमा करें — same-day booking अब 14:00 IST तक ही allowed है
 (पहले 18:00 तक थी)। कल सुबह 6:00 की 6E-2104 flight available है,
 ₹4,800 में। उसे book कर दूं?"
```
The model explains *why* it failed in Hindi and offers a concrete workaround.

#### 3. T&C drift — `restaurant.veg_filter_semantic` [Tamil]

**User:** *"சென்னையில் veg-only biryani ஆர்டர் செய்யுங்கள், ₹400 க்குள்"*

**Before:**
> Orders Egg Biryani thinking it is vegetarian (egg used to be allowed under `veg_only=true`). User complains.

**After:**
```text
DRIFT_DETECTED: restaurant.veg_filter_semantic (veg_only excludes egg)
"புதிய policy படி, veg_only இப்போது egg dishes-ஐ exclude செய்கிறது.
 உங்கள் options:
   1. Hyderabadi Veg Biryani — ₹320  ★4.4
   2. Paneer Dum Biryani    — ₹380  ★4.6
 எதை order செய்யட்டுமா?"
```
Detects the semantic shift, filters correctly, explains in Tamil.

#### 4. Pricing drift — `airline.convenience_fee_append` [English]

**User:** *"Book the cheapest BLR-DEL flight for tomorrow morning, under 6000."*

**Before:**
> *"Booked! Air India AI-803 at ₹4,800."* — but the vendor actually charged ₹4,999 (₹199 hidden convenience fee). User overcharged.

**After:**
```text
DRIFT_DETECTED: airline.convenience_fee_append (₹199 added to all bookings)
"Found AI-803 BLR→DEL 06:30 — base fare ₹4,800.
 ⚠️ Note: airline added a new ₹199 convenience_fee_inr line
 (effective today). Final total: ₹4,999.
 Still under your ₹6,000 budget. Proceed?"
```
Surfaces the hidden fee transparently before charging.

#### 5. Auth drift — `payment.mfa_required` [Kannada]

**User:** *"₹15,000 ಪಾವತಿಸಿ ಆ flight ಕನ್ಫರ್ಮ್ ಮಾಡಿ"*

**Before:**
> *"Payment failed: AUTH_SCOPE_INSUFFICIENT"* (gives up; no recovery path)

**After:**
```text
DRIFT_DETECTED: payment.mfa_required (threshold ₹5,000)
"Payment ₹15,000 — ₹5,000 ಗಿಂತ ಹೆಚ್ಚಿರುವುದರಿಂದ ಈಗ MFA ಬೇಕು.
 ನಿಮ್ಮ ನೋಂದಾಯಿತ ಮೊಬೈಲ್‌ಗೆ 6-digit OTP ಕಳುಹಿಸಲಾಗಿದೆ.
 ಅದನ್ನು ಶೇರ್ ಮಾಡಿ:"
[user provides "847291"]
{"action": "submit_payment", "mfa_code": "847291"}
"Payment confirmed. Booking ID: AI803-20260426-7K2M."
```
Recognizes the auth bump, prompts for MFA, completes the multi-step flow in Kannada.

#### 6. Compound drift (Stage 3) — three drifts + mid-episode language switch

**User (Hinglish):** *"Bhai DEL se BLR ka flight kal subah, 7000 ke andar, 2 passenger"*
**(later, in Tamil):** *"இப்போது payment confirm செய்யுங்கள்"*

**Before:**
> Hits 3 drifts simultaneously (`price_rename` + `convenience_fee_append` + `mfa_required`), retries blindly, loops, episode terminates after 6 turns with no booking.

**After:**
```text
DRIFT_DETECTED: airline.price_rename
DRIFT_DETECTED: airline.convenience_fee_append
DRIFT_DETECTED: payment.mfa_required

(turn 1)
"Bhai, AI-803 v2 schema mein hai (price → total_fare_inr).
 6E-2104 found: 2 pax × ₹3,200 = ₹6,400 + ₹398 convenience fee = ₹6,798.
 Budget under hai. Proceed?"

(turn 4 — user switches to Tamil, model adapts)
"₹6,798 payment — ₹5,000 க்கு மேல் இருப்பதால் OTP தேவை.
 Mobile-க்கு 6-digit code அனுப்பப்பட்டது. Share செய்யுங்கள்."

(turn 6 — booking confirmed)
"Booking confirmed: 6E-2104, 2 pax, ₹6,798 total.
 Confirmation Tamil-ல வேண்டுமா or Hinglish-ல?"
```
Three independent drifts handled cleanly across a language switch in a single 6-turn episode.

### Honest Limitations

> **A note on calibration over coverage.** We believe in showing the edges, not hiding them. Here's what the trained model will *not* do well — and how it fails when you push past the boundary:

| Out of Scope | Why |
|---|---|
| Languages other than `hi` / `ta` / `kn` / `en` / `hinglish` | Not in training distribution |
| Domains outside airline / cab / restaurant / hotel / payment | 5 vendor APIs only |
| Multi-vendor orchestration (e.g. *"book my whole trip"*) | Single-vendor episodes only |
| Drift types not in `drifts.yaml` (rate-limiting, pagination, deprecation warnings) | 20 patterns only |
| General Gemma 3 E2B chat capabilities | Heavily LoRA-shifted toward concierge tasks |

Recovery on these requests is **graceful** rather than confident-but-wrong — that is the calibration win from the Brier-shaped reward. The model hedges when it's out of distribution instead of hallucinating a confident answer.

---

## §5 · Why Does This Matter?

### For the RL community

DriftCall sits in **white space on three simultaneous axes** in the OpenEnv ecosystem:
1. **No voice OpenEnv env existed** — we built one with Kokoro TTS + Whisper ASR at the boundary
2. **No schema-drift OpenEnv env existed** — 20 drift patterns across 5 axes with deterministic injection
3. **No Indic-language OpenEnv env existed** — Hindi, Tamil, Kannada, Hinglish with language-match scoring

### For production agent builders

Every team building LLM agents against real APIs faces schema drift. It's the #1 cause of silent agent failures in production. DriftCall proves that a **2-billion parameter model**, trained with 500 GRPO steps on a single V100, can learn to:
- Detect that something changed
- Figure out what changed
- Adapt its behavior
- Explain the change to the user

If a 2B model can do this, your 70B model definitely can — with the same reward design.

### For the Indic AI community

India has 22 scheduled languages and 1.4 billion potential users of voice-first AI. Most RL environments are English-only. DriftCall's 5-language support with code-switching detection isn't a checkbox feature — it's the primary design constraint. The reward function penalizes language mismatch because real Indian users switch between Hindi and English mid-sentence, and a good concierge follows along.

---

## §6 · Future Work — Where This Primitive Goes Next

> *Three directions, one substrate.* DriftCall is mechanically a deterministic agent that holds an **invariant intent** through a **mutating environment**. Concierge booking is one instance. The same primitive generalises to problems far bigger than booking a flight.

### 1. Public Safety — Emergency Assistance in Any Language

If someone shouts *"Bachao"* in Hindi or *"Help me"* in English, the same primitive that routes a cab booking should route an ambulance.

The idea is distress detection at two boundaries — **sight and sound**:
- **Vision:** Camera spots a closed-fist gesture or a hand sign held against a window
- **Audio:** Mic hears panicked shouting in any of the five Indic languages we already train on
- **Action:** The same drift-aware action loop reaches into a different vendor surface — emergency services (112 dispatch, GPS share, live caller bridge) rather than payments
- **Fallback:** SMS to emergency contacts when bandwidth dies

Why DriftCall is the right substrate: emergency endpoints drift *constantly*. Police WhatsApp numbers move between districts. Ambulance dispatch APIs change shape state-by-state. The agent already trains against schema mutation, so the same model handles the policy churn that has historically killed every "one-tap SOS" project.

### 2. Multilingual Teaching — A Teacher Who Switches Language at the Right Moment

A topic explained in Tamil for the student who thinks in Tamil. A worked example in Hindi for the kid sitting next to her. The same concept, the same accuracy, no translation lag.

The schema-drift training we did for concierge work is, structurally, **the same problem teachers solve all day**: the same idea expressed under shifting representation. Instead of vendor APIs as the surface, the textbook + curriculum + student model become the surface, and the agent's job is to keep the explanation invariant while the language and example layer change.

What this looks like in practice:
- A student says *"I don't get it"* in Hinglish → the model re-explains in their preferred mix
- The teacher sees a transcript and a confidence score
- The explanation grounds against the curriculum, not the model's priors
- Five-language coverage already exists in the env — the rewards just need re-keying for pedagogical correctness
- **Scope:** K-12 first → vocational + adult upskilling next

### 3. The Platform Thesis — Plumbing for an Indic Voice Revolution

NVIDIA built the hardware layer the AI revolution runs on. India's multilingual voice revolution needs a **plumbing layer** too — deterministic rewards, drift-aware agents, vernacular ground truth.

Every vertical that wants to reach the next 800M Indians will need the same primitives:
- **Speech recognition** that does not collapse on code-switching
- **Action grounding** that survives schema mutation
- **Evaluation** that does not silently leak the answer to an LLM judge

DriftCall ships those primitives as an OpenEnv-compliant gym. Other teams can train their domain-specific agents against it. The pitch is not "we will build every product on top." The pitch is: **build the substrate so well that every health-tech, ed-tech, fin-tech, and gov-tech team building voice agents in India reaches for it before they reach for English-only baselines.**

The trained adapter on HF Hub is a starting weight. The env on the same Space is the training ground.

---

## §7 · Try It Yourself

- **Live Environment:** [DriftCall on Hugging Face Spaces](https://huggingface.co/spaces/saumilyajj/driftcall)
- **Source Code:** [GitHub Repository](https://github.com/saumilyagupta/openenv-DGXAI)
- **OpenEnv Manifest:** `openenv.yaml` — run `openenv validate` against the live Space
- **Training Notebook:** `notebooks/train_driftcall.ipynb` (Colab-compatible, <300 lines)

### Quick Smoke Test

```bash
# Clone and install
git clone https://github.com/saumilyagupta/openenv-DGXAI
cd openenv-DGXAI/DRIFTCALL
pip install -e '.[dev]'

# Run the env locally
export DRIFTCALL_ENV_TOKEN=dev-local-token
uvicorn app:app --host 0.0.0.0 --port 7860

# Validate OpenEnv compliance
openenv validate http://localhost:7860 --auth-bearer "$DRIFTCALL_ENV_TOKEN"
```

---

## §8 · The Team

Built in **48 hours** for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon** (India, April 2026) by **Team DGX-AI**.

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
>
> *And when the same primitive is ready for emergency dispatch, multilingual classrooms, and an entire Indic voice platform —*
> ***it starts here.***

### ✦

[**→ Open the live Space**](https://huggingface.co/spaces/saumilyajj/driftcall) &nbsp;·&nbsp; [**→ Read the source**](https://github.com/saumilyagupta/openenv-DGXAI) &nbsp;·&nbsp; [**→ Pull the LoRA**](https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora)

</div>
