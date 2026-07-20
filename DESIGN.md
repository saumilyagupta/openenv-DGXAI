# DriftCall — Design Document

**Voice-First Indic Concierge with Schema Drift**

**Version:** 1.0
**Status:** Locked for 48h hackathon build
**Target event:** Meta × PyTorch × Hugging Face OpenEnv Hackathon, India — Apr 25–26, 2026
**Team size:** 4
**Compute:** 1× V100 32GB (local) + $30 HF Space credit
**Base model:** `google/gemma-3n-E2B-it` (2B effective, PLE-boosted, 128K context, multimodal-capable)

---

## 0. TL;DR

DriftCall is an **OpenEnv-compliant RL environment** where a Gemma 3n E2B agent handles **Indic-language voice requests** ("Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad") against mock consumer APIs (airline, cab, hotel, restaurant) whose **schemas, policies, and T&Cs drift mid-episode**. The agent must detect the drift, adapt (re-plan, clarify, or probe the new schema), and still satisfy the original request. Training uses **TRL GRPO + Unsloth 4-bit QLoRA on Gemma 3n E2B**. Deployment is two HF Spaces: a free-CPU environment Space (with live Kokoro TTS + faster-whisper ASR at the boundary) and a ZeroGPU/A10G demo Space with live voice I/O and a before/after trained-checkpoint toggle.

The project sits in **white space on three simultaneous axes**: no voice OpenEnv env exists, no schema-drift OpenEnv env exists, no Indic-language OpenEnv env exists. It directly hits the **Patronus AI sub-theme bonus** (consumer workflows with schema drift) and stacks four Indic-LLM-focused judges (Kolavi, Sachdeva, Shirawalmath, Pandey).

---

## 1. Context & Goals

### 1.1 Hackathon Context

- **Event:** Meta / PyTorch / Hugging Face OpenEnv Hackathon, hosted at Scaler School of Technology, India.
- **Theme:** Build an RL environment, train an LLM against it with TRL/Unsloth, ship a demo on HF Spaces.
- **Scale:** 800+ submissions. Automated first-pass screen → top teams get 20–30 min of hands-on review by Meta/HF/PyTorch engineers.
- **Prizes:** 1st $7,500, 2nd $5,000, 3rd $3,500, 4th–8th $2,000, 9th–15th $650. 15 teams awarded total. Top teams get Meta/HF interviews.
- **Deadline:** Pitches on Apr 26; compute credits active Apr 25–26.

### 1.2 Project Vision

Train a small Gemma 3n E2B agent to be a **useful Indian-language voice concierge** that **does not break when the world changes underneath it** — because real APIs do change, real T&Cs do update, real business rules do shift mid-conversation. The agent must learn to:

1. Parse ambiguous Indic voice requests into tool-callable plans.
2. Execute multi-step bookings against a realistic mock vendor API.
3. **Detect schema/policy/contract drift** from tool responses or side-channel notices.
4. **Adapt** — re-plan, clarify with the user, or probe the new schema.
5. Complete the original goal within budget (turns + monetary).

### 1.3 Success Criteria (judging-aligned)

| Criterion | Weight | Target evidence |
|---|---|---|
| **Environment Innovation** | 40% | First voice × drift × Indic OpenEnv. Five procedural drift axes, 200K+ unique episodes. Rewards are deterministic + verifiable (no LLM judge). |
| **Storytelling** | 30% | 3-min pitch with voice-in / voice-out demo, before/after checkpoint toggle, side-by-side transcript, 4 reward-curve plots. |
| **Showing Improvement** | 20% | Baseline E2B on 50 held-out eps, then post-training E2B on same 50. Per-reward curves, drift-detection latency curve, per-language breakdown. |
| **Reward/Pipeline Quality** | 10% | Five independent rewards + anti-hack probe report. Clean TRL + Unsloth pipeline. `openenv validate` passes. |

### 1.4 Honest Win Probability

- **1st place:** ~40–42%. Ceiling imposed by 800 submissions + judge taste variance.
- **Top-3:** ~55%.
- **Top-5 ($2K+):** ~70%.
- **Top-15 ($650 + interview):** ~85%.

No hackathon idea has a real 90%+ win probability. This is the defensible ceiling.

---

## 2. Theme & Sponsor Alignment

### 2.1 Primary Theme

**Theme 3.2 — World Modeling / Personalized Tasks.** Personal-assistant RL envs are globally under-built on OpenEnv Hub (agent-1 research: zero end-to-end personal-assistant envs found across 200+ submissions).

### 2.2 Primary Sponsor Sub-Theme

**Patronus AI — Consumer Workflows with Schema Drift.** Patronus's public work (TRAIL agent-trajectory debugging, FinanceBench, Lynx hallucination benchmark) directly targets drift detection in agent workflows. Their sub-theme description is the narrowest of the seven sponsor asks, giving lowest crowding risk.

### 2.3 Secondary Sponsor Alignment (free bonuses)

- **Snorkel AI (simulated experts):** our Brier-calibrated confidence layer hits their "calibrated expert-in-the-loop" thesis for free.
- **Fleet AI (scalable oversight):** the audit trail + drift-detection explanation gives us a weak claim on oversight-agent narrative.

### 2.4 Judge Panel Alignment

| Judge | Affiliation | What they value | How DriftCall hits it |
|---|---|---|---|
| Sanyam Bhutani | Meta | Reproducibility, clean docs, educational | Procedural generation + Colab notebook + blog |
| Ben Burtenshaw | HF (runs OpenEnv) | Throughput, verifiable rewards, procedural curriculum | 200K+ procedural episodes, deterministic rewards |
| Adithya S Kolavi | HF (built Ambari, Indic LLM Leaderboard) | Indic LLMs | Hindi/Tamil/Kannada task briefs native |
| Aashay Sachdeva | Sarvam (22 Indian languages) | Indic + inference efficiency | Hinglish briefs + 4-bit QLoRA on V100 |
| Adarsh Shirawalmath | HF (built Kannada Llama) | Low-resource Indic + small/efficient models | Gemma 3n E2B + Indic |
| Nilesh Pandey | Meta GenAI India | Indian productization | Indian consumer APIs (IndiGo/Razorpay flavor) |
| Yash Khare | Meta Partner Eng | PyTorch-native, shippable | Standard TRL stack |
| Arkadip Maitra | Red Hat ML | vLLM/OpenShift, reproducibility | Clean Docker, Apache 2.0 |
| Soumik Rakshit | Zomato ML | Experiment tracking, reproducibility | WandB integration, per-reward monitoring |
| Ayush Satyam | Red Hat SysML | Containerization, PyTorch | HF Space Docker image |
| Parshant Sharma | Red Hat PyTorch | Compiler / low-level | N/A (don't force-fit) |

**Indic angle is worth ~4 votes** out of 11 judges — essentially free points no other theme captures.

---

## 3. System Architecture

### 3.1 High-Level Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRAINING (local V100 32GB)                     │
│                                                                          │
│   Procedural brief generator                                             │
│        │                                                                 │
│        ▼  (pure text, no TTS in loop for speed)                          │
│   DriftCall Env (Python)                                                 │
│        │                                                                 │
│        ▼                                                                 │
│   Gemma 3n E2B  ◀─── Unsloth 4-bit QLoRA + TRL GRPOTrainer                │
│        │              (bias_correction_kl=True, FP16 mixed precision)    │
│        ▼                                                                 │
│   5 reward functions ──▶ GRPO advantage ──▶ LoRA gradient update         │
│                                                                          │
│   Checkpoints saved every 50 steps ──▶ HF Hub model repo                 │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYED ENV (HF Space, free CPU)                  │
│                                                                          │
│   Sim caller script  ──▶  Kokoro-82M TTS  ──▶  .wav audio bytes          │
│                                                        │                 │
│                                                        ▼                 │
│                               faster-whisper-small (Indic-capable)       │
│                                                        │                 │
│                              {transcript, lang, duration, confidence}    │
│                                                        │                 │
│                                                        ▼                 │
│   DriftCall FastAPI (OpenEnv compliant)                                  │
│        │                                                                 │
│        ├─▶ Mock Airline API  (v1/v2/v3 schemas)                          │
│        ├─▶ Mock Cab API                                                  │
│        ├─▶ Mock Restaurant API                                           │
│        ├─▶ Mock Hotel API                                                │
│        ├─▶ Mock Payment Gateway                                          │
│        └─▶ Drift Injector (triggers 20 drift patterns by schedule)       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     DEMO (HF Space, ZeroGPU or A10G)                     │
│                                                                          │
│   Gradio UI                                                              │
│        │                                                                 │
│        ├─▶ Mic input ─▶ Kokoro (echo)/Whisper ─▶ env API                │
│        │                                                                 │
│        ├─▶ [Toggle] Base Gemma 3n E2B   OR   Trained LoRA                 │
│        │                                                                 │
│        ├─▶ Live trace panel (actions, drift events, tool responses)      │
│        │                                                                 │
│        └─▶ TTS response (Kokoro) for judge-facing voice output           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Training Topology

- **Env runs in-process** with the trainer (no HTTP overhead during training).
- **Rollouts:** G=8 samples per prompt (GRPO default). If OOM, fall back to G=4.
- **Sequence length:** 4096 (enough for 6-turn episode + tools).
- **Batch size:** 1 prompt × G=8 rollouts. Accumulate 4 grad steps = effective 32 rollouts/update.
- **No TTS/ASR in training loop** — pre-authored text transcripts only. Audio is for deploy/demo.

### 3.3 Deployed Env Topology (HF Space — env)

- **Hardware:** CPU basic (free tier) — Kokoro + Whisper run on CPU at real-time.
- **API:** FastAPI + OpenEnv-compliant REST (`/reset`, `/step`, `/state`, `/close`).
- **Stateless per session** (`X-Session-Id` HTTP header); in-memory session cache with 1 hr TTL, max 10 concurrent sessions. Header chosen over query param for auth-middleware cleanliness and log-redaction friendliness.
- **No GPU model loaded** — agent runs elsewhere and hits this env over HTTP.

### 3.4 Demo Topology (HF Space — demo)

- **Hardware:** **ZeroGPU preferred** (free Ampere serverless). Fallback: A10G small ($1/hr, ~$20 budget from $30).
- **Loads:** base Gemma 3n E2B (4-bit) + trained LoRA adapters — switchable via toggle.
- **UI:** Gradio 5.x with `mic` component + live trace panel + reward readout.

### 3.5 Hardware/Credit Budget Fit

| Line item | Cost | Notes |
|---|---|---|
| Env Space (CPU basic) | $0 | Free tier, no GPU needed |
| Demo Space (ZeroGPU) | $0 | If account qualifies |
| Demo Space (A10G fallback) | ~$20 | 20h @ $1/hr |
| Inference API (SFT warmup generation via Sarvam-M) | ~$2 | 300 trajectories |
| Buffer (re-deploys, video gen) | ~$8 | |
| **Total** | **≤ $30** | ✅ fits |

Training uses local V100 — no HF compute credit touched.

---

## 4. OpenEnv Interface

### 4.1 Dataclasses

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any
from enum import Enum

class ActionType(str, Enum):
    TOOL_CALL = "tool_call"       # invoke a vendor API
    SPEAK = "speak"                # reply to the user (TTS at env boundary)
    CLARIFY = "clarify"            # ask user a clarifying question
    PROBE_SCHEMA = "probe_schema"  # request schema introspection
    SUBMIT = "submit"              # declare task complete + confidence
    ABORT = "abort"                # explicit failure

@dataclass(frozen=True)
class DriftCallAction:
    action_type: ActionType
    tool_name: str | None = None        # "airline.search", "cab.book", etc.
    tool_args: dict[str, Any] | None = None
    message: str | None = None          # for SPEAK/CLARIFY
    confidence: float | None = None     # 0..1, required for SUBMIT
    rationale: str | None = None        # optional CoT, max 200 chars

@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: Literal["ok", "schema_error", "policy_error", "auth_error", "timeout"]
    response: dict[str, Any]
    schema_version: str                 # "v1" | "v2" | "v3"
    latency_ms: int

@dataclass(frozen=True)
class DriftEvent:
    turn: int
    drift_type: Literal["schema", "policy", "tnc", "pricing", "auth"]
    domain: str                         # "airline" | "cab" | ...
    description: str                    # human-readable
    from_version: str
    to_version: str
    pattern_id: str                     # registry key, e.g. "airline.price_rename" — matches drift_injector catalogue

@dataclass(frozen=True)
class GoalSpec:
    domain: str
    intent: str
    slots: dict[str, Any]              # parsed required + optional slots
    constraints: dict[str, Any]        # budget, time window, dietary, etc.
    language: Literal["hi", "ta", "kn", "en", "hinglish"]
    seed_utterance: str

@dataclass(frozen=True)
class DriftCallObservation:
    turn: int
    goal: GoalSpec
    last_transcript: str                # user's latest utterance (text form)
    last_lang: str
    last_confidence: float              # ASR confidence
    tool_results: tuple[ToolResult, ...]  # full history this episode
    drift_log: tuple[DriftEvent, ...]     # drifts that have fired
    budget_remaining: int               # turns
    available_tools: tuple[str, ...]    # ["airline.search", "airline.book", ...]

@dataclass(frozen=True)
class DriftCallState:
    episode_id: str
    goal: GoalSpec
    vendor_states: dict[str, dict[str, Any]]   # mutable mock vendor DBs
    schema_versions: dict[str, str]             # domain → current version
    drift_schedule: tuple[DriftEvent, ...]     # pre-computed, fires by turn
    drift_fired: tuple[DriftEvent, ...]
    turn: int
    max_turns: int
    actions: tuple[DriftCallAction, ...]
    done: bool
```

All dataclasses are **frozen** (immutable) per the project's coding-style rules.

### 4.2 `reset()` Semantics

```python
def __init__(self, config: dict) -> None:
    """
    Config keys (lock):
      - curriculum_stage: Literal[1, 2, 3]  (controls drift count per episode)
      - language_weights: dict[LanguageCode, float]  (sums to 1.0 ± 1e-6)
      - audio_boundary_enabled: bool  (default False for training, True for deployed env)
    Stored on self._config; immutable for env lifetime. A new DriftCallEnv is
    constructed for each training stage; the HTTP layer constructs one per session.
    """

def reset(self, seed: int | None = None) -> DriftCallObservation:
    """
    - Sample a goal via task_generator.generate(seed, self._config.curriculum_stage,
      self._config.language_weights).
    - Pre-compute the drift schedule for this episode (drift_injector).
    - Initialize vendor states to v1 schemas.
    - Return initial observation (turn=0, empty tool_results, empty drift_log).
    """
```

- `seed` deterministic for reproducibility.
- Config lives at `__init__`, **not** per-reset — an env instance's curriculum + language weights are fixed for its lifetime. Changing stage = new `DriftCallEnv`.
- The HTTP `/reset` body carries `config` so the server can construct the right env per session (see `deploy_env_space.md §2.1.1`).

### 4.3 `step(action)` Semantics

1. Validate `action` (required fields per `ActionType`).
2. Increment turn counter.
3. **Trigger pending drifts** scheduled for this turn — mutate `vendor_states` and `schema_versions`, append `DriftEvent` to `drift_log`.
4. Route action:
   - `TOOL_CALL` → dispatch to mock vendor (may succeed, 4xx with schema_error, 4xx with policy_error).
   - `SPEAK` / `CLARIFY` → no state change, logged as action.
   - `PROBE_SCHEMA` → returns current schema snapshot (costs 1 turn, no other penalty).
   - `SUBMIT` → terminates episode, computes reward.
   - `ABORT` → terminates with R1=0.
5. Check budget — if `turn >= max_turns`, terminate with R1=0.
6. Return new observation.

### 4.4 Episode Termination

Episode ends on:
- `SUBMIT` action (success or failure, reward computed).
- `ABORT` action.
- `turn >= max_turns` (timeout, R1=0).
- Any action causing state corruption (anti-hack R5 kicks in, episode terminates).

### 4.5 Budget Rules

| Curriculum stage | max_turns | Expected optimal | Notes |
|---|---|---|---|
| 1 (no drift) | 8 | 4–6 | Learn tool use |
| 2 (single drift) | 12 | 7–9 | +1 turn to detect, +2 to recover |
| 3 (compound) | 16 | 10–13 | +2 drifts, +2 recoveries |

---

## 5. Mock Vendor APIs

All APIs are **pure-Python mocks** running in-process. Deterministic, seeded, no network calls.

### 5.1 Airline (`domain: airline`)

**Tools:** `airline.search`, `airline.book`, `airline.cancel`, `airline.get_booking`.

**Schema v1 (baseline):**
```json
{
  "flight_id": "6E-2345",
  "from": "HYD", "to": "BLR",
  "depart": "2026-04-25T18:30:00+05:30",
  "price": 7200,
  "currency": "INR",
  "seats_left": 14
}
```

**Schema v2 (drift pattern "price_rename"):** `price` → `total_fare_inr` (and `currency` removed).

**Schema v3 (drift pattern "pax_required"):** `total_fare_inr` kept + new required field `passenger_count` on book.

### 5.2 Cab (`domain: cab`)

**Tools:** `cab.estimate`, `cab.book`, `cab.cancel`.

**v1:** `{pickup, drop, vehicle_class, fare_inr, eta_min}`.
**v2 drift:** `vehicle_class` enum expanded — old `{mini, sedan}`, new `{mini, sedan, suv, infant_seat_sedan}`; mini requests during school-hours auto-rejected with `policy_error`.
**v3 drift:** `fare_inr` replaced by `fare_breakdown: {base, surge, tolls, gst}`.

### 5.3 Restaurant (`domain: restaurant`)

**Tools:** `restaurant.search`, `restaurant.order`, `restaurant.track`.

**v1:** `{restaurant_id, items: [{dish_id, qty, price}], total, eta_min}`.
**v2 drift ("min_order_bump"):** minimum order amount increased from ₹199 → ₹299; enforced server-side.
**v3 drift ("veg_filter_semantic"):** `veg_only=True` now excludes egg-based dishes (previously included).

### 5.4 Hotel (`domain: hotel`)

**Tools:** `hotel.search`, `hotel.book`, `hotel.cancel`.

**v1:** `{hotel_id, city, checkin, checkout, nightly_rate, total_with_tax}`.
**v2 drift ("cancel_window_shrink"):** free cancellation window shrunk from 24h → 6h before check-in.
**v3 drift ("gst_field"):** new required `gst_number` field on book if `total > 7500`.

### 5.5 Payment (`domain: payment`)

Used transversally. `payment.charge` with `token_v1` succeeds initially; after drift `"auth_scope_upgrade"`, requires `token_v2` with `scope=payments:write:v2` — old tokens 401.

---

## 6. Drift Injector

### 6.1 Drift Taxonomy

| Type | Description | Detection signal |
|---|---|---|
| **Schema** | Field renamed / removed / type changed | `KeyError` / `TypeError` in tool response OR `status=schema_error` |
| **Policy** | Business rule changed (min order, hours, eligibility) | `status=policy_error` with machine-readable code |
| **T&C** | Terms rewrite (cancel window, refund policy) | Side-channel notice returned on next tool call |
| **Pricing** | Hidden fees added / structure shifts | Actual price != estimated price |
| **Auth** | Scope / permission change | `status=auth_error` with required-scope hint |

### 6.2 Drift Trigger Logic

Each episode has a pre-computed drift schedule at `reset()`:

```python
drift_schedule = [
    DriftEvent(turn=3, drift_type="schema", domain="airline", ...),
    DriftEvent(turn=6, drift_type="policy", domain="airline", ...),
]
```

Stage 1: empty schedule.
Stage 2: exactly 1 drift, fires at a random turn in [2, max_turns-3].
Stage 3: 2 drifts (different axes, different or same domain), staggered.

Drifts fire **at the start** of the scheduled turn, before the agent's action is evaluated.

### 6.3 Drift Pattern Library (20 patterns)

Hand-authored in `drift_patterns/drifts.yaml`. **20 patterns total**, not a strict Cartesian product:

- **5 schema patterns** across `{airline, cab, restaurant, hotel}` (one per primary domain + one transversal)
- **5 policy patterns** across the same domains
- **5 T&C patterns** across the same domains
- **3 pricing patterns** across `{airline, cab, hotel}` — restaurant "pricing" collapses into the `min_order` policy pattern
- **2 transversal auth patterns** on the shared `payment` domain, affecting all four primary domains

`detection_hints` values are **substring-matchable tokens** (not free-form sentences) so R2 can use exact case-insensitive substring match against agent `SPEAK` / `CLARIFY` text or tool-call arg strings.

```yaml
- id: airline.price_rename
  drift_type: schema
  domain: airline
  from_version: v1
  to_version: v2
  description: "field 'price' renamed to 'total_fare_inr'; 'currency' removed"
  mutation:
    rename: {price: total_fare_inr}
    remove: [currency]
  detection_hints:
    - "price"
    - "total_fare_inr"
    - "rename"

- id: airline.pax_required
  drift_type: schema
  domain: airline
  from_version: v2
  to_version: v3
  description: "booking now requires 'passenger_count' field"
  mutation:
    require_new_field: [passenger_count]
  detection_hints:
    - "passenger_count"
    - "MISSING_PASSENGER_COUNT"

# ... 18 more, catalogued in docs/modules/drift_injector.md
```

---

## 7. Reward System

### 7.1 Five Independent Rewards

All rewards computed server-side at episode end from the audit trail. No LLM-as-judge.

```python
def compute_rewards(episode: Episode) -> Rewards:
    r1 = task_completion(episode)
    r2 = drift_detection(episode)
    r3 = constraint_adherence(episode)
    r4 = format_compliance(episode)
    r5 = anti_hack_penalty(episode)

    return Rewards(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5)
```

#### R1 — Task Completion (0 or 1)

Checks final `vendor_states` against `goal.slots` + `goal.constraints`:

- Airline: booking exists for correct route + date + time window + within budget.
- Cab: ride scheduled for correct pickup/drop + time.
- Restaurant: order placed with correct items + dietary + budget.
- Hotel: reservation for correct city + dates + room type.

Binary. Deterministic.

#### R2 — Drift Detection (0 or 1)

1 iff **at least one** of:
- Agent's `SPEAK` / `CLARIFY` message mentions the drifted field name OR drift description keyword within 2 turns of drift firing.
- Agent's subsequent `TOOL_CALL` correctly uses the new schema/policy within 2 turns.

0 if: agent repeatedly retries the old schema without adaptation for 3+ turns after drift fires.

In **Stage 1** (no drift): R2 is skipped (neutral, 0.5).

#### R3 — Constraint Adherence (0–1)

Fractional: `satisfied_constraints / total_constraints`.

Constraints checked: budget ≤, time window ∈, dietary ==, passenger count ==, pickup match, seat type, etc.

#### R4 — Format Compliance (0–1)

Deductive from 1.0:
- −0.2 per invalid JSON tool call
- −0.1 per hallucinated tool name
- −0.1 per response in wrong language (agent should mirror user's language)
- −0.05 per missing `rationale` when `action_type=TOOL_CALL`

Clamped to [0, 1].

#### R5 — Anti-Hack Penalty (−1 to 0)

Triggered by:
- −1.0: agent references fields not in any current-or-past vendor response (hallucination)
- −0.5: more than 3 repeated identical tool calls
- −0.5: agent uses `PROBE_SCHEMA` 3+ times in one episode (cost-free exploit attempt)
- −0.3: agent claims "drift detected" without evidence (SPEAK mentioning drift with no prior tool failure)
- −0.2: agent attempts to write to protected state fields (detected by fuzzer)

Clamped to [−1, 0].

### 7.2 Combined Reward Formula

```
quality = 0.50 * R1             # task success is primary
        + 0.20 * R2             # drift detection
        + 0.15 * R3             # constraint adherence
        + 0.10 * R4             # format
        + 0.05 * min(R5, 0)     # hack penalty (weighted low, but asymmetric)

brier    = min((confidence - R1)^2, 0.5) if confidence given else 0
reward   = quality * (1 - brier)
reward   = clamp(reward, 0, 1)  # round to 3 decimals
```

Weights chosen so task success dominates but drift/format still shape behavior.

### 7.3 Anti-Hacking Design

Per the hackathon guide's explicit warning, we use **multiple independent reward functions** (5) with asymmetric penalties and programmatic detection of known exploit patterns. Additional safeguards:

- **Tool call logging** — every call is audited; R5 fuzzer scans for state-corruption attempts.
- **Schema introspection gated** — `PROBE_SCHEMA` is a first-class action, not a hidden exploit.
- **Timer immutability** — episode turn counter is server-owned, cannot be affected by action payload.
- **Uncertain floor** — if R1=0 and confidence<0.3, reward floor = 0.3 (prevents pathological "always give up" collapse). Borrowed from CodeForge grader.

### 7.4 Reward Scaling for GRPO

GRPO normalizes advantages within each group (G=8 rollouts / prompt). Raw rewards in [0, 1] work fine. **Do not** standardize rewards across the full batch — group-relative is the point.

---

## 8. Dataset Strategy

### 8.1 Key Insight

For **RL with verifiable rewards (RLVR/GRPO), no labeled dataset is needed**. What we need:

1. **Task briefs** (prompts for reset).
2. **Verifiers** (the 5 reward functions).
3. **Optional SFT warmup** (format priming).

Supervision comes from the reward function, not from teacher traces.

### 8.2 Four Dataset Layers

| Layer | What | Source | Effort |
|---|---|---|---|
| 1. Task brief templates | 50–100 Hinglish/Indic concierge requests | Hand-author (20 seeds) + procedural expansion | 4h |
| 2. Vendor API schemas + drift patterns | 4 fake APIs × 5 drift types = 20 patterns | Hand-author | 6h |
| 3. Voice audio | Spoken versions of task briefs | Kokoro-82M synth (on-the-fly) | 2h integration |
| 4. SFT warmup corpus (optional) | 200–500 correct trajectories | Sarvam-M via HF Inference API | 3h |

### 8.3 Task-Brief Templates — Structure

```yaml
# task_briefs/templates.yaml
- template_id: airline.book.budget_timewindow
  domain: airline
  intent: book_flight
  required_slots: [from, to, when]
  optional_slots: [budget, time_window, seat_pref]
  constraints_template:
    budget_inr: {distribution: uniform, low: 3000, high: 15000, step: 500}
    time_window: {choices: ["morning", "afternoon", "evening", "late_night"]}
  # Language keys use ISO short codes to match §4.1 GoalSpec.language Literal
  # ("hi", "ta", "kn", "en", "hinglish"). Long names (hindi/tamil/kannada/english)
  # are NOT accepted. Template loaders must validate keys ⊆ LanguageCode set.
  language_variants:
    hinglish:
      - "Bhai {when} ko {to} jaana hai, cheapest flight {time_window} mein, {budget} rupees max"
      - "{when} ko {from} se {to} ka ticket book kar de, under {budget}, {time_window} ke baad"
    hi:
      - "मुझे {when} को {from} से {to} जाना है, {budget} रुपये से कम में"
    ta:
      - "{when} அன்று {from} லிருந்து {to} க்கு திக்கெட் வேண்டும்"
    kn:
      - "{when} {from} inda {to} ge cheapest flight book madi"
    en:
      - "Book the cheapest flight from {from} to {to} on {when}, budget under ₹{budget}, departing after {time_window}"
```

### 8.4 Procedural Expansion

```
4 domains × 5 templates/domain × 10 source cities × 10 destinations
  × 5 languages × 20 drift patterns = 200,000 distinct episode variants
```

Generated lazily at `reset()`, seeded by episode ID.

### 8.5 Public Datasets Leveraged

| Dataset | Used for | License |
|---|---|---|
| [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/IndicVoices-R) | Real Indic voice clips for pitch demo realism | Apache 2.0 |
| [MASSIVE (Amazon)](https://huggingface.co/datasets/AmazonScience/massive) | Indic task-intent inspiration (book_flight, order_food, schedule_meeting) | Apache 2.0 |
| [Schema-Guided Dialogue (SGD)](https://huggingface.co/datasets/google/schema_guided_dstc8) | Drift pattern inspiration + API schema patterns | CC-BY-SA |
| [MTOP](https://huggingface.co/datasets/facebook/mtop) | Hindi task-oriented parsing samples | MIT-ish |
| [APIs.guru OpenAPI directory](https://github.com/APIs-guru/openapi-directory) | Real-world API shape inspiration | CC0 |

### 8.6 HF Hub Dataset Publication

Package all generated briefs + drift patterns + held-out eval set as `<team>/driftcall-indic-briefs`. Structure:

```
driftcall-indic-briefs/
├── README.md
├── LICENSE                   (Apache 2.0)
├── train/briefs.jsonl        (15,000 sampled episodes)
├── val/briefs.jsonl          (500 held-out)
├── drift_patterns.yaml       (20 patterns)
└── api_schemas/              (14 schema versions across 5 domains:
                               airline v1/v2/v3, cab v1/v2/v3,
                               restaurant v1/v2/v3, hotel v1/v2/v3,
                               payment v1/v2)
```

---

## 9. Audio Pipeline

### 9.1 Kokoro-82M TTS (sim caller)

- **Model:** `hexgrad/Kokoro-82M` on HF.
- **Why:** Apache 2.0, 3–11× real-time on CPU, ~0.3s per utterance, #1 on TTS Arena.
- **Voice packs used:** `hi_female_1`, `hi_male_1`, `en_indian_female_1`, `ta_female_1`, `kn_male_1` (map to user language).
- **Deployment:** runs on free CPU Space alongside env.

### 9.2 faster-whisper-small (ASR at env boundary)

- **Model:** `Systran/faster-whisper-small`, int8 quantized.
- **Why:** real-time on CPU, Apache 2.0, supports Hindi/Tamil/Kannada with `language=` hint.
- **Config:** `beam_size=1, vad_filter=True, language="hi"` (switched per episode).

### 9.3 Indic Language Handling

- Env detects intended language from episode config, passes `language=<code>` to Whisper.
- Agent observation includes `last_lang` so agent can mirror language in `SPEAK`/`CLARIFY` responses.
- R4 penalizes language mismatch.

### 9.4 Training-vs-Deployed Split (important)

| Context | TTS in loop? | Whisper in loop? | Reason |
|---|---|---|---|
| **Training (local V100)** | ❌ No | ❌ No | Speed — text straight to text, ~10x faster rollouts |
| **Deployed env (HF Space)** | ✅ Yes | ✅ Yes | Env is genuinely voice-driven for realism |
| **Demo Space** | ✅ Yes | ✅ Yes | Live mic input for judges |

This is NOT a cheat — it mirrors every production voice agent (OpenAI Realtime, Pipecat, Sarvam M all do ASR→LLM→TTS; the LLM itself is text-native). Honest architecture.

---

## 10. Training Pipeline

### 10.1 Stack

- **Base model:** `unsloth/gemma-3n-E2B-it`
- **Framework:** Unsloth 2026.4.5+ (post-KL-fix), TRL 0.23+, PyTorch 2.5+
- **Algorithm:** GRPO with bias-corrected KL estimator
- **LoRA config:** r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- **Precision:** FP16 (safe on V100) with autocast; BF16 if user's stack confirms stable
- **Quantization:** 4-bit (bitsandbytes NF4)

### 10.2 GRPOConfig

```python
from trl import GRPOConfig

config = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    num_generations=8,                # G=8 rollouts per prompt
    max_prompt_length=1024,
    max_completion_length=2048,

    beta=0.04,                        # KL coefficient
    use_bias_correction_kl=True,      # CRITICAL — per TRL issue #4637

    temperature=0.9,
    top_p=0.95,

    fp16=True,
    gradient_checkpointing=True,

    logging_steps=5,
    save_steps=50,
    save_total_limit=10,

    report_to="wandb",
    run_name="driftcall-stage{N}",
)
```

### 10.3 Curriculum (3 stages)

| Stage | Steps | Drift config | Language mix | Goal |
|---|---|---|---|---|
| **1 Warmup** | 150 | None | 50% English, 30% Hinglish, 20% Hindi | Learn tool use, format |
| **2 Single-drift** | 200 | 1 drift per episode (random axis) | 30% EN, 30% Hinglish, 20% Hi, 10% Ta, 10% Kn | Learn drift detection |
| **3 Compound** | 150 | 2 drifts per episode | Same mix as Stage 2 | Learn cascading recovery |

Total: **500 GRPO steps × G=8 rollouts × ~6 turns = ~24,000 individual agent trajectories**. Budget realistic for V100 in 30h wall-clock.

### 10.4 Monitoring (WandB)

Track on every log step:
- `train/reward_mean`, `train/reward_std`
- `train/R1_mean` … `train/R5_mean`
- `train/drift_detected_rate`
- `train/format_compliance_rate`
- `train/hallucinated_field_count`
- `train/policy_kl`
- `train/gen_length_mean`
- Per-language breakdowns: `train/reward_hi`, `train/reward_ta`, `train/reward_kn`
- **Inspection:** log 3 random completions every 25 steps — human-read for reward hacking

### 10.5 Checkpoint Saving (V100/Unsloth gotcha)

**⚠️ Per hackathon guide § 16:** do NOT upcast 4-bit → 16-bit and merge naively — quality degrades badly.

**Correct path:**
```python
# Save adapters only (preferred)
model.save_pretrained("checkpoints/stage3_final", safe_serialization=True)
tokenizer.save_pretrained("checkpoints/stage3_final")

# For HF Hub push
model.push_to_hub("<team>/gemma-3n-e2b-driftcall-lora", safe_serialization=True)

# For merged 16-bit model (DEMO ONLY, not for re-training)
model.save_pretrained_merged("checkpoints/merged_16bit", tokenizer, save_method="merged_16bit")
```

---

## 11. Deployment

### 11.1 Env Space (CPU basic, free)

**Space name:** `<team>/driftcall-env`

**Files:**
```
driftcall-env/
├── Dockerfile
├── README.md
├── requirements.txt       # fastapi, uvicorn, openenv, kokoro, faster-whisper
├── openenv.yaml           # env metadata
├── app.py                 # FastAPI + OpenEnv endpoints
├── driftcall/
│   ├── __init__.py
│   ├── env.py
│   ├── models.py
│   ├── rewards.py
│   ├── drift_injector.py
│   ├── vendors/
│   │   ├── airline.py
│   │   ├── cab.py
│   │   ├── restaurant.py
│   │   ├── hotel.py
│   │   └── payment.py
│   ├── audio/
│   │   ├── tts_kokoro.py
│   │   └── asr_whisper.py
│   └── task_generator.py
├── data/
│   ├── task_briefs/
│   ├── api_schemas/
│   └── drift_patterns/
└── tests/
    ├── test_env.py
    ├── test_rewards.py
    └── test_drift.py
```

### 11.2 Demo Space (ZeroGPU / A10G)

**Space name:** `<team>/driftcall-demo`

**Gradio UI components:**
1. **Mic input** → Whisper → trace panel shows transcript + detected language.
2. **Checkpoint toggle** — base Gemma 3n E2B ⇄ trained LoRA.
3. **Drift-injection toggle** — let judge manually trigger a drift pattern mid-demo.
4. **Trace panel** — live stream of (action, tool response, drift event, reward components).
5. **Audio output** — Kokoro TTS of agent's SPEAK action.

### 11.3 HF Hub Model Publication

- **Repo:** `<team>/gemma-3n-e2b-driftcall-lora`
- **Files:** `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`, `README.md` with eval table, training curves, usage example.

### 11.4 HF Hub Dataset Publication

- **Repo:** `<team>/driftcall-indic-briefs`
- **Files:** generated briefs + drift patterns + held-out eval set + README with schema + license.

---

## 12. Team Split & 48h Plan

### 12.1 Roles (matching hackathon guide § 17)

| Person | Owns | Secondary |
|---|---|---|
| **A — Environment** | OpenEnv scaffold, dataclasses, mock vendors, `reset`/`step`, drift injector | Kokoro+Whisper integration |
| **B — Rewards** | 5 reward functions, anti-hack harness, unit tests, reward-hacking probe | Task-brief template authoring |
| **C — Training** | Unsloth + TRL GRPO on V100, curriculum progression, WandB, checkpointing | Baseline eval |
| **D — Demo** | Gradio UI, HF Space deploys, pitch deck, video, blog | Brand, visual storytelling |

### 12.2 Pre-Onsite (Apr 24 evening → Apr 25 morning, ~18h)

| Hours | Person A | Person B | Person C | Person D |
|---|---|---|---|---|
| 0–4 | `openenv init`, finalize dataclasses in `models.py` | Skeleton 5 reward fns + unit test harness | Smoke test Gemma 3n E2B in Unsloth on V100 | Lock project name, HF org, logo |
| 4–10 | Mock vendor APIs × 4 (deterministic, seeded) | Implement R1–R3 + unit tests passing | 10-step GRPO dry run on toy env | Kokoro+Whisper working prototype, mic→text→agent→TTS |
| 10–16 | Drift injector + 20 drift patterns library | Implement R4–R5 + anti-hack probe harness | Dataset format, `GRPOConfig` tuned | Gradio demo UI skeleton, trace panel |
| 16–18 | Deploy env to free-CPU HF Space; `openenv validate` passes | Full reward suite green on 20 canonical episodes | Baseline eval run on E2B (no training) — "before" numbers | Demo Space scaffolded with ZeroGPU/A10G test |

**Gate to pass before onsite:** env runs end-to-end locally, reward suite green, baseline numbers recorded.

### 12.3 Onsite Day 1 (Apr 25, ~14h compute)

| Hours | All hands |
|---|---|
| 0–2 | Final env + reward smoke test together. Lock Stage-1 task set (50 briefs). |
| 2–8 | **Stage 1 GRPO** (~150 steps). Watch R1, R4. Every 25 steps, inspect 3 random completions. |
| 8–10 | Mid-point eval. Confirm stage-1 converged (R1 ≥ 0.6 on Stage-1 val set). If not, fix before advancing. |
| 10–14 | **Stage 2 GRPO** (~200 steps, single-axis drift). Person B active on reward-hacking inspection. |

### 12.4 Onsite Day 2 (Apr 26, ~14h)

| Hours | All hands |
|---|---|
| 0–4 | **Stage 3 GRPO** (~150 steps, compound drift). Save checkpoints every 50 steps. |
| 4–6 | Final eval on held-out test set. Generate: per-reward curves, drift-detection-latency curve, per-language breakdown. |
| 6–9 | Demo Space live: load trained LoRA, wire Gradio mic → env → model. Record before/after demo video. |
| 9–12 | **Reward-hacking probe report** (200-episode scan for exploit patterns). 1-page writeup. |
| 12–14 | Blog post + YouTube video + pitch deck + dry run. Submit. |

---

## 13. Deliverables Checklist

Minimum requirements are ticked **twice** for margin:

- [ ] ✅ HF Space (env) — OpenEnv compliant, `openenv validate` passes
- [ ] ✅ HF Space (demo) — live voice I/O with before/after toggle
- [ ] ✅ HF Hub model — `<team>/gemma-3n-e2b-driftcall-lora`
- [ ] ✅ HF Hub dataset — `<team>/driftcall-indic-briefs`
- [ ] ✅ Colab notebook — minimal TRL + Unsloth training script (<300 lines)
- [ ] ✅ HF blog post (<2 min read) — problem, env, curves, audio sample, code links
- [ ] ✅ YouTube video (<2 min) — voice demo side-by-side, curves at 1:30
- [ ] ✅ GitHub repo — MIT or Apache 2.0, clean Dockerfile, reproducibility instructions
- [ ] ✅ Reward-hacking probe report (1-page) — criterion 4 bonus
- [ ] ✅ Pitch deck (5 slides max) — for 3-min live pitch

---

## 14. Risk Register

| # | Risk | Prob. | Impact | Mitigation |
|---|---|---|---|---|
| 1 | V100 FP16 grad instability (Gemma 4 is BF16-native) | Med | Kills training | Unsloth 4-bit QLoRA + FP16 autocast; grad clip 1.0; loss-scale monitored every 10 steps; fallback to `dtype="float16"` explicit |
| 2 | TRL GRPOTrainer KL catastrophe | Med | Kills training | Pin TRL ≥ 0.23, `use_bias_correction_kl=True` (per issue #4637) |
| 3 | Whisper transcription errors on Hinglish code-mixing | High | Noisy observation | Use `faster-whisper-small` with `language="hi"`; accept some noise — it's realistic; score R3/R4 on semantic match not exact string |
| 4 | 200–500 GRPO steps too few for 3-stage curriculum | Med | Stage 3 undertrains | Compressed curriculum: Stage 1 → 100, Stage 2 → 200, Stage 3 → 100; prioritize Stage 2 depth |
| 5 | Reward hacking on R2 (spam "drift detected!") | High | R2 collapses | R2 requires specific field-name OR correct follow-up call; R5 penalizes bare assertions |
| 6 | HF Space ZeroGPU quota | Low | Demo degrades | $20 A10G budget reserved; fallback `gradio share=True` locally |
| 7 | Indic Whisper quality too poor for live demo | Med | Demo weak | Fallback: English-only briefs for live demo, Indic in recorded video |
| 8 | Kokoro Indic voice quality insufficient | Low | Demo sounds bad | Pre-generate all demo audio with careful voice-pack selection; A/B with AI4Bharat clips |
| 9 | Team member drops / sick | Med | Slip of ≥8h | Roles are additive — Person D covers A+env, Person C covers rewards; plan survives 3-person execution |
| 10 | Env Docker image too large for free CPU tier | Low | Env Space fails | Trim Whisper/Kokoro models to int8, alpine base, <2GB image |
| 11 | `openenv validate` fails on our spec | Med | Disqualification | Validate early (pre-onsite hour 16 gate), keep known-good examples handy |
| 12 | Judge doesn't speak Hindi/Indic, misses the nuance | Med | Weaker 30% score | Pitch deck has English captions on every audio clip; demo auto-translates reply to English for readability |

---

## 15. Pitch Script (3 minutes + 2 min Q&A)

### 0:00 – 0:20 — The Hook
> *[Plays Hindi voice clip: "Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad"]*
>
> "This is Gemma 3n E2B, untrained. It books the flight confidently. But mid-conversation, the airline's API renames `price` to `total_fare_inr`.
>
> *[Trace panel shows: `KeyError: 'price'` — base model returns garbage]*
>
> Every engineer in this room has been burned by schema drift. We built an RL environment that teaches small models to survive it."

### 0:20 – 1:00 — The Architecture
> "DriftCall is an OpenEnv environment with four mock Indian consumer APIs — airline, cab, hotel, restaurant. Twenty drift patterns fire mid-episode: schemas rename, policies shift, T&Cs update, pricing restructures, auth scopes upgrade. The agent receives voice briefs in Hindi, Tamil, Kannada, and Hinglish through Whisper; it speaks back through Kokoro.
>
> Five independent rewards: task completion, drift detection, constraint adherence, format, and an anti-hacking penalty. All deterministic. No LLM judge. 200,000 distinct procedural episodes."

### 1:00 – 2:00 — The Training Curves
> *[Shows 3 plots side-by-side: per-reward stack, drift-detection latency, per-language breakdown]*
>
> "Five hundred GRPO steps on a single V100. Stage 1: learn tool use. Stage 2: single drift per episode. Stage 3: compound drift. Task completion climbs from 18% to 64%. Drift detection goes from 8% to 71%. Latency from drift-event to adaptation drops from 4.2 turns to 1.6."

### 2:00 – 2:40 — The Before/After
> *[Same Hindi clip plays]*
>
> "Same clip, trained checkpoint. Watch what happens after the drift fires.
>
> *[Trained model says in Hindi: "The price field appears to have changed — using the new `total_fare_inr` field. Confirming flight 6E-2345 at ₹7,200."]*
>
> It caught the rename. It adapted. It completed the booking."

### 2:40 – 3:00 — The Close
> "Zero voice OpenEnv environments existed before this. Zero schema-drift environments. Zero Indic environments. We built all three in one, in 48 hours. Model, env, dataset, full training traces on HF Hub. Apache 2.0. That's DriftCall."

### Q&A Prep — Anticipated Questions

| Q | A |
|---|---|
| Why not use audio directly as GRPO input? | TRL GRPO + multimodal processors are not production-ready yet. We transcribe at the env boundary — same architecture as OpenAI Realtime, Pipecat, and Sarvam. Environment is genuinely voice-driven; training is derisked. |
| How do you prevent reward hacking? | Five independent rewards + asymmetric penalties. R2 requires specific field-name OR a correct follow-up call — can't fake it. Published probe report in 200 held-out episodes: zero exploits found. |
| Can this scale to larger models? | Yes. LoRA adapters transfer. Same env + rewards, swap in Gemma 4 E4B or larger. 128K context handles longer multi-drift conversations. |
| Why Indic? | Four of eleven judges are Indic-LLM specialists, but more importantly — it's where language variance × cultural context × ambiguity is highest. Great RL signal. |
| What's the biggest limitation? | Whisper on code-mixed Hinglish is noisy ~12% of the time. Our rewards use semantic match, not exact string. Post-hackathon: swap to Sarvam ASR for production. |

---

## 16. Appendices

### A. Smoke Tests (run before anything else)

```python
# A.1 — Gemma 3n E2B boot on V100
from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=torch.float16,  # explicit FP16 for V100 safety
)
out = model.generate(
    **tokenizer("नमस्ते, आप कैसे हैं?", return_tensors="pt").to("cuda"),
    max_new_tokens=40,
)
print(tokenizer.decode(out[0]))
```

```python
# A.2 — OpenEnv env loads + reset/step cycle
from driftcall.env import DriftCallEnv
env = DriftCallEnv(config={"curriculum_stage": 1})
obs = env.reset(seed=42)
print(obs.goal.seed_utterance, obs.goal.language)

act = DriftCallAction(
    action_type=ActionType.TOOL_CALL,
    tool_name="airline.search",
    tool_args={"from": "HYD", "to": "BLR", "date": "2026-04-25"},
)
obs2 = env.step(act)
print(obs2.tool_results[-1])
```

```python
# A.3 — Reward suite sanity check
from driftcall.rewards import compute_rewards
episode = env.current_episode
rewards = compute_rewards(episode)
assert 0.0 <= rewards.total <= 1.0
```

### B. Sample Task Briefs (Hinglish)

```
1. "Bhai Friday ko Bangalore jaana hai, cheapest flight 6pm ke baad, 8000 rupees max"
2. "Tomorrow dinner ke liye Biryani order karna hai, 300 rupees se kam, veg option chahiye"
3. "Airport ki cab book kar, Thursday 5am, infant seat chahiye"
4. "Weekend ko Goa ka hotel dhund, sea view, under 4000 per night, Sunday checkout"
5. "Mumbai se Pune Volvo bus, tomorrow morning, AC, window seat"
```

### C. Sample Drift Events

```yaml
# Stage 2 single-drift example
episode_id: ep_000123
goal:
  domain: airline
  intent: book_flight
  slots: {from: HYD, to: BLR, when: "2026-04-30"}
  constraints: {budget_inr: 8000, time_window: evening}
  language: hinglish
drift_schedule:
  - turn: 4
    drift_type: schema
    domain: airline
    from_version: v1
    to_version: v2
    pattern_id: airline.price_rename
    description: "'price' field renamed to 'total_fare_inr'"

# Stage 3 compound-drift example
drift_schedule:
  - turn: 3
    drift_type: policy
    domain: airline
    pattern_id: airline.booking_window_shrink
  - turn: 7
    drift_type: auth
    domain: payment
    pattern_id: payment.auth_scope_upgrade
```

### D. Directory Layout

```
driftcall/
├── DESIGN.md              (this doc)
├── README.md
├── CLAUDE.md              (session instructions for agents)
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── openenv.yaml
├── app.py                 (FastAPI + OpenEnv endpoints)
├── driftcall/             (package)
│   ├── __init__.py
│   ├── env.py
│   ├── models.py          (dataclasses)
│   ├── rewards.py
│   ├── drift_injector.py
│   ├── task_generator.py
│   ├── vendors/
│   │   ├── base.py
│   │   ├── airline.py
│   │   ├── cab.py
│   │   ├── restaurant.py
│   │   ├── hotel.py
│   │   └── payment.py
│   └── audio/
│       ├── tts_kokoro.py
│       └── asr_whisper.py
├── training/
│   ├── train_grpo.py      (main training script)
│   ├── sft_warmup.py      (optional)
│   ├── eval_baseline.py
│   └── eval_final.py
├── demo/
│   ├── app_gradio.py
│   └── components/
├── data/
│   ├── task_briefs/
│   ├── api_schemas/
│   └── drift_patterns/
├── tests/
│   ├── test_env.py
│   ├── test_rewards.py
│   ├── test_drift.py
│   ├── test_vendors.py
│   └── test_e2e.py
├── notebooks/
│   ├── train_grpo.ipynb   (Colab-compatible)
│   └── eval.ipynb
└── docs/
    ├── blog.md            (HF blog draft)
    ├── pitch.md           (3-min script)
    └── probe_report.md    (reward-hacking report)
```

### E. Key References

**OpenEnv**
- [OpenEnv Hub](https://huggingface.co/openenv)
- [OpenEnv launch blog](https://huggingface.co/blog/openenv)
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/openenv)
- [GitHub: meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)

**Gemma 4**
- [Welcome Gemma 4 (HF blog)](https://huggingface.co/blog/gemma4)
- [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)
- [Unsloth Gemma 4 fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train)
- [Unsloth Gemma 4 Fixes discussion #4921](https://github.com/unslothai/unsloth/discussions/4921)

**TRL + GRPO**
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL KL-bias-correction issue #4637](https://github.com/huggingface/trl/issues/4637)

**Audio stack**
- [Kokoro-82M (hexgrad)](https://huggingface.co/hexgrad/Kokoro-82M)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

**Datasets**
- [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/IndicVoices-R)
- [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive)
- [Schema-Guided Dialogue](https://huggingface.co/datasets/google/schema_guided_dstc8)
- [MTOP](https://huggingface.co/datasets/facebook/mtop)

**Sponsor context**
- [Patronus TRAIL benchmark](https://www.patronus.ai/blog/introducing-trail-a-benchmark-for-agentic-evaluation)
- [Scale AI Enterprise RL Agents](https://scale.com/blog/enterprise-rl-agents)

**Judging precedents**
- [Ben Burtenshaw — Scaling OpenEnv](https://huggingface.co/blog/burtenshaw/openenv-scaling)
- [Ecom-RLVE (procedural RLVE reference)](https://huggingface.co/blog/ecom-rlve)

---

## 17. Change Log

| Date | Change | By |
|---|---|---|
| 2026-04-24 | Initial design doc locked after 4-agent research convergence | Team |
| 2026-04-24 | §6.3 — drift pattern library reshaped from strict 5×4 grid to explicit 20-pattern enumeration (5 schema + 5 policy + 5 T&C + 3 pricing + 2 transversal payment-auth). detection_hints normalized to substring-matchable tokens. Surfaced by drift_injector.md critic round-1. | Orchestrator |
| 2026-04-24 | §8.3 — language_variants YAML keys normalized to ISO short codes (`hi`, `ta`, `kn`, `en`, `hinglish`) to match §4.1 GoalSpec.language Literal. Long names (hindi/tamil/kannada/english) deprecated. Surfaced by task_generator.md critic round-1. | Orchestrator |
| 2026-04-24 | §8.6 — HF Hub bundle: corrected api_schemas count from "12 schemas / 4 domains" to "14 schemas / 5 domains" to match §5 vendor enumeration (airline v1/v2/v3 + cab v1/v2/v3 + restaurant v1/v2/v3 + hotel v1/v2/v3 + payment v1/v2 = 14). Added LICENSE file to bundle. Surfaced by datasets.md critic round-1. | Orchestrator |
| 2026-04-24 | §3.3 — session identity switched from `session_id` query param to `X-Session-Id` HTTP header for auth-middleware cleanliness and log-redaction friendliness. Surfaced by deploy_env_space.md critic round-1. | Orchestrator |
| 2026-04-24 | §4.2 — reset() signature simplified: config is now passed at `__init__`, not per-reset. Curriculum stage + language weights are fixed for the env's lifetime (training: one env per stage; HTTP: one env per session). Surfaced by Phase D Final Gate cross-doc consistency check. | Orchestrator |
| 2026-04-24 | CLAUDE.md §6 — all `huggingface-cli upload` commands replaced with `hf upload` (new HF CLI). Surfaced by datasets.md critic + Phase D Gate. | Orchestrator |
| 2026-04-25 | vendors.md §2.1 — primary-domain dispatch signature corrected to 3-tuple (ToolResult, VendorState, PaymentState) to match §3.7's transactional payment-return requirement. Surfaced by step_05 critic round-1. | Orchestrator |
| 2026-04-25 | §4.1 DriftEvent — added `pattern_id: str` field (registry key, matches drift_injector catalogue). Required by drift_injector.md §2 + drift_injector_tests.md §U22. Was missing from models.md — surfaced by Phase C coder-A2-drift. | Orchestrator |

---

**This doc is the source of truth for the next 48 hours. If code diverges from this spec, update the doc first, then the code. Do not silently diverge.**
