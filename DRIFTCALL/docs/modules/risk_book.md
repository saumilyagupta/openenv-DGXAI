# risk_book.md — DriftCall Consolidated Risk Register

**Version:** v1.0
**Owner:** Person B (Rewards & Tests; CLAUDE.md §2.2)
**Primary source:** DESIGN.md §14 (12 risks) + CLAUDE.md §11 (escalation & stop conditions)
**Consumes from:** training.md §5, vendors.md §5, audio.md §5
**Consumed by:** env.md, training.md, audio.md, deploy_env_space.md, deploy_demo_space.md, pitch_demo.md, evaluation.md

---

## 1. Purpose

The **Risk Book** is the single consolidated risk register for DriftCall's 48-hour onsite execution (Apr 25–26, 2026) plus the ≈18 hours of pre-onsite doc + smoke work. It unifies three previously-separate concerns:

1. **DESIGN.md §14 static register** — 12 risks known at spec-lock time, each with P × I × mitigation.
2. **Module-level error modes** — concrete exceptions declared in `training.md §5`, `vendors.md §5`, `audio.md §5` that, when they fire at runtime, are RISK EVENTS this book owns routing for.
3. **CLAUDE.md §11 escalate / stop conditions** — the gates that halt agent dispatch or terminate the plan.

Without this consolidation every teammate would independently rediscover stop conditions, escalation paths, and who owns which mitigation. This doc centralizes the answer so that — under the hackathon clock — decisions are table-lookups, not debates.

**The Risk Book is live.** It is consulted at three moments:

- **Onsite start (hour 0 of Apr 25):** Person B runs `Risk.assess()`, reviews every entry's owner + trigger_signal with that owner, and blesses the register as the active contract.
- **Runtime (continuously):** the orchestrator and module owners call `Risk.triage(signal)` whenever a module-level error mode fires or an observable trigger is crossed. Triage returns the rooted mitigation + any escalation path.
- **Post-mortem (hour 48+):** the `RiskLog` (append-only) feeds the reward-hacking probe (DESIGN.md §13 deliverable #9) and the blog post (§15 Pitch).

**Goal:** every mitigation has a named owner and an observable trigger signal. No "we'll handle it when it comes up." If a risk fires and no one knows they own it, that is the design failure this book exists to prevent.

---

## 2. Interface

`driftcall/risk.py` (authored in Phase C by Person B) exposes the following. Phase D spec only — no code yet.

### 2.1 `RiskEntry` (frozen dataclass)

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

@dataclass(frozen=True)
class RiskEntry:
    id: str                       # "R01" ... "R15", "R01-STOP", "R06-STOP", "R99" (stable)
    description: str              # one-sentence human-readable risk
    probability: Probability      # Low | Med | High
    impact: Impact                # low | med | high | kills
    mitigation: str               # concrete action; cites module doc §
    owner: str                    # "A" | "B" | "C" | "D" | "orchestrator" | "team"
    trigger_signal: TriggerSignal # observable condition that fires mitigation
    stop_condition: Optional[StopCondition] = None  # None if not a hard-stop risk
    design_refs: tuple[str, ...] = ()                # e.g. ("DESIGN.md §14 #1", "training.md §7a")
```

### 2.2 Enums

```python
class Probability(str, Enum):
    LOW = "Low"
    MED = "Med"
    HIGH = "High"

class Impact(str, Enum):
    LOW = "low"            # nuisance; deliverable unaffected
    MED = "med"            # one deliverable degraded
    HIGH = "high"          # ≥ 2 deliverables at risk
    KILLS = "kills"        # training or demo cannot ship

class TriggerSignal(str, Enum):
    # Training-loop signals (mirrors training.md §5)
    GRAD_NORM_INF            = "grad_norm_inf"
    POLICY_KL_OVER_10        = "policy_kl_over_10"
    R5_DROP_WITH_HACK_SPIKE  = "r5_drop_with_hack_spike"
    CHECKPOINT_IO_ERROR      = "checkpoint_io_error"
    OOM_AT_G8                = "oom_at_g8"
    STAGE1_R1_BELOW_0_4_AT_100 = "stage1_r1_below_0_4_at_step_100"
    # Vendor signals (mirrors vendors.md §5)
    UNEXPECTED_STATUS_VALUE  = "vendor_status_not_in_five"
    ERROR_ENVELOPE_VIOLATION = "vendor_response_field_outside_catalogue"
    # Audio signals (mirrors audio.md §5)
    KOKORO_LOAD_FAILED       = "kokoro_model_load_error"
    WHISPER_LOAD_FAILED      = "whisper_model_load_error"
    INDIC_VOICE_PACK_MISSING = "indic_voice_pack_missing"
    # Infra / process signals
    V100_DOWN                = "v100_unreachable"
    V100_DOWN_OVER_8H        = "v100_unavailable_over_8h"
    HF_HUB_OUTAGE            = "hf_hub_or_spaces_5xx"
    HF_HUB_OUTAGE_OVER_2H    = "hf_hub_outage_over_2h"
    WANDB_STARTUP_FAIL       = "wandb_init_failed_nonoffline"
    DOCKER_IMAGE_OVER_2GB    = "docker_image_over_2gb"
    OPENENV_VALIDATE_FAIL    = "openenv_validate_failure"
    GEMMA4_SMOKE_FAIL        = "gemma4_e2b_smoke_test_failed"
    MERGE_CONFLICT_CROSS_OWNER = "merge_conflict_a_and_b_owned_files"
    TEAM_MEMBER_DROP         = "team_member_dropped"
    TEAM_3PERSON_BELOW_GATE  = "three_person_cannot_meet_phase_d_gate"
    JUDGE_LANGUAGE_MISMATCH  = "judge_does_not_speak_indic"
    # Meta
    UNKNOWN                  = "unknown"

class StopCondition(str, Enum):
    ESCALATE_TO_USER = "escalate"   # pause dispatch; ping user
    HARD_STOP        = "hard_stop"  # terminate plan; re-plan with user
```

### 2.3 Assessment + triage API

```python
class Risk:
    @staticmethod
    def assess() -> list[RiskEntry]:
        """Return the frozen register (§6 below, as data).
        Called once at onsite start by Person B. No side effects.
        Invariant: `len(assess()) >= 15` and every entry has a non-None owner."""

    @staticmethod
    def triage(signal: TriggerSignal, *, context: Optional[str] = None) -> TriageResult:
        """Look up the RiskEntry keyed by `signal`, return the handling plan.
        Idempotent. Never raises — if `signal == UNKNOWN` or no entry matches,
        returns a TriageResult pointing to R99 (unknown-class risk)."""

@dataclass(frozen=True)
class TriageResult:
    entry: RiskEntry
    action: str            # copy of entry.mitigation, ready to execute
    escalate: bool         # True iff entry.stop_condition == ESCALATE_TO_USER or HARD_STOP
    hard_stop: bool        # True iff entry.stop_condition == HARD_STOP
    log_line: str          # one-line summary for RiskLog append

class RiskLog:
    """Append-only audit log. One line per triage call during the 48h.
    Serialized to `risk_log.jsonl` at episode end; feeds reward-hacking probe."""
    def append(self, t: TriageResult, ts_iso: str) -> None: ...
    def to_jsonl(self, path: str) -> None: ...
    def summary(self) -> dict[str, int]:
        """{risk_id: fire_count} for the post-mortem report."""
```

### 2.4 Wiring points

| Caller | Trigger | How |
|---|---|---|
| `training/train_grpo.py` callback | `KLDivergenceExplosion`, `RewardCollapseError`, `CheckpointIOError`, `OutOfMemoryError` | `Risk.triage(signal)` then honor `hard_stop` / `escalate` |
| `driftcall/env.py` | `AudioDecodeError`, `ModelLoadError`, `UnsupportedLanguageError` | `Risk.triage(signal)` for logging; env already has local mitigations |
| `driftcall/vendors/*.py` | envelope-shape test fires | `Risk.triage(ERROR_ENVELOPE_VIOLATION)` — always hard_stop |
| Orchestrator (this CLAUDE session) | `openenv validate` fail, smoke-test fail, team drop | `Risk.triage(signal)` before deciding to escalate |
| `deploy_env_space.md` health check | HF Hub 5xx, Docker image > 2 GB | `Risk.triage(signal)` for the deploy runbook |

---

## 3. Behavior Spec

### 3.1 Invariants

1. **Minimum 17 entries.** `len(Risk.assess()) >= 17` — 12 from DESIGN.md §14 (R01–R12) + 3 mandated here (R13–R15) + R01-STOP + R06-STOP + R99 + room for post-onsite additions. Asserted in `tests/test_risk_book.py`.
2. **Every entry has a named owner.** Owner ∈ {A, B, C, D, orchestrator, team}. No `None`, no "TBD". Asserted.
3. **Every entry has an observable trigger signal.** Either a module error class (training.md §5, vendors.md §5, audio.md §5) or an explicit infra condition. No "we'll notice it." Asserted.
4. **Every CLAUDE.md §11 condition is represented.** Both escalation items (5) AND hard-stop items (3) appear as `StopCondition`-bearing entries. Asserted by a coverage test.
5. **Mitigations cite their source.** `design_refs` is non-empty for every entry. This lets a reviewer trace any mitigation back to the canonical spec.

### 3.2 Triage loop (runtime)

```
on signal s fired at time t:
    1. triage = Risk.triage(s)
    2. append triage to RiskLog with ts=t
    3. if triage.hard_stop:
           orchestrator halts all Agent dispatch
           SendMessage team-lead: "HARD_STOP fired for {triage.entry.id}: {triage.log_line}"
           await user intervention
    4. elif triage.escalate:
           orchestrator pauses new dispatch in affected scope
           SendMessage team-lead: "ESCALATE {triage.entry.id}: {triage.log_line}"
           continue other independent work
    5. else:
           owner executes triage.action in their worktree
           orchestrator continues
```

### 3.3 Escalation path (from CLAUDE.md §11 verbatim)

| Signal | Scope | First responder | Escalates to |
|---|---|---|---|
| Gemma 4 E2B smoke fails on V100 | training | Person C | User (block; may downshift to Gemma 3 4B) |
| `openenv validate` fails 3× | deploy | Person D | User (block; may need schema deviation approval) |
| Stage-1 R1 < 0.4 at step 100 | training | Person C → Person B | User (curriculum/reward redesign required) |
| Critic flags consistent DESIGN.md flaw | any phase | Orchestrator | User (update spec before continuing) |
| Merge conflict across A-owned and B-owned files | any phase | Orchestrator | User — ownership was wrong (process fix) |
| V100 unavailable > 8h during onsite | training | Person C | **Hard stop** — re-plan with user |
| HF Hub / Spaces outage > 2h | deploy | Person D | **Hard stop** |
| Team member drops AND 3-person cannot meet Phase D gate | all | Orchestrator | **Hard stop** |

### 3.4 Decision thresholds

| Impact | Probability | Action |
|---|---|---|
| `kills` | any | Immediate surface on first trigger; mitigation is non-negotiable |
| `high` | `High` or `Med` | Immediate surface; mitigation inline |
| `high` | `Low` | Pre-stage mitigation; surface on first trigger |
| `med` | `High` | Pre-stage mitigation; surface on repeat |
| `med` | `Med` or `Low` | Defer to next batch boundary; log only on first trigger |
| `low` | any | Defer; log only |

"Surface" = `SendMessage team-lead` + `RiskLog.append`. "Defer" = `RiskLog.append` only, no interruption.

### 3.5 Retry semantics

Mitigations that retry (e.g., HF Hub upload with 3× backoff) do so at their own layer (training.md §5 `CheckpointIOError`). This book does NOT implement retries — it routes the *final* failure after retries are exhausted. A `CheckpointIOError` in `Risk.triage` means 3 retries already failed.

### 3.6 Post-mortem

At hour 48, Person B runs `RiskLog.summary()`. Any risk that fired ≥ 3× is a candidate spec bug — update DESIGN.md §14 + this doc in the post-hackathon merge (not during the 48h).

---

## 4. Data Structures

### 4.1 `RiskEntry` — frozen, immutable, append-once at `assess()`.

Serialization: `to_dict()` / `from_dict()` round-trip stable. `id` is the primary key; changing any other field requires a new entry (versioning by file revision, not in-memory mutation). Enforced by `frozen=True`.

### 4.2 `RiskLog` — append-only in-memory list wrapped as immutable view per append.

- Storage format: JSONL (one `TriageResult.to_dict()` per line).
- Path: `logs/risk_log.jsonl` within the run directory (DESIGN.md §13 deliverable #6 traces).
- Never truncated during a run. A new run writes a new file (run_id prefix).

### 4.3 `TriggerSignal` — string Enum; stable wire format.

Adding a new signal requires (a) a new `RiskEntry` with that signal OR routing to R99 (unknown), and (b) a new line in the §2.2 enum above. Removing a signal is forbidden during the 48h.

### 4.4 `StopCondition` — three-valued sum type (`None | ESCALATE | HARD_STOP`).

`None` is the default (most risks have a local mitigation, not an escalation). `ESCALATE` and `HARD_STOP` are reserved for the CLAUDE.md §11 set.

---

## 5. Error Modes

These are the error classes **this module itself** can raise or encounter — distinct from the risks it catalogs.

| Situation | Handling |
|---|---|
| **Unknown-class risk (not in register)** — `Risk.triage(UNKNOWN)` or a signal with no matching entry | Returns `TriageResult(entry=R99, escalate=True, log_line="unknown signal: <raw>")`. Does NOT raise. Orchestrator is notified via SendMessage. R99 is the catch-all so `triage()` is total. |
| **Stop-condition met** (V100 down > 8h, HF outage > 2h, team drop + gate-miss) | `triage().hard_stop == True`. Orchestrator halts dispatch. This is **not** an error in `risk.py` — it is the correct behavior. |
| **Unresolvable mitigation** — owner reports mitigation attempt failed | Owner calls `Risk.triage(signal)` a second time within 5 min with `context="mitigation_failed"`. If `stop_condition` is already `ESCALATE`, upgrade path to `HARD_STOP` is explicitly a user decision (not automatic). |
| **Duplicate registration** — same `id` appears twice in `assess()` | `AssertionError` at import-time test (`tests/test_risk_book.py::test_unique_ids`). The register is a static data structure; duplicates are a code bug. |
| **`design_refs` cites a non-existent section** | Caught by `tests/test_risk_book.py::test_design_refs_valid` which greps the referenced files for the cited headings. Missing → test failure → fix before merge. |
| **`RiskLog` write fails** (disk full) | Log to stderr + continue. The run is more important than the log line. This is a silent-log fallback, not a silent-failure of the risk itself. |
| **`Risk.triage` called before `Risk.assess`** | Safe — `assess()` returns pure data; `triage()` internally calls it if not cached. No initialization-order coupling. |

**Policy:** `risk.py` never crashes the run. If in doubt, route to R99 and escalate.

---

## 6. Dependencies

### 6.1 Consumes

- `DESIGN.md §14` — the 12-risk table (R01–R12 below).
- `CLAUDE.md §11` — escalation + stop conditions (R13–R15 and `StopCondition` wiring).
- `docs/modules/training.md §5` — exception class names mapped to `TriggerSignal` values for training risks.
- `docs/modules/vendors.md §5` — envelope-shape invariant mapped to `ERROR_ENVELOPE_VIOLATION`.
- `docs/modules/audio.md §5` — model-load / voice-pack failures mapped to `KOKORO_LOAD_FAILED`, `WHISPER_LOAD_FAILED`, `INDIC_VOICE_PACK_MISSING`.

### 6.2 Consumed by

- **`env.md`** — catches audio errors; calls `Risk.triage` for logging. `env.py` already has local fallbacks (audio.md §5 table) so the triage call is informational.
- **`training.md`** — the training callback wiring (training.md §5) maps `KLDivergenceExplosion`, `RewardCollapseError`, `CheckpointIOError`, `OutOfMemoryError`, `LanguageCohortCollapseError`, `WandBStartupError` to triggers. Callback delegates to `Risk.triage`.
- **`audio.md`** — reports startup / missing-pack conditions but owns local mitigations.
- **`deploy_env_space.md`**, **`deploy_demo_space.md`** — the deploy runbooks reference R06 (ZeroGPU quota), R10 (Docker image size), R11 (openenv validate), R06-STOP (HF Hub/Spaces outage > 2h) for their Day-2 checklists.
- **`pitch_demo.md`** — references R06 (ZeroGPU) as the reason for the `gradio share=True` fallback demo path; references R12 (judge language) for the English-captions design choice.
- **`evaluation.md`** — references R13 (Stage 1 convergence) as the gate for final-eval vs retrain-decision.

### 6.3 Does not depend on

- The agent / LLM — this book is pure environment + process code.
- HF Hub at runtime — `assess()` reads from local Python data only.
- Any vendor module — vendors report errors *to* this book, not the other way.

---

## 7. Edge Cases

Minimum 5 required. Ten here for completeness.

### 7.1 Compound risk trigger (two risks fire in the same step)

**Scenario:** During Stage-3 training, at step 247, the V100 briefly disconnects (R01-V100-adjacent) AND `train/policy_kl` crosses 10.0 (R13 KL catastrophe) within a 30-second window.

**Handling:** `Risk.triage` is called twice — once per signal — in whatever order the callback observes them. Both append to `RiskLog`. The orchestrator sees two `HARD_STOP` / `ESCALATE` messages and treats them as independent incidents. `RiskLog.summary()` post-mortem will show both. We do NOT try to merge compound events into a single "meta-risk" entry — that invites silent de-duplication bugs.

### 7.2 New risk surfaced mid-event

**Scenario:** Hour 22 of onsite, Person D discovers the HF Hub deprecated `huggingface-cli upload` in favor of `hf upload` and the Dockerfile command silently no-ops. This is not in the register.

**Handling:** Person D calls `Risk.triage(UNKNOWN, context="hf-cli-deprecated")`. Returns R99 (unknown-class). Person D messages the team lead. A one-line entry is appended **to the doc** (not the frozen register) under §11 "Live additions". The frozen register gets updated in the post-hackathon PR — during the 48h we never mutate `assess()` because test suites pin its length + contents.

### 7.3 Mitigation fails (first attempt)

**Scenario:** R01 (V100 FP16 instability) fires. Mitigation: `learning_rate 5e-6 → 2e-6, resume from last checkpoint`. Person C does that. Step 30 later, `grad_norm` spikes inf again.

**Handling:** Person C re-triages with `context="mitigation_failed_round_1"`. R01's `stop_condition` is `None` (not originally escalate). But two consecutive mitigation failures promote this to `ESCALATE` — the promotion is a **human decision**, encoded in the triage-loop §3.2 step 3 ("elif triage.escalate"). Person C `SendMessage` team lead with the promoted severity. Never silent, never automatic.

### 7.4 Team member reassign (A covers B)

**Scenario:** Person B gets a fever at hour 30. Person A picks up rewards + test plans. R09 (team drop) fires with `TEAM_MEMBER_DROP` signal.

**Handling:** R09's mitigation is "roles are additive — Person D covers A+env, Person C covers rewards". But in this specific case Person A takes B's load (A has capacity because env code is already in C-batch). Orchestrator updates the live-ownership mapping in memory (not in CLAUDE.md §2.2 — that is a spec, not a runtime roster). If 3-person coverage cannot make the Phase D gate on time (`TEAM_3PERSON_BELOW_GATE` signal), R09 promotes to HARD_STOP.

### 7.5 Mitigation conflicts with deliverable

**Scenario:** R03 (Hinglish Whisper noise) mitigation: "score R3/R4 on semantic match not exact string". This was a DESIGN.md §14 decision pre-locked. At hour 36, Person B proposes tightening R4 to exact-string to fix a specific reward-hack exploit found in the probe (DESIGN.md §13 #9).

**Handling:** This is a spec conflict, not a triage event. Person B updates rewards.md AND DESIGN.md §14 AND this doc's R03 entry. Critic-gated change. NOT a `Risk.triage` call — `triage` is runtime, the spec conflict is authoring. The rule: mitigation edits go through the doc workflow, not through the runtime log.

### 7.6 False-positive stop (V100 dropped, came back in 4 min)

**Scenario:** V100 SSH drops at hour 20 minute 8. Returns at hour 20 minute 12. `V100_DOWN` fires; `V100_DOWN_OVER_8H` does NOT fire (not enough elapsed time).

**Handling:** R01-adjacent `V100_DOWN` has `stop_condition=None` — it's a nuisance. `RiskLog.append` only. Resume work. The HARD_STOP is only on the `_OVER_8H` variant, which requires a wall-clock threshold. This matches CLAUDE.md §11 literally ("V100 unavailable for > 8h").

### 7.7 ZeroGPU quota depleted during demo day

**Scenario:** Saturday midday, demo Space has been hit by spectators; ZeroGPU quota for `krrishchoudhary109` is exhausted.

**Handling:** R07 (HF Space ZeroGPU) trigger fires. Mitigation: fallback to `gradio share=True` local tunnel. Person D already has $20 A10G reserved — switch to that first (cheaper for real judge eval). If A10G pulls a 5xx, fall to `share=True` on the local V100 training box (after Stage-3 completes). RiskLog records which path was taken for the blog.

### 7.8 `openenv validate` fails on a subtle schema issue

**Scenario:** `openenv validate .` reports `observation.tool_results[].schema_version` type is `str` but validator expects `Literal["v1", "v2", "v3"]`.

**Handling:** R11 (openenv validate fails) fires. Mitigation: "Validate early (pre-onsite hour 16 gate)." If we are past that gate and still failing, after 3 fix attempts `OPENENV_VALIDATE_FAIL` promotes to `ESCALATE`. Person D + Orchestrator decide whether to (a) update our models.md to use the Literal, (b) ask user for schema-deviation approval, or (c) drop the validator check — never (c) without user approval.

### 7.9 Reward-hacking spike during Stage 2

**Scenario:** `train/R5_mean` drops to -0.32 at step 180 of Stage 2. `train/reward_mean` stays flat — classic hack indicator. `training.md §5 RewardCollapseError` fires.

**Handling:** R05 (reward hacking on R2) trigger. Mitigation chain: (i) training halts, `RewardCollapseError` raised; (ii) Person B runs reward-hacking probe on last 200 episodes per DESIGN.md §13 #9; (iii) if a new exploit pattern is found, update `anti_hack_penalty` logic in `rewards.py` per rewards.md §3.6; (iv) resume from **pre-regression** checkpoint, NOT the current one. This case is the single most important runtime mitigation in the book — reward hacking propagates through the model weights and cannot be fixed by just resuming.

### 7.10 Judge doesn't speak Hindi

**Scenario:** Demo day, the judge hearing our booth is fluent only in English + Mandarin (neither Indic).

**Handling:** R12 pre-staged mitigation — every audio clip in the deck has English captions, and the demo Gradio UI auto-translates agent replies to English for readability. No runtime trigger; this is a design-time mitigation surfaced at the pitch-script level. RiskLog records which judges the English path fired for (instrumentation lives in the demo, not in this module).

---

## 8. Examples

Three concrete worked examples, runnable if implemented.

### 8.1 R01 — V100 FP16 gradient instability → detection + mitigation + rollback

**Setup.** Hour 6 of onsite. Stage-1 training starts. Fresh V100 box; Unsloth 2026.4.5 pinned; `use_bias_correction_kl=True`. Everything looks right.

**Detection (step 14).** Training callback reports `grad_norm=inf`. `NonFiniteGradientError` would be raised on the 4th consecutive skip. At step 14, this is the first occurrence.

```
signal = TriggerSignal.GRAD_NORM_INF
triage = Risk.triage(signal)
# → TriageResult(
#     entry=R01,
#     action="Unsloth 4-bit QLoRA + FP16 autocast; grad clip 1.0; loss-scale monitored every 10 steps; fallback to dtype=float16 explicit",
#     escalate=False, hard_stop=False,
#     log_line="R01 fired (step 14): grad_norm=inf; mitigation=fp16_autocast+gradclip1.0"
#   )
RiskLog.append(triage, ts_iso="2026-04-25T15:22:04+05:30")
# Person C inspects: loss-scale has halved twice in last 50 steps — precursor to underflow.
# Action (from training.md §7a mitigation 5): drop learning_rate 5e-6 → 2e-6, resume from last checkpoint.
```

**Mitigation (step 15 onward).** Person C: `train(..., learning_rate=2e-6, resume_from=Path("checkpoints/stage1_step_10"))`. Resume starts cleanly.

**Rollback guardrail.** Training continues for 20 steps. `grad_norm` stays finite. `train/skipped_updates==0` rolling mean. R01 does NOT fire again.

**If it had fired again:** Person C would triage with `context="mitigation_failed_round_1"`. Second-round failure promotes R01 to escalation per §3.3; user is paged; we consider downshifting to Gemma 3 4B per CLAUDE.md §11 first-responder case.

### 8.2 R05 — Reward hacking spike on R2 → probe → reward re-weighting

**Setup.** Hour 28 of onsite, Stage-2 (single-drift) training. `train/R5_mean` has floated near 0.0 for 170 steps. Then across steps 171–180, it drops to -0.31.

**Detection (step 180).** The training callback computes `R5_mean_10` (10-step moving mean) = -0.31 AND checks `train/hallucinated_field_count` simultaneously (training.md §7d) — hallucinations spiked. `train/reward_mean` is flat (not dropping proportionally). `RewardCollapseError` is raised.

```
signal = TriggerSignal.R5_DROP_WITH_HACK_SPIKE
triage = Risk.triage(signal)
# → TriageResult(
#     entry=R05,
#     action="R2 requires specific field-name OR correct follow-up call; R5 penalizes bare assertions. "
#            "Halt training; surface to Person B for probe inspection (training.md §7d).",
#     escalate=True, hard_stop=False,
#     log_line="R05 fired (step 180): R5_mean_10=-0.31; reward_mean flat; halt + probe"
#   )
```

**Probe (hour 28 + 30 min).** Person B runs `python3 eval/reward_hack_probe.py --episodes-from step_170_to_180.jsonl --top-k-suspicious 20`. Output reveals: the agent learned to emit `"drift detected on unknown_field"` on EVERY turn, which scores R2 = +0.3 via the substring "drift" token match AND R5 = -0.3 for bare assertion — net 0.0, but the hallucinated R2 credit still feeds advantage variance and the model is "rewarded by gradient" even if the mean reward is unchanged.

**Fix (hour 29).** rewards.md §3.6 update: tighten R2 to require the drift-pattern's canonical `error_code` token (from vendors.md §5.2) as substring, not a generic `"drift"` token. Bump `config_sha256`.

**Resume (hour 30).** Person C resumes from the **pre-regression** checkpoint `checkpoints/stage2_step_160` — NOT 170 — because weights from 160 onward already encoded the exploit pattern in latent gradient directions. Stage-2 restarts with updated rewards. Runs clean for 20 steps; R05 does not re-fire.

**Book-keeping.** The reward-hacking probe result is appended to `reward_hacks.jsonl` (DESIGN.md §13 deliverable #9). The blog post cites this incident as "one of three exploits caught by the probe, all fixed without silent re-weighting".

### 8.3 Stop-condition trigger — V100 down ≥ 8h

**Setup.** Hour 14 of onsite. V100 box drops SSH at hour 14:02. Root cause: data-center UPS maintenance neither DGX ops nor the hackathon organizers flagged.

**Detection (continuous).** Orchestrator watchdog `ssh -o ConnectTimeout=5 v100-box echo ok` fails. Signal `V100_DOWN` fires at hour 14:02.

```
triage = Risk.triage(TriggerSignal.V100_DOWN)
# → R01-adjacent entry, non-stop, log_only.
# RiskLog.append. Continue non-training work (Person B on test plans, Person D on Dockerfile).
```

**Escalation (hour 14:02 → hour 22:02).** Watchdog flips every 2 min. At wall-clock 8 hours elapsed without successful SSH, signal `V100_DOWN_OVER_8H` fires. CLAUDE.md §11 says: hard stop.

```
triage = Risk.triage(TriggerSignal.V100_DOWN_OVER_8H)
# → TriageResult(
#     entry=R01-STOP variant (id="R01-STOP"),
#     action="V100 unreachable > 8h — terminate plan; re-plan with user.",
#     escalate=True, hard_stop=True,
#     log_line="R01-STOP fired: V100 down 8h+; HARD_STOP"
#   )
# Orchestrator:
#   - halts all Agent dispatch immediately
#   - SendMessage team-lead: "HARD_STOP R01-STOP: V100 down ≥ 8h. Plan must be re-scoped with user."
#   - awaits user decision: (a) cancel onsite, (b) pivot to Colab T4 at reduced scale, (c) submit baseline-only.
# Person C preserves whatever checkpoint is on last-good-rsync to HF Hub.
```

**Resolution.** User decides: pivot to Colab T4 for Stage-1 only; skip Stage-3; record the incident in the blog. The blog-post honesty about this is, per DESIGN.md §14 #9-adjacent, an asset not a liability.

---

## 9. Open Questions

1. **Should `Risk.triage` auto-promote to ESCALATE on second-round mitigation failure, or stay manual?** Current spec: manual (§7.3). Rationale: automated promotion has surprised us in prior incidents. But manual requires an attentive operator. DECISION during Batch D3 critic round: stay manual; revisit post-hackathon.

2. **Do we include supply-chain risks (PyPI malicious package)?** Currently out of scope — we pin every dep in `requirements.txt`, and `pip-audit` is a Phase C C1 task. If a malicious package lands during onsite, it becomes R99 (unknown). Not worth a dedicated entry for a 48h event.

3. **`RiskLog` public vs private.** We write `risk_log.jsonl` to the run directory. Do we push it to HF Hub as part of the training traces (DESIGN.md §13 #6)? Pros: full transparency for judges. Cons: the log names V100 incidents which is operationally sensitive. Proposed resolution: push with operator identifiers scrubbed to `team` / `A..D` only. Confirm with Person D before Phase C5.

4. **R14 threshold specifics.** Currently "Stage 1 doesn't converge (R1 < 0.4 at step 100)". Is step 100 the right checkpoint, or should it be step 75? DESIGN.md §14 risk #4 says Stage 1 targets 100 steps total, so step 100 = end-of-stage — already too late to recover in-stage. Proposed: `STAGE1_R1_BELOW_0_4_AT_STEP_75` is the actual signal, but we call it "at step 100" in prose to match CLAUDE.md §11. Confirm with Person C during their training callback implementation.

5. **Judge language (R12) beyond Hindi.** Our mitigation is "English captions on every audio clip." What if the judge is fluent only in Mandarin and English? Captions are still English. Demo voice reply is Hindi. Acceptable? YES — English is the hackathon lingua franca; Mandarin judges on an India-first hackathon is low-probability. Not worth adding a second caption track.

---

## 10. The Register (data for §2.3 `Risk.assess()`)

Fifteen entries. R01–R12 are DESIGN.md §14 verbatim, augmented with owner + trigger signal. R13–R15 are CLAUDE.md §11 adds.

### R01 — V100 FP16 gradient instability (Gemma 4 is BF16-native)

- **Probability:** Med
- **Impact:** kills (training)
- **Mitigation:** Unsloth 4-bit QLoRA + FP16 autocast; `max_grad_norm=1.0`; loss-scale monitored every 10 steps; fallback to `dtype=torch.float16` explicit at `FastModel.from_pretrained`; if instability persists, `learning_rate 5e-6 → 2e-6` and resume from last checkpoint.
- **Owner:** C (Training & Data)
- **Trigger signal:** `GRAD_NORM_INF` (training callback observes `grad_norm` is inf or NaN; training.md §5 `NonFiniteGradientError` after 3 consecutive skips).
- **Stop condition:** None (local mitigation). **See R01-STOP** for the > 8h variant.
- **design_refs:** DESIGN.md §14 #1, training.md §5, training.md §7a, §7c.

### R01-STOP — V100 unavailable for more than 8 hours

- **Probability:** Low
- **Impact:** kills (training AND demo)
- **Mitigation:** HARD STOP. Preserve last-good checkpoint to HF Hub. Re-plan with user: (a) cancel onsite, (b) pivot to Colab T4 with reduced scope, (c) submit baseline-only.
- **Owner:** Orchestrator → user
- **Trigger signal:** `V100_DOWN_OVER_8H` (wall-clock ≥ 8h since first `V100_DOWN` event, per CLAUDE.md §11).
- **Stop condition:** `HARD_STOP`.
- **design_refs:** CLAUDE.md §11 (hard-stop #1).

### R02 — TRL GRPOTrainer KL catastrophe

- **Probability:** Med
- **Impact:** kills (training)
- **Mitigation:** Pin TRL ≥ 0.23; `use_bias_correction_kl=True` (invariant asserted in `build_grpo_config`); `beta=0.04`; training callback raises `KLDivergenceExplosion` if `policy_kl` 10-step mean > 10.0; halt + dump last 20 rollout groups to `debug/kl_explosion_dump.jsonl` and escalate to user.
- **Owner:** C (Training & Data)
- **Trigger signal:** `POLICY_KL_OVER_10` (training.md §5 `KLDivergenceExplosion`).
- **Stop condition:** `ESCALATE_TO_USER` (halt training; no silent recovery attempt; root-cause is almost always config-invariant violation).
- **design_refs:** DESIGN.md §14 #2, training.md §5, training.md §7c, TRL issue #4637.

### R03 — Whisper transcription errors on Hinglish code-mixing

- **Probability:** High
- **Impact:** med (noisy observation)
- **Mitigation:** Use `faster-whisper-small` with `language="hi"`; accept noise — it's realistic; score R3/R4 on semantic match not exact string. TTS/ASR is ONLY at env boundary (DESIGN.md §9.4), never in training loop — so noisy ASR cannot break gradient. Reflected in `DriftCallObservation.last_confidence`; agent may `CLARIFY` to re-prompt.
- **Owner:** C (Audio subsystem; env-side consumer is A)
- **Trigger signal:** observable as low `TranscriptResult.confidence` in observation; not an exception. Does not fire `Risk.triage` unless confidence < 0.3 on ≥ 30% of episodes over a 50-episode window (indicating systemic degradation, not natural variance).
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #3, audio.md §5, audio.md §7.1, §7.4, rewards.md (semantic-match specification).

### R04 — 200–500 GRPO steps too few for 3-stage curriculum

- **Probability:** Med
- **Impact:** med (Stage 3 undertrains)
- **Mitigation:** Compressed curriculum: Stage 1 → 100, Stage 2 → 200, Stage 3 → 100 (total 400). Prioritize Stage 2 depth. If time budget slips further, drop Stage 3 entirely and ship Stage-2-final as the LoRA — this is still better than untrained.
- **Owner:** C (Training & Data)
- **Trigger signal:** not a runtime signal per se; checked at each stage boundary via `train/step` vs plan. Triggers `Risk.triage` only if Stage-3 has < 50 steps available at its start.
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #4, DESIGN.md §10.3, training.md §6.

### R05 — Reward hacking on R2 (spam "drift detected!")

- **Probability:** High
- **Impact:** high (R2 collapses as a signal)
- **Mitigation:** R2 requires specific `error_code` field-name substring OR correct follow-up call (rewards.md §3.6); R5 penalizes bare assertions (-0.3); penalties stack additively to -1.0 floor. Runtime: `RewardCollapseError` raises on `R5_mean_10 ≤ -0.3` AND `hallucinated_field_count` spike AND `reward_mean` flat — halt and run reward-hacking probe (DESIGN.md §13 deliverable #9); resume from **pre-regression** checkpoint, not the current one.
- **Owner:** B (Rewards & Tests)
- **Trigger signal:** `R5_DROP_WITH_HACK_SPIKE` (training.md §5 `RewardCollapseError`).
- **Stop condition:** `ESCALATE_TO_USER` (halt + probe; does not auto-hard-stop, but requires B attention).
- **design_refs:** DESIGN.md §14 #5, DESIGN.md §7.3, training.md §7d, rewards.md §3.6.

### R06 — HF Space ZeroGPU quota exhausted

- **Probability:** Low
- **Impact:** med (demo degrades)
- **Mitigation:** $20 A10G budget reserved for paid fallback; secondary fallback `gradio share=True` locally from the V100 training box after Stage-3 completes. Pre-generated demo audio means a static deck is the tertiary fallback.
- **Owner:** D (Deploy & Story)
- **Trigger signal:** HF Space returns quota-exceeded HTTP; surfaced by deploy-demo-space.md health probe.
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #6, deploy_demo_space.md §4.

### R06-STOP — HF Hub / Spaces outage persisting > 2 hours
- **Description:** Hugging Face Hub or Spaces serving infrastructure is unreachable for > 2 hours during onsite; env Space / demo Space / Hub model pushes cannot be verified or redeployed.
- **P:** Low · **I:** kills
- **Mitigation:** pause all HF-dependent tasks; fall back to local `gradio share=True` demo and local-only Docker testing; if not restored by onsite hour −2 (2 hours before pitch), orchestrator declares `HARD_STOP` and team pivots to pre-recorded pitch video.
- **Owner:** orchestrator
- **Trigger signal:** `HF_HUB_OUTAGE_OVER_2H`
- **Stop condition:** `HARD_STOP`
- **Design refs:** `CLAUDE.md §11 Hard-stop #2`, `DESIGN.md §13` (deployment), `deploy_env_space.md §9 OQ3`

### R07 — Indic Whisper quality too poor for live demo

- **Probability:** Med
- **Impact:** med (live demo weak)
- **Mitigation:** Fallback — English-only briefs for live demo; Indic briefs in recorded video. Decision point is hour 40 (after Stage-3 complete) based on whisper quality measured during final eval.
- **Owner:** D (Deploy & Story), decision co-signed by C.
- **Trigger signal:** final-eval Indic-cohort R4 < 0.3 (a proxy for "agent can't parse Indic transcripts reliably").
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #7, audio.md §1.1, pitch_demo.md.

### R08 — Kokoro Indic voice quality insufficient

- **Probability:** Low
- **Impact:** med (demo sounds bad)
- **Mitigation:** Pre-generate ALL demo audio offline with careful voice-pack selection; A/B each generated clip against AI4Bharat reference clips before shipping; if still poor, use AI4Bharat's TTS for demo-only clips (training is text-only so this has no training-side impact).
- **Owner:** D (Deploy & Story), audio pipeline owned by C.
- **Trigger signal:** `INDIC_VOICE_PACK_MISSING` (audio.md §5) OR subjective quality fail at pre-demo rehearsal.
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #8, audio.md §4.3.1 (voice-pack fallback chain), audio.md §7.2.

### R09 — Team member drops / sick

- **Probability:** Med
- **Impact:** high (≥ 8h slip)
- **Mitigation:** Roles are additive by design — Person D covers A + env if A drops; Person C covers rewards if B drops; Person A covers demo if D drops. Plan survives 3-person execution. If 3-person execution cannot hit Phase D gate on time, escalate.
- **Owner:** Orchestrator (re-planning); affected teammate for hand-off.
- **Trigger signal:** `TEAM_MEMBER_DROP` (self-reported via SendMessage to team-lead).
- **Stop condition:** `ESCALATE_TO_USER`. Promotes to `HARD_STOP` only if `TEAM_3PERSON_BELOW_GATE` also fires.
- **design_refs:** DESIGN.md §14 #9, CLAUDE.md §11 (hard-stop #3), CLAUDE.md §2.2.

### R10 — Env Docker image too large for free CPU tier

- **Probability:** Low
- **Impact:** med (env Space fails to deploy)
- **Mitigation:** Trim Whisper/Kokoro models to int8; use Alpine base (or `python:3.11-slim` with aggressive layer pruning); target < 2 GB total (audio.md §6.4 already budgets ~450 MB for weights). Measured at `docker build` via `docker images` → image size.
- **Owner:** D (Deploy & Story)
- **Trigger signal:** `DOCKER_IMAGE_OVER_2GB` — Dockerfile build prints image size; CI lint fails if > 2 GB.
- **Stop condition:** None.
- **design_refs:** DESIGN.md §14 #10, deploy_env_space.md, audio.md §6.4.

### R11 — `openenv validate` fails on our spec

- **Probability:** Med
- **Impact:** high (disqualification risk)
- **Mitigation:** Validate early — pre-onsite hour 16 gate (CLAUDE.md §8 smoke-test checklist expansion). Keep `openenv`'s known-good example envs side-by-side for diffing. After 3 fix attempts without success, escalate to user for schema-deviation approval.
- **Owner:** D (Deploy & Story), env schema owned by A.
- **Trigger signal:** `OPENENV_VALIDATE_FAIL` after 3 fix attempts.
- **Stop condition:** `ESCALATE_TO_USER` (CLAUDE.md §11 escalate #2).
- **design_refs:** DESIGN.md §14 #11, CLAUDE.md §11 (escalate #2), env.md.

### R12 — Judge doesn't speak Hindi / Indic, misses the nuance

- **Probability:** Med
- **Impact:** med (weaker 30% score component)
- **Mitigation:** Pitch deck has English captions on every audio clip; demo UI auto-translates agent replies to English; pitch script explicitly calls out schema-drift which is language-agnostic as the primary technical contribution (Indic is the substrate, drift is the innovation).
- **Owner:** D (Deploy & Story)
- **Trigger signal:** `JUDGE_LANGUAGE_MISMATCH` — observed at booth (not automated).
- **Stop condition:** None (design-time mitigation, not runtime).
- **design_refs:** DESIGN.md §14 #12, pitch_demo.md, DESIGN.md §15.

### R13 — Gemma 4 E2B smoke test fails on V100 (NEW per CLAUDE.md §11)

- **Probability:** Med
- **Impact:** kills (block; may need downshift to Gemma 3 4B)
- **Mitigation:** Pre-onsite smoke test per DESIGN.md §16.A.1 — run BEFORE Batch D1 kickoff (CLAUDE.md §8 checklist). If fails: diagnose via `nvidia-smi`, CUDA driver version, Unsloth pin. Two fallbacks in priority order: (i) downshift to `unsloth/gemma-3-4b-it-bnb-4bit` — same GRPO pipeline, known-working on V100; (ii) use `unsloth/gemma-2-2b-it-bnb-4bit` — smaller, older, very stable. Either downshift requires updating DESIGN.md §0 model ref AND escalation to user (architecture-level decision).
- **Owner:** C (Training & Data); escalation via Orchestrator.
- **Trigger signal:** `GEMMA4_SMOKE_FAIL`.
- **Stop condition:** `ESCALATE_TO_USER` (CLAUDE.md §11 escalate #1). Hard-stop only if all three candidates (E2B, 3-4B, 2-2B) fail — then hardware is the problem.
- **design_refs:** CLAUDE.md §11 (escalate #1), DESIGN.md §16.A.1, training.md §6.3.

### R14 — Stage 1 training doesn't converge (NEW per CLAUDE.md §11)

- **Probability:** Med
- **Impact:** high (reward/curriculum redesign required; Stage 2/3 blocked)
- **Mitigation:** Detect at step 100 (end-of-Stage-1): if `eval/R1 < 0.4` on the 50-episode held-out set, halt. Diagnose via per-reward breakdown: is R1 flat (tool-use not learned) or R4 flat (format not learned)? Fix candidates: (i) raise `learning_rate` 5e-6 → 1e-5 for Stage 1 only (Stage 2/3 stay at 5e-6); (ii) over-weight R4 during Stage 1 only (rewards.md temporary override); (iii) extend Stage 1 from 100 → 150 steps by borrowing from Stage 3 budget. All three are reversible; pick one, re-run Stage 1.
- **Owner:** C (Training & Data), reward-side decisions co-signed by B.
- **Trigger signal:** `STAGE1_R1_BELOW_0_4_AT_100`.
- **Stop condition:** `ESCALATE_TO_USER` (CLAUDE.md §11 escalate #3).
- **design_refs:** CLAUDE.md §11 (escalate #3), training.md §6, DESIGN.md §10.3.

### R15 — Merge conflict across owned files — ownership violation (NEW per CLAUDE.md §11)

- **Probability:** Low
- **Impact:** med (short slip; trust hit)
- **Mitigation:** First rule — orchestrator resolves merge conflicts, never delegates to an agent (CLAUDE.md §5.4). If a conflict touches both A-owned (env/vendors) and B-owned (rewards/tests) files, the ownership was wrong somewhere. Orchestrator: (i) halt both agents, (ii) identify the overlap file, (iii) reassign that file to a single owner, (iv) update CLAUDE.md §2.2 if the reassignment is permanent, (v) restart the affected batch.
- **Owner:** Orchestrator.
- **Trigger signal:** `MERGE_CONFLICT_CROSS_OWNER`.
- **Stop condition:** `ESCALATE_TO_USER` (process-level; CLAUDE.md §11 escalate #5). Never hard-stop — this is a process bug, not a technical one.
- **design_refs:** CLAUDE.md §11 (escalate #5), CLAUDE.md §5.4, CLAUDE.md §2.2.

### R99 — Unknown-class risk (not in register)

- **Probability:** Low (by construction — the register aims to be exhaustive)
- **Impact:** unknown (treated as high until classified)
- **Mitigation:** `Risk.triage(UNKNOWN)` routes here. Log line to `risk_log.jsonl`; SendMessage team-lead; human classifies within 15 min: (a) add a new entry to this doc (post-hackathon PR) and use R99's mitigation template for now; (b) map to an existing entry if the signal was mislabeled.
- **Owner:** Orchestrator → team lead.
- **Trigger signal:** `UNKNOWN` or any signal not matched by R01–R15.
- **Stop condition:** `ESCALATE_TO_USER` (default-pessimistic).
- **design_refs:** CLAUDE.md §11 (catch-all), §7.2 of this doc.

---

## 11. Live additions (populated during the 48h)

Append-only notes about risks discovered after hour 0 of onsite. These go here NOT into `Risk.assess()` because the register's length is pinned in tests. Post-hackathon PR will promote these into numbered entries.

_(empty at v1.0)_

---

## 12. Cross-doc consistency table

| Risk | DESIGN.md §14 # | CLAUDE.md §11 | training.md §5 | vendors.md §5 | audio.md §5 | rewards.md |
|---|---|---|---|---|---|---|
| R01 | #1 | — | `NonFiniteGradientError` | — | — | — |
| R01-STOP | — | hard-stop #1 | — | — | — | — |
| R02 | #2 | — | `KLDivergenceExplosion` | — | — | — |
| R03 | #3 | — | — | — | confidence field | §3.6 semantic match |
| R04 | #4 | — | — | — | — | — |
| R05 | #5 | — | `RewardCollapseError` | — | — | §3.6 |
| R06 | #6 | — | — | — | — | — |
| R06-STOP | — | hard-stop #2 | — | — | — | — |
| R07 | #7 | — | — | — | — | — |
| R08 | #8 | — | — | — | `INDIC_VOICE_PACK_MISSING` | — |
| R09 | #9 | hard-stop #3 | — | — | — | — |
| R10 | #10 | — | — | — | §6.4 budget | — |
| R11 | #11 | escalate #2 | — | — | — | — |
| R12 | #12 | — | — | — | — | — |
| R13 | — | escalate #1 | — | — | — | — |
| R14 | — | escalate #3 | — | — | — | §3 (curriculum impact) |
| R15 | — | escalate #5 | — | — | — | — |
| R99 | — | catch-all | — | — | — | — |

Every cell in this table corresponds to an actual citation in the source doc. If a cell is filled but the cited section does not actually describe that risk → spec-drift bug → fix the cite.
