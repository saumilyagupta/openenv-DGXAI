# rewards.md — DriftCall Reward System

**Module:** `driftcall/rewards.py`
**Owner:** Person B (Rewards & Tests)
**Implements:** DESIGN.md §7 (Reward System), §4.1 (Dataclasses), §4.4 (Episode Termination)
**Consumers:** `driftcall/env.py` (on `SUBMIT` / `ABORT` / timeout), `training/train_grpo.py`, `training/eval_baseline.py`, `training/eval_final.py`
**Status:** Design spec — code precedes only after ≥ 2 fresh critic rounds return `NOTHING_FURTHER`.

---

## 1. Purpose

The reward module is the **sole arbiter of episode outcome** in DriftCall. It converts a terminated episode — a frozen, append-only record of goal, actions, tool results, and drift events — into five independent scalar reward signals (`R1..R5`), combines them into a single scalar `reward` for GRPO advantage computation, and emits a structured `Rewards` record for audit and WandB logging.

This module is the **ground truth** that supervises training. It exists to:

1. Enforce DESIGN.md's **"environment is the judge, no LLM-as-judge"** invariant (§7.1, §7.3).
2. Provide **deterministic, verifiable** rewards so that every training signal is reproducible from the episode transcript alone.
3. Implement **anti-hack** defenses (§7.3) so GRPO cannot collapse into pathological policies (spam-submit, hallucinated drift claims, schema introspection abuse, etc.).
4. Apply the **Brier-calibration** penalty that rewards well-calibrated confidence and hits the Snorkel AI sub-theme bonus (§2.3).

No side effects. Pure function of `Episode → Rewards`. Called exactly once per episode, at termination, from `DriftCallEnv.step()` when it sets `done=True`.

---

## 2. Interface

### 2.1 Top-level entry point

```python
def compute_rewards(episode: Episode) -> Rewards: ...
```

### 2.2 Per-reward primitives (all pure, all deterministic)

```python
def task_completion(episode: Episode) -> float: ...          # R1 ∈ {0.0, 1.0}
def drift_detection(episode: Episode) -> float: ...          # R2 ∈ {0.0, 0.5, 1.0}
def constraint_adherence(episode: Episode) -> float: ...     # R3 ∈ [0.0, 1.0]
def format_compliance(episode: Episode) -> float: ...        # R4 ∈ [0.0, 1.0]
def anti_hack_penalty(episode: Episode) -> float: ...        # R5 ∈ [-1.0, 0.0]
```

### 2.3 Combination helpers

```python
def combine_quality(r1: float, r2: float, r3: float, r4: float, r5: float) -> float: ...
def brier_penalty(confidence: float | None, r1: float) -> float: ...
def apply_uncertain_floor(reward: float, r1: float, confidence: float | None) -> float: ...
def final_reward(quality: float, brier: float, r1: float, confidence: float | None) -> float: ...
```

**Per-helper contracts** (exact pre/post-conditions — this is the CONTRACT, not a sketch):

**`combine_quality(r1, r2, r3, r4, r5) -> float`**
- **Pre:** `r1 ∈ {0,1}`, `r2 ∈ {0, 0.5, 1}`, `r3 ∈ [0,1]`, `r4 ∈ [0,1]`, `r5 ∈ [-1, 0]`. All finite.
- **Post:** returns the weighted sum exactly per §7.2 / DESIGN.md §7.2:
  `0.50*r1 + 0.20*r2 + 0.15*r3 + 0.10*r4 + 0.05*min(r5, 0.0)`.
- Returns a raw float in approximately `[-0.05, 1.00]`. **Does NOT clamp. Does NOT round.**

**`brier_penalty(confidence, r1) -> float`**
- **Pre:** `r1 ∈ {0.0, 1.0}`; `confidence is None` or `confidence ∈ [0.0, 1.0]` (out-of-range is clamped for this call only per §5).
- **Post:** returns `min((confidence - r1) ** 2, 0.5)` iff `confidence is not None`, else `0.0`.
- Returns a float in `[0.0, 0.5]`. **Does NOT clamp outside [0, 0.5]. Does NOT round.**

**`apply_uncertain_floor(reward, r1, confidence) -> float`**
- **Signature note:** first arg is the **PRE-clamp reward** (i.e. `quality * (1 - brier)`), NOT the raw quality.
- **Pre:** `reward` is any finite float; `r1 ∈ {0.0, 1.0}`; `confidence is None` or a finite float.
- **Post:** applies the uncertain floor **iff** `r1 == 0.0 AND confidence is not None AND confidence < 0.3`:
  returns `max(reward, 0.3)`. Otherwise returns `reward` unchanged (identity).
- Side-signal: the caller is responsible for recording `floor_applied = (returned_value != input_reward and floor-condition-true)`.
- **Does NOT clamp to [0,1]. Does NOT round.**

**`final_reward(quality, brier, r1, confidence) -> float`**
- **Orchestration helper.** This is the ONLY helper that clamps and rounds.
- **Pre:** `quality` from `combine_quality`; `brier` from `brier_penalty`; `r1 ∈ {0.0, 1.0}`; `confidence is None` or finite.
- **Post:** computes
  1. `reward = quality * (1.0 - brier)`
  2. `reward = apply_uncertain_floor(reward, r1, confidence)`
  3. `reward = max(0.0, min(1.0, reward))` — clamp to [0, 1]
  4. `reward = round(reward, 3)` — 3-decimal rounding
- Returns a float in `[0.000, 1.000]` with at most 3 decimals.

**Call order invariant (enforced by `compute_rewards`):**

```
combine_quality  →  brier_penalty  →  multiply (quality * (1 - brier))
                                         ↓
                                 apply_uncertain_floor
                                         ↓
                                       clamp
                                         ↓
                                       round
                                         ↓
                                   final reward
```

No helper may reorder these steps. `combine_quality`, `brier_penalty`, and `apply_uncertain_floor` each produce unclamped, unrounded floats; clamp + round happens exactly once, inside `final_reward`.

### 2.4 `Episode` — input contract (consumed, not owned)

`Episode` is defined in `models.md` / `models.py`. For this module's contract, it MUST carry (frozen):

```python
@dataclass(frozen=True)
class Episode:
    episode_id: str
    goal: GoalSpec                              # see DESIGN.md §4.1
    actions: tuple[DriftCallAction, ...]        # append-only, turn-indexed
    tool_results: tuple[ToolResult, ...]        # append-only
    drift_log: tuple[DriftEvent, ...]           # drifts that FIRED this episode
    vendor_states_final: dict[str, dict[str, Any]]  # post-termination vendor DB
    schema_versions_final: dict[str, str]       # post-termination schema map
    max_turns: int
    turns_used: int                             # == len(actions)
    terminated_by: Literal["SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"]
    stage: Literal[1, 2, 3]                     # curriculum stage, from reset config
```

If a field is missing, see §5 Error modes.

### 2.5 `Rewards` — output contract (returned, fully immutable)

```python
@dataclass(frozen=True)
class Rewards:
    r1: float                 # task_completion, {0.0, 1.0}
    r2: float                 # drift_detection, {0.0, 0.5, 1.0}
    r3: float                 # constraint_adherence, [0.0, 1.0]
    r4: float                 # format_compliance, [0.0, 1.0]
    r5: float                 # anti_hack_penalty, [-1.0, 0.0]
    quality: float            # weighted combination, [−0.05, 1.0] before clamp
    brier: float              # Brier penalty, [0.0, 0.5]
    reward: float             # final scalar for GRPO, [0.0, 1.0], 3 decimals
    confidence: float | None  # echoed from SUBMIT action, None if ABORT/TIMEOUT
    floor_applied: bool       # True iff §7.3 uncertain-floor clamp raised reward
    breakdown: dict[str, Any] # diagnostic per-reward evidence; see §4.2
```

`reward` is the scalar consumed by TRL GRPO. Everything else is for WandB + audit + reward-hacking probe.

---

## 3. Behavior Spec

### 3.1 Determinism & evaluation-time-only invariants

- **Pure function.** Given two `Episode` values with equal field content, `compute_rewards` MUST return equal `Rewards`. No RNG, no clock reads, no network, no disk. (Implementation checked by property test — see `docs/tests/rewards_tests.md`.)
- **No LLM-as-judge.** No HTTP calls, no model inference, no fuzzy matching against an external model. All string checks are regex / substring / exact-match only. This is the hard invariant from DESIGN.md §7.1 and §7.3.
- **Evaluation-time-only.** `compute_rewards` is called **exactly once**, at episode termination. Never mid-episode. Never on a partial transcript. The env enforces this by setting `done=True` before calling.
- **Idempotent.** Calling `compute_rewards(ep)` twice with the same episode yields the same result.
- **Frozen inputs.** `Episode` is frozen; rewards MUST NOT mutate it. Detected by `dataclasses.is_dataclass(ep) and ep.__dataclass_params__.frozen`.

### 3.2 R1 — Task Completion (DESIGN.md §7.1 / R1)

**Signature:** `task_completion(episode: Episode) -> float`
**Range:** `{0.0, 1.0}` — strict binary.

**Algorithm (pseudocode):**

```
if episode.terminated_by != "SUBMIT":
    return 0.0

goal = episode.goal
final = episode.vendor_states_final

# Per-domain check — EXACT match on required slots + constraint satisfaction
# The predicate for each domain lives in driftcall.rewards.checkers and mirrors
# the slot/constraint fields from the task-brief template (datasets.md §8.3).
match goal.domain:
    case "airline":   ok = _check_airline_booking(goal, final)
    case "cab":       ok = _check_cab_booking(goal, final)
    case "restaurant":ok = _check_restaurant_order(goal, final)
    case "hotel":     ok = _check_hotel_booking(goal, final)
    case _:           ok = False   # unknown domain — R1=0, flagged in breakdown

return 1.0 if ok else 0.0
```

**Per-domain success criteria (verbatim from DESIGN.md §7.1):**

| Domain | Success predicate |
|---|---|
| airline | booking exists for correct route + date + time window + within budget |
| cab | ride scheduled for correct pickup/drop + time |
| restaurant | order placed with correct items + dietary + budget |
| hotel | reservation for correct city + dates + room type |

"Correct route" = `goal.slots.from == booking.from AND goal.slots.to == booking.to`. "Time window" = parsed window (e.g. "evening" → 18:00–22:00 IST) contains `booking.depart`. "Budget" = `booking.total <= goal.constraints.budget_inr`. No fuzzy string match — slot IDs normalize through the same canonicaliser the task generator uses.

### 3.3 R2 — Drift Detection (DESIGN.md §7.1 / R2)

**Signature:** `drift_detection(episode: Episode) -> float`
**Range:** `{0.0, 0.5, 1.0}`.

**Algorithm:**

```
if episode.stage == 1 OR len(episode.drift_log) == 0:
    return 0.5                                      # neutral / skipped

for drift in episode.drift_log:
    # Guard: empty hints → structural bug, raise (see §5)
    if not drift.detection_hints or all(not h for h in drift.detection_hints):
        raise RewardComputationError(
            f"drift {drift.id} has empty detection_hints", episode.episode_id
        )

    window = [turn in [drift.turn, drift.turn + 1, drift.turn + 2]]
    actions_in_window = [a for a in episode.actions if a.turn in window]

    # Branch 1: speech channel — case-insensitive substring match on SPEAK/CLARIFY text
    hit_by_speech = any(
        a.action_type in {SPEAK, CLARIFY}
        and _mentions_drift(a.message, drift)
        for a in actions_in_window
    )
    # Branch 2: tool-call-args-hint channel — case-insensitive substring match
    #           on the JSON-stringified tool_args AND its string arg VALUES
    hit_by_args_hint = any(
        a.action_type == TOOL_CALL
        and _args_mention_drift(a.tool_args, drift)
        for a in actions_in_window
    )
    # Branch 3: structural-adaptation channel — tool_args conforms to post-drift schema
    hit_by_adaptation = any(
        a.action_type == TOOL_CALL
        and _uses_new_schema(a.tool_args, drift)
        for a in actions_in_window
    )
    if hit_by_speech or hit_by_args_hint or hit_by_adaptation:
        continue
    else:
        return 0.0                                  # one miss → whole-episode miss

# Fail-fast: 3+ consecutive retries of the OLD schema after drift fires
if _has_3plus_old_schema_retries(episode):
    return 0.0

return 1.0
```

**R2 has THREE independent detection branches** (any one positive → drift detected for that event):

1. **Speech channel** (`_mentions_drift`) — on `SPEAK` / `CLARIFY` `action.message`.
2. **Tool-call-args-hint channel** (`_args_mention_drift`) — on `TOOL_CALL` `action.tool_args`, matched as a stringified JSON payload AND on concatenated string arg values.
3. **Structural-adaptation channel** (`_uses_new_schema`) — on `TOOL_CALL` `action.tool_args` conforming to the post-drift schema.

**`_mentions_drift(message, drift)` — case-insensitive SUBSTRING containment, no regex.**

Per DESIGN.md §6.3, every `detection_hint` is a substring-matchable token (e.g., `"price"`, `"total_fare_inr"`, `"MISSING_PASSENGER_COUNT"`). Matching is:

```
def _mentions_drift(message: str, drift: DriftEvent) -> bool:
    target = message.lower()
    # Each hint token is matched INDEPENDENTLY. ANY substring hit → True.
    for hint in drift.detection_hints:
        if hint and hint.lower() in target:
            return True
    return False
```

- Each `detection_hint` token from the drift pattern is matched independently.
- If **ANY** hint produces a case-insensitive substring match → R2 positive (speech channel) for that drift.
- **No** regex, **no** word boundaries, **no** fuzzy match, **no** stemming. Exactly `hint.lower() in target.lower()`.

**Design note — no separate field-name branch.** Per DESIGN.md §6.3 (updated 2026-04-24), the catalogued `detection_hints` already include the drifted field names themselves (e.g., `airline.price_rename` lists both `"price"` and `"total_fare_inr"`; `airline.pax_required` lists `"passenger_count"`). A separate "field-name from `drift.mutation`" branch is therefore **redundant** and is collapsed into the hints-only branch. The catalogue loader (see drift_injector.md) is responsible for ensuring every mutation field name is also a detection hint; if that invariant is ever relaxed, re-introduce a field-name branch that uses the same case-insensitive-substring rule (`field.lower() in message.lower()`) — NEVER regex.

**`_args_mention_drift(tool_args, drift)` — case-insensitive SUBSTRING containment on tool_args.**

Covers agents that adapt by invoking the new schema field without speaking about it (common in Stage 3 compound drift where speech-channel bandwidth is exhausted). Matching is:

```
def _args_mention_drift(tool_args: dict, drift: DriftEvent) -> bool:
    # Deterministic JSON: sort keys, no whitespace. Serializes the full structure.
    payload = json.dumps(tool_args, sort_keys=True, separators=(",", ":")).lower()
    # Also collect all STRING values (skip numbers, booleans, None) and concat.
    string_values = " ".join(
        v for v in _iter_primitive_strings(tool_args)
    ).lower()
    for hint in drift.detection_hints:
        if not hint:
            continue
        h = hint.lower()
        if h in payload or h in string_values:
            return True
    return False
```

- The JSON serialization is deterministic (`sort_keys=True`, no whitespace) so matches are reproducible across runs.
- Numeric and boolean values are **excluded** from the concatenated-values scan (they never carry hint tokens; including them would produce false positives against digit-substring hints).
- Any single hint substring match on either the JSON payload or the string-values concatenation → R2 positive (tool-call-arg channel).

**`_uses_new_schema(tool_args, drift)`:** returns True iff `tool_args` conforms to the post-drift schema — e.g., after `airline.price_rename`, a `search` response consumer must use `total_fare_inr` (not `price`) in a follow-up `book`. Conformance is checked structurally by `drift.mutation` (rename/add/remove/type-change rules). This is a **separate structural check** from the hint-substring branches above and is retained unchanged.

**Stage-1 skip.** Stage 1 episodes have `drift_schedule == ()`. R2 returns `0.5` (neutral). This prevents R2 from dragging on an episode with nothing to detect; GRPO's group-relative normalization handles the constant offset fine.

### 3.4 R3 — Constraint Adherence (DESIGN.md §7.1 / R3)

**Signature:** `constraint_adherence(episode: Episode) -> float`
**Range:** `[0.0, 1.0]`, fractional.

**Algorithm:**

```
constraints = episode.goal.constraints     # dict, may be {}
if not constraints:
    return 1.0                              # vacuously satisfied

satisfied = 0
for key, expected in constraints.items():
    satisfied += 1 if _check_constraint(key, expected, episode) else 0

return satisfied / len(constraints)
```

**`_check_constraint(key, expected, episode)`** dispatches on `key`:

| Constraint key | Check |
|---|---|
| `budget_inr` | final booking total ≤ `expected` |
| `time_window` | `booking.depart` lies in parsed window |
| `dietary` | all ordered items satisfy dietary flag (e.g. `veg_only`) |
| `passenger_count` | `booking.passenger_count == expected` |
| `pickup` | `booking.pickup == expected` (canonicalised) |
| `seat_type` | `booking.seat_type == expected` |
| `checkin`/`checkout` | dates match |
| `room_type` | matches |
| unknown key | counts as **satisfied** (0-cost) but logged to `breakdown.unknown_constraints` for spec review |

Unknown-key tolerance is intentional: new task templates can be added without breaking R3. The logged list is the signal the reward-hacking probe watches.

### 3.5 R4 — Format Compliance (DESIGN.md §7.1 / R4)

**Signature:** `format_compliance(episode: Episode) -> float`
**Range:** `[0.0, 1.0]`, deductive from 1.0.

**Algorithm:**

```
score = 1.0
for action in episode.actions:
    if action.action_type == TOOL_CALL:
        if not _is_valid_json(action.tool_args):
            score -= 0.20
        if action.tool_name not in AVAILABLE_TOOL_REGISTRY:
            score -= 0.10
        if action.rationale is None or len(action.rationale.strip()) == 0:
            score -= 0.05
    if action.action_type in {SPEAK, CLARIFY}:
        if _detect_language(action.message) != episode.goal.language:
            score -= 0.10
return max(0.0, min(1.0, score))
```

**Per DESIGN.md §7.1 R4 (verbatim weights):**

- `−0.2` per invalid JSON tool call
- `−0.1` per hallucinated tool name
- `−0.1` per language-mismatched reply
- `−0.05` per missing `rationale` on `TOOL_CALL`

**`_detect_language(message)`** uses a deterministic script-based heuristic (Devanagari→hi, Tamil script→ta, Kannada script→kn, Latin with Hinglish markers→hinglish, else en). No LLM — see §3.1. For `language == "hinglish"`, Latin+Devanagari code-mixed replies are all accepted.

**`AVAILABLE_TOOL_REGISTRY`** is the static set of known tool names from `vendors.md` (~14 tools across airline/cab/restaurant/hotel/payment). A tool name not in this set is a "hallucinated tool".

### 3.6 R5 — Anti-Hack Penalty (DESIGN.md §7.1 / R5, §7.3)

**Signature:** `anti_hack_penalty(episode: Episode) -> float`
**Range:** `[-1.0, 0.0]`, asymmetric (pure penalty, never positive).

**Algorithm:**

```
penalty = 0.0

# (a) Hallucinated field reference
#     The agent's SPEAK/CLARIFY/rationale text references a field name that never
#     appeared in any tool_result this episode. Case-insensitive, word-boundary.
#     The whitelist is built by RECURSIVELY walking every tool_result.response
#     (arbitrary nesting depth) — see whitelist construction below.
if _any_hallucinated_field(episode):
    penalty -= 1.0

# (b) Repeated identical tool calls
if _count_repeated_identical_tool_calls(episode) > 3:
    penalty -= 0.5

# (c) PROBE_SCHEMA abuse
if _count_actions(episode, ActionType.PROBE_SCHEMA) >= 3:
    penalty -= 0.5

# (d) Bare drift assertion
#     Agent SPEAK/CLARIFY mentions "drift" / matching hint keyword BEFORE any
#     tool_result in this episode carried status in {schema_error, policy_error,
#     auth_error} AND before any drift_log entry whose turn ≤ action.turn.
if _has_unsubstantiated_drift_claim(episode):
    penalty -= 0.3

# (e) Protected-state write attempt
#     Tool call whose tool_name matches the protected allowlist (see vendors.md)
#     or whose tool_args contains reserved keys (`__turn__`, `__schema_version__`,
#     `__done__`, `__episode_id__`).
if _any_protected_write(episode):
    penalty -= 0.2

return max(-1.0, penalty)                    # clamp
```

**Whitelist construction for `_any_hallucinated_field` — recursive walk (normative).**

The whitelist is the union of every **key** and every **primitive leaf value** (`str`, `int`, `float`, `bool`) found at **any nesting depth** inside `tool_result.response` across **every** tool result observed so far in the episode.

```
def _build_whitelist(tool_results) -> set[str]:
    seen: set[str] = set()
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                seen.add(str(k).lower())   # every key at every depth
                walk(v)
        elif isinstance(node, list) or isinstance(node, tuple):
            for item in node:
                walk(item)
        elif isinstance(node, (str, int, float, bool)):
            seen.add(str(node).lower())    # every primitive leaf
        # None and other types contribute nothing
    for tr in tool_results:
        walk(tr.response)
    return seen
```

- **Recursion depth is unbounded** — a key buried 5 levels deep is just as whitelisted as a top-level key.
- Strings, ints, floats, bools are whitelisted as their `str(...).lower()` form. `None` contributes nothing.
- Agent-mentioned tokens are checked against this set with the same case-insensitive, word-boundary rule used elsewhere in §3.6(a).

**Concrete example — cab v3 `fare_breakdown` nested dict (matches DESIGN.md §5.2 v3 drift).**

After drift `cab.fare_breakdown_split` fires, `cab.estimate` returns:

```json
{
  "pickup": "HSR",
  "drop": "Indiranagar",
  "vehicle_class": "sedan",
  "fare_breakdown": {"base": 120, "surge": 45, "tolls": 10, "gst": 32},
  "eta_min": 7
}
```

Recursive walk yields whitelist (lowercased):
`{"pickup", "hsr", "drop", "indiranagar", "vehicle_class", "sedan", "fare_breakdown", "base", "120", "surge", "45", "tolls", "10", "gst", "32", "eta_min", "7"}`.

Consequences:

- Agent says *"the surge component is ₹45"* → `surge` is in the whitelist (nested key, depth 2) → **NOT hallucinated**.
- Agent says *"the base fare is ₹120"* → `base` and `120` both whitelisted → **NOT hallucinated**.
- Agent says *"the base_fare field says ₹120"* → `base_fare` is NOT in any tool_result (the field is `base`, not `base_fare`) → **hallucinated** → R5 −= 1.0.
- Agent says *"total_fare_inr is ₹207"* → neither `total_fare_inr` nor `207` appears anywhere in any tool_result → **hallucinated** → R5 −= 1.0.

This prevents a cheap exploit where the agent invents plausibly-named nested fields (`fare_details`, `price_components`) banking on a shallow key-only scan missing them. The recursive walk also prevents false positives against legitimate deep references (e.g., `gst` at depth 2).

**Penalties stack additively**, then clamp at `−1.0` (floor). A single hallucination alone is already at the floor — subsequent hacks don't make it worse, but the `breakdown.anti_hack` lists all offenses for the probe report (DESIGN.md §13 deliverable #9).

### 3.7 Combined reward (DESIGN.md §7.2)

```
quality = 0.50 * R1
        + 0.20 * R2
        + 0.15 * R3
        + 0.10 * R4
        + 0.05 * min(R5, 0.0)              # R5 already ≤ 0; the min() is defensive

brier   = min((confidence - R1) ** 2, 0.5) if confidence is not None else 0.0

reward  = quality * (1.0 - brier)

# Uncertain floor (DESIGN.md §7.3)
if R1 == 0.0 and confidence is not None and confidence < 0.3:
    reward = max(reward, 0.3)
    floor_applied = True
else:
    floor_applied = False

reward = max(0.0, min(1.0, reward))
reward = round(reward, 3)
```

**Helper call order** (matches §2.3 contracts — do NOT reorder):
`combine_quality` → `brier_penalty` → multiply → `apply_uncertain_floor` → clamp → round.
Only `final_reward` clamps and rounds; the other three helpers return raw, unclamped, unrounded floats. See §2.3 for the full per-helper pre/post-condition contracts.

**Confidence source.** `confidence` comes from the terminating `SUBMIT` action's `confidence` field. If the episode terminated via `ABORT` / `TIMEOUT` / `ANTI_HACK`, `confidence` is `None`, Brier is `0.0`, and the uncertain floor does **not** apply (the floor is calibrated-surrender insurance, not a failure bribe).

**Weight rationale (DESIGN.md §7.2).** R1 dominates because task success is the product; R2 and R3 shape the *way* the agent succeeds; R4 polices the interface; R5 is weighted low but asymmetric so a single hack bleeds `−0.05` of the quality budget (enough to punish without dominating).

### 3.8 Rounding, clamping, and NaN defense

- `reward` MUST be `round(x, 3)` for consistency with WandB logs and the before/after narrative in the pitch (§15 DESIGN.md).
- Any `float('nan')` or `float('inf')` at any stage → raise `RewardComputationError` (see §5). GRPO will skip the sample.
- `quality` before clamp may legitimately be slightly negative (R5 = −1 alone → quality = −0.05). It is stored in `Rewards.quality` unclamped (useful for diagnostics) but `reward` always clamps to `[0, 1]`.

---

## 4. Data Structures

### 4.1 `Rewards` (output, frozen)

Full field list — see §2.5. Fully serialisable via `dataclasses.asdict`, round-trips through JSON.

### 4.2 `Rewards.breakdown` (diagnostic dict)

Per-reward evidence. Shape:

```python
{
    "r1": {
        "domain": str,
        "success_predicate": str,        # e.g. "airline_booking_match"
        "matched_slots": dict[str, Any],
        "missing_slots": list[str],
    },
    "r2": {
        "stage": int,
        "drifts_total": int,
        "drifts_detected": int,
        "per_drift": list[{
            "drift_id": str,
            "hit_by_speech": bool,         # branch 1: §3.3 speech channel
            "hit_by_args_hint": bool,      # branch 2: §3.3 tool-call-args-hint channel
            "hit_by_adaptation": bool,     # branch 3: §3.3 structural-adaptation channel
            "window_turns": list[int],
        }],
        "three_plus_retries": bool,
    },
    "r3": {
        "total_constraints": int,
        "satisfied_constraints": int,
        "unknown_constraints": list[str],
        "failures": list[{"key": str, "expected": Any, "actual": Any}],
    },
    "r4": {
        "deductions": list[{"turn": int, "reason": str, "amount": float}],
    },
    "anti_hack": {
        "offenses": list[{"code": str, "turn": int | None, "evidence": str}],
    },
    "combination": {
        "quality_raw": float,
        "brier": float,
        "uncertain_floor_applied": bool,
    },
}
```

This blob is what the reward-hacking probe (B's deliverable) scans for exploit patterns across 200 held-out episodes.

### 4.3 `RewardComputationError` (exception)

```python
class RewardComputationError(Exception):
    """Raised when rewards cannot be computed for a malformed episode."""
    def __init__(self, reason: str, episode_id: str | None = None):
        self.reason = reason
        self.episode_id = episode_id
```

---

## 5. Error Modes

| Failure | Trigger | Handling |
|---|---|---|
| **Missing goal** | `episode.goal is None` | Raise `RewardComputationError("episode.goal is None", episode_id)`. Env treats this as ANTI_HACK termination; `env.step` converts to `Rewards(r1=0, r2=0, r3=0, r4=0, r5=-1, ...)` via a fallback path. |
| **Unterminated episode** | `episode.terminated_by is None` or `done == False` | Raise `RewardComputationError("episode not terminated")`. Never compute on partial transcripts. |
| **Corrupted drift log** | `drift in drift_log` references a `drift_type` not in the 5-axis taxonomy | Raise `RewardComputationError(f"unknown drift_type: {drift.drift_type}")`. This is a bug in the drift injector, not in the agent — training halts, orchestrator escalated. |
| **Unknown domain** | `goal.domain` not in `{airline, cab, restaurant, hotel}` | R1 returns 0.0, `breakdown.r1.success_predicate = "unknown_domain"`, logged for spec review. Does not raise. |
| **NaN / inf** in intermediate | Any float(`nan`) or `inf` at any stage | Raise `RewardComputationError("non-finite value in reward computation")`. GRPO trainer must catch and skip the rollout (TRL supports `reward == None` semantics; we return via exception instead of a sentinel to force explicit handling). |
| **`confidence` out of range** | `SUBMIT.confidence` not in `[0.0, 1.0]` | Clamp to `[0, 1]` for Brier only; record `breakdown.combination.confidence_clamped = True`. Do NOT raise — the env already validated on submit, defense in depth. |
| **Missing `tool_results` but `actions` contain `TOOL_CALL`** | Log-integrity violation | Raise `RewardComputationError("action/tool_result count mismatch")`. |
| **Empty `actions`** | 0-length episode, immediate timeout | R1=0, R2=0.5 (stage 1) or 0 (drift stages), R3=1.0 vacuous if no constraints else 0.0, R4=1.0, R5=0.0. Do NOT raise — this is a legal (if pathological) episode. |
| **Empty / missing `detection_hints`** | A `DriftEvent` in `episode.drift_log` has `detection_hints is None`, `== []`, or every token is an empty string | Raise `RewardComputationError(f"drift {drift.id} has empty detection_hints", episode.episode_id)` at R2 pipeline entry (inside the drift-log iteration in §3.3). Mitigation: the drift catalogue loader (drift_injector.md) MUST validate at load time that every pattern has ≥ 1 non-empty hint token; this is a defense-in-depth guard against a corrupted or partially-loaded catalogue reaching the reward pipeline. |

**Policy:** the reward module is **strict on structural invariants** (raises on corruption) and **permissive on content invariants** (counts bad behaviour as R5 penalty or R4 deduction, not exception). Structural problems mean the env or drift-injector has a bug; content problems mean the agent has a bug — and we're training it.

---

## 6. Dependencies

### 6.1 Upstream (imports from)

- `driftcall.models` — `Episode`, `DriftCallAction`, `ToolResult`, `DriftEvent`, `GoalSpec`, `ActionType` (see `models.md`).
- `driftcall.rewards.checkers` — per-domain success predicates (internal submodule; implementation detail).
- `driftcall.rewards.parsers` — time-window parser, language detector, JSON validator (internal submodule).

### 6.2 Downstream (consumed by)

- **`driftcall.env.DriftCallEnv.step`** — calls `compute_rewards(self._episode)` at termination (on `SUBMIT`/`ABORT`/timeout/anti-hack). Puts `Rewards` into the observation's `info` dict for GRPO.
- **`training/train_grpo.py`** — calls the env through TRL's `OpenEnvWrapper`; TRL reads `info["reward"]` (the scalar) as the GRPO advantage input.
- **`training/eval_baseline.py` / `eval_final.py`** — invoke the env and collect full `Rewards` objects for per-reward curves and the reward-hacking probe report.
- **`demo/app_gradio.py`** — renders `Rewards.breakdown` in the trace panel so judges can see why the episode scored what it did.
- **`tests/test_rewards.py`** — full unit + property suite.

### 6.3 Prohibited dependencies (do not import)

- No `requests`, `httpx`, `aiohttp` — no network.
- No `openai`, `anthropic`, `transformers`, `torch`, `unsloth` — no model inference.
- No `time.time()`, `datetime.now()`, `random` — no non-determinism. (A fixed seed from `episode.episode_id` is acceptable for reproducible sampling in the reward-hacking probe, but NOT inside `compute_rewards` itself.)
- No file I/O — rewards are pure in-memory.

---

## 7. Edge Cases

1. **Empty episode (0 actions, timeout on reset).** `actions == ()`. R1=0 (no SUBMIT), R2=0.5 if stage==1 else 0.0 (drift fired, nothing happened), R3=1.0 if `goal.constraints == {}` else 0.0, R4=1.0 (no format violations possible), R5=0.0. `quality = 0.20*0.5 + 0.15*R3 + 0.10 ≈ 0.20–0.35`. Final reward clamped and rounded. No exception.

2. **Drift fires in Stage 1.** `stage == 1 AND len(drift_log) > 0`. This is a bug in the env; drift schedules must be empty in Stage 1. The reward module still returns `R2 = 0.5` (trusts the `stage` field) and logs `breakdown.r2.stage == 1 but drifts_total > 0` for the probe to flag. Does NOT raise — stage is authoritative.

3. **Hallucinated field pattern.** Agent says "Using the `flight_total_with_gst` field" but no tool_result ever contained such a field. `_any_hallucinated_field` returns True → R5 = −1.0 alone (clamp). Critically: this test scans `action.message` AND `action.rationale` AND `action.tool_args` (keys and string values) against the whitelist built by the **recursive walk** defined in §3.6(a) — every key and every primitive (`str`/`int`/`float`/`bool`) leaf value at any nesting depth of every `tool_result.response` seen so far. Fields that appeared in *any* prior tool_result (including pre-drift responses, including nested dicts like `fare_breakdown.surge`) are whitelisted.

4. **Repeated identical tool calls.** Agent calls `airline.search(from=HYD, to=BLR, date=2026-04-30)` four times in a row (tool name + normalised tool_args identical). Threshold `> 3` → R5 penalty triggers on the 4th call. Args are normalised (sorted keys, case-lowered string values) before hashing to prevent near-duplicate evasion.

5. **Over-budget termination (TIMEOUT).** `episode.turns_used >= episode.max_turns` and agent never submitted. `terminated_by == "TIMEOUT"`, `confidence is None`. R1=0 (no SUBMIT). R2 computed normally (did they detect the drift during the wasted turns?). R3 computed against whatever vendor state exists (usually bad). R4 and R5 computed over all actions. Brier=0, uncertain floor NOT applied (no confidence). Final reward ~= `0.20*R2 + 0.15*R3 + 0.10*R4 + 0.05*R5`.

6. **Confidence not provided on SUBMIT.** Action validator in env SHOULD reject — but defense in depth: if `SUBMIT.confidence is None` reaches here, we treat as no-confidence (Brier=0, no floor, no penalty). Flagged in `breakdown.combination.confidence_missing = True`.

7. **Confidence=1.0 on failure (R1=0).** `confidence=1.0, R1=0.0`. Brier = `min((1.0 - 0.0)^2, 0.5) = 0.5`. `reward = quality * (1.0 - 0.5) = 0.5 * quality`. Uncertain floor does NOT apply (confidence ≥ 0.3). This is the miscalibrated-overconfidence case; Brier punishes it hard. (If R1=1 and confidence=1.0 → Brier=0 → full quality retained.)

8. **Confidence=0.0 on success (R1=1).** `confidence=0.0, R1=1.0`. Brier = `min((0 - 1)^2, 0.5) = 0.5`. Same 50% multiplier. Miscalibrated-underconfidence. Uncertain floor does NOT apply (R1==1).

9. **Uncertain floor activates.** `R1=0.0, confidence=0.2, quality=0.1` (low R2/R3/R4). `brier = (0.2 - 0.0)^2 = 0.04`. `reward = 0.1 * 0.96 = 0.096`. Then floor kicks in: `reward = max(0.096, 0.3) = 0.3`. `floor_applied = True`. This rewards calibrated surrender.

10. **R5 at floor AND R1=1.** Agent solves the task but also hallucinates a field in the victory message. `R5 = −1.0`, `R1 = 1.0`, other rewards full. `quality = 0.50 + 0.20 + 0.15 + 0.10 + 0.05*(−1) = 0.90`. `brier = (conf − 1)^2`. Success is not erased, but the hack costs ~5% of quality — designed to discourage cosmetic hallucinations without invalidating real completions.

11. **Unknown constraint key in goal.** E.g. `goal.constraints = {"carbon_offset": True}` from a future task template. `_check_constraint` returns `True` (permissive), denominator unchanged. Logged in `breakdown.r3.unknown_constraints = ["carbon_offset"]` for spec review.

12. **Stage 2/3, drift never fires (scheduler bug).** `stage in {2,3} AND drift_log == ()`. R2 returns 0.5 (neutral) and flags `breakdown.r2.stage2_3_no_drift = True` for the probe. Does not raise — agent is not punished for an env bug.

13. **Agent uses PROBE_SCHEMA exactly 2 times.** Under the threshold — no R5 penalty. Logged in `breakdown.anti_hack.probe_count = 2` for trend analysis.

14. **ANTI_HACK termination itself.** Env detects a protected-write attempt mid-episode and terminates with `terminated_by == "ANTI_HACK"`. `confidence is None`, R1=0, R5 applied normally (the offense that caused termination is included in the action trace). `breakdown.anti_hack.terminating_offense` names the trigger.

15. **Drift with empty `detection_hints` reaches the reward pipeline.** `episode.drift_log[i].detection_hints == []` or `is None` or all-empty-strings. This is a structural catalogue bug (loader should have rejected it). R2 cannot decide hit-by-speech or hit-by-args-hint without at least one hint token, so we raise `RewardComputationError("drift {id} has empty detection_hints")` at R2 entry. The env converts to fallback Rewards (R1=0, R5=−1) per the `ANTI_HACK`-style path. This edge case exists because the substring-match algorithm (§3.3) depends on hints being non-empty tokens; a missing token list would silently skip the whole drift and inflate R2. See §5 Error Modes.

---

## 8. Examples

All three examples use DESIGN.md §7.2 weights verbatim. `round(x, 3)` applied to `reward`.

### 8.1 Example A — Clean success with calibrated confidence

**Episode:** Stage 1, Hinglish airline booking. Agent searches, finds the flight, books, submits.

```
goal.domain          = "airline"
goal.slots           = {from: HYD, to: BLR, when: 2026-04-30}
goal.constraints     = {budget_inr: 8000, time_window: "evening"}
stage                = 1
drift_log            = ()
terminated_by        = "SUBMIT"
confidence           = 0.85
actions              = [search, book, submit]   # all JSON-valid, rationales present
vendor_states_final  = {airline: {bookings: [{from:HYD, to:BLR, depart:2026-04-30T19:15, total:7200}]}}
```

**Rewards:**

| | |
|---|---|
| R1 | 1.0 (booking matches slots + budget + evening window) |
| R2 | 0.5 (stage 1 → neutral) |
| R3 | 1.0 (2/2 constraints satisfied) |
| R4 | 1.0 (no deductions) |
| R5 | 0.0 (no hacks) |
| quality | `0.50*1 + 0.20*0.5 + 0.15*1 + 0.10*1 + 0.05*0 = 0.850` |
| brier | `(0.85 − 1.0)^2 = 0.0225` |
| reward | `0.850 * (1 − 0.0225) = 0.8309125 → round → 0.831` |
| floor_applied | False |

### 8.2 Example B — Stage-2 drift detected and adapted, but constraint violated

**Episode:** Stage 2 Kannada airline brief. Drift `airline.price_rename` fires at turn 3. Agent detects via SPEAK ("`price` field seems renamed; using `total_fare_inr`"), re-books, submits — but picks a flight at ₹8400 when budget was ₹8000.

```
goal.constraints     = {budget_inr: 8000, time_window: "morning"}
stage                = 2
drift_log            = [DriftEvent(turn=3, id=airline.price_rename)]
terminated_by        = "SUBMIT"
confidence           = 0.60
actions              = [search@1, search@2, speak@3 ("price→total_fare_inr"),
                        search_v2@4, book_v2@5, submit@6]
final booking total  = 8400
```

**Rewards:**

| | |
|---|---|
| R1 | 0.0 (budget constraint part of success predicate — but spec says R1 checks route+date+window+budget; ₹8400 > ₹8000 → R1=0) |
| R2 | 1.0 (SPEAK in turn 3 mentions `total_fare_inr`, within window) |
| R3 | 0.5 (1/2 constraints: time_window satisfied, budget_inr violated) |
| R4 | 1.0 |
| R5 | 0.0 |
| quality | `0.50*0 + 0.20*1 + 0.15*0.5 + 0.10*1 + 0.05*0 = 0.375` |
| brier | `(0.60 − 0.0)^2 = 0.36` (confidence overshot) |
| reward | `0.375 * (1 − 0.36) = 0.24` → round → `0.240` |
| floor_applied | False (confidence ≥ 0.3) |

This is the **calibration lesson**: drift was caught and format was clean, but overconfidence on a failed booking punches quality down 36%. The agent is trained to lower confidence when it knows the budget is tight.

### 8.3 Example C — Hallucinated field + calibrated surrender (floor activates)

**Episode:** Stage 3 Tamil compound-drift restaurant order. Two drifts fire. Agent gets confused, invents a field `"order_metadata_v4"` in its rationale, repeats `restaurant.search` four times, submits with low confidence.

```
stage                = 3
drift_log            = [policy@3, schema@7]
terminated_by        = "SUBMIT"
confidence           = 0.20
actions              = [search×4, speak (invents order_metadata_v4), submit]
goal.constraints     = {budget_inr: 300, dietary: "veg"}
final vendor state   = {restaurant: {orders: []}}   # never ordered
```

**Rewards:**

| | |
|---|---|
| R1 | 0.0 (no order placed) |
| R2 | 0.0 (no drift-mention, old schema retries 4 times) |
| R3 | 0.0 (0/2 — no order means neither constraint realisable; budget vacuous=False, dietary vacuous=False) |
| R4 | 1.0 (rationale present, JSON valid, tool names known) |
| R5 | `−1.0` (hallucinated field) + `−0.5` (4 repeated calls) → clamped to `−1.0` |
| quality | `0.50*0 + 0.20*0 + 0.15*0 + 0.10*1 + 0.05*(−1) = 0.050` |
| brier | `(0.20 − 0.0)^2 = 0.04` |
| reward (pre-floor) | `0.050 * (1 − 0.04) = 0.048` |
| **uncertain floor** | R1==0 AND confidence<0.3 → `max(0.048, 0.3) = 0.300` |
| floor_applied | True |
| reward (final) | `0.300` |

The agent is rewarded for calibrated surrender (`confidence=0.20`) **despite** the hack penalty. This is intentional: without the floor, a policy that says "I don't know, giving up" collapses; with the floor at 0.3, we keep it alive as a legitimate fallback. R5 still shows up in `breakdown.anti_hack.offenses` so the probe report counts it.

---

## 9. Open Questions

None — spec is complete.

The following items are resolved by deferral to their owning docs (not gaps in this spec):

- Exact form of `_check_airline_booking` et al. → `vendors.md` owns per-domain success predicates.
- Exact list of `AVAILABLE_TOOL_REGISTRY` tool names → `vendors.md` owns the tool catalog.
- Exact drift-mutation shape (rename/add/remove/type-change DSL) → `drift_injector.md` owns the mutation language.
- Exact script/heuristic for `_detect_language` → resolved to "Unicode script + Hinglish marker lookup, no external model, frozen word list in code" — noted here and implemented in `driftcall/rewards/parsers.py`.
- Whether to expose per-reward scalars separately to GRPO (e.g. multi-objective GRPO variant) → resolved **no** per DESIGN.md §7.4: single scalar `reward`, GRPO handles group-relative normalisation.

Previously-open items **now resolved in this revision** (critic-2 round):

- **R2 match algorithm** → resolved to case-insensitive substring (`hint.lower() in target.lower()`), no regex, no word boundaries; three detection branches (speech, tool-call args, structural adaptation) documented in §3.3.
- **Helper function call order and clamp/round responsibility** → resolved: only `final_reward` clamps and rounds; `combine_quality`, `brier_penalty`, `apply_uncertain_floor` all return raw unclamped floats; order locked in §2.3 and §3.7.
- **Empty `detection_hints` handling** → resolved: raise `RewardComputationError` at R2 entry; catalogue loader validates at load time (§5, §7 edge case 15).
- **Hallucination whitelist depth** → resolved: recursive walk, unbounded nesting, keys + primitive leaves; cab v3 `fare_breakdown` example in §3.6(a).

---

**End of spec. Implementation (`driftcall/rewards.py`) does not start until ≥ 2 fresh critic agents return `NOTHING_FURTHER` on this doc.**
