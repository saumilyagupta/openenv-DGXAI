# drift_injector.md — Drift Injection Subsystem

**Module:** `driftcall/drift_injector.py`
**Owner:** Person A (Environment)
**Implements:** DESIGN.md §4.3 (step semantics — drift trigger point), §6 (full module), §7.1 (R2 detection signal)
**Status:** Design spec — pre-critic-gate

---

## 1. Purpose

The Drift Injector is the subsystem that makes DriftCall *DriftCall*. It is responsible for:

1. **Scheduling** a deterministic sequence of `DriftEvent`s for each episode at `reset()` time, based on the curriculum stage and the episode seed.
2. **Applying** each scheduled drift to the immutable `DriftCallState` at the start of the scheduled turn, mutating the relevant vendor's schema version, business policy, T&C text, pricing structure, or auth scope — and producing a new state with the drift appended to `drift_fired`.
3. **Cataloguing** the 20 canonical drift patterns (per DESIGN.md §6.3 explicit enumeration: 5 schema + 5 policy + 5 T&C + 3 pricing + 2 transversal payment-auth) loaded from `data/drift_patterns/drifts.yaml`, keyed by `pattern_id` and exposing a stable discovery interface so the test harness and R2 scorer can enumerate them.

The injector is the **only** code path allowed to change `schema_versions` or vendor-state shape mid-episode. Vendors themselves are pure — they read whichever schema version the injector has installed. This keeps the mutation surface one-file-deep and auditable, which is required for the anti-hack guarantees in DESIGN.md §7.3.

The injector's output is both the direct trigger for state mutation *and* the ground truth against which R2 (drift detection) is scored — every `DriftEvent` carries `detection_hints` (substring-matchable tokens per DESIGN.md §6.3) that the R2 reward scorer matches case-insensitively against the agent's subsequent `SPEAK` / `CLARIFY` text and against its subsequent `TOOL_CALL` argument strings (DESIGN.md §7.1 R2).

---

## 2. Interface

All signatures are frozen (pure functions / frozen dataclasses). Every mutation returns a new `DriftCallState`; no in-place modification.

```python
from __future__ import annotations
from typing import Protocol

def build_schedule(
    stage: int,
    episode_seed: int,
    goal: GoalSpec,
) -> tuple[DriftEvent, ...]:
    """
    Build the full drift schedule for an episode at reset() time.

    Determinism: (stage, episode_seed, goal.domain) → identical tuple every call.
    Stage-based count (DESIGN.md §6.2):
        stage == 1 → ()                  # empty tuple, no drifts
        stage == 2 → (one DriftEvent,)   # exactly one
        stage == 3 → (event_a, event_b)  # exactly two, different pattern_ids,
                                         # different turns, staggered

    Turn placement rules:
        stage 2: turn ∈ [2, max_turns - 3]  (RNG seeded on episode_seed)
        stage 3: first in [2, max_turns // 2], second in [first+2, max_turns - 3]

    Domain selection:
        - First drift always targets `goal.domain` (the primary vendor).
        - Second drift (stage 3) may target `goal.domain` OR `payment`
          (payment cross-cuts airline/hotel flows; see §7 edge case on cascades).

    Returns: a tuple of DriftEvent in turn-ascending order.
    Raises: ValueError on stage ∉ {1, 2, 3}.
    """

def apply_drift(
    state: DriftCallState,
    event: DriftEvent,
) -> DriftCallState:
    """
    Apply a single drift event to an immutable state and return the new state.

    Behavior (DESIGN.md §4.3 step #3):
        1. Look up the DriftPattern by event.pattern_id in the registry.
        2. Execute pattern.mutation against a deep copy of state.vendor_states[event.domain]
           (mutation ops: rename / remove / require_new_field / enum_expand /
            numeric_bump / policy_flag_flip / tnc_text_swap / pricing_restructure /
            auth_scope_bump — full enum in §4 Data Structures).
        3. Update schema_versions[event.domain] = event.to_version.
        4. Append event to drift_fired (tuple, preserving order).
        5. Return new frozen DriftCallState with these updates;
           all other fields unchanged.

    Idempotence: calling apply_drift twice with the same event on the same state
    is detected and raises DriftReapplicationError (§5). The injector is trusted
    to only call this once per scheduled event — the guard is defence-in-depth.

    Returns: new DriftCallState (frozen).
    Raises:
        UnknownDriftPatternError: event.pattern_id not in registry
        DriftDomainMismatchError: event.domain not present in state.vendor_states
        DriftReapplicationError:  event already present in state.drift_fired
    """

def list_patterns() -> tuple[DriftPattern, ...]:
    """
    Return all 20 registered drift patterns, sorted by pattern_id (stable ordering).

    Used by:
      - tests/test_drift.py — enumerates every pattern, fires each in isolation,
        verifies detection_hints trip R2.
      - R2 scorer (rewards.py) — retrieves pattern.detection_hints by pattern_id.
      - MCP discovery / openenv introspection — reports the pattern catalogue.

    Patterns are loaded once on first call from data/drift_patterns/drifts.yaml
    and cached in a module-level tuple. Concurrency-safe because the tuple is
    immutable after load.

    Returns: tuple[DriftPattern, ...] of length 20.
    Raises: DriftCatalogueError if fewer than 20 patterns load (startup invariant).
    """
```

**Protocol for testability (optional seam):**

```python
class DriftScheduler(Protocol):
    """Abstract scheduler; the default is build_schedule above.
       Test harness may inject a scripted scheduler to force specific events."""
    def __call__(
        self, stage: int, episode_seed: int, goal: GoalSpec
    ) -> tuple[DriftEvent, ...]: ...
```

---

## 3. Behavior Spec

### 3.1 Firing Logic (when drift applies within a turn)

Per DESIGN.md §4.3, the `step()` sequence is:

```
turn_before  = state.turn
turn_current = turn_before + 1         # step 2 in DESIGN.md §4.3
pending = [e for e in drift_schedule if e.turn == turn_current and e not in drift_fired]
for e in pending:                      # step 3 in DESIGN.md §4.3
    state = apply_drift(state, e)
# THEN the agent's action is evaluated (step 4):
state = dispatch_action(state, action)
```

**Invariant: drift applies at the START of a scheduled turn, BEFORE the agent's action for that turn is evaluated.** This matters because it means if the drift scheduled for turn 4 renames `price` → `total_fare_inr`, and the agent's turn-4 action is `airline.search`, the *response* to that search already uses the new schema. The agent is expected to detect this on the *next* turn via R2's 2-turn window (DESIGN.md §7.1 R2).

### 3.2 Stage-based Drift Count

| Stage | Drift count | Placement | Notes |
|---|---|---|---|
| 1 | 0 | N/A | `build_schedule` returns `()`. `apply_drift` is never invoked. R2 is neutral (0.5) per DESIGN.md §7.1 R2. |
| 2 | exactly 1 | random turn ∈ [2, max_turns−3] | All 5 drift types are eligible; domain == goal.domain. Random selection seeded. |
| 3 | exactly 2 | staggered across episode | Different `pattern_id`s. Second may cross domains (payment). Distance ≥ 2 turns between them. |

### 3.3 Schedule Determinism

- The scheduler is a pure function of `(stage, episode_seed, goal)`. Two `reset()` calls with the same seed produce byte-identical schedules.
- The RNG is a `random.Random(episode_seed)` instance local to the scheduler — never the global RNG — so parallel episode construction doesn't race.
- The schedule is frozen into `state.drift_schedule` at `reset()` and never consulted again beyond the firing check in `step()`. It is visible to the R2 scorer as ground truth.

### 3.4 Mutation Types (mapped to the 5 drift types)

Each `DriftPattern` carries a `mutation: dict[str, Any]` field. The injector dispatches on mutation operator keys:

| Drift type | Mutation operators | Applied to |
|---|---|---|
| `schema` | `rename`, `remove`, `require_new_field`, `change_type` | `state.vendor_states[domain]['schema']` |
| `policy` | `numeric_bump` (min/max), `enum_expand`, `policy_flag_flip`, `time_window_shrink` | `state.vendor_states[domain]['policy']` |
| `tnc` | `tnc_text_swap`, `side_channel_notice_append` | `state.vendor_states[domain]['tnc']` + `state.vendor_states[domain]['side_channel']` |
| `pricing` | `pricing_restructure`, `fee_append` | `state.vendor_states[domain]['pricing']` |
| `auth` | `auth_scope_bump`, `token_version_bump` | `state.vendor_states['payment']['auth']` (always payment, cross-cutting) |

Vendors (DESIGN.md §5) read from `state.vendor_states[domain]` using whichever `schema_versions[domain]` is current. The injector is the sole writer.

### 3.5 Post-conditions of `apply_drift`

- `returned.turn == state.turn` (the injector does not advance the clock; `step()` does)
- `returned.schema_versions[event.domain] == event.to_version`
- `returned.drift_fired == state.drift_fired + (event,)`
- `len(returned.vendor_states) == len(state.vendor_states)` (no vendor added/removed mid-episode)
- `returned is not state` (new object; frozen dataclass `replace`)

---

## 4. Data Structures

### 4.1 `DriftEvent` (frozen, defined in models.md, referenced here)

```python
@dataclass(frozen=True)
class DriftEvent:
    turn: int
    drift_type: Literal["schema", "policy", "tnc", "pricing", "auth"]
    domain: str
    description: str
    from_version: str
    to_version: str
    pattern_id: str                          # key into the pattern registry
```

### 4.2 `DriftPattern` (frozen; loaded from YAML; module-internal)

Per DESIGN.md §6.3 schema:

```python
@dataclass(frozen=True)
class DriftPattern:
    id: str                                  # e.g. "airline.price_rename"
    drift_type: Literal["schema", "policy", "tnc", "pricing", "auth"]
    domain: str                              # "airline" | "cab" | "restaurant" | "hotel" | "payment"
    from_version: str                        # "v1" | "v2" | "v3"
    to_version: str
    description: str                         # human-readable, shown in drift_log
    mutation: Mapping[str, Any]              # operator-keyed dict (§3.4)
    detection_hints: tuple[str, ...]         # feeds R2 scoring (see §6)
```

All fields immutable. `mutation` is wrapped in `types.MappingProxyType` on load to prevent accidental write.

### 4.3 YAML Schema (on disk — `data/drift_patterns/drifts.yaml`)

```yaml
- id: airline.price_rename
  drift_type: schema
  domain: airline
  from_version: v1
  to_version: v2
  description: "field 'price' renamed to 'total_fare_inr'; 'currency' removed"
  mutation:
    rename:  {price: total_fare_inr}
    remove:  [currency]
  detection_hints:
    - "total_fare_inr"                       # new field name token
    - "price"                                # old field name token
    - "rename"                               # keyword token
```

Per DESIGN.md §6.3, `detection_hints` are **substring-matchable tokens** (short keywords / field names / enum values / error codes — not free-form sentences). R2 applies case-insensitive substring match against the agent's subsequent `SPEAK` / `CLARIFY` `message` and `rationale` fields, AND against any subsequent `TOOL_CALL.tool_args` JSON-serialized strings. Either text-path or tool-args-path match counts for R2=1 per DESIGN.md §7.1 R2.

### 4.4 Full Pattern Registry (20 patterns — ids only; full YAML in `data/drift_patterns/drifts.yaml`)

Per DESIGN.md §6.3 explicit enumeration — **20 patterns total, not a strict Cartesian product**: 5 schema + 5 policy + 5 T&C + 3 pricing (airline/cab/hotel; restaurant "pricing" collapses into the min-order policy pattern) + 2 transversal payment-auth patterns. Payment auth patterns cross-cut all four primary domains as cascade inducers.

**Schema drifts (5):**

1. `airline.price_rename` — `price` → `total_fare_inr`; `currency` removed (DESIGN.md §5.1 v1→v2).
2. `airline.pax_required` — booking requires new `passenger_count` field (§5.1 v2→v3).
3. `cab.fare_breakdown` — `fare_inr` replaced by nested `fare_breakdown: {base, surge, tolls, gst}` (§5.2 v3).
4. `restaurant.items_shape_bump` — `items: [{dish_id, qty, price}]` gains required `modifiers: []` array.
5. `hotel.gst_field` — `hotel.book` requires `gst_number` when `total > 7500` (§5.4 v3).

**Policy drifts (5):**

6. `airline.booking_window_shrink` — same-day bookings rejected after 14:00 IST (was: same-day always allowed).
7. `cab.school_hours_mini_reject` — `vehicle_class=mini` during 07:00–09:00 IST auto-rejects with `policy_error` (DESIGN.md §5.2 v2).
8. `restaurant.min_order_bump` — minimum order ₹199 → ₹299 (§5.3 v2).
9. `hotel.cancel_window_shrink` — free-cancellation window 24h → 6h (§5.4 v2).
10. `cab.vehicle_class_expand` — enum gains `suv`, `infant_seat_sedan`; old `sedan` may map to `suv` for some routes (§5.2 v2).

**T&C drifts (5):**

11. `airline.baggage_tnc_rewrite` — free-cabin allowance reduced 7kg → 5kg; announced via `side_channel_notice` on next response.
12. `cab.surge_policy_tnc` — surge may apply retroactively if ride extended; side-channel notice.
13. `restaurant.veg_filter_semantic` — `veg_only=True` now excludes egg dishes (was: included) (§5.3 v3). Notice appended.
14. `hotel.early_checkin_tnc` — early check-in before 12:00 now billed at 50% of nightly rate; side-channel.
15. `airline.reschedule_tnc` — reschedule fee previously waived; now 10% of fare + side-channel text.

**Pricing drifts (3):**

16. `airline.convenience_fee_append` — hidden ₹199 convenience fee added to booking confirmation; search estimate unchanged.
17. `cab.toll_unbundle` — tolls previously included; now a separate line item appearing only on booking, not estimate.
18. `hotel.resort_fee_append` — ₹500/night resort fee added at booking; not shown in `nightly_rate`.

**Auth drifts (2 — cross-cutting via payment):**

19. `payment.auth_scope_upgrade` — `token_v1` starts returning 401; requires `token_v2` with `scope=payments:write:v2` (DESIGN.md §5.5). Affects every domain's booking call.
20. `payment.mfa_required` — transactions > ₹5000 now require `mfa_code` in payload; auth_error with `mfa_required` code.

**Catalogue invariant:** `list_patterns()` returns exactly 20 `DriftPattern`s. If fewer load (file edited, YAML malformed), startup raises `DriftCatalogueError` and the env refuses to serve `reset()`.

**Coverage check (matches DESIGN.md §6.3 explicit enumeration):**
- Schema (5): airline.price_rename, airline.pax_required, cab.fare_breakdown, restaurant.items_shape_bump, hotel.gst_field.
- Policy (5): airline.booking_window_shrink, cab.school_hours_mini_reject, restaurant.min_order_bump, hotel.cancel_window_shrink, cab.vehicle_class_expand.
- T&C (5): airline.baggage_tnc_rewrite, cab.surge_policy_tnc, restaurant.veg_filter_semantic, hotel.early_checkin_tnc, airline.reschedule_tnc (two airline T&Cs, zero cab/hotel overlap allowed — the axis is 5 patterns, not one-per-domain).
- Pricing (3): airline.convenience_fee_append, cab.toll_unbundle, hotel.resort_fee_append (restaurant pricing collapses into `restaurant.min_order_bump` policy pattern per DESIGN.md §6.3).
- Auth (2, transversal via payment): payment.auth_scope_upgrade, payment.mfa_required — these affect booking calls in every primary domain.

Total: 5 + 5 + 5 + 3 + 2 = 20, byte-identical to DESIGN.md §6.3.

---

## 5. Error Modes

| Error | When raised | Handler |
|---|---|---|
| `ValueError("unknown stage")` | `build_schedule(stage)` with `stage ∉ {1, 2, 3}` | `env.reset()` catches and returns HTTP 400 via `app.py` |
| `UnknownDriftPatternError` | `apply_drift` called with `event.pattern_id` not in registry | Fatal — indicates schedule built against stale catalogue. `env.step()` aborts episode with R5=-1.0 (state corruption, per DESIGN.md §7.1 R5) |
| `DriftDomainMismatchError` | `event.domain` not a key in `state.vendor_states` | Fatal — same handling as above |
| `DriftReapplicationError` | `event` already in `state.drift_fired` | Fatal; signals schedule/firing bug |
| `DriftCatalogueError` | `list_patterns()` loads fewer than 20 patterns on startup | Env refuses to serve `reset()`. App returns 503 until the YAML is fixed |
| `DriftScheduleConflictError` | Stage 3 schedule generated duplicate `pattern_id` or overlapping turns < 2 apart | Fatal during `build_schedule`; retry with next RNG draw up to 5 times, then raise |

**No silent failures.** Every error path is caught at the `env.step()` / `env.reset()` boundary and converted to either a structured `ToolResult(status="schema_error"|"policy_error"|"auth_error", ...)` — when the error is the *intended* effect of the drift on a vendor call — or to an episode-terminating `R5` penalty when it's a real bug. The injector never returns a partially-mutated state.

### 5.1 Four Specific Error Scenarios the Spec Addresses

1. **Drift targets a tool the agent hasn't called yet.** Not an error — drift mutates `vendor_states` unconditionally. The mutation takes effect on the *next* time any tool for that domain is called. R2's 2-turn window starts at drift firing, regardless of whether the agent touches the domain that turn.
2. **Duplicate drift in stage 3.** `build_schedule` must reject any schedule where both events share a `pattern_id`. If RNG generates a duplicate, retry the second draw; after 5 failed retries, raise `DriftScheduleConflictError`. Guard with a property test (§7 edge case E4).
3. **Drift scheduled after `max_turns`.** `build_schedule` enforces `turn ≤ max_turns - 3`. If a caller somehow constructs a `DriftEvent` with `turn > max_turns`, the firing check in `step()` will never match — the drift never fires, which is wrong. The safeguard is that only `build_schedule` constructs events; tests assert `all(e.turn <= max_turns - 3 for e in schedule)`.
4. **Drift on a domain not in `goal`.** Stage 2 forbids this by construction. Stage 3's second drift may target `payment` even if goal domain is `airline` — this is the intended cross-domain cascade (edge case E5). All other cross-domain combinations are rejected at `build_schedule` time with `DriftScheduleConflictError`.

---

## 6. Dependencies

### 6.1 Consumes

- `DriftCallState` (from `models.py`, DESIGN.md §4.1) — read, returns new instance.
- `GoalSpec` (from `models.py`, DESIGN.md §4.1) — read-only; injector uses `goal.domain` for scheduling and `goal.constraints` only for edge-case checks (e.g., `hotel.gst_field` cares about total > 7500).
- `data/drift_patterns/drifts.yaml` — disk resource, loaded once, cached immutably.
- `random.Random(episode_seed)` — local RNG for determinism.

### 6.2 Produces

- New `DriftCallState` with updated `schema_versions`, mutated `vendor_states`, extended `drift_fired`.
- `tuple[DriftEvent, ...]` schedule for `reset()` to install on state.
- `tuple[DriftPattern, ...]` for introspection.

### 6.3 Consumed By

- **`env.step()`** — calls `apply_drift` at the drift-firing point in the step sequence (DESIGN.md §4.3 step 3). Before action dispatch. Always. This is the ONLY call site of `apply_drift` in the product code.
- **`env.reset()`** — calls `build_schedule` once per episode and installs the result on `state.drift_schedule`.
- **`rewards.r2_drift_detection(episode)`** — reads `episode.state.drift_fired` (ground truth) and `pattern.detection_hints` (from the registry keyed by `event.pattern_id`) to score the agent's adaptation. This is the tightest coupling in the system: **if `detection_hints` on a pattern are wrong, R2 is wrong.** Per DESIGN.md §6.3, hints are substring-matchable tokens. R2 treats a hint as matched via **case-insensitive substring match** applied against both (a) the agent's `SPEAK.message | CLARIFY.message | rationale` text within 2 turns of firing, AND (b) the JSON-serialized string of any subsequent `TOOL_CALL.tool_args` (so renamed-to fields, newly-required fields, and removed-field diagnostics trip the match structurally via the arg keys/values). A hit on either channel counts. No stemming, no edit distance, no fuzzy matching — exact case-insensitive substring. The injector guarantees that at least one detection path is mechanically available for every pattern — verified by the detection-smoke test in `docs/tests/drift_injector_tests.md`.
- **Vendors (`driftcall/vendors/*.py`)** — read `state.schema_versions[domain]` and `state.vendor_states[domain]` to shape their responses. They never write. They surface drift effects as `ToolResult.status` + `response` shape changes.
- **Tests (`tests/test_drift.py`)** — enumerate all 20 patterns via `list_patterns()`, fire each against a fresh state, assert the mutation took effect and that at least one detection hint trips against a canned correct-adaptation trajectory.

### 6.4 Does NOT Depend On

- The agent's actions (drift firing is independent of agent behavior, except for the turn clock which `env.step` owns).
- TTS/ASR (drift is a state mutation; audio is at the env boundary).
- The reward functions (injector produces ground truth; rewards consume it).

---

## 7. Edge Cases

**E1 — Drift on turn 0.** `build_schedule` never places a drift on turn 0 or turn 1. The minimum is turn 2, because the agent needs at least one successful pre-drift interaction to establish a "before" expectation that R2 can measure a delta against. Asserted in schedule validator; property-tested.

**E2 — Drift after episode terminates early (SUBMIT / ABORT before scheduled turn).** The schedule is static but the episode may end early. Drifts scheduled for turns after termination simply never fire. `drift_fired ⊆ drift_schedule` always. R2 scorer only considers fired drifts; unfired drifts are not held against the agent.

**E3 — Drift when the target tool is already failing for other reasons.** If a tool was returning `timeout` before the drift, applying the drift still mutates vendor state; once the tool recovers (or is retried), the new schema is in effect. The injector does not inspect `ToolResult` history.

**E4 — Duplicate `pattern_id` in a stage-3 schedule.** `build_schedule` must never produce this. Detection: hashset-dedup during construction; retry up to 5 times. If the pattern catalogue has fewer than 2 patterns for the target domain (edge case with payment-only second slot), the scheduler may reach for a cross-domain pattern. Property-tested with 10,000 seeds per stage.

**E5 — Cross-domain cascade: `payment.auth_scope_upgrade` fires when the agent is mid-booking on `airline`.** This is the *intended* hard case. The airline-booking call will fail at the payment step with `auth_error` even though airline's own schema is unchanged. R2 credits the agent if it either (a) mentions token/auth/scope keywords or (b) probes and retries with the new scope. `build_schedule` explicitly allows payment as a legitimate second-drift domain in stage 3 for any primary goal domain. Tests assert this scenario fires in ≥ 10% of stage-3 seeds (so the agent actually sees it during training).

**E6 — Drift whose target vendor is already "failing" this turn** (e.g., agent triggered a timeout and drift fires the same turn). The drift applies first (at the start of the turn), then the action is evaluated. If the action happens to be a tool call on the drifted domain, the response reflects the new schema AND whatever failure mode the action induces. Both effects compose; neither is suppressed.

**E7 — Two drifts on the same domain in stage 3** (e.g., `airline.price_rename` at turn 3 then `airline.pax_required` at turn 7). This is allowed — the airline schema walks v1 → v2 → v3 within one episode. The `from_version`/`to_version` on the second event must match the current `schema_versions['airline']` at firing time; `build_schedule` chains them. If the chain breaks (stale from_version), `apply_drift` still executes the mutation (mutation is schema-version-agnostic) but logs a warning; tests enforce the chain at build time.

**E8 — T&C drift arriving via side channel vs. tool response.** T&C drifts with operator `side_channel_notice_append` do *not* change tool response *shape* — they append a machine-readable notice string to `state.vendor_states[domain]['side_channel']`. The next tool result for any tool in that domain carries the notice in its `response["_notice"]` field. R2 credits detection if the agent quotes or references the notice. Distinguished from schema drift at the mutation-operator level (§3.4). Vendors check `state.vendor_states[domain]['side_channel']` and attach the notice once per episode per notice.

**E9 — Drift schedule length mismatch with curriculum stage.** Invariant: `len(schedule) == {1: 0, 2: 1, 3: 2}[stage]`. If an external caller builds a state with a schedule length inconsistent with stage, `env.step` does not re-validate — the invariant is enforced only at `build_schedule`. This is documented so test fixtures know not to hand-craft mismatched schedules.

**E10 — Stage-3 drift placement with small `max_turns`.** If `max_turns == 16` (DESIGN.md §4.5 stage 3), valid placement is roomy. But if a caller passes a smaller `max_turns` via config override, the schedule may fail the distance-≥2 constraint. `build_schedule` raises `DriftScheduleConflictError` if `max_turns < 8` for stage 3.

---

## 8. Examples

### 8.1 Stage-2 schedule: `airline.price_rename` at turn 4

**Input:**
```python
stage = 2
episode_seed = 1234
goal = GoalSpec(
    domain="airline",
    intent="book_flight",
    slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
    constraints={"budget_inr": 8000, "time_window": "evening"},
    language="hinglish",
    seed_utterance="Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad",
)
```

**Output (`build_schedule`):**
```python
(
    DriftEvent(
        turn=4,
        drift_type="schema",
        domain="airline",
        description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.price_rename",
    ),
)
```

**Lifecycle at turn 4:**
1. `env.step` increments `state.turn` from 3 → 4.
2. Scheduler check finds `drift_schedule[0].turn == 4`, not yet in `drift_fired`.
3. `apply_drift(state, event)` is invoked:
   - Looks up pattern `airline.price_rename` in registry.
   - Deep-copies `state.vendor_states["airline"]`.
   - Applies `rename {price → total_fare_inr}` and `remove [currency]` to the schema dict.
   - Sets `schema_versions["airline"] = "v2"`.
   - Appends the event to `drift_fired`.
   - Returns new frozen state.
4. Agent's turn-4 action (say `airline.search`) is now evaluated against v2 — response contains `total_fare_inr`, no `price`, no `currency`.
5. R2 scorer on episode-end inspects `drift_fired` and matches detection hints `["total_fare_inr", "price", "rename", "field"]` against agent's turn-4, turn-5 `SPEAK`/`CLARIFY` text AND against any turn-5 `TOOL_CALL.tool_args` containing key `total_fare_inr`.

### 8.2 Stage-3 compound schedule: airline policy drift at turn 3 + payment auth drift at turn 7

**Input:**
```python
stage = 3
episode_seed = 9001
goal = GoalSpec(
    domain="airline",
    intent="book_flight",
    slots={"from": "BOM", "to": "DEL", "when": "2026-05-02"},
    constraints={"budget_inr": 12000},
    language="hi",
    seed_utterance="मुझे शनिवार को दिल्ली जाना है, 12000 रुपये से कम में",
)
```

**Output (`build_schedule`):**
```python
(
    DriftEvent(
        turn=3,
        drift_type="policy",
        domain="airline",
        description="same-day booking window shrunk from 24h to 6h before departure",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.booking_window_shrink",
    ),
    DriftEvent(
        turn=7,
        drift_type="auth",
        domain="payment",
        description="token_v1 now 401s; requires token_v2 with scope=payments:write:v2",
        from_version="v1",
        to_version="v2",
        pattern_id="payment.auth_scope_upgrade",
    ),
)
```

**Cascade behavior:** Agent successfully adapts to the booking-window policy at turn 3–5. At turn 7, it attempts `airline.book` → internally calls `payment.charge` → payment gateway returns `auth_error` with `{required_scope: "payments:write:v2"}`. This is the cross-domain cascade (E5). R2 credits the agent if it subsequently re-issues payment with updated scope or even references "auth"/"scope"/"token" keywords in its SPEAK message before retry.

### 8.3 Stage-1 empty schedule

**Input:**
```python
stage = 1
episode_seed = 42
goal = GoalSpec(
    domain="restaurant",
    intent="order_food",
    slots={"city": "Bengaluru", "cuisine": "biryani"},
    constraints={"budget_inr": 300, "dietary": "veg"},
    language="hinglish",
    seed_utterance="Tomorrow dinner ke liye Biryani order karna hai, 300 rupees se kam, veg option chahiye",
)
```

**Output (`build_schedule`):**
```python
()   # empty tuple
```

**Behavior:** No drifts ever fire. `apply_drift` is never called. `state.drift_fired` remains `()` for the whole episode. R2 is neutral-scored at 0.5 per DESIGN.md §7.1 R2. The agent's job is pure tool use + format + constraint adherence.

---

## 9. Open Questions

**Resolved (recorded here for audit; no action required):**

- ~~Q1 — Exact 5×4 grid vs 20-pattern catalogue shape.~~ **RESOLVED** by DESIGN.md §6.3 update (2026-04-24 changelog): the catalogue is the explicit enumeration `5 schema + 5 policy + 5 T&C + 3 pricing + 2 transversal payment-auth = 20`, not a strict Cartesian product. §4.4 of this doc now mirrors that enumeration verbatim.
- ~~Q2 — Detection hint matching: exact vs fuzzy.~~ **RESOLVED** by DESIGN.md §6.3: `detection_hints` are substring-matchable tokens; R2 uses **case-insensitive substring match** against agent `SPEAK`/`CLARIFY` text AND against `TOOL_CALL.tool_args` JSON-serialized strings. No stemming, no edit distance. Documented in §6.3 of this doc.
- ~~Q3 — Cross-domain cascade in stage-3 compound drift.~~ **RESOLVED** by DESIGN.md §6: any two patterns from the 20-catalogue may co-occur in a stage-3 schedule, subject to the non-overlapping-turns constraint (distance ≥ 2, both turns within `[2, max_turns - 3]`) and the different-`pattern_id` constraint. Explicitly, a `payment.*` auth drift MAY co-occur with an `airline.*` schema drift — this is the intended cross-domain cascade (E5). `build_schedule` biases the second-drift domain 80% same-domain-as-`goal.domain` / 20% cross-domain (= `payment`), which satisfies E5's "≥ 10% cross-domain in stage-3 seeds" property test.
- ~~Q4 — Side-channel T&C notice lifecycle.~~ **RESOLVED**: a T&C drift with operator `side_channel_notice_append` returns a `side_channel_notice` field on the **next** tool call's `ToolResult.response` (single-turn visibility) AND fires a `DriftEvent` recorded in `state.drift_log` / `drift_fired`. The notice is not persistent across subsequent tool calls in the domain; R2 credits detection if the agent references the notice within the 2-turn window from firing. Vendors (`vendors.md`) wire the read side as a one-shot attachment.

**Remaining:**

**Q1 (deferred post-hackathon) — Catalogue-hash in seed.**
Should `drift_schedule` seeding incorporate a hash of the drift catalogue so that changing the catalogue (adding/removing/editing a pattern) changes all episode seeds and invalidates old held-out eval replays? A single `random.Random(episode_seed)` is drawn for turn placement AND pattern selection today; if the catalogue changes, previously-seeded episodes silently produce different schedules. Recommendation for production: include `catalogue_hash` in `DriftCallState.metadata` and mix it into the seed. **Decision deferred post-hackathon** — for the event we pin the catalogue at v1 in `data/drift_patterns/drifts.yaml` and ship held-out eval episodes computed against that exact file; any future catalogue edit will rev the dataset version rather than silently rerolling seeds. Flagged for `models.md` author as a metadata addition in a subsequent milestone.

---

*End of drift_injector.md.*
