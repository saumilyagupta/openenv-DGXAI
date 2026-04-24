# drift_injector_tests.md — Test Plan for `driftcall/drift_injector.py`

**Module under test:** `driftcall/drift_injector.py`
**Design doc:** `DRIFTCALL/docs/modules/drift_injector.md` (final sealed)
**Owner:** Person B (Rewards & Tests); reviewed by Person A (Environment)
**Implements test coverage for:** DESIGN.md §4.3 (step semantics — drift trigger point), §6 (full module), §7.1 (R2 detection signal)
**Status:** Test plan — pre-critic-gate
**Last updated:** 2026-04-24

**Framework:** `pytest` + `pytest-cov` + `hypothesis` (+ `pyyaml` for fixture loading)
**Coverage tool:** `pytest --cov=driftcall.drift_injector --cov-branch --cov-report=term-missing`
**File under test (100% line, 95% branch required):** `driftcall/drift_injector.py`

---

## 0. Scope & Contract

This plan covers the **entire public surface** of `driftcall/drift_injector.py`:

- `build_schedule(stage, episode_seed, goal) -> tuple[DriftEvent, ...]`
- `apply_drift(state, event) -> DriftCallState`
- `list_patterns() -> tuple[DriftPattern, ...]`
- The module-private registry loader and operator dispatch table (covered via the public surface — no direct tests against private functions; 100% coverage is achieved via the public API).
- All 6 error types declared in drift_injector.md §5: `ValueError`, `UnknownDriftPatternError`, `DriftDomainMismatchError`, `DriftReapplicationError`, `DriftCatalogueError`, `DriftScheduleConflictError`.

**Non-goals (covered elsewhere):**
- Full `env.step()` orchestration — lives in `docs/tests/env_tests.md`. This plan fires events through a **mock env.step loop** (§3) to confirm wiring but does not re-test env-level semantics.
- Vendor response shaping after drift — lives in `docs/tests/vendors_tests.md`.
- R2 reward scoring — lives in `docs/tests/rewards_tests.md`. This plan only asserts that `detection_hints` are non-empty short substring-matchable tokens (the structural precondition R2 depends on).

**Test count target:** ≥ 53 test cases total (20 unit + 5 property + 12 integration + 16 parametrized pattern-smoke = 53). Actual inventory below sums to **55**.

---

## 1. Unit Tests (≥ 20 cases)

All unit tests live in `tests/test_drift_injector.py`. Each test is hermetic — no network, no disk writes, loads the authoritative `drift_patterns_fixture.yaml` from §5 once per session via a `pytest.fixture(scope="session")`.

### 1.1 `build_schedule` — stage-count invariants

| # | Test | Setup | Assertion |
|---|---|---|---|
| U1 | `test_build_schedule_stage1_returns_empty_tuple` | `stage=1, seed=42, goal=goal_airline` | `build_schedule(...) == ()` exactly. `len() == 0`. Type is `tuple`. |
| U2 | `test_build_schedule_stage2_returns_exactly_one_event` | `stage=2, seed=1234, goal=goal_airline` | `len(schedule) == 1`. `schedule[0]` is a `DriftEvent` instance. |
| U3 | `test_build_schedule_stage3_returns_exactly_two_events` | `stage=3, seed=9001, goal=goal_airline` | `len(schedule) == 2`. Both are `DriftEvent` instances. |
| U4 | `test_build_schedule_stage3_events_are_turn_ascending` | Same as U3 | `schedule[0].turn < schedule[1].turn`. |
| U5 | `test_build_schedule_stage3_distance_ge_2_turns` | Parametrized over 50 seeds | `schedule[1].turn - schedule[0].turn >= 2` for every seed. |
| U6 | `test_build_schedule_invalid_stage_raises_value_error` | `stage=0, 4, -1, 99` (parametrized) | `pytest.raises(ValueError, match="stage")`. |

### 1.2 `build_schedule` — determinism

| # | Test | Setup | Assertion |
|---|---|---|---|
| U7 | `test_build_schedule_is_deterministic_same_inputs` | Two calls with `(stage=2, seed=1234, goal_airline)` | Call A `==` Call B element-wise (all `DriftEvent` fields). |
| U8 | `test_build_schedule_different_seeds_produce_different_schedules` | `(stage=2, seed=1234, goal_airline)` vs `(stage=2, seed=1235, goal_airline)` | Schedules differ in at least one of `{turn, pattern_id}`. Sampled over 100 seed pairs; require ≥ 95% divergence (deterministic hash collisions tolerated). |
| U9 | `test_build_schedule_does_not_use_global_rng` | Seed `random.seed(0)`, call `build_schedule(stage=2, seed=1234, goal_airline)`, then `random.random()` | The subsequent `random.random()` call returns the same value as if `build_schedule` were never called. Proves the scheduler owns a local `random.Random(episode_seed)`. |

### 1.3 `build_schedule` — placement windows

| # | Test | Setup | Assertion |
|---|---|---|---|
| U10 | `test_build_schedule_stage2_turn_within_window` | Parametrized over 100 seeds, `max_turns=12` | For every seed, `2 <= schedule[0].turn <= max_turns - 3` (i.e., `[2, 9]`). |
| U11 | `test_build_schedule_stage3_first_turn_first_half` | Parametrized over 100 seeds, `max_turns=16` | `2 <= schedule[0].turn <= max_turns // 2` (i.e., `[2, 8]`). |
| U12 | `test_build_schedule_stage3_second_turn_after_first_plus_2` | Parametrized over 100 seeds, `max_turns=16` | `schedule[0].turn + 2 <= schedule[1].turn <= max_turns - 3`. |
| U13 | `test_build_schedule_stage3_raises_conflict_when_max_turns_too_small` | `max_turns=7, stage=3` | `pytest.raises(DriftScheduleConflictError, match="max_turns")`. |

### 1.4 `build_schedule` — domain & pattern selection

| # | Test | Setup | Assertion |
|---|---|---|---|
| U14 | `test_build_schedule_stage2_targets_goal_domain` | Parametrized over all 4 `goal_*` fixtures × 20 seeds each | `schedule[0].domain == goal.domain` for all 80 cases. |
| U15 | `test_build_schedule_stage3_first_drift_targets_goal_domain` | Parametrized over all 4 goals × 20 seeds | `schedule[0].domain == goal.domain`. |
| U16 | `test_build_schedule_stage3_no_pattern_id_collision` | Parametrized over 200 seeds, `goal=goal_airline` | `schedule[0].pattern_id != schedule[1].pattern_id` for every seed. |

### 1.5 `apply_drift` — mutation semantics & purity

| # | Test | Setup | Assertion |
|---|---|---|---|
| U17 | `test_apply_drift_returns_new_object_not_mutated` | `state0 = fresh_state(domain="airline")`, event = `airline.price_rename` | `new_state = apply_drift(state0, event)`. `new_state is not state0`. `state0.schema_versions["airline"] == "v1"` (unchanged). `state0.drift_fired == ()` (unchanged). `state0.vendor_states["airline"]` unchanged (deep-equal to pre-call snapshot). |
| U18 | `test_apply_drift_updates_schema_version` | Same | `new_state.schema_versions["airline"] == event.to_version`. |
| U19 | `test_apply_drift_appends_event_to_drift_fired` | Same | `new_state.drift_fired == state0.drift_fired + (event,)`. `isinstance(new_state.drift_fired, tuple)`. |
| U20 | `test_apply_drift_does_not_change_turn` | Same | `new_state.turn == state0.turn`. |
| U21 | `test_apply_drift_vendor_states_length_preserved` | Same | `len(new_state.vendor_states) == len(state0.vendor_states)`. Key set identical. |
| U22 | `test_apply_drift_unknown_pattern_raises` | Craft a `DriftEvent(pattern_id="bogus.nonsense", ...)` | `pytest.raises(UnknownDriftPatternError)`. |
| U23 | `test_apply_drift_unknown_domain_raises` | Event with `domain="martian_colony"` | `pytest.raises(DriftDomainMismatchError)`. |
| U24 | `test_apply_drift_reapplication_raises` | Apply once → state1; apply same event to state1 | `pytest.raises(DriftReapplicationError)`. |

### 1.6 `list_patterns` — catalogue invariants

| # | Test | Setup | Assertion |
|---|---|---|---|
| U25 | `test_list_patterns_returns_exactly_20` | Direct call | `len(list_patterns()) == 20`. Result is a `tuple`. |
| U26 | `test_list_patterns_sorted_by_id` | Direct call | `[p.id for p in list_patterns()] == sorted(p.id for p in list_patterns())`. |
| U27 | `test_list_patterns_cached_on_second_call` | Two calls | `list_patterns() is list_patterns()` (identity — module-level cached tuple). |
| U28 | `test_list_patterns_all_ids_unique` | Direct call | `len({p.id for p in list_patterns()}) == 20`. |
| U29 | `test_list_patterns_axis_counts_match_design` | Direct call | `sum(p.drift_type == "schema")==5`, `policy==5`, `tnc==5`, `pricing==3`, `auth==2`. |
| U30 | `test_list_patterns_detection_hints_are_substring_tokens` | For every pattern | `all(isinstance(h, str) and 1 <= len(h) <= 40 and " " not in h.strip() or h.count(" ") <= 2 for h in p.detection_hints)`. Each hint is a short token (≤ 40 chars, ≤ 2 spaces — explicitly NOT sentences). Each `detection_hints` tuple has length ≥ 2. |
| U31 | `test_list_patterns_empty_catalogue_raises` | Monkeypatch loader to return `[]` and clear cache | `pytest.raises(DriftCatalogueError)`. |

### 1.7 All-pattern applicability smoke (parametrized, counts as 20 cases)

| # | Test | Setup | Assertion |
|---|---|---|---|
| U32 | `test_every_pattern_is_applyable[<pattern_id>]` | `@pytest.mark.parametrize("pattern", list_patterns())` (20 rows) | For each of the 20 patterns: build a fresh `state0` whose `schema_versions[pattern.domain] == pattern.from_version`, construct `DriftEvent(turn=2, drift_type=pattern.drift_type, domain=pattern.domain, description=pattern.description, from_version=pattern.from_version, to_version=pattern.to_version, pattern_id=pattern.id)`, call `apply_drift(state0, event)` — no exception, `new_state.schema_versions[pattern.domain] == pattern.to_version`, `event in new_state.drift_fired`, **and** for at least one detection_hint in the pattern, substring-matching the hint against a canonical "post-drift vendor response" payload (loaded from the fixture) returns `True` case-insensitively. Confirms detection-path mechanical availability (drift_injector.md §6.3). |

This single parametrized test expands to **20 concrete test cases at collection time**, satisfying the "all 20 drift patterns + 2 transversal payment-auth patterns applyable" clause (the 2 payment-auth patterns are included in the 20 — §4.4 of the design doc). It also exercises the full `mutation` operator dispatch (rename / remove / require_new_field / enum_expand / numeric_bump / policy_flag_flip / tnc_text_swap / pricing_restructure / auth_scope_bump / time_window_shrink / side_channel_notice_append / fee_append / token_version_bump / change_type), which is how we hit **100% line coverage** in one stroke.

### 1.8 Unit test inventory summary

- Stage count: U1–U6 (6)
- Determinism: U7–U9 (3)
- Placement: U10–U13 (4)
- Domain / pattern selection: U14–U16 (3)
- `apply_drift` semantics & errors: U17–U24 (8)
- `list_patterns` catalogue: U25–U31 (7)
- Every-pattern applicability (parametrized × 20): U32 (20)

**Total unit test cases: 6 + 3 + 4 + 3 + 8 + 7 + 20 = 51, well above the ≥ 20 bar.**

---

## 2. Property Tests (≥ 5 properties — `hypothesis`-based)

All property tests live in `tests/test_drift_injector_properties.py`. Strategies are composed from a shared `hypothesis_strategies.py` so `env_tests.md` can reuse them.

### 2.1 Strategies

```python
# tests/hypothesis_strategies.py (shared)
import hypothesis.strategies as st

stages    = st.sampled_from([1, 2, 3])
seeds     = st.integers(min_value=0, max_value=2**31 - 1)
goals     = st.sampled_from([goal_airline, goal_cab, goal_restaurant, goal_hotel])  # §5 fixtures
max_turns = st.integers(min_value=8, max_value=20)
```

### 2.2 Properties

| # | Property | Strategy | Assertion |
|---|---|---|---|
| P1 | **build_schedule is deterministic/pure** — `build_schedule(s, k, g) == build_schedule(s, k, g)` for every `(s, k, g)` | `stages × seeds × goals`; `@settings(max_examples=500)` | Equality of returned tuples, field-by-field. Two calls in sequence — no input mutation (goal dataclass hash unchanged pre/post). |
| P2 | **apply_drift never returns the input identity** — for every valid `(state, event)` where event targets an in-scope domain, `apply_drift(state, event) is not state`, and the input `state` is structurally unchanged after the call | Synthesized states × randomly-drawn patterns from `list_patterns()`; `max_examples=500` | `new_state is not state`. `state.schema_versions`, `state.drift_fired`, `state.vendor_states` all structurally equal to snapshot taken before the call (deep-equal via `copy.deepcopy` compare). |
| P3 | **Stage-3 schedule never has colliding pattern_ids** — for every stage-3 schedule, `schedule[0].pattern_id != schedule[1].pattern_id` | `stages=st.just(3) × seeds × goals`; `max_examples=2000` | Uniqueness; also asserts distance-≥-2 on `turn` as a secondary property on the same generated data. |
| P4 | **All detection_hints are non-empty short tokens** — for every pattern in `list_patterns()`, every hint is `str`, `1 <= len(hint) <= 40`, `hint.strip() == hint`, and no hint contains a newline or tab. Each pattern has at least 2 hints. | Enumerates all 20 patterns (not randomized — this is a Hypothesis-driven invariant over the catalogue); also `@given(st.sampled_from(list_patterns()))` sweep with `max_examples=200` | All assertions hold for all hints of all patterns. |
| P5 | **Stage-3 schedule turns always within window** — for every `(seed, goal, max_turns)`, `all(2 <= e.turn <= max_turns - 3 for e in schedule)` and `schedule[1].turn - schedule[0].turn >= 2`. | `st.just(3) × seeds × goals × max_turns` with `max_turns >= 8`; `max_examples=2000` | Window + distance invariants; shrinks to the smallest failing `(seed, max_turns)` pair if broken. |

### 2.3 Optional sixth property (safety net for payment cascade bias)

| # | Property | Strategy | Assertion |
|---|---|---|---|
| P6 | **Cross-domain cascade fires in ≥ 10% of stage-3 schedules** — sample 1,000 `(seed, goal=goal_airline)` stage-3 schedules; count schedules where `schedule[1].domain == "payment"`. | Fixed batch of 1,000 seeds (deterministic hypothesis seed) | `count >= 100` (≥ 10% per drift_injector.md §7 E5). This is a statistical check but with fixed seeds it is deterministic — guarantees CI reproducibility. |

**Total properties: 6.** Well above the ≥ 5 bar.

---

## 3. Integration Tests

All integration tests live in `tests/test_drift_injector_integration.py`. They use the authoritative fixtures from §5 and a lightweight `MockEnv` (defined in `tests/mock_env.py`) that implements the drift-firing point from DESIGN.md §4.3 step 3, but stubs everything else (no vendors, no reward, no audio). Only `build_schedule` + `apply_drift` are exercised against real code — the rest is scaffolding.

### 3.1 Matrix: 3 stages × 4 goal domains (I1–I12)

| # | Test | Setup | Assertion |
|---|---|---|---|
| I1 | `test_integration_stage1_airline` | `stage=1, seed=100, goal=goal_airline` | Generate schedule, run 16-turn mock loop, no drifts ever fire. `fired_events == []`. `final_state.drift_fired == ()`. |
| I2 | `test_integration_stage1_cab` | `stage=1, seed=101, goal=goal_cab` | Same pattern for cab. |
| I3 | `test_integration_stage1_restaurant` | `stage=1, seed=102, goal=goal_restaurant` | Same pattern for restaurant. |
| I4 | `test_integration_stage1_hotel` | `stage=1, seed=103, goal=goal_hotel` | Same pattern for hotel. |
| I5 | `test_integration_stage2_airline` | `stage=2, seed=200, goal=goal_airline` | Exactly 1 drift fires. The fired `DriftEvent` is `schedule[0]`. `final_state.schema_versions["airline"]` advanced. `fired_events == list(schedule)`. |
| I6 | `test_integration_stage2_cab` | `stage=2, seed=201, goal=goal_cab` | Same. |
| I7 | `test_integration_stage2_restaurant` | `stage=2, seed=202, goal=goal_restaurant` | Same. |
| I8 | `test_integration_stage2_hotel` | `stage=2, seed=203, goal=goal_hotel` | Same. |
| I9 | `test_integration_stage3_airline` | `stage=3, seed=300, goal=goal_airline` | Exactly 2 drifts fire at their scheduled turns. `fired_events == list(schedule)`. Schema versions advanced per-domain appropriately. |
| I10 | `test_integration_stage3_cab` | `stage=3, seed=301, goal=goal_cab` | Same. |
| I11 | `test_integration_stage3_restaurant` | `stage=3, seed=302, goal=goal_restaurant` | Same. |
| I12 | `test_integration_stage3_hotel` | `stage=3, seed=303, goal=goal_hotel` | Same. |

**Core invariant asserted across I1–I12:** `every DriftEvent that was fired by the mock env.step loop is an element of the schedule produced by build_schedule` (checked via set membership on `pattern_id` + `turn`). Equivalent to: `set(fired_events) ⊆ set(schedule)` and, for completed episodes (no early SUBMIT/ABORT), `set(fired_events) == set(schedule)`.

### 3.2 Cross-domain cascade (I13)

| # | Test | Setup | Assertion |
|---|---|---|---|
| I13 | `test_integration_stage3_airline_with_payment_auth_cascade` | Seed-hunt: iterate seeds `[0..5000]` with `stage=3, goal=goal_airline`, pick the first seed whose schedule has `schedule[1].pattern_id == "payment.auth_scope_upgrade"` (guaranteed to exist given P6's ≥ 10% cross-domain rate). Set up mock env with airline + payment vendors wired. Agent trajectory: book airline at turn ≥ `schedule[1].turn`; internal `payment.charge` call is made. | After drift at `schedule[1].turn` fires, an `airline.book` tool call whose handler internally calls `payment.charge` returns `ToolResult(status="auth_error", response={"required_scope": "payments:write:v2", ...})`. Assertions: (a) `final_state.schema_versions["payment"] == "v2"`, (b) the `airline.book` call's `ToolResult.status == "auth_error"` (NOT `"ok"`, NOT `"schema_error"`), (c) response JSON contains substring `"scope"` and `"payments:write:v2"` so R2 detection hints `["scope", "token_v2", "auth", "401"]` can trip in downstream scoring. |

This is the DESIGN.md §6 / drift_injector.md §7 E5 "intended hard case" — cross-domain cascade. It is the single most important integration test in this plan because it is the drift pattern most likely to be silently broken by a future refactor.

### 3.3 Early-termination determinism (I14)

| # | Test | Setup | Assertion |
|---|---|---|---|
| I14 | `test_integration_early_submit_before_scheduled_drift` | `stage=3, seed=300, goal=goal_airline`, mock loop runs only up to `schedule[0].turn - 1` (agent SUBMITs early) | Exactly 0 drifts fire. `final_state.drift_fired == ()`. Schedule remains `len == 2` on `final_state.drift_schedule`. Confirms E2 from the design doc: unfired drifts don't retroactively apply. |

### 3.4 Two-drift same-domain chain (I15)

| # | Test | Setup | Assertion |
|---|---|---|---|
| I15 | `test_integration_stage3_two_drifts_same_domain_chain` | Seed-hunt for a stage-3 airline goal whose schedule is `[airline.price_rename @ turn t1, airline.pax_required @ turn t2]` (both airline, chained v1→v2→v3). | After both fire, `final_state.schema_versions["airline"] == "v3"`. `final_state.vendor_states["airline"]["schema"]` reflects both mutations composed (no `price`, no `currency`, has `total_fare_inr`, has `passenger_count`). E7 from the design doc. |

### 3.5 Fixture reuse with `env_tests.md` (I16)

| # | Test | Setup | Assertion |
|---|---|---|---|
| I16 | `test_integration_fixture_shared_with_env_tests` | Import `drift_patterns_fixture` and `goal_airline/cab/restaurant/hotel` from the shared `tests/conftest.py` | Assert the fixture object identities match those imported by `test_env.py` (via `sys.modules`-level identity). Sentinel to catch fixture drift between the two test plans. |

**Integration inventory: 16 tests (I1–I16).** Exceeds the §3 requirement (the design spec asks for "each of 3 stages × 4 goal domains" which is I1–I12, plus the cascade, plus the supporting 4 — total 16).

---

## 4. Coverage Target

### 4.1 Goals

| Metric | Target | Enforced via |
|---|---|---|
| **Line coverage** on `driftcall/drift_injector.py` | **100%** | `pytest --cov=driftcall.drift_injector --cov-fail-under=100` in CI |
| **Branch coverage** on same file | **≥ 95%** | `pytest --cov-branch --cov-fail-under=95` (separate invocation, branch is tracked separately) |
| **All 20 drift patterns exercised** | **20/20** | Parametrized test U32 (collection count exactly 20) — enforced by an explicit assertion at session end: `assert PATTERNS_EXERCISED == set(p.id for p in list_patterns())` registered via a `pytest_sessionfinish` hook in `tests/conftest.py` |

### 4.2 How each line is reached

| Source-file region | Test(s) that cover it |
|---|---|
| `build_schedule` stage=1 branch | U1, I1–I4, P1 |
| `build_schedule` stage=2 branch | U2, U10, U14, I5–I8, P1, P2 |
| `build_schedule` stage=3 branch | U3, U4, U5, U11, U12, U15, U16, U13, I9–I12, I13, I15, P3, P5, P6 |
| `build_schedule` ValueError branch | U6 |
| `build_schedule` DriftScheduleConflictError (max_turns) | U13 |
| `build_schedule` RNG-retry-on-duplicate branch | Covered indirectly via P3 at `max_examples=2000` — if the retry never fires on 2,000 seeds we add a targeted unit test U16b with a monkeypatched RNG that forces 4 consecutive duplicates, proving the retry loop executes and caps at 5 |
| `apply_drift` happy path | U17–U21, U32 (×20), I5–I12, I13, I15 |
| `apply_drift` UnknownDriftPatternError | U22 |
| `apply_drift` DriftDomainMismatchError | U23 |
| `apply_drift` DriftReapplicationError | U24 |
| Every mutation operator dispatch (`rename`, `remove`, `require_new_field`, `change_type`, `enum_expand`, `numeric_bump`, `policy_flag_flip`, `time_window_shrink`, `tnc_text_swap`, `side_channel_notice_append`, `pricing_restructure`, `fee_append`, `auth_scope_bump`, `token_version_bump`) | U32 parametrization (each pattern hits at least one operator; across all 20 patterns every operator in the dispatch table is exercised at least once — a conftest-level assertion verifies this by tracking which operator keys are dispatched) |
| `list_patterns` happy path | U25–U30, U32 |
| `list_patterns` cached-second-call branch | U27 |
| `list_patterns` DriftCatalogueError (malformed / empty YAML) | U31 |

### 4.3 Branch-coverage note

The 5% unreachable-branch budget is reserved for: (a) defensive `assert` statements never expected to fail in production (e.g., `assert isinstance(schedule, tuple)` internal to `build_schedule`), and (b) the `else` branches of exhaustive enum-dispatch `if/elif` chains that are fully covered on the positive side. These are acceptable un-hit branches per `pytest-cov` conventions.

### 4.4 Mandatory CI gate

```
pytest tests/ --cov=driftcall.drift_injector --cov-branch \
  --cov-fail-under=100 --cov-report=term-missing \
  -q
```

Fails the build if line coverage drops below 100% OR if any of the 20 pattern IDs was not exercised (session-finish hook assertion).

---

## 5. Fixtures

All fixtures live in `tests/conftest.py` (session-scoped) and are **shared with `docs/tests/env_tests.md`** — fixture identity is asserted by I16.

### 5.1 `drift_patterns_fixture.yaml` — authoritative 20-pattern catalogue

**Location:** `tests/fixtures/drift_patterns_fixture.yaml`
**Authoritative source:** this fixture file is a byte-identical copy of the production `data/drift_patterns/drifts.yaml`, committed to the test tree so the tests don't accidentally depend on a stale or mutated production file. A test (`U33` — an additional check outside the 51 above) asserts the two files are byte-identical via `hashlib.sha256`. If the production file changes, the fixture must be re-synced in the same commit.

**Shape:** list of 20 `DriftPattern` YAML documents with fields `{id, drift_type, domain, from_version, to_version, description, mutation, detection_hints}` per drift_injector.md §4.3.

**Patterns (byte-identical to drift_injector.md §4.4):**

Schema (5): `airline.price_rename`, `airline.pax_required`, `cab.fare_breakdown`, `restaurant.items_shape_bump`, `hotel.gst_field`.
Policy (5): `airline.booking_window_shrink`, `cab.school_hours_mini_reject`, `restaurant.min_order_bump`, `hotel.cancel_window_shrink`, `cab.vehicle_class_expand`.
T&C (5): `airline.baggage_tnc_rewrite`, `cab.surge_policy_tnc`, `restaurant.veg_filter_semantic`, `hotel.early_checkin_tnc`, `airline.reschedule_tnc`.
Pricing (3): `airline.convenience_fee_append`, `cab.toll_unbundle`, `hotel.resort_fee_append`.
Auth (2, transversal via payment): `payment.auth_scope_upgrade`, `payment.mfa_required`.

**Loader:**

```python
@pytest.fixture(scope="session")
def drift_patterns_fixture() -> tuple[DriftPattern, ...]:
    import yaml, pathlib
    raw = yaml.safe_load(pathlib.Path("tests/fixtures/drift_patterns_fixture.yaml").read_text())
    patterns = tuple(DriftPattern(**entry) for entry in raw)
    assert len(patterns) == 20
    return patterns
```

### 5.2 `goal_airline`, `goal_cab`, `goal_restaurant`, `goal_hotel`

Four frozen `GoalSpec` instances, session-scoped, covering all four primary domains. Reused verbatim in `env_tests.md` via `from tests.conftest import goal_airline, goal_cab, goal_restaurant, goal_hotel`.

```python
@pytest.fixture(scope="session")
def goal_airline() -> GoalSpec:
    return GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
        language="hinglish",
        seed_utterance="Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad",
    )

@pytest.fixture(scope="session")
def goal_cab() -> GoalSpec:
    return GoalSpec(
        domain="cab",
        intent="book_cab",
        slots={"pickup": "Koramangala", "drop": "Kempegowda Airport", "when": "2026-04-26 06:30"},
        constraints={"budget_inr": 900, "vehicle_class": "sedan"},
        language="en",
        seed_utterance="Need a sedan from Koramangala to the airport at 6:30 AM, under 900 rupees.",
    )

@pytest.fixture(scope="session")
def goal_restaurant() -> GoalSpec:
    return GoalSpec(
        domain="restaurant",
        intent="order_food",
        slots={"city": "Bengaluru", "cuisine": "biryani"},
        constraints={"budget_inr": 300, "dietary": "veg"},
        language="hinglish",
        seed_utterance="Tomorrow dinner ke liye Biryani order karna hai, 300 rupees se kam, veg option chahiye",
    )

@pytest.fixture(scope="session")
def goal_hotel() -> GoalSpec:
    return GoalSpec(
        domain="hotel",
        intent="book_room",
        slots={"city": "Goa", "check_in": "2026-05-10", "nights": 2},
        constraints={"budget_inr": 7000, "room_type": "deluxe"},
        language="hi",
        seed_utterance="दस मई को गोवा में दो रात, deluxe room, 7000 रुपये से कम में",
    )
```

### 5.3 Helper fixtures (module-scoped, also shared)

| Fixture | Purpose |
|---|---|
| `fresh_state(domain)` | Factory returning a pristine `DriftCallState` at turn 0 with `schema_versions = {domain: "v1", "payment": "v1"}` and default vendor states loaded from `tests/fixtures/vendor_initial_states.yaml`. Used by U17–U24, U32, and every integration test. |
| `mock_env` | Lightweight env harness implementing DESIGN.md §4.3 drift-firing point only; no vendors/rewards/audio. Used by I1–I15. |
| `canonical_post_drift_responses` | Dict keyed by `pattern_id` → example `ToolResult.response` payload that a correct vendor would return after that drift fires. Used by U32 to confirm detection-hint substring-matchability. |

### 5.4 Fixture sharing contract with `env_tests.md`

The 6 fixtures (`drift_patterns_fixture`, `goal_airline`, `goal_cab`, `goal_restaurant`, `goal_hotel`, `fresh_state`) are defined once in `tests/conftest.py`. Both this test plan and `docs/tests/env_tests.md` import them by name. Test I16 asserts fixture-object identity across the two test modules — if either file copies-rather-than-imports a fixture, I16 fails loud.

---

## 6. Test Execution

### 6.1 Commands

```bash
# Full suite (unit + property + integration) with coverage gate
python3 -m pytest DRIFTCALL/tests/ \
  --cov=driftcall.drift_injector \
  --cov-branch \
  --cov-fail-under=100 \
  --cov-report=term-missing -v

# Unit-only (fast feedback while developing)
python3 -m pytest DRIFTCALL/tests/test_drift_injector.py -v

# Properties only (slower; set a lower max_examples locally)
python3 -m pytest DRIFTCALL/tests/test_drift_injector_properties.py -v \
  --hypothesis-seed=0

# Integration only
python3 -m pytest DRIFTCALL/tests/test_drift_injector_integration.py -v
```

### 6.2 CI gates (must all pass before PR merge)

1. `pytest` exits 0
2. Line coverage == 100%, branch coverage ≥ 95% on `drift_injector.py`
3. All 20 `pattern_id`s appear in the session-finish exerciser set
4. `ruff check` clean on `tests/`
5. `mypy --strict` clean on `tests/` (except `tests/fixtures/` which is data)
6. No hypothesis flaky test warnings (`hypothesis-stats` summary clean)

---

## 7. Cross-References

- Design: `DRIFTCALL/docs/modules/drift_injector.md` (final sealed) — every section of §§ 1–8 of the design is hit by at least one test above. §9 (open questions) is not tested (it's deferred).
- Master: `DRIFTCALL/DESIGN.md` §4.3 (firing point — I13, I14), §6 (module — all), §6.3 (20-pattern catalogue — U29, U32, §5.1), §7.1 R2 (detection-hint wiring — U30, P4, U32).
- Methodology: `DRIFTCALL/CLAUDE.md` §3.1 (test plan doc must contain Unit / Property / Integration / Coverage / Fixtures — this plan hits all 5 sections).
- Sibling test plan: `DRIFTCALL/docs/tests/env_tests.md` (consumes the same fixtures; I16 asserts fixture-identity).
- Reward tests: `DRIFTCALL/docs/tests/rewards_tests.md` (consumes `list_patterns()` detection_hints; owns the agent-text-vs-hints assertions — not duplicated here).

---

*End of drift_injector_tests.md.*
