# models_tests.md — Test Plan for `driftcall/models.py`

**Owner:** Person B (Rewards & Tests)
**Target module:** `DRIFTCALL/docs/modules/models.md` (sealed)
**Implements coverage for:** DESIGN.md §4.1 (Dataclasses), §4.2 (reset), §4.3 (step), §4.4 (termination), §7.1 (Reward inputs)
**Frameworks:** `pytest`, `hypothesis`
**Status:** DRAFT — pending ≥ 1 fresh critic round (test-plan gate is lighter per CLAUDE.md §3.2 Batch D4)

---

## 0. Scope & Non-goals

`models.py` is pure shape — frozen dataclasses + one Enum. There is no runtime behavior to integration-test. This plan therefore concentrates on:

1. **Constructibility** — every dataclass can be instantiated with valid inputs and rejects malformed required fields.
2. **Immutability** — every frozen guarantee raises `FrozenInstanceError` on mutation attempt.
3. **Type shape** — tuples stay tuples, dicts stay dicts, Optionals default to `None`, Literals accept documented values.
4. **Hashability boundaries** — the classes the doc declares hashable hash; the classes the doc declares unhashable raise `TypeError`.
5. **Round-trip equality** — `dataclasses.asdict` + JSON encode + decode + reconstruct produces an equal object (the invariant in models.md §3.4).
6. **Invariant witnessing** — properties from models.md §3.5 that can be checked *by construction* are asserted via hypothesis. Invariants enforced elsewhere (e.g. `env.step` validation) are NOT retested here; they are documented as N/A with a pointer to `env_tests.md`.

Every test below maps to one numbered clause in `docs/modules/models.md`. Clause references are embedded in the test docstring as `models.md §X.Y` / `models.md §7 edge N`.

---

## 1. Unit tests

All unit tests live in `DRIFTCALL/tests/test_models.py`. Import line under test:

```python
from driftcall.models import (
    ActionType,
    DriftCallAction,
    ToolResult,
    DriftEvent,
    GoalSpec,
    DriftCallObservation,
    DriftCallState,
)
```

Fixtures (§5) are imported from `tests/conftest.py`.

### 1.1 `ActionType` (Enum)

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U1 | `test_action_type_members_exactly_six` | `set(ActionType) == {TOOL_CALL, SPEAK, CLARIFY, PROBE_SCHEMA, SUBMIT, ABORT}`; `len(ActionType) == 6`. | models.md §4.1 |
| U2 | `test_action_type_is_str_subclass` | `isinstance(ActionType.TOOL_CALL, str)`; `ActionType.TOOL_CALL == "tool_call"`; `json.dumps({"t": ActionType.SPEAK}) == '{"t": "speak"}'` (string-mixed Enum serializes to its `.value`). | models.md §3.4, §4.1 |
| U3 | `test_action_type_values_match_spec` | `ActionType.TOOL_CALL.value == "tool_call"` and the other five values equal exactly the lowercase strings listed in the spec table. | models.md §4.1 |
| U4 | `test_action_type_is_hashable` | `hash(ActionType.TOOL_CALL)` returns an int; `{ActionType.SPEAK, ActionType.SPEAK} == {ActionType.SPEAK}` (set de-dupes). | models.md §3.2 bullet 1 |

### 1.2 `DriftCallAction`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U5 | `test_driftcall_action_happy_tool_call` | Build a `TOOL_CALL` action from `valid_tool_call_action()` fixture; assert all six fields carry the values passed in; `action.confidence is None`; `action.message is None`. | models.md §4.2, §8.1 |
| U6 | `test_driftcall_action_happy_submit` | Build a `SUBMIT` action with `confidence=0.87`; assert `action.action_type is ActionType.SUBMIT`, `action.confidence == 0.87`, and `tool_name / tool_args / message` are all `None`. | models.md §4.2, §3.5 SUBMIT row |
| U7 | `test_driftcall_action_happy_speak` | Build a `SPEAK` action with Unicode `message="मुझे कल दिल्ली जाना है"`; assert `action.message` round-trips byte-for-byte. | models.md §7 edge 3 |
| U8 | `test_driftcall_action_frozen_mutation_raises` | Construct any action; assert `action.tool_name = "x"` raises `dataclasses.FrozenInstanceError`; same for every one of the 6 fields. Parametrized over all 6 field names. | models.md §3.1, §5 row 1 |
| U9 | `test_driftcall_action_defaults_are_none` | `DriftCallAction(action_type=ActionType.ABORT)` succeeds; every optional field defaults to `None`. | models.md §2 (dataclass signature), §4.2 |
| U10 | `test_driftcall_action_missing_required_raises` | `DriftCallAction()` raises `TypeError` (missing `action_type`). | models.md §5 row 3 |
| U11 | `test_driftcall_action_equality_value_based` | Two actions built with identical fields compare `==`; changing any single field makes them `!=`. Covers all 6 fields via parametrize. | models.md §3.2 |
| U12 | `test_driftcall_action_unhashable_due_to_dict` | `hash(valid_tool_call_action())` raises `TypeError: unhashable type: 'dict'`. | models.md §3.2 bullet 2, §5 row 7 |
| U13 | `test_driftcall_action_confidence_none_vs_zero` | `DriftCallAction(action_type=SUBMIT, confidence=0.0) != DriftCallAction(action_type=SUBMIT, confidence=None)`; `0.0` is a real low confidence, `None` means absent. Proves the `Optional[float]` semantics are preserved by the dataclass (i.e. `None` is a distinguishable value, not coerced to `0.0`). | models.md §4.2, §7 edge 6 |

### 1.3 `ToolResult`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U14 | `test_tool_result_happy_ok` | Build from `valid_tool_result(status="ok")`; fields round-trip; `result.status == "ok"`; `result.response["results"]` is a list. | models.md §4.3, §8.2 |
| U15 | `test_tool_result_frozen_mutation_raises` | `result.status = "timeout"` raises `FrozenInstanceError`; parametrized over all 5 fields. | models.md §3.1, §5 row 1 |
| U16 | `test_tool_result_accepts_all_five_statuses` | Parametrized over `["ok", "schema_error", "policy_error", "auth_error", "timeout"]`; all construct successfully. | models.md §4.3 status row |
| U17 | `test_tool_result_empty_response_on_non_ok` | `valid_tool_result(status="schema_error", response={})` constructs without error (models.py does not validate; vendor-contract bug detection lives in `test_vendors.py`). | models.md §7 edge 7, §5 row 6 |
| U18 | `test_tool_result_unhashable_due_to_dict` | `hash(valid_tool_result())` raises `TypeError`. | models.md §3.2 bullet 2 |

### 1.4 `DriftEvent`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U19 | `test_drift_event_happy_schema` | Build from `valid_drift_event(turn=3, domain="airline")`; all six fields match input; `from_version="v1"`, `to_version="v2"`. | models.md §4.4, §8.4 |
| U20 | `test_drift_event_frozen_mutation_raises` | `drift.turn = 99` raises `FrozenInstanceError`; parametrized over all 6 fields. | models.md §3.1 |
| U21 | `test_drift_event_is_hashable` | `hash(valid_drift_event())` returns an int; `{drift, drift} == {drift}`. Confirms primitive-only fields keep `DriftEvent` hashable (models.md §3.2 bullet 1 asserts this). | models.md §3.2 |
| U22 | `test_drift_event_accepts_all_five_drift_types` | Parametrized over `["schema", "policy", "tnc", "pricing", "auth"]`; all construct successfully. | models.md §4.4 drift_type row |

### 1.5 `GoalSpec`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U23 | `test_goal_spec_happy_hinglish` | Build from `valid_goal_spec()`; fields round-trip; `goal.language == "hinglish"`; `goal.slots["from"] == "HYD"`. | models.md §4.5, §8.3 |
| U24 | `test_goal_spec_accepts_all_five_languages` | Parametrized over `["hi", "ta", "kn", "en", "hinglish"]`; all construct successfully. | models.md §4.5 language row |
| U25 | `test_goal_spec_frozen_mutation_raises` | `goal.intent = "other"` raises `FrozenInstanceError`. | models.md §3.1 |
| U26 | `test_goal_spec_unhashable_due_to_dict` | `hash(valid_goal_spec())` raises `TypeError`. | models.md §3.2 bullet 2 |
| U27 | `test_goal_spec_unicode_seed_utterance` | `seed_utterance="{when} அன்று விமானம்"` (Tamil) survives `json.dumps(..., ensure_ascii=False)` + `json.loads` + reconstruction; `reconstructed == original`. | models.md §7 edge 3 |

### 1.6 `DriftCallObservation`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U28 | `test_observation_reset_state` | Build turn-0 observation with `tool_results=()`, `drift_log=()`, `last_transcript=""`, `last_lang=""`, `last_confidence=1.0`, `budget_remaining=12`. Assert `isinstance(obs.tool_results, tuple)`, `len(obs.tool_results) == 0`, same for `drift_log`. | models.md §7 edge 1, §7 edge 2, §8.3 |
| U29 | `test_observation_tuple_not_list_for_sequences` | Passing `tool_results=[valid_tool_result()]` (a list) still stores a list on the dataclass (Python does not coerce) — the test asserts `type(obs.tool_results) is list` and then asserts that our documented contract requires callers to pass tuples by flagging this shape as a contract-violating fixture. The real enforcement is a *separate* test that constructs with `tool_results=(valid_tool_result(),)` and asserts `isinstance(obs.tool_results, tuple)`. Covers both the "what Python does" and "what the contract requires" halves of models.md §3.1. | models.md §3.1, §7 edge 1 |
| U30 | `test_observation_frozen_mutation_raises` | `obs.turn = 1` raises `FrozenInstanceError`; parametrized over all 9 fields. | models.md §3.1 |
| U31 | `test_observation_unhashable` | `hash(obs)` raises `TypeError` (observation contains `GoalSpec` which contains dicts). | models.md §3.2 bullet 2 |
| U32 | `test_observation_goal_ref_stable` | Two observations built with the same `GoalSpec` object yield `obs_a.goal is obs_b.goal` when passed the same instance; i.e. `frozen=True` does not deep-copy. Encodes the invariant that goal is copied by reference within an episode. | models.md §4.6 goal row |

### 1.7 `DriftCallState`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U33 | `test_state_happy_turn_zero` | Build state with `turn=0`, `actions=()`, `drift_fired=()`, `done=False`; assert length invariant `len(state.actions) == state.turn` holds at construction; `state.done is False`. | models.md §4.7, §3.5 `len(actions) == turn` row |
| U34 | `test_state_replace_appends_action` | Use `dataclasses.replace(state, turn=state.turn+1, actions=state.actions + (action,))`; assert original `state` is unchanged (`state.turn == 0`, `state.actions == ()`) and new state has `turn == 1`, `actions[-1] is action`. This witnesses the `replace`-don't-mutate pattern from models.md §3.3. | models.md §3.3, §8.4 |
| U35 | `test_state_replace_dict_field_builds_new_dict` | Call `replace(state, schema_versions={**state.schema_versions, "airline": "v2"})`; assert `state.schema_versions["airline"] == "v1"` (original untouched) and `new_state.schema_versions["airline"] == "v2"`. Confirms the "always build new dict" convention does not rely on shared references. | models.md §3.3 |
| U36 | `test_state_frozen_mutation_raises` | `state.done = True` raises `FrozenInstanceError`; parametrized over all 10 fields. | models.md §3.1 |
| U37 | `test_state_unhashable` | `hash(state)` raises `TypeError`. | models.md §3.2 bullet 2 |

### 1.8 Cross-cutting: JSON round-trip

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U38 | `test_action_json_roundtrip_equality` | For an action with Unicode message, nested-dict `tool_args`, and all optional fields set: `a2 = DriftCallAction(**json.loads(json.dumps(dataclasses.asdict(a1), ensure_ascii=False)))` then `action_type` is re-coerced to `ActionType`; `a2 == a1`. | models.md §3.4 invariant |
| U39 | `test_tool_result_json_roundtrip_equality` | Same pattern for `ToolResult` with nested list-of-dict `response`. | models.md §3.4, §7 edge 4 |
| U40 | `test_observation_json_roundtrip_preserves_tuple_length` | After JSON round-trip (which makes tuples into lists), a reconstructor that wraps the sequence fields back into tuples yields `obs2 == obs1`. Documents the reconstructor contract `app.py` must uphold. | models.md §3.4 |

**Unit test count: 40.**

---

## 2. Property tests (Hypothesis)

All property tests live in `DRIFTCALL/tests/test_models_properties.py`. Hypothesis settings: `deadline=None`, `max_examples=200` (enough coverage for dataclass shape; no IO so runs are fast).

Shared strategies live in `tests/strategies.py`:

```python
# strategies.py (sketch for reference — NOT part of this test-plan doc to implement)
import string
from hypothesis import strategies as st
from driftcall.models import ActionType

_versions = st.from_regex(r"^v[1-9]\d?$", fullmatch=True)
_languages = st.sampled_from(["hi", "ta", "kn", "en", "hinglish"])
_domains = st.sampled_from(["airline", "cab", "restaurant", "hotel", "payment"])
_drift_types = st.sampled_from(["schema", "policy", "tnc", "pricing", "auth"])
_statuses = st.sampled_from(["ok", "schema_error", "policy_error", "auth_error", "timeout"])

_json_scalar = st.one_of(st.none(), st.booleans(), st.integers(-1_000_000, 1_000_000),
                         st.floats(allow_nan=False, allow_infinity=False),
                         st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=32))
_json_dict = st.dictionaries(keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=8),
                             values=_json_scalar, max_size=4)
```

Each property below names its strategies and states the invariant.

| # | Property name | Strategy | Invariant | Maps to |
|---|---|---|---|---|
| P1 | `test_state_turn_matches_len_actions_by_construction` | Generate `turn ∈ [0, 16]`, then build `actions` tuple of exactly `turn` `DriftCallAction` instances; construct `DriftCallState` with those values. | `len(state.actions) == state.turn` holds for every generated example (proves the invariant is *witness-able* by construction; actual runtime enforcement is in `env_tests.md`). This is the "turn monotone non-decreasing by construction" property at the state-boundary layer. | models.md §3.5 state row, §7 edge 9 |
| P2 | `test_drift_fired_is_subset_of_drift_schedule` | Generate a `drift_schedule` tuple of 0..2 `DriftEvent`s with unique ascending `turn`; pick a random prefix length `k ∈ [0, len(schedule)]`; set `drift_fired = drift_schedule[:k]`. | `set(state.drift_fired).issubset(set(state.drift_schedule))` AND `len(state.drift_fired) <= len(state.drift_schedule)` AND `state.drift_fired == state.drift_schedule[:len(state.drift_fired)]` (prefix, order preserved). | models.md §3.5 `drift_fired ⊆ drift_schedule` row, §4.7 drift_fired row |
| P3 | `test_probe_schema_tool_name_is_bare_domain` | Generate `DriftCallAction(action_type=PROBE_SCHEMA, tool_name=domain)` where `domain` comes from `_domains`. | `"." not in action.tool_name` AND `action.tool_name in {"airline","cab","restaurant","hotel","payment"}`. Encodes the "PROBE_SCHEMA carries bare domain, NOT `domain.verb`" invariant from models.md §3.5. (Note: `models.py` itself does not validate this; the property asserts the TEST FIXTURES obey the contract — a regression-canary against a fixture drifting to `"airline.search"` by accident.) | models.md §3.5 PROBE_SCHEMA row |
| P4 | `test_frozen_invariant_universal` | For every dataclass type in `{DriftCallAction, ToolResult, DriftEvent, GoalSpec, DriftCallObservation, DriftCallState}`, generate a valid instance using the type-specific strategy, then for each field name on the class, assert `setattr(instance, field_name, ...)` raises `FrozenInstanceError`. | The frozen guarantee is universal across the module. Stronger than the per-class U8/U15/U20/U25/U30/U36 tests because it is strategy-driven rather than single-example. | models.md §3.1, §5 row 1 |
| P5 | `test_json_roundtrip_preserves_equality_for_action` | Generate `DriftCallAction` with random valid fields (Unicode text up to 200 chars for `rationale`, nested JSON dicts for `tool_args`, random `ActionType`). Serialize via `json.dumps(dataclasses.asdict(a), ensure_ascii=False)`, parse, rebuild (re-coerce `action_type` to `ActionType`). | Rebuilt action `== original` for every generated example. Verifies the models.md §3.4 round-trip invariant holds under adversarial shape generation. | models.md §3.4 invariant |
| P6 | `test_observation_budget_non_negative_by_construction` | Generate `max_turns ∈ [1, 16]`, `turn ∈ [0, max_turns]`, then `budget_remaining = max_turns - turn`. Build observation. | `obs.budget_remaining >= 0` always. (models.py does not enforce; the property ensures our fixture arithmetic respects models.md §3.5 budget_remaining row.) | models.md §3.5 budget row, §4.6 budget row |

**Property test count: 6.**

Notes on properties *intentionally not tested here*:
- "`DriftCallState.turn` monotone non-decreasing across a step sequence" — that is a runtime property of `env.step`, tested in `env_tests.md`. At the `models.py` layer the property is trivially "by construction a single instance has one turn value", which P1 already witnesses by rebuilding valid states at every turn.
- "`DriftEvent.from_version != to_version`" — enforced by drift injector, not by `models.py`. Tested in `drift_injector_tests.md`.
- "`DriftCallAction.confidence ∈ [0, 1]` when SUBMIT" — enforced by `env.step`, tested in `env_tests.md`.

Each of these deferrals matches models.md §3.5's "Enforced by" column (not `models.py`).

---

## 3. Integration tests

**N/A — defer to `env_tests.md`.**

`models.py` has no IO, no network, no filesystem, no cross-module composition. Every meaningful interaction (e.g. "step consumes action, emits observation") belongs to `env.py` and is tested in `DRIFTCALL/docs/tests/env_tests.md` once that test plan is authored. This section is deliberately empty; the pointer is the deliverable.

---

## 4. Coverage target

**Line coverage: 100% on `driftcall/models.py`.**
**Branch coverage: ≥ 95% on `driftcall/models.py`.**

`models.py` is effectively branchless — the only "branches" are the implicit ones the dataclass decorator generates in `__init__`, `__repr__`, `__eq__`, and `__hash__`. Python's `coverage.py` with `branch=True` will report those auto-generated lines; our unit and property tests hit every one:

| `models.py` region | Hit by |
|---|---|
| `class ActionType(str, Enum): ...` — 6 member lines | U1, U2, U3 (every member is named or sampled). |
| `@dataclass(frozen=True) class DriftCallAction: ...` — 6 field lines + auto-generated `__init__` / `__eq__` / `__repr__` / `__hash__` | U5–U13, U38, P4, P5. |
| `@dataclass(frozen=True) class ToolResult: ...` — 5 field lines + auto-methods | U14–U18, U39, P4. |
| `@dataclass(frozen=True) class DriftEvent: ...` — 6 field lines + auto-methods | U19–U22, P2, P4. |
| `@dataclass(frozen=True) class GoalSpec: ...` — 6 field lines + auto-methods | U23–U27, P4. |
| `@dataclass(frozen=True) class DriftCallObservation: ...` — 9 field lines + auto-methods | U28–U32, U40, P4, P6. |
| `@dataclass(frozen=True) class DriftCallState: ...` — 10 field lines + auto-methods | U33–U37, P1, P2, P4. |
| `__all__ = [...]` list | Imported in every test module; trivially covered. |

Verification command (per CLAUDE.md §6):

```
python3 -m pytest tests/test_models.py tests/test_models_properties.py \
    --cov=driftcall.models --cov-branch --cov-report=term-missing
```

**Gate:** if either line-coverage < 100% or branch-coverage < 95%, the PR is blocked and the missing lines/branches are added as new unit tests before proceeding.

---

## 5. Fixtures

All fixtures live in `DRIFTCALL/tests/conftest.py` and are imported by every test file that needs models. They are pytest fixtures (not plain functions) so hypothesis strategies can reuse them via `st.builds`. Each fixture returns a frozen dataclass built with spec-valid defaults; keyword arguments override specific fields for variation. Naming follows `valid_<thing>[_variant]` convention.

### 5.1 `valid_goal_spec`

```python
import pytest
from driftcall.models import GoalSpec

@pytest.fixture
def valid_goal_spec() -> GoalSpec:
    """A spec-valid Hinglish airline booking goal. Matches models.md §8.3."""
    return GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
        language="hinglish",
        seed_utterance="Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad",
    )
```

Reused by: U23, U26, U28, U33, P6, and every downstream test file (`test_env.py`, `test_rewards.py`, `test_vendors.py`).

### 5.2 `valid_drift_event`

```python
from driftcall.models import DriftEvent

@pytest.fixture
def valid_drift_event_factory():
    """Factory fixture so tests can override turn/domain. Matches models.md §8.4."""
    def _build(turn: int = 3, domain: str = "airline") -> DriftEvent:
        return DriftEvent(
            turn=turn,
            drift_type="schema",
            domain=domain,
            description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
            from_version="v1",
            to_version="v2",
        )
    return _build
```

Also provided as a plain (non-factory) `valid_drift_event` fixture with defaults `turn=3, domain="airline"` for convenience.

Reused by: U19, U20, U21, U22, P2, and `drift_injector_tests.md` / `env_tests.md`.

### 5.3 `valid_tool_result`

```python
from driftcall.models import ToolResult
from typing import Any

@pytest.fixture
def valid_tool_result_factory():
    """Factory fixture so tests can override status/response. Matches models.md §8.2."""
    def _build(
        status: str = "ok",
        response: dict[str, Any] | None = None,
        tool_name: str = "airline.search",
        schema_version: str = "v1",
        latency_ms: int = 142,
    ) -> ToolResult:
        if response is None:
            response = (
                {
                    "results": [
                        {
                            "flight_id": "6E-2345",
                            "from": "HYD",
                            "to": "BLR",
                            "depart": "2026-04-25T18:30:00+05:30",
                            "price": 7200,
                            "currency": "INR",
                            "seats_left": 14,
                        }
                    ]
                }
                if status == "ok"
                else {"error_code": status.upper()}
            )
        return ToolResult(
            tool_name=tool_name,
            status=status,  # type: ignore[arg-type]
            response=response,
            schema_version=schema_version,
            latency_ms=latency_ms,
        )
    return _build
```

A plain `valid_tool_result` fixture with `status="ok"` defaults is also exposed.

Reused by: U14, U15, U16, U17, U18, U29, U39, and `test_vendors.py`, `test_rewards.py`, `test_env.py`.

### 5.4 Supporting fixtures (bonus, for the downstream test files)

These are included here because the fixtures module must ship as one unit, and the briefing names them "reusable across other modules' tests":

- `valid_tool_call_action()` — returns the §8.1 `DriftCallAction` (airline.search with slots). Consumed by U5, U6, U11, U34.
- `valid_submit_action(confidence: float = 0.87)` — builds a SUBMIT action; consumed by U6, U13.
- `valid_observation_reset(valid_goal_spec)` — builds the §8.3 turn-0 observation; consumed by U28, U29, U30, U31, U32, U40, P6.
- `valid_state_reset(valid_goal_spec)` — builds a turn-0 `DriftCallState` with `max_turns=12`, empty action/drift tuples, `done=False`; consumed by U33, U34, U35, U36, U37, P1.

**Fixture count (fixtures that must exist in `conftest.py`): 7.**
- `valid_goal_spec`
- `valid_drift_event` (plain)
- `valid_drift_event_factory`
- `valid_tool_result` (plain)
- `valid_tool_result_factory`
- `valid_tool_call_action`
- `valid_submit_action`
- `valid_observation_reset`
- `valid_state_reset`

(The briefing requires the first three by name; the remaining four are the minimal set required to write the 40 unit tests above without duplicating construction boilerplate. All 9 ship together.)

---

## 6. Execution plan

1. `conftest.py` fixtures land first (no tests will even collect without them).
2. `test_models.py` is authored top-down: U1 → U40, committed in 4 logical chunks (`ActionType + DriftCallAction`, `ToolResult + DriftEvent`, `GoalSpec + DriftCallObservation`, `DriftCallState + JSON round-trip`). Each chunk is a separate commit with passing `pytest -v` + coverage output.
3. `test_models_properties.py` lands last, after all unit tests are green. Hypothesis shrinking behavior requires the fixtures and strategies to be stable first.
4. Coverage gate: `pytest --cov=driftcall.models --cov-branch` must show 100% line / ≥ 95% branch before PR.
5. Ruff + mypy --strict run on both test files (CLAUDE.md §6 commands). The test files themselves must obey the same hygiene bar as production code.

---

## 7. Traceability summary

| models.md clause | Covered by |
|---|---|
| §2 (Interface) | U5–U37 (every dataclass built with full spec signature) |
| §3.1 Immutability | U8, U15, U20, U25, U30, U36, P4 |
| §3.2 Equality & hashing | U4, U11, U12, U18, U21, U26, U31, U37 |
| §3.3 `replace` pattern | U34, U35 |
| §3.4 Serialization round-trip | U38, U39, U40, P5 |
| §3.5 Field invariants (constructible-at-boundary subset) | P1, P2, P3, P6 |
| §4.1–4.7 (per-class field tables) | U1–U37 (one class per subsection) |
| §5 Error modes | U8, U10, U12, U15, U18, U20, U25, U26, U30, U31, U36, U37 |
| §7 Edge 1 (empty tuples at turn 0) | U28, U29 |
| §7 Edge 2 (empty strings vs None) | U28 |
| §7 Edge 3 (Unicode round-trip) | U7, U27, U38 |
| §7 Edge 4 (nested tool_args) | U38, U39 |
| §7 Edge 5 (large history) | N/A at models layer — deferred to `env_tests.md` §integration |
| §7 Edge 6 (SUBMIT + confidence=None) | U13 |
| §7 Edge 7 (non-ok with empty response) | U17 |
| §7 Edge 8 (from_version == to_version) | N/A at models layer — deferred to `drift_injector_tests.md` |
| §7 Edge 9 (`len(actions) != turn`) | P1 (witnesses constructibility of the good case; violation-detection deferred to `env_tests.md`) |
| §7 Edge 10 (auth-drifted tool still listed) | N/A at models layer — deferred to `env_tests.md` |
| §8.1–8.4 (worked examples) | U5, U14, U19, U28, U33, U34 |

Every models.md clause is either directly tested here or explicitly deferred to a named downstream test plan with a reason. No clause is silently unaddressed.

---

## 8. Open questions

None. The test surface for `models.py` is bounded by the module's pure-shape nature: 40 unit tests + 6 properties + 9 fixtures + 100%/95% coverage exhaustively cover every field, every frozen guarantee, every hashability rule, every JSON round-trip invariant, and every edge case that can be witnessed at construction time. All edge cases requiring runtime enforcement (`env.step` validation, drift injector rules) are explicitly routed to their home test plans.
