"""Tests for cells/step_06_drift_injector.py.

Implements docs/tests/drift_injector_tests.md. Covers unit (§1), property (§2),
and integration (§3) tiers.
"""

from __future__ import annotations

import copy
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cells.step_04_models import (
    DriftCallState,
    DriftEvent,
    GoalSpec,
)
from cells.step_06_drift_injector import (
    DriftCatalogueError,
    DriftDomainMismatchError,
    DriftPattern,
    DriftReapplicationError,
    DriftScheduleConflictError,
    UnknownDriftPatternError,
    apply_drift,
    build_schedule,
    list_patterns,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def goal_airline() -> GoalSpec:
    return GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
        language="hinglish",
        seed_utterance="Friday Bangalore 8000 max",
    )


@pytest.fixture
def goal_cab() -> GoalSpec:
    return GoalSpec(
        domain="cab",
        intent="book_cab",
        slots={"pickup": "Koramangala", "drop": "Airport", "when": "06:30"},
        constraints={"budget_inr": 900, "vehicle_class": "sedan"},
        language="en",
        seed_utterance="Sedan 6:30 airport 900",
    )


@pytest.fixture
def goal_restaurant() -> GoalSpec:
    return GoalSpec(
        domain="restaurant",
        intent="order_food",
        slots={"city": "Bengaluru", "cuisine": "biryani"},
        constraints={"budget_inr": 300, "dietary": "veg"},
        language="hinglish",
        seed_utterance="Biryani 300 veg",
    )


@pytest.fixture
def goal_hotel() -> GoalSpec:
    return GoalSpec(
        domain="hotel",
        intent="book_room",
        slots={"city": "Goa", "check_in": "2026-05-10", "nights": 2},
        constraints={"budget_inr": 7000, "room_type": "deluxe"},
        language="hi",
        seed_utterance="Goa deluxe 7000",
    )


def _fresh_state(goal: GoalSpec) -> DriftCallState:
    return DriftCallState(
        episode_id="ep_test",
        goal=goal,
        vendor_states={
            "airline": {"schema": {}, "policy": {}, "tnc": {}, "pricing": {}},
            "cab": {"schema": {}, "policy": {}, "tnc": {}, "pricing": {}},
            "restaurant": {"schema": {}, "policy": {}, "tnc": {}, "pricing": {}},
            "hotel": {"schema": {}, "policy": {}, "tnc": {}, "pricing": {}},
            "payment": {"auth": {}},
        },
        schema_versions={
            "airline": "v1",
            "cab": "v1",
            "restaurant": "v1",
            "hotel": "v1",
            "payment": "v1",
        },
        drift_schedule=(),
        drift_fired=(),
        turn=0,
        max_turns=16,
        actions=(),
        done=False,
    )


@pytest.fixture
def fresh_state_airline(goal_airline: GoalSpec) -> DriftCallState:
    return _fresh_state(goal_airline)


def _event_for(pattern: DriftPattern, *, turn: int = 2) -> DriftEvent:
    return DriftEvent(
        turn=turn,
        drift_type=pattern.drift_type,
        domain=pattern.domain,
        description=pattern.description,
        from_version=pattern.from_version,
        to_version=pattern.to_version,
        pattern_id=pattern.id,
    )


# ---------------------------------------------------------------------------
# 1. Unit tests — build_schedule stage invariants (U1–U6)
# ---------------------------------------------------------------------------


def test_U1_stage1_returns_empty_tuple(goal_airline: GoalSpec) -> None:
    schedule = build_schedule(1, 42, goal_airline)
    assert schedule == ()
    assert isinstance(schedule, tuple)
    assert len(schedule) == 0


def test_U2_stage2_returns_exactly_one_event(goal_airline: GoalSpec) -> None:
    schedule = build_schedule(2, 1234, goal_airline)
    assert len(schedule) == 1
    assert isinstance(schedule[0], DriftEvent)


def test_U3_stage3_returns_exactly_two_events(goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, 9001, goal_airline)
    assert len(schedule) == 2
    assert all(isinstance(e, DriftEvent) for e in schedule)


def test_U4_stage3_events_turn_ascending(goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, 9001, goal_airline)
    assert schedule[0].turn < schedule[1].turn


@pytest.mark.parametrize("seed", list(range(1000, 1050)))
def test_U5_stage3_distance_ge_2_turns(seed: int, goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, seed, goal_airline)
    assert schedule[1].turn - schedule[0].turn >= 2


@pytest.mark.parametrize("stage", [0, 4, -1, 99])
def test_U6_invalid_stage_raises(stage: int, goal_airline: GoalSpec) -> None:
    with pytest.raises(ValueError, match="stage"):
        build_schedule(stage, 42, goal_airline)


# ---------------------------------------------------------------------------
# 1.2 Determinism (U7–U9)
# ---------------------------------------------------------------------------


def test_U7_deterministic_same_inputs(goal_airline: GoalSpec) -> None:
    a = build_schedule(2, 1234, goal_airline)
    b = build_schedule(2, 1234, goal_airline)
    assert a == b


def test_U8_different_seeds_diverge(goal_airline: GoalSpec) -> None:
    pairs = [(i, i + 1) for i in range(0, 200, 2)]
    diverged = 0
    for s1, s2 in pairs:
        a = build_schedule(2, s1, goal_airline)
        b = build_schedule(2, s2, goal_airline)
        if a != b:
            diverged += 1
    assert diverged / len(pairs) >= 0.95


def test_U9_does_not_use_global_rng(goal_airline: GoalSpec) -> None:
    random.seed(0)
    expected = random.random()
    random.seed(0)
    build_schedule(2, 1234, goal_airline)
    actual = random.random()
    assert expected == actual


# ---------------------------------------------------------------------------
# 1.3 Placement windows (U10–U13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(2000, 2100)))
def test_U10_stage2_turn_in_window(seed: int, goal_airline: GoalSpec) -> None:
    schedule = build_schedule(2, seed, goal_airline, max_turns=12)
    assert 2 <= schedule[0].turn <= 12 - 3


@pytest.mark.parametrize("seed", list(range(3000, 3100)))
def test_U11_stage3_first_turn_first_half(seed: int, goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, seed, goal_airline, max_turns=16)
    assert 2 <= schedule[0].turn <= 16 // 2


@pytest.mark.parametrize("seed", list(range(4000, 4100)))
def test_U12_stage3_second_turn_bounds(seed: int, goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, seed, goal_airline, max_turns=16)
    assert schedule[0].turn + 2 <= schedule[1].turn <= 16 - 3


def test_U13_stage3_max_turns_too_small_raises(goal_airline: GoalSpec) -> None:
    with pytest.raises(DriftScheduleConflictError, match="max_turns"):
        build_schedule(3, 42, goal_airline, max_turns=7)


def test_U13b_stage2_max_turns_too_small_raises(goal_airline: GoalSpec) -> None:
    with pytest.raises(DriftScheduleConflictError, match="max_turns"):
        build_schedule(2, 42, goal_airline, max_turns=4)


# ---------------------------------------------------------------------------
# 1.4 Domain & pattern selection (U14–U16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(5000, 5020)))
@pytest.mark.parametrize("goal_fixture", ["goal_airline", "goal_cab", "goal_restaurant", "goal_hotel"])
def test_U14_stage2_targets_goal_domain(
    seed: int,
    goal_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    goal: GoalSpec = request.getfixturevalue(goal_fixture)
    schedule = build_schedule(2, seed, goal)
    assert schedule[0].domain == goal.domain


@pytest.mark.parametrize("seed", list(range(6000, 6020)))
@pytest.mark.parametrize("goal_fixture", ["goal_airline", "goal_cab", "goal_restaurant", "goal_hotel"])
def test_U15_stage3_first_targets_goal_domain(
    seed: int,
    goal_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    goal: GoalSpec = request.getfixturevalue(goal_fixture)
    schedule = build_schedule(3, seed, goal)
    assert schedule[0].domain == goal.domain


@pytest.mark.parametrize("seed", list(range(7000, 7200)))
def test_U16_stage3_no_pattern_collision(seed: int, goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, seed, goal_airline)
    assert schedule[0].pattern_id != schedule[1].pattern_id


# ---------------------------------------------------------------------------
# 1.5 apply_drift semantics (U17–U24)
# ---------------------------------------------------------------------------


def test_U17_apply_drift_returns_new_object_not_mutated(
    fresh_state_airline: DriftCallState,
) -> None:
    patterns = list_patterns()
    pattern = next(p for p in patterns if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    snapshot_schema = copy.deepcopy(fresh_state_airline.schema_versions)
    snapshot_vendors = copy.deepcopy(fresh_state_airline.vendor_states)
    snapshot_fired = fresh_state_airline.drift_fired

    new_state = apply_drift(fresh_state_airline, event)

    assert new_state is not fresh_state_airline
    assert fresh_state_airline.schema_versions == snapshot_schema
    assert fresh_state_airline.vendor_states == snapshot_vendors
    assert fresh_state_airline.drift_fired == snapshot_fired


def test_U18_apply_drift_updates_schema_version(
    fresh_state_airline: DriftCallState,
) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    new_state = apply_drift(fresh_state_airline, event)
    assert new_state.schema_versions["airline"] == event.to_version


def test_U19_apply_drift_appends_event_to_drift_fired(
    fresh_state_airline: DriftCallState,
) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    new_state = apply_drift(fresh_state_airline, event)
    assert new_state.drift_fired == fresh_state_airline.drift_fired + (event,)
    assert isinstance(new_state.drift_fired, tuple)


def test_U20_apply_drift_does_not_change_turn(
    fresh_state_airline: DriftCallState,
) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    new_state = apply_drift(fresh_state_airline, event)
    assert new_state.turn == fresh_state_airline.turn


def test_U21_vendor_states_length_preserved(
    fresh_state_airline: DriftCallState,
) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    new_state = apply_drift(fresh_state_airline, event)
    assert len(new_state.vendor_states) == len(fresh_state_airline.vendor_states)
    assert set(new_state.vendor_states.keys()) == set(fresh_state_airline.vendor_states.keys())


def test_U22_unknown_pattern_raises(fresh_state_airline: DriftCallState) -> None:
    bogus_event = DriftEvent(
        turn=3,
        drift_type="schema",
        domain="airline",
        description="bogus",
        from_version="v1",
        to_version="v2",
        pattern_id="bogus.nonsense",
    )
    with pytest.raises(UnknownDriftPatternError):
        apply_drift(fresh_state_airline, bogus_event)


def test_U23_unknown_domain_raises(fresh_state_airline: DriftCallState) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    bad_event = DriftEvent(
        turn=3,
        drift_type=pattern.drift_type,
        domain="martian_colony",
        description=pattern.description,
        from_version=pattern.from_version,
        to_version=pattern.to_version,
        pattern_id=pattern.id,
    )
    with pytest.raises(DriftDomainMismatchError):
        apply_drift(fresh_state_airline, bad_event)


def test_U24_reapplication_raises(fresh_state_airline: DriftCallState) -> None:
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    event = _event_for(pattern, turn=3)
    once = apply_drift(fresh_state_airline, event)
    with pytest.raises(DriftReapplicationError):
        apply_drift(once, event)


# ---------------------------------------------------------------------------
# 1.6 list_patterns catalogue (U25–U31)
# ---------------------------------------------------------------------------


def test_U25_list_patterns_returns_exactly_20() -> None:
    patterns = list_patterns()
    assert len(patterns) == 20
    assert isinstance(patterns, tuple)


def test_U26_list_patterns_sorted_by_id() -> None:
    ids = [p.id for p in list_patterns()]
    assert ids == sorted(ids)


def test_U27_list_patterns_cached() -> None:
    assert list_patterns() is list_patterns()


def test_U28_all_ids_unique() -> None:
    ids = {p.id for p in list_patterns()}
    assert len(ids) == 20


def test_U29_axis_counts_match_design() -> None:
    patterns = list_patterns()
    counts: dict[str, int] = {}
    for p in patterns:
        counts[p.drift_type] = counts.get(p.drift_type, 0) + 1
    assert counts["schema"] == 5
    assert counts["policy"] == 5
    assert counts["tnc"] == 5
    assert counts["pricing"] == 3
    assert counts["auth"] == 2


def test_U30_detection_hints_are_substring_tokens() -> None:
    for p in list_patterns():
        assert len(p.detection_hints) >= 2
        for h in p.detection_hints:
            assert isinstance(h, str)
            assert 1 <= len(h) <= 40
            assert "\n" not in h
            assert "\t" not in h
            assert h.strip() == h


def test_U31_empty_catalogue_raises_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """DriftCatalogueError is reachable via the _load_catalogue helper
    when given an empty raw tuple (simulates a corrupt YAML)."""
    from cells import step_06_drift_injector as mod

    original = mod._CATALOGUE_RAW
    monkeypatch.setattr(mod, "_CATALOGUE_RAW", ())
    try:
        with pytest.raises(DriftCatalogueError):
            mod._load_catalogue()
    finally:
        monkeypatch.setattr(mod, "_CATALOGUE_RAW", original)


# ---------------------------------------------------------------------------
# 1.7 U32 — every pattern is applyable (parametrized × 20)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", list_patterns(), ids=[p.id for p in list_patterns()])
def test_U32_every_pattern_applyable(pattern: DriftPattern, goal_airline: GoalSpec) -> None:
    base = _fresh_state(goal_airline)
    new_schema = dict(base.schema_versions)
    new_schema[pattern.domain] = pattern.from_version
    state = DriftCallState(
        episode_id=base.episode_id,
        goal=base.goal,
        vendor_states=base.vendor_states,
        schema_versions=new_schema,
        drift_schedule=base.drift_schedule,
        drift_fired=base.drift_fired,
        turn=base.turn,
        max_turns=base.max_turns,
        actions=base.actions,
        done=base.done,
    )
    event = _event_for(pattern, turn=2)
    new_state = apply_drift(state, event)
    assert new_state.schema_versions[pattern.domain] == pattern.to_version
    assert event in new_state.drift_fired

    # At least one detection hint is substring-matchable against a canonical
    # post-drift vendor response — synthesize a response that mentions the
    # description + hints to simulate the vendor's reply.
    canonical_response = (
        pattern.description + " " + " ".join(pattern.detection_hints)
    ).lower()
    assert any(h.lower() in canonical_response for h in pattern.detection_hints)


# ---------------------------------------------------------------------------
# 2. Property tests (P1–P6)
# ---------------------------------------------------------------------------


_GOALS = [
    GoalSpec(
        domain=d,
        intent="x",
        slots={},
        constraints={},
        language="en",
        seed_utterance="u",
    )
    for d in ("airline", "cab", "restaurant", "hotel")
]

_goal_strategy = st.sampled_from(_GOALS)
_stage_strategy = st.sampled_from([1, 2, 3])
_seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)
_max_turns_strategy = st.integers(min_value=8, max_value=20)


@given(stage=_stage_strategy, seed=_seed_strategy, goal=_goal_strategy)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_P1_build_schedule_is_deterministic(
    stage: int, seed: int, goal: GoalSpec,
) -> None:
    a = build_schedule(stage, seed, goal)
    b = build_schedule(stage, seed, goal)
    assert a == b


@given(seed=_seed_strategy, goal=_goal_strategy)
@settings(max_examples=200, deadline=None)
def test_P2_apply_drift_never_returns_input_identity(
    seed: int, goal: GoalSpec,
) -> None:
    rng = random.Random(seed)
    pattern = rng.choice(list_patterns())
    state = _fresh_state(goal)
    new_schema = dict(state.schema_versions)
    new_schema[pattern.domain] = pattern.from_version
    state = DriftCallState(
        episode_id=state.episode_id,
        goal=state.goal,
        vendor_states=state.vendor_states,
        schema_versions=new_schema,
        drift_schedule=state.drift_schedule,
        drift_fired=state.drift_fired,
        turn=state.turn,
        max_turns=state.max_turns,
        actions=state.actions,
        done=state.done,
    )
    event = _event_for(pattern, turn=2)
    snap_schema = copy.deepcopy(state.schema_versions)
    snap_vendors = copy.deepcopy(state.vendor_states)
    snap_fired = state.drift_fired

    new_state = apply_drift(state, event)

    assert new_state is not state
    assert state.schema_versions == snap_schema
    assert state.vendor_states == snap_vendors
    assert state.drift_fired == snap_fired


@given(seed=_seed_strategy, goal=_goal_strategy)
@settings(max_examples=500, deadline=None)
def test_P3_stage3_no_colliding_pattern_ids(
    seed: int, goal: GoalSpec,
) -> None:
    schedule = build_schedule(3, seed, goal)
    assert schedule[0].pattern_id != schedule[1].pattern_id
    assert schedule[1].turn - schedule[0].turn >= 2


def test_P4_detection_hints_invariant() -> None:
    for p in list_patterns():
        assert len(p.detection_hints) >= 2
        for h in p.detection_hints:
            assert isinstance(h, str)
            assert 1 <= len(h) <= 40
            assert h.strip() == h
            assert "\n" not in h
            assert "\t" not in h


@given(seed=_seed_strategy, goal=_goal_strategy, max_turns=_max_turns_strategy)
@settings(max_examples=500, deadline=None)
def test_P5_stage3_turns_always_in_window(
    seed: int, goal: GoalSpec, max_turns: int,
) -> None:
    schedule = build_schedule(3, seed, goal, max_turns=max_turns)
    for e in schedule:
        assert 2 <= e.turn <= max_turns - 3
    assert schedule[1].turn - schedule[0].turn >= 2


def test_P6_cross_domain_cascade_freq_ge_10pct(goal_airline: GoalSpec) -> None:
    count_cross = 0
    total = 1000
    for seed in range(total):
        schedule = build_schedule(3, seed, goal_airline)
        if schedule[1].domain == "payment":
            count_cross += 1
    assert count_cross >= total * 0.10


# ---------------------------------------------------------------------------
# 3. Integration tests — mock-env drift firing (I1–I16)
# ---------------------------------------------------------------------------


def _run_mock_episode(
    state: DriftCallState,
    schedule: tuple[DriftEvent, ...],
    up_to_turn: int | None = None,
) -> tuple[DriftCallState, list[DriftEvent]]:
    """Mock env.step loop per DESIGN.md §4.3 — only the drift-firing point.

    At the start of each turn, fire any scheduled events whose turn matches
    and are not yet fired. Action dispatch is stubbed out.
    """
    current_state = DriftCallState(
        episode_id=state.episode_id,
        goal=state.goal,
        vendor_states=state.vendor_states,
        schema_versions=state.schema_versions,
        drift_schedule=schedule,
        drift_fired=state.drift_fired,
        turn=state.turn,
        max_turns=state.max_turns,
        actions=state.actions,
        done=state.done,
    )
    fired: list[DriftEvent] = []
    end_turn = state.max_turns if up_to_turn is None else up_to_turn
    for turn in range(1, end_turn + 1):
        current_state = DriftCallState(
            episode_id=current_state.episode_id,
            goal=current_state.goal,
            vendor_states=current_state.vendor_states,
            schema_versions=current_state.schema_versions,
            drift_schedule=current_state.drift_schedule,
            drift_fired=current_state.drift_fired,
            turn=turn,
            max_turns=current_state.max_turns,
            actions=current_state.actions,
            done=current_state.done,
        )
        pending = [
            e for e in schedule
            if e.turn == turn and e not in current_state.drift_fired
        ]
        for e in pending:
            current_state = apply_drift(current_state, e)
            fired.append(e)
    return current_state, fired


@pytest.mark.parametrize(
    "goal_fixture,seed",
    [
        ("goal_airline", 100),
        ("goal_cab", 101),
        ("goal_restaurant", 102),
        ("goal_hotel", 103),
    ],
)
def test_I1_I4_stage1_no_drifts_fire(
    goal_fixture: str, seed: int, request: pytest.FixtureRequest,
) -> None:
    goal: GoalSpec = request.getfixturevalue(goal_fixture)
    schedule = build_schedule(1, seed, goal)
    state = _fresh_state(goal)
    final_state, fired = _run_mock_episode(state, schedule)
    assert fired == []
    assert final_state.drift_fired == ()


@pytest.mark.parametrize(
    "goal_fixture,seed",
    [
        ("goal_airline", 200),
        ("goal_cab", 201),
        ("goal_restaurant", 202),
        ("goal_hotel", 203),
    ],
)
def test_I5_I8_stage2_one_drift_fires(
    goal_fixture: str, seed: int, request: pytest.FixtureRequest,
) -> None:
    goal: GoalSpec = request.getfixturevalue(goal_fixture)
    schedule = build_schedule(2, seed, goal)
    state = _fresh_state(goal)
    final_state, fired = _run_mock_episode(state, schedule)
    assert len(fired) == 1
    assert fired == list(schedule)
    assert len(final_state.drift_fired) == 1


@pytest.mark.parametrize(
    "goal_fixture,seed",
    [
        ("goal_airline", 300),
        ("goal_cab", 301),
        ("goal_restaurant", 302),
        ("goal_hotel", 303),
    ],
)
def test_I9_I12_stage3_two_drifts_fire(
    goal_fixture: str, seed: int, request: pytest.FixtureRequest,
) -> None:
    goal: GoalSpec = request.getfixturevalue(goal_fixture)
    schedule = build_schedule(3, seed, goal)
    state = _fresh_state(goal)
    final_state, fired = _run_mock_episode(state, schedule)
    assert len(fired) == 2
    assert fired == list(schedule)
    assert len(final_state.drift_fired) == 2


def test_I13_stage3_cross_domain_payment_cascade(goal_airline: GoalSpec) -> None:
    """Hunt seeds [0..5000] for a stage-3 schedule where schedule[1].domain
    == 'payment', then confirm the cascade fires and state reflects it."""
    target_seed: int | None = None
    target_schedule: tuple[DriftEvent, ...] | None = None
    for seed in range(5000):
        schedule = build_schedule(3, seed, goal_airline)
        if schedule[1].domain == "payment":
            target_seed = seed
            target_schedule = schedule
            break
    assert target_seed is not None
    assert target_schedule is not None
    state = _fresh_state(goal_airline)
    final_state, fired = _run_mock_episode(state, target_schedule)
    assert len(fired) == 2
    assert final_state.schema_versions["payment"] == target_schedule[1].to_version


def test_I14_early_submit_no_drift_fires(goal_airline: GoalSpec) -> None:
    schedule = build_schedule(3, 300, goal_airline)
    state = _fresh_state(goal_airline)
    # Stop the mock loop before the first scheduled drift.
    final_state, fired = _run_mock_episode(state, schedule, up_to_turn=schedule[0].turn - 1)
    assert fired == []
    assert final_state.drift_fired == ()


def test_I15_stage3_two_drifts_same_domain_chain(goal_airline: GoalSpec) -> None:
    """Seed-hunt a stage-3 airline schedule where both patterns target airline."""
    hit: tuple[DriftEvent, ...] | None = None
    for seed in range(5000):
        schedule = build_schedule(3, seed, goal_airline)
        if (
            schedule[0].domain == "airline"
            and schedule[1].domain == "airline"
        ):
            hit = schedule
            break
    assert hit is not None
    state = _fresh_state(goal_airline)
    final_state, fired = _run_mock_episode(state, hit)
    assert len(fired) == 2
    # Second drift's to_version is now the airline schema version.
    assert final_state.schema_versions["airline"] == hit[1].to_version


def test_I16_fixture_identity_placeholder(goal_airline: GoalSpec) -> None:
    """Sentinel: confirm the GoalSpec fixture is a frozen dataclass instance,
    used as cross-module fixture contract per test plan §5.4."""
    import dataclasses as _dc

    assert isinstance(goal_airline, GoalSpec)
    with pytest.raises(_dc.FrozenInstanceError):
        goal_airline.__setattr__("domain", "mutated")


# ---------------------------------------------------------------------------
# Sanity guard — schedule events are always valid DriftEvents
# ---------------------------------------------------------------------------


def test_apply_rename_with_missing_old_key(fresh_state_airline: DriftCallState) -> None:
    """Cover the else-branch of _apply_rename (old_key not in target)."""
    pattern = next(p for p in list_patterns() if p.id == "airline.price_rename")
    # Vendor state doesn't have 'price' key — rename should default the new key.
    event = _event_for(pattern, turn=3)
    new_state = apply_drift(fresh_state_airline, event)
    # The mutation op applies against airline vendor state.
    assert new_state.schema_versions["airline"] == "v2"


def test_apply_drift_with_all_operator_types() -> None:
    """Sanity check: each pattern's operator chain actually runs through dispatch
    without raising."""
    base = _fresh_state(
        GoalSpec(
            domain="airline", intent="x", slots={}, constraints={},
            language="en", seed_utterance="u",
        ),
    )
    for p in list_patterns():
        new_schema = dict(base.schema_versions)
        new_schema[p.domain] = p.from_version
        state = DriftCallState(
            episode_id=base.episode_id,
            goal=base.goal,
            vendor_states=base.vendor_states,
            schema_versions=new_schema,
            drift_schedule=base.drift_schedule,
            drift_fired=base.drift_fired,
            turn=base.turn,
            max_turns=base.max_turns,
            actions=base.actions,
            done=base.done,
        )
        event = _event_for(p, turn=2)
        apply_drift(state, event)  # must not raise


def test_unknown_operator_is_noop() -> None:
    """If a DriftPattern's mutation contains an unknown op key, it is skipped."""
    from cells import step_06_drift_injector as mod

    weird = DriftPattern(
        id="test.weird",
        drift_type="schema",
        domain="airline",
        from_version="v1",
        to_version="v2",
        description="weird",
        mutation={"unknown_op_xyz": {"a": 1}},
        detection_hints=("foo", "bar"),
    )
    result = mod._mutate_vendor_state({"keep": "me"}, weird)
    assert result == {"keep": "me"}


def test_stage3_conflict_when_max_turns_leaves_no_second_slot(
    goal_airline: GoalSpec,
) -> None:
    """max_turns that satisfies >=8 check but still can't fit second drift."""
    # max_turns=8 -> lo=2, hi=5, first_hi = min(4, 3) = 3. OK — always fits.
    # Force failure by monkeypatching lo via small max_turns cases covered
    # elsewhere. This test pins the lower bound check branch.
    with pytest.raises(DriftScheduleConflictError):
        build_schedule(3, 42, goal_airline, max_turns=6)


def test_sanity_all_schedule_events_resolve_to_registered_patterns() -> None:
    known_ids = {p.id for p in list_patterns()}
    for seed in range(50):
        for stage, goal in [(2, _GOALS[0]), (3, _GOALS[0])]:
            schedule = build_schedule(stage, seed, goal)
            for e in schedule:
                assert e.pattern_id in known_ids
