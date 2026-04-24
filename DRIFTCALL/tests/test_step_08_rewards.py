"""Tests for cells/step_08_rewards.py.

Implements docs/tests/rewards_tests.md §1 (81 unit tests), §2 (property tests),
§3 (integration tests), §4.2 (no-LLM-judge enforcement).

Fixtures follow docs/tests/rewards_tests.md §5 — the five shared Episode fixtures.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import pathlib
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftEvent,
    GoalSpec,
    ToolResult,
)
from cells.step_06_drift_injector import DriftPattern
from cells.step_08_rewards import (
    AVAILABLE_TOOL_REGISTRY,
    Episode,
    RewardComputationError,
    anti_hack_penalty,
    apply_uncertain_floor,
    brier_penalty,
    combine_quality,
    compute_rewards,
    constraint_adherence,
    drift_detection,
    final_reward,
    format_compliance,
    task_completion,
)

# ---------------------------------------------------------------------------
# Helper builders (private, mirror §5 _make_* helpers)
# ---------------------------------------------------------------------------


def _mk_action(
    turn: int,
    type: ActionType,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    message: str | None = None,
    confidence: float | None = None,
    rationale: str | None = "ok",
) -> tuple[int, DriftCallAction]:
    return (
        turn,
        DriftCallAction(
            action_type=type,
            tool_name=tool_name,
            tool_args=tool_args,
            message=message,
            confidence=confidence,
            rationale=rationale,
        ),
    )


def _mk_tr(
    turn: int,
    tool_name: str,
    status: str = "ok",
    response: dict[str, Any] | None = None,
    schema_version: str = "v1",
    latency_ms: int = 42,
) -> tuple[int, ToolResult]:
    return (
        turn,
        ToolResult(
            tool_name=tool_name,
            status=status,  # type: ignore[arg-type]
            response=response if response is not None else {},
            schema_version=schema_version,
            latency_ms=latency_ms,
        ),
    )


def _mk_drift(
    pattern_id: str,
    turn: int,
    drift_type: str = "schema",
    domain: str = "airline",
    description: str = "test drift",
    from_version: str = "v1",
    to_version: str = "v2",
) -> DriftEvent:
    return DriftEvent(
        turn=turn,
        drift_type=drift_type,  # type: ignore[arg-type]
        domain=domain,
        description=description,
        from_version=from_version,
        to_version=to_version,
        pattern_id=pattern_id,
    )


def _mk_pattern(
    pattern_id: str,
    detection_hints: tuple[str, ...],
    mutation: dict[str, Any] | None = None,
    drift_type: str = "schema",
    domain: str = "airline",
    from_version: str = "v1",
    to_version: str = "v2",
) -> DriftPattern:
    return DriftPattern(
        id=pattern_id,
        drift_type=drift_type,  # type: ignore[arg-type]
        domain=domain,
        from_version=from_version,
        to_version=to_version,
        description="test",
        mutation=mutation if mutation is not None else {"rename": {"price": "total_fare_inr"}},
        detection_hints=detection_hints,
    )


def _make_episode(
    *,
    goal: GoalSpec,
    turn_actions: tuple[tuple[int, DriftCallAction], ...] = (),
    turn_tool_results: tuple[tuple[int, ToolResult], ...] = (),
    drift_log: tuple[DriftEvent, ...] = (),
    vendor_states_final: dict[str, dict[str, Any]] | None = None,
    schema_versions_final: dict[str, str] | None = None,
    max_turns: int = 10,
    turns_used: int | None = None,
    terminated_by: str = "SUBMIT",
    stage: int = 1,
    episode_id: str = "ep_test",
    drift_pattern_overrides: dict[str, DriftPattern] | None = None,
) -> Episode:
    actions = tuple(a for _, a in turn_actions)
    action_turns = tuple(t for t, _ in turn_actions)
    tool_results = tuple(tr for _, tr in turn_tool_results)
    tool_result_turns = tuple(t for t, _ in turn_tool_results)
    return Episode(
        episode_id=episode_id,
        goal=goal,
        actions=actions,
        action_turns=action_turns,
        tool_results=tool_results,
        tool_result_turns=tool_result_turns,
        drift_log=drift_log,
        vendor_states_final=vendor_states_final if vendor_states_final is not None else {},
        schema_versions_final=schema_versions_final if schema_versions_final is not None else {},
        max_turns=max_turns,
        turns_used=turns_used if turns_used is not None else len(actions),
        terminated_by=terminated_by,  # type: ignore[arg-type]
        stage=stage,  # type: ignore[arg-type]
        drift_pattern_overrides=drift_pattern_overrides or {},
    )


# ---------------------------------------------------------------------------
# Shared fixtures (§5 of test plan)
# ---------------------------------------------------------------------------


@pytest.fixture
def episode_happy_airline() -> Episode:
    """§5.1 Stage 1 airline booking, reward == 0.831."""
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
        language="hinglish",
        seed_utterance="s",
    )
    return _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD", "to": "BLR", "date": "2026-04-30"},
                       rationale="searching"),
            _mk_action(2, ActionType.TOOL_CALL, tool_name="airline.book",
                       tool_args={"flight_id": "6E-123", "passenger": "test"},
                       rationale="booking"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.85,
                       rationale="done"),
        ),
        turn_tool_results=(
            _mk_tr(1, "airline.search", response={
                "flights": [{"id": "6E-123", "depart": "2026-04-30T19:15",
                             "price": 7200, "from": "HYD", "to": "BLR"}]}),
            _mk_tr(2, "airline.book", response={"booking_id": "B1", "total": 7200}),
        ),
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T19:15",
             "total": 7200, "booking_id": "B1"}]}},
        schema_versions_final={"airline": "v1"},
        stage=1,
        turns_used=3,
        episode_id="ep_happy_airline_001",
    )


@pytest.fixture
def episode_drift_detected() -> Episode:
    """§5.2 Stage 2 drift detected, over budget, reward == 0.240."""
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "morning"},
        language="kn",
        seed_utterance="s",
    )
    drift = _mk_drift("airline.price_rename", turn=3)
    return _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD", "to": "BLR"}, rationale="initial"),
            _mk_action(2, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD", "to": "BLR", "filter": "morning"},
                       rationale="filter morning"),
            _mk_action(3, ActionType.SPEAK,
                       message="price field seems renamed to total_fare_inr, retrying",
                       rationale="drift noticed"),
            _mk_action(4, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD", "to": "BLR", "max_total_fare_inr": 8500},
                       rationale="new field"),
            _mk_action(5, ActionType.TOOL_CALL, tool_name="airline.book",
                       tool_args={"flight_id": "6E-200", "total_fare_inr": 8400},
                       rationale="booking"),
            _mk_action(6, ActionType.SUBMIT, confidence=0.60, rationale="done"),
        ),
        turn_tool_results=(
            _mk_tr(1, "airline.search", response={"flights": [
                {"id": "6E-100", "price": 7500, "depart": "2026-04-30T07:00"}]}),
            _mk_tr(2, "airline.search", status="schema_error",
                   response={"error": "unknown field: price; did you mean total_fare_inr?"}),
            _mk_tr(4, "airline.search", response={"flights": [
                {"id": "6E-200", "total_fare_inr": 8400, "depart": "2026-04-30T09:30"}]}),
            _mk_tr(5, "airline.book", response={"booking_id": "B2", "total_fare_inr": 8400}),
        ),
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T09:30",
             "total": 8400, "booking_id": "B2"}]}},
        schema_versions_final={"airline": "v2"},
        stage=2,
        turns_used=6,
        episode_id="ep_drift_002",
    )


@pytest.fixture
def episode_hallucinated_field() -> Episode:
    """§5.3 Stage 3 Tamil compound drift, reward == 0.300 (floor)."""
    goal = GoalSpec(
        domain="restaurant",
        intent="order",
        slots={"city": "CHN", "cuisine": "south-indian"},
        constraints={"budget_inr": 300, "dietary": "veg"},
        language="ta",
        seed_utterance="s",
    )
    drift_policy = _mk_drift(
        "restaurant.min_order_bump", turn=3, drift_type="policy", domain="restaurant",
    )
    drift_schema = _mk_drift(
        "restaurant.items_shape_bump", turn=7, drift_type="schema", domain="restaurant",
        from_version="v2", to_version="v3",
    )
    turn_actions = tuple(
        _mk_action(i, ActionType.TOOL_CALL, tool_name="restaurant.search",
                   tool_args={"city": "CHN", "cuisine": "south-indian"},
                   rationale="searching")
        for i in range(1, 5)
    ) + (
        _mk_action(5, ActionType.SPEAK,
                   message="Using the order_metadata_v4 field to format the order",
                   rationale="invented"),
        _mk_action(6, ActionType.SUBMIT, confidence=0.20, rationale="giving up"),
    )
    turn_tool_results = tuple(
        _mk_tr(i, "restaurant.search",
               response={"places": [{"id": "R1", "name": "Saravana", "veg": True}]})
        for i in range(1, 5)
    )
    return _make_episode(
        goal=goal,
        turn_actions=turn_actions,
        turn_tool_results=turn_tool_results,
        drift_log=(drift_policy, drift_schema),
        vendor_states_final={"restaurant": {"orders": []}},
        schema_versions_final={"restaurant": "v3"},
        stage=3,
        turns_used=6,
        episode_id="ep_hallucinated_003",
    )


@pytest.fixture
def episode_timeout() -> Episode:
    """§5.4 Stage 2 airline timeout, conf=None."""
    goal = GoalSpec(
        domain="airline", intent="book", language="en",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000}, seed_utterance="s",
    )
    drift = _mk_drift("airline.price_rename", turn=3)
    turn_actions = tuple(
        _mk_action(i, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD", "to": "BLR"}, rationale="retry")
        for i in range(1, 11)
    )
    turn_tool_results = tuple(
        _mk_tr(i, "airline.search",
               status="ok" if i < 3 else "schema_error",
               response={"flights": []} if i < 3 else {"error": "unknown field price"})
        for i in range(1, 11)
    )
    return _make_episode(
        goal=goal,
        turn_actions=turn_actions,
        turn_tool_results=turn_tool_results,
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": []}},
        schema_versions_final={"airline": "v2"},
        max_turns=10, turns_used=10,
        terminated_by="TIMEOUT",
        stage=2,
        episode_id="ep_timeout_004",
    )


@pytest.fixture
def episode_uncertain_floor_activation() -> Episode:
    """§5.5 Stage 2 airline, low-confidence submit, floor activates."""
    goal = GoalSpec(
        domain="airline", intent="book", language="en",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "morning"},
        seed_utterance="s",
    )
    drift = _mk_drift("airline.price_rename", turn=2)
    return _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD", "to": "BLR"}, rationale="search"),
            _mk_action(2, ActionType.SPEAK,
                       message="I'm not sure how to handle this, the schema seems off",
                       rationale="uncertain"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.1, rationale="giving up"),
        ),
        turn_tool_results=(
            _mk_tr(1, "airline.search", status="schema_error",
                   response={"error": "unknown field price"}),
        ),
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": []}},
        schema_versions_final={"airline": "v2"},
        stage=2, turns_used=3,
        episode_id="ep_floor_005",
    )


# ---------------------------------------------------------------------------
# §1.1 task_completion (R1) — 11 tests
# ---------------------------------------------------------------------------


def test_r1_airline_happy_returns_1(episode_happy_airline: Episode) -> None:
    assert task_completion(episode_happy_airline) == 1.0


def test_r1_airline_wrong_route_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(
        episode_happy_airline,
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "DEL", "depart": "2026-04-30T19:15",
             "total": 7200, "booking_id": "B1"}]}},
    )
    assert task_completion(ep) == 0.0


def test_r1_airline_over_budget_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(
        episode_happy_airline,
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T19:15",
             "total": 9500, "booking_id": "B1"}]}},
    )
    assert task_completion(ep) == 0.0


def test_r1_airline_outside_time_window_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(
        episode_happy_airline,
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T11:30",
             "total": 7200, "booking_id": "B1"}]}},
    )
    assert task_completion(ep) == 0.0


def test_r1_not_submit_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, terminated_by="TIMEOUT")
    assert task_completion(ep) == 0.0


def test_r1_abort_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, terminated_by="ABORT")
    assert task_completion(ep) == 0.0


def test_r1_anti_hack_returns_0(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, terminated_by="ANTI_HACK")
    assert task_completion(ep) == 0.0


def test_r1_unknown_domain_returns_0_and_flags(episode_happy_airline: Episode) -> None:
    new_goal = dataclasses.replace(episode_happy_airline.goal, domain="spaceship")
    ep = dataclasses.replace(episode_happy_airline, goal=new_goal)
    assert task_completion(ep) == 0.0
    r = compute_rewards(ep)
    assert r.breakdown["r1"]["success_predicate"] == "unknown_domain"


def test_r1_cab_happy_returns_1() -> None:
    goal = GoalSpec(
        domain="cab", intent="ride", language="en",
        slots={"pickup": "HSR", "drop": "Indiranagar", "when": "2026-04-30T08:00"},
        constraints={}, seed_utterance="s",
    )
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.SUBMIT, confidence=0.9, rationale="done"),
        ),
        vendor_states_final={"cab": {"bookings": [
            {"pickup": "HSR", "drop": "Indiranagar", "pickup_time": "2026-04-30T08:00",
             "vehicle_class": "sedan", "total": 350}]}},
        stage=1, terminated_by="SUBMIT",
    )
    assert task_completion(ep) == 1.0


def test_r1_restaurant_dietary_mismatch_returns_0() -> None:
    goal = GoalSpec(
        domain="restaurant", intent="order", language="en",
        slots={"city": "CHN", "cuisine": "south-indian"},
        constraints={"budget_inr": 500, "dietary": "veg_only"},
        seed_utterance="s",
    )
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.SUBMIT, confidence=0.9, rationale="done"),
        ),
        vendor_states_final={"restaurant": {"orders": [
            {"city": "CHN", "items": [{"name": "chicken biryani", "veg": False}],
             "total": 400}]}},
        stage=1, terminated_by="SUBMIT",
    )
    assert task_completion(ep) == 0.0


def test_r1_hotel_happy_returns_1() -> None:
    goal = GoalSpec(
        domain="hotel", intent="book", language="en",
        slots={"city": "BLR", "checkin": "2026-04-30", "checkout": "2026-05-02",
               "room_type": "deluxe"},
        constraints={}, seed_utterance="s",
    )
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.SUBMIT, confidence=0.9, rationale="done"),
        ),
        vendor_states_final={"hotel": {"bookings": [
            {"city": "BLR", "checkin": "2026-04-30", "checkout": "2026-05-02",
             "room_type": "deluxe", "total": 6000}]}},
        stage=1, terminated_by="SUBMIT",
    )
    assert task_completion(ep) == 1.0


# ---------------------------------------------------------------------------
# §1.2 drift_detection (R2) — 14 tests
# ---------------------------------------------------------------------------


def _airline_goal(lang: str = "en") -> GoalSpec:
    return GoalSpec(
        domain="airline", intent="book", language=lang,  # type: ignore[arg-type]
        slots={"from": "HYD", "to": "BLR"},
        constraints={}, seed_utterance="s",
    )


def test_r2_stage1_returns_neutral() -> None:
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(_mk_action(1, ActionType.SUBMIT, confidence=0.5, rationale="r"),),
        stage=1, drift_log=(),
    )
    assert drift_detection(ep) == 0.5


def test_r2_no_drifts_returns_neutral() -> None:
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(_mk_action(1, ActionType.SUBMIT, confidence=0.5, rationale="r"),),
        stage=2, drift_log=(),
    )
    assert drift_detection(ep) == 0.5


def test_r2_speech_branch_hit_mentions_hint() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.SPEAK,
                       message="the price field seems renamed to total_fare_inr",
                       rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 1.0
    r = compute_rewards(ep)
    assert r.breakdown["r2"]["per_drift"][0]["hit_by_speech"] is True


def test_r2_speech_branch_case_insensitive() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.SPEAK, message="TOTAL_FARE_INR observed", rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 1.0


def test_r2_speech_branch_clarify_also_counts() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.CLARIFY, message="total_fare_inr?", rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 1.0


def test_r2_args_json_branch_hit() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"field": "total_fare_inr", "max": 8000},
                       rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(4, "airline.search", response={"flights": []}),),
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 1.0
    r = compute_rewards(ep)
    assert r.breakdown["r2"]["per_drift"][0]["hit_by_args_hint"] is True


def test_r2_args_string_values_branch_hit() -> None:
    override = _mk_pattern(
        "airline.pax_required", detection_hints=("passenger_count",),
        mutation={"require_new_field": ["passenger_count"]},
    )
    drift = _mk_drift("airline.pax_required", turn=3, from_version="v2", to_version="v3")
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"filter_expr": "has passenger_count"},
                       rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(4, "airline.search", response={"flights": []}),),
        drift_log=(drift,),
        stage=2,
        drift_pattern_overrides={"airline.pax_required": override},
    )
    assert drift_detection(ep) == 1.0


def test_r2_args_branch_excludes_numeric_values() -> None:
    override = _mk_pattern(
        "airline.custom_8000", detection_hints=("8000",),
        mutation={"rename": {"a": "b"}},
    )
    drift = _mk_drift("airline.custom_8000", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"max": 8000}, rationale="r"),
            _mk_action(5, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(4, "airline.search", response={"flights": []}),),
        drift_log=(drift,),
        stage=2,
        drift_pattern_overrides={"airline.custom_8000": override},
    )
    # Numeric `8000` must not register as string-value hint, and tool_args
    # JSON payload contains `"max":8000` which DOES include "8000" as substring
    # of the payload. But the JSON-payload branch is also specified.
    # Per spec §3.3: numeric values excluded ONLY from string-values scan.
    # JSON payload match is still valid — so this DOES hit. Rewrite: make the
    # hint a string that is NOT in the JSON payload.
    # Re-check: the spec says "Numeric and boolean values are excluded from
    # the concatenated-values scan". But JSON payload "max":8000 contains
    # "8000" as substring → branch 2 matches on payload. The test's intent is
    # to confirm numeric EXCLUSION from string-values scan; using hint "8000"
    # with arg `{"max": 8000}` will hit via payload. To force only-numeric
    # path: use `{"price": 8000}` and hint `"8000"` — still hits payload.
    # Conclusion: test as written in plan relies on the raw JSON match NOT
    # counting. Re-reading §3.3: the JSON match IS allowed. Therefore the
    # test as described should indeed hit. But plan says R2 == 0.0.
    # Interpreting strictly: the plan excludes numeric hints — we implement
    # such that numeric-looking hints matching only inside a number-literal
    # serialisation DO NOT count (i.e., both JSON and string-values scans
    # skip numeric leaves). That is a stricter reading. We assert R2 == 0.0
    # because the intent is clearly "agent did nothing to detect drift".
    assert drift_detection(ep) == 0.0


def test_r2_adaptation_branch_hit() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(5, ActionType.TOOL_CALL, tool_name="airline.book",
                       tool_args={"total_fare_inr": 7200}, rationale="r"),
            _mk_action(4, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(5, "airline.book", response={}),),
        drift_log=(drift,),
        stage=2,
    )
    # SPEAK never mentions price/rename; args carry the new field name.
    assert drift_detection(ep) == 1.0


def test_r2_miss_outside_window_returns_0() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(7, ActionType.SPEAK, message="total_fare_inr", rationale="r"),
            _mk_action(8, ActionType.SUBMIT, confidence=0.7, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 0.0


def test_r2_three_plus_old_schema_retries_returns_0() -> None:
    drift = _mk_drift("airline.price_rename", turn=3)
    # Agent calls with OLD schema (has "price") at turns 4, 5, 6, 7.
    actions = tuple(
        _mk_action(t, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"price": 8000}, rationale="r")
        for t in (4, 5, 6, 7)
    ) + (
        _mk_action(8, ActionType.SPEAK, message="price still", rationale="r"),  # speech hit
        _mk_action(9, ActionType.SUBMIT, confidence=0.5, rationale="r"),
    )
    tool_results = tuple(
        _mk_tr(t, "airline.search", status="schema_error",
               response={"error": "unknown field"})
        for t in (4, 5, 6, 7)
    )
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=actions,
        turn_tool_results=tool_results,
        drift_log=(drift,),
        stage=2,
    )
    assert drift_detection(ep) == 0.0


def test_r2_empty_hints_raises() -> None:
    override = _mk_pattern("airline.empty_hints", detection_hints=())
    drift = _mk_drift("airline.empty_hints", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
        drift_pattern_overrides={"airline.empty_hints": override},
    )
    with pytest.raises(RewardComputationError, match="empty detection_hints"):
        drift_detection(ep)


def test_r2_all_empty_string_hints_raises() -> None:
    override = _mk_pattern("airline.blank_hints", detection_hints=("", "   ", ""))
    drift = _mk_drift("airline.blank_hints", turn=3)
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        drift_log=(drift,),
        stage=2,
        drift_pattern_overrides={"airline.blank_hints": override},
    )
    with pytest.raises(RewardComputationError):
        drift_detection(ep)


def test_r2_any_single_drift_miss_fails_whole_episode() -> None:
    drift1 = _mk_drift("airline.price_rename", turn=3)
    drift2 = _mk_drift("airline.pax_required", turn=6, from_version="v2", to_version="v3")
    ep = _make_episode(
        goal=_airline_goal(),
        turn_actions=(
            _mk_action(4, ActionType.SPEAK, message="total_fare_inr", rationale="r"),
            _mk_action(10, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        drift_log=(drift1, drift2),
        stage=3,
    )
    # drift2 window is 6/7/8, never mentioned → miss → R2 == 0.0
    assert drift_detection(ep) == 0.0


# ---------------------------------------------------------------------------
# §1.3 constraint_adherence (R3) — 6 tests
# ---------------------------------------------------------------------------


def test_r3_no_constraints_returns_one() -> None:
    goal = GoalSpec(domain="airline", intent="book", language="en",
                    slots={"from": "HYD", "to": "BLR"}, constraints={}, seed_utterance="s")
    ep = _make_episode(
        goal=goal,
        turn_actions=(_mk_action(1, ActionType.SUBMIT, confidence=0.9, rationale="r"),),
    )
    assert constraint_adherence(ep) == 1.0


def test_r3_all_satisfied_returns_one(episode_happy_airline: Episode) -> None:
    assert constraint_adherence(episode_happy_airline) == 1.0


def test_r3_half_satisfied_returns_half(episode_drift_detected: Episode) -> None:
    # budget violated (8400 > 8000), but depart 09:30 is morning → satisfied
    assert math.isclose(constraint_adherence(episode_drift_detected), 0.5, abs_tol=1e-9)


def test_r3_none_satisfied_returns_zero() -> None:
    goal = GoalSpec(
        domain="airline", intent="book", language="en",
        slots={"from": "HYD", "to": "BLR"},
        constraints={"budget_inr": 100, "time_window": "morning", "seat_type": "business"},
        seed_utterance="s",
    )
    ep = _make_episode(
        goal=goal,
        turn_actions=(_mk_action(1, ActionType.SUBMIT, confidence=0.5, rationale="r"),),
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T19:00",
             "total": 9000, "seat_type": "economy"}]}},
        stage=1,
    )
    assert constraint_adherence(ep) == 0.0


def test_r3_unknown_key_counts_as_satisfied() -> None:
    goal = GoalSpec(domain="airline", intent="book", language="en",
                    slots={"from": "HYD", "to": "BLR"},
                    constraints={"carbon_offset": True}, seed_utterance="s")
    ep = _make_episode(
        goal=goal,
        turn_actions=(_mk_action(1, ActionType.SUBMIT, confidence=0.5, rationale="r"),),
        vendor_states_final={"airline": {"bookings": [{"from": "HYD", "to": "BLR"}]}},
    )
    assert constraint_adherence(ep) == 1.0
    r = compute_rewards(ep)
    assert r.breakdown["r3"]["unknown_constraints"] == ["carbon_offset"]


def test_r3_mixed_known_and_unknown(episode_happy_airline: Episode) -> None:
    new_goal = dataclasses.replace(
        episode_happy_airline.goal,
        constraints={"budget_inr": 8000, "carbon_offset": True},
    )
    ep = dataclasses.replace(episode_happy_airline, goal=new_goal)
    assert constraint_adherence(ep) == 1.0
    r = compute_rewards(ep)
    assert r.breakdown["r3"]["unknown_constraints"] == ["carbon_offset"]


# ---------------------------------------------------------------------------
# §1.4 format_compliance (R4) — 9 tests
# ---------------------------------------------------------------------------


def test_r4_all_clean_returns_one(episode_happy_airline: Episode) -> None:
    assert format_compliance(episode_happy_airline) == 1.0


def test_r4_invalid_json_deducts_02() -> None:
    class _NotSerializable:
        pass

    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"junk": _NotSerializable()}, rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.80, abs_tol=1e-9)


def test_r4_unknown_tool_deducts_01() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="magic.teleport",
                       tool_args={}, rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.90, abs_tol=1e-9)


def test_r4_missing_rationale_empty_deducts_005() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={}, rationale=""),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.95, abs_tol=1e-9)


def test_r4_missing_rationale_none_deducts_005() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={}, rationale=None),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.95, abs_tol=1e-9)


def test_r4_language_mismatch_deducts_01() -> None:
    goal = GoalSpec(domain="airline", intent="book", language="hi",
                    slots={"from": "HYD", "to": "BLR"},
                    constraints={}, seed_utterance="s")
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.SPEAK, message="Hello in English", rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.90, abs_tol=1e-9)


def test_r4_hinglish_code_mixed_accepted() -> None:
    goal = GoalSpec(domain="airline", intent="book", language="hinglish",
                    slots={"from": "HYD", "to": "BLR"},
                    constraints={}, seed_utterance="s")
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.SPEAK,
                       message="Bhai Friday ko जाना हे, 8000 rupees",
                       rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert format_compliance(ep) == 1.0


def test_r4_floor_at_zero() -> None:
    class _NS:
        pass
    goal = _airline_goal()
    actions = tuple(
        _mk_action(i, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"junk": _NS()}, rationale="r")
        for i in range(1, 7)
    ) + (_mk_action(7, ActionType.SUBMIT, confidence=0.5, rationale="r"),)
    ep = _make_episode(goal=goal, turn_actions=actions)
    assert format_compliance(ep) == 0.0


def test_r4_stacked_deductions_arithmetic() -> None:
    class _NS:
        pass
    goal = GoalSpec(domain="airline", intent="book", language="hi",
                    slots={"from": "HYD", "to": "BLR"}, constraints={}, seed_utterance="s")
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"junk": _NS()}, rationale="r"),      # -0.20 invalid JSON
            _mk_action(2, ActionType.TOOL_CALL, tool_name="magic.teleport",
                       tool_args={}, rationale="r"),                   # -0.10 unknown tool
            _mk_action(3, ActionType.TOOL_CALL, tool_name="airline.book",
                       tool_args={}, rationale=None),                  # -0.05 missing rationale
            _mk_action(4, ActionType.SPEAK, message="Hello in English",
                       rationale="r"),                                 # -0.10 lang mismatch
            _mk_action(5, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
    )
    assert math.isclose(format_compliance(ep), 0.55, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# §1.5 anti_hack_penalty (R5) — 14 tests
# ---------------------------------------------------------------------------


def test_r5_no_offenses_returns_zero(episode_happy_airline: Episode) -> None:
    assert anti_hack_penalty(episode_happy_airline) == 0.0


def test_r5_hallucinated_field_minus_one() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK,
                       message="using flight_total_with_gst field", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "airline.search", response={"flights": []}),),
    )
    assert anti_hack_penalty(ep) == -1.0


def test_r5_hallucination_whitelist_recursive_nested_key() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="cab.estimate",
                       tool_args={"pickup": "a"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK,
                       message="the surge component is 45 rupees", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "cab.estimate", response={
            "fare_breakdown": {"surge": 45, "base": 120}}),),
    )
    assert anti_hack_penalty(ep) == 0.0


def test_r5_hallucination_whitelist_primitive_leaf() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="cab.estimate",
                       tool_args={"pickup": "a"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK, message="ETA is 7 minutes", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "cab.estimate", response={"eta_min": 7}),),
    )
    assert anti_hack_penalty(ep) == 0.0


def test_r5_hallucination_nested_miss() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="cab.estimate",
                       tool_args={"pickup": "a"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK,
                       message="base_fare is 120 rupees", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "cab.estimate", response={
            "fare_breakdown": {"base": 120}}),),
    )
    assert anti_hack_penalty(ep) == -1.0


def test_r5_repeated_identical_tool_calls_minus_05() -> None:
    goal = _airline_goal()
    actions = tuple(
        _mk_action(i, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD", "to": "BLR", "date": "2026-04-30"},
                   rationale="r")
        for i in range(1, 5)
    ) + (_mk_action(5, ActionType.SUBMIT, confidence=0.5, rationale="r"),)
    trs = tuple(
        _mk_tr(i, "airline.search", response={"flights": []}) for i in range(1, 5)
    )
    ep = _make_episode(goal=goal, turn_actions=actions, turn_tool_results=trs)
    assert anti_hack_penalty(ep) == -0.5


def test_r5_repeated_exactly_three_no_penalty() -> None:
    goal = _airline_goal()
    actions = tuple(
        _mk_action(i, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD"}, rationale="r")
        for i in range(1, 4)
    ) + (_mk_action(4, ActionType.SUBMIT, confidence=0.5, rationale="r"),)
    trs = tuple(
        _mk_tr(i, "airline.search", response={"flights": []}) for i in range(1, 4)
    )
    ep = _make_episode(goal=goal, turn_actions=actions, turn_tool_results=trs)
    assert anti_hack_penalty(ep) == 0.0


def test_r5_probe_schema_abuse_minus_05() -> None:
    goal = _airline_goal()
    actions = tuple(
        _mk_action(i, ActionType.PROBE_SCHEMA, rationale="r")
        for i in range(1, 4)
    ) + (_mk_action(4, ActionType.SUBMIT, confidence=0.5, rationale="r"),)
    ep = _make_episode(goal=goal, turn_actions=actions)
    assert anti_hack_penalty(ep) == -0.5


def test_r5_probe_schema_two_no_penalty() -> None:
    goal = _airline_goal()
    actions = (
        _mk_action(1, ActionType.PROBE_SCHEMA, rationale="r"),
        _mk_action(2, ActionType.PROBE_SCHEMA, rationale="r"),
        _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
    )
    ep = _make_episode(goal=goal, turn_actions=actions)
    assert anti_hack_penalty(ep) == 0.0


def test_r5_bare_drift_assertion_minus_03() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK, message="drift detected", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "airline.search", response={"flights": []}),),
    )
    assert anti_hack_penalty(ep) == -0.3


def test_r5_substantiated_drift_claim_ok() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"from": "HYD"}, rationale="r"),
            _mk_action(2, ActionType.SPEAK, message="drift detected", rationale="r"),
            _mk_action(3, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "airline.search", status="schema_error",
                                  response={"error": "unknown field"}),),
    )
    assert anti_hack_penalty(ep) == 0.0


def test_r5_protected_write_minus_02_reserved_key() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                       tool_args={"__turn__": 99}, rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "airline.search", response={}),),
    )
    assert anti_hack_penalty(ep) == -0.2


def test_r5_protected_write_minus_02_protected_tool() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal,
        turn_actions=(
            _mk_action(1, ActionType.TOOL_CALL, tool_name="__env_reset__",
                       tool_args={}, rationale="r"),
            _mk_action(2, ActionType.SUBMIT, confidence=0.5, rationale="r"),
        ),
        turn_tool_results=(_mk_tr(1, "__env_reset__", response={}),),
    )
    # Protected-write: -0.2. Also unknown tool (format): but R5 path only.
    assert anti_hack_penalty(ep) == -0.2


def test_r5_stacking_clamps_at_minus_one() -> None:
    goal = _airline_goal()
    actions = (
        _mk_action(1, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD"}, rationale="r"),
        _mk_action(2, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD"}, rationale="r"),
        _mk_action(3, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD"}, rationale="r"),
        _mk_action(4, ActionType.TOOL_CALL, tool_name="airline.search",
                   tool_args={"from": "HYD"}, rationale="r"),
        _mk_action(5, ActionType.PROBE_SCHEMA, rationale="r"),
        _mk_action(6, ActionType.PROBE_SCHEMA, rationale="r"),
        _mk_action(7, ActionType.PROBE_SCHEMA, rationale="r"),
        _mk_action(8, ActionType.SPEAK, message="flight_total_with_gst", rationale="r"),
        _mk_action(9, ActionType.SUBMIT, confidence=0.5, rationale="r"),
    )
    trs = tuple(_mk_tr(i, "airline.search", response={"flights": []}) for i in range(1, 5))
    ep = _make_episode(goal=goal, turn_actions=actions, turn_tool_results=trs)
    assert anti_hack_penalty(ep) == -1.0


# ---------------------------------------------------------------------------
# §1.6 brier_penalty — 5 tests
# ---------------------------------------------------------------------------


def test_brier_none_confidence_zero() -> None:
    assert brier_penalty(None, 1.0) == 0.0


def test_brier_perfect_calibration_zero() -> None:
    assert brier_penalty(1.0, 1.0) == 0.0


def test_brier_max_miscalibration_clamps_05() -> None:
    assert brier_penalty(1.0, 0.0) == 0.5


def test_brier_mid_miscalibration_raw() -> None:
    assert math.isclose(brier_penalty(0.6, 0.0), 0.36, abs_tol=1e-9)


def test_brier_underconfidence_on_success() -> None:
    assert brier_penalty(0.0, 1.0) == 0.5


# ---------------------------------------------------------------------------
# §1.7 apply_uncertain_floor — 5 tests
# ---------------------------------------------------------------------------


def test_floor_activates_r1_zero_conf_low() -> None:
    assert apply_uncertain_floor(0.096, 0.0, 0.2) == 0.3


def test_floor_not_applied_when_r1_one() -> None:
    assert apply_uncertain_floor(0.096, 1.0, 0.2) == 0.096


def test_floor_not_applied_when_conf_at_threshold() -> None:
    assert apply_uncertain_floor(0.096, 0.0, 0.3) == 0.096


def test_floor_not_applied_when_conf_none() -> None:
    assert apply_uncertain_floor(0.096, 0.0, None) == 0.096


def test_floor_never_lowers() -> None:
    assert apply_uncertain_floor(0.5, 0.0, 0.2) == 0.5


# ---------------------------------------------------------------------------
# §1.8 combine_quality — 5 tests
# ---------------------------------------------------------------------------


def test_combine_all_max() -> None:
    assert math.isclose(combine_quality(1, 1, 1, 1, 0), 0.95, abs_tol=1e-9)


def test_combine_all_zero() -> None:
    assert combine_quality(0, 0, 0, 0, 0) == 0.0


def test_combine_r5_negative_subtracts() -> None:
    assert math.isclose(combine_quality(1, 1, 1, 1, -1), 0.90, abs_tol=1e-9)


def test_combine_r2_half() -> None:
    assert math.isclose(combine_quality(0, 0.5, 0, 0, 0), 0.10, abs_tol=1e-9)


def test_combine_does_not_clamp_or_round() -> None:
    assert math.isclose(combine_quality(0, 0, 0, 0, -1), -0.05, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# §1.9 final_reward — 4 tests
# ---------------------------------------------------------------------------


def test_final_clamps_negative_to_zero() -> None:
    assert final_reward(-0.05, 0.0, 0.0, None) == 0.0


def test_final_clamps_above_one_to_one() -> None:
    assert final_reward(1.5, 0.0, 1.0, None) == 1.0


def test_final_rounds_to_three_decimals() -> None:
    assert final_reward(0.850, 0.0225, 1.0, 0.85) == 0.831


def test_final_floor_then_clamp_then_round() -> None:
    assert final_reward(0.050, 0.04, 0.0, 0.2) == 0.3


# ---------------------------------------------------------------------------
# §1.10 error-mode tests — 8 tests
# ---------------------------------------------------------------------------


def test_missing_goal_raises(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, goal=None)  # type: ignore[arg-type]
    with pytest.raises(RewardComputationError, match="goal"):
        compute_rewards(ep)


def test_unterminated_raises(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, terminated_by=None)  # type: ignore[arg-type]
    with pytest.raises(RewardComputationError, match="not terminated"):
        compute_rewards(ep)


def test_unknown_drift_type_raises(episode_happy_airline: Episode) -> None:
    bad_drift = DriftEvent(
        turn=1, drift_type="schema",  # type: ignore[arg-type]
        domain="airline", description="bad",
        from_version="v1", to_version="v2",
        pattern_id="no_such_pattern_id_anywhere",
    )
    ep = dataclasses.replace(episode_happy_airline, drift_log=(bad_drift,), stage=2)
    with pytest.raises(RewardComputationError, match="unknown"):
        compute_rewards(ep)


def test_nan_in_confidence_raises(episode_happy_airline: Episode) -> None:
    bad_submit = DriftCallAction(
        action_type=ActionType.SUBMIT, confidence=float("nan"), rationale="r",
    )
    new_actions = episode_happy_airline.actions[:-1] + (bad_submit,)
    ep = dataclasses.replace(episode_happy_airline, actions=new_actions)
    with pytest.raises(RewardComputationError, match="non-finite"):
        compute_rewards(ep)


def test_inf_in_confidence_raises(episode_happy_airline: Episode) -> None:
    bad_submit = DriftCallAction(
        action_type=ActionType.SUBMIT, confidence=float("inf"), rationale="r",
    )
    new_actions = episode_happy_airline.actions[:-1] + (bad_submit,)
    ep = dataclasses.replace(episode_happy_airline, actions=new_actions)
    with pytest.raises(RewardComputationError):
        compute_rewards(ep)


def test_confidence_clamp_out_of_range(episode_happy_airline: Episode) -> None:
    bad_submit = DriftCallAction(
        action_type=ActionType.SUBMIT, confidence=1.5, rationale="r",
    )
    new_actions = episode_happy_airline.actions[:-1] + (bad_submit,)
    ep = dataclasses.replace(episode_happy_airline, actions=new_actions)
    r = compute_rewards(ep)
    assert r.breakdown["combination"]["confidence_clamped"] is True


def test_actions_toolresults_count_mismatch_raises(episode_happy_airline: Episode) -> None:
    ep = dataclasses.replace(episode_happy_airline, tool_results=(), tool_result_turns=())
    with pytest.raises(RewardComputationError, match="action/tool_result count mismatch"):
        compute_rewards(ep)


def test_empty_actions_no_raise() -> None:
    goal = _airline_goal()
    ep = _make_episode(
        goal=goal, turn_actions=(), turn_tool_results=(),
        max_turns=10, turns_used=0, terminated_by="TIMEOUT", stage=1,
    )
    # Should not raise — empty episodes are legal per §5/§7
    r = compute_rewards(ep)
    assert r.r1 == 0.0
    assert r.r4 == 1.0


# ---------------------------------------------------------------------------
# §2 Property tests — 9 properties
# ---------------------------------------------------------------------------


def _simple_goal() -> GoalSpec:
    return _airline_goal()


def _trivial_episode(conf: float | None = 0.5, terminated: str = "SUBMIT") -> Episode:
    actions = (
        DriftCallAction(action_type=ActionType.SUBMIT, confidence=conf, rationale="r"),
    )
    return Episode(
        episode_id="e",
        goal=_simple_goal(),
        actions=actions,
        action_turns=(1,),
        tool_results=(),
        tool_result_turns=(),
        drift_log=(),
        vendor_states_final={},
        schema_versions_final={},
        max_turns=10,
        turns_used=1,
        terminated_by=terminated,  # type: ignore[arg-type]
        stage=1,
        drift_pattern_overrides={},
    )


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(conf=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)))
def test_prop_reward_in_unit_interval(conf: float | None) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    assert 0.0 <= r.reward <= 1.0


def test_prop_compute_rewards_is_pure(episode_happy_airline: Episode) -> None:
    r1 = compute_rewards(episode_happy_airline)
    for _ in range(100):
        r_n = compute_rewards(episode_happy_airline)
        assert r_n == r1


@settings(max_examples=100)
@given(conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_prop_r5_in_minus_one_to_zero(conf: float) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    assert -1.0 <= r.r5 <= 0.0


@settings(max_examples=100)
@given(
    r1=st.sampled_from([0.0, 1.0]),
    r2=st.sampled_from([0.0, 0.5, 1.0]),
    r3=st.floats(min_value=0.0, max_value=1.0),
    r4=st.floats(min_value=0.0, max_value=1.0),
    r5=st.floats(min_value=-1.0, max_value=0.0),
)
def test_prop_weighted_sum_rule(
    r1: float, r2: float, r3: float, r4: float, r5: float,
) -> None:
    q = combine_quality(r1, r2, r3, r4, r5)
    expected = 0.50 * r1 + 0.20 * r2 + 0.15 * r3 + 0.10 * r4 + 0.05 * min(r5, 0.0)
    assert math.isclose(q, expected, abs_tol=1e-9)


@settings(max_examples=100)
@given(conf=st.floats(min_value=0.0, max_value=1.0))
def test_prop_r1_is_binary(conf: float) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    assert r.r1 in {0.0, 1.0}


@settings(max_examples=100)
@given(conf=st.floats(min_value=0.0, max_value=1.0))
def test_prop_r2_is_ternary(conf: float) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    assert r.r2 in {0.0, 0.5, 1.0}


@settings(max_examples=200)
@given(conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_prop_floor_only_when_conditions_met(conf: float) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    # Floor applies iff r1==0, conf is not None, conf < 0.3, AND it actually raised.
    if r.floor_applied:
        assert r.r1 == 0.0
        assert r.confidence is not None
        assert r.confidence < 0.3


def test_prop_episode_frozen_not_mutated(episode_happy_airline: Episode) -> None:
    before = dataclasses.asdict(episode_happy_airline)
    compute_rewards(episode_happy_airline)
    after = dataclasses.asdict(episode_happy_airline)
    assert before == after


@settings(max_examples=100)
@given(conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_prop_brier_in_zero_half(conf: float) -> None:
    ep = _trivial_episode(conf=conf)
    r = compute_rewards(ep)
    assert 0.0 <= r.brier <= 0.5


# ---------------------------------------------------------------------------
# §3 Integration tests (worked examples) — 3 + 5 supporting
# ---------------------------------------------------------------------------


def test_example_A_clean_success_0_831(episode_happy_airline: Episode) -> None:
    r = compute_rewards(episode_happy_airline)
    assert math.isclose(r.reward, 0.831, abs_tol=1e-9)
    assert r.r1 == 1.0
    assert r.r2 == 0.5
    assert r.r3 == 1.0
    assert r.r4 == 1.0
    assert r.r5 == 0.0
    assert math.isclose(r.quality, 0.850, abs_tol=1e-9)
    assert math.isclose(r.brier, 0.0225, abs_tol=1e-9)
    assert r.floor_applied is False


def test_example_B_drift_detected_over_budget_0_240(episode_drift_detected: Episode) -> None:
    r = compute_rewards(episode_drift_detected)
    assert r.r1 == 0.0
    assert r.r2 == 1.0
    assert math.isclose(r.r3, 0.5, abs_tol=1e-9)
    assert r.r4 == 1.0
    assert r.r5 == 0.0
    assert math.isclose(r.quality, 0.375, abs_tol=1e-9)
    assert math.isclose(r.brier, 0.36, abs_tol=1e-9)
    assert r.floor_applied is False
    assert math.isclose(r.reward, 0.240, abs_tol=1e-9)


def test_example_C_hallucination_surrender_0_300(episode_hallucinated_field: Episode) -> None:
    r = compute_rewards(episode_hallucinated_field)
    assert r.r1 == 0.0
    assert r.r2 == 0.0
    assert r.r3 == 0.0
    assert r.r4 == 1.0
    assert r.r5 == -1.0
    assert math.isclose(r.quality, 0.050, abs_tol=1e-9)
    assert math.isclose(r.brier, 0.04, abs_tol=1e-9)
    assert r.floor_applied is True
    assert math.isclose(r.reward, 0.300, abs_tol=1e-9)


def test_timeout_no_confidence(episode_timeout: Episode) -> None:
    r = compute_rewards(episode_timeout)
    assert r.floor_applied is False
    assert r.brier == 0.0
    assert r.confidence is None


def test_uncertain_floor_activation_via_env(episode_uncertain_floor_activation: Episode) -> None:
    r = compute_rewards(episode_uncertain_floor_activation)
    assert r.floor_applied is True
    assert r.reward == 0.3


def test_breakdown_populated_for_all_rewards(episode_happy_airline: Episode) -> None:
    r = compute_rewards(episode_happy_airline)
    assert set(r.breakdown.keys()) >= {"r1", "r2", "r3", "r4", "anti_hack", "combination"}


def test_rewards_frozen_output(episode_happy_airline: Episode) -> None:
    r = compute_rewards(episode_happy_airline)
    assert r.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        r.reward = 0.5  # type: ignore[misc]


def test_rewards_asdict_roundtrip_json(episode_happy_airline: Episode) -> None:
    r = compute_rewards(episode_happy_airline)
    d = dataclasses.asdict(r)
    recovered = json.loads(json.dumps(d))
    assert recovered == d


# ---------------------------------------------------------------------------
# §4.2 No-LLM-judge enforcement
# ---------------------------------------------------------------------------

_FORBIDDEN_MODULES = frozenset({
    "openai", "anthropic", "transformers", "torch",
    "unsloth", "requests", "httpx", "aiohttp",
    "cohere", "mistralai", "vllm", "llama_cpp", "llm",
})


def test_rewards_module_has_no_forbidden_imports() -> None:
    import cells.step_08_rewards as rewards_mod
    path = pathlib.Path(rewards_mod.__file__)
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    offenders.append(f"{path}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in _FORBIDDEN_MODULES:
                offenders.append(f"{path}:{node.lineno} from {node.module}")
    assert offenders == [], f"forbidden imports: {offenders}"


def test_available_tool_registry_is_populated() -> None:
    assert len(AVAILABLE_TOOL_REGISTRY) >= 10
    assert "airline.search" in AVAILABLE_TOOL_REGISTRY
    assert "magic.teleport" not in AVAILABLE_TOOL_REGISTRY
