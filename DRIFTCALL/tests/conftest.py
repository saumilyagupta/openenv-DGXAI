"""Shared pytest fixtures for DriftCall tests.

Per docs/tests/models_tests.md §5, these fixtures build spec-valid instances
with keyword overrides for variation. They are imported by every test module
that needs models.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftCallObservation,
    DriftCallState,
    DriftEvent,
    GoalSpec,
    ToolResult,
)


@pytest.fixture
def valid_goal_spec() -> GoalSpec:
    """Spec-valid Hinglish airline booking goal. Matches models.md §8.3."""
    return GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
        language="hinglish",
        seed_utterance="Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad",
    )


@pytest.fixture
def valid_drift_event_factory() -> Callable[..., DriftEvent]:
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


@pytest.fixture
def valid_drift_event(
    valid_drift_event_factory: Callable[..., DriftEvent],
) -> DriftEvent:
    """Plain drift event with defaults turn=3, domain='airline'."""
    return valid_drift_event_factory()


@pytest.fixture
def valid_tool_result_factory() -> Callable[..., ToolResult]:
    """Factory fixture for ToolResult. Matches models.md §8.2."""

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


@pytest.fixture
def valid_tool_result(
    valid_tool_result_factory: Callable[..., ToolResult],
) -> ToolResult:
    """Plain ok-status airline.search ToolResult."""
    return valid_tool_result_factory()


@pytest.fixture
def valid_tool_call_action() -> DriftCallAction:
    """TOOL_CALL action for airline.search. Matches models.md §8.1."""
    return DriftCallAction(
        action_type=ActionType.TOOL_CALL,
        tool_name="airline.search",
        tool_args={
            "from": "HYD",
            "to": "BLR",
            "date": "2026-04-25",
            "max_price_inr": 8000,
            "time_window": "evening",
        },
        rationale="User asked for cheapest evening flight under 8000",
    )


@pytest.fixture
def valid_submit_action() -> Callable[..., DriftCallAction]:
    """Factory: builds a SUBMIT action with given confidence."""

    def _build(confidence: float = 0.87) -> DriftCallAction:
        return DriftCallAction(
            action_type=ActionType.SUBMIT,
            confidence=confidence,
        )

    return _build


@pytest.fixture
def valid_observation_reset(valid_goal_spec: GoalSpec) -> DriftCallObservation:
    """Turn-0 observation matching models.md §8.3."""
    return DriftCallObservation(
        turn=0,
        goal=valid_goal_spec,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=(),
        drift_log=(),
        budget_remaining=12,
        available_tools=(
            "airline.search",
            "airline.book",
            "airline.cancel",
            "airline.get_booking",
            "payment.charge",
        ),
    )


@pytest.fixture
def valid_state_reset(valid_goal_spec: GoalSpec) -> DriftCallState:
    """Turn-0 DriftCallState with max_turns=12 and empty histories."""
    return DriftCallState(
        episode_id="ep_000123",
        goal=valid_goal_spec,
        vendor_states={
            "airline": {},
            "cab": {},
            "restaurant": {},
            "hotel": {},
            "payment": {},
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
        max_turns=12,
        actions=(),
        done=False,
    )
