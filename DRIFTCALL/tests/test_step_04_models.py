"""Tests for cells/step_04_models.py.

Implements docs/tests/models_tests.md §1 (40 unit tests) + §2 (6 property tests).
Fixtures live in tests/conftest.py (§5 of the test plan).
"""

from __future__ import annotations

import dataclasses
import json
import string
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftCallObservation,
    DriftCallState,
    DriftEvent,
    GoalSpec,
    ToolResult,
)

DriftTypeLiteral = Literal["schema", "policy", "tnc", "pricing", "auth"]
LanguageLiteral = Literal["hi", "ta", "kn", "en", "hinglish"]

# ---------------------------------------------------------------------------
# Hypothesis strategies (shared between properties). Inline rather than in a
# separate strategies.py to keep the test surface self-contained per briefing.
# ---------------------------------------------------------------------------

_LANGUAGES = ["hi", "ta", "kn", "en", "hinglish"]
_DOMAINS = ["airline", "cab", "restaurant", "hotel", "payment"]
_DRIFT_TYPES = ["schema", "policy", "tnc", "pricing", "auth"]
_STATUSES = ["ok", "schema_error", "policy_error", "auth_error", "timeout"]
_ACTION_TYPES = list(ActionType)

_languages = st.sampled_from(_LANGUAGES)
_domains = st.sampled_from(_DOMAINS)
_drift_types = st.sampled_from(_DRIFT_TYPES)
_statuses = st.sampled_from(_STATUSES)
_versions = st.from_regex(r"^v[1-9]\d?$", fullmatch=True)

_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-1_000_000, 1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(
        alphabet=st.characters(blacklist_categories=["Cs"]),
        max_size=32,
    ),
)
_json_key = st.text(alphabet=string.ascii_letters, min_size=1, max_size=8)
_json_dict = st.dictionaries(keys=_json_key, values=_json_scalar, max_size=4)


# ---------------------------------------------------------------------------
# §1.1 ActionType (Enum) — U1–U4
# ---------------------------------------------------------------------------


def test_action_type_members_exactly_six() -> None:
    """U1 — models.md §4.1: enum has exactly six canonical members."""
    assert set(ActionType) == {
        ActionType.TOOL_CALL,
        ActionType.SPEAK,
        ActionType.CLARIFY,
        ActionType.PROBE_SCHEMA,
        ActionType.SUBMIT,
        ActionType.ABORT,
    }
    assert len(ActionType) == 6


def test_action_type_is_str_subclass() -> None:
    """U2 — models.md §3.4, §4.1: ActionType is a str-mixed Enum."""
    assert isinstance(ActionType.TOOL_CALL, str)
    assert ActionType.TOOL_CALL.value == "tool_call"
    assert str(ActionType.TOOL_CALL) in ("ActionType.TOOL_CALL", "tool_call")
    assert json.dumps({"t": ActionType.SPEAK}) == '{"t": "speak"}'


def test_action_type_values_match_spec() -> None:
    """U3 — models.md §4.1: value strings are the lowercase names exactly."""
    assert ActionType.TOOL_CALL.value == "tool_call"
    assert ActionType.SPEAK.value == "speak"
    assert ActionType.CLARIFY.value == "clarify"
    assert ActionType.PROBE_SCHEMA.value == "probe_schema"
    assert ActionType.SUBMIT.value == "submit"
    assert ActionType.ABORT.value == "abort"


def test_action_type_is_hashable() -> None:
    """U4 — models.md §3.2 bullet 1: enum members are hashable."""
    assert isinstance(hash(ActionType.TOOL_CALL), int)
    assert {ActionType.SPEAK, ActionType.SPEAK} == {ActionType.SPEAK}


# ---------------------------------------------------------------------------
# §1.2 DriftCallAction — U5–U13
# ---------------------------------------------------------------------------


def test_driftcall_action_happy_tool_call(
    valid_tool_call_action: DriftCallAction,
) -> None:
    """U5 — models.md §4.2, §8.1: TOOL_CALL action populates the right fields."""
    action = valid_tool_call_action
    assert action.action_type is ActionType.TOOL_CALL
    assert action.tool_name == "airline.search"
    assert action.tool_args is not None
    assert action.tool_args["from"] == "HYD"
    assert action.tool_args["to"] == "BLR"
    assert action.rationale == "User asked for cheapest evening flight under 8000"
    assert action.confidence is None
    assert action.message is None


def test_driftcall_action_happy_submit(
    valid_submit_action: Callable[..., DriftCallAction],
) -> None:
    """U6 — models.md §4.2, §3.5 SUBMIT row."""
    action = valid_submit_action(0.87)
    assert action.action_type is ActionType.SUBMIT
    assert action.confidence == 0.87
    assert action.tool_name is None
    assert action.tool_args is None
    assert action.message is None


def test_driftcall_action_happy_speak() -> None:
    """U7 — models.md §7 edge 3: SPEAK action carries Unicode round-trip-safe."""
    original = "मुझे कल दिल्ली जाना है"
    action = DriftCallAction(
        action_type=ActionType.SPEAK,
        message=original,
    )
    assert action.message == original
    assert action.action_type is ActionType.SPEAK


_ACTION_FIELDS = [
    "action_type",
    "tool_name",
    "tool_args",
    "message",
    "confidence",
    "rationale",
]


@pytest.mark.parametrize("field_name", _ACTION_FIELDS)
def test_driftcall_action_frozen_mutation_raises(
    valid_tool_call_action: DriftCallAction, field_name: str
) -> None:
    """U8 — models.md §3.1, §5 row 1: every field is frozen."""
    with pytest.raises(FrozenInstanceError):
        setattr(valid_tool_call_action, field_name, "anything")


def test_driftcall_action_defaults_are_none() -> None:
    """U9 — models.md §2: every optional field defaults to None."""
    action = DriftCallAction(action_type=ActionType.ABORT)
    assert action.tool_name is None
    assert action.tool_args is None
    assert action.message is None
    assert action.confidence is None
    assert action.rationale is None


def test_driftcall_action_missing_required_raises() -> None:
    """U10 — models.md §5 row 3: action_type is required."""
    with pytest.raises(TypeError):
        cls: type = DriftCallAction
        cls()


@pytest.mark.parametrize(
    ("field_name", "alt_value"),
    [
        ("action_type", ActionType.SPEAK),
        ("tool_name", "airline.book"),
        ("tool_args", {"x": 1}),
        ("message", "hi"),
        ("confidence", 0.5),
        ("rationale", "because"),
    ],
)
def test_driftcall_action_equality_value_based(
    field_name: str, alt_value: Any
) -> None:
    """U11 — models.md §3.2: equality is value-based across all 6 fields."""
    base_kwargs: dict[str, Any] = {
        "action_type": ActionType.TOOL_CALL,
        "tool_name": "airline.search",
        "tool_args": {"a": 1},
        "message": "msg",
        "confidence": 0.1,
        "rationale": "r",
    }
    a1 = DriftCallAction(**base_kwargs)
    a2 = DriftCallAction(**base_kwargs)
    assert a1 == a2
    modified = {**base_kwargs, field_name: alt_value}
    a3 = DriftCallAction(**modified)
    assert a1 != a3


def test_driftcall_action_unhashable_due_to_dict(
    valid_tool_call_action: DriftCallAction,
) -> None:
    """U12 — models.md §3.2 bullet 2, §5 row 7."""
    with pytest.raises(TypeError):
        hash(valid_tool_call_action)


def test_driftcall_action_confidence_none_vs_zero() -> None:
    """U13 — models.md §4.2, §7 edge 6: None and 0.0 are distinguishable."""
    with_zero = DriftCallAction(action_type=ActionType.SUBMIT, confidence=0.0)
    with_none = DriftCallAction(action_type=ActionType.SUBMIT, confidence=None)
    assert with_zero != with_none
    assert with_zero.confidence == 0.0
    assert with_none.confidence is None


# ---------------------------------------------------------------------------
# §1.3 ToolResult — U14–U18
# ---------------------------------------------------------------------------


def test_tool_result_happy_ok(valid_tool_result: ToolResult) -> None:
    """U14 — models.md §4.3, §8.2."""
    assert valid_tool_result.status == "ok"
    assert valid_tool_result.tool_name == "airline.search"
    assert valid_tool_result.schema_version == "v1"
    assert valid_tool_result.latency_ms == 142
    assert isinstance(valid_tool_result.response["results"], list)


_TOOL_RESULT_FIELDS = ["tool_name", "status", "response", "schema_version", "latency_ms"]


@pytest.mark.parametrize("field_name", _TOOL_RESULT_FIELDS)
def test_tool_result_frozen_mutation_raises(
    valid_tool_result: ToolResult, field_name: str
) -> None:
    """U15 — models.md §3.1, §5 row 1."""
    with pytest.raises(FrozenInstanceError):
        setattr(valid_tool_result, field_name, "anything")


@pytest.mark.parametrize("status", _STATUSES)
def test_tool_result_accepts_all_five_statuses(
    status: str,
    valid_tool_result_factory: Callable[..., ToolResult],
) -> None:
    """U16 — models.md §4.3 status row: all five statuses construct cleanly."""
    result = valid_tool_result_factory(status=status)
    assert result.status == status


def test_tool_result_empty_response_on_non_ok(
    valid_tool_result_factory: Callable[..., ToolResult],
) -> None:
    """U17 — models.md §7 edge 7: empty response on non-ok is not rejected here."""
    result = valid_tool_result_factory(status="schema_error", response={})
    assert result.status == "schema_error"
    assert result.response == {}


def test_tool_result_unhashable_due_to_dict(valid_tool_result: ToolResult) -> None:
    """U18 — models.md §3.2 bullet 2."""
    with pytest.raises(TypeError):
        hash(valid_tool_result)


# ---------------------------------------------------------------------------
# §1.4 DriftEvent — U19–U22
# ---------------------------------------------------------------------------


def test_drift_event_happy_schema(valid_drift_event: DriftEvent) -> None:
    """U19 — models.md §4.4, §8.4."""
    assert valid_drift_event.turn == 3
    assert valid_drift_event.domain == "airline"
    assert valid_drift_event.drift_type == "schema"
    assert valid_drift_event.from_version == "v1"
    assert valid_drift_event.to_version == "v2"


_DRIFT_EVENT_FIELDS = [
    "turn",
    "drift_type",
    "domain",
    "description",
    "from_version",
    "to_version",
    "pattern_id",
]


@pytest.mark.parametrize("field_name", _DRIFT_EVENT_FIELDS)
def test_drift_event_frozen_mutation_raises(
    valid_drift_event: DriftEvent, field_name: str
) -> None:
    """U20 — models.md §3.1."""
    with pytest.raises(FrozenInstanceError):
        setattr(valid_drift_event, field_name, "anything")


def test_drift_event_is_hashable(valid_drift_event: DriftEvent) -> None:
    """U21 — models.md §3.2: primitive-only fields make DriftEvent hashable."""
    assert isinstance(hash(valid_drift_event), int)
    assert {valid_drift_event, valid_drift_event} == {valid_drift_event}


@pytest.mark.parametrize("drift_type", _DRIFT_TYPES)
def test_drift_event_accepts_all_five_drift_types(drift_type: str) -> None:
    """U22 — models.md §4.4 drift_type row."""
    event = DriftEvent(
        turn=2,
        drift_type=cast("DriftTypeLiteral", drift_type),
        domain="airline",
        description="d",
        from_version="v1",
        to_version="v2",
        pattern_id="airline.test",
    )
    assert event.drift_type == drift_type


# ---------------------------------------------------------------------------
# §1.5 GoalSpec — U23–U27
# ---------------------------------------------------------------------------


def test_goal_spec_happy_hinglish(valid_goal_spec: GoalSpec) -> None:
    """U23 — models.md §4.5, §8.3."""
    assert valid_goal_spec.language == "hinglish"
    assert valid_goal_spec.slots["from"] == "HYD"
    assert valid_goal_spec.intent == "book_flight"
    assert valid_goal_spec.domain == "airline"


@pytest.mark.parametrize("lang", _LANGUAGES)
def test_goal_spec_accepts_all_five_languages(lang: str) -> None:
    """U24 — models.md §4.5 language row."""
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD"},
        constraints={},
        language=cast("LanguageLiteral", lang),
        seed_utterance="x",
    )
    assert goal.language == lang


def test_goal_spec_frozen_mutation_raises(valid_goal_spec: GoalSpec) -> None:
    """U25 — models.md §3.1."""
    obj: Any = valid_goal_spec
    with pytest.raises(FrozenInstanceError):
        obj.intent = "other"


def test_goal_spec_unhashable_due_to_dict(valid_goal_spec: GoalSpec) -> None:
    """U26 — models.md §3.2 bullet 2."""
    with pytest.raises(TypeError):
        hash(valid_goal_spec)


def test_goal_spec_unicode_seed_utterance() -> None:
    """U27 — models.md §7 edge 3: Tamil utterance survives JSON round-trip."""
    original = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={"from": "HYD"},
        constraints={},
        language="ta",
        seed_utterance="{when} அன்று விமானம்",
    )
    payload = json.dumps(dataclasses.asdict(original), ensure_ascii=False)
    assert "அன்று" in payload
    loaded = json.loads(payload)
    rebuilt = GoalSpec(**loaded)
    assert rebuilt == original


# ---------------------------------------------------------------------------
# §1.6 DriftCallObservation — U28–U32
# ---------------------------------------------------------------------------


def test_observation_reset_state(valid_observation_reset: DriftCallObservation) -> None:
    """U28 — models.md §7 edges 1+2, §8.3."""
    obs = valid_observation_reset
    assert obs.turn == 0
    assert isinstance(obs.tool_results, tuple)
    assert len(obs.tool_results) == 0
    assert isinstance(obs.drift_log, tuple)
    assert len(obs.drift_log) == 0
    assert obs.last_transcript == ""
    assert obs.last_lang == ""
    assert obs.last_confidence == 1.0
    assert obs.budget_remaining == 12


def test_observation_tuple_not_list_for_sequences(
    valid_goal_spec: GoalSpec,
    valid_tool_result: ToolResult,
) -> None:
    """U29 — models.md §3.1, §7 edge 1: Python does not coerce list→tuple.

    Passing a list stores a list (documents "what Python does"); passing a
    tuple stores a tuple (documents "what the contract requires").
    """
    # What Python actually does: no coercion.
    obs_list = DriftCallObservation(
        turn=1,
        goal=valid_goal_spec,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=cast("tuple[ToolResult, ...]", [valid_tool_result]),
        drift_log=(),
        budget_remaining=11,
        available_tools=(),
    )
    assert isinstance(obs_list.tool_results, list)
    assert not isinstance(obs_list.tool_results, tuple)

    # What the contract requires: callers pass tuples.
    obs_tuple = DriftCallObservation(
        turn=1,
        goal=valid_goal_spec,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=(valid_tool_result,),
        drift_log=(),
        budget_remaining=11,
        available_tools=(),
    )
    assert isinstance(obs_tuple.tool_results, tuple)


_OBSERVATION_FIELDS = [
    "turn",
    "goal",
    "last_transcript",
    "last_lang",
    "last_confidence",
    "tool_results",
    "drift_log",
    "budget_remaining",
    "available_tools",
]


@pytest.mark.parametrize("field_name", _OBSERVATION_FIELDS)
def test_observation_frozen_mutation_raises(
    valid_observation_reset: DriftCallObservation, field_name: str
) -> None:
    """U30 — models.md §3.1."""
    with pytest.raises(FrozenInstanceError):
        setattr(valid_observation_reset, field_name, "anything")


def test_observation_unhashable(valid_observation_reset: DriftCallObservation) -> None:
    """U31 — models.md §3.2 bullet 2."""
    with pytest.raises(TypeError):
        hash(valid_observation_reset)


def test_observation_goal_ref_stable(valid_goal_spec: GoalSpec) -> None:
    """U32 — models.md §4.6 goal row: frozen=True does not deep-copy."""
    obs_a = DriftCallObservation(
        turn=0,
        goal=valid_goal_spec,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=(),
        drift_log=(),
        budget_remaining=12,
        available_tools=(),
    )
    obs_b = DriftCallObservation(
        turn=1,
        goal=valid_goal_spec,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=(),
        drift_log=(),
        budget_remaining=11,
        available_tools=(),
    )
    assert obs_a.goal is obs_b.goal


# ---------------------------------------------------------------------------
# §1.7 DriftCallState — U33–U37
# ---------------------------------------------------------------------------


def test_state_happy_turn_zero(valid_state_reset: DriftCallState) -> None:
    """U33 — models.md §4.7, §3.5 len(actions) == turn row."""
    state = valid_state_reset
    assert state.turn == 0
    assert len(state.actions) == state.turn
    assert state.drift_fired == ()
    assert state.done is False


def test_state_replace_appends_action(
    valid_state_reset: DriftCallState,
    valid_tool_call_action: DriftCallAction,
) -> None:
    """U34 — models.md §3.3, §8.4: replace produces a new state, original untouched."""
    new_state = dataclasses.replace(
        valid_state_reset,
        turn=valid_state_reset.turn + 1,
        actions=valid_state_reset.actions + (valid_tool_call_action,),
    )
    # Original untouched.
    assert valid_state_reset.turn == 0
    assert valid_state_reset.actions == ()
    # New state has the increment.
    assert new_state.turn == 1
    assert new_state.actions[-1] is valid_tool_call_action


def test_state_replace_dict_field_builds_new_dict(
    valid_state_reset: DriftCallState,
) -> None:
    """U35 — models.md §3.3: replace with a fresh dict preserves original."""
    new_state = dataclasses.replace(
        valid_state_reset,
        schema_versions={**valid_state_reset.schema_versions, "airline": "v2"},
    )
    assert valid_state_reset.schema_versions["airline"] == "v1"
    assert new_state.schema_versions["airline"] == "v2"


_STATE_FIELDS = [
    "episode_id",
    "goal",
    "vendor_states",
    "schema_versions",
    "drift_schedule",
    "drift_fired",
    "turn",
    "max_turns",
    "actions",
    "done",
]


@pytest.mark.parametrize("field_name", _STATE_FIELDS)
def test_state_frozen_mutation_raises(
    valid_state_reset: DriftCallState, field_name: str
) -> None:
    """U36 — models.md §3.1."""
    with pytest.raises(FrozenInstanceError):
        setattr(valid_state_reset, field_name, "anything")


def test_state_unhashable(valid_state_reset: DriftCallState) -> None:
    """U37 — models.md §3.2 bullet 2."""
    with pytest.raises(TypeError):
        hash(valid_state_reset)


# ---------------------------------------------------------------------------
# §1.8 Cross-cutting: JSON round-trip — U38–U40
# ---------------------------------------------------------------------------


def test_action_json_roundtrip_equality() -> None:
    """U38 — models.md §3.4 invariant."""
    original = DriftCallAction(
        action_type=ActionType.TOOL_CALL,
        tool_name="airline.search",
        tool_args={"filters": {"class": ["economy", "premium"], "max_stops": 1}},
        message="मुझे उड़ान चाहिए",
        confidence=0.5,
        rationale="because user asked",
    )
    payload = json.dumps(dataclasses.asdict(original), ensure_ascii=False)
    decoded = json.loads(payload)
    decoded["action_type"] = ActionType(decoded["action_type"])
    rebuilt = DriftCallAction(**decoded)
    assert rebuilt == original


def test_tool_result_json_roundtrip_equality(
    valid_tool_result: ToolResult,
) -> None:
    """U39 — models.md §3.4, §7 edge 4."""
    payload = json.dumps(dataclasses.asdict(valid_tool_result), ensure_ascii=False)
    decoded = json.loads(payload)
    rebuilt = ToolResult(**decoded)
    assert rebuilt == valid_tool_result


def test_observation_json_roundtrip_preserves_tuple_length(
    valid_goal_spec: GoalSpec,
    valid_tool_result: ToolResult,
    valid_drift_event: DriftEvent,
) -> None:
    """U40 — models.md §3.4: round-trip + re-tuple yields equal observation."""
    original = DriftCallObservation(
        turn=2,
        goal=valid_goal_spec,
        last_transcript="hello",
        last_lang="hinglish",
        last_confidence=0.9,
        tool_results=(valid_tool_result,),
        drift_log=(valid_drift_event,),
        budget_remaining=10,
        available_tools=("airline.search", "airline.book"),
    )
    payload = json.dumps(dataclasses.asdict(original), ensure_ascii=False)
    decoded = json.loads(payload)

    # Reconstructor contract: wrap sequences back into tuples, rebuild nested types.
    goal_dict = decoded.pop("goal")
    tool_results = tuple(ToolResult(**r) for r in decoded.pop("tool_results"))
    drift_log = tuple(DriftEvent(**d) for d in decoded.pop("drift_log"))
    available_tools = tuple(decoded.pop("available_tools"))
    rebuilt = DriftCallObservation(
        goal=GoalSpec(**goal_dict),
        tool_results=tool_results,
        drift_log=drift_log,
        available_tools=available_tools,
        **decoded,
    )
    assert rebuilt == original


# ---------------------------------------------------------------------------
# §2 Property tests — P1–P6
# ---------------------------------------------------------------------------


_SETTINGS = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=(HealthCheck.too_slow,),
)


def _make_action(at: ActionType) -> DriftCallAction:
    return DriftCallAction(action_type=at)


@_SETTINGS
@given(turn=st.integers(min_value=0, max_value=16))
def test_state_turn_matches_len_actions_by_construction(turn: int) -> None:
    """P1 — models.md §3.5 state row, §7 edge 9.

    For every turn ∈ [0, 16] we can construct a state where
    len(actions) == turn — witnessing the constructibility half of the invariant.
    """
    actions = tuple(_make_action(ActionType.ABORT) for _ in range(turn))
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={},
        constraints={},
        language="en",
        seed_utterance="x",
    )
    state = DriftCallState(
        episode_id="ep_prop",
        goal=goal,
        vendor_states={},
        schema_versions={},
        drift_schedule=(),
        drift_fired=(),
        turn=turn,
        max_turns=16,
        actions=actions,
        done=False,
    )
    assert len(state.actions) == state.turn


@_SETTINGS
@given(
    schedule_len=st.integers(min_value=0, max_value=2),
    fired_len=st.integers(min_value=0, max_value=2),
)
def test_drift_fired_is_subset_of_drift_schedule(
    schedule_len: int, fired_len: int
) -> None:
    """P2 — models.md §3.5 drift_fired ⊆ drift_schedule row."""
    fired_len = min(fired_len, schedule_len)
    schedule = tuple(
        DriftEvent(
            turn=i + 1,
            drift_type="schema",
            domain="airline",
            description=f"d{i}",
            from_version="v1",
            to_version="v2",
            pattern_id=f"airline.test_{i}",
        )
        for i in range(schedule_len)
    )
    fired = schedule[:fired_len]
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={},
        constraints={},
        language="en",
        seed_utterance="x",
    )
    state = DriftCallState(
        episode_id="ep_prop",
        goal=goal,
        vendor_states={},
        schema_versions={},
        drift_schedule=schedule,
        drift_fired=fired,
        turn=0,
        max_turns=16,
        actions=(),
        done=False,
    )
    assert set(state.drift_fired).issubset(set(state.drift_schedule))
    assert len(state.drift_fired) <= len(state.drift_schedule)
    assert state.drift_fired == state.drift_schedule[: len(state.drift_fired)]


@_SETTINGS
@given(domain=_domains)
def test_probe_schema_tool_name_is_bare_domain(domain: str) -> None:
    """P3 — models.md §3.5 PROBE_SCHEMA row: fixtures obey bare-domain contract."""
    action = DriftCallAction(
        action_type=ActionType.PROBE_SCHEMA,
        tool_name=domain,
    )
    assert action.tool_name is not None
    assert "." not in action.tool_name
    assert action.tool_name in {"airline", "cab", "restaurant", "hotel", "payment"}


def _sample_action_strategy() -> st.SearchStrategy[DriftCallAction]:
    return st.builds(
        DriftCallAction,
        action_type=st.sampled_from(_ACTION_TYPES),
        tool_name=st.one_of(st.none(), st.text(max_size=16)),
        tool_args=st.one_of(st.none(), _json_dict),
        message=st.one_of(st.none(), st.text(max_size=64)),
        confidence=st.one_of(
            st.none(), st.floats(min_value=0.0, max_value=1.0, width=32)
        ),
        rationale=st.one_of(st.none(), st.text(max_size=64)),
    )


_DATACLASS_SAMPLES: list[tuple[str, Callable[[], Any], list[str]]] = [
    (
        "DriftCallAction",
        lambda: DriftCallAction(action_type=ActionType.ABORT),
        _ACTION_FIELDS,
    ),
    (
        "ToolResult",
        lambda: ToolResult(
            tool_name="airline.search",
            status="ok",
            response={"x": 1},
            schema_version="v1",
            latency_ms=10,
        ),
        _TOOL_RESULT_FIELDS,
    ),
    (
        "DriftEvent",
        lambda: DriftEvent(
            turn=1,
            drift_type="schema",
            domain="airline",
            description="d",
            from_version="v1",
            to_version="v2",
            pattern_id="airline.test",
        ),
        _DRIFT_EVENT_FIELDS,
    ),
    (
        "GoalSpec",
        lambda: GoalSpec(
            domain="airline",
            intent="book_flight",
            slots={},
            constraints={},
            language="en",
            seed_utterance="x",
        ),
        [
            "domain",
            "intent",
            "slots",
            "constraints",
            "language",
            "seed_utterance",
        ],
    ),
    (
        "DriftCallObservation",
        lambda: DriftCallObservation(
            turn=0,
            goal=GoalSpec(
                domain="airline",
                intent="book_flight",
                slots={},
                constraints={},
                language="en",
                seed_utterance="x",
            ),
            last_transcript="",
            last_lang="",
            last_confidence=1.0,
            tool_results=(),
            drift_log=(),
            budget_remaining=12,
            available_tools=(),
        ),
        _OBSERVATION_FIELDS,
    ),
    (
        "DriftCallState",
        lambda: DriftCallState(
            episode_id="ep_prop",
            goal=GoalSpec(
                domain="airline",
                intent="book_flight",
                slots={},
                constraints={},
                language="en",
                seed_utterance="x",
            ),
            vendor_states={},
            schema_versions={},
            drift_schedule=(),
            drift_fired=(),
            turn=0,
            max_turns=16,
            actions=(),
            done=False,
        ),
        _STATE_FIELDS,
    ),
]


@pytest.mark.parametrize(
    ("cls_name", "builder", "fields"),
    _DATACLASS_SAMPLES,
    ids=[s[0] for s in _DATACLASS_SAMPLES],
)
def test_frozen_invariant_universal(
    cls_name: str, builder: Callable[[], Any], fields: list[str]
) -> None:
    """P4 — models.md §3.1, §5 row 1: frozen guarantee is universal.

    Stronger than the per-class U8/U15/U20/U25/U30/U36 tests because it is
    driven across every dataclass in one sweep.
    """
    instance = builder()
    for field_name in fields:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field_name, "anything")


@_SETTINGS
@given(action=_sample_action_strategy())
def test_json_roundtrip_preserves_equality_for_action(action: DriftCallAction) -> None:
    """P5 — models.md §3.4 invariant."""
    payload = json.dumps(dataclasses.asdict(action), ensure_ascii=False)
    decoded = json.loads(payload)
    decoded["action_type"] = ActionType(decoded["action_type"])
    rebuilt = DriftCallAction(**decoded)
    assert rebuilt == action


@_SETTINGS
@given(
    max_turns=st.integers(min_value=1, max_value=16),
    data=st.data(),
)
def test_observation_budget_non_negative_by_construction(
    max_turns: int, data: st.DataObject
) -> None:
    """P6 — models.md §3.5 budget row, §4.6 budget row."""
    turn = data.draw(st.integers(min_value=0, max_value=max_turns))
    goal = GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={},
        constraints={},
        language="en",
        seed_utterance="x",
    )
    obs = DriftCallObservation(
        turn=turn,
        goal=goal,
        last_transcript="",
        last_lang="",
        last_confidence=1.0,
        tool_results=(),
        drift_log=(),
        budget_remaining=max_turns - turn,
        available_tools=(),
    )
    assert obs.budget_remaining >= 0
