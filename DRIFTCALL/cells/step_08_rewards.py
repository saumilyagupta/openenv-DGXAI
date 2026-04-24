"""DriftCall reward pipeline.

Implements docs/modules/rewards.md and DESIGN.md §7. Pure-functional: no I/O,
no clock, no RNG, no LLM. Every reward is deterministic on the input Episode.

Public surface:
    Episode, Rewards, RewardComputationError, AVAILABLE_TOOL_REGISTRY,
    task_completion, drift_detection, constraint_adherence,
    format_compliance, anti_hack_penalty,
    combine_quality, brier_penalty, apply_uncertain_floor, final_reward,
    compute_rewards.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftEvent,
    GoalSpec,
    ToolResult,
)
from cells.step_05_vendors import TOOLS as _VENDOR_TOOLS
from cells.step_06_drift_injector import DriftPattern, list_patterns

__all__ = [
    "AVAILABLE_TOOL_REGISTRY",
    "Episode",
    "RewardComputationError",
    "Rewards",
    "anti_hack_penalty",
    "apply_uncertain_floor",
    "brier_penalty",
    "combine_quality",
    "compute_rewards",
    "constraint_adherence",
    "drift_detection",
    "final_reward",
    "format_compliance",
    "task_completion",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


AVAILABLE_TOOL_REGISTRY: frozenset[str] = frozenset(_VENDOR_TOOLS)

_RESERVED_KEYS: frozenset[str] = frozenset(
    {"__turn__", "__schema_version__", "__done__", "__episode_id__"},
)

_VALID_DRIFT_TYPES: frozenset[str] = frozenset(
    {"schema", "policy", "tnc", "pricing", "auth"},
)

_VALID_TERMINATIONS: frozenset[str] = frozenset(
    {"SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"},
)

# Hour windows (24h IST). "night" wraps midnight; encoded as (lo, hi+24).
_TIME_WINDOWS: dict[str, tuple[int, int]] = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 22),
    "night": (22, 30),
}

_FAILURE_STATUSES: frozenset[str] = frozenset(
    {"schema_error", "policy_error", "auth_error"},
)

# snake_case identifier with at least one underscore between alphanumeric segments
_SNAKE_FIELD_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

_PATTERNS_BY_ID: dict[str, DriftPattern] = {p.id: p for p in list_patterns()}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RewardComputationError(Exception):
    """Raised when rewards cannot be computed for a malformed episode."""

    def __init__(self, reason: str, episode_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.episode_id = episode_id


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    episode_id: str
    goal: GoalSpec
    actions: tuple[DriftCallAction, ...]
    action_turns: tuple[int, ...]
    tool_results: tuple[ToolResult, ...]
    tool_result_turns: tuple[int, ...]
    drift_log: tuple[DriftEvent, ...]
    vendor_states_final: dict[str, dict[str, Any]]
    schema_versions_final: dict[str, str]
    max_turns: int
    turns_used: int
    terminated_by: Literal["SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"]
    stage: Literal[1, 2, 3]
    drift_pattern_overrides: dict[str, DriftPattern] = field(default_factory=dict)


@dataclass(frozen=True)
class Rewards:
    r1: float
    r2: float
    r3: float
    r4: float
    r5: float
    quality: float
    brier: float
    reward: float
    confidence: float | None
    floor_applied: bool
    breakdown: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_pattern(episode: Episode, drift: DriftEvent) -> DriftPattern:
    """Look up the DriftPattern via episode overrides, then global registry."""
    pattern_id = drift.pattern_id
    if pattern_id in episode.drift_pattern_overrides:
        return episode.drift_pattern_overrides[pattern_id]
    if pattern_id in _PATTERNS_BY_ID:
        return _PATTERNS_BY_ID[pattern_id]
    raise RewardComputationError(
        f"unknown pattern_id: {pattern_id}",
        episode.episode_id,
    )


def _validate_hints(pattern: DriftPattern, episode: Episode) -> tuple[str, ...]:
    """Return non-empty stripped hints; raise on empty."""
    cleaned = tuple(h for h in pattern.detection_hints if h and h.strip())
    if not cleaned:
        raise RewardComputationError(
            f"drift {pattern.id} has empty detection_hints",
            episode.episode_id,
        )
    return cleaned


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _safe_lower(text: str | None) -> str:
    return text.lower() if text else ""


def _iter_string_values(node: Any) -> list[str]:
    """Recursively collect string values (numerics/booleans excluded)."""
    out: list[str] = []
    if isinstance(node, bool):
        return out
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            out.extend(_iter_string_values(v))
    elif isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_iter_string_values(item))
    return out


def _iter_keys(node: Any) -> list[str]:
    """Recursively collect dict keys."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(str(k))
            out.extend(_iter_keys(v))
    elif isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_iter_keys(item))
    return out


def _build_args_search_corpus(tool_args: dict[str, Any] | None) -> str:
    """Lowercased keys + string values; numeric/boolean leaves excluded."""
    if not tool_args:
        return ""
    keys = _iter_keys(tool_args)
    strings = _iter_string_values(tool_args)
    return " ".join(keys + strings).lower()


def _mentions_drift(message: str | None, hints: tuple[str, ...]) -> bool:
    if not message:
        return False
    target = message.lower()
    return any(hint.lower() in target for hint in hints)


def _args_mention_drift(
    tool_args: dict[str, Any] | None,
    hints: tuple[str, ...],
) -> bool:
    corpus = _build_args_search_corpus(tool_args)
    if not corpus:
        return False
    return any(hint.lower() in corpus for hint in hints)


def _new_field_names(pattern: DriftPattern) -> tuple[str, ...]:
    """Field names introduced by the drift mutation (post-drift schema)."""
    mutation = pattern.mutation
    out: list[str] = []
    rename = mutation.get("rename")
    if isinstance(rename, dict):
        out.extend(str(v) for v in rename.values())
    new_fields = mutation.get("require_new_field")
    if isinstance(new_fields, (list, tuple)):
        out.extend(str(v) for v in new_fields)
    change = mutation.get("change_type")
    if isinstance(change, dict):
        out.extend(str(v) for v in change.values())
    return tuple(out)


def _old_field_names(pattern: DriftPattern) -> tuple[str, ...]:
    """Field names from the pre-drift schema."""
    mutation = pattern.mutation
    out: list[str] = []
    rename = mutation.get("rename")
    if isinstance(rename, dict):
        out.extend(str(k) for k in rename)
    removed = mutation.get("remove")
    if isinstance(removed, (list, tuple)):
        out.extend(str(v) for v in removed)
    change = mutation.get("change_type")
    if isinstance(change, dict):
        out.extend(str(k) for k in change)
    return tuple(out)


def _uses_new_schema(
    tool_args: dict[str, Any] | None,
    pattern: DriftPattern,
) -> bool:
    if not tool_args:
        return False
    new_fields = _new_field_names(pattern)
    if not new_fields:
        return False
    keys_lower = {k.lower() for k in _iter_keys(tool_args)}
    return any(f.lower() in keys_lower for f in new_fields)


def _uses_old_schema(
    tool_args: dict[str, Any] | None,
    pattern: DriftPattern,
) -> bool:
    if not tool_args:
        return False
    old_fields = _old_field_names(pattern)
    if not old_fields:
        return False
    keys_lower = {k.lower() for k in _iter_keys(tool_args)}
    return any(f.lower() in keys_lower for f in old_fields)


def _has_3plus_old_schema_retries(
    episode: Episode,
    pattern: DriftPattern,
    drift_turn: int,
) -> bool:
    """True iff >= 3 TOOL_CALLs after drift_turn use OLD schema."""
    count = 0
    for action, turn in zip(episode.actions, episode.action_turns, strict=True):
        if turn <= drift_turn:
            continue
        if action.action_type != ActionType.TOOL_CALL:
            continue
        if _uses_old_schema(action.tool_args, pattern):
            count += 1
    return count >= 3


# ---------------------------------------------------------------------------
# R1 — Task Completion
# ---------------------------------------------------------------------------


def _parse_iso_hour(timestamp: str) -> int | None:
    """Parse 'YYYY-MM-DDTHH:MM[:SS]' and return hour, or None on failure."""
    if "T" not in timestamp:
        return None
    try:
        time_part = timestamp.split("T", 1)[1]
        return int(time_part[:2])
    except (ValueError, IndexError):
        return None


def _hour_in_window(hour: int, window: str) -> bool:
    win = _TIME_WINDOWS.get(window)
    if win is None:
        return True
    lo, hi = win
    if hi <= 24:
        return lo <= hour < hi
    return hour >= lo or hour < (hi - 24)


def _check_airline_booking(
    goal: GoalSpec,
    vendor_states: dict[str, dict[str, Any]],
) -> bool:
    state = vendor_states.get("airline", {})
    if not isinstance(state, dict):
        return False
    bookings = state.get("bookings", [])
    if not isinstance(bookings, list) or not bookings:
        return False
    expected_from = goal.slots.get("from")
    expected_to = goal.slots.get("to")
    budget = goal.constraints.get("budget_inr")
    window = goal.constraints.get("time_window")
    for booking in bookings:
        if not isinstance(booking, dict):
            continue
        if expected_from is not None and booking.get("from") != expected_from:
            continue
        if expected_to is not None and booking.get("to") != expected_to:
            continue
        if budget is not None:
            total = booking.get("total")
            if total is None or total > budget:
                continue
        if window is not None:
            depart = booking.get("depart")
            if not isinstance(depart, str):
                continue
            hour = _parse_iso_hour(depart)
            if hour is None or not _hour_in_window(hour, str(window)):
                continue
        return True
    return False


def _check_cab_booking(
    goal: GoalSpec,
    vendor_states: dict[str, dict[str, Any]],
) -> bool:
    state = vendor_states.get("cab", {})
    if not isinstance(state, dict):
        return False
    bookings = state.get("bookings", [])
    if not isinstance(bookings, list) or not bookings:
        return False
    expected_pickup = goal.slots.get("pickup")
    expected_drop = goal.slots.get("drop")
    expected_when = goal.slots.get("when")
    for booking in bookings:
        if not isinstance(booking, dict):
            continue
        if expected_pickup is not None and booking.get("pickup") != expected_pickup:
            continue
        if expected_drop is not None and booking.get("drop") != expected_drop:
            continue
        if expected_when is not None and booking.get("pickup_time") != expected_when:
            continue
        return True
    return False


def _check_restaurant_order(
    goal: GoalSpec,
    vendor_states: dict[str, dict[str, Any]],
) -> bool:
    state = vendor_states.get("restaurant", {})
    if not isinstance(state, dict):
        return False
    orders = state.get("orders", [])
    if not isinstance(orders, list) or not orders:
        return False
    budget = goal.constraints.get("budget_inr")
    dietary = goal.constraints.get("dietary")
    for order in orders:
        if not isinstance(order, dict):
            continue
        if budget is not None:
            total = order.get("total")
            if total is None or total > budget:
                continue
        if dietary is not None:
            items = order.get("items", [])
            if dietary in {"veg", "veg_only"} and not all(
                isinstance(it, dict) and it.get("veg") is True for it in items
            ):
                continue
        return True
    return False


def _check_hotel_booking(
    goal: GoalSpec,
    vendor_states: dict[str, dict[str, Any]],
) -> bool:
    state = vendor_states.get("hotel", {})
    if not isinstance(state, dict):
        return False
    bookings = state.get("bookings", [])
    if not isinstance(bookings, list) or not bookings:
        return False
    expected_city = goal.slots.get("city")
    expected_in = goal.slots.get("checkin")
    expected_out = goal.slots.get("checkout")
    expected_room = goal.slots.get("room_type")
    for booking in bookings:
        if not isinstance(booking, dict):
            continue
        if expected_city is not None and booking.get("city") != expected_city:
            continue
        if expected_in is not None and booking.get("checkin") != expected_in:
            continue
        if expected_out is not None and booking.get("checkout") != expected_out:
            continue
        if expected_room is not None and booking.get("room_type") != expected_room:
            continue
        return True
    return False


def task_completion(episode: Episode) -> float:
    """R1: 1.0 iff terminated by SUBMIT and per-domain success predicate holds."""
    if episode.terminated_by != "SUBMIT":
        return 0.0
    domain = episode.goal.domain
    final = episode.vendor_states_final
    if domain == "airline":
        ok = _check_airline_booking(episode.goal, final)
    elif domain == "cab":
        ok = _check_cab_booking(episode.goal, final)
    elif domain == "restaurant":
        ok = _check_restaurant_order(episode.goal, final)
    elif domain == "hotel":
        ok = _check_hotel_booking(episode.goal, final)
    else:
        ok = False
    return 1.0 if ok else 0.0


def _r1_breakdown(episode: Episode) -> dict[str, Any]:
    domain = episode.goal.domain
    if domain not in {"airline", "cab", "restaurant", "hotel"}:
        return {
            "domain": domain,
            "success_predicate": "unknown_domain",
            "matched_slots": {},
            "missing_slots": [],
        }
    return {
        "domain": domain,
        "success_predicate": f"{domain}_booking_match",
        "matched_slots": dict(episode.goal.slots),
        "missing_slots": [],
    }


# ---------------------------------------------------------------------------
# R2 — Drift Detection
# ---------------------------------------------------------------------------


def _drift_detection_with_breakdown(
    episode: Episode,
) -> tuple[float, dict[str, Any]]:
    breakdown: dict[str, Any] = {
        "stage": int(episode.stage),
        "drifts_total": len(episode.drift_log),
        "drifts_detected": 0,
        "per_drift": [],
        "three_plus_retries": False,
    }
    if episode.stage == 1 or len(episode.drift_log) == 0:
        if episode.stage in (2, 3) and len(episode.drift_log) == 0:
            breakdown["stage2_3_no_drift"] = True
        return 0.5, breakdown

    score = 1.0
    detected = 0
    any_old_schema_retries = False

    for drift in episode.drift_log:
        pattern = _resolve_pattern(episode, drift)
        hints = _validate_hints(pattern, episode)
        window_turns = [drift.turn, drift.turn + 1, drift.turn + 2]
        actions_in_window = [
            (a, t)
            for a, t in zip(episode.actions, episode.action_turns, strict=True)
            if t in window_turns
        ]
        hit_speech = False
        hit_args = False
        hit_adapt = False
        for action, _turn in actions_in_window:
            if (
                action.action_type in {ActionType.SPEAK, ActionType.CLARIFY}
                and _mentions_drift(action.message, hints)
            ):
                hit_speech = True
            if action.action_type == ActionType.TOOL_CALL:
                if _args_mention_drift(action.tool_args, hints):
                    hit_args = True
                if _uses_new_schema(action.tool_args, pattern):
                    hit_adapt = True

        breakdown["per_drift"].append({
            "drift_id": drift.pattern_id,
            "hit_by_speech": hit_speech,
            "hit_by_args_hint": hit_args,
            "hit_by_adaptation": hit_adapt,
            "window_turns": list(window_turns),
        })

        if hit_speech or hit_args or hit_adapt:
            detected += 1
        else:
            score = 0.0

        if _has_3plus_old_schema_retries(episode, pattern, drift.turn):
            any_old_schema_retries = True

    breakdown["drifts_detected"] = detected
    breakdown["three_plus_retries"] = any_old_schema_retries
    if any_old_schema_retries:
        score = 0.0
    return score, breakdown


def drift_detection(episode: Episode) -> float:
    """R2: stage-1/no-drift → 0.5; per-drift any-branch hit → 1.0; one miss → 0.0."""
    score, _ = _drift_detection_with_breakdown(episode)
    return score


# ---------------------------------------------------------------------------
# R3 — Constraint Adherence
# ---------------------------------------------------------------------------


_KNOWN_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        "budget_inr",
        "time_window",
        "dietary",
        "passenger_count",
        "pickup",
        "seat_type",
        "checkin",
        "checkout",
        "room_type",
    },
)


def _final_booking(episode: Episode) -> dict[str, Any] | None:
    """Return the most recent booking/order from vendor_states_final."""
    domain = episode.goal.domain
    state = episode.vendor_states_final.get(domain, {})
    if not isinstance(state, dict):
        return None
    items = (
        state.get("orders", []) if domain == "restaurant" else state.get("bookings", [])
    )
    if not isinstance(items, list) or not items:
        return None
    last = items[-1]
    return last if isinstance(last, dict) else None


def _check_constraint(
    key: str,
    expected: Any,
    booking: dict[str, Any] | None,
) -> bool:
    if booking is None:
        return False
    if key == "budget_inr":
        total = booking.get("total")
        if total is None:
            return False
        try:
            return float(total) <= float(expected)
        except (TypeError, ValueError):
            return False
    if key == "time_window":
        depart = booking.get("depart") or booking.get("pickup_time")
        if not isinstance(depart, str):
            return False
        hour = _parse_iso_hour(depart)
        if hour is None:
            return False
        return _hour_in_window(hour, str(expected))
    if key == "dietary":
        items = booking.get("items", [])
        if not isinstance(items, list):
            return False
        if expected in {"veg", "veg_only"}:
            return all(
                isinstance(it, dict) and it.get("veg") is True for it in items
            )
        return True
    if key == "passenger_count":
        return bool(booking.get("passenger_count") == expected)
    if key == "pickup":
        return bool(booking.get("pickup") == expected)
    if key == "seat_type":
        return bool(booking.get("seat_type") == expected)
    if key == "checkin":
        return bool(booking.get("checkin") == expected)
    if key == "checkout":
        return bool(booking.get("checkout") == expected)
    if key == "room_type":
        return bool(booking.get("room_type") == expected)
    return False


def _r3_with_breakdown(episode: Episode) -> tuple[float, dict[str, Any]]:
    constraints = episode.goal.constraints
    if not constraints:
        return 1.0, {
            "total_constraints": 0,
            "satisfied_constraints": 0,
            "unknown_constraints": [],
            "failures": [],
        }
    booking = _final_booking(episode)
    satisfied = 0
    unknown: list[str] = []
    failures: list[dict[str, Any]] = []
    for key, expected in constraints.items():
        if key not in _KNOWN_CONSTRAINT_KEYS:
            unknown.append(key)
            satisfied += 1
            continue
        if _check_constraint(key, expected, booking):
            satisfied += 1
        else:
            actual = booking.get(key) if booking else None
            failures.append({"key": key, "expected": expected, "actual": actual})
    total = len(constraints)
    return satisfied / total, {
        "total_constraints": total,
        "satisfied_constraints": satisfied,
        "unknown_constraints": unknown,
        "failures": failures,
    }


def constraint_adherence(episode: Episode) -> float:
    """R3: fraction of goal.constraints satisfied by the final booking."""
    score, _ = _r3_with_breakdown(episode)
    return score


# ---------------------------------------------------------------------------
# R4 — Format Compliance
# ---------------------------------------------------------------------------


def _is_valid_json(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def _has_tamil(text: str) -> bool:
    return any("஀" <= c <= "௿" for c in text)


def _has_kannada(text: str) -> bool:
    return any("ಀ" <= c <= "೿" for c in text)


def _has_indic(text: str) -> bool:
    return _has_devanagari(text) or _has_tamil(text) or _has_kannada(text)


def _language_mismatch(message: str, goal_language: str) -> bool:
    """Asymmetric heuristic per rewards.md §3.5; permissive for ta/kn/hinglish.

    - "en"      : mismatch iff message contains any Indic script.
    - "hi"      : mismatch iff message contains no Devanagari.
    - others    : Latin or local script accepted (transliteration is common).
    """
    if not message:
        return False
    if goal_language == "en":
        return _has_indic(message)
    if goal_language == "hi":
        return not _has_devanagari(message)
    return False


def _r4_with_breakdown(episode: Episode) -> tuple[float, dict[str, Any]]:
    score = 1.0
    deductions: list[dict[str, Any]] = []
    for action, turn in zip(episode.actions, episode.action_turns, strict=True):
        if action.action_type == ActionType.TOOL_CALL:
            if not _is_valid_json(action.tool_args):
                score -= 0.20
                deductions.append({"turn": turn, "reason": "invalid_json", "amount": 0.20})
            if action.tool_name not in AVAILABLE_TOOL_REGISTRY:
                score -= 0.10
                deductions.append({"turn": turn, "reason": "unknown_tool", "amount": 0.10})
            if action.rationale is None or len(action.rationale.strip()) == 0:
                score -= 0.05
                deductions.append({
                    "turn": turn,
                    "reason": "missing_rationale",
                    "amount": 0.05,
                })
        if action.action_type in {ActionType.SPEAK, ActionType.CLARIFY}:
            msg = action.message or ""
            if _language_mismatch(msg, episode.goal.language):
                score -= 0.10
                deductions.append({
                    "turn": turn,
                    "reason": "language_mismatch",
                    "amount": 0.10,
                })
    score = max(0.0, min(1.0, score))
    return score, {"deductions": deductions}


def format_compliance(episode: Episode) -> float:
    """R4: deductive from 1.0; clamped to [0, 1]."""
    score, _ = _r4_with_breakdown(episode)
    return score


# ---------------------------------------------------------------------------
# R5 — Anti-Hack Penalty
# ---------------------------------------------------------------------------


def _build_whitelist(tool_results: tuple[ToolResult, ...]) -> set[str]:
    """Recursive walk: every key + every primitive leaf, lowercased."""
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            seen.add(str(node).lower())
            return
        if isinstance(node, dict):
            for k, v in node.items():
                seen.add(str(k).lower())
                walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, (str, int, float)):
            seen.add(str(node).lower())

    for tr in tool_results:
        walk(tr.response)
    return seen


def _extract_field_tokens(text: str | None) -> list[str]:
    """Return lowercased snake_case identifier tokens (>=1 underscore)."""
    if not text:
        return []
    return _SNAKE_FIELD_RE.findall(text.lower())


def _hallucinated_offenses(
    episode: Episode,
    whitelist: set[str],
) -> list[tuple[int, str]]:
    """(turn, token) pairs where agent referenced a snake_case field not in whitelist.

    Scans natural-language surfaces only (`message`, `rationale`). Tool-call
    `tool_args` keys are the agent's own request payload — they are policed by
    R4 (unknown_tool / invalid_json) and the protected-write check, not as
    hallucinated field references.
    """
    offenses: list[tuple[int, str]] = []
    for action, turn in zip(episode.actions, episode.action_turns, strict=True):
        candidates: list[str] = []
        candidates.extend(_extract_field_tokens(action.message))
        candidates.extend(_extract_field_tokens(action.rationale))
        for token in candidates:
            if token not in whitelist:
                offenses.append((turn, token))
    return offenses


def _canonical_args(tool_args: dict[str, Any] | None) -> str:
    if tool_args is None:
        return "null"
    try:
        return json.dumps(tool_args, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        try:
            return repr(sorted(tool_args.items()))
        except TypeError:
            return repr(tool_args)


def _max_repeated_tool_calls(episode: Episode) -> int:
    """Maximum repetition count of any (tool_name, normalised_args) pair."""
    counts: dict[tuple[str | None, str], int] = {}
    for action in episode.actions:
        if action.action_type != ActionType.TOOL_CALL:
            continue
        key = (action.tool_name, _canonical_args(action.tool_args))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def _count_actions(episode: Episode, action_type: ActionType) -> int:
    return sum(1 for a in episode.actions if a.action_type == action_type)


def _has_unsubstantiated_drift_claim(
    episode: Episode,
) -> tuple[bool, int | None]:
    """SPEAK/CLARIFY mentions 'drift' before any prior failure or drift_log entry."""
    for action, turn in zip(episode.actions, episode.action_turns, strict=True):
        if action.action_type not in {ActionType.SPEAK, ActionType.CLARIFY}:
            continue
        if "drift" not in _safe_lower(action.message):
            continue
        prior_failure = any(
            tr.status in _FAILURE_STATUSES and tr_turn <= turn
            for tr, tr_turn in zip(
                episode.tool_results, episode.tool_result_turns, strict=True,
            )
        )
        if prior_failure:
            continue
        prior_drift = any(d.turn <= turn for d in episode.drift_log)
        if prior_drift:
            continue
        return True, turn
    return False, None


def _is_protected_tool(tool_name: str | None) -> bool:
    if not tool_name:
        return False
    return tool_name.startswith("__") and tool_name.endswith("__")


def _has_protected_write(episode: Episode) -> tuple[bool, int | None]:
    for action, turn in zip(episode.actions, episode.action_turns, strict=True):
        if action.action_type != ActionType.TOOL_CALL:
            continue
        if _is_protected_tool(action.tool_name):
            return True, turn
        if action.tool_args:
            for key in action.tool_args:
                if key in _RESERVED_KEYS:
                    return True, turn
    return False, None


def _r5_with_breakdown(episode: Episode) -> tuple[float, dict[str, Any]]:
    penalty = 0.0
    offenses: list[dict[str, Any]] = []

    whitelist = _build_whitelist(episode.tool_results)
    hallucinations = _hallucinated_offenses(episode, whitelist)
    if hallucinations:
        penalty -= 1.0
        first_turn, first_token = hallucinations[0]
        offenses.append({
            "code": "hallucinated_field",
            "turn": first_turn,
            "evidence": first_token,
        })

    repeats = _max_repeated_tool_calls(episode)
    if repeats > 3:
        penalty -= 0.5
        offenses.append({
            "code": "repeated_tool_calls",
            "turn": None,
            "evidence": f"max_repetition={repeats}",
        })

    probe_count = _count_actions(episode, ActionType.PROBE_SCHEMA)
    if probe_count >= 3:
        penalty -= 0.5
        offenses.append({
            "code": "probe_schema_abuse",
            "turn": None,
            "evidence": f"probe_count={probe_count}",
        })

    bare, bare_turn = _has_unsubstantiated_drift_claim(episode)
    if bare:
        penalty -= 0.3
        offenses.append({
            "code": "bare_drift_assertion",
            "turn": bare_turn,
            "evidence": "speak mentions drift without prior failure",
        })

    protected, protected_turn = _has_protected_write(episode)
    if protected:
        penalty -= 0.2
        offenses.append({
            "code": "protected_write",
            "turn": protected_turn,
            "evidence": "reserved key or protected tool",
        })

    penalty = max(-1.0, penalty)
    return penalty, {"offenses": offenses}


def anti_hack_penalty(episode: Episode) -> float:
    """R5: additive penalties, clamped to [-1.0, 0.0]."""
    score, _ = _r5_with_breakdown(episode)
    return score


# ---------------------------------------------------------------------------
# Combination helpers
# ---------------------------------------------------------------------------


def combine_quality(
    r1: float,
    r2: float,
    r3: float,
    r4: float,
    r5: float,
) -> float:
    """Weighted sum (0.50/0.20/0.15/0.10/0.05). Does not clamp or round."""
    return 0.50 * r1 + 0.20 * r2 + 0.15 * r3 + 0.10 * r4 + 0.05 * min(r5, 0.0)


def brier_penalty(confidence: float | None, r1: float) -> float:
    """min((conf - r1)^2, 0.5) when confidence given; else 0.0."""
    if confidence is None:
        return 0.0
    raw = (confidence - r1) ** 2
    return raw if raw <= 0.5 else 0.5


def apply_uncertain_floor(
    reward: float,
    r1: float,
    confidence: float | None,
) -> float:
    """Floor at 0.3 iff r1==0, confidence is not None, confidence < 0.3."""
    if r1 == 0.0 and confidence is not None and confidence < 0.3:
        return max(reward, 0.3)
    return reward


def final_reward(
    quality: float,
    brier: float,
    r1: float,
    confidence: float | None,
) -> float:
    """multiply -> floor -> clamp [0,1] -> round 3dp."""
    reward = quality * (1.0 - brier)
    reward = apply_uncertain_floor(reward, r1, confidence)
    reward = max(0.0, min(1.0, reward))
    return round(reward, 3)


# ---------------------------------------------------------------------------
# compute_rewards orchestration
# ---------------------------------------------------------------------------


def _validate_episode_structure(episode: Episode) -> None:
    if episode.goal is None:
        raise RewardComputationError("episode.goal is None", episode.episode_id)
    if episode.terminated_by is None:
        raise RewardComputationError("episode not terminated", episode.episode_id)
    if episode.terminated_by not in _VALID_TERMINATIONS:
        raise RewardComputationError(
            f"episode not terminated (invalid terminated_by={episode.terminated_by!r})",
            episode.episode_id,
        )
    for drift in episode.drift_log:
        if drift.drift_type not in _VALID_DRIFT_TYPES:
            raise RewardComputationError(
                f"unknown drift_type: {drift.drift_type}",
                episode.episode_id,
            )
        if (
            drift.pattern_id not in episode.drift_pattern_overrides
            and drift.pattern_id not in _PATTERNS_BY_ID
        ):
            raise RewardComputationError(
                f"unknown pattern_id: {drift.pattern_id}",
                episode.episode_id,
            )
    n_tool_calls = sum(
        1 for a in episode.actions if a.action_type == ActionType.TOOL_CALL
    )
    if n_tool_calls != len(episode.tool_results):
        raise RewardComputationError(
            "action/tool_result count mismatch",
            episode.episode_id,
        )


def _extract_confidence(episode: Episode) -> tuple[float | None, bool]:
    """Return (raw_confidence, clamped_flag). Raises on non-finite."""
    if episode.terminated_by != "SUBMIT":
        return None, False
    submit_conf: float | None = None
    for action in reversed(episode.actions):
        if action.action_type == ActionType.SUBMIT:
            submit_conf = action.confidence
            break
    if submit_conf is None:
        return None, False
    if not _is_finite(float(submit_conf)):
        raise RewardComputationError(
            "non-finite value in reward computation",
            episode.episode_id,
        )
    if submit_conf < 0.0 or submit_conf > 1.0:
        return submit_conf, True
    return submit_conf, False


def compute_rewards(episode: Episode) -> Rewards:
    """Convert a terminated Episode into a frozen Rewards record."""
    _validate_episode_structure(episode)

    raw_confidence, clamped = _extract_confidence(episode)
    confidence_for_brier = raw_confidence
    if clamped and raw_confidence is not None:
        confidence_for_brier = max(0.0, min(1.0, raw_confidence))

    r1 = task_completion(episode)
    r2, r2_breakdown = _drift_detection_with_breakdown(episode)
    r3, r3_breakdown = _r3_with_breakdown(episode)
    r4, r4_breakdown = _r4_with_breakdown(episode)
    r5, r5_breakdown = _r5_with_breakdown(episode)

    if not (
        _is_finite(r1)
        and _is_finite(r2)
        and _is_finite(r3)
        and _is_finite(r4)
        and _is_finite(r5)
    ):
        raise RewardComputationError(
            "non-finite value in reward computation",
            episode.episode_id,
        )

    quality = combine_quality(r1, r2, r3, r4, r5)
    brier = brier_penalty(confidence_for_brier, r1)
    if not (_is_finite(quality) and _is_finite(brier)):
        raise RewardComputationError(
            "non-finite value in reward computation",
            episode.episode_id,
        )

    pre_floor = quality * (1.0 - brier)
    floored = apply_uncertain_floor(pre_floor, r1, confidence_for_brier)
    floor_applied = floored != pre_floor
    reward_clamped = max(0.0, min(1.0, floored))
    reward = round(reward_clamped, 3)

    breakdown: dict[str, Any] = {
        "r1": _r1_breakdown(episode),
        "r2": r2_breakdown,
        "r3": r3_breakdown,
        "r4": r4_breakdown,
        "anti_hack": r5_breakdown,
        "combination": {
            "quality_raw": quality,
            "brier": brier,
            "uncertain_floor_applied": floor_applied,
            "confidence_clamped": clamped,
            "confidence_missing": (
                episode.terminated_by == "SUBMIT" and raw_confidence is None
            ),
        },
    }

    return Rewards(
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        r5=r5,
        quality=quality,
        brier=brier,
        reward=reward,
        confidence=raw_confidence,
        floor_applied=floor_applied,
        breakdown=breakdown,
    )
