"""DriftCall drift injector.

Implements docs/modules/drift_injector.md. Public surface:

- build_schedule(stage, episode_seed, goal) -> tuple[DriftEvent, ...]
- apply_drift(state, event) -> DriftCallState
- list_patterns() -> tuple[DriftPattern, ...]

The 20-pattern catalogue is embedded as a module-level constant (one
source of truth; no YAML dependency at runtime). Patterns are keyed by
`pattern_id` per drift_injector.md §4.1.

Error taxonomy (drift_injector.md §5):

- ValueError                    — stage not in {1,2,3}
- UnknownDriftPatternError      — event.pattern_id not in registry
- DriftDomainMismatchError      — event.domain not in state.vendor_states
- DriftReapplicationError       — event already present in state.drift_fired
- DriftCatalogueError           — catalogue loads < 20 patterns (startup)
- DriftScheduleConflictError    — stage-3 schedule cannot be built within
                                  retry budget, or max_turns < 8 for stage 3
"""

from __future__ import annotations

import copy
import hashlib
import random
import struct
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

from cells.step_04_models import DriftCallState, DriftEvent, GoalSpec

DriftTypeLiteral = Literal["schema", "policy", "tnc", "pricing", "auth"]

__all__ = [
    "DriftCatalogueError",
    "DriftDomainMismatchError",
    "DriftPattern",
    "DriftReapplicationError",
    "DriftScheduleConflictError",
    "UnknownDriftPatternError",
    "apply_drift",
    "build_schedule",
    "list_patterns",
]


# ---------------------------------------------------------------------------
# Errors (drift_injector.md §5)
# ---------------------------------------------------------------------------


class UnknownDriftPatternError(Exception):
    """Raised when apply_drift receives a DriftEvent whose description is
    not a key in the pattern registry."""


class DriftDomainMismatchError(Exception):
    """Raised when the event's domain is not a key of state.vendor_states."""


class DriftReapplicationError(Exception):
    """Raised when apply_drift is called with an event already present in
    state.drift_fired. Defence-in-depth per spec §2."""


class DriftCatalogueError(Exception):
    """Raised at startup when the embedded catalogue contains fewer than
    20 patterns."""


class DriftScheduleConflictError(Exception):
    """Raised when build_schedule cannot produce a valid stage-3 schedule
    (max_turns too small, or retry budget exhausted)."""


# ---------------------------------------------------------------------------
# DriftPattern dataclass (drift_injector.md §4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftPattern:
    id: str
    drift_type: DriftTypeLiteral
    domain: str
    from_version: str
    to_version: str
    description: str
    mutation: Mapping[str, Any]
    detection_hints: tuple[str, ...]

    def __post_init__(self) -> None:
        # Wrap mutation in MappingProxyType for immutability without mutating
        # a frozen instance — use object.__setattr__ (frozen-safe per stdlib).
        if not isinstance(self.mutation, MappingProxyType):
            object.__setattr__(self, "mutation", MappingProxyType(dict(self.mutation)))


# ---------------------------------------------------------------------------
# 20-pattern catalogue (drift_injector.md §4.4, byte-identical to DESIGN.md §6.3)
# ---------------------------------------------------------------------------


_CATALOGUE_RAW: tuple[dict[str, Any], ...] = (
    # Schema (5)
    {
        "id": "airline.price_rename",
        "drift_type": "schema",
        "domain": "airline",
        "from_version": "v1",
        "to_version": "v2",
        "description": "field 'price' renamed to 'total_fare_inr'; 'currency' removed",
        "mutation": {
            "rename": {"price": "total_fare_inr"},
            "remove": ["currency"],
        },
        "detection_hints": ("total_fare_inr", "price", "rename"),
    },
    {
        "id": "airline.pax_required",
        "drift_type": "schema",
        "domain": "airline",
        "from_version": "v2",
        "to_version": "v3",
        "description": "booking now requires 'passenger_count' field",
        "mutation": {
            "require_new_field": ["passenger_count"],
        },
        "detection_hints": ("passenger_count", "MISSING_PASSENGER_COUNT"),
    },
    {
        "id": "cab.fare_breakdown",
        "drift_type": "schema",
        "domain": "cab",
        "from_version": "v2",
        "to_version": "v3",
        "description": "'fare_inr' replaced by nested 'fare_breakdown' object",
        "mutation": {
            "change_type": {"fare_inr": "fare_breakdown"},
            "require_new_field": ["fare_breakdown"],
            "remove": ["fare_inr"],
        },
        "detection_hints": ("fare_breakdown", "base", "surge", "tolls", "gst"),
    },
    {
        "id": "restaurant.items_shape_bump",
        "drift_type": "schema",
        "domain": "restaurant",
        "from_version": "v1",
        "to_version": "v2",
        "description": "items[] entries now require a 'modifiers' array",
        "mutation": {
            "require_new_field": ["modifiers"],
        },
        "detection_hints": ("modifiers", "items", "require"),
    },
    {
        "id": "hotel.gst_field",
        "drift_type": "schema",
        "domain": "hotel",
        "from_version": "v2",
        "to_version": "v3",
        "description": "hotel.book requires 'gst_number' when total > 7500",
        "mutation": {
            "require_new_field": ["gst_number"],
        },
        "detection_hints": ("gst_number", "gst", "7500"),
    },
    # Policy (5)
    {
        "id": "airline.booking_window_shrink",
        "drift_type": "policy",
        "domain": "airline",
        "from_version": "v1",
        "to_version": "v2",
        "description": "same-day bookings rejected after 14:00 IST",
        "mutation": {
            "time_window_shrink": {"same_day_cutoff": "14:00"},
            "policy_flag_flip": {"same_day_allowed": False},
        },
        "detection_hints": ("14:00", "same-day", "policy_error", "booking_window"),
    },
    {
        "id": "cab.school_hours_mini_reject",
        "drift_type": "policy",
        "domain": "cab",
        "from_version": "v1",
        "to_version": "v2",
        "description": "vehicle_class=mini rejected during 07:00-09:00 IST",
        "mutation": {
            "time_window_shrink": {"mini_blackout": ["07:00", "09:00"]},
            "policy_flag_flip": {"mini_school_hours": False},
        },
        "detection_hints": ("mini", "07:00", "09:00", "policy_error", "school"),
    },
    {
        "id": "restaurant.min_order_bump",
        "drift_type": "policy",
        "domain": "restaurant",
        "from_version": "v1",
        "to_version": "v2",
        "description": "minimum order raised from 199 to 299 INR",
        "mutation": {
            "numeric_bump": {"min_order_inr": {"from": 199, "to": 299}},
        },
        "detection_hints": ("299", "199", "min_order", "minimum"),
    },
    {
        "id": "hotel.cancel_window_shrink",
        "drift_type": "policy",
        "domain": "hotel",
        "from_version": "v1",
        "to_version": "v2",
        "description": "free cancellation window shrunk 24h to 6h",
        "mutation": {
            "numeric_bump": {"cancel_window_hours": {"from": 24, "to": 6}},
        },
        "detection_hints": ("6h", "24h", "cancel_window", "cancel"),
    },
    {
        "id": "cab.vehicle_class_expand",
        "drift_type": "policy",
        "domain": "cab",
        "from_version": "v1",
        "to_version": "v2",
        "description": "vehicle_class enum expanded with suv and infant_seat_sedan",
        "mutation": {
            "enum_expand": {"vehicle_class": ["suv", "infant_seat_sedan"]},
        },
        "detection_hints": ("suv", "infant_seat_sedan", "vehicle_class"),
    },
    # T&C (5)
    {
        "id": "airline.baggage_tnc_rewrite",
        "drift_type": "tnc",
        "domain": "airline",
        "from_version": "v1",
        "to_version": "v2",
        "description": "cabin baggage allowance reduced from 7kg to 5kg",
        "mutation": {
            "tnc_text_swap": {
                "from": "free cabin baggage 7kg",
                "to": "free cabin baggage 5kg",
            },
            "side_channel_notice_append": "baggage_allowance_change_7_to_5",
        },
        "detection_hints": ("5kg", "7kg", "baggage", "cabin"),
    },
    {
        "id": "cab.surge_policy_tnc",
        "drift_type": "tnc",
        "domain": "cab",
        "from_version": "v1",
        "to_version": "v2",
        "description": "surge may apply retroactively if ride extended",
        "mutation": {
            "tnc_text_swap": {
                "from": "surge fixed at booking",
                "to": "surge applies retroactively on extension",
            },
            "side_channel_notice_append": "surge_retroactive_notice",
        },
        "detection_hints": ("surge", "retroactive", "extend", "tnc"),
    },
    {
        "id": "restaurant.veg_filter_semantic",
        "drift_type": "tnc",
        "domain": "restaurant",
        "from_version": "v2",
        "to_version": "v3",
        "description": "veg_only=True now excludes egg dishes (was included)",
        "mutation": {
            "tnc_text_swap": {
                "from": "veg_only includes egg",
                "to": "veg_only excludes egg",
            },
            "side_channel_notice_append": "veg_only_egg_exclusion",
        },
        "detection_hints": ("veg_only", "egg", "exclude"),
    },
    {
        "id": "hotel.early_checkin_tnc",
        "drift_type": "tnc",
        "domain": "hotel",
        "from_version": "v1",
        "to_version": "v2",
        "description": "early check-in before 12:00 billed at 50% of nightly rate",
        "mutation": {
            "tnc_text_swap": {
                "from": "early check-in free subject to availability",
                "to": "early check-in billed 50% of nightly rate",
            },
            "side_channel_notice_append": "early_checkin_billed",
        },
        "detection_hints": ("early", "check-in", "50%", "12:00"),
    },
    {
        "id": "airline.reschedule_tnc",
        "drift_type": "tnc",
        "domain": "airline",
        "from_version": "v2",
        "to_version": "v3",
        "description": "reschedule fee previously waived is now 10% of fare",
        "mutation": {
            "tnc_text_swap": {
                "from": "reschedule waived",
                "to": "reschedule fee 10% of fare",
            },
            "side_channel_notice_append": "reschedule_fee_10pct",
        },
        "detection_hints": ("reschedule", "10%", "fare", "fee"),
    },
    # Pricing (3)
    {
        "id": "airline.convenience_fee_append",
        "drift_type": "pricing",
        "domain": "airline",
        "from_version": "v2",
        "to_version": "v3",
        "description": "hidden INR 199 convenience fee added at booking",
        "mutation": {
            "fee_append": {"convenience_fee_inr": 199},
            "pricing_restructure": {"hidden_fees": True},
        },
        "detection_hints": ("199", "convenience_fee", "fee", "hidden"),
    },
    {
        "id": "cab.toll_unbundle",
        "drift_type": "pricing",
        "domain": "cab",
        "from_version": "v2",
        "to_version": "v3",
        "description": "tolls previously included, now separate line item at booking",
        "mutation": {
            "fee_append": {"tolls_inr": 0},
            "pricing_restructure": {"toll_unbundled": True},
        },
        "detection_hints": ("toll", "tolls", "unbundle", "line item"),
    },
    {
        "id": "hotel.resort_fee_append",
        "drift_type": "pricing",
        "domain": "hotel",
        "from_version": "v2",
        "to_version": "v3",
        "description": "resort fee of INR 500 per night added at booking",
        "mutation": {
            "fee_append": {"resort_fee_inr": 500},
            "pricing_restructure": {"resort_fee_hidden": True},
        },
        "detection_hints": ("resort_fee", "500", "per night", "resort"),
    },
    # Auth (2, transversal on payment)
    {
        "id": "payment.auth_scope_upgrade",
        "drift_type": "auth",
        "domain": "payment",
        "from_version": "v1",
        "to_version": "v2",
        "description": "token_v1 401s; token_v2 with scope=payments:write:v2 required",
        "mutation": {
            "auth_scope_bump": {"required_scope": "payments:write:v2"},
            "token_version_bump": {"from": "token_v1", "to": "token_v2"},
        },
        "detection_hints": ("token_v2", "payments:write:v2", "scope", "401", "auth"),
    },
    {
        "id": "payment.mfa_required",
        "drift_type": "auth",
        "domain": "payment",
        "from_version": "v2",
        "to_version": "v3",
        "description": "transactions above INR 5000 require mfa_code in payload",
        "mutation": {
            "auth_scope_bump": {"required_field": "mfa_code"},
            "token_version_bump": {"threshold_inr": 5000},
        },
        "detection_hints": ("mfa_code", "mfa_required", "5000", "mfa"),
    },
)


def _load_catalogue() -> tuple[DriftPattern, ...]:
    patterns = tuple(
        DriftPattern(
            id=entry["id"],
            drift_type=entry["drift_type"],
            domain=entry["domain"],
            from_version=entry["from_version"],
            to_version=entry["to_version"],
            description=entry["description"],
            mutation=entry["mutation"],
            detection_hints=tuple(entry["detection_hints"]),
        )
        for entry in _CATALOGUE_RAW
    )
    if len(patterns) < 20:
        raise DriftCatalogueError(
            f"expected 20 patterns in catalogue, got {len(patterns)}",
        )
    # Sort by id for stable ordering (spec §2 list_patterns contract).
    return tuple(sorted(patterns, key=lambda p: p.id))


_PATTERNS: tuple[DriftPattern, ...] = _load_catalogue()
_PATTERNS_BY_ID: dict[str, DriftPattern] = {p.id: p for p in _PATTERNS}
_PATTERNS_BY_DOMAIN: dict[str, tuple[DriftPattern, ...]] = {}
for _p in _PATTERNS:
    _PATTERNS_BY_DOMAIN.setdefault(_p.domain, ())
    _PATTERNS_BY_DOMAIN[_p.domain] = (*_PATTERNS_BY_DOMAIN[_p.domain], _p)


def list_patterns() -> tuple[DriftPattern, ...]:
    """Return all 20 registered drift patterns, sorted by id."""
    return _PATTERNS


# ---------------------------------------------------------------------------
# Deterministic RNG helpers (drift_injector.md §3.3)
# ---------------------------------------------------------------------------


def _derive_seed(stage: int, episode_seed: int, domain: str) -> int:
    """Blake2b-based seed derivation — hash-stable across PYTHONHASHSEED."""
    payload = f"drift|{stage}|{episode_seed}|{domain}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    (seed,) = struct.unpack("<Q", digest)
    return int(seed)


# ---------------------------------------------------------------------------
# Schedule construction (drift_injector.md §2, §3.2, §7)
# ---------------------------------------------------------------------------


_DEFAULT_MAX_TURNS: int = 16


def _pick_pattern_for_domain(
    rng: random.Random,
    domain: str,
    exclude_ids: frozenset[str],
) -> DriftPattern | None:
    pool = tuple(
        p for p in _PATTERNS_BY_DOMAIN.get(domain, ()) if p.id not in exclude_ids
    )
    if not pool:
        return None
    return rng.choice(pool)


def _event_from_pattern(pattern: DriftPattern, turn: int) -> DriftEvent:
    return DriftEvent(
        turn=turn,
        drift_type=pattern.drift_type,
        domain=pattern.domain,
        description=pattern.description,
        from_version=pattern.from_version,
        to_version=pattern.to_version,
        pattern_id=pattern.id,
    )


def build_schedule(
    stage: int,
    episode_seed: int,
    goal: GoalSpec,
    *,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> tuple[DriftEvent, ...]:
    """Build the drift schedule for an episode. See drift_injector.md §2."""
    if stage not in (1, 2, 3):
        raise ValueError(f"unknown stage: {stage!r} (expected 1, 2, or 3)")

    if stage == 1:
        return ()

    rng = random.Random(_derive_seed(stage, episode_seed, goal.domain))
    lo = 2
    hi = max_turns - 3
    if hi < lo:
        raise DriftScheduleConflictError(
            f"max_turns={max_turns} too small for any drift placement",
        )

    first_pattern = _pick_pattern_for_domain(rng, goal.domain, frozenset())
    if first_pattern is None:
        # Fallback: goal.domain has no pattern; pick any.
        first_pattern = rng.choice(_PATTERNS)

    if stage == 2:
        turn = rng.randint(lo, hi)
        return (_event_from_pattern(first_pattern, turn),)

    # stage == 3 — need two drifts, distance >= 2, different pattern_ids.
    if max_turns < 8:
        raise DriftScheduleConflictError(
            f"max_turns={max_turns} too small for stage-3 schedule (need >= 8)",
        )

    # first_turn must leave room for second_turn >= first_turn + 2 within [lo, hi].
    first_hi_by_window = max_turns // 2
    first_hi = min(first_hi_by_window, hi - 2)
    if first_hi < lo:
        raise DriftScheduleConflictError(
            f"max_turns={max_turns} leaves no room for stage-3 first drift",
        )
    first_turn = rng.randint(lo, first_hi)

    second_lo = first_turn + 2
    if second_lo > hi:
        raise DriftScheduleConflictError(
            f"max_turns={max_turns} leaves no room for stage-3 second drift",
        )
    second_turn = rng.randint(second_lo, hi)

    # Second-drift domain: 80% same as goal.domain, 20% payment cross-domain.
    cross_domain_roll = rng.random()
    second_domain = "payment" if cross_domain_roll < 0.20 else goal.domain

    second_pattern: DriftPattern | None = None
    for _attempt in range(5):
        candidate = _pick_pattern_for_domain(
            rng,
            second_domain,
            frozenset({first_pattern.id}),
        )
        if candidate is not None:
            second_pattern = candidate
            break
        # Swap domain on miss (e.g., if same-domain pool is already exhausted).
        second_domain = "payment" if second_domain == goal.domain else goal.domain

    if second_pattern is None:
        # Last resort: any pattern in catalogue other than first.
        remaining = tuple(p for p in _PATTERNS if p.id != first_pattern.id)
        if not remaining:
            raise DriftScheduleConflictError(
                "unable to build stage-3 schedule: no distinct second pattern",
            )
        second_pattern = rng.choice(remaining)

    return (
        _event_from_pattern(first_pattern, first_turn),
        _event_from_pattern(second_pattern, second_turn),
    )


# ---------------------------------------------------------------------------
# Mutation dispatch (drift_injector.md §3.4)
# ---------------------------------------------------------------------------


def _apply_rename(target: dict[str, Any], rename_map: Mapping[str, str]) -> None:
    for old_key, new_key in rename_map.items():
        if old_key in target:
            target[new_key] = target.pop(old_key)
        else:
            target.setdefault(new_key, None)


def _apply_remove(target: dict[str, Any], remove_keys: list[str]) -> None:
    for key in remove_keys:
        target.pop(key, None)


def _apply_require_new_field(target: dict[str, Any], fields: list[str]) -> None:
    existing = target.setdefault("required_fields", [])
    if isinstance(existing, list):
        for f in fields:
            if f not in existing:
                existing.append(f)


def _apply_change_type(target: dict[str, Any], types_map: Mapping[str, str]) -> None:
    bucket = target.setdefault("type_changes", {})
    if isinstance(bucket, dict):
        bucket.update({k: v for k, v in types_map.items()})


def _apply_enum_expand(target: dict[str, Any], enum_map: Mapping[str, list[str]]) -> None:
    for enum_name, additions in enum_map.items():
        current = target.setdefault(enum_name, [])
        if isinstance(current, list):
            for v in additions:
                if v not in current:
                    current.append(v)


def _apply_numeric_bump(target: dict[str, Any], bumps: Mapping[str, Mapping[str, Any]]) -> None:
    for key, change in bumps.items():
        if "to" in change:
            target[key] = change["to"]


def _apply_policy_flag_flip(target: dict[str, Any], flags: Mapping[str, bool]) -> None:
    flag_bucket = target.setdefault("flags", {})
    if isinstance(flag_bucket, dict):
        for k, v in flags.items():
            flag_bucket[k] = v


def _apply_time_window_shrink(target: dict[str, Any], windows: Mapping[str, Any]) -> None:
    bucket = target.setdefault("time_windows", {})
    if isinstance(bucket, dict):
        for k, v in windows.items():
            bucket[k] = v


def _apply_tnc_text_swap(target: dict[str, Any], swap: Mapping[str, str]) -> None:
    target["tnc_text"] = swap.get("to", target.get("tnc_text"))


def _apply_side_channel_notice(target: dict[str, Any], notice: str) -> None:
    notices = target.setdefault("side_channel", [])
    if isinstance(notices, list):
        notices.append(notice)


def _apply_pricing_restructure(target: dict[str, Any], change: Mapping[str, Any]) -> None:
    bucket = target.setdefault("pricing_flags", {})
    if isinstance(bucket, dict):
        for k, v in change.items():
            bucket[k] = v


def _apply_fee_append(target: dict[str, Any], fees: Mapping[str, Any]) -> None:
    bucket = target.setdefault("fees", {})
    if isinstance(bucket, dict):
        for k, v in fees.items():
            bucket[k] = v


def _apply_auth_scope_bump(target: dict[str, Any], scope: Mapping[str, Any]) -> None:
    bucket = target.setdefault("auth", {})
    if isinstance(bucket, dict):
        for k, v in scope.items():
            bucket[k] = v


def _apply_token_version_bump(target: dict[str, Any], bump: Mapping[str, Any]) -> None:
    bucket = target.setdefault("auth", {})
    if isinstance(bucket, dict):
        for k, v in bump.items():
            bucket[k] = v


_OPERATOR_DISPATCH: dict[str, Any] = {
    "rename": _apply_rename,
    "remove": _apply_remove,
    "require_new_field": _apply_require_new_field,
    "change_type": _apply_change_type,
    "enum_expand": _apply_enum_expand,
    "numeric_bump": _apply_numeric_bump,
    "policy_flag_flip": _apply_policy_flag_flip,
    "time_window_shrink": _apply_time_window_shrink,
    "tnc_text_swap": _apply_tnc_text_swap,
    "side_channel_notice_append": _apply_side_channel_notice,
    "pricing_restructure": _apply_pricing_restructure,
    "fee_append": _apply_fee_append,
    "auth_scope_bump": _apply_auth_scope_bump,
    "token_version_bump": _apply_token_version_bump,
}


def _mutate_vendor_state(
    vendor_state: dict[str, Any],
    pattern: DriftPattern,
) -> dict[str, Any]:
    """Return a mutated deep copy of the vendor state for the given pattern.
    Pure with respect to inputs (input dict is not modified)."""
    mutated = copy.deepcopy(vendor_state)
    for op_key, op_payload in pattern.mutation.items():
        handler = _OPERATOR_DISPATCH.get(op_key)
        if handler is None:
            # Unknown operator keys are tolerated as no-ops so catalogue
            # extensions don't break existing callers.
            continue
        handler(mutated, op_payload)
    return mutated


# ---------------------------------------------------------------------------
# apply_drift (drift_injector.md §2, §3.5)
# ---------------------------------------------------------------------------


def apply_drift(state: DriftCallState, event: DriftEvent) -> DriftCallState:
    """Apply a drift event to immutable state; return a new DriftCallState."""
    pattern = _PATTERNS_BY_ID.get(event.pattern_id)
    if pattern is None:
        raise UnknownDriftPatternError(
            f"no pattern registered for pattern_id: {event.pattern_id!r}",
        )
    if event.domain not in state.vendor_states:
        raise DriftDomainMismatchError(
            f"event.domain={event.domain!r} not in state.vendor_states",
        )
    if event in state.drift_fired:
        raise DriftReapplicationError(
            f"event already in drift_fired: {event!r}",
        )

    # Build new vendor_states dict with mutated copy for event.domain.
    new_vendor_states: dict[str, dict[str, Any]] = {
        k: copy.deepcopy(v) for k, v in state.vendor_states.items()
    }
    new_vendor_states[event.domain] = _mutate_vendor_state(
        state.vendor_states[event.domain],
        pattern,
    )

    new_schema_versions = dict(state.schema_versions)
    new_schema_versions[event.domain] = event.to_version

    new_drift_fired = state.drift_fired + (event,)

    return replace(
        state,
        vendor_states=new_vendor_states,
        schema_versions=new_schema_versions,
        drift_fired=new_drift_fired,
    )


