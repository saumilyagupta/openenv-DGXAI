"""Cell 05 — Mock vendor APIs.

Consolidated cell implementing five vendor submodules (airline, cab,
restaurant, hotel, payment) as namespaces on a single module. Every vendor
exposes: frozen ``*State`` dataclass, ``initial_state``, ``dispatch``,
``apply_schema_mutation``, ``describe_schema``, ``emit_side_channel_if_pending``,
and ``TOOLS`` tuple. Implements ``docs/modules/vendors.md`` §§2–8.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from cells.step_04_models import GoalSpec, ToolResult

if TYPE_CHECKING:
    from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownSchemaVersionError(ValueError):
    """Raised by a serializer when an unrecognised schema_version is passed."""


class UnknownMutationOperatorError(ValueError):
    """Raised by apply_schema_mutation when the operator key is not known."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_LATENCY_OK_LO, _LATENCY_OK_HI = 50, 400
_LATENCY_TIMEOUT_LO, _LATENCY_TIMEOUT_HI = 5000, 7000
_TIMEOUT_MASK = 0x7F  # 1-in-128 trigger rate


def _canonical_args_json(tool_args: Mapping[str, Any] | None) -> str:
    """Stable sorted whitespace-free JSON for hashing (vendors.md §3.1)."""

    return json.dumps(
        dict(tool_args or {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _stable_digest(*parts: Any) -> int:
    """Cross-process-stable 64-bit integer digest.

    Python's built-in ``hash()`` is PYTHONHASHSEED-randomized for strings, so
    it cannot be used for replay-stable determinism (vendors.md §3.1). We use
    blake2b truncated to 8 bytes instead.
    """

    blob = "||".join(repr(p) for p in parts).encode("utf-8")
    digest_bytes = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest_bytes, "big", signed=False)


def _is_timeout(episode_seed: int, tool_name: str, tool_args: Mapping[str, Any] | None) -> bool:
    """Deterministic 1/128 timeout trigger — vendors.md §3.1."""

    digest = _stable_digest(episode_seed, tool_name, _canonical_args_json(tool_args))
    return (digest & _TIMEOUT_MASK) == 0


def _seeded_uniform(episode_seed: int, tag: str, lo: int, hi: int) -> int:
    """Deterministic uniform int in ``[lo, hi]``. No wall clock."""

    h = _stable_digest(episode_seed, tag) & 0x7FFFFFFF
    span = hi - lo + 1
    return lo + (h % span)


def _make_id(domain: str, episode_seed: int, op: str, key: Any, records: Mapping[str, Any]) -> str:
    """Deterministic 4-hex ID with ``-R{retry}`` suffix on prefix collisions.

    ``records`` is scanned for prefix matches to derive the replay-stable
    retry counter (vendors.md §3.8).
    """

    prefix = f"{domain[:3].upper()}-{_stable_digest(episode_seed, op, key) & 0xFFFF:04X}"
    matches = sum(1 for existing_id in records if existing_id.startswith(prefix))
    if matches == 0:
        return prefix
    return f"{prefix}-R{matches + 1}"


def _integer_inr(value: Any) -> int:
    """Coerce to int, rejecting bools. Uses ``math.floor(x + 0.5)`` for rounding."""

    if isinstance(value, bool):
        raise TypeError("monetary fields must be int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(math.floor(value + 0.5))
    raise TypeError(f"non-numeric monetary value: {value!r}")


def _timeout_result(
    tool_name: str,
    episode_seed: int,
    schema_version: str,
) -> ToolResult:
    latency = _seeded_uniform(episode_seed, f"{tool_name}:timeout", _LATENCY_TIMEOUT_LO, _LATENCY_TIMEOUT_HI)
    return ToolResult(
        tool_name=tool_name,
        status="timeout",
        response={"error_code": "TIMEOUT", "hint": "retry with same args"},
        schema_version=schema_version,
        latency_ms=latency,
    )


def _ok_latency(episode_seed: int, tool_name: str) -> int:
    return _seeded_uniform(episode_seed, f"{tool_name}:ok", _LATENCY_OK_LO, _LATENCY_OK_HI)


def _normalize_items(items: list[dict[str, Any]]) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """Normalise restaurant items for idempotency keying (vendors.md §3.9)."""

    out: list[tuple[str, int, tuple[str, ...]]] = []
    for item in items:
        dish_id = str(item["dish_id"]).strip().lower()
        qty = int(item["qty"])
        mods_raw = item.get("modifiers", []) or []
        mods = tuple(sorted(str(m).strip().lower() for m in mods_raw))
        out.append((dish_id, qty, mods))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Airline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AirlinePolicy:
    booking_window_hours: int = 24
    required_book_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class AirlineTnC:
    baggage_cabin_kg: int = 7
    reschedule_fee_pct: int = 0


@dataclass(frozen=True)
class AirlinePricing:
    convenience_fee_inr: int = 0


@dataclass(frozen=True)
class AirlineState:
    schema_version: str
    bookings: dict[str, dict[str, Any]]
    flight_roster_cache: dict[str, tuple[dict[str, Any], ...]]
    policy: AirlinePolicy
    tnc: AirlineTnC
    pricing: AirlinePricing
    side_channel_notice: str | None


_AIRLINE_BASE_FLIGHTS: tuple[dict[str, Any], ...] = (
    {"flight_id": "6E-2345", "depart_hour": 18, "depart_min": 30, "base_price": 7200, "seats": 14},
    {"flight_id": "AI-501", "depart_hour": 20, "depart_min": 15, "base_price": 6800, "seats": 3},
    {"flight_id": "UK-878", "depart_hour": 9, "depart_min": 10, "base_price": 5200, "seats": 9},
    {"flight_id": "SG-102", "depart_hour": 14, "depart_min": 50, "base_price": 8400, "seats": 22},
)


def _airline_time_window(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late_night"


def _airline_search_flights(
    from_: str, to: str, date: str, episode_seed: int
) -> tuple[dict[str, Any], ...]:
    key = f"{from_}->{to}|{date}"
    h = _stable_digest(episode_seed, key) & 0xFFFF
    count = 3 + (h % 3)
    return _AIRLINE_BASE_FLIGHTS[:count]


def _airline_serialize_flight(flight: dict[str, Any], from_: str, to: str, date: str, version: str) -> dict[str, Any]:
    depart = f"{date}T{flight['depart_hour']:02d}:{flight['depart_min']:02d}:00+05:30"
    base: dict[str, Any] = {
        "flight_id": flight["flight_id"],
        "from": from_,
        "to": to,
        "depart": depart,
        "seats_left": int(flight["seats"]),
    }
    if version == "v1":
        base["price"] = int(flight["base_price"])
        base["currency"] = "INR"
    elif version in ("v2", "v3"):
        base["total_fare_inr"] = int(flight["base_price"])
    else:
        raise UnknownSchemaVersionError(version)
    return base


def airline_initial_state(episode_seed: int, goal: GoalSpec) -> AirlineState:
    _ = (episode_seed, goal)
    return AirlineState(
        schema_version="v1",
        bookings={},
        flight_roster_cache={},
        policy=AirlinePolicy(booking_window_hours=24, required_book_fields=()),
        tnc=AirlineTnC(),
        pricing=AirlinePricing(),
        side_channel_notice=None,
    )


def airline_search(
    vendor_state: AirlineState,
    schema_version: str,
    from_: str,
    to: str,
    date: str,
    max_price_inr: int | None = None,
    time_window: Literal["morning", "afternoon", "evening", "late_night"] | None = None,
    episode_seed: int = 0,
) -> ToolResult:
    flights = _airline_search_flights(from_, to, date, episode_seed)
    serialized: list[dict[str, Any]] = []
    for f in flights:
        if time_window is not None and _airline_time_window(f["depart_hour"]) != time_window:
            continue
        if max_price_inr is not None and int(f["base_price"]) > int(max_price_inr):
            continue
        serialized.append(_airline_serialize_flight(f, from_, to, date, schema_version))
    return ToolResult(
        tool_name="airline.search",
        status="ok",
        response={"results": serialized},
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "airline.search"),
    )


def _airline_book_impl(
    vendor_state: AirlineState,
    schema_version: str,
    payment_state: PaymentState,
    flight_id: str,
    payment_token: str,
    passenger_count: int | None,
    passenger_name: str | None,
    episode_seed: int,
    now_ist: datetime,
) -> tuple[ToolResult, AirlineState, PaymentState]:
    flight = next((f for f in _AIRLINE_BASE_FLIGHTS if f["flight_id"] == flight_id), None)
    if flight is None:
        return (
            ToolResult(
                tool_name="airline.book",
                status="schema_error",
                response={
                    "error_code": "MISSING_FIELD",
                    "field_name": "flight_id",
                    "hint": "unknown flight_id",
                },
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "airline.book"),
            ),
            vendor_state,
            payment_state,
        )

    if schema_version == "v3" and passenger_count is None:
        return (
            ToolResult(
                tool_name="airline.book",
                status="schema_error",
                response={
                    "error_code": "MISSING_PASSENGER_COUNT",
                    "hint": "v3 requires passenger_count on book",
                },
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "airline.book"),
            ),
            vendor_state,
            payment_state,
        )

    depart_date = now_ist.date().isoformat()
    depart_dt = now_ist.replace(
        hour=int(flight["depart_hour"]),
        minute=int(flight["depart_min"]),
        second=0,
        microsecond=0,
    )
    window_hours = int(vendor_state.policy.booking_window_hours)
    if (
        depart_dt - now_ist < timedelta(hours=window_hours)
        and depart_dt >= now_ist
        and window_hours < 24
        and now_ist.hour >= 14
    ):
        return (
                ToolResult(
                    tool_name="airline.book",
                    status="policy_error",
                    response={
                        "error_code": "BOOKING_WINDOW_CLOSED",
                        "hint": "same-day booking closed after 14:00 IST",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "airline.book"),
                ),
                vendor_state,
                payment_state,
            )

    idempotency_key = (flight_id, (passenger_name or "").strip().lower(), depart_date)
    for existing_id, record in vendor_state.bookings.items():
        existing_key = (
            record.get("flight_id"),
            str(record.get("passenger_name") or "").strip().lower(),
            record.get("depart_date"),
        )
        if existing_key == idempotency_key:
            return (
                ToolResult(
                    tool_name="airline.book",
                    status="policy_error",
                    response={
                        "error_code": "DUPLICATE_BOOKING",
                        "existing_id": existing_id,
                        "original_ts": str(record.get("created_at_ist", "")),
                        "hint": "identical booking already exists",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "airline.book"),
                ),
                vendor_state,
                payment_state,
            )

    amount = int(flight["base_price"])
    charge_result, new_payment_state = _payment_charge_internal(
        payment_state=payment_state,
        amount_inr=amount,
        payment_token=payment_token,
        mfa_code=None,
        episode_seed=episode_seed,
        order_ref=f"airline:{flight_id}:{depart_date}",
    )
    if charge_result.status != "ok":
        propagated = _propagate_payment_error(charge_result, "airline.book", schema_version, episode_seed)
        return propagated, vendor_state, payment_state

    booking_id = _make_id("airline", episode_seed, "book", (flight_id, passenger_name, depart_date), vendor_state.bookings)
    new_record: dict[str, Any] = {
        "booking_id": booking_id,
        "flight_id": flight_id,
        "depart": f"{depart_date}T{flight['depart_hour']:02d}:{flight['depart_min']:02d}:00+05:30",
        "depart_date": depart_date,
        "passenger_name": passenger_name,
        "seats_confirmed": int(passenger_count or 1),
        "payment_status": "captured",
        "created_at_ist": now_ist.isoformat(),
    }
    if schema_version == "v1":
        new_record["price"] = amount
    else:
        new_record["total_fare_inr"] = amount
    if schema_version == "v3":
        new_record["passenger_count"] = int(passenger_count or 1)

    new_bookings = {**vendor_state.bookings, booking_id: new_record}
    new_state = replace(vendor_state, bookings=new_bookings)
    response = {k: v for k, v in new_record.items() if k not in ("depart_date", "created_at_ist", "passenger_name")}
    return (
        ToolResult(
            tool_name="airline.book",
            status="ok",
            response=response,
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "airline.book"),
        ),
        new_state,
        new_payment_state,
    )


def airline_cancel(
    vendor_state: AirlineState,
    schema_version: str,
    booking_id: str,
    episode_seed: int = 0,
) -> tuple[ToolResult, AirlineState]:
    if booking_id not in vendor_state.bookings:
        return (
            ToolResult(
                tool_name="airline.cancel",
                status="policy_error",
                response={"error_code": "MISSING_FIELD", "hint": "booking_id not found"},
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "airline.cancel"),
            ),
            vendor_state,
        )
    new_bookings = {k: v for k, v in vendor_state.bookings.items() if k != booking_id}
    new_state = replace(vendor_state, bookings=new_bookings)
    return (
        ToolResult(
            tool_name="airline.cancel",
            status="ok",
            response={"booking_id": booking_id, "cancelled": True},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "airline.cancel"),
        ),
        new_state,
    )


def airline_get_booking(
    vendor_state: AirlineState,
    schema_version: str,
    booking_id: str,
    episode_seed: int = 0,
) -> ToolResult:
    record = vendor_state.bookings.get(booking_id)
    if record is None:
        return ToolResult(
            tool_name="airline.get_booking",
            status="schema_error",
            response={"error_code": "MISSING_FIELD", "field_name": "booking_id", "hint": "unknown booking_id"},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "airline.get_booking"),
        )
    payload = {k: v for k, v in record.items() if k not in ("depart_date", "created_at_ist", "passenger_name")}
    return ToolResult(
        tool_name="airline.get_booking",
        status="ok",
        response=payload,
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "airline.get_booking"),
    )


def airline_apply_schema_mutation(
    vendor_state: AirlineState, mutation: Mapping[str, Any]
) -> AirlineState:
    state = vendor_state
    next_version = state.schema_version
    policy = state.policy
    for op, payload in mutation.items():
        if op == "rename":
            if "price" in payload and payload["price"] == "total_fare_inr":
                next_version = "v2"
        elif op == "remove":
            fields = payload if isinstance(payload, list) else [payload]
            if "currency" in fields and next_version == "v1":
                next_version = "v2"
        elif op == "require_new_field":
            if isinstance(payload, dict) and "passenger_count" in payload:
                policy = replace(policy, required_book_fields=tuple(sorted(set(policy.required_book_fields) | {"passenger_count"})))
                next_version = "v3"
        elif op == "time_window_shrink":
            if isinstance(payload, dict) and "booking_window_hours" in payload:
                policy = replace(policy, booking_window_hours=int(payload["booking_window_hours"]))
        elif op == "change_type" or op == "tnc_text_swap":
            continue
        elif op == "side_channel_notice_append":
            state = replace(state, side_channel_notice=str(payload))
        elif op == "fee_append":
            if isinstance(payload, dict) and "convenience_fee_inr" in payload:
                state = replace(state, pricing=replace(state.pricing, convenience_fee_inr=int(payload["convenience_fee_inr"])))
        elif op == "pricing_restructure" or op in {"numeric_bump", "enum_expand", "policy_flag_flip", "auth_scope_bump", "token_version_bump"}:
            continue
        else:
            raise UnknownMutationOperatorError(op)
    return replace(state, schema_version=next_version, policy=policy)


def airline_describe_schema(vendor_state: AirlineState, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        fields = {
            "flight_id": "str",
            "from": "str",
            "to": "str",
            "depart": "str",
            "price": "int",
            "currency": "str",
            "seats_left": "int",
        }
        removed: list[str] = []
    elif schema_version == "v2":
        fields = {
            "flight_id": "str",
            "from": "str",
            "to": "str",
            "depart": "str",
            "total_fare_inr": "int",
            "seats_left": "int",
        }
        removed = ["price", "currency"]
    elif schema_version == "v3":
        fields = {
            "flight_id": "str",
            "from": "str",
            "to": "str",
            "depart": "str",
            "total_fare_inr": "int",
            "seats_left": "int",
            "passenger_count": "int",
        }
        removed = ["price", "currency"]
    else:
        raise UnknownSchemaVersionError(schema_version)
    return {"version": schema_version, "fields": fields, "removed_from_prior": removed}


def airline_emit_side_channel_if_pending(
    vendor_state: AirlineState,
) -> tuple[str | None, AirlineState]:
    if vendor_state.side_channel_notice is None:
        return None, vendor_state
    notice = vendor_state.side_channel_notice
    return notice, replace(vendor_state, side_channel_notice=None)


AIRLINE_TOOLS: tuple[str, ...] = (
    "airline.search",
    "airline.book",
    "airline.cancel",
    "airline.get_booking",
)


# ---------------------------------------------------------------------------
# Cab
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CabPolicy:
    vehicle_class_enum: tuple[str, ...] = ("mini", "sedan")
    mini_reject_school_hours: bool = False


@dataclass(frozen=True)
class CabPricing:
    base_per_km_inr: int = 12
    surge_factor_pct: int = 100
    toll_bundled: bool = True
    fare_breakdown: bool = False


@dataclass(frozen=True)
class CabTnC:
    cancel_fee_inr: int = 0


@dataclass(frozen=True)
class CabState:
    schema_version: str
    rides: dict[str, dict[str, Any]]
    policy: CabPolicy
    pricing: CabPricing
    tnc: CabTnC
    side_channel_notice: str | None


def cab_initial_state(episode_seed: int, goal: GoalSpec) -> CabState:
    _ = (episode_seed, goal)
    return CabState(
        schema_version="v1",
        rides={},
        policy=CabPolicy(),
        pricing=CabPricing(),
        tnc=CabTnC(),
        side_channel_notice=None,
    )


def _cab_fare(pickup: str, drop: str, vehicle_class: str, episode_seed: int) -> int:
    base = 80
    key_hash = _stable_digest(pickup.strip().lower(), drop.strip().lower(), episode_seed) & 0x3FF
    distance = 50 + (key_hash % 250)
    multipliers = {"mini": 100, "sedan": 130, "suv": 170, "infant_seat_sedan": 150}
    mul = multipliers.get(vehicle_class, 100)
    return int(base + (distance * mul) // 100)


def _cab_eta(pickup: str, episode_seed: int) -> int:
    return 3 + (_stable_digest(pickup.strip().lower(), episode_seed) & 0xF)


def _cab_serialize(
    pickup: str,
    drop: str,
    vehicle_class: str,
    fare: int,
    eta_min: int,
    schema_version: str,
    pricing: CabPricing,
) -> dict[str, Any]:
    if schema_version == "v1":
        return {
            "pickup": pickup,
            "drop": drop,
            "vehicle_class": vehicle_class,
            "fare_inr": int(fare),
            "eta_min": int(eta_min),
        }
    if schema_version == "v2":
        return {
            "pickup": pickup,
            "drop": drop,
            "vehicle_class": vehicle_class,
            "fare_inr": int(fare),
            "eta_min": int(eta_min),
        }
    if schema_version == "v3":
        base = int(fare * 75 // 100)
        surge = int(fare * 12 // 100)
        tolls = int(fare * 6 // 100)
        gst = int(fare - base - surge - tolls)
        breakdown = {"base": base, "surge": surge, "tolls": tolls, "gst": gst}
        total = base + surge + tolls + gst
        if total != int(fare):
            # Defensive self-check — adjust gst to preserve invariant
            breakdown["gst"] = int(fare) - base - surge - tolls
        return {
            "pickup": pickup,
            "drop": drop,
            "vehicle_class": vehicle_class,
            "fare_breakdown": breakdown,
            "total_inr": int(fare),
            "eta_min": int(eta_min),
        }
    raise UnknownSchemaVersionError(schema_version)


def cab_estimate(
    vendor_state: CabState,
    schema_version: str,
    pickup: str,
    drop: str,
    vehicle_class: str,
    pickup_time_ist: str,
    episode_seed: int = 0,
) -> ToolResult:
    if vehicle_class not in vendor_state.policy.vehicle_class_enum:
        return ToolResult(
            tool_name="cab.estimate",
            status="policy_error",
            response={
                "error_code": "VEHICLE_CLASS_UNAVAILABLE",
                "available": list(vendor_state.policy.vehicle_class_enum),
                "hint": "requested vehicle_class not in current enum",
            },
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "cab.estimate"),
        )
    fare = _cab_fare(pickup, drop, vehicle_class, episode_seed)
    eta = _cab_eta(pickup, episode_seed)
    payload = _cab_serialize(pickup, drop, vehicle_class, fare, eta, schema_version, vendor_state.pricing)
    return ToolResult(
        tool_name="cab.estimate",
        status="ok",
        response=payload,
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "cab.estimate"),
    )


def _cab_book_impl(
    vendor_state: CabState,
    schema_version: str,
    payment_state: PaymentState,
    pickup: str,
    drop: str,
    vehicle_class: str,
    pickup_time_ist: str,
    payment_token: str,
    episode_seed: int,
    now_ist: datetime,
) -> tuple[ToolResult, CabState, PaymentState]:
    if vehicle_class not in vendor_state.policy.vehicle_class_enum:
        return (
            ToolResult(
                tool_name="cab.book",
                status="policy_error",
                response={
                    "error_code": "VEHICLE_CLASS_UNAVAILABLE",
                    "available": list(vendor_state.policy.vehicle_class_enum),
                    "hint": "requested vehicle_class not in current enum",
                },
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "cab.book"),
            ),
            vendor_state,
            payment_state,
        )

    if (
        vendor_state.policy.mini_reject_school_hours
        and vehicle_class == "mini"
        and 7 <= now_ist.hour < 9
    ):
        return (
                ToolResult(
                    tool_name="cab.book",
                    status="policy_error",
                    response={
                        "error_code": "SCHOOL_HOURS_MINI_REJECTED",
                        "available": [v for v in vendor_state.policy.vehicle_class_enum if v != "mini"],
                        "hint": "mini rejected during 07:00-09:00 IST",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "cab.book"),
                ),
                vendor_state,
                payment_state,
            )

    idempotency_key = (
        pickup.strip().lower(),
        drop.strip().lower(),
        pickup_time_ist.strip(),
        vehicle_class,
    )
    for existing_id, record in vendor_state.rides.items():
        existing_key = (
            str(record.get("pickup") or "").strip().lower(),
            str(record.get("drop") or "").strip().lower(),
            str(record.get("pickup_time_ist") or "").strip(),
            record.get("vehicle_class"),
        )
        if existing_key == idempotency_key:
            return (
                ToolResult(
                    tool_name="cab.book",
                    status="policy_error",
                    response={
                        "error_code": "DUPLICATE_RIDE",
                        "existing_id": existing_id,
                        "original_ts": str(record.get("created_at_ist", "")),
                        "hint": "identical ride already booked",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "cab.book"),
                ),
                vendor_state,
                payment_state,
            )

    fare = _cab_fare(pickup, drop, vehicle_class, episode_seed)
    charge_result, new_payment_state = _payment_charge_internal(
        payment_state=payment_state,
        amount_inr=fare,
        payment_token=payment_token,
        mfa_code=None,
        episode_seed=episode_seed,
        order_ref=f"cab:{pickup}:{drop}:{pickup_time_ist}",
    )
    if charge_result.status != "ok":
        return (
            _propagate_payment_error(charge_result, "cab.book", schema_version, episode_seed),
            vendor_state,
            payment_state,
        )

    ride_id = _make_id("cab", episode_seed, "ride", idempotency_key, vendor_state.rides)
    eta = _cab_eta(pickup, episode_seed)
    serialized = _cab_serialize(pickup, drop, vehicle_class, fare, eta, schema_version, vendor_state.pricing)
    new_record: dict[str, Any] = {
        "ride_id": ride_id,
        **serialized,
        "pickup_time_ist": pickup_time_ist,
        "created_at_ist": now_ist.isoformat(),
        "payment_status": "captured",
    }
    new_rides = {**vendor_state.rides, ride_id: new_record}
    new_state = replace(vendor_state, rides=new_rides)
    response = {k: v for k, v in new_record.items() if k != "created_at_ist"}
    return (
        ToolResult(
            tool_name="cab.book",
            status="ok",
            response=response,
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "cab.book"),
        ),
        new_state,
        new_payment_state,
    )


def cab_cancel(
    vendor_state: CabState,
    schema_version: str,
    ride_id: str,
    episode_seed: int = 0,
) -> tuple[ToolResult, CabState]:
    if ride_id not in vendor_state.rides:
        return (
            ToolResult(
                tool_name="cab.cancel",
                status="policy_error",
                response={"error_code": "MISSING_FIELD", "hint": "ride_id not found"},
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "cab.cancel"),
            ),
            vendor_state,
        )
    new_rides = {k: v for k, v in vendor_state.rides.items() if k != ride_id}
    new_state = replace(vendor_state, rides=new_rides)
    return (
        ToolResult(
            tool_name="cab.cancel",
            status="ok",
            response={"ride_id": ride_id, "cancelled": True},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "cab.cancel"),
        ),
        new_state,
    )


def cab_apply_schema_mutation(
    vendor_state: CabState, mutation: Mapping[str, Any]
) -> CabState:
    state = vendor_state
    next_version = state.schema_version
    policy = state.policy
    pricing = state.pricing
    for op, payload in mutation.items():
        if op == "enum_expand":
            new_vals = payload.get("vehicle_class_enum", []) if isinstance(payload, dict) else []
            enum = tuple(dict.fromkeys([*policy.vehicle_class_enum, *new_vals]))
            policy = replace(policy, vehicle_class_enum=enum)
            if next_version == "v1":
                next_version = "v2"
        elif op == "policy_flag_flip":
            if isinstance(payload, dict) and "mini_reject_school_hours" in payload:
                policy = replace(policy, mini_reject_school_hours=bool(payload["mini_reject_school_hours"]))
            if next_version == "v1":
                next_version = "v2"
        elif op == "pricing_restructure":
            pricing = replace(pricing, fare_breakdown=True)
            if next_version in ("v1", "v2"):
                next_version = "v3"
        elif op == "fee_append":
            continue
        elif op == "side_channel_notice_append":
            state = replace(state, side_channel_notice=str(payload))
        elif op == "tnc_text_swap":
            if isinstance(payload, dict) and "cancel_fee_inr" in payload:
                state = replace(state, tnc=replace(state.tnc, cancel_fee_inr=int(payload["cancel_fee_inr"])))
        elif op in {"rename", "remove", "require_new_field", "change_type", "numeric_bump", "time_window_shrink", "auth_scope_bump", "token_version_bump"}:
            continue
        else:
            raise UnknownMutationOperatorError(op)
    return replace(state, schema_version=next_version, policy=policy, pricing=pricing)


def cab_describe_schema(vendor_state: CabState, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        fields = {
            "pickup": "str",
            "drop": "str",
            "vehicle_class": "str",
            "fare_inr": "int",
            "eta_min": "int",
        }
        removed: list[str] = []
    elif schema_version == "v2":
        fields = {
            "pickup": "str",
            "drop": "str",
            "vehicle_class": "str",
            "fare_inr": "int",
            "eta_min": "int",
        }
        removed = []
    elif schema_version == "v3":
        fields = {
            "pickup": "str",
            "drop": "str",
            "vehicle_class": "str",
            "fare_breakdown": "dict[str, int]",
            "total_inr": "int",
            "eta_min": "int",
        }
        removed = ["fare_inr"]
    else:
        raise UnknownSchemaVersionError(schema_version)
    return {"version": schema_version, "fields": fields, "removed_from_prior": removed}


def cab_emit_side_channel_if_pending(vendor_state: CabState) -> tuple[str | None, CabState]:
    if vendor_state.side_channel_notice is None:
        return None, vendor_state
    notice = vendor_state.side_channel_notice
    return notice, replace(vendor_state, side_channel_notice=None)


CAB_TOOLS: tuple[str, ...] = ("cab.estimate", "cab.book", "cab.cancel")


# ---------------------------------------------------------------------------
# Restaurant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestaurantPolicy:
    min_order_inr: int = 199
    require_modifiers: bool = False


@dataclass(frozen=True)
class RestaurantSemantics:
    veg_only_excludes_egg: bool = False


@dataclass(frozen=True)
class RestaurantTnC:
    refund_window_min: int = 10


@dataclass(frozen=True)
class RestaurantState:
    schema_version: str
    orders: dict[str, dict[str, Any]]
    menu_cache: dict[str, tuple[dict[str, Any], ...]]
    policy: RestaurantPolicy
    semantics: RestaurantSemantics
    tnc: RestaurantTnC
    side_channel_notice: str | None


_RESTAURANT_MENU: tuple[dict[str, Any], ...] = (
    {"restaurant_id": "BLR-BIR-0123", "city": "Bengaluru", "cuisine": "biryani",
     "dishes": (
         {"dish_id": "BIR-001", "name": "Chicken Biryani", "price": 220, "is_veg": False, "has_egg": False},
         {"dish_id": "BIR-002", "name": "Egg Biryani", "price": 180, "is_veg": True, "has_egg": True},
         {"dish_id": "BIR-003", "name": "Veg Biryani", "price": 160, "is_veg": True, "has_egg": False},
     )},
    {"restaurant_id": "BLR-SOU-0456", "city": "Bengaluru", "cuisine": "south_indian",
     "dishes": (
         {"dish_id": "DOS-001", "name": "Masala Dosa", "price": 120, "is_veg": True, "has_egg": False},
         {"dish_id": "DOS-002", "name": "Egg Dosa", "price": 140, "is_veg": True, "has_egg": True},
     )},
)


def restaurant_initial_state(episode_seed: int, goal: GoalSpec) -> RestaurantState:
    _ = (episode_seed, goal)
    return RestaurantState(
        schema_version="v1",
        orders={},
        menu_cache={},
        policy=RestaurantPolicy(min_order_inr=199),
        semantics=RestaurantSemantics(veg_only_excludes_egg=False),
        tnc=RestaurantTnC(),
        side_channel_notice=None,
    )


def restaurant_search(
    vendor_state: RestaurantState,
    schema_version: str,
    city: str,
    cuisine: str | None = None,
    veg_only: bool = False,
    max_price_inr: int | None = None,
    episode_seed: int = 0,
) -> ToolResult:
    results: list[dict[str, Any]] = []
    for rec in _RESTAURANT_MENU:
        if rec["city"].lower() != city.strip().lower():
            continue
        if cuisine is not None and rec["cuisine"] != cuisine:
            continue
        dishes = []
        for dish in rec["dishes"]:
            if veg_only and not dish["is_veg"]:
                continue
            if veg_only and vendor_state.semantics.veg_only_excludes_egg and dish["has_egg"]:
                continue
            if max_price_inr is not None and int(dish["price"]) > int(max_price_inr):
                continue
            dishes.append({"dish_id": dish["dish_id"], "name": dish["name"], "price": int(dish["price"])})
        if dishes:
            results.append({
                "restaurant_id": rec["restaurant_id"],
                "city": rec["city"],
                "cuisine": rec["cuisine"],
                "dishes": dishes,
            })
    return ToolResult(
        tool_name="restaurant.search",
        status="ok",
        response={"results": results},
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "restaurant.search"),
    )


def _restaurant_lookup_price(dish_id: str) -> int | None:
    for rec in _RESTAURANT_MENU:
        for dish in rec["dishes"]:
            if dish["dish_id"] == dish_id:
                return int(dish["price"])
    return None


def _restaurant_order_impl(
    vendor_state: RestaurantState,
    schema_version: str,
    payment_state: PaymentState,
    restaurant_id: str,
    items: list[dict[str, Any]],
    payment_token: str,
    episode_seed: int,
    now_ist: datetime,
) -> tuple[ToolResult, RestaurantState, PaymentState]:
    if schema_version == "v3" or vendor_state.policy.require_modifiers:
        for it in items:
            if "modifiers" not in it:
                return (
                    ToolResult(
                        tool_name="restaurant.order",
                        status="schema_error",
                        response={
                            "error_code": "INVALID_ITEMS_SHAPE",
                            "field_name": "items",
                            "hint": "v3 requires modifiers list on every item",
                        },
                        schema_version=schema_version,
                        latency_ms=_ok_latency(episode_seed, "restaurant.order"),
                    ),
                    vendor_state,
                    payment_state,
                )

    total = 0
    for it in items:
        price = _restaurant_lookup_price(str(it["dish_id"]))
        if price is None:
            return (
                ToolResult(
                    tool_name="restaurant.order",
                    status="schema_error",
                    response={
                        "error_code": "MISSING_FIELD",
                        "field_name": "dish_id",
                        "hint": "unknown dish_id",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "restaurant.order"),
                ),
                vendor_state,
                payment_state,
            )
        total += price * int(it["qty"])

    if total < int(vendor_state.policy.min_order_inr):
        return (
            ToolResult(
                tool_name="restaurant.order",
                status="policy_error",
                response={
                    "error_code": "MIN_ORDER_NOT_MET",
                    "min_order_inr": int(vendor_state.policy.min_order_inr),
                    "got_total_inr": int(total),
                    "hint": "order total below minimum",
                },
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "restaurant.order"),
            ),
            vendor_state,
            payment_state,
        )

    idempotency_key = (restaurant_id, _normalize_items(items))
    for existing_id, record in vendor_state.orders.items():
        existing_key = (
            record.get("restaurant_id"),
            _normalize_items(list(record.get("items") or [])),
        )
        if existing_key == idempotency_key:
            return (
                ToolResult(
                    tool_name="restaurant.order",
                    status="policy_error",
                    response={
                        "error_code": "DUPLICATE_ORDER",
                        "existing_id": existing_id,
                        "original_ts": str(record.get("created_at_ist", "")),
                        "hint": "identical order already placed",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "restaurant.order"),
                ),
                vendor_state,
                payment_state,
            )

    charge_result, new_payment_state = _payment_charge_internal(
        payment_state=payment_state,
        amount_inr=total,
        payment_token=payment_token,
        mfa_code=None,
        episode_seed=episode_seed,
        order_ref=f"restaurant:{restaurant_id}",
    )
    if charge_result.status != "ok":
        return (
            _propagate_payment_error(charge_result, "restaurant.order", schema_version, episode_seed),
            vendor_state,
            payment_state,
        )

    order_id = _make_id("restaurant", episode_seed, "order", idempotency_key, vendor_state.orders)
    record_items: list[dict[str, Any]] = []
    for it in items:
        entry: dict[str, Any] = {"dish_id": str(it["dish_id"]), "qty": int(it["qty"])}
        price = _restaurant_lookup_price(str(it["dish_id"]))
        entry["price"] = int(price) if price is not None else 0
        if "modifiers" in it:
            entry["modifiers"] = list(it["modifiers"])
        record_items.append(entry)
    record = {
        "order_id": order_id,
        "restaurant_id": restaurant_id,
        "items": record_items,
        "total": int(total),
        "eta_min": 30 + (_stable_digest(episode_seed, order_id) & 0x1F),
        "created_at_ist": now_ist.isoformat(),
        "payment_status": "captured",
    }
    new_orders = {**vendor_state.orders, order_id: record}
    new_state = replace(vendor_state, orders=new_orders)
    response = {k: v for k, v in record.items() if k != "created_at_ist"}
    return (
        ToolResult(
            tool_name="restaurant.order",
            status="ok",
            response=response,
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "restaurant.order"),
        ),
        new_state,
        new_payment_state,
    )


def restaurant_track(
    vendor_state: RestaurantState,
    schema_version: str,
    order_id: str,
    episode_seed: int = 0,
) -> ToolResult:
    record = vendor_state.orders.get(order_id)
    if record is None:
        return ToolResult(
            tool_name="restaurant.track",
            status="schema_error",
            response={"error_code": "MISSING_FIELD", "field_name": "order_id", "hint": "unknown order_id"},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "restaurant.track"),
        )
    items = []
    for it in record.get("items", []):
        entry = dict(it)
        if schema_version == "v3" and "modifiers" not in entry:
            entry["modifiers"] = []
        items.append(entry)
    payload = {
        "order_id": record["order_id"],
        "restaurant_id": record["restaurant_id"],
        "items": items,
        "total": int(record["total"]),
        "eta_min": int(record["eta_min"]),
        "status": "in_transit",
    }
    return ToolResult(
        tool_name="restaurant.track",
        status="ok",
        response=payload,
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "restaurant.track"),
    )


def restaurant_apply_schema_mutation(
    vendor_state: RestaurantState, mutation: Mapping[str, Any]
) -> RestaurantState:
    state = vendor_state
    next_version = state.schema_version
    policy = state.policy
    semantics = state.semantics
    for op, payload in mutation.items():
        if op == "numeric_bump":
            if isinstance(payload, dict) and "min_order_inr" in payload:
                policy = replace(policy, min_order_inr=int(payload["min_order_inr"]))
                if next_version == "v1":
                    next_version = "v2"
        elif op == "require_new_field":
            if isinstance(payload, dict) and "modifiers" in payload:
                policy = replace(policy, require_modifiers=True)
                if next_version in ("v1", "v2"):
                    next_version = "v3"
        elif op == "side_channel_notice_append":
            state = replace(state, side_channel_notice=str(payload))
            semantics = replace(semantics, veg_only_excludes_egg=True)
            if next_version in ("v1", "v2"):
                next_version = "v3"
        elif op == "change_type" or op in {"rename", "remove", "enum_expand", "policy_flag_flip", "time_window_shrink", "tnc_text_swap", "pricing_restructure", "fee_append", "auth_scope_bump", "token_version_bump"}:
            continue
        else:
            raise UnknownMutationOperatorError(op)
    return replace(state, schema_version=next_version, policy=policy, semantics=semantics)


def restaurant_describe_schema(vendor_state: RestaurantState, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        fields = {
            "restaurant_id": "str",
            "items": "list[dict]",
            "total": "int",
            "eta_min": "int",
            "min_order_inr": "int",
        }
        removed: list[str] = []
    elif schema_version == "v2":
        fields = {
            "restaurant_id": "str",
            "items": "list[dict]",
            "total": "int",
            "eta_min": "int",
            "min_order_inr": "int",
        }
        removed = []
    elif schema_version == "v3":
        fields = {
            "restaurant_id": "str",
            "items": "list[dict{dish_id,qty,modifiers}]",
            "total": "int",
            "eta_min": "int",
            "min_order_inr": "int",
        }
        removed = []
    else:
        raise UnknownSchemaVersionError(schema_version)
    return {"version": schema_version, "fields": fields, "removed_from_prior": removed}


def restaurant_emit_side_channel_if_pending(
    vendor_state: RestaurantState,
) -> tuple[str | None, RestaurantState]:
    if vendor_state.side_channel_notice is None:
        return None, vendor_state
    notice = vendor_state.side_channel_notice
    return notice, replace(vendor_state, side_channel_notice=None)


RESTAURANT_TOOLS: tuple[str, ...] = ("restaurant.search", "restaurant.order", "restaurant.track")


# ---------------------------------------------------------------------------
# Hotel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HotelPolicy:
    cancel_window_hours: int = 24
    gst_required_threshold_inr: int = 0  # 0 disables


@dataclass(frozen=True)
class HotelPricing:
    resort_fee_inr: int = 0


@dataclass(frozen=True)
class HotelTnC:
    early_checkin_fee_pct: int = 0


@dataclass(frozen=True)
class HotelState:
    schema_version: str
    bookings: dict[str, dict[str, Any]]
    inventory_cache: dict[str, tuple[dict[str, Any], ...]]
    policy: HotelPolicy
    pricing: HotelPricing
    tnc: HotelTnC
    side_channel_notice: str | None


_HOTEL_INVENTORY: tuple[dict[str, Any], ...] = (
    {"hotel_id": "GOA-BEACH-007", "city": "Goa", "nightly_rate": 3500, "rooms": 12},
    {"hotel_id": "GOA-RESORT-012", "city": "Goa", "nightly_rate": 4200, "rooms": 8},
    {"hotel_id": "BLR-TECH-001", "city": "Bengaluru", "nightly_rate": 2800, "rooms": 30},
    {"hotel_id": "HYD-PARK-022", "city": "Hyderabad", "nightly_rate": 1800, "rooms": 20},
)


def hotel_initial_state(episode_seed: int, goal: GoalSpec) -> HotelState:
    _ = (episode_seed, goal)
    return HotelState(
        schema_version="v1",
        bookings={},
        inventory_cache={},
        policy=HotelPolicy(cancel_window_hours=24, gst_required_threshold_inr=0),
        pricing=HotelPricing(resort_fee_inr=0),
        tnc=HotelTnC(),
        side_channel_notice=None,
    )


def _hotel_nights(checkin: str, checkout: str) -> int:
    ci = datetime.fromisoformat(checkin)
    co = datetime.fromisoformat(checkout)
    return max(1, (co.date() - ci.date()).days)


def _hotel_compute_total(rate: int, nights: int, resort_fee: int) -> int:
    subtotal = rate * nights + resort_fee * nights
    gst = (subtotal * 18) // 100
    return int(subtotal + gst)


def hotel_search(
    vendor_state: HotelState,
    schema_version: str,
    city: str,
    checkin: str,
    checkout: str,
    max_nightly_rate_inr: int | None = None,
    episode_seed: int = 0,
) -> ToolResult:
    nights = _hotel_nights(checkin, checkout)
    results: list[dict[str, Any]] = []
    for rec in _HOTEL_INVENTORY:
        if rec["city"].lower() != city.strip().lower():
            continue
        if max_nightly_rate_inr is not None and int(rec["nightly_rate"]) > int(max_nightly_rate_inr):
            continue
        total = _hotel_compute_total(int(rec["nightly_rate"]), nights, int(vendor_state.pricing.resort_fee_inr))
        results.append({
            "hotel_id": rec["hotel_id"],
            "city": rec["city"],
            "checkin": checkin,
            "checkout": checkout,
            "nightly_rate": int(rec["nightly_rate"]),
            "total_with_tax": int(total),
            "cancel_window_hours": int(vendor_state.policy.cancel_window_hours),
        })
    return ToolResult(
        tool_name="hotel.search",
        status="ok",
        response={"results": results},
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "hotel.search"),
    )


def _hotel_book_impl(
    vendor_state: HotelState,
    schema_version: str,
    payment_state: PaymentState,
    hotel_id: str,
    checkin: str,
    checkout: str,
    payment_token: str,
    gst_number: str | None,
    episode_seed: int,
    now_ist: datetime,
    primary_guest: str | None = None,
) -> tuple[ToolResult, HotelState, PaymentState]:
    rec = next((h for h in _HOTEL_INVENTORY if h["hotel_id"] == hotel_id), None)
    if rec is None:
        return (
            ToolResult(
                tool_name="hotel.book",
                status="schema_error",
                response={"error_code": "MISSING_FIELD", "field_name": "hotel_id", "hint": "unknown hotel"},
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "hotel.book"),
            ),
            vendor_state,
            payment_state,
        )

    nights = _hotel_nights(checkin, checkout)
    total = _hotel_compute_total(int(rec["nightly_rate"]), nights, int(vendor_state.pricing.resort_fee_inr))

    threshold = int(vendor_state.policy.gst_required_threshold_inr)
    if threshold > 0 and total > threshold and not gst_number:
        return (
            ToolResult(
                tool_name="hotel.book",
                status="schema_error",
                response={
                    "error_code": "MISSING_GST_NUMBER",
                    "gst_threshold_inr": threshold,
                    "computed_total_inr": int(total),
                    "hint": "provide gst_number for bookings above threshold",
                },
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "hotel.book"),
            ),
            vendor_state,
            payment_state,
        )

    idempotency_key = (
        hotel_id,
        checkin,
        checkout,
        (primary_guest or "").strip().lower(),
    )
    for existing_id, existing in vendor_state.bookings.items():
        existing_key = (
            existing.get("hotel_id"),
            existing.get("checkin"),
            existing.get("checkout"),
            str(existing.get("primary_guest") or "").strip().lower(),
        )
        if existing_key == idempotency_key:
            return (
                ToolResult(
                    tool_name="hotel.book",
                    status="policy_error",
                    response={
                        "error_code": "DUPLICATE_BOOKING",
                        "existing_id": existing_id,
                        "original_ts": str(existing.get("created_at_ist", "")),
                        "hint": "identical hotel booking already exists",
                    },
                    schema_version=schema_version,
                    latency_ms=_ok_latency(episode_seed, "hotel.book"),
                ),
                vendor_state,
                payment_state,
            )

    charge_result, new_payment_state = _payment_charge_internal(
        payment_state=payment_state,
        amount_inr=total,
        payment_token=payment_token,
        mfa_code=None,
        episode_seed=episode_seed,
        order_ref=f"hotel:{hotel_id}:{checkin}:{checkout}",
    )
    if charge_result.status != "ok":
        return (
            _propagate_payment_error(charge_result, "hotel.book", schema_version, episode_seed),
            vendor_state,
            payment_state,
        )

    booking_id = _make_id("hotel", episode_seed, "book", idempotency_key, vendor_state.bookings)
    record: dict[str, Any] = {
        "booking_id": booking_id,
        "hotel_id": hotel_id,
        "city": rec["city"],
        "checkin": checkin,
        "checkout": checkout,
        "nightly_rate": int(rec["nightly_rate"]),
        "total_with_tax": int(total),
        "cancel_window_hours": int(vendor_state.policy.cancel_window_hours),
        "primary_guest": primary_guest,
        "created_at_ist": now_ist.isoformat(),
        "payment_status": "captured",
    }
    if vendor_state.pricing.resort_fee_inr > 0:
        record["resort_fee_inr"] = int(vendor_state.pricing.resort_fee_inr)
    if gst_number:
        record["gst_number"] = gst_number
    new_bookings = {**vendor_state.bookings, booking_id: record}
    new_state = replace(vendor_state, bookings=new_bookings)
    response = {k: v for k, v in record.items() if k not in ("created_at_ist", "primary_guest")}
    return (
        ToolResult(
            tool_name="hotel.book",
            status="ok",
            response=response,
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "hotel.book"),
        ),
        new_state,
        new_payment_state,
    )


def hotel_cancel(
    vendor_state: HotelState,
    schema_version: str,
    booking_id: str,
    episode_seed: int = 0,
    now_ist: datetime | None = None,
) -> tuple[ToolResult, HotelState]:
    record = vendor_state.bookings.get(booking_id)
    if record is None:
        return (
            ToolResult(
                tool_name="hotel.cancel",
                status="policy_error",
                response={"error_code": "MISSING_FIELD", "hint": "booking not found"},
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "hotel.cancel"),
            ),
            vendor_state,
        )
    if now_ist is not None:
        try:
            checkin_dt = datetime.fromisoformat(record["checkin"]).replace(tzinfo=now_ist.tzinfo)
            window = timedelta(hours=int(vendor_state.policy.cancel_window_hours))
            if checkin_dt - now_ist < window:
                return (
                    ToolResult(
                        tool_name="hotel.cancel",
                        status="policy_error",
                        response={"error_code": "CANCEL_WINDOW_EXPIRED", "hint": "cancel window has passed"},
                        schema_version=schema_version,
                        latency_ms=_ok_latency(episode_seed, "hotel.cancel"),
                    ),
                    vendor_state,
                )
        except (ValueError, KeyError):
            pass
    new_bookings = {k: v for k, v in vendor_state.bookings.items() if k != booking_id}
    new_state = replace(vendor_state, bookings=new_bookings)
    return (
        ToolResult(
            tool_name="hotel.cancel",
            status="ok",
            response={"booking_id": booking_id, "cancelled": True},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "hotel.cancel"),
        ),
        new_state,
    )


def hotel_apply_schema_mutation(
    vendor_state: HotelState, mutation: Mapping[str, Any]
) -> HotelState:
    state = vendor_state
    next_version = state.schema_version
    policy = state.policy
    pricing = state.pricing
    tnc = state.tnc
    for op, payload in mutation.items():
        if op == "time_window_shrink":
            if isinstance(payload, dict) and "cancel_window_hours" in payload:
                policy = replace(policy, cancel_window_hours=int(payload["cancel_window_hours"]))
                if next_version == "v1":
                    next_version = "v2"
        elif op == "fee_append":
            if isinstance(payload, dict) and "resort_fee_inr" in payload:
                pricing = replace(pricing, resort_fee_inr=int(payload["resort_fee_inr"]))
                if next_version == "v1":
                    next_version = "v2"
        elif op == "require_new_field":
            if isinstance(payload, dict) and "gst_number" in payload:
                if policy.gst_required_threshold_inr == 0:
                    policy = replace(policy, gst_required_threshold_inr=7500)
                if next_version in ("v1", "v2"):
                    next_version = "v3"
        elif op == "policy_flag_flip":
            if isinstance(payload, dict) and "gst_required_threshold_inr" in payload:
                policy = replace(policy, gst_required_threshold_inr=int(payload["gst_required_threshold_inr"]))
                if next_version in ("v1", "v2"):
                    next_version = "v3"
        elif op == "tnc_text_swap":
            if isinstance(payload, dict) and "early_checkin_fee_pct" in payload:
                tnc = replace(tnc, early_checkin_fee_pct=int(payload["early_checkin_fee_pct"]))
        elif op == "side_channel_notice_append":
            state = replace(state, side_channel_notice=str(payload))
        elif op in {"rename", "remove", "change_type", "numeric_bump", "enum_expand", "pricing_restructure", "auth_scope_bump", "token_version_bump"}:
            continue
        else:
            raise UnknownMutationOperatorError(op)
    return replace(state, schema_version=next_version, policy=policy, pricing=pricing, tnc=tnc)


def hotel_describe_schema(vendor_state: HotelState, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        fields = {
            "hotel_id": "str",
            "city": "str",
            "checkin": "str",
            "checkout": "str",
            "nightly_rate": "int",
            "total_with_tax": "int",
            "cancel_window_hours": "int",
        }
        removed: list[str] = []
    elif schema_version == "v2":
        fields = {
            "hotel_id": "str",
            "city": "str",
            "checkin": "str",
            "checkout": "str",
            "nightly_rate": "int",
            "total_with_tax": "int",
            "cancel_window_hours": "int",
            "resort_fee_inr": "int",
        }
        removed = []
    elif schema_version == "v3":
        fields = {
            "hotel_id": "str",
            "city": "str",
            "checkin": "str",
            "checkout": "str",
            "nightly_rate": "int",
            "total_with_tax": "int",
            "cancel_window_hours": "int",
            "resort_fee_inr": "int",
            "gst_number": "str",
        }
        removed = []
    else:
        raise UnknownSchemaVersionError(schema_version)
    return {"version": schema_version, "fields": fields, "removed_from_prior": removed}


def hotel_emit_side_channel_if_pending(vendor_state: HotelState) -> tuple[str | None, HotelState]:
    if vendor_state.side_channel_notice is None:
        return None, vendor_state
    notice = vendor_state.side_channel_notice
    return notice, replace(vendor_state, side_channel_notice=None)


HOTEL_TOOLS: tuple[str, ...] = ("hotel.search", "hotel.book", "hotel.cancel")


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaymentState:
    schema_version: str
    charges: dict[str, dict[str, Any]]
    accepted_token_version: Literal["v1", "v2"]
    required_scope: str
    mfa_threshold_inr: int
    side_channel_notice: str | None


_VALID_TOKENS = {"token_v1", "token_v2"}


def payment_initial_state(episode_seed: int, goal: GoalSpec) -> PaymentState:
    _ = (episode_seed, goal)
    return PaymentState(
        schema_version="v1",
        charges={},
        accepted_token_version="v1",
        required_scope="payments:write:v1",
        mfa_threshold_inr=0,
        side_channel_notice=None,
    )


def _token_scope(token: str) -> str | None:
    if token == "token_v1":
        return "payments:write:v1"
    if token == "token_v2":
        return "payments:write:v2"
    return None


def _payment_charge_internal(
    payment_state: PaymentState,
    amount_inr: int,
    payment_token: str,
    mfa_code: str | None,
    episode_seed: int,
    order_ref: str,
) -> tuple[ToolResult, PaymentState]:
    """Pure subroutine invoked by primary-domain book/order handlers."""

    sv = payment_state.schema_version
    scope = _token_scope(payment_token)
    if scope is None:
        return (
            ToolResult(
                tool_name="payment.charge",
                status="auth_error",
                response={"error_code": "TOKEN_INVALID", "hint": "malformed payment_token"},
                schema_version=sv,
                latency_ms=_ok_latency(episode_seed, "payment.charge"),
            ),
            payment_state,
        )
    if payment_state.accepted_token_version == "v2" and payment_token == "token_v1":
        return (
            ToolResult(
                tool_name="payment.charge",
                status="auth_error",
                response={
                    "error_code": "AUTH_SCOPE_INSUFFICIENT",
                    "required_scope": payment_state.required_scope,
                    "hint": "request a v2 token",
                },
                schema_version=sv,
                latency_ms=_ok_latency(episode_seed, "payment.charge"),
            ),
            payment_state,
        )
    if payment_state.mfa_threshold_inr > 0 and int(amount_inr) > payment_state.mfa_threshold_inr and not mfa_code:
        return (
            ToolResult(
                tool_name="payment.charge",
                status="auth_error",
                response={
                    "error_code": "MFA_REQUIRED",
                    "mfa_threshold_inr": int(payment_state.mfa_threshold_inr),
                    "mfa_required": True,
                    "hint": "provide mfa_code for amounts above threshold",
                },
                schema_version=sv,
                latency_ms=_ok_latency(episode_seed, "payment.charge"),
            ),
            payment_state,
        )

    idempotency_key = (order_ref, int(amount_inr), scope)
    for existing_id, existing in payment_state.charges.items():
        existing_key = (
            existing.get("order_ref"),
            int(existing.get("amount_inr", -1)),
            existing.get("token_scope"),
        )
        if existing_key == idempotency_key:
            return (
                ToolResult(
                    tool_name="payment.charge",
                    status="policy_error",
                    response={
                        "error_code": "DUPLICATE_CHARGE",
                        "existing_id": existing_id,
                        "original_ts": str(existing.get("created_at_ist", "")),
                        "hint": "duplicate charge request",
                    },
                    schema_version=sv,
                    latency_ms=_ok_latency(episode_seed, "payment.charge"),
                ),
                payment_state,
            )

    charge_id = _make_id("payment", episode_seed, "charge", idempotency_key, payment_state.charges)
    record = {
        "charge_id": charge_id,
        "amount_inr": int(amount_inr),
        "order_ref": order_ref,
        "token_scope": scope,
        "status": "captured",
        "created_at_ist": "",
    }
    new_charges = {**payment_state.charges, charge_id: record}
    new_state = replace(payment_state, charges=new_charges)
    response = {k: v for k, v in record.items() if k != "created_at_ist"}
    return (
        ToolResult(
            tool_name="payment.charge",
            status="ok",
            response=response,
            schema_version=sv,
            latency_ms=_ok_latency(episode_seed, "payment.charge"),
        ),
        new_state,
    )


def payment_charge(
    vendor_state: PaymentState,
    schema_version: str,
    amount_inr: int,
    payment_token: str,
    mfa_code: str | None = None,
    episode_seed: int = 0,
    now_ist: datetime | None = None,
    order_ref: str | None = None,
) -> tuple[ToolResult, PaymentState]:
    _integer_inr(amount_inr)
    ref = order_ref or f"direct:{payment_token}:{amount_inr}"
    result, new_state = _payment_charge_internal(
        payment_state=vendor_state,
        amount_inr=int(amount_inr),
        payment_token=payment_token,
        mfa_code=mfa_code,
        episode_seed=episode_seed,
        order_ref=ref,
    )
    if result.status == "ok" and now_ist is not None:
        updated_record = {**new_state.charges[result.response["charge_id"]]}
        updated_record["created_at_ist"] = now_ist.isoformat()
        new_charges = {**new_state.charges, result.response["charge_id"]: updated_record}
        new_state = replace(new_state, charges=new_charges)
    return result, new_state


def payment_refund(
    vendor_state: PaymentState,
    schema_version: str,
    charge_id: str,
    amount_inr: int,
    episode_seed: int = 0,
) -> tuple[ToolResult, PaymentState]:
    _integer_inr(amount_inr)
    if charge_id not in vendor_state.charges:
        return (
            ToolResult(
                tool_name="payment.refund",
                status="policy_error",
                response={"error_code": "MISSING_FIELD", "hint": "charge_id not found"},
                schema_version=schema_version,
                latency_ms=_ok_latency(episode_seed, "payment.refund"),
            ),
            vendor_state,
        )
    refund_id = _make_id("payment", episode_seed, "refund", (charge_id, int(amount_inr)), vendor_state.charges)
    record = {
        "refund_id": refund_id,
        "charge_id": charge_id,
        "amount_inr": int(amount_inr),
        "order_ref": f"refund:{charge_id}",
        "token_scope": vendor_state.required_scope,
        "status": "refunded",
    }
    new_charges = {**vendor_state.charges, refund_id: record}
    new_state = replace(vendor_state, charges=new_charges)
    return (
        ToolResult(
            tool_name="payment.refund",
            status="ok",
            response=record,
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "payment.refund"),
        ),
        new_state,
    )


def payment_get_token(
    vendor_state: PaymentState,
    schema_version: str,
    requested_scope: str,
    episode_seed: int = 0,
) -> ToolResult:
    if requested_scope == "payments:write:v1":
        token = "token_v1"
    elif requested_scope == "payments:write:v2":
        token = "token_v2"
    else:
        return ToolResult(
            tool_name="payment.get_token",
            status="auth_error",
            response={"error_code": "TOKEN_INVALID", "hint": "unknown scope"},
            schema_version=schema_version,
            latency_ms=_ok_latency(episode_seed, "payment.get_token"),
        )
    return ToolResult(
        tool_name="payment.get_token",
        status="ok",
        response={"token": token, "scope": requested_scope},
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, "payment.get_token"),
    )


def payment_apply_schema_mutation(
    vendor_state: PaymentState, mutation: Mapping[str, Any]
) -> PaymentState:
    state = vendor_state
    next_version = state.schema_version
    for op, payload in mutation.items():
        if op == "auth_scope_bump":
            required = "payments:write:v2"
            if isinstance(payload, dict) and "required_scope" in payload:
                required = str(payload["required_scope"])
            state = replace(state, accepted_token_version="v2", required_scope=required)
            if next_version == "v1":
                next_version = "v2"
        elif op == "token_version_bump":
            state = replace(state, accepted_token_version="v2")
            if next_version == "v1":
                next_version = "v2"
        elif op == "policy_flag_flip":
            if isinstance(payload, dict) and "mfa_threshold_inr" in payload:
                state = replace(state, mfa_threshold_inr=int(payload["mfa_threshold_inr"]))
                if next_version in ("v1", "v2"):
                    next_version = "v3"
        elif op == "side_channel_notice_append":
            state = replace(state, side_channel_notice=str(payload))
        elif op in {"rename", "remove", "require_new_field", "change_type", "numeric_bump", "enum_expand", "time_window_shrink", "tnc_text_swap", "pricing_restructure", "fee_append"}:
            continue
        else:
            raise UnknownMutationOperatorError(op)
    return replace(state, schema_version=next_version)


def payment_describe_schema(vendor_state: PaymentState, schema_version: str) -> dict[str, Any]:
    fields = {"amount_inr": "int", "payment_token": "str"}
    removed: list[str] = []
    if schema_version == "v1":
        pass
    elif schema_version == "v2":
        fields["required_scope"] = "str"
    elif schema_version == "v3":
        fields["required_scope"] = "str"
        fields["mfa_code"] = "str"
    else:
        raise UnknownSchemaVersionError(schema_version)
    return {"version": schema_version, "fields": fields, "removed_from_prior": removed}


def payment_emit_side_channel_if_pending(
    vendor_state: PaymentState,
) -> tuple[str | None, PaymentState]:
    if vendor_state.side_channel_notice is None:
        return None, vendor_state
    notice = vendor_state.side_channel_notice
    return notice, replace(vendor_state, side_channel_notice=None)


PAYMENT_TOOLS: tuple[str, ...] = ("payment.charge", "payment.refund", "payment.get_token")


# ---------------------------------------------------------------------------
# Auth cascade propagation (payment → primary domain)
# ---------------------------------------------------------------------------


def _propagate_payment_error(
    charge_result: ToolResult,
    caller_tool: str,
    schema_version: str,
    episode_seed: int,
) -> ToolResult:
    response: dict[str, Any] = {"error_code": "PAYMENT_AUTH_FAILED"}
    if charge_result.status == "auth_error":
        inner = charge_result.response
        if "required_scope" in inner:
            response["required_scope"] = inner["required_scope"]
        if inner.get("mfa_required") or inner.get("error_code") == "MFA_REQUIRED":
            response["mfa_required"] = True
        response["hint"] = inner.get("hint", "payment auth failed")
        status: Literal["ok", "schema_error", "policy_error", "auth_error", "timeout"] = "auth_error"
    else:
        response = dict(charge_result.response)
        status = charge_result.status
    return ToolResult(
        tool_name=caller_tool,
        status=status,
        response=response,
        schema_version=schema_version,
        latency_ms=_ok_latency(episode_seed, caller_tool),
    )


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


TOOLS: tuple[str, ...] = (
    *AIRLINE_TOOLS,
    *CAB_TOOLS,
    *RESTAURANT_TOOLS,
    *HOTEL_TOOLS,
    *PAYMENT_TOOLS,
)


def _split_tool(tool_name: str) -> tuple[str, str]:
    if "." not in tool_name:
        raise ValueError(f"tool_name must be '<domain>.<verb>', got {tool_name!r}")
    domain, verb = tool_name.split(".", 1)
    return domain, verb


def airline_dispatch(
    tool_name: str,
    tool_args: Mapping[str, Any],
    vendor_state: AirlineState,
    schema_version: str,
    episode_seed: int,
    now_ist: datetime,
    payment_state: PaymentState | None = None,
) -> tuple[ToolResult, AirlineState, PaymentState | None]:
    if _is_timeout(episode_seed, tool_name, tool_args):
        return _timeout_result(tool_name, episode_seed, schema_version), vendor_state, payment_state

    if tool_name == "airline.search":
        result = airline_search(
            vendor_state=vendor_state,
            schema_version=schema_version,
            from_=str(tool_args.get("from", tool_args.get("from_", ""))),
            to=str(tool_args.get("to", "")),
            date=str(tool_args.get("date", "")),
            max_price_inr=tool_args.get("max_price_inr"),
            time_window=tool_args.get("time_window"),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    if tool_name == "airline.book":
        if payment_state is None:
            payment_state = payment_initial_state(episode_seed, _stub_goal())
        result, new_state, new_payment = _airline_book_impl(
            vendor_state=vendor_state,
            schema_version=schema_version,
            payment_state=payment_state,
            flight_id=str(tool_args.get("flight_id", "")),
            payment_token=str(tool_args.get("payment_token", "")),
            passenger_count=tool_args.get("passenger_count"),
            passenger_name=tool_args.get("passenger_name"),
            episode_seed=episode_seed,
            now_ist=now_ist,
        )
        return result, new_state, new_payment
    if tool_name == "airline.cancel":
        result, new_state = airline_cancel(
            vendor_state=vendor_state,
            schema_version=schema_version,
            booking_id=str(tool_args.get("booking_id", "")),
            episode_seed=episode_seed,
        )
        return result, new_state, payment_state
    if tool_name == "airline.get_booking":
        result = airline_get_booking(
            vendor_state=vendor_state,
            schema_version=schema_version,
            booking_id=str(tool_args.get("booking_id", "")),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    raise ValueError(f"unknown airline tool: {tool_name}")


def cab_dispatch(
    tool_name: str,
    tool_args: Mapping[str, Any],
    vendor_state: CabState,
    schema_version: str,
    episode_seed: int,
    now_ist: datetime,
    payment_state: PaymentState | None = None,
) -> tuple[ToolResult, CabState, PaymentState | None]:
    if _is_timeout(episode_seed, tool_name, tool_args):
        return _timeout_result(tool_name, episode_seed, schema_version), vendor_state, payment_state
    if tool_name == "cab.estimate":
        result = cab_estimate(
            vendor_state=vendor_state,
            schema_version=schema_version,
            pickup=str(tool_args.get("pickup", "")),
            drop=str(tool_args.get("drop", "")),
            vehicle_class=str(tool_args.get("vehicle_class", "mini")),
            pickup_time_ist=str(tool_args.get("pickup_time_ist", "")),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    if tool_name == "cab.book":
        if payment_state is None:
            payment_state = payment_initial_state(episode_seed, _stub_goal())
        result, new_state, new_payment = _cab_book_impl(
            vendor_state=vendor_state,
            schema_version=schema_version,
            payment_state=payment_state,
            pickup=str(tool_args.get("pickup", "")),
            drop=str(tool_args.get("drop", "")),
            vehicle_class=str(tool_args.get("vehicle_class", "mini")),
            pickup_time_ist=str(tool_args.get("pickup_time_ist", "")),
            payment_token=str(tool_args.get("payment_token", "")),
            episode_seed=episode_seed,
            now_ist=now_ist,
        )
        return result, new_state, new_payment
    if tool_name == "cab.cancel":
        result, new_state = cab_cancel(
            vendor_state=vendor_state,
            schema_version=schema_version,
            ride_id=str(tool_args.get("ride_id", "")),
            episode_seed=episode_seed,
        )
        return result, new_state, payment_state
    raise ValueError(f"unknown cab tool: {tool_name}")


def restaurant_dispatch(
    tool_name: str,
    tool_args: Mapping[str, Any],
    vendor_state: RestaurantState,
    schema_version: str,
    episode_seed: int,
    now_ist: datetime,
    payment_state: PaymentState | None = None,
) -> tuple[ToolResult, RestaurantState, PaymentState | None]:
    if _is_timeout(episode_seed, tool_name, tool_args):
        return _timeout_result(tool_name, episode_seed, schema_version), vendor_state, payment_state
    if tool_name == "restaurant.search":
        result = restaurant_search(
            vendor_state=vendor_state,
            schema_version=schema_version,
            city=str(tool_args.get("city", "")),
            cuisine=tool_args.get("cuisine"),
            veg_only=bool(tool_args.get("veg_only", False)),
            max_price_inr=tool_args.get("max_price_inr"),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    if tool_name == "restaurant.order":
        if payment_state is None:
            payment_state = payment_initial_state(episode_seed, _stub_goal())
        items = list(tool_args.get("items") or [])
        result, new_state, new_payment = _restaurant_order_impl(
            vendor_state=vendor_state,
            schema_version=schema_version,
            payment_state=payment_state,
            restaurant_id=str(tool_args.get("restaurant_id", "")),
            items=items,
            payment_token=str(tool_args.get("payment_token", "")),
            episode_seed=episode_seed,
            now_ist=now_ist,
        )
        return result, new_state, new_payment
    if tool_name == "restaurant.track":
        result = restaurant_track(
            vendor_state=vendor_state,
            schema_version=schema_version,
            order_id=str(tool_args.get("order_id", "")),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    raise ValueError(f"unknown restaurant tool: {tool_name}")


def hotel_dispatch(
    tool_name: str,
    tool_args: Mapping[str, Any],
    vendor_state: HotelState,
    schema_version: str,
    episode_seed: int,
    now_ist: datetime,
    payment_state: PaymentState | None = None,
) -> tuple[ToolResult, HotelState, PaymentState | None]:
    if _is_timeout(episode_seed, tool_name, tool_args):
        return _timeout_result(tool_name, episode_seed, schema_version), vendor_state, payment_state
    if tool_name == "hotel.search":
        result = hotel_search(
            vendor_state=vendor_state,
            schema_version=schema_version,
            city=str(tool_args.get("city", "")),
            checkin=str(tool_args.get("checkin", "")),
            checkout=str(tool_args.get("checkout", "")),
            max_nightly_rate_inr=tool_args.get("max_nightly_rate_inr"),
            episode_seed=episode_seed,
        )
        return result, vendor_state, payment_state
    if tool_name == "hotel.book":
        if payment_state is None:
            payment_state = payment_initial_state(episode_seed, _stub_goal())
        result, new_state, new_payment = _hotel_book_impl(
            vendor_state=vendor_state,
            schema_version=schema_version,
            payment_state=payment_state,
            hotel_id=str(tool_args.get("hotel_id", "")),
            checkin=str(tool_args.get("checkin", "")),
            checkout=str(tool_args.get("checkout", "")),
            payment_token=str(tool_args.get("payment_token", "")),
            gst_number=tool_args.get("gst_number"),
            episode_seed=episode_seed,
            now_ist=now_ist,
            primary_guest=tool_args.get("primary_guest"),
        )
        return result, new_state, new_payment
    if tool_name == "hotel.cancel":
        result, new_state = hotel_cancel(
            vendor_state=vendor_state,
            schema_version=schema_version,
            booking_id=str(tool_args.get("booking_id", "")),
            episode_seed=episode_seed,
            now_ist=now_ist,
        )
        return result, new_state, payment_state
    raise ValueError(f"unknown hotel tool: {tool_name}")


def payment_dispatch(
    tool_name: str,
    tool_args: Mapping[str, Any],
    vendor_state: PaymentState,
    schema_version: str,
    episode_seed: int,
    now_ist: datetime,
) -> tuple[ToolResult, PaymentState]:
    if _is_timeout(episode_seed, tool_name, tool_args):
        return _timeout_result(tool_name, episode_seed, schema_version), vendor_state
    if tool_name == "payment.charge":
        return payment_charge(
            vendor_state=vendor_state,
            schema_version=schema_version,
            amount_inr=int(tool_args.get("amount_inr", 0)),
            payment_token=str(tool_args.get("payment_token", "")),
            mfa_code=tool_args.get("mfa_code"),
            episode_seed=episode_seed,
            now_ist=now_ist,
            order_ref=tool_args.get("order_ref"),
        )
    if tool_name == "payment.refund":
        return payment_refund(
            vendor_state=vendor_state,
            schema_version=schema_version,
            charge_id=str(tool_args.get("charge_id", "")),
            amount_inr=int(tool_args.get("amount_inr", 0)),
            episode_seed=episode_seed,
        )
    if tool_name == "payment.get_token":
        result = payment_get_token(
            vendor_state=vendor_state,
            schema_version=schema_version,
            requested_scope=str(tool_args.get("requested_scope", "")),
            episode_seed=episode_seed,
        )
        return result, vendor_state
    raise ValueError(f"unknown payment tool: {tool_name}")


def _stub_goal() -> GoalSpec:
    return GoalSpec(
        domain="airline",
        intent="book_flight",
        slots={},
        constraints={},
        language="en",
        seed_utterance="",
    )


# ---------------------------------------------------------------------------
# Vendor namespace registry — exposes the per-domain "module" surface the
# spec calls for while keeping everything in a single cell.
# ---------------------------------------------------------------------------


airline = SimpleNamespace(
    initial_state=airline_initial_state,
    search=airline_search,
    cancel=airline_cancel,
    get_booking=airline_get_booking,
    apply_schema_mutation=airline_apply_schema_mutation,
    describe_schema=airline_describe_schema,
    emit_side_channel_if_pending=airline_emit_side_channel_if_pending,
    dispatch=airline_dispatch,
    TOOLS=AIRLINE_TOOLS,
)

cab = SimpleNamespace(
    initial_state=cab_initial_state,
    estimate=cab_estimate,
    cancel=cab_cancel,
    apply_schema_mutation=cab_apply_schema_mutation,
    describe_schema=cab_describe_schema,
    emit_side_channel_if_pending=cab_emit_side_channel_if_pending,
    dispatch=cab_dispatch,
    TOOLS=CAB_TOOLS,
)

restaurant = SimpleNamespace(
    initial_state=restaurant_initial_state,
    search=restaurant_search,
    track=restaurant_track,
    apply_schema_mutation=restaurant_apply_schema_mutation,
    describe_schema=restaurant_describe_schema,
    emit_side_channel_if_pending=restaurant_emit_side_channel_if_pending,
    dispatch=restaurant_dispatch,
    TOOLS=RESTAURANT_TOOLS,
)

hotel = SimpleNamespace(
    initial_state=hotel_initial_state,
    search=hotel_search,
    cancel=hotel_cancel,
    apply_schema_mutation=hotel_apply_schema_mutation,
    describe_schema=hotel_describe_schema,
    emit_side_channel_if_pending=hotel_emit_side_channel_if_pending,
    dispatch=hotel_dispatch,
    TOOLS=HOTEL_TOOLS,
)

payment = SimpleNamespace(
    initial_state=payment_initial_state,
    charge=payment_charge,
    refund=payment_refund,
    get_token=payment_get_token,
    apply_schema_mutation=payment_apply_schema_mutation,
    describe_schema=payment_describe_schema,
    emit_side_channel_if_pending=payment_emit_side_channel_if_pending,
    dispatch=payment_dispatch,
    TOOLS=PAYMENT_TOOLS,
)


VENDOR_REGISTRY: dict[str, SimpleNamespace] = {
    "airline": airline,
    "cab": cab,
    "restaurant": restaurant,
    "hotel": hotel,
    "payment": payment,
}


__all__ = [
    "AirlinePolicy",
    "AirlineTnC",
    "AirlinePricing",
    "AirlineState",
    "CabPolicy",
    "CabPricing",
    "CabTnC",
    "CabState",
    "RestaurantPolicy",
    "RestaurantSemantics",
    "RestaurantTnC",
    "RestaurantState",
    "HotelPolicy",
    "HotelPricing",
    "HotelTnC",
    "HotelState",
    "PaymentState",
    "UnknownSchemaVersionError",
    "UnknownMutationOperatorError",
    "TOOLS",
    "AIRLINE_TOOLS",
    "CAB_TOOLS",
    "RESTAURANT_TOOLS",
    "HOTEL_TOOLS",
    "PAYMENT_TOOLS",
    "VENDOR_REGISTRY",
    "airline",
    "cab",
    "restaurant",
    "hotel",
    "payment",
    "airline_initial_state",
    "airline_search",
    "airline_cancel",
    "airline_get_booking",
    "airline_apply_schema_mutation",
    "airline_describe_schema",
    "airline_emit_side_channel_if_pending",
    "airline_dispatch",
    "cab_initial_state",
    "cab_estimate",
    "cab_cancel",
    "cab_apply_schema_mutation",
    "cab_describe_schema",
    "cab_emit_side_channel_if_pending",
    "cab_dispatch",
    "restaurant_initial_state",
    "restaurant_search",
    "restaurant_track",
    "restaurant_apply_schema_mutation",
    "restaurant_describe_schema",
    "restaurant_emit_side_channel_if_pending",
    "restaurant_dispatch",
    "hotel_initial_state",
    "hotel_search",
    "hotel_cancel",
    "hotel_apply_schema_mutation",
    "hotel_describe_schema",
    "hotel_emit_side_channel_if_pending",
    "hotel_dispatch",
    "payment_initial_state",
    "payment_charge",
    "payment_refund",
    "payment_get_token",
    "payment_apply_schema_mutation",
    "payment_describe_schema",
    "payment_emit_side_channel_if_pending",
    "payment_dispatch",
]
