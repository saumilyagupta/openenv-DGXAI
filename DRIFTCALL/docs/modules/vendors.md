# vendors.md — Mock Vendor API Subsystem

**Module root:** `driftcall/vendors/` (6 files: `base.py`, `airline.py`, `cab.py`, `restaurant.py`, `hotel.py`, `payment.py`)
**Owner:** Person A (Environment)
**Implements:** DESIGN.md §5 (all subsections §5.1–§5.5), §4.3 (tool dispatch in `step()`), §6 (as mutation target via `drift_injector`), §7.1 R1 / R3 (constraint checking inputs)
**Status:** Design spec — pre-critic-gate

---

## 1. Purpose

The vendor subsystem is the five pure-Python **mock consumer APIs** that DriftCall's agent interacts with every time it emits a `TOOL_CALL` action. They are the *world* that drifts: their schemas, policies, T&Cs, pricing, and (transversally) auth tokens mutate mid-episode under the drift injector's control, and every behavioral signal the agent learns from flows out of their `ToolResult` returns.

The subsystem serves four consumers:

1. **`env.step`** (DESIGN.md §4.3) — dispatches each `TOOL_CALL` to a vendor tool handler and receives a `(ToolResult, VendorState)` pair (from `models.md` §4.3). The env installs the returned `VendorState` into `DriftCallState.vendor_states[domain]` whether or not it differs from the input — the returned-state contract is uniform.
2. **`drift_injector`** (drift_injector.md §3.4, §6) — writes to `DriftCallState.vendor_states[domain]` via the `apply_schema_mutation` helper; vendors read from the same dict using whichever `schema_versions[domain]` is currently installed.
3. **`rewards`** (DESIGN.md §7.1) — consumes `ToolResult` history embedded in the episode trail to compute R1 (task completion — was a booking created?), R3 (constraint adherence — did the booking satisfy budget/time/dietary?), and R4 (format compliance — was the `response` shape legal?).
4. **`audio/asr_whisper`** and `audio/tts_kokoro` — are *independent* of the vendor layer. Audio converts user utterances ↔ text at the env boundary (DESIGN.md §9); vendors never touch audio.

**Mutation authority.** Two and only two code paths commit changes to a `VendorState`:

- **Drift path** — `drift_injector.apply_drift` → vendor-module `apply_schema_mutation(state, mutation) → state'`. Schemas, policies, T&Cs, pricing, auth shift under this path.
- **Commit path** — `dispatch(...) → (ToolResult, state')`. Write tools (`*.book`, `*.order`, `*.cancel`, `payment.charge`, `payment.refund`, `payment.get_token`) commit new records into `state'.bookings / .orders / .rides / .charges / .accepted_token_version`. Non-write tools (`*.search`, `*.estimate`, `*.get_booking`, `*.track`) return the same state by identity (`returned_state is input_state`).

Both paths are pure functions: new frozen `VendorState` out, never in-place mutation. Any code that does `state.bookings[k] = v` or `state.charges[k] = v` (dict mutation on a field of a frozen `VendorState`) is a contract violation and is caught by the frozen-dataclass equality tests in `tests/test_vendors.py`.

Every tool handler is a **deterministic pure function** given `(vendor_state, tool_args, seed)`. A seeded `random.Random` (sourced from `DriftCallState.episode_id` hash) drives any synthetic listing (flight rosters, restaurant menus, hotel inventory) so reset replay is bit-identical.

Cites DESIGN.md §5.1 (airline), §5.2 (cab), §5.3 (restaurant), §5.4 (hotel), §5.5 (payment), and §6 drift patterns #1–#20 as enumerated in drift_injector.md §4.4.

---

## 2. Interface

Every signature below is the exact target. Additions require a DESIGN.md update first. All tool handlers return a `ToolResult` (from `models.py`, see models.md §4.3).

### 2.1 Common per-vendor module structure

Every vendor module (`airline.py`, `cab.py`, `restaurant.py`, `hotel.py`, `payment.py`) exports exactly this surface:

```python
from __future__ import annotations

# Public tool dispatch — called by env.step
def dispatch(
    tool_name: str,                          # e.g. "airline.search" (fully-qualified)
    tool_args: dict[str, Any],
    vendor_state: VendorState,               # frozen dataclass, per-vendor
    schema_version: str,                     # "v1" | "v2" | "v3"
    episode_seed: int,
    now_ist: datetime,                       # env-owned clock, IST-tzaware (episode-constant, see §3.5)
) -> tuple[ToolResult, VendorState]: ...
    # Returns (result, new_state).
    # For non-write ops (*.search, *.estimate, *.get_booking, *.track): new_state IS vendor_state
    # (identity equality: `returned_state is input_state`).
    # For write ops (*.book, *.order, *.cancel, payment.charge/refund/get_token): new_state is a fresh
    # frozen VendorState built via dataclasses.replace(old, records={**old.records, new_id: record}).
    # In-place mutation of any dict/list field on vendor_state is a contract violation.

# State bootstrap — called by env.reset() once per episode
def initial_state(
    episode_seed: int,
    goal: GoalSpec,                          # only read for domain-local hints
) -> VendorState: ...

# Drift mutation helper — called ONLY by drift_injector.apply_drift
def apply_schema_mutation(
    vendor_state: VendorState,
    mutation: Mapping[str, Any],             # operator-keyed dict, per drift_injector §3.4
) -> VendorState: ...

# Introspection (for PROBE_SCHEMA action)
def describe_schema(
    vendor_state: VendorState,
    schema_version: str,
) -> dict[str, Any]: ...

# Side-channel notice emission — called ONCE per env.step BEFORE dispatch
def emit_side_channel_if_pending(
    vendor_state: VendorState,
) -> tuple[str | None, VendorState]: ...
    # Returns (notice_string_or_None, new_state_with_cleared_channel).
    # Purpose: consumed-on-read pattern for one-shot T&C / pricing / auth
    # notices placed on vendor_state.side_channel_notice by drift_injector's
    # side_channel_notice_append operator (drift_injector.md §3.4).
    # Semantics: one-shot — if side_channel_notice is set, returns it and
    # returns a new VendorState with side_channel_notice = None (the clear
    # is VENDOR-INTERNAL; no drift-injector operator is involved). If nothing
    # pending, returns (None, vendor_state) unchanged.
    # env.step calls this once per step before dispatch and attaches the
    # returned notice to the next ToolResult.response["_notice"] surface.

# Tool catalogue — static tuple, used to populate DriftCallObservation.available_tools
TOOLS: tuple[str, ...]
```

`VendorState` is a per-module frozen dataclass (§4 below). `dispatch` is a pure function: same inputs → same `(ToolResult, VendorState)` pair, no hidden randomness beyond the seeded RNG derived from `episode_seed + tool_name + tool_args`. **`dispatch`'s signature is fixed: `(...) → tuple[ToolResult, VendorState]`.** Commit-path transitions (a new booking/order/ride/charge record) flow through `dispatch`'s returned state; drift-path transitions flow through `apply_schema_mutation`; initial construction flows through `initial_state`; side-channel consumption flows through `emit_side_channel_if_pending`. All four are pure functions returning a new frozen `VendorState`.

**Mechanical rule for implementers.** Every dict-field update uses `{**old_dict, key: value}` to construct a new dict, and `dataclasses.replace(state, field=new_dict)` builds the new frozen state. For example, committing a booking:

```python
new_bookings = {**vendor_state.bookings, booking_id: record}
new_state = dataclasses.replace(vendor_state, bookings=new_bookings)
return ToolResult(...), new_state
```

In-place mutation of any state dict (`vendor_state.bookings[k] = v`) is a contract violation and will be caught by the frozen-equality test in `tests/test_vendors.py` (which asserts `id(returned_state.bookings) != id(input_state.bookings)` whenever a write occurred, and `returned_state is input_state` whenever a read occurred).

### 2.2 Airline (`driftcall/vendors/airline.py`)

Implements DESIGN.md §5.1.

```python
TOOLS: tuple[str, ...] = (
    "airline.search",
    "airline.book",
    "airline.cancel",
    "airline.get_booking",
)

def airline_search(
    vendor_state: AirlineState,
    schema_version: str,
    from_: str, to: str, date: str,
    max_price_inr: int | None = None,
    time_window: Literal["morning", "afternoon", "evening", "late_night"] | None = None,
    episode_seed: int = 0,
) -> ToolResult: ...

def airline_book(
    vendor_state: AirlineState,
    schema_version: str,
    flight_id: str,
    payment_token: str,
    passenger_count: int | None = None,       # required in v3
    passenger_name: str | None = None,
    episode_seed: int = 0,
) -> ToolResult: ...

def airline_cancel(
    vendor_state: AirlineState,
    schema_version: str,
    booking_id: str,
    episode_seed: int = 0,
) -> ToolResult: ...

def airline_get_booking(
    vendor_state: AirlineState,
    schema_version: str,
    booking_id: str,
) -> ToolResult: ...
```

### 2.3 Cab (`driftcall/vendors/cab.py`)

Implements DESIGN.md §5.2.

```python
TOOLS: tuple[str, ...] = (
    "cab.estimate",
    "cab.book",
    "cab.cancel",
)

def cab_estimate(
    vendor_state: CabState,
    schema_version: str,
    pickup: str, drop: str,
    vehicle_class: Literal["mini", "sedan", "suv", "infant_seat_sedan"],
    pickup_time_ist: str,
    episode_seed: int = 0,
) -> ToolResult: ...

def cab_book(
    vendor_state: CabState,
    schema_version: str,
    pickup: str, drop: str,
    vehicle_class: str,
    pickup_time_ist: str,
    payment_token: str,
    episode_seed: int = 0,
) -> ToolResult: ...

def cab_cancel(
    vendor_state: CabState,
    schema_version: str,
    ride_id: str,
    episode_seed: int = 0,
) -> ToolResult: ...
```

### 2.4 Restaurant (`driftcall/vendors/restaurant.py`)

Implements DESIGN.md §5.3.

```python
TOOLS: tuple[str, ...] = (
    "restaurant.search",
    "restaurant.order",
    "restaurant.track",
)

def restaurant_search(
    vendor_state: RestaurantState,
    schema_version: str,
    city: str,
    cuisine: str | None = None,
    veg_only: bool = False,
    max_price_inr: int | None = None,
    episode_seed: int = 0,
) -> ToolResult: ...

def restaurant_order(
    vendor_state: RestaurantState,
    schema_version: str,
    restaurant_id: str,
    items: list[dict[str, Any]],              # [{dish_id, qty, modifiers?: [...]}]
    payment_token: str,
    episode_seed: int = 0,
) -> ToolResult: ...

def restaurant_track(
    vendor_state: RestaurantState,
    schema_version: str,
    order_id: str,
) -> ToolResult: ...
```

### 2.5 Hotel (`driftcall/vendors/hotel.py`)

Implements DESIGN.md §5.4.

```python
TOOLS: tuple[str, ...] = (
    "hotel.search",
    "hotel.book",
    "hotel.cancel",
)

def hotel_search(
    vendor_state: HotelState,
    schema_version: str,
    city: str,
    checkin: str, checkout: str,
    max_nightly_rate_inr: int | None = None,
    episode_seed: int = 0,
) -> ToolResult: ...

def hotel_book(
    vendor_state: HotelState,
    schema_version: str,
    hotel_id: str,
    checkin: str, checkout: str,
    payment_token: str,
    gst_number: str | None = None,            # required in v3 if total > 7500
    episode_seed: int = 0,
) -> ToolResult: ...

def hotel_cancel(
    vendor_state: HotelState,
    schema_version: str,
    booking_id: str,
    episode_seed: int = 0,
) -> ToolResult: ...
```

### 2.6 Payment (`driftcall/vendors/payment.py`)

Implements DESIGN.md §5.5 — **transversal**; every primary domain's `*_book` / `*_order` handler calls `payment.charge` under the hood.

```python
TOOLS: tuple[str, ...] = (
    "payment.charge",
    "payment.refund",
    "payment.get_token",
)

def payment_charge(
    vendor_state: PaymentState,
    schema_version: str,
    amount_inr: int,                          # integer INR only
    payment_token: str,
    mfa_code: str | None = None,              # required in v3 if amount > 5000
    episode_seed: int = 0,
) -> ToolResult: ...

def payment_refund(
    vendor_state: PaymentState,
    schema_version: str,
    charge_id: str,
    amount_inr: int,
    episode_seed: int = 0,
) -> ToolResult: ...

def payment_get_token(
    vendor_state: PaymentState,
    schema_version: str,
    requested_scope: str,                     # "payments:write:v1" | "payments:write:v2"
    episode_seed: int = 0,
) -> ToolResult: ...
```

---

## 3. Behavior Spec

### 3.1 Determinism

Every vendor handler is a pure function of `(vendor_state, schema_version, tool_args, episode_seed, now_ist)`. The `episode_seed` deterministically produces:

- Flight roster at `airline.search` (seeded `random.Random(episode_seed ^ hash(("airline.search", from_, to, date)))` — 3 to 8 flights).
- Restaurant listings at `restaurant.search` (same pattern).
- Hotel inventory at `hotel.search` (same pattern).
- `cab.estimate` fare (deterministic function of `(pickup, drop, vehicle_class, episode_seed)` — no noise).
- `latency_ms` (seeded uniform sample 50–400 for `ok`, 5000–7000 for `timeout`).

**Timeout trigger is deterministic.** A dispatch returns `status="timeout"` iff:

```python
def _canonical_args_json(tool_args: dict[str, Any]) -> str:
    # Stable, sorted, whitespace-free JSON rep of tool_args.
    return json.dumps(tool_args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

is_timeout = (hash((episode_seed, tool_name, _canonical_args_json(tool_args))) & 0x7F) == 0
```

This gives ~0.78% rate (1 in 128 dispatches), is bit-identical across replays, and uses the same formula in every vendor. No vendor ever instantiates `random.Random()` for timeout selection — timeouts flow solely from this hash-bit check. The seeded latency RNG is used only to sample the numeric `latency_ms` value after the timeout branch is chosen.

Two `dispatch()` calls with identical inputs MUST return equal `(ToolResult, VendorState)` pairs (`ToolResult` equal by `==`; `VendorState` identical by `is` if no commit occurred, else structurally equal by `==`). No global RNG, no wall-clock reads (the env injects `now_ist`), no environment variables.

### 3.2 Schema versioning

Each vendor exposes exactly three schema versions — `v1`, `v2`, `v3`. Version transitions are **clean**: the drift injector mutates `vendor_state` via `apply_schema_mutation`, and the *next* tool call reads the mutated state through whichever schema-version branch applies. There is no rollback.

Every handler branches on `schema_version` at its top:

```python
if schema_version == "v1":   return _serialize_v1(...)
elif schema_version == "v2": return _serialize_v2(...)
elif schema_version == "v3": return _serialize_v3(...)
else: raise UnknownSchemaVersionError(schema_version)
```

Transition semantics:

- **v1 → v2** (one drift pattern per domain) — field renames, required-field additions, enum expansions, policy bumps.
- **v2 → v3** (one drift pattern per domain, where applicable) — second-order changes; typically adds another required field or semantic shift.
- Versions advance monotonically within an episode (drift injector never decrements).
- Stage-3 episodes can chain `v1 → v2 → v3` on the same domain within 16 turns (drift_injector §7 E7).

### 3.3 ToolResult construction contract

Every vendor handler returns a `ToolResult` with exactly these semantics:

| Field | Rule |
|---|---|
| `tool_name` | Echoes the fully-qualified tool name (e.g. `"airline.search"`). MUST match the dispatched name. |
| `status` | Exactly one of `"ok"`, `"schema_error"`, `"policy_error"`, `"auth_error"`, `"timeout"` — enumerated in §5. |
| `response` | JSON-roundtrip-safe dict. No `set`, no `bytes`, no `tuple`-as-value, no `datetime` objects (serialize to ISO-8601 strings), no non-primitive custom classes. On non-ok status, MUST include `error_code: str` key. |
| `schema_version` | Current `schema_versions[domain]` at call time. Echoes even on error. |
| `latency_ms` | Seeded sample; `≥ 0`. |

The JSON-roundtrip invariant is enforced at test time by `tests/test_vendors.py` running `json.loads(json.dumps(result.response))` on every return path (models.md §5 covers the surface).

### 3.4 Monetary semantics

**All amounts are integer INR.** No floats. `budget_inr`, `fare_inr`, `total_with_tax`, `amount_inr` — everywhere `int`. The cab v3 `fare_breakdown` sub-fields (`base`, `surge`, `tolls`, `gst`) are each integer INR and MUST sum to the `total_inr` top-level field. Rounding (when derived from deterministic ratios — e.g. GST at 18% — uses `int(round(x))` with banker's rounding explicitly avoided via Python's `math.floor(x + 0.5)` for reproducibility across platforms).

### 3.5 Clock & IST timezone

`now_ist` is **episode-constant**. It is set exactly once at `env.reset()` from a deterministic formula and carried in `DriftCallState` for the life of the episode:

```python
# Pinned at env construction (process start): today's date in Asia/Kolkata, truncated to date.
BASE_DATE_IST: datetime = datetime.now(ZoneInfo("Asia/Kolkata")).replace(hour=0, minute=0, second=0, microsecond=0)

# At env.reset(seed=episode_seed):
offset_seconds = (episode_seed * 37) % 86400
now_ist = (BASE_DATE_IST + timedelta(seconds=offset_seconds)).replace(second=0, microsecond=0)
# → pinned, minute-truncated, IST-tzaware
```

Properties:

- **Deterministic per seed.** Two episodes with the same `episode_seed` produce the same `now_ist`, regardless of wall-clock replay time (within the same env-process lifetime; `BASE_DATE_IST` is pinned at process start).
- **Constant across the episode.** Every `dispatch` call in episode `N` receives the same `now_ist` — no advancement per turn, no policy flips from clock drift.
- **Minute-truncated.** Seconds and microseconds zeroed, so policy boundaries (14:00, 07:00, 09:00, 12:00) are unambiguous.

Vendors use `now_ist` for:

- `airline.booking_window_shrink` policy check (same-day after 14:00 IST).
- `cab.school_hours_mini_reject` policy (07:00–09:00 IST).
- `hotel.early_checkin_tnc` (check-in before 12:00 IST).

**Vendors MUST NOT call `datetime.now()`, `time.time()`, or any wall-clock reader.** The sole time source is the `now_ist` parameter threaded from `env.step`. This is enforced by an AST-grep test in `tests/test_vendors.py` that rejects any `datetime.now` / `time.time` / `date.today` name reference in `driftcall/vendors/*.py`.

**Interaction with timing drifts.** Drifts that mutate policy timing (e.g., `airline.booking_window_shrink` shrinking the booking-window threshold from 24h to 2h) work by `apply_schema_mutation` changing the threshold field inside `AirlinePolicy` — NOT by `now_ist` crossing a static threshold. Since `now_ist` is constant within an episode, the only way a policy can flip mid-episode is via a drift mutation. This guarantees that a replay from the same seed produces bit-identical policy decisions.

### 3.6 Interaction with drift_injector

Per drift_injector.md §3.4, mutation operators (`rename`, `remove`, `require_new_field`, `change_type`, `numeric_bump`, `enum_expand`, `policy_flag_flip`, `time_window_shrink`, `tnc_text_swap`, `side_channel_notice_append`, `pricing_restructure`, `fee_append`, `auth_scope_bump`, `token_version_bump`) are applied via each vendor's `apply_schema_mutation(vendor_state, mutation) -> vendor_state'` helper. The helper is:

- A **pure function** (frozen `VendorState` in, new frozen `VendorState` out).
- Idempotent per-operator (applying the same mutation twice is a no-op beyond the first — but `drift_injector` itself guards via `DriftReapplicationError`).
- Domain-scoped: it only mutates the keys relevant to `domain` (airline mutation never touches hotel state).

For T&C / pricing / auth drifts whose operator is `side_channel_notice_append`, `apply_schema_mutation` sets `vendor_state.side_channel_notice` (Optional[str]). The drift injector's job ends there — it only ever **sets** the notice. The **clear** is a pure vendor-internal state transition: `env.step` calls `emit_side_channel_if_pending(vendor_state)` ONCE per step, BEFORE dispatch. That helper returns `(notice_or_None, new_vendor_state_with_cleared_channel)` — the consumed-on-read pattern. If a notice was pending, the env installs the returned new vendor state via `vendor_states[domain] = new_state` and attaches the notice string to the next `ToolResult.response["_notice"]` in that domain. This is the "one-shot notice" resolved in drift_injector.md §9 Q4.

**There is no `clear_side_channel` mutation operator.** The 14 operators defined in drift_injector.md §3.4 (`rename`, `remove`, `require_new_field`, `change_type`, `numeric_bump`, `enum_expand`, `policy_flag_flip`, `time_window_shrink`, `tnc_text_swap`, `side_channel_notice_append`, `pricing_restructure`, `fee_append`, `auth_scope_bump`, `token_version_bump`) are the complete set. Clearing the side-channel notice is not a drift — it is the vendor consuming a pending notice during normal read. The drift injector never calls any helper to clear it.

**`dispatch` is the commit path; `apply_schema_mutation` is the drift path.** Both are pure functions returning a new frozen `VendorState`. `dispatch` takes `(tool_name, tool_args, vendor_state, schema_version, episode_seed, now_ist)` and returns `(ToolResult, VendorState')`. For read tools (`*.search`, `*.estimate`, `*.get_booking`, `*.track`), the returned state is identical to the input by identity (`returned_state is vendor_state`) — the dispatch reads but does not commit. For write tools (`*.book`, `*.order`, `*.cancel`, `payment.charge`, `payment.refund`, `payment.get_token`), the returned state is a freshly constructed frozen `VendorState` with the commit delta applied via `dataclasses.replace(old, field={**old.field, key: record})`. `dispatch` never mutates `vendor_state` in place. `env.step` threads the returned state back into `DriftCallState.vendor_states[domain]` on every call — the install is uniform; non-write dispatches simply re-install the same object.

### 3.7 Auth & payment cascades

Every `*_book` / `*_order` handler that finalizes a transaction internally calls `payment.payment_charge(vendor_state=state.vendor_states["payment"], ...)`, which itself returns `(ToolResult, PaymentState')`. If payment returns `auth_error` (token mismatch, MFA required), the calling handler surfaces that error **upward** — the airline/cab/hotel/restaurant `ToolResult` returns `status="auth_error"` with `response={"error_code": "PAYMENT_AUTH_FAILED", "required_scope": "payments:write:v2"}` (or `"mfa_required": true`), propagating the payment gateway's diagnostic. This is the intended cross-domain cascade (drift_injector.md §7 E5).

The caller handler MUST NOT partially commit: if payment fails, no booking is created, no order is placed, and no state transition is recorded in the domain-specific vendor state. Concretely, the caller returns its input `VendorState` by identity (no `dataclasses.replace`), AND it returns the `PaymentState` by identity (the payment handler itself did not commit a new charge on the failure path). `env.step` receives this pair and installs the domain state — both the caller's vendor state and the payment state remain identical to pre-call objects. A cross-domain cascade failure therefore leaves the full `DriftCallState.vendor_states` dict structurally unchanged (same object identities).

### 3.8 Uniqueness of booking/order/ride IDs

Generated IDs are deterministic strings: `f"{domain[:3].upper()}-{hash((episode_seed, op, key)) & 0xFFFF:04X}"` where `op` is `"book" | "order" | "ride" | "charge"` and `key` is tool-arg-derived. This gives 4-hex IDs like `AIR-3F2A`, stable per seed. Collisions within an episode are vanishingly rare (< 0.001% across 16-turn episodes). The vendor reads the current record dict (`vendor_state.bookings` / `.orders` / `.rides` / `.charges`) to detect whether the candidate ID is already present BEFORE constructing the commit state — the read is against the input `vendor_state`, and the commit (if any) is applied via `dataclasses.replace` on that same input.

**`-R{retry}` is for HASH COLLISIONS ONLY.** The `-R{retry}` suffix is appended when two calls with *different* inputs collide on the same 4-hex prefix (a pure hash accident). It is NOT the mechanism for handling duplicate intent (same caller, same key, same parameters) — duplicate-intent calls are rejected by the idempotency guard in §3.9 with a `DUPLICATE_*` policy error BEFORE the ID generator runs. The two paths are disjoint: idempotency runs first (and may short-circuit with an error result and unchanged state), then ID generation runs (and may append `-R{retry}` if the prefix collides).

**Retry counter derivation (deterministic, replayable).** On hash collision — the vendor detects that the candidate 4-hex ID already exists in `vendor_state.bookings` / `.orders` / `.rides` / `.charges` — the vendor appends a `-R{retry}` suffix where `retry` is a per-episode, per-operation monotonic counter derived entirely from replay-stable inputs:

```
retry = 1 + sum(
    1 for existing_id in vendor_state.<records>
    if existing_id.startswith(f"{domain[:3].upper()}-{hash((episode_seed, op, key)) & 0xFFFF:04X}")
)
```

Equivalently: `retry` counts how many already-stored records in the vendor's record dict share the same 4-hex prefix, plus one. Because `vendor_state.<records>` is the deterministic result of all prior tool calls in this episode (each of which was itself a pure function of `(episode_seed, prior_state, tool_args)`), and because the collision-triggering call's `(episode_seed, op, key)` tuple is itself deterministic, `retry` is identical across two runs of the same `episode_seed`. No wall-clock, no global RNG, no process-local counter — the value is reconstructable from state alone.

Worked example: seed `1234`, op `"book"`, key derived from `"6E-2345"` hashes to `3F2A`. Prior state has `AIR-3F2A` already present (from a *different* flight whose key happened to hash to the same prefix). Next call with the same hash prefix → scan finds one prefix match → `retry = 1 + 1 = 2` → ID becomes `AIR-3F2A-R2`. Replaying the episode from seed `1234` reconstructs the identical prior state at the identical turn, finds the same one prefix match, and computes the same `retry = 2`. The ID stream is bit-identical across runs.

Tests assert no first-level collisions for the curated seed set (so `-R{retry}` never fires on the canonical seeds); replay determinism for the `-R{retry}` path is covered by a dedicated stress seed that intentionally collides.

### 3.9 Duplicate-intent idempotency (write-tool guard)

Every write tool runs an **idempotency check** before constructing an ID or committing. If an existing record in `vendor_state.<records>` has the same idempotency key as the incoming request, the tool returns a `policy_error` and **does not** commit — the returned `VendorState` is the input by identity.

Idempotency key per domain (all fields normalized: trimmed whitespace, lowercased where free-text, sorted where list-like):

| Tool | Idempotency key | Error code |
|---|---|---|
| `airline.book` | `(flight_id, passenger_name, depart_date)` | `DUPLICATE_BOOKING` |
| `hotel.book` | `(hotel_id, checkin, checkout, primary_guest)` | `DUPLICATE_BOOKING` |
| `cab.book` | `(pickup, drop, depart_time, vehicle_class)` | `DUPLICATE_RIDE` |
| `restaurant.order` | `(restaurant_id, normalized_items_sorted)` | `DUPLICATE_ORDER` |
| `payment.charge` | `(order_ref, amount_inr, token_scope)` | `DUPLICATE_CHARGE` |

Where `normalized_items_sorted = tuple(sorted((item["dish_id"], item["qty"], tuple(sorted(item.get("modifiers", [])))) for item in items))`.

**Error envelope on duplicate:**

```python
ToolResult(
    tool_name=<as dispatched>,
    status="policy_error",
    response={
        "error_code": "DUPLICATE_BOOKING",     # or DUPLICATE_RIDE / _ORDER / _CHARGE
        "existing_id": "AIR-3F2A",             # the ID of the record that already satisfies this intent
        "original_ts": "2026-04-25T18:32:00+05:30",  # now_ist at the time the original record was created, ISO-8601
        "hint": "an identical booking already exists; cancel the existing one to rebook",
    },
    schema_version=<current>,
    latency_ms=<seeded>,
)
```

**What the original_ts field tracks.** When a write tool commits a record, the record stores `"created_at_ist": now_ist.isoformat()` inside its dict. The idempotency guard reads this back as `original_ts` when rejecting a duplicate. Because `now_ist` is episode-constant (§3.5), `original_ts` equals the `now_ist` the episode was pinned to at `reset()` — not wall-clock time.

**Order of checks in a write handler** (mandatory):

1. Schema validation (required fields, types) → `schema_error` on fail.
2. Policy validation (min order, booking window, enum, GST gating) → `policy_error` on fail.
3. **Idempotency check** (this section) → `policy_error` with `DUPLICATE_*` on fail. Returns `(result, input_state)` — no commit.
4. Auth cascade (payment subcall) → propagates `auth_error` upward on fail. Returns `(result, input_state)` — no commit on either domain.
5. ID generation with `-R{retry}` for hash collisions (§3.8).
6. Commit via `dataclasses.replace` — construct new state and return `(result, new_state)`.

The idempotency guard and the `-R{retry}` mechanism are **disjoint**: duplicate-intent rejects before ID generation; `-R{retry}` only fires when *different* inputs collide on the same 4-hex prefix.

---

## 4. Data Structures

### 4.1 `AirlineState` (frozen dataclass)

```python
@dataclass(frozen=True)
class AirlineState:
    schema_version: str                              # "v1" | "v2" | "v3"
    bookings: dict[str, dict[str, Any]]              # booking_id → booking dict (current-version shape)
    flight_roster_cache: dict[str, tuple[dict[str, Any], ...]]  # search_key → flights tuple
    policy: AirlinePolicy                            # nested frozen policy (booking_window, etc.)
    tnc: AirlineTnC                                  # nested frozen T&C text
    pricing: AirlinePricing                          # convenience_fee, etc.
    side_channel_notice: str | None                  # set by drift_injector; attached once
```

#### Schema field tables

**v1 (baseline, DESIGN.md §5.1):**

| Field | Type | Example | Notes |
|---|---|---|---|
| `flight_id` | str | `"6E-2345"` | Indigo-style code |
| `from` | str (IATA) | `"HYD"` | Origin |
| `to` | str (IATA) | `"BLR"` | Destination |
| `depart` | str (ISO-8601 IST) | `"2026-04-25T18:30:00+05:30"` | Timezone-aware |
| `price` | int | `7200` | Integer INR |
| `currency` | str | `"INR"` | Redundant, removed in v2 |
| `seats_left` | int | `14` | |

**v2 (after `airline.price_rename`, DESIGN.md §5.1):**

| Field | Type | Example | Notes |
|---|---|---|---|
| `flight_id` | str | `"6E-2345"` | unchanged |
| `from` / `to` | str | unchanged | |
| `depart` | str | unchanged | |
| `total_fare_inr` | int | `7200` | **was `price`** |
| `seats_left` | int | unchanged | |
| *(`currency` removed)* | — | — | |

**v3 (after `airline.pax_required`, DESIGN.md §5.1):**

| Field | Type | Example | Notes |
|---|---|---|---|
| *(all v2 fields)* | — | — | |
| `passenger_count` | int | `1` | **New required field on `airline.book`.** Search responses may include occupancy, but book calls now 4xx without it. |

### 4.2 `CabState` (frozen)

```python
@dataclass(frozen=True)
class CabState:
    schema_version: str
    rides: dict[str, dict[str, Any]]
    policy: CabPolicy                                # mini_reject_school_hours flag, vehicle_class_enum
    pricing: CabPricing                              # base_per_km, surge_factor, toll_bundled
    tnc: CabTnC
    side_channel_notice: str | None
```

**v1:**

| Field | Type | Example |
|---|---|---|
| `pickup` | str | `"HYD airport T1"` |
| `drop` | str | `"Banjara Hills"` |
| `vehicle_class` | Literal | `"mini" | "sedan"` |
| `fare_inr` | int | `320` |
| `eta_min` | int | `7` |

**v2 (after `cab.vehicle_class_expand` OR `cab.school_hours_mini_reject`):**

| Field | Type | Example |
|---|---|---|
| `vehicle_class` | Literal | `"mini" | "sedan" | "suv" | "infant_seat_sedan"` |
| `fare_inr` | int | unchanged |
| policy | — | `mini` during 07:00–09:00 IST → `policy_error` |
| (other fields unchanged) | — | — |

**v3 (after `cab.fare_breakdown`):**

| Field | Type | Example | Notes |
|---|---|---|---|
| `pickup`, `drop`, `vehicle_class`, `eta_min` | — | unchanged | |
| ~~`fare_inr`~~ | — | **removed** | Replaced by breakdown |
| `fare_breakdown` | dict | `{"base": 240, "surge": 40, "tolls": 20, "gst": 20}` | Four required int sub-fields |
| `total_inr` | int | `320` | Sum invariant: `base + surge + tolls + gst == total_inr` |

### 4.3 `RestaurantState` (frozen)

```python
@dataclass(frozen=True)
class RestaurantState:
    schema_version: str
    orders: dict[str, dict[str, Any]]
    menu_cache: dict[str, tuple[dict[str, Any], ...]]  # city:cuisine → listings
    policy: RestaurantPolicy                            # min_order_inr
    semantics: RestaurantSemantics                      # veg_only_excludes_egg flag
    tnc: RestaurantTnC
    side_channel_notice: str | None
```

**v1:**

| Field | Type | Example | Notes |
|---|---|---|---|
| `restaurant_id` | str | `"BLR-BIR-0123"` | |
| `items` | list[dict] | `[{"dish_id": "BIR-001", "qty": 1, "price": 220}]` | |
| `total` | int | `220` | Sum of `qty * price` per item |
| `eta_min` | int | `35` | |
| `min_order_inr` | int | `199` | Server enforces ≥ this |

**v2 (after `restaurant.min_order_bump`):**

| Field | Type | Change | Notes |
|---|---|---|---|
| `min_order_inr` | int | `199` → `299` | Enforced server-side; orders below → `policy_error` with `MIN_ORDER_NOT_MET` |
| (all others) | — | unchanged | |

**v3 (after `restaurant.veg_filter_semantic` AND/OR `restaurant.items_shape_bump`):**

| Field | Type | Example | Notes |
|---|---|---|---|
| `items` | list[dict] | `[{"dish_id": "BIR-001", "qty": 1, "modifiers": ["no-onion"]}]` | **`modifiers: list[str]` now required** on each item (empty list allowed) |
| `veg_only` (search arg semantics) | bool | — | `True` now *excludes* egg dishes (previously included). No field rename — behavior shift only. Declared via `side_channel_notice`. |

### 4.4 `HotelState` (frozen)

```python
@dataclass(frozen=True)
class HotelState:
    schema_version: str
    bookings: dict[str, dict[str, Any]]
    inventory_cache: dict[str, tuple[dict[str, Any], ...]]
    policy: HotelPolicy                              # cancel_window_hours, early_checkin_fee_pct
    pricing: HotelPricing                            # resort_fee_inr (v2+)
    tnc: HotelTnC
    side_channel_notice: str | None
```

**v1:**

| Field | Type | Example |
|---|---|---|
| `hotel_id` | str | `"GOA-BEACH-007"` |
| `city` | str | `"Goa"` |
| `checkin` / `checkout` | str (ISO date) | `"2026-04-27"` / `"2026-04-29"` |
| `nightly_rate` | int | `3500` |
| `total_with_tax` | int | `8260` (2 nights × 3500 + 18% GST) |
| `cancel_window_hours` | int | `24` |

**v2 (after `hotel.cancel_window_shrink` OR `hotel.resort_fee_append`):**

| Field | Type | Change |
|---|---|---|
| `cancel_window_hours` | int | `24 → 6` (policy) |
| `resort_fee_inr` | int | `0 → 500/night` (pricing; surfaces only on `hotel.book`) |
| (all others) | — | unchanged |

**v3 (after `hotel.gst_field`):**

| Field | Type | Example | Notes |
|---|---|---|---|
| `gst_number` | str (optional until total > 7500) | `"29ABCDE1234F1Z5"` | **Required when `total_with_tax > 7500`**; missing → `policy_error` with `GST_REQUIRED` |

### 4.5 `PaymentState` (frozen)

```python
@dataclass(frozen=True)
class PaymentState:
    schema_version: str                              # "v1" | "v2" | "v3"
    charges: dict[str, dict[str, Any]]               # charge_id → charge record
    accepted_token_version: Literal["v1", "v2"]      # "v1" until auth drift
    required_scope: str                              # "payments:write:v1" | "payments:write:v2"
    mfa_threshold_inr: int                           # 0 → disabled; 5000 after mfa_required drift
    side_channel_notice: str | None
```

**v1:** Accepts `payment_token="token_v1"` with `scope=payments:write:v1`. No MFA. All charges ok.

**v2 (after `payment.auth_scope_upgrade`, DESIGN.md §5.5):** Requires `payment_token="token_v2"` with `scope=payments:write:v2`. `token_v1` calls return `auth_error` with `{"error_code": "AUTH_SCOPE_INSUFFICIENT", "required_scope": "payments:write:v2"}`.

**v3 (after `payment.mfa_required`):** On top of v2 scope requirement, any `amount_inr > 5000` demands `mfa_code` arg. Missing → `auth_error` with `{"error_code": "MFA_REQUIRED", "mfa_threshold_inr": 5000}`.

---

## 5. Error Modes

Every vendor handler returns one of exactly five `status` values. No exceptions escape `dispatch()` — all errors are encoded in `ToolResult`.

### 5.1 Status values & triggers

| `status` | Trigger condition | `error_code` (in `response`) | Drift types that produce it |
|---|---|---|---|
| `ok` | Successful call; `response` is domain-appropriate payload | *(absent)* | n/a |
| `schema_error` | Missing required field, wrong type, removed field referenced, or type mismatch the vendor cannot coerce | `MISSING_FIELD`, `UNKNOWN_FIELD`, `TYPE_MISMATCH`, `MISSING_PASSENGER_COUNT`, `MISSING_GST_NUMBER`, `INVALID_ITEMS_SHAPE` | schema |
| `policy_error` | Request violates a current business rule (min order, booking window, school-hours cab) | `MIN_ORDER_NOT_MET`, `BOOKING_WINDOW_CLOSED`, `SCHOOL_HOURS_MINI_REJECTED`, `VEHICLE_CLASS_UNAVAILABLE`, `CANCEL_WINDOW_EXPIRED`, `GST_REQUIRED` | policy, some T&C |
| `auth_error` | Payment token invalid, scope insufficient, MFA missing | `AUTH_SCOPE_INSUFFICIENT`, `MFA_REQUIRED`, `TOKEN_INVALID`, `PAYMENT_AUTH_FAILED` (propagated upward) | auth |
| `timeout` | Simulated network timeout — triggered deterministically when `(hash((episode_seed, tool_name, _canonical_args_json(tool_args))) & 0x7F) == 0` (~0.78%, 1 in 128; formula in §3.1). **Not** drift-triggered; stress-tests R2 false-positives. | `TIMEOUT` | n/a (noise) |

### 5.2 Error-code catalogue (machine-readable, stable)

Every `error_code` is an uppercase snake-case string. R2 detection hints in `data/drift_patterns/drifts.yaml` use these codes as substring tokens (drift_injector.md §6.3), so renames here require a drift-catalogue bump.

**Schema error codes:**
- `MISSING_FIELD` — generic, `response.field_name: str` names the field.
- `MISSING_PASSENGER_COUNT` — airline v3 specific.
- `MISSING_GST_NUMBER` — hotel v3 specific.
- `INVALID_ITEMS_SHAPE` — restaurant v3 items missing `modifiers`.
- `TYPE_MISMATCH` — generic; `response.expected: str`, `response.got: str`.
- `UNKNOWN_FIELD` — caller sent a field the current schema doesn't recognize (not strictly an error in permissive mode; v2+ reject strictly).

**Policy error codes:**
- `MIN_ORDER_NOT_MET` — `response.min_order_inr: int`, `response.got_total_inr: int`.
- `BOOKING_WINDOW_CLOSED` — airline v2 same-day after 14:00.
- `SCHOOL_HOURS_MINI_REJECTED` — cab v2, 07:00–09:00 IST with `vehicle_class=mini`.
- `VEHICLE_CLASS_UNAVAILABLE` — cab enum caller used outside current enum set.
- `CANCEL_WINDOW_EXPIRED` — hotel v2 cancel after 6h-before-checkin.
- `GST_REQUIRED` — hotel v3 for totals > 7500.

**Auth error codes:**
- `AUTH_SCOPE_INSUFFICIENT` — payment v2; `response.required_scope: str`.
- `MFA_REQUIRED` — payment v3; `response.mfa_threshold_inr: int`.
- `TOKEN_INVALID` — malformed payment_token.
- `PAYMENT_AUTH_FAILED` — propagated upward from `*_book` callers. `response` carries original `required_scope` or `mfa_required`.

### 5.2.1 Error envelope canonical fields

When `ToolResult.status != "ok"`, the `response` dict conforms to the canonical envelope defined here. No other shapes are permitted.

**Required field (every non-ok response):**

- `error_code: str` — one of the codes pinned in §5.2 (enumerated in the table below).

**Optional fields (only those listed here may appear; no ad-hoc keys):**

- `hint: str` — user-friendly next-step guidance.
- `field_name: str` — names the offending field (schema errors).
- `required_scope: str` — the payment scope the caller needs.
- `min_order_inr: int` — the policy-enforced minimum order threshold.
- `got_total_inr: int` — the caller's actual order total.
- `computed_total_inr: int` — server-derived total (e.g. nights × rate + GST).
- `gst_threshold_inr: int` — hotel v3 GST-gating threshold.
- `mfa_threshold_inr: int` — payment v3 MFA-gating threshold.
- `mfa_required: bool` — propagated MFA flag on cross-domain cascades.
- `expected: Any` — expected type/shape (type mismatch).
- `got: Any` — observed type/shape (type mismatch).
- `available: list` — the current enum set (vehicle class unavailable).
- `existing_id: str` — ID of the prior record (`DUPLICATE_*` only).
- `original_ts: str` — ISO-8601 `now_ist` at the prior record's creation (`DUPLICATE_*` only).

**Per-error-code field pinning.** Every code introduced in §5.2 maps to a fixed field set. Implementers MUST NOT add fields outside these rows; callers MUST NOT assume any field outside this table.

| `error_code` | `status` | Required extra fields | Optional fields |
|---|---|---|---|
| `MISSING_FIELD` | `schema_error` | `field_name` | `hint` |
| `MISSING_PASSENGER_COUNT` | `schema_error` | *(none)* | `hint` |
| `MISSING_GST_NUMBER` | `schema_error` | `gst_threshold_inr`, `computed_total_inr` | `hint` |
| `INVALID_ITEMS_SHAPE` | `schema_error` | `field_name` | `hint` |
| `TYPE_MISMATCH` | `schema_error` | `field_name`, `expected`, `got` | `hint` |
| `UNKNOWN_FIELD` | `schema_error` | `field_name` | `hint` |
| `MIN_ORDER_NOT_MET` | `policy_error` | `min_order_inr`, `got_total_inr` | `hint` |
| `BOOKING_WINDOW_CLOSED` | `policy_error` | *(none)* | `hint` |
| `SCHOOL_HOURS_MINI_REJECTED` | `policy_error` | *(none)* | `hint`, `available` |
| `VEHICLE_CLASS_UNAVAILABLE` | `policy_error` | `available` | `hint` |
| `CANCEL_WINDOW_EXPIRED` | `policy_error` | *(none)* | `hint` |
| `GST_REQUIRED` | `policy_error` | `gst_threshold_inr`, `computed_total_inr` | `hint` |
| `DUPLICATE_BOOKING` | `policy_error` | `existing_id`, `original_ts` | `hint` |
| `DUPLICATE_RIDE` | `policy_error` | `existing_id`, `original_ts` | `hint` |
| `DUPLICATE_ORDER` | `policy_error` | `existing_id`, `original_ts` | `hint` |
| `DUPLICATE_CHARGE` | `policy_error` | `existing_id`, `original_ts` | `hint` |
| `AUTH_SCOPE_INSUFFICIENT` | `auth_error` | `required_scope` | `hint` |
| `MFA_REQUIRED` | `auth_error` | `mfa_threshold_inr` | `hint`, `mfa_required` |
| `TOKEN_INVALID` | `auth_error` | *(none)* | `hint` |
| `PAYMENT_AUTH_FAILED` | `auth_error` | *(none)* | `required_scope`, `mfa_required`, `hint` |
| `TIMEOUT` | `timeout` | *(none)* | `hint` |
| `INTERNAL_SUM_MISMATCH` | `schema_error` | *(none)* | `hint` |

**Contract invariant.** No `response` dict outside the schemas defined here is permitted. §8 examples and all §5.2 code descriptions MUST use only fields declared here. Any vendor handler returning a key not listed above is a contract violation and is caught by the envelope-shape test in `tests/test_vendors.py`.

### 5.3 Informational notice codes

These are **not errors**. They ride on `status="ok"` responses inside `response._notice` (via the side-channel one-shot surface from §3.6) or directly in the response body. They exist to signal semantic shifts that do not change response shape and thus cannot be expressed as `schema_error` / `policy_error`. They are not enumerated in the `status` field (which stays at the five values from §5.1) and they never appear as `error_code`.

- `VEG_ONLY_EXCLUDES_EGG` — restaurant v3. Attached to `restaurant.search` response as `response._notice = "veg_only now excludes egg dishes"` (or the equivalent notice string installed by the `restaurant.veg_filter_semantic` drift's `side_channel_notice_append` operator). The call itself returns `status="ok"` with filtered results. R2 detection hints match on `veg_only | egg | exclude | notice`. See E2 for the full scenario.

Additional informational notices are carried the same way whenever a drift operator is `side_channel_notice_append` (T&C, pricing, auth) — they are surfaced via `response._notice` on the next tool call in the affected domain, once, then cleared by the vendor's `emit_side_channel_if_pending` helper (§3.6).

### 5.4 What is NOT an error

- A successful `airline.search` returning zero matching flights → `status="ok"`, `response={"results": []}`. Empty is a legitimate answer.
- A `cab.estimate` in a schema that's since been mutated to v3 → returns v3 `fare_breakdown`, not an error. Schema-drift is seamless from the vendor's side; it's the *agent*'s job to notice the shape changed.

---

## 6. Dependencies

### 6.1 Consumes

- `driftcall/models.py` — `ToolResult`, `GoalSpec`, `DriftCallState` (read only).
- `datetime` (stdlib) + `zoneinfo` for IST — injected, never sourced from wall clock.
- `random.Random(episode_seed)` — local RNG per tool call.
- `types.MappingProxyType` — frozen sub-dict views.
- `json` — used only in `tests/test_vendors.py` for the round-trip guard; vendors themselves never JSON-serialize.

**No third-party imports.** Matches `models.py`'s zero-dependency posture (models.md §6.1). This ensures vendors import cleanly inside both the FastAPI process and the Unsloth training loop.

### 6.2 Produces

- `ToolResult` (to `env.step`).
- New `VendorState` (from `apply_schema_mutation`, returned to `drift_injector`).
- `dict[str, Any]` schema snapshot (from `describe_schema`, for `PROBE_SCHEMA` action).

### 6.3 Consumed by

- **`env.step`** — dispatches `TOOL_CALL` to the correct vendor module by prefix (`tool_name.split(".")[0]`), receives `ToolResult`, appends to state.
- **`drift_injector`** — calls `apply_schema_mutation(vendor_state, pattern.mutation)` and installs the returned state via `DriftCallState.replace(...)`.
- **`rewards`** — consumes the `ToolResult` tuple in `episode.tool_results` and `DriftCallState.vendor_states` end-state to compute R1 (was a booking/order/ride actually created in vendor_state), R3 (does the created artifact satisfy `goal.constraints`), R4 (shape legality via a per-schema-version validator).
- **`app.py`** — does not touch vendors directly; all vendor access flows through `env.step`.

### 6.4 Does NOT depend on

- `audio/*` — the audio layer (DESIGN.md §9) is strictly at the env boundary; it converts user utterances to/from text. Vendors operate purely on text tool-args. `ToolResult.response` is never synthesized to audio (agent's `SPEAK` action does that).
- The agent / model / training loop — vendors are environment code.
- `rewards.py` — the arrow points the other way; vendors have zero knowledge of rewards.

---

## 7. Edge Cases

Numbered edge cases with expected behavior. Referenced by `docs/tests/vendors_tests.md`.

**E1 — Payment auth drift cascades into `airline.book`.**
Scenario: `payment.auth_scope_upgrade` fires at turn 5 (drift pattern #19). Agent calls `airline.book` at turn 6 with `payment_token="token_v1"`. `airline.book` internally calls `payment.charge`, which returns `auth_error` with `{"error_code": "AUTH_SCOPE_INSUFFICIENT", "required_scope": "payments:write:v2"}`. `airline.book` propagates upward: returns `ToolResult(status="auth_error", response={"error_code": "PAYMENT_AUTH_FAILED", "required_scope": "payments:write:v2"}, schema_version="v1")`. No booking created. R2 credits if agent references `auth|scope|token|payments:write` within 2 turns (detection hints). Cites drift_injector.md §7 E5 and drift pattern #19.

**E2 — Restaurant v3 `veg_only` semantic change.**
Scenario: `restaurant.veg_filter_semantic` drift fires (pattern #13). Agent calls `restaurant.search(city="Bengaluru", veg_only=True)`. Pre-drift, results include egg biryani. Post-drift, same call excludes egg dishes but still returns `status="ok"` with modified `response.results` AND a `response._notice: "veg_only now excludes egg dishes"` one-shot. The `veg_only` arg's signature didn't change — **the semantic did**. R2 credits keyword match on `veg_only | egg | exclude | notice`. This is the sole drift that touches **semantic meaning** without a shape change — critics must note the subtlety.

**E3 — Hotel v3 GST field conditional gating.**
Scenario: `hotel.gst_field` drift fires (pattern #5). Agent calls `hotel.book` with `total_with_tax=9500` and no `gst_number`. Returns `schema_error` with `{"error_code": "MISSING_GST_NUMBER", "gst_threshold_inr": 7500}`. Agent calls same with `total_with_tax=4200` and no `gst_number` → `status="ok"` (under threshold). The gating is **conditional on computed total**, not a blanket requirement — tests cover both branches.

**E4 — Cab vehicle_class enum expansion strictly enforced.**
Scenario: Pre-drift (`cab.vehicle_class_expand`, pattern #10 not yet fired). Agent calls `cab.book(vehicle_class="suv")`. `CabState.policy.vehicle_class_enum == ("mini", "sedan")`, so `"suv"` is not in the enum. Returns `policy_error` with `{"error_code": "VEHICLE_CLASS_UNAVAILABLE", "available": ["mini", "sedan"]}`. Post-drift, same call succeeds. Tests verify both temporal halves.

**E5 — `payment.auth_scope_upgrade` mid-booking.**
Scenario: Agent has an open `restaurant.order` in progress (2-call sequence: search → order). Between `search` (turn 3) and `order` (turn 5), the auth drift fires at turn 4. `restaurant.order` at turn 5 triggers `payment.charge` which now rejects `token_v1`. Order is **not** placed (§3.7 no-partial-commit). Agent must call `payment.get_token(requested_scope="payments:write:v2")` → receives new `token_v2` → retry `restaurant.order` with new token. R2 awards on token-scope keyword match; R1 awards only if the order ultimately succeeds within budget.

**E6 — Airline v2 `price_rename` with zero matching flights.**
Scenario: Agent calls `airline.search(from="HYD", to="BLR", max_price_inr=500)` post-drift. No flights at that price exist. Returns `status="ok"`, `response={"results": []}`, `schema_version="v2"`. The absence of `price` fields is moot — there are no result objects to carry them. Agent must recognize empty results, not interpret as schema error. Distinction documented for R5 anti-hack: agent claiming "drift detected" on empty results → R5 penalty (DESIGN.md §7.1 R5 −0.3).

**E7 — Side-channel notice lifecycle (one-shot).**
Scenario: `hotel.early_checkin_tnc` drift fires at turn 3 with `side_channel_notice="early check-in before 12:00 IST now incurs 50% of nightly rate"`. At turn 4, agent calls `hotel.search`. Response carries `response._notice = "early check-in before 12:00 IST now incurs 50% of nightly rate"`. At turn 5, agent calls `hotel.book`. Response does **NOT** re-carry the notice (one-shot, per drift_injector.md §9 Q4 resolution). R2's 2-turn window (turn 3, 4, 5) must credit the agent who mentions the notice at turn 4 or turn 5 (the notice lives in the agent's conversation history).

**E8 — Payment MFA on payment + airline cascade.**
Scenario: `payment.mfa_required` fires (pattern #20). Agent calls `airline.book` with `total_fare_inr=8500` (> 5000). Internal `payment.charge` returns `auth_error` with `MFA_REQUIRED`. Agent must (a) emit SPEAK/CLARIFY asking user for MFA code OR (b) directly call `payment.charge(..., mfa_code="123456")` if the task brief included one in slots. Tests cover both paths. Note: task_generator surfaces `mfa_code` as a slot only in stage-3 compound-drift episodes — otherwise the agent must CLARIFY to obtain it.

**E9 — Cab v3 `fare_breakdown` sum invariant.**
Scenario: Post `cab.fare_breakdown` drift, every `cab.estimate` / `cab.book` response MUST satisfy `base + surge + tolls + gst == total_inr`. Enforced by `_serialize_v3` helper with an internal assert; if violated (would indicate a vendor bug), returns `schema_error` with `error_code="INTERNAL_SUM_MISMATCH"` — this should never fire in practice and is a defensive self-check.

**E10 — `PROBE_SCHEMA` action returns current snapshot for the named domain.**
Scenario: Agent emits `DriftCallAction(action_type=PROBE_SCHEMA, tool_name="airline")` (models.md §3.5 — tool_name is bare domain). Env dispatches to `airline.describe_schema(vendor_state, schema_version)` which returns `{"version": "v2", "fields": {"flight_id": "str", "total_fare_inr": "int", ...}, "removed_from_prior": ["price", "currency"]}`. Wrapped in `ToolResult(tool_name="airline.describe", status="ok", response=<the dict>, schema_version="v2", latency_ms=5)`. This costs 1 turn, per DESIGN.md §4.3; R5 penalizes if used 3+ times (DESIGN.md §7.1 R5 −0.5).

---

## 8. Examples

Three concrete traces `tool_call → ToolResult`.

### 8.1 `airline.search` v1 returning a flight list

**Inputs:**
```python
action = DriftCallAction(
    action_type=ActionType.TOOL_CALL,
    tool_name="airline.search",
    tool_args={
        "from": "HYD",
        "to": "BLR",
        "date": "2026-04-25",
        "max_price_inr": 8000,
        "time_window": "evening",
    },
    rationale="User wants evening flight under 8000",
)
vendor_state = AirlineState(
    schema_version="v1",
    bookings={},
    flight_roster_cache={},
    policy=AirlinePolicy(booking_window_hours=24),
    tnc=AirlineTnC(baggage_cabin_kg=7, reschedule_fee_pct=0),
    pricing=AirlinePricing(convenience_fee_inr=0),
    side_channel_notice=None,
)
schema_version = "v1"
episode_seed = 1234
```

**Output `ToolResult`:**
```python
ToolResult(
    tool_name="airline.search",
    status="ok",
    response={
        "results": [
            {
                "flight_id": "6E-2345",
                "from": "HYD",
                "to": "BLR",
                "depart": "2026-04-25T18:30:00+05:30",
                "price": 7200,
                "currency": "INR",
                "seats_left": 14,
            },
            {
                "flight_id": "AI-501",
                "from": "HYD",
                "to": "BLR",
                "depart": "2026-04-25T20:15:00+05:30",
                "price": 6800,
                "currency": "INR",
                "seats_left": 3,
            },
        ]
    },
    schema_version="v1",
    latency_ms=142,
)
```

### 8.2 `airline.book` v2 after `airline.price_rename` drift

**Inputs (after drift fired, so `schema_version="v2"`, `AirlineState` mutated):**
```python
action = DriftCallAction(
    action_type=ActionType.TOOL_CALL,
    tool_name="airline.book",
    tool_args={
        "flight_id": "6E-2345",
        "payment_token": "token_v1",
    },
    rationale="Booking cheapest evening flight",
)
# vendor_state.schema_version == "v2"; no `price`/`currency` keys
```

**Output `ToolResult`:**
```python
ToolResult(
    tool_name="airline.book",
    status="ok",
    response={
        "booking_id": "AIR-3F2A",
        "flight_id": "6E-2345",
        "total_fare_inr": 7200,         # NOTE: renamed from v1 "price"
        "depart": "2026-04-25T18:30:00+05:30",
        "seats_confirmed": 1,
        "payment_status": "captured",
    },
    schema_version="v2",
    latency_ms=287,
)
```

Drift-detection note: R2 credits the agent if, on the prior turn or this turn, SPEAK/CLARIFY text or `tool_args` JSON contains `"total_fare_inr"`, `"price"`, or `"rename"` (detection hints from pattern `airline.price_rename`, drift_injector.md §4.3).

### 8.3 `hotel.book` v3 with GST required

**Inputs:**
```python
action = DriftCallAction(
    action_type=ActionType.TOOL_CALL,
    tool_name="hotel.book",
    tool_args={
        "hotel_id": "GOA-BEACH-007",
        "checkin": "2026-04-27",
        "checkout": "2026-04-29",
        "payment_token": "token_v2",
        # gst_number intentionally omitted
    },
    rationale="Booking sea-view hotel for weekend",
)
# vendor_state.schema_version == "v3"; `hotel.gst_field` drift has fired
# computed total_with_tax == 9500 (> 7500 threshold)
```

**Output `ToolResult`:**
```python
ToolResult(
    tool_name="hotel.book",
    status="schema_error",
    response={
        "error_code": "MISSING_GST_NUMBER",
        "gst_threshold_inr": 7500,
        "computed_total_inr": 9500,
        "hint": "provide gst_number (15-char GSTIN) for bookings above threshold",
    },
    schema_version="v3",
    latency_ms=89,
)
```

Follow-up: agent calls `hotel.book` again with `"gst_number": "29ABCDE1234F1Z5"` → `status="ok"`, booking created.

Second follow-up (edge case on pattern fire): if `computed_total_inr == 4200 < 7500`, the same call without `gst_number` succeeds — gating is conditional (E3).

---

## 9. Open Questions

**Q1 (deferred post-hackathon) — Cab v2 vs v3 co-existence.**
Two drift patterns apply to cab: `cab.vehicle_class_expand` (policy, v1→v2) and `cab.fare_breakdown` (schema, v2→v3 OR v1→v3 depending on firing order). In stage-3 episodes where both fire, their relative order changes the effective v-bump chain. Current spec: drift_injector.md §7 E7 allows chaining; `CabState.schema_version` moves monotonically v1→v2→v3. If only `cab.fare_breakdown` fires (no prior `vehicle_class_expand`), the transition is v1→v3 directly — technically skipping v2's enum expansion. The fare_breakdown mutation still applies cleanly, but the enum stays at v1's `{mini, sedan}`. This is intentional (each drift is independent) but surprising. Flagged for critic review; post-hackathon consider an "effective schema version" derived from the set of applied mutations rather than a monotonic counter.

**Q2 (deferred) — Restaurant v2 + v3 compound: `modifiers` requirement retroactive?**
If `restaurant.items_shape_bump` (pattern #4, v2→v3, requires `modifiers`) fires at turn 5 and the agent had placed an order at turn 3 (without `modifiers`), the earlier order stays valid (its record in `RestaurantState.orders` already has whatever shape it had). But if the agent queries `restaurant.track(order_id=...)` post-drift, the response is serialized under v3 — does it backfill `modifiers: []` on the historical record, or return the pre-drift shape? Decision for the hackathon: **backfill with `modifiers: []`** on read (cleaner for agent), but do not rewrite `RestaurantState.orders` records in place (immutability). Tracker pattern: the v3 serializer reads the stored record and augments with the default `[]` if absent. Documented in behavior spec §3.2 as part of version-transition semantics; opening for post-hackathon as it has implications for audit replay.

**Q3 (deferred) — Latency distribution calibration.**
Current spec: `ok` latency uniform 50–400 ms; `timeout` 5000–7000 ms. These are placeholder ranges. Judges may notice unrealistic uniformity if they inspect `ToolResult.latency_ms` in the audit trail. Post-hackathon: sample from a log-normal distribution calibrated to IRCTC/IndiGo/Uber real-world latency percentiles. No training-impact; flagged for polish.

**Q4 (deferred) — Test coverage for cross-domain drift chains.**
Spec covers E1 (payment→airline), E5 (payment→restaurant), E8 (MFA→airline). What about payment→hotel (hotel.book calls payment.charge)? And payment→cab? These are structurally identical cascades but not explicitly enumerated in `tests/test_vendors.py`. The test plan (`docs/tests/vendors_tests.md`, not yet written) should enumerate all 4 × 2 = 8 primary-domain × auth-drift combinations. Flagged for Person B (tests owner) to pick up in Batch D4.

---

*End of vendors.md.*
