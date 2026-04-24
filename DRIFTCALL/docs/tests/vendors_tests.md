# vendors_tests.md — Test Plan for `driftcall/vendors/*.py`

**Target module:** `driftcall/vendors/` (`base.py`, `airline.py`, `cab.py`, `restaurant.py`, `hotel.py`, `payment.py`)
**Spec doc:** `DRIFTCALL/docs/modules/vendors.md` (final, sealed)
**Framework:** `pytest` + `hypothesis`
**Owner:** Person B (Rewards & Tests) — domain-reviewed by Person A (Environment)
**Implements:** vendors.md §§2–8 behavioral coverage; drift_injector.md §3.4 (14 mutation operators); DRIFTCALL/CLAUDE.md §3.1 (nine-section test-plan doc)
**Numeric invariants:** Integer INR only — tests MUST assert `isinstance(value, int) and not isinstance(value, bool)` on every monetary field. No `math.isclose`, no float tolerance; amounts are exact.
**Mandatory assertion on every `ToolResult.response`:** `json.loads(json.dumps(response)) == response` — enforced by a helper fixture `assert_json_roundtrip(response)` that all response-path tests call.

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `driftcall/vendors/*.py`. All 16 tool functions (airline × 4, cab × 3, restaurant × 3, hotel × 3, payment × 3) plus the 5 shared helpers (`dispatch`, `initial_state`, `apply_schema_mutation`, `describe_schema`, `emit_side_channel_if_pending`) in every vendor module are exercised. Every one of the 14 mutation operators from drift_injector.md §3.4 has at least one unit test applying it and asserting the post-state shape.

Fixtures defined in §5 are **shared** with `env_tests.md` (same names, same canonicalised content). If any fixture content changes here, the shared copy in `tests/conftest.py` MUST be updated in lockstep, and `env_tests.md §5` cross-checked.

---

## 1. Unit Tests

**Organisation:** one `pytest` sub-package per vendor, plus a shared helpers sub-package. File layout under `tests/test_vendors/`:

```
tests/test_vendors/
  __init__.py
  conftest.py                        # fixtures from §5, plus assert_json_roundtrip helper
  test_base_helpers.py               # dispatch wiring, emit_side_channel_if_pending, ID generation, timeout hash
  test_airline_v1.py
  test_airline_v2.py
  test_airline_v3.py
  test_cab_v1.py
  test_cab_v2.py
  test_cab_v3.py
  test_restaurant_v1.py
  test_restaurant_v2.py
  test_restaurant_v3.py
  test_hotel_v1.py
  test_hotel_v2.py
  test_hotel_v3.py
  test_payment_v1.py
  test_payment_v2.py
  test_payment_v3.py
  test_apply_schema_mutation.py      # 14-operator coverage, all domains
  test_idempotency.py                # DUPLICATE_* guard, per domain
  test_auth_cascades.py              # cross-domain propagation (payment → airline/cab/hotel/restaurant)
  test_integer_inr_invariant.py      # no-float guard on every monetary field
  test_json_roundtrip.py             # catchall: every ToolResult.response is JSON-roundtrip-safe
  test_determinism.py                # same inputs → same outputs across two runs
  test_now_ist_constant.py           # wall-clock readers forbidden + episode-constant
```

**Unit test case inventory — 52 cases total (exceeds the ≥ 40 requirement; ≥ 3 per tool per schema version for the 16 tools = 48, plus 4 shared-helper edge cases).**

### 1.1 Airline — `test_airline_v{1,2,3}.py`

**Scope:** 4 tools × 3 versions × (happy + ≥ 1 error) = 13 cases.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_airline_search_v1_happy` | `vendor_states_v1["airline"]`, args `{from:"HYD", to:"BLR", date:"2026-04-25", max_price_inr:8000, time_window:"evening"}`, `episode_seed=1234`, `now_ist=now_ist_evening`. | `status=="ok"`, `response["results"]` non-empty, every flight dict has keys `{flight_id, from, to, depart, price, currency, seats_left}`, every `price` is `int`, `schema_version=="v1"`, `latency_ms` in [50,400], `returned_state is input_state` (identity). JSON roundtrip. |
| U2 | `test_airline_search_v1_empty_results_ok` | As U1 but `max_price_inr=500`. | `status=="ok"`, `response=={"results": []}`, identity-preserved state, no `error_code` key. JSON roundtrip. |
| U3 | `test_airline_book_v1_happy` | `vendor_states_v1["airline"]`, args `{flight_id:"6E-2345", payment_token:"token_v1"}`, payment state v1. | `status=="ok"`, `response["booking_id"]` matches `^AIR-[0-9A-F]{4}$`, `response["payment_status"]=="captured"`, returned state `is not` input state, `returned_state.bookings` contains the new id, old dict object id differs (`id(returned.bookings) != id(input.bookings)`). JSON roundtrip. |
| U4 | `test_airline_book_v1_timeout_path` | As U3 but use a seed whose `(hash & 0x7F) == 0` for those args (seed `42` curated). | `status=="timeout"`, `response=={"error_code":"TIMEOUT"}` (optional `hint`), `latency_ms` in [5000,7000], returned state `is input_state`. |
| U5 | `test_airline_cancel_v1_happy` | State pre-seeded with a booking `AIR-3F2A`. | `status=="ok"`, booking removed or flagged cancelled per spec; returned state `is not` input. JSON roundtrip. |
| U6 | `test_airline_get_booking_v1_not_found` | Empty bookings, args `{booking_id:"AIR-0000"}`. | `status=="schema_error"` with `error_code=="MISSING_FIELD"` **or** domain-scoped `policy_error` per spec; returned state identity-preserved. JSON roundtrip. |
| U7 | `test_airline_search_v2_uses_total_fare_inr` | `vendor_states_v2["airline"]` (post `airline.price_rename`). | Every result dict has `total_fare_inr: int`, lacks `price` and `currency` keys, `schema_version=="v2"`. JSON roundtrip. |
| U8 | `test_airline_book_v2_happy` | `vendor_states_v2["airline"]`, args as U3. | `response["total_fare_inr"]: int`, no `price`, `schema_version=="v2"`. Booking committed. |
| U9 | `test_airline_book_v2_booking_window_closed` | `now_ist = now_ist_evening` (post-14:00 IST), same-day flight; `policy.booking_window_hours=2`. | `status=="policy_error"`, `error_code=="BOOKING_WINDOW_CLOSED"`, state identity-preserved. |
| U10 | `test_airline_book_v3_missing_passenger_count_schema_error` | `vendor_states_v3["airline"]`, args without `passenger_count`. | `status=="schema_error"`, `error_code=="MISSING_PASSENGER_COUNT"`, no `field_name` required, state identity-preserved. JSON roundtrip. |
| U11 | `test_airline_book_v3_happy_with_passenger_count` | As U10 but `passenger_count=2`. | `status=="ok"`, booking record stores `passenger_count=2`, returned state committed. |
| U12 | `test_airline_search_v3_response_shape` | `vendor_states_v3["airline"]`, `airline.search(...)`. | Results use `total_fare_inr`, no `price`, `schema_version=="v3"`. |
| U13 | `test_airline_book_v1_duplicate_intent_returns_DUPLICATE_BOOKING` | Pre-seed state with an existing booking keyed `(flight_id, passenger_name, depart_date)`; re-call with identical args. | `status=="policy_error"`, `error_code=="DUPLICATE_BOOKING"`, `response["existing_id"]` matches prior record, `response["original_ts"]` equals `now_ist.isoformat()`, state identity-preserved (no commit). |

### 1.2 Cab — `test_cab_v{1,2,3}.py`

**Scope:** 3 tools × 3 versions × (happy + ≥ 1 error) = 10 cases.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U14 | `test_cab_estimate_v1_happy` | `vendor_states_v1["cab"]`, `{pickup:"HYD T1", drop:"Banjara Hills", vehicle_class:"mini", pickup_time_ist:"2026-04-25T10:00+05:30"}`. | `status=="ok"`, `response["fare_inr"]: int`, `eta_min: int`, no `fare_breakdown` key, identity-preserved. JSON roundtrip. |
| U15 | `test_cab_book_v1_happy` | As U14 plus `payment_token="token_v1"`. | `status=="ok"`, `response["ride_id"]` matches `^CAB-[0-9A-F]{4}$`, returned state committed. |
| U16 | `test_cab_cancel_v1_happy` | State pre-seeded with ride `CAB-1234`. | `status=="ok"`, returned state `is not` input. JSON roundtrip. |
| U17 | `test_cab_book_v2_school_hours_mini_rejected` | `vendor_states_v2["cab"]` (post `policy_flag_flip`), `now_ist = now_ist_morning` (08:00 IST), `vehicle_class="mini"`. | `status=="policy_error"`, `error_code=="SCHOOL_HOURS_MINI_REJECTED"`, `response.get("available", [])` lists current enum excluding mini, state identity-preserved. |
| U18 | `test_cab_book_v2_suv_accepted_after_enum_expand` | `vendor_states_v2["cab"]` (post `cab.vehicle_class_expand`), `vehicle_class="suv"`. | `status=="ok"`, committed. |
| U19 | `test_cab_book_v1_suv_rejected_before_enum_expand` | `vendor_states_v1["cab"]`, `vehicle_class="suv"`. | `status=="policy_error"`, `error_code=="VEHICLE_CLASS_UNAVAILABLE"`, `available=="('mini','sedan')"` (list form). |
| U20 | `test_cab_estimate_v3_fare_breakdown_sum_invariant` | `vendor_states_v3["cab"]` (post `cab.fare_breakdown`). | `response["fare_breakdown"]` keys `{base, surge, tolls, gst}` all ints, `sum(...) == response["total_inr"]`, no `fare_inr` key. |
| U21 | `test_cab_book_v3_fare_breakdown_persisted` | As U20 plus book. | Committed record contains `fare_breakdown` dict with same invariant. |
| U22 | `test_cab_cancel_v3_happy` | v3 state pre-seeded. | `status=="ok"`, returned state is new object. |
| U23 | `test_cab_book_v1_duplicate_returns_DUPLICATE_RIDE` | Re-book identical key `(pickup, drop, depart_time, vehicle_class)`. | `status=="policy_error"`, `error_code=="DUPLICATE_RIDE"`, `existing_id` populated, state identity-preserved. |

### 1.3 Restaurant — `test_restaurant_v{1,2,3}.py`

**Scope:** 3 tools × 3 versions × (happy + ≥ 1 error) = 10 cases.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U24 | `test_restaurant_search_v1_happy` | `vendor_states_v1["restaurant"]`, `{city:"Bengaluru", cuisine:"biryani", veg_only:false, max_price_inr:400}`. | `status=="ok"`, `response["results"]` non-empty, each item has integer prices, identity-preserved. |
| U25 | `test_restaurant_order_v1_min_order_met` | Items summing to 220 ≥ 199. | `status=="ok"`, `response["order_id"]` matches `^RES-[0-9A-F]{4}$`, committed. |
| U26 | `test_restaurant_track_v1_happy` | State pre-seeded with order. | `status=="ok"`, response is JSON-roundtrip-safe, identity-preserved. |
| U27 | `test_restaurant_order_v2_min_order_not_met` | `vendor_states_v2["restaurant"]` (min_order 299), total 220. | `status=="policy_error"`, `error_code=="MIN_ORDER_NOT_MET"`, `response["min_order_inr"]==299`, `response["got_total_inr"]==220`, state identity-preserved. |
| U28 | `test_restaurant_order_v2_min_order_met_after_bump` | v2, total 350. | `status=="ok"`, committed. |
| U29 | `test_restaurant_search_v3_veg_only_excludes_egg` | v3 post `restaurant.veg_filter_semantic`, `veg_only=True`. Pre-seed menu with egg biryani + veg biryani. | `status=="ok"`, no result contains `dish_id` tagged `egg`, **no `_notice` key** (notice is emitted by `emit_side_channel_if_pending`, which is tested separately — the search handler itself does not attach notices). |
| U30 | `test_restaurant_order_v3_missing_modifiers_schema_error` | v3, items without `modifiers` key. | `status=="schema_error"`, `error_code=="INVALID_ITEMS_SHAPE"`, `field_name=="items"`, state identity-preserved. |
| U31 | `test_restaurant_order_v3_happy_with_modifiers` | v3, items include `modifiers: []`. | `status=="ok"`, committed. |
| U32 | `test_restaurant_track_v3_backfills_modifiers_on_historical_record` | Order committed at v2 then schema drifts to v3, call track. Ref: vendors.md §9 Q2. | Response record contains `modifiers: []` even though stored record lacks it; storage dict unchanged (`state.orders[id]` has no `modifiers` key — read-side augmentation only). |
| U33 | `test_restaurant_order_v1_duplicate_returns_DUPLICATE_ORDER` | Re-order with identical `(restaurant_id, normalized_items_sorted)`. | `status=="policy_error"`, `error_code=="DUPLICATE_ORDER"`, state identity-preserved. |

### 1.4 Hotel — `test_hotel_v{1,2,3}.py`

**Scope:** 3 tools × 3 versions × (happy + ≥ 1 error) = 10 cases.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U34 | `test_hotel_search_v1_happy` | `vendor_states_v1["hotel"]`, `{city:"Goa", checkin:"2026-04-27", checkout:"2026-04-29", max_nightly_rate_inr:4000}`. | `status=="ok"`, `results` non-empty, integer nightly rates, identity-preserved. |
| U35 | `test_hotel_book_v1_happy` | As U34 plus `payment_token="token_v1"`. | `status=="ok"`, `response["booking_id"]` matches `^HOT-[0-9A-F]{4}$`, `total_with_tax: int`, committed. |
| U36 | `test_hotel_cancel_v1_happy` | Pre-seeded booking. | `status=="ok"`, committed new state. |
| U37 | `test_hotel_cancel_v2_window_expired` | `vendor_states_v2["hotel"]` (`cancel_window_hours=6`), `now_ist` inside the 6h cutoff. | `status=="policy_error"`, `error_code=="CANCEL_WINDOW_EXPIRED"`, state identity-preserved. |
| U38 | `test_hotel_book_v2_resort_fee_applied` | `vendor_states_v2["hotel"]` post `hotel.resort_fee_append`. | `response["resort_fee_inr"]==500` (per night, integer), total arithmetic holds. |
| U39 | `test_hotel_book_v3_missing_gst_number_over_threshold` | v3, `total_with_tax=9500`, no `gst_number`. | `status=="schema_error"`, `error_code=="MISSING_GST_NUMBER"`, `response["gst_threshold_inr"]==7500`, `response["computed_total_inr"]==9500`. |
| U40 | `test_hotel_book_v3_under_threshold_no_gst_required` | v3, `total_with_tax=4200`, no `gst_number`. | `status=="ok"`, committed. |
| U41 | `test_hotel_book_v3_with_gst_number_happy` | v3, `total_with_tax=9500`, `gst_number="29ABCDE1234F1Z5"`. | `status=="ok"`, committed, record stores `gst_number`. |
| U42 | `test_hotel_book_v1_duplicate_returns_DUPLICATE_BOOKING` | Re-book identical `(hotel_id, checkin, checkout, primary_guest)`. | `status=="policy_error"`, `error_code=="DUPLICATE_BOOKING"`, `existing_id`+`original_ts` populated. |
| U43 | `test_hotel_search_v3_shape_stability` | v3 search. | Response shape matches v3 serializer; no `price` key; all integer INR. |

### 1.5 Payment — `test_payment_v{1,2,3}.py`

**Scope:** 3 tools × 3 versions × (happy + ≥ 1 error) = 10 cases.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U44 | `test_payment_charge_v1_happy` | `vendor_states_v1["payment"]`, `amount_inr=500`, `payment_token="token_v1"`. | `status=="ok"`, `response["charge_id"]` matches `^PAY-[0-9A-F]{4}$`, committed. |
| U45 | `test_payment_refund_v1_happy` | Pre-seeded charge. | `status=="ok"`, refund record committed. |
| U46 | `test_payment_get_token_v1_returns_token_v1` | `requested_scope="payments:write:v1"`. | `status=="ok"`, `response["token"]=="token_v1"`, scope echoed. |
| U47 | `test_payment_charge_v2_token_v1_rejected` | `vendor_states_v2["payment"]` (post auth_scope_bump), `payment_token="token_v1"`. | `status=="auth_error"`, `error_code=="AUTH_SCOPE_INSUFFICIENT"`, `required_scope=="payments:write:v2"`, state identity-preserved. |
| U48 | `test_payment_charge_v2_token_v2_accepted` | v2, `payment_token="token_v2"`. | `status=="ok"`, committed. |
| U49 | `test_payment_get_token_v2_returns_token_v2` | v2, `requested_scope="payments:write:v2"`. | Returns `token_v2`. |
| U50 | `test_payment_charge_v3_over_threshold_no_mfa_rejected` | v3, `amount_inr=8500`, `mfa_code=None`. | `status=="auth_error"`, `error_code=="MFA_REQUIRED"`, `mfa_threshold_inr==5000`. |
| U51 | `test_payment_charge_v3_over_threshold_with_mfa_accepted` | v3, `amount_inr=8500`, `mfa_code="123456"`. | `status=="ok"`, committed. |
| U52 | `test_payment_charge_v3_under_threshold_no_mfa_ok` | v3, `amount_inr=500`, no `mfa_code`. | `status=="ok"` (MFA not required below threshold). |
| U53 | `test_payment_charge_v1_duplicate_returns_DUPLICATE_CHARGE` | Re-charge identical `(order_ref, amount_inr, token_scope)`. | `status=="policy_error"`, `error_code=="DUPLICATE_CHARGE"`, state identity-preserved. |
| U54 | `test_payment_token_invalid_malformed` | `payment_token="garbage-token"`. | `status=="auth_error"`, `error_code=="TOKEN_INVALID"`. |

### 1.6 Shared dispatch helpers — `test_base_helpers.py`

**Scope:** `dispatch`, `emit_side_channel_if_pending`, `initial_state`, `describe_schema`, ID generator + retry, timeout hash, `now_ist` constant.

| # | Name | Assertion |
|---|---|---|
| U55 | `test_dispatch_returns_tuple_toolresult_vendorstate` | For every domain + tool, `dispatch(...)` returns `tuple` of length 2; `[0]` is `ToolResult`, `[1]` is the module's `*State`. |
| U56 | `test_dispatch_read_tools_return_identity_state` | For `*.search`, `*.estimate`, `*.get_booking`, `*.track`: `returned_state is input_state`. |
| U57 | `test_dispatch_write_tools_return_new_state` | For `*.book`, `*.order`, `*.cancel`, `payment.charge|refund|get_token`: `returned_state is not input_state` AND `id(returned.<record_dict>) != id(input.<record_dict>)` on success. |
| U58 | `test_emit_side_channel_if_pending_consume_on_read` | Pre-seed state with `side_channel_notice="..."`. First call returns `(<notice>, new_state_with_None)`; second call on the returned state returns `(None, same_state)`. Consume-on-read pattern (vendors.md §3.6). |
| U59 | `test_emit_side_channel_if_pending_none_returns_identity` | State with `side_channel_notice=None` → `(None, vendor_state)` with `returned_state is vendor_state`. |
| U60 | `test_timeout_hash_formula_1_in_128` | Exhaustive check: for 10,000 seeded `(seed, tool, canonical_args)` triples, exactly ~78 (1/128 ± 3σ binomial bound) match `(hash(...) & 0x7F) == 0`. Asserts the hash formula matches vendors.md §3.1. |
| U61 | `test_timeout_hash_deterministic_replay` | Same `(seed, tool_name, tool_args)` → `(hash & 0x7F)` identical across two Python interpreter invocations (uses `PYTHONHASHSEED=0` pinned in `conftest.py`). |
| U62 | `test_id_generation_retry_counter_monotonic` | Force two identical key hashes (pre-seeded state with one colliding prefix) → second call produces `-R2` suffix, third produces `-R3`. |
| U63 | `test_now_ist_is_episode_constant` | Dispatch same tool at two wall-clock-distant moments with same `now_ist` argument → returned `ToolResult` (modulo `latency_ms`) structurally equal. |
| U64 | `test_vendors_have_no_wall_clock_reads` | AST-grep over `driftcall/vendors/*.py`: no reference to `datetime.now`, `time.time`, `date.today`, `time.monotonic`. Raises `AssertionError` listing any offending file:line. |
| U65 | `test_describe_schema_returns_current_version_fields` | For each domain × version: `describe_schema(state, version)` returns `{"version": version, "fields": {<expected>}, "removed_from_prior": [<expected>]}` exactly matching vendors.md §4 field tables. |
| U66 | `test_initial_state_deterministic_per_seed` | Two calls `initial_state(seed=1234, goal=g)` → structurally equal `VendorState` instances. |

**Shared helper cases: 12 (U55–U66).**

### 1.7 `apply_schema_mutation` — all 14 operators — `test_apply_schema_mutation.py`

**Scope:** One test per operator × at least one domain where the operator is used (drift_injector.md §3.4). All 14 operators exercised.

| # | Operator | Domain/pattern | Assertion |
|---|---|---|---|
| U67 | `rename` | `airline.price_rename` | Post: every cached flight carries `total_fare_inr` not `price`; `schema_version=="v2"`; returned state frozen + new object. |
| U68 | `remove` | `airline.price_rename` (removes `currency`) | Post: cached flights lack `currency`; `describe_schema(...)["removed_from_prior"]` contains `"currency"`. |
| U69 | `require_new_field` | `airline.pax_required` | Post: `policy.required_book_fields` now includes `"passenger_count"`; v3 `airline.book` without it returns `MISSING_PASSENGER_COUNT`. |
| U70 | `change_type` | synthetic `restaurant.total_to_string` (test-only pattern, covered for operator-code reach even if not in canonical catalogue) | Post: serializer returns `total` as str; vendor's `change_type` helper updates a field-type map. |
| U71 | `numeric_bump` | `restaurant.min_order_bump` | Post: `policy.min_order_inr == 299` (was 199); order below 299 → `MIN_ORDER_NOT_MET`. |
| U72 | `enum_expand` | `cab.vehicle_class_expand` | Post: `policy.vehicle_class_enum == ("mini","sedan","suv","infant_seat_sedan")`; `cab.book(vehicle_class="suv")` succeeds. |
| U73 | `policy_flag_flip` | `cab.school_hours_mini_reject` | Post: `policy.mini_reject_school_hours is True`; mini book at 08:00 IST → `SCHOOL_HOURS_MINI_REJECTED`. |
| U74 | `time_window_shrink` | `airline.booking_window_shrink` | Post: `policy.booking_window_hours` shrunk from 24 to 2; same-day post-14:00 book → `BOOKING_WINDOW_CLOSED`. |
| U75 | `tnc_text_swap` | `hotel.cancel_tnc_swap` | Post: `tnc.cancel_clause` text changed; stored as immutable tuple or frozen dataclass replacement. |
| U76 | `side_channel_notice_append` | `hotel.early_checkin_tnc` | Post: `vendor_state.side_channel_notice` set to the notice string; `emit_side_channel_if_pending` will surface it once. |
| U77 | `pricing_restructure` | `cab.fare_breakdown` | Post: `pricing` replaced; v3 serializer emits `fare_breakdown` sub-dict; sum invariant holds on next `cab.estimate`. |
| U78 | `fee_append` | `hotel.resort_fee_append` | Post: `pricing.resort_fee_inr == 500`; next `hotel.book` response includes fee; `total_with_tax` reflects addition. |
| U79 | `auth_scope_bump` | `payment.auth_scope_upgrade` | Post: `accepted_token_version == "v2"`, `required_scope == "payments:write:v2"`; `token_v1` charge → `AUTH_SCOPE_INSUFFICIENT`. |
| U80 | `token_version_bump` | `payment.mfa_required` (also bumps scope indirectly) / dedicated pattern | Post: `accepted_token_version == "v2"`; bookkeeping-only bump verified by `describe_schema`. |
| U81 | `test_apply_schema_mutation_is_pure_frozen_returns_new_object` | Any operator: `returned is not input`, `input.<all_fields>` byte-identical before and after (deep-equality check), `dataclasses.is_dataclass(returned) and returned.__dataclass_params__.frozen`. |
| U82 | `test_apply_schema_mutation_unknown_operator_raises` | `mutation={"unknown_op": {...}}` → raises `UnknownMutationOperatorError` (vendor-local exception). State untouched. |
| U83 | `test_apply_schema_mutation_domain_scope_enforced` | Airline mutation applied to `AirlineState` does NOT reach `HotelState` (separate call). Verified by passing both, asserting hotel is untouched. |

**Operator coverage: 14/14 operators + 3 meta-properties = 17 cases (U67–U83).**

### 1.8 Idempotency — `test_idempotency.py`

**Scope:** 5 domains × idempotency key per vendors.md §3.9. Disjoint from `-R{retry}` (collision) path.

| # | Name | Assertion |
|---|---|---|
| U84 | `test_airline_book_duplicate_key_triggers_DUPLICATE_BOOKING` | `(flight_id, passenger_name, depart_date)` identical → `policy_error`/`DUPLICATE_BOOKING`; `existing_id`+`original_ts` populated; `original_ts == now_ist.isoformat()`; state identity-preserved. |
| U85 | `test_hotel_book_duplicate_key_triggers_DUPLICATE_BOOKING` | Same for `(hotel_id, checkin, checkout, primary_guest)`. |
| U86 | `test_cab_book_duplicate_key_triggers_DUPLICATE_RIDE` | Same for `(pickup, drop, depart_time, vehicle_class)`. |
| U87 | `test_restaurant_order_duplicate_key_triggers_DUPLICATE_ORDER` | Same for `(restaurant_id, normalized_items_sorted)`; `modifiers` normalization: `[["no-onion"], ["extra-raita"]]` vs `[["extra-raita"], ["no-onion"]]` are the same key. |
| U88 | `test_payment_charge_duplicate_key_triggers_DUPLICATE_CHARGE` | Same for `(order_ref, amount_inr, token_scope)`. |
| U89 | `test_idempotency_runs_before_auth_cascade` | Pre-seed duplicate booking; call with a token that would fail auth. Result: `DUPLICATE_BOOKING` (not `PAYMENT_AUTH_FAILED`) — idempotency short-circuits first (vendors.md §3.9 Step order). |
| U90 | `test_retry_suffix_disjoint_from_idempotency` | Call with *different* args whose hash prefix collides with an existing record → ID gets `-R2` suffix; `DUPLICATE_*` NOT returned. |

**Idempotency cases: 7 (U84–U90).**

### 1.9 Auth cascades — `test_auth_cascades.py`

**Scope:** Every primary domain × every auth drift (payment.auth_scope_upgrade, payment.mfa_required) = 8 cases covering the Q4 gap from vendors.md §9.

| # | Name | Assertion |
|---|---|---|
| U91 | `test_airline_book_propagates_AUTH_SCOPE_INSUFFICIENT` | Payment v2, token_v1 → `airline.book` returns `auth_error`/`PAYMENT_AUTH_FAILED` with `required_scope="payments:write:v2"`. No airline booking committed. Returned airline state `is` input; returned payment state `is` input. |
| U92 | `test_airline_book_propagates_MFA_REQUIRED` | Payment v3, amount>5000, no mfa_code → `airline.book` returns `auth_error`/`PAYMENT_AUTH_FAILED` with `mfa_required=True`. No commit in either domain. |
| U93 | `test_cab_book_propagates_AUTH_SCOPE_INSUFFICIENT` | Same as U91 for cab. |
| U94 | `test_cab_book_propagates_MFA_REQUIRED` | Same as U92 for cab. |
| U95 | `test_hotel_book_propagates_AUTH_SCOPE_INSUFFICIENT` | Same for hotel. |
| U96 | `test_hotel_book_propagates_MFA_REQUIRED` | Same for hotel. |
| U97 | `test_restaurant_order_propagates_AUTH_SCOPE_INSUFFICIENT` | Same for restaurant. |
| U98 | `test_restaurant_order_propagates_MFA_REQUIRED` | Same for restaurant. |

**Cascade cases: 8 (U91–U98).**

### 1.10 Monetary invariant + JSON roundtrip + determinism

| # | Name | Assertion |
|---|---|---|
| U99 | `test_every_amount_field_is_int_not_bool_not_float` | Iterates every response path in §1.1–§1.5 (via parametrised fixture `every_successful_call`), asserts `type(value) is int` for every key matching `^(.*_inr|total|fare|price|amount|eta_min|seats_left|.*_hours|.*_kg)$`. Rejects `True/False` (Python `bool` is `int` subclass). |
| U100 | `test_every_response_json_roundtrip_safe` | `json.loads(json.dumps(result.response)) == result.response` for every tool × every version × every happy/error path. |
| U101 | `test_dispatch_determinism_same_inputs_same_outputs` | For a curated set of (seed, tool, args, state, version, now_ist): two `dispatch` calls return structurally equal `ToolResult` and identical `VendorState` object identity semantics (identity-preserved for reads; structurally equal dataclasses for writes). |

**Invariant cases: 3 (U99–U101).**

**Total unit tests: U1–U101 = 101 cases (far exceeds ≥ 40).**

---

## 2. Property Tests

**Framework:** `hypothesis`. All property tests use custom `hypothesis.strategies` composing `vendor_states_v{1,2,3}` fixtures with randomly-sampled tool-args from pinned domain alphabets. `max_examples=100` default, `max_examples=500` for purity/immutability properties.

### 2.1 Properties (6 total, exceeds ≥ 5 requirement)

| # | Property | Statement | Strategy |
|---|---|---|---|
| P1 | `dispatch_is_pure_given_state` | For any `(tool, args, state, version, seed, now_ist)`, calling `dispatch` twice yields equal `ToolResult` (by `==`) and the same `VendorState` identity semantics (either `is` for reads or structurally equal for writes). No global state mutation. | Compose `st.sampled_from(ALL_TOOLS)`, `vendor_state_strategy()`, `st.integers(min_value=0, max_value=2**32-1)`, `st.datetimes(...)` IST-pinned. |
| P2 | `spread_pattern_never_mutates_input` | For any `old_state` and any operator mutation dict, after `apply_schema_mutation(old, mut)`, the input `old` is byte-equal to a deep-copy taken beforehand. Verifies the `{**old, k: v}` + `dataclasses.replace` discipline. | `vendor_state_strategy()` × `mutation_strategy()` from the 14-op catalogue. |
| P3 | `dispatch_write_returns_distinct_record_dict` | For any successful write tool call, `id(returned_state.<record_dict>) != id(input_state.<record_dict>)`. Guards against in-place `vendor_state.bookings[k] = v` slip-ups. | `st.sampled_from(WRITE_TOOLS)` × `args_strategy()` × `vendor_state_strategy()`. |
| P4 | `timeout_hash_rate_is_approximately_1_in_128` | Hypothesis draws 10,000 `(seed, tool, canonical_args)` triples; count of `(hash & 0x7F) == 0` lies in `[55, 105]` (binomial 99.9% CI around mean 78.1). | `st.tuples(st.integers(), st.sampled_from(ALL_TOOLS), st.text())`. |
| P5 | `integer_inr_invariant_all_paths` | For any `(tool, args, state, version)`, every integer-typed response field (matched by name regex in U99) is `type(value) is int` and `value >= 0`. No floats leaked. | Union of happy-path strategies over all 16 tools. |
| P6 | `fare_breakdown_sum_invariant_v3_cab` | Post `cab.fare_breakdown` drift: for any `cab.estimate` / `cab.book` successful response, `sum(response["fare_breakdown"].values()) == response["total_inr"]`. | `st.sampled_from(["cab.estimate","cab.book"])` × cab-v3 state strategy. |
| P7 | `idempotency_key_normalization_commutative` | For `restaurant.order`, any permutation of `items` list + permutation of inner `modifiers` lists with identical multiset content produces the same `DUPLICATE_ORDER` verdict on a pre-seeded state. | `st.permutations(items_list)` and `st.permutations(modifiers_list)`. |

**Property test count: 7 (exceeds ≥ 5 requirement).**

**Settings pinning:** `@settings(deadline=None, max_examples=100, derandomize=False, print_blob=True)` on every property. `hypothesis.seed` pinned per property for replay; the suite uses the `database_file` in `.hypothesis/` committed to git for deterministic CI.

---

## 3. Integration Tests

**Location:** `tests/test_vendors/test_integration_flows.py`.

### 3.1 IT1 — Full airline booking flow

```
airline.search(HYD→BLR, max_price=8000, window=evening)  # ok, 2 flights
  → airline.book(flight_id=picked_id, payment_token=token_v1)  # payment.charge called
    → ToolResult status=="ok", booking_id present, payment_status=="captured"
```

Assertions:
- Terminal `DriftCallState.vendor_states["airline"].bookings` has exactly one record; prior searches added nothing.
- Terminal `vendor_states["payment"].charges` has one record with `amount_inr` = booked flight's `price`/`total_fare_inr`.
- Both vendor states committed (new object identities vs initial).
- Every `ToolResult.response` passes `assert_json_roundtrip`.
- All `*_inr` fields in every response are `int`.

### 3.2 IT2 — Cross-vendor cascade: payment auth drift breaks `airline.book`

```
env.reset(seed=1234) → drift_schedule places payment.auth_scope_upgrade at turn 5
turn 1..4: airline.search + hotel.search (reads, no commits)
turn 5: DriftPattern fires → payment state mutates to v2
turn 6: airline.book(flight_id, payment_token="token_v1")
  → payment.charge → auth_error AUTH_SCOPE_INSUFFICIENT
  → airline.book propagates: status="auth_error", error_code="PAYMENT_AUTH_FAILED", required_scope="payments:write:v2"
turn 7: payment.get_token(requested_scope="payments:write:v2") → token_v2
turn 8: airline.book(flight_id, payment_token="token_v2") → ok
```

Assertions:
- At turn 6, `vendor_states["airline"]` and `vendor_states["payment"]` have IDENTICAL object identities to turn 5 (no partial commit).
- At turn 8, both are new objects; `airline.bookings` has one record; `payment.charges` has one record.
- JSON roundtrip on every `ToolResult.response` across all 8 turns.

### 3.3 IT3 — Restaurant min-order bump mid-session

```
turn 1: restaurant.order(items total 220) at v1 → ok, committed (min_order=199)
turn 2: drift restaurant.min_order_bump fires → v2 (min_order=299)
turn 3: restaurant.order(identical items total 220) → policy_error MIN_ORDER_NOT_MET
turn 4: restaurant.order(items total 350) → ok, committed
```

Assertions: version bump visible in `schema_version` field of every turn-3+ ToolResult; terminal state has 2 orders (turns 1 and 4).

### 3.4 IT4 — Hotel v3 conditional GST gating

```
turn 1: hotel.book(nightly=3500, 2 nights, with_tax=8260, no gst) at v3 → MISSING_GST_NUMBER
turn 2: hotel.book(nightly=1800, 2 nights, with_tax=4248, no gst) at v3 → ok
turn 3: hotel.book(nightly=3500, 2 nights, with_tax=8260, gst="29ABCDE1234F1Z5") → ok
```

Assertions: `computed_total_inr` in turn-1 error matches server arithmetic; terminal state has 2 bookings.

### 3.5 IT5 — Side-channel notice lifecycle (consume-on-read)

```
turn 2: drift hotel.early_checkin_tnc fires → vendor_state.side_channel_notice set
turn 3: env.step calls emit_side_channel_if_pending BEFORE dispatch → notice surfaced in ToolResult.response["_notice"]; vendor_state.side_channel_notice cleared
turn 4: env.step → emit_side_channel_if_pending returns (None, same_state); no _notice on ToolResult
```

Assertions: `_notice` appears exactly once; at turn 4, `response.get("_notice") is None` (key absent).

### 3.6 IT6 — Every ToolResult.response JSON-roundtrips (catchall harness)

Parametrise over every unit test in §1; after each run, the shared fixture captures the `ToolResult` and asserts `json.loads(json.dumps(r.response)) == r.response`. Any failure here supersedes the per-test JSON assertion.

**Integration tests: 6 scenarios.**

---

## 4. Coverage Target

**Required:** 100% line coverage + ≥ 95% branch coverage on `driftcall/vendors/base.py`, `airline.py`, `cab.py`, `restaurant.py`, `hotel.py`, `payment.py`.

**Enforcement:**
```
python3 -m pytest tests/test_vendors/ \
  --cov=driftcall/vendors \
  --cov-report=term-missing \
  --cov-branch \
  --cov-fail-under=100
```
(`--cov-fail-under` enforces line; branch coverage is asserted by a separate parser of `coverage.xml` in CI — failing if any module under `driftcall/vendors/` drops below 95% branch.)

**Coverage mapping:**

| Surface | Covered by |
|---|---|
| **16 tool functions** (4+3+3+3+3) | U1–U54 (happy + error per version) + U84–U98 (idempotency + cascades) |
| **5 shared helpers per vendor** (`dispatch`, `initial_state`, `apply_schema_mutation`, `describe_schema`, `emit_side_channel_if_pending`) | U55–U83 |
| **14 mutation operators** | U67–U80 |
| **5 error statuses × all error codes** (table in vendors.md §5.2.1) | Distributed across U4, U9, U10, U13, U17, U19, U23, U27, U30, U33, U37, U39, U42, U47, U50, U53, U54, U91–U98 |
| **JSON roundtrip + integer INR** | U99–U100 + P5 + IT6 |
| **Determinism** | U61, U63, U101, P1 |
| **`now_ist` episode constancy + no wall-clock reads** | U63, U64 |

**Expected uncovered branches (< 5%):** defensive `else: raise UnknownSchemaVersionError` paths where a well-typed `Literal["v1","v2","v3"]` argument reaches the else branch only via deliberate test fixtures (covered by U82 pattern); `INTERNAL_SUM_MISMATCH` defensive assert in `cab._serialize_v3` (covered by a synthetic monkey-patched state that violates the invariant).

---

## 5. Fixtures

All fixtures defined in `tests/test_vendors/conftest.py`. Shared with `tests/test_env/conftest.py` via a common `tests/conftest.py` shim (single source of truth).

### 5.1 Vendor-state fixtures (15 total)

Per-domain × per-schema-version frozen `VendorState` instances. Canonical content; do not drift between test files.

```python
# tests/conftest.py (shim imported by both test_vendors/ and test_env/)

@pytest.fixture
def vendor_states_v1() -> dict[str, Any]:
    """All five domains at v1 baseline. Empty record dicts. Canonical policy/pricing/tnc."""
    return {
        "airline":    airline.initial_state(episode_seed=1234, goal=_GOAL_STUB),
        "cab":        cab.initial_state(episode_seed=1234, goal=_GOAL_STUB),
        "restaurant": restaurant.initial_state(episode_seed=1234, goal=_GOAL_STUB),
        "hotel":      hotel.initial_state(episode_seed=1234, goal=_GOAL_STUB),
        "payment":    payment.initial_state(episode_seed=1234, goal=_GOAL_STUB),
    }

@pytest.fixture
def vendor_states_v2(vendor_states_v1) -> dict[str, Any]:
    """All five domains advanced to v2 via the canonical v1→v2 mutation per domain."""
    return {
        "airline":    airline.apply_schema_mutation(vendor_states_v1["airline"],
                         {"rename": {"price": "total_fare_inr"}, "remove": ["currency"]}),
        "cab":        cab.apply_schema_mutation(vendor_states_v1["cab"],
                         {"enum_expand": {"vehicle_class_enum": ["suv","infant_seat_sedan"]}}),
        "restaurant": restaurant.apply_schema_mutation(vendor_states_v1["restaurant"],
                         {"numeric_bump": {"min_order_inr": 299}}),
        "hotel":      hotel.apply_schema_mutation(vendor_states_v1["hotel"],
                         {"time_window_shrink": {"cancel_window_hours": 6}}),
        "payment":    payment.apply_schema_mutation(vendor_states_v1["payment"],
                         {"auth_scope_bump": {"required_scope": "payments:write:v2"}}),
    }

@pytest.fixture
def vendor_states_v3(vendor_states_v2) -> dict[str, Any]:
    """All five domains advanced to v3 via the canonical v2→v3 mutation per domain."""
    return {
        "airline":    airline.apply_schema_mutation(vendor_states_v2["airline"],
                         {"require_new_field": {"passenger_count": "int"}}),
        "cab":        cab.apply_schema_mutation(vendor_states_v2["cab"],
                         {"pricing_restructure": {"fare_breakdown": True}}),
        "restaurant": restaurant.apply_schema_mutation(vendor_states_v2["restaurant"],
                         {"require_new_field": {"modifiers": "list[str]"},
                          "side_channel_notice_append": "veg_only now excludes egg dishes"}),
        "hotel":      hotel.apply_schema_mutation(vendor_states_v2["hotel"],
                         {"require_new_field": {"gst_number": "str"},
                          "policy_flag_flip": {"gst_required_threshold_inr": 7500}}),
        "payment":    payment.apply_schema_mutation(vendor_states_v2["payment"],
                         {"policy_flag_flip": {"mfa_threshold_inr": 5000}}),
    }
```

Per-domain aliases also exported for convenience:

```python
@pytest.fixture
def airline_v1(vendor_states_v1): return vendor_states_v1["airline"]
# ...same for airline_v2, airline_v3, cab_v1..payment_v3
```

### 5.2 Clock fixtures (2 total)

```python
@pytest.fixture
def now_ist_morning() -> datetime:
    """08:00 IST — inside school-hours window (07:00–09:00). Used by U17, cab policy tests."""
    return datetime(2026, 4, 25, 8, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

@pytest.fixture
def now_ist_evening() -> datetime:
    """18:30 IST — post 14:00 (airline booking-window-shrink), past 12:00 (hotel early-checkin)."""
    return datetime(2026, 4, 25, 18, 30, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
```

Both are episode-constant (vendors.md §3.5): within a test, `now_ist` MUST be passed unchanged across all `dispatch` calls.

### 5.3 JSON-roundtrip helper

```python
@pytest.fixture
def assert_json_roundtrip():
    def _assert(response: dict) -> None:
        assert json.loads(json.dumps(response)) == response, \
            f"ToolResult.response is not JSON-roundtrip-safe: {response!r}"
    return _assert
```

### 5.4 Hypothesis strategies module

`tests/test_vendors/strategies.py` (imported by property tests):

```python
ALL_TOOLS: tuple[str, ...] = (
    "airline.search", "airline.book", "airline.cancel", "airline.get_booking",
    "cab.estimate", "cab.book", "cab.cancel",
    "restaurant.search", "restaurant.order", "restaurant.track",
    "hotel.search", "hotel.book", "hotel.cancel",
    "payment.charge", "payment.refund", "payment.get_token",
)
WRITE_TOOLS: frozenset[str] = frozenset({
    "airline.book", "airline.cancel", "cab.book", "cab.cancel",
    "restaurant.order", "hotel.book", "hotel.cancel",
    "payment.charge", "payment.refund", "payment.get_token",
})
# vendor_state_strategy(), args_strategy(), mutation_strategy() — composed via @st.composite
```

### 5.5 Shared-with-env contract

`env_tests.md §5` imports these fixtures verbatim. If a field, name, or structure changes here, `env_tests.md` MUST be updated in the same PR. The `tests/conftest.py` shim enforces a single source of truth; test-file-local redefinition of these names is forbidden by a `conftest.py`-scope `pytest.fixture` collision check.

---

*End of vendors_tests.md.*
