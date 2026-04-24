# rewards_tests.md — Test Plan for `driftcall/rewards.py`

**Target module:** `driftcall/rewards.py`
**Spec doc:** `DRIFTCALL/docs/modules/rewards.md` (final, critic-gated)
**Framework:** `pytest` + `hypothesis`
**Owner:** Person B (Rewards & Tests)
**Implements:** DESIGN.md §7 test coverage; DRIFTCALL/CLAUDE.md §3.1 (nine-section test-plan doc)
**Numeric tolerance on worked examples:** `1e-9` (exact IEEE-754 float equality via `math.isclose(a, b, abs_tol=1e-9, rel_tol=0.0)`; `reward` values are 3-decimal-rounded, so tolerance is trivially exceeded but we still assert with `abs_tol=1e-9`).

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `driftcall/rewards.py`. Every public helper in `rewards.md §2` and every branch in `§3.2–§3.7` has at least one dedicated test. Every error mode in `§5` has a test raising `RewardComputationError`. Every edge case enumerated in `§7` has a test asserting its exact behavior.

Fixtures defined in §5 are **shared** with `training_tests.md` and `evaluation_tests.md` (same names, same canonicalised content). If any fixture content changes here, the shared copy in `tests/conftest.py` MUST be updated in lockstep.

---

## 1. Unit tests

**Organisation:** one `pytest` module per reward + combination helpers + error modes.
File layout under `tests/test_rewards/`:

```
tests/test_rewards/
  __init__.py
  test_r1_task_completion.py
  test_r2_drift_detection.py
  test_r3_constraint_adherence.py
  test_r4_format_compliance.py
  test_r5_anti_hack_penalty.py
  test_combine_quality.py
  test_brier_penalty.py
  test_apply_uncertain_floor.py
  test_final_reward.py
  test_compute_rewards_error_modes.py
  test_no_llm_judge.py
```

**Unit test case inventory — 34 cases total (exceeds the ≥ 30 requirement):**

### 1.1 `task_completion` (R1) — `test_r1_task_completion.py`

**Scope:** R1 ∈ {0.0, 1.0}. Binary, exact per-domain match, zero iff not terminated by SUBMIT.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_r1_airline_happy_returns_1` | Use `episode_happy_airline()`: `SUBMIT`, booking HYD→BLR 2026-04-30T19:15 total ₹7200, budget ₹8000, time_window "evening". | `task_completion(ep) == 1.0` |
| U2 | `test_r1_airline_wrong_route_returns_0` | Same as U1 but `vendor_states_final["airline"]["bookings"][0]["to"] = "DEL"`. | `task_completion(ep) == 0.0` |
| U3 | `test_r1_airline_over_budget_returns_0` | As U1 but booking `total = 9500 > budget 8000`. | `task_completion(ep) == 0.0` |
| U4 | `test_r1_airline_outside_time_window_returns_0` | As U1 but depart `2026-04-30T11:30`, window "evening". | `task_completion(ep) == 0.0` |
| U5 | `test_r1_not_submit_returns_0` | `terminated_by == "TIMEOUT"`, everything else like U1. | `task_completion(ep) == 0.0` |
| U6 | `test_r1_abort_returns_0` | `terminated_by == "ABORT"`. | `task_completion(ep) == 0.0` |
| U7 | `test_r1_anti_hack_returns_0` | `terminated_by == "ANTI_HACK"`. | `task_completion(ep) == 0.0` |
| U8 | `test_r1_unknown_domain_returns_0_and_flags` | `goal.domain = "spaceship"`. | `task_completion(ep) == 0.0`; `compute_rewards(ep).breakdown["r1"]["success_predicate"] == "unknown_domain"`. |
| U9 | `test_r1_cab_happy_returns_1` | cab domain fixture, pickup/drop/time match. | `== 1.0` |
| U10 | `test_r1_restaurant_dietary_mismatch_returns_0` | restaurant fixture, order has a non-veg item, `goal.constraints.dietary = "veg_only"`. | `== 0.0` (the success predicate includes dietary constraint per rewards.md §3.2). |
| U11 | `test_r1_hotel_happy_returns_1` | hotel fixture: correct city + dates + room type. | `== 1.0` |

### 1.2 `drift_detection` (R2) — `test_r2_drift_detection.py`

**Scope:** R2 ∈ {0.0, 0.5, 1.0}; three detection branches (speech / tool-args / structural-adaptation); substring match on SPEAK/CLARIFY text; substring match on TOOL_CALL args JSON + on string arg values; 3+ old-schema retries fail-fast.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U12 | `test_r2_stage1_returns_neutral` | `stage == 1`, `drift_log == ()`. | `drift_detection(ep) == 0.5` |
| U13 | `test_r2_no_drifts_returns_neutral` | `stage == 2`, `drift_log == ()`. | `drift_detection(ep) == 0.5` |
| U14 | `test_r2_speech_branch_hit_mentions_hint` | Drift `airline.price_rename` at turn 3 (`detection_hints = ["price","total_fare_inr"]`). Action `SPEAK` at turn 4: `"the price field seems renamed to total_fare_inr"`. | `== 1.0` AND `breakdown["r2"]["per_drift"][0]["hit_by_speech"] is True`. |
| U15 | `test_r2_speech_branch_case_insensitive` | Same drift, SPEAK text `"TOTAL_FARE_INR"` (uppercase). | `== 1.0` (case-insensitive substring). |
| U16 | `test_r2_speech_branch_clarify_also_counts` | Action type `CLARIFY` (not SPEAK) carrying the hint. | `== 1.0` (both SPEAK and CLARIFY are in the branch-1 allowlist per §3.3). |
| U17 | `test_r2_args_json_branch_hit` | Drift `airline.price_rename`. Action `TOOL_CALL` at turn 4: `tool_args = {"field": "total_fare_inr", "max": 8000}`. No SPEAK mentions it. | `== 1.0` AND `breakdown["r2"]["per_drift"][0]["hit_by_args_hint"] is True`. The hint `"total_fare_inr"` appears in the deterministic JSON payload. |
| U18 | `test_r2_args_string_values_branch_hit` | Drift with hint `"passenger_count"`. Action `TOOL_CALL` with `tool_args = {"filter_expr": "has passenger_count"}`. Hint is embedded in a string VALUE, not a key. | `== 1.0` (branch 2 checks the concatenated string values separately from the JSON payload). |
| U19 | `test_r2_args_branch_excludes_numeric_values` | Hint is `"8000"` (numeric-looking). `tool_args = {"max": 8000}` (integer value). The integer 8000 serialises in JSON as `8000` but must NOT register as a string-value hit. Assertion: if the agent also has NO speech/adaptation hit, R2 == 0.0 (numeric values are excluded per §3.3). | `drift_detection(ep) == 0.0` |
| U20 | `test_r2_adaptation_branch_hit` | Drift `airline.price_rename`. Post-drift schema requires `total_fare_inr`. Action `TOOL_CALL` `book` at turn 5 uses `{"total_fare_inr": 7200}`. SPEAK never mentions the drift. | `== 1.0` AND `breakdown["r2"]["per_drift"][0]["hit_by_adaptation"] is True`. |
| U21 | `test_r2_miss_outside_window_returns_0` | Drift at turn 3, SPEAK mentions hint at turn 7 (outside window [3, 4, 5]). | `== 0.0` |
| U22 | `test_r2_three_plus_old_schema_retries_returns_0` | Drift at turn 3 renames `price→total_fare_inr`. Agent calls OLD schema (`{"price": ...}`) at turns 4, 5, 6, 7 (≥ 3 consecutive). | `== 0.0` AND `breakdown["r2"]["three_plus_retries"] is True`. |
| U23 | `test_r2_empty_hints_raises` | `drift_log[0].detection_hints == []`. | Raises `RewardComputationError` with substring `"empty detection_hints"`. |
| U24 | `test_r2_all_empty_string_hints_raises` | `detection_hints = ["", "   ", ""]` (all empty/whitespace). | Raises `RewardComputationError`. |
| U25 | `test_r2_any_single_drift_miss_fails_whole_episode` | Two drifts: first detected, second missed. | `== 0.0` (per §3.3 "one miss → whole-episode miss"). |

### 1.3 `constraint_adherence` (R3) — `test_r3_constraint_adherence.py`

**Scope:** R3 ∈ [0, 1], arithmetic = satisfied / total; empty dict → 1.0; unknown keys count as satisfied.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U26 | `test_r3_no_constraints_returns_one` | `goal.constraints == {}`. | `constraint_adherence(ep) == 1.0` |
| U27 | `test_r3_all_satisfied_returns_one` | 2/2 constraints satisfied. | `== 1.0` |
| U28 | `test_r3_half_satisfied_returns_half` | `{budget_inr: 8000 (satisfied), time_window: "morning" (violated)}`. | `math.isclose(constraint_adherence(ep), 0.5, abs_tol=1e-9)` |
| U29 | `test_r3_none_satisfied_returns_zero` | 0/3 constraints satisfied. | `== 0.0` |
| U30 | `test_r3_unknown_key_counts_as_satisfied` | `{carbon_offset: True}`. | `== 1.0` AND `breakdown["r3"]["unknown_constraints"] == ["carbon_offset"]`. |
| U31 | `test_r3_mixed_known_and_unknown` | `{budget_inr: 8000 (satisfied), carbon_offset: True (unknown=satisfied)}`. | `== 1.0`; `unknown_constraints == ["carbon_offset"]`. |

### 1.4 `format_compliance` (R4) — `test_r4_format_compliance.py`

**Scope:** R4 ∈ [0, 1] deductive from 1.0; −0.20 invalid JSON; −0.10 unknown tool; −0.05 missing rationale; −0.10 language mismatch; clamp at 0.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U32 | `test_r4_all_clean_returns_one` | Fixture with all valid TOOL_CALLs, rationales, matching language. | `format_compliance(ep) == 1.0` |
| U33 | `test_r4_invalid_json_deducts_02` | One TOOL_CALL with `tool_args = <non-JSON-serializable sentinel>`. | `math.isclose(format_compliance(ep), 0.80, abs_tol=1e-9)` |
| U34 | `test_r4_unknown_tool_deducts_01` | One TOOL_CALL with `tool_name = "magic.teleport"` (not in registry). | `math.isclose(format_compliance(ep), 0.90, abs_tol=1e-9)` |
| U35 | `test_r4_missing_rationale_deducts_005` | One TOOL_CALL with `rationale = ""`. | `math.isclose(format_compliance(ep), 0.95, abs_tol=1e-9)` |
| U36 | `test_r4_missing_rationale_none_deducts_005` | `rationale = None`. | `math.isclose(format_compliance(ep), 0.95, abs_tol=1e-9)` |
| U37 | `test_r4_language_mismatch_deducts_01` | SPEAK in English when `goal.language == "hi"` (Devanagari expected). | `math.isclose(format_compliance(ep), 0.90, abs_tol=1e-9)` |
| U38 | `test_r4_hinglish_code_mixed_accepted` | `goal.language == "hinglish"`, SPEAK mixes Latin + Devanagari. | `== 1.0` (no deduction). |
| U39 | `test_r4_floor_at_zero` | Many cumulative deductions pushing below 0 (e.g., 6× invalid JSON = −1.2). | `format_compliance(ep) == 0.0` (clamped). |
| U40 | `test_r4_stacked_deductions_arithmetic` | One invalid JSON + one unknown tool + one missing rationale + one lang mismatch = 1.0 − 0.20 − 0.10 − 0.05 − 0.10 = 0.55. | `math.isclose(format_compliance(ep), 0.55, abs_tol=1e-9)` |

### 1.5 `anti_hack_penalty` (R5) — `test_r5_anti_hack_penalty.py`

**Scope:** R5 ∈ [−1, 0]; five offense codes (a–e); additive stacking with floor at −1.0; recursive whitelist walk.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U41 | `test_r5_no_offenses_returns_zero` | Clean episode. | `anti_hack_penalty(ep) == 0.0` |
| U42 | `test_r5_hallucinated_field_minus_one` | Agent SPEAK: `"using flight_total_with_gst field"`; `flight_total_with_gst` never in any `tool_result.response`. | `anti_hack_penalty(ep) == -1.0` AND `breakdown["anti_hack"]["offenses"]` contains a code `"hallucinated_field"`. |
| U43 | `test_r5_hallucination_whitelist_recursive_nested_key` | `tool_result.response = {"fare_breakdown": {"surge": 45, "base": 120}}`. Agent SPEAK: `"the surge component is ₹45"`. | `anti_hack_penalty(ep) == 0.0` (`surge` is whitelisted at depth 2). |
| U44 | `test_r5_hallucination_whitelist_primitive_leaf` | `tool_result.response = {"eta_min": 7}`. Agent SPEAK: `"ETA is 7 minutes"`. | `anti_hack_penalty(ep) == 0.0` (`7` is whitelisted as primitive leaf). |
| U45 | `test_r5_hallucination_nested_miss` | Agent SPEAK: `"base_fare is ₹120"`; whitelist contains `base` and `120` but NOT `base_fare`. | `anti_hack_penalty(ep) == -1.0`. |
| U46 | `test_r5_repeated_identical_tool_calls_minus_05` | 4 identical `airline.search(from=HYD, to=BLR, date=2026-04-30)` calls. | `anti_hack_penalty(ep) == -0.5`; offense code `"repeated_tool_calls"`. |
| U47 | `test_r5_repeated_exactly_three_no_penalty` | Exactly 3 identical calls (threshold is `> 3`). | `anti_hack_penalty(ep) == 0.0`. |
| U48 | `test_r5_probe_schema_abuse_minus_05` | 3× `PROBE_SCHEMA` actions. | `anti_hack_penalty(ep) == -0.5`; offense code `"probe_schema_abuse"`. |
| U49 | `test_r5_probe_schema_two_no_penalty` | Exactly 2 `PROBE_SCHEMA` (threshold is `≥ 3`). | `anti_hack_penalty(ep) == 0.0`. |
| U50 | `test_r5_bare_drift_assertion_minus_03` | SPEAK at turn 2 says `"drift detected"`; no `drift_log` entry with turn ≤ 2; no prior `tool_result.status` in {schema_error, policy_error, auth_error}. | `anti_hack_penalty(ep) == -0.3`; offense code `"bare_drift_assertion"`. |
| U51 | `test_r5_substantiated_drift_claim_ok` | Same but a `tool_result` at turn 1 has `status = "schema_error"`. | `anti_hack_penalty(ep) == 0.0`. |
| U52 | `test_r5_protected_write_minus_02_reserved_key` | TOOL_CALL with `tool_args = {"__turn__": 99}`. | `anti_hack_penalty(ep) == -0.2`; offense code `"protected_write"`. |
| U53 | `test_r5_protected_write_minus_02_protected_tool` | TOOL_CALL with `tool_name` in protected allowlist (e.g., `"__env_reset__"`). | `anti_hack_penalty(ep) == -0.2`. |
| U54 | `test_r5_stacking_clamps_at_minus_one` | Hallucinated field (−1.0) + repeated calls (−0.5) + probe abuse (−0.5). Raw sum = −2.0. | `anti_hack_penalty(ep) == -1.0` (floor); `offenses` list has 3 entries. |

### 1.6 `brier_penalty` — `test_brier_penalty.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U55 | `test_brier_none_confidence_zero` | `confidence = None`. | `brier_penalty(None, 1.0) == 0.0` |
| U56 | `test_brier_perfect_calibration_zero` | `confidence = 1.0, r1 = 1.0`. | `brier_penalty(1.0, 1.0) == 0.0` |
| U57 | `test_brier_max_miscalibration_clamps_05` | `confidence = 1.0, r1 = 0.0`. Raw `(1-0)^2 = 1.0`. | `brier_penalty(1.0, 0.0) == 0.5` (cap). |
| U58 | `test_brier_mid_miscalibration_raw` | `confidence = 0.6, r1 = 0.0`. Raw `0.36`. | `math.isclose(brier_penalty(0.6, 0.0), 0.36, abs_tol=1e-9)` |
| U59 | `test_brier_underconfidence_on_success` | `confidence = 0.0, r1 = 1.0`. Raw `1.0`. | `brier_penalty(0.0, 1.0) == 0.5` |

### 1.7 `apply_uncertain_floor` — `test_apply_uncertain_floor.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U60 | `test_floor_activates_r1_zero_conf_low` | `reward = 0.096, r1 = 0.0, confidence = 0.2`. | `apply_uncertain_floor(0.096, 0.0, 0.2) == 0.3` |
| U61 | `test_floor_not_applied_when_r1_one` | `reward = 0.096, r1 = 1.0, confidence = 0.2`. | `apply_uncertain_floor(0.096, 1.0, 0.2) == 0.096` (identity). |
| U62 | `test_floor_not_applied_when_conf_at_threshold` | `reward = 0.096, r1 = 0.0, confidence = 0.3`. Threshold is strict `< 0.3`. | `apply_uncertain_floor(0.096, 0.0, 0.3) == 0.096`. |
| U63 | `test_floor_not_applied_when_conf_none` | `confidence = None`. | `apply_uncertain_floor(0.096, 0.0, None) == 0.096`. |
| U64 | `test_floor_never_lowers` | `reward = 0.5, r1 = 0.0, confidence = 0.2`. Already above floor. | `apply_uncertain_floor(0.5, 0.0, 0.2) == 0.5` (no change). |

### 1.8 `combine_quality` — `test_combine_quality.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U65 | `test_combine_all_max` | `(1, 1, 1, 1, 0)`. | `math.isclose(combine_quality(1,1,1,1,0), 0.95, abs_tol=1e-9)` |
| U66 | `test_combine_all_zero` | `(0, 0, 0, 0, 0)`. | `combine_quality(0,0,0,0,0) == 0.0` |
| U67 | `test_combine_r5_negative_subtracts` | `(1, 1, 1, 1, -1)` → `0.5+0.2+0.15+0.1 + 0.05*(-1) = 0.90`. | `math.isclose(combine_quality(1,1,1,1,-1), 0.90, abs_tol=1e-9)` |
| U68 | `test_combine_r2_half` | `(0, 0.5, 0, 0, 0)`. | `math.isclose(combine_quality(0,0.5,0,0,0), 0.10, abs_tol=1e-9)` |
| U69 | `test_combine_does_not_clamp_or_round` | `(0, 0, 0, 0, -1)` → `-0.05`. Helper must return raw float. | `math.isclose(combine_quality(0,0,0,0,-1), -0.05, abs_tol=1e-9)` (NOT clamped to 0). |

### 1.9 `final_reward` — `test_final_reward.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U70 | `test_final_clamps_negative_to_zero` | `quality = -0.05, brier = 0.0, r1 = 0.0, conf = None`. Pre-clamp `-0.05`. | `final_reward(-0.05, 0.0, 0.0, None) == 0.0` |
| U71 | `test_final_clamps_above_one_to_one` | Pathological `quality = 1.5, brier = 0.0`. | `final_reward(1.5, 0.0, 1.0, None) == 1.0` |
| U72 | `test_final_rounds_to_three_decimals` | `quality = 0.850, brier = 0.0225, r1 = 1, conf = 0.85`. Pre-round `0.8309125`. | `final_reward(0.850, 0.0225, 1.0, 0.85) == 0.831` |
| U73 | `test_final_floor_then_clamp_then_round` | Floor applies (`r1=0, conf=0.2`), pre-floor `0.048` → floor to `0.3`. | `final_reward(0.050, 0.04, 0.0, 0.2) == 0.3` |

### 1.10 Error-mode unit tests — `test_compute_rewards_error_modes.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U74 | `test_missing_goal_raises` | `episode.goal = None`. | Raises `RewardComputationError`, `.reason contains "goal"`. |
| U75 | `test_unterminated_raises` | `terminated_by = None`. | Raises `RewardComputationError`, `.reason contains "not terminated"`. |
| U76 | `test_unknown_drift_type_raises` | `drift.drift_type = "warp_drive"`. | Raises `RewardComputationError`, `.reason contains "unknown drift_type"`. |
| U77 | `test_nan_in_confidence_raises` | `confidence = float("nan")` reaches compute. | Raises `RewardComputationError("non-finite value ...")`. |
| U78 | `test_inf_in_confidence_raises` | `confidence = float("inf")`. | Raises `RewardComputationError`. |
| U79 | `test_confidence_clamp_out_of_range` | `confidence = 1.5` (out-of-[0,1] but finite). | Does NOT raise; `breakdown["combination"]["confidence_clamped"] is True`; Brier uses clamped value 1.0. |
| U80 | `test_actions_toolresults_count_mismatch_raises` | `len(actions with TOOL_CALL) != len(tool_results)`. | Raises `RewardComputationError`, `.reason contains "action/tool_result count mismatch"`. |
| U81 | `test_empty_actions_no_raise` | `actions = ()`, `tool_results = ()`, timeout. | Does NOT raise. R1=0, R4=1.0, R5=0.0; R2=0.5 for stage 1. |

**Total unit tests: 81 cases. Requirement was ≥ 30. Delivered 81.**

---

## 2. Property tests (hypothesis)

**Framework:** `hypothesis`. Strategies live in `tests/strategies/rewards.py` (shared with `training_tests.md` and `evaluation_tests.md`).

| # | Name | Invariant | Strategy |
|---|---|---|---|
| P1 | `prop_reward_in_unit_interval` | For any valid `Episode`, `0.0 <= compute_rewards(ep).reward <= 1.0`. Asserts the hard invariant in rewards.md §3.7. | `st.builds(Episode, ...)` with constrained slots; 1000 examples. |
| P2 | `prop_compute_rewards_is_pure` | Given one `Episode` value, `compute_rewards` called 1000 times returns equal `Rewards` each time (dataclass equality). Asserts §3.1 determinism. | Build one example, assert equality across 1000 calls. |
| P3 | `prop_r5_in_minus_one_to_zero` | For any valid `Episode`, `-1.0 <= Rewards.r5 <= 0.0`. Asserts §3.6 range. | `st.builds(Episode, ...)`; 500 examples. |
| P4 | `prop_weighted_sum_rule` | `math.isclose(Rewards.quality, 0.50*r1 + 0.20*r2 + 0.15*r3 + 0.10*r4 + 0.05*min(r5,0), abs_tol=1e-9)` for every example. | `st.builds(Rewards subcomponents)`; 500 examples. |
| P5 | `prop_r1_is_binary` | `Rewards.r1 in {0.0, 1.0}` always. | 500 examples. |
| P6 | `prop_r2_is_ternary` | `Rewards.r2 in {0.0, 0.5, 1.0}` always. | 500 examples. |
| P7 | `prop_floor_only_when_conditions_met` | `Rewards.floor_applied is True` iff `(r1 == 0.0) AND (confidence is not None) AND (confidence < 0.3) AND (reward == 0.3 AND pre_floor_reward < 0.3)`. | 500 examples covering both halves of the biconditional. |
| P8 | `prop_episode_frozen_not_mutated` | Snapshot `hash(ep)` before `compute_rewards(ep)`; assert equal after. Confirms no mutation per §3.1. | 500 examples. |
| P9 | `prop_brier_in_zero_half` | `0.0 <= Rewards.brier <= 0.5`. Asserts §2.3 `brier_penalty` post. | 500 examples. |

**Total properties: 9. Requirement was ≥ 5. Delivered 9.**

---

## 3. Integration tests

**Scope:** Full `Episode → compute_rewards` round trip using production-shaped fixtures. Covers the three worked examples in rewards.md §3.7 (Example A/B/C — actually documented in rewards.md §8.1/§8.2/§8.3 — §3.7 references the formula). Exact numeric equality required.

File: `tests/test_rewards_integration.py`.

### 3.1 Worked examples (rewards.md §8) — exact numeric reproduction

| # | Name | Fixture | Expected `reward` | Supporting assertions |
|---|---|---|---|---|
| I1 | `test_example_A_clean_success_0_831` | `episode_happy_airline()` (stage 1, conf=0.85, SUBMIT, booking matches). | `math.isclose(r.reward, 0.831, abs_tol=1e-9)` | `r.r1 == 1.0`, `r.r2 == 0.5`, `r.r3 == 1.0`, `r.r4 == 1.0`, `r.r5 == 0.0`, `math.isclose(r.quality, 0.850, abs_tol=1e-9)`, `math.isclose(r.brier, 0.0225, abs_tol=1e-9)`, `r.floor_applied is False`. |
| I2 | `test_example_B_drift_detected_over_budget_0_240` | `episode_drift_detected()` (stage 2, Kannada, drift `airline.price_rename` at turn 3, SPEAK mentions `total_fare_inr`, books ₹8400 > budget ₹8000, conf=0.60). | `math.isclose(r.reward, 0.240, abs_tol=1e-9)` | `r.r1 == 0.0`, `r.r2 == 1.0`, `r.r3 == 0.5`, `r.r4 == 1.0`, `r.r5 == 0.0`, `math.isclose(r.quality, 0.375, abs_tol=1e-9)`, `math.isclose(r.brier, 0.36, abs_tol=1e-9)`, `r.floor_applied is False`. |
| I3 | `test_example_C_hallucination_surrender_0_300` | `episode_hallucinated_field()` (stage 3 Tamil, 2 drifts, invents `order_metadata_v4`, 4× repeated search, conf=0.20, no order placed). | `math.isclose(r.reward, 0.300, abs_tol=1e-9)` | `r.r1 == 0.0`, `r.r2 == 0.0`, `r.r3 == 0.0`, `r.r4 == 1.0`, `r.r5 == -1.0` (floor from −1 + −0.5), `math.isclose(r.quality, 0.050, abs_tol=1e-9)`, `math.isclose(r.brier, 0.04, abs_tol=1e-9)`, `r.floor_applied is True`. Pre-floor reward `0.048`, floored to `0.3`. |

### 3.2 Additional integration scenarios

| # | Name | Fixture | Assertion |
|---|---|---|---|
| I4 | `test_timeout_no_confidence` | `episode_timeout()` (TIMEOUT, conf=None). | `r.floor_applied is False`, `r.brier == 0.0`, reward = `0.20*r2 + 0.15*r3 + 0.10*r4 + 0.05*min(r5,0)` clamped+rounded. |
| I5 | `test_uncertain_floor_activation_via_env` | `episode_uncertain_floor_activation()` (r1=0, conf=0.1, low quality). | `r.floor_applied is True`, `r.reward == 0.3`. |
| I6 | `test_breakdown_populated_for_all_rewards` | Any fixture. | `set(r.breakdown.keys()) >= {"r1","r2","r3","r4","anti_hack","combination"}`; every subdict has the fields spec'd in rewards.md §4.2. |
| I7 | `test_rewards_frozen_output` | Any fixture. | `r.__dataclass_params__.frozen is True`; mutation attempt raises `FrozenInstanceError`. |
| I8 | `test_rewards_asdict_roundtrip_json` | Any fixture. | `json.loads(json.dumps(dataclasses.asdict(r)))` equals the asdict form (round-trip check). |

---

## 4. Coverage target

### 4.1 Line + branch coverage

- **Target:** 100% line coverage and ≥ 95% branch coverage on `driftcall/rewards.py` and every submodule under `driftcall/rewards/` (`checkers.py`, `parsers.py`).
- **Tool:** `pytest --cov=driftcall.rewards --cov-branch --cov-report=term-missing --cov-fail-under=100` for lines; branch threshold checked via `--cov-report=xml` and a CI script that asserts the `<branches-covered>/<branches-valid>` ratio ≥ 0.95.
- **CI gate:** in `pyproject.toml` / `pytest.ini`:
  ```
  [tool.coverage.report]
  fail_under = 100
  show_missing = true
  skip_covered = false
  ```
  Plus a separate `scripts/check_branch_coverage.py` that parses the XML and fails if branch coverage < 95%.

### 4.2 No-LLM-judge enforcement test

File: `tests/test_no_llm_judge.py`.

Purpose: encode the rewards.md §3.1 and §6.3 invariant that the reward module MUST NOT import any LLM / network / non-determinism package. This is the hardest invariant in the system and must be enforced by an automated test, not just code review.

```python
# Pseudocode — see the actual test file for exact implementation
FORBIDDEN_MODULES = frozenset({
    "openai", "anthropic", "transformers", "torch",
    "unsloth", "requests", "httpx", "aiohttp",
    "google.generativeai", "cohere", "mistralai",
    "vllm", "llama_cpp", "llm",
})
FORBIDDEN_ATTRS = frozenset({
    ("time", "time"), ("datetime", "now"), ("datetime", "utcnow"),
    ("random", "random"), ("random", "randint"), ("random", "choice"),
})

def test_rewards_module_has_no_forbidden_imports():
    """Static AST scan of driftcall/rewards.py + driftcall/rewards/*.py."""
    import ast, pathlib
    root = pathlib.Path(driftcall.rewards.__file__).parent
    files = list(root.rglob("*.py"))
    assert files, "expected at least rewards.py"
    offenders: list[str] = []
    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_MODULES:
                        offenders.append(f"{f}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in FORBIDDEN_MODULES:
                        offenders.append(f"{f}:{node.lineno} from {node.module}")
    assert offenders == [], f"forbidden imports detected: {offenders}"

def test_rewards_module_does_not_hit_network_or_clock_at_runtime():
    """Runtime monkeypatch: make forbidden calls raise; run compute_rewards on
    every shared fixture; confirm none of the forbidden hooks fired."""
    # monkeypatch socket.socket → raise, time.time → raise, datetime.now → raise,
    # random.random → raise, then call compute_rewards(episode_happy_airline())
    # and every other fixture. None must raise.
```

**Assertion:** any future `import openai`, `import anthropic`, `import torch`, `import requests`, `time.time()`, `datetime.now()`, or `random.*` call in `driftcall/rewards.py` (or its submodules) MUST fail the test suite. This gates every PR against accidental non-determinism or LLM-judge reintroduction.

---

## 5. Fixtures

**Location:** `tests/conftest.py` (auto-discovered by pytest at collection). **Shared** with `training_tests.md` and `evaluation_tests.md` — same names, identical bodies, single source of truth. Any change here MUST be reflected in the other two test plans on the same PR.

All fixtures return a frozen `Episode` dataclass conforming to `models.md`. Helper builders (`_make_action`, `_make_tool_result`, `_make_drift_event`) live in `tests/builders.py`. Each fixture is also registered in `tests/strategies/rewards.py` as a `hypothesis` base-case example used by the property tests.

### 5.1 `episode_happy_airline()`

```python
@pytest.fixture
def episode_happy_airline() -> Episode:
    """Stage 1 Hinglish airline booking, clean success, confidence=0.85.
    Reproduces rewards.md §8.1 Example A. Expected reward == 0.831."""
    goal = GoalSpec(
        domain="airline",
        language="hinglish",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "evening"},
    )
    actions = (
        _make_action(turn=1, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR", "date": "2026-04-30"},
                     rationale="Searching HYD-BLR evening flights under 8000 INR"),
        _make_action(turn=2, type=ActionType.TOOL_CALL, tool_name="airline.book",
                     tool_args={"flight_id": "6E-123", "passenger": "test"},
                     rationale="Booking the 19:15 flight at 7200 INR, within budget"),
        _make_action(turn=3, type=ActionType.SUBMIT, confidence=0.85,
                     rationale="Booking confirmed, within budget and evening window"),
    )
    tool_results = (
        _make_tool_result(turn=1, tool_name="airline.search", status="ok",
                          response={"flights": [{"id": "6E-123", "depart": "2026-04-30T19:15",
                                                 "price": 7200, "from": "HYD", "to": "BLR"}]}),
        _make_tool_result(turn=2, tool_name="airline.book", status="ok",
                          response={"booking_id": "B1", "total": 7200}),
    )
    return Episode(
        episode_id="ep_happy_airline_001",
        goal=goal, actions=actions, tool_results=tool_results,
        drift_log=(), vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T19:15",
             "total": 7200, "booking_id": "B1"}]}},
        schema_versions_final={"airline": "v1"},
        max_turns=10, turns_used=3,
        terminated_by="SUBMIT", stage=1,
    )
```

### 5.2 `episode_drift_detected()`

```python
@pytest.fixture
def episode_drift_detected() -> Episode:
    """Stage 2 Kannada airline, drift airline.price_rename at turn 3, agent
    detects via SPEAK, re-books over budget (₹8400 > ₹8000), confidence=0.60.
    Reproduces rewards.md §8.2 Example B. Expected reward == 0.240."""
    goal = GoalSpec(
        domain="airline", language="kn",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "morning"},
    )
    drift = _make_drift_event(
        pattern_id="airline.price_rename", turn=3,
        drift_type="schema",
        detection_hints=("price", "total_fare_inr"),
        mutation={"kind": "rename", "from": "price", "to": "total_fare_inr"},
    )
    actions = (
        _make_action(turn=1, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR"}, rationale="initial search"),
        _make_action(turn=2, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR", "filter": "morning"},
                     rationale="filter morning"),
        _make_action(turn=3, type=ActionType.SPEAK,
                     message="price field seems renamed to total_fare_inr, retrying",
                     rationale="drift noticed"),
        _make_action(turn=4, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR", "max_total_fare_inr": 8500},
                     rationale="using new field"),
        _make_action(turn=5, type=ActionType.TOOL_CALL, tool_name="airline.book",
                     tool_args={"flight_id": "6E-200", "total_fare_inr": 8400},
                     rationale="booking best morning option"),
        _make_action(turn=6, type=ActionType.SUBMIT, confidence=0.60,
                     rationale="booked morning flight, slightly over budget"),
    )
    # tool_results mirror actions 1,2,4,5 (TOOL_CALL only)
    tool_results = (
        _make_tool_result(turn=1, tool_name="airline.search", status="ok",
                          response={"flights": [{"id": "6E-100", "price": 7500,
                                                 "depart": "2026-04-30T07:00"}]}),
        _make_tool_result(turn=2, tool_name="airline.search", status="schema_error",
                          response={"error": "unknown field: price; did you mean total_fare_inr?"}),
        _make_tool_result(turn=4, tool_name="airline.search", status="ok",
                          response={"flights": [{"id": "6E-200", "total_fare_inr": 8400,
                                                 "depart": "2026-04-30T09:30"}]}),
        _make_tool_result(turn=5, tool_name="airline.book", status="ok",
                          response={"booking_id": "B2", "total_fare_inr": 8400}),
    )
    return Episode(
        episode_id="ep_drift_002",
        goal=goal, actions=actions, tool_results=tool_results,
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": [
            {"from": "HYD", "to": "BLR", "depart": "2026-04-30T09:30",
             "total": 8400, "booking_id": "B2"}]}},
        schema_versions_final={"airline": "v2"},
        max_turns=10, turns_used=6, terminated_by="SUBMIT", stage=2,
    )
```

### 5.3 `episode_hallucinated_field()`

```python
@pytest.fixture
def episode_hallucinated_field() -> Episode:
    """Stage 3 Tamil restaurant compound drift, agent invents `order_metadata_v4`,
    4× repeated search, confidence=0.20, no order placed.
    Reproduces rewards.md §8.3 Example C. Expected reward == 0.300 (floor)."""
    goal = GoalSpec(
        domain="restaurant", language="ta",
        slots={"city": "CHN", "cuisine": "south-indian"},
        constraints={"budget_inr": 300, "dietary": "veg"},
    )
    drift_policy = _make_drift_event(
        pattern_id="restaurant.min_order_bump", turn=3, drift_type="policy",
        detection_hints=("cutoff", "closing_time"),
        mutation={"kind": "policy", "cutoff_after": "21:00"})
    drift_schema = _make_drift_event(
        pattern_id="restaurant.items_shape_bump", turn=7, drift_type="schema",
        detection_hints=("order_metadata",),
        mutation={"kind": "add", "field": "order_metadata"})
    actions = tuple(
        _make_action(turn=i, type=ActionType.TOOL_CALL, tool_name="restaurant.search",
                     tool_args={"city": "CHN", "cuisine": "south-indian"},
                     rationale="searching")
        for i in range(1, 5)
    ) + (
        _make_action(turn=5, type=ActionType.SPEAK,
                     message="Using the order_metadata_v4 field to format the order",
                     rationale="invented field for format"),
        _make_action(turn=6, type=ActionType.SUBMIT, confidence=0.20,
                     rationale="giving up, uncertain"),
    )
    tool_results = tuple(
        _make_tool_result(turn=i, tool_name="restaurant.search", status="ok",
                          response={"places": [{"id": "R1", "name": "Saravana", "veg": True}]})
        for i in range(1, 5)
    )
    return Episode(
        episode_id="ep_hallucinated_003",
        goal=goal, actions=actions, tool_results=tool_results,
        drift_log=(drift_policy, drift_schema),
        vendor_states_final={"restaurant": {"orders": []}},
        schema_versions_final={"restaurant": "v3"},
        max_turns=10, turns_used=6, terminated_by="SUBMIT", stage=3,
    )
```

### 5.4 `episode_timeout()`

```python
@pytest.fixture
def episode_timeout() -> Episode:
    """Stage 2 airline, agent churns 10 turns without SUBMIT, TIMEOUT, conf=None."""
    goal = GoalSpec(
        domain="airline", language="en",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000})
    drift = _make_drift_event(
        pattern_id="airline.price_rename", turn=3, drift_type="schema",
        detection_hints=("price", "total_fare_inr"),
        mutation={"kind": "rename", "from": "price", "to": "total_fare_inr"})
    actions = tuple(
        _make_action(turn=i, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR"}, rationale="retry")
        for i in range(1, 11)
    )
    tool_results = tuple(
        _make_tool_result(turn=i, tool_name="airline.search",
                          status="ok" if i < 3 else "schema_error",
                          response={"flights": []} if i < 3 else
                                   {"error": "unknown field price"})
        for i in range(1, 11)
    )
    return Episode(
        episode_id="ep_timeout_004", goal=goal,
        actions=actions, tool_results=tool_results,
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": []}},
        schema_versions_final={"airline": "v2"},
        max_turns=10, turns_used=10,
        terminated_by="TIMEOUT", stage=2,
    )
```

### 5.5 `episode_uncertain_floor_activation()`

```python
@pytest.fixture
def episode_uncertain_floor_activation() -> Episode:
    """Stage 2 airline, agent submits with low confidence=0.1, r1=0, low quality.
    Asserts the uncertain floor path in isolation from the hallucination case.
    Expected: pre-floor reward < 0.3, post-floor reward == 0.3, floor_applied True."""
    goal = GoalSpec(
        domain="airline", language="en",
        slots={"from": "HYD", "to": "BLR", "when": "2026-04-30"},
        constraints={"budget_inr": 8000, "time_window": "morning"})
    drift = _make_drift_event(
        pattern_id="airline.price_rename", turn=2, drift_type="schema",
        detection_hints=("price", "total_fare_inr"),
        mutation={"kind": "rename", "from": "price", "to": "total_fare_inr"})
    actions = (
        _make_action(turn=1, type=ActionType.TOOL_CALL, tool_name="airline.search",
                     tool_args={"from": "HYD", "to": "BLR"}, rationale="search"),
        _make_action(turn=2, type=ActionType.SPEAK,
                     message="I'm not sure how to handle this, the schema seems off",
                     rationale="expressing uncertainty"),
        _make_action(turn=3, type=ActionType.SUBMIT, confidence=0.1,
                     rationale="giving up calibrated"),
    )
    tool_results = (
        _make_tool_result(turn=1, tool_name="airline.search", status="schema_error",
                          response={"error": "unknown field price"}),
    )
    return Episode(
        episode_id="ep_floor_005", goal=goal,
        actions=actions, tool_results=tool_results,
        drift_log=(drift,),
        vendor_states_final={"airline": {"bookings": []}},
        schema_versions_final={"airline": "v2"},
        max_turns=10, turns_used=3,
        terminated_by="SUBMIT", stage=2,
    )
```

### 5.6 Fixture contract (shared-registry invariant)

A CI test `tests/test_fixture_registry.py` asserts that the five fixtures listed above are importable from `tests.conftest` and that their `episode_id` prefixes are unique (`ep_happy_airline_`, `ep_drift_`, `ep_hallucinated_`, `ep_timeout_`, `ep_floor_`). This enforces that `training_tests.md` and `evaluation_tests.md` share the same fixture identities — any rename breaks the whole suite at once, forcing a coordinated update.

---

## 6. Summary (auto-check)

| Metric | Target | Delivered |
|---|---|---|
| Unit test cases | ≥ 30 | **81** |
| Property test invariants | ≥ 5 | **9** |
| Worked-example numeric reproductions (§8.1/§8.2/§8.3 of rewards.md) | 3 | **3** (0.831, 0.240, 0.300) |
| Line coverage on `driftcall/rewards.py` | 100% | enforced via `--cov-fail-under=100` |
| Branch coverage | ≥ 95% | enforced via `scripts/check_branch_coverage.py` |
| No-LLM-judge enforcement | 1 static + 1 runtime test | both specified in §4.2 |
| Shared fixtures | 5 | `episode_happy_airline`, `episode_drift_detected`, `episode_hallucinated_field`, `episode_timeout`, `episode_uncertain_floor_activation` |

**End of test plan. Implementation (`tests/test_rewards/*.py`, `tests/test_rewards_integration.py`, `tests/test_no_llm_judge.py`, `tests/conftest.py`, `tests/builders.py`, `tests/strategies/rewards.py`) does not start until ≥ 2 fresh critic agents return `NOTHING_FURTHER` on this doc.**
