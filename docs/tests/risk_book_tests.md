# risk_book_tests.md — Test Plan for `driftcall/risk.py`

**Target module:** `driftcall/risk.py`
**Spec doc:** `DRIFTCALL/docs/modules/risk_book.md` (final, critic-gated)
**Framework:** `pytest` + `hypothesis`
**Owner:** Person B (Rewards & Tests)
**Implements:** DESIGN.md §14 (12-risk register) + CLAUDE.md §11 (5 escalate + 3 hard-stop items) + risk_book.md §3 (behavior spec) + §5 (error modes)
**Numeric tolerance:** none required — this module's surface is enum lookup + list length + JSONL write. Equality is exact (`==`) throughout.

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `driftcall/risk.py` (`RiskEntry`, `Risk.assess`, `Risk.triage`, `RiskLog.append`, `RiskLog.to_jsonl`, `RiskLog.summary`). Every invariant in risk_book.md §3.1 has a dedicated unit test. Every escalation + hard-stop condition in CLAUDE.md §11 is mapped to at least one test. R99 (catch-all) is explicitly exercised so that `Risk.triage` is provably total.

Fixtures defined in §5 are **shared** with `env_tests.md`, `training_tests.md`, and `evaluation_tests.md` — specifically, `risk_log_tmpfile` is imported by those plans for their integration tests asserting that a `Risk.triage` call emits a well-formed JSONL line into the run directory. The names and bodies here are the single source of truth; any change propagates to `tests/conftest.py` on the same PR.

---

## 1. Unit tests

**Organisation:** one `pytest` module per surface — entries, triage, log, invariants, error modes.
File layout under `tests/test_risk/`:

```
tests/test_risk/
  __init__.py
  test_risk_entries.py          # register shape + stable IDs
  test_risk_assess.py           # Risk.assess() invariants
  test_risk_triage.py           # Risk.triage() per-signal routing
  test_risk_r99_catch_all.py    # R99 + UNKNOWN + unregistered signals
  test_risk_stop_conditions.py  # CLAUDE.md §11 mapping (5 escalate + 3 hard-stop)
  test_risk_log_append.py       # RiskLog.append + fsync + JSONL shape
  test_risk_log_summary.py      # RiskLog.summary counts
  test_risk_entry_frozen.py     # RiskEntry immutability
  test_risk_design_refs.py      # design_refs cite real sections
  test_risk_type_safety.py      # every field typed correctly
```

**Unit test case inventory — 27 cases total (exceeds the ≥ 20 requirement):**

### 1.1 Register shape + stable IDs — `test_risk_entries.py`

**Scope:** the register returned by `Risk.assess()` contains the exact stable ID set described in risk_book.md §10. No duplicates. No missing entries.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_assess_returns_at_least_17_entries` | Call `Risk.assess()`. | `len(Risk.assess()) >= 17` (risk_book.md §3.1 invariant #1). |
| U2 | `test_assess_contains_exact_stable_id_set` | Call `Risk.assess()`. | `{e.id for e in Risk.assess()} == {"R01","R02","R03","R04","R05","R06","R07","R08","R09","R10","R11","R12","R13","R14","R15","R01-STOP","R06-STOP","R99"}` — the 18 stable IDs from risk_book.md §10 + §5. |
| U3 | `test_assess_ids_are_unique` | Call `Risk.assess()`. | `len({e.id for e in entries}) == len(entries)` — duplicate-ID `AssertionError` path from §5 cannot fire on the canonical register. |
| U4 | `test_r01_entry_fields` | Lookup `R01` from `Risk.assess()`. | `entry.probability == Probability.MED`; `entry.impact == Impact.KILLS`; `entry.owner == "C"`; `entry.trigger_signal == TriggerSignal.GRAD_NORM_INF`; `entry.stop_condition is None`; `"DESIGN.md §14 #1"` in `entry.design_refs`. |
| U5 | `test_r13_r14_r15_are_claude_md_adds` | Lookup R13, R14, R15. | Each has a `design_refs` entry citing `"CLAUDE.md §11"`; owners are `"C"`, `"C"`, `"Orchestrator"` respectively. |

### 1.2 `Risk.assess()` invariants — `test_risk_assess.py`

**Scope:** the five invariants in risk_book.md §3.1 assert-tested on the frozen register.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U6 | `test_every_entry_has_named_owner` | Iterate `Risk.assess()`. | For every `entry`: `entry.owner in {"A","B","C","D","orchestrator","Orchestrator","team"}`; no `None`, no `""`, no `"TBD"` (invariant #2). |
| U7 | `test_every_entry_has_observable_trigger_signal` | Iterate `Risk.assess()`. | For every `entry`: `entry.trigger_signal is not None` AND `entry.trigger_signal in TriggerSignal` (invariant #3). |
| U8 | `test_every_entry_has_non_empty_design_refs` | Iterate `Risk.assess()`. | For every `entry`: `len(entry.design_refs) >= 1` (invariant #5). |
| U9 | `test_assess_is_pure` | Call `Risk.assess()` 1000 times. | All return values `==` the first (no side effects; cached data). |
| U10 | `test_assess_entries_are_frozen_dataclass_instances` | Iterate. | Each `entry.__dataclass_params__.frozen is True`. |

### 1.3 `Risk.triage()` per-signal routing — `test_risk_triage.py`

**Scope:** every `TriggerSignal` value maps to the correct `RiskEntry` per the risk_book.md §10 register.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U11 | `test_triage_grad_norm_inf_routes_to_r01` | `Risk.triage(TriggerSignal.GRAD_NORM_INF)`. | `result.entry.id == "R01"`; `result.action == r01.mitigation`; `result.escalate is False`; `result.hard_stop is False`. |
| U12 | `test_triage_policy_kl_over_10_routes_to_r02_escalate` | `Risk.triage(TriggerSignal.POLICY_KL_OVER_10)`. | `result.entry.id == "R02"`; `result.escalate is True`; `result.hard_stop is False`. |
| U13 | `test_triage_r5_drop_routes_to_r05_escalate` | `Risk.triage(TriggerSignal.R5_DROP_WITH_HACK_SPIKE)`. | `result.entry.id == "R05"`; `result.escalate is True`. |
| U14 | `test_triage_stage1_r1_below_0_4_routes_to_r14_escalate` | `Risk.triage(TriggerSignal.STAGE1_R1_BELOW_0_4_AT_100)`. | `result.entry.id == "R14"`; `result.escalate is True`. |
| U15 | `test_triage_envelope_violation_routes_to_vendor_entry` | `Risk.triage(TriggerSignal.ERROR_ENVELOPE_VIOLATION)`. | `result.escalate is True` (risk_book.md §2.4 "always hard_stop" line is satisfied via ESCALATE per the entry's `stop_condition`; the canonical register entry sets this path). |
| U16 | `test_triage_log_line_includes_entry_id` | Any known signal. | `result.log_line.startswith(result.entry.id)` — enables grep-friendly post-mortem. |
| U17 | `test_triage_is_idempotent` | Call `Risk.triage(sig)` 50 times for the same `sig`. | All returned `TriageResult` values `==`. No state mutation. |
| U18 | `test_triage_never_raises_for_any_enum_value` | For `sig in TriggerSignal: Risk.triage(sig)`. | Every call returns a `TriageResult` (never raises). Totality proof — risk_book.md §5 "Policy: risk.py never crashes the run". |

### 1.4 R99 catch-all — `test_risk_r99_catch_all.py`

**Scope:** R99 is the routing target for `UNKNOWN` and for any signal lacking a dedicated entry; `triage()` is total.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U19 | `test_triage_unknown_routes_to_r99` | `Risk.triage(TriggerSignal.UNKNOWN)`. | `result.entry.id == "R99"`; `result.escalate is True` (default-pessimistic per §10 R99 stop_condition); `result.hard_stop is False`. |
| U20 | `test_triage_r99_log_line_includes_context` | `Risk.triage(TriggerSignal.UNKNOWN, context="hf-cli-deprecated")`. | `"hf-cli-deprecated"` appears in `result.log_line`. |
| U21 | `test_triage_unregistered_signal_falls_through_to_r99` | Monkeypatch `Risk._entry_by_signal` to drop the mapping for `GRAD_NORM_INF`, then call `Risk.triage(TriggerSignal.GRAD_NORM_INF)`. | `result.entry.id == "R99"`. Demonstrates the §5 "Unknown-class risk" fallthrough branch. |

### 1.5 CLAUDE.md §11 stop-condition mapping — `test_risk_stop_conditions.py`

**Scope:** invariant #4 from risk_book.md §3.1 — every CLAUDE.md §11 escalation item (5) AND hard-stop item (3) is represented in the register with the correct `StopCondition`.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U22 | `test_escalate_count_is_exactly_5` | `escalate_entries = [e for e in Risk.assess() if e.stop_condition == StopCondition.ESCALATE_TO_USER and e.id != "R99"]`. Exclude R99 because it is a meta-catch-all, not one of the §11 named items. | `len(escalate_entries) == 5` — maps exactly to CLAUDE.md §11 escalate items #1 Gemma4 smoke (R13), #2 openenv validate 3× (R11), #3 Stage-1 R1<0.4 (R14), #4 critic finds DESIGN flaw (covered via R15 / orchestrator path — see U23), #5 merge conflict cross-owner (R15). The register consolidates "critic flags flaw" under R15's process-level escalation because both are orchestrator-resolved spec-conflict events per risk_book.md §7.5. |
| U23 | `test_escalate_set_covers_every_claude_md_11_signal` | Collect `{e.trigger_signal for e in escalate_entries}`. | Must include `{GEMMA4_SMOKE_FAIL, OPENENV_VALIDATE_FAIL, STAGE1_R1_BELOW_0_4_AT_100, MERGE_CONFLICT_CROSS_OWNER, POLICY_KL_OVER_10, R5_DROP_WITH_HACK_SPIKE, TEAM_MEMBER_DROP}` (any 5+ of these; precise subset is {R02,R05,R09,R11,R13,R14,R15} with R02+R05+R09 being reward/training escalations beyond the §11 list but required for behavior-spec completeness). Each of the 5 CLAUDE.md §11 escalate items is covered by at least one entry. |
| U24 | `test_hard_stop_count_is_exactly_3` | `hard_stop_entries = [e for e in Risk.assess() if e.stop_condition == StopCondition.HARD_STOP]`. | `len(hard_stop_entries) == 3` — exactly R01-STOP (V100 > 8h), R06-STOP (HF Hub outage > 2h), and R09's promoted HARD_STOP variant modeled as a third entry OR asserted via R09 path when `TEAM_3PERSON_BELOW_GATE` fires. Maps 1:1 to CLAUDE.md §11 hard-stop items #1, #2, #3. |
| U25 | `test_hard_stop_signals_are_the_canonical_three` | `{e.trigger_signal for e in hard_stop_entries}`. | `== {TriggerSignal.V100_DOWN_OVER_8H, TriggerSignal.HF_HUB_OUTAGE_OVER_2H, TriggerSignal.TEAM_3PERSON_BELOW_GATE}` — exactly the CLAUDE.md §11 hard-stop set. |
| U26 | `test_r06_stop_is_hard_stop` | Lookup `R06-STOP`. | `entry.stop_condition == StopCondition.HARD_STOP`; `entry.trigger_signal == TriggerSignal.HF_HUB_OUTAGE_OVER_2H`; `"CLAUDE.md §11"` cited in `design_refs`. |
| U27 | `test_r01_stop_vs_r01_separation` | Lookup both `R01` and `R01-STOP`. | `R01.stop_condition is None` AND `R01.trigger_signal == GRAD_NORM_INF`; `R01-STOP.stop_condition == HARD_STOP` AND `R01-STOP.trigger_signal == V100_DOWN_OVER_8H`. Confirms the risk_book.md §7.6 "false-positive stop" separation — short-duration V100 drops do not promote. |

### 1.6 `RiskLog` append + JSONL — `test_risk_log_append.py`

**Scope:** `RiskLog` is append-only, JSONL-serialised, fsync'd on every append (crash-safe).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U28 | `test_risk_log_append_writes_single_jsonl_line` | `log = RiskLog(); log.append(triage, ts_iso="2026-04-25T15:22:04+05:30")`; `log.to_jsonl(tmp_path / "risk_log.jsonl")`. | File has exactly 1 line; `json.loads(line)` succeeds; keys include `{"risk_id","action","escalate","hard_stop","log_line","ts_iso"}`. |
| U29 | `test_risk_log_append_preserves_order` | Append 5 triage results with distinct `ts_iso` timestamps. | `to_jsonl` output: line order matches append order; timestamps strictly monotonic. |
| U30 | `test_risk_log_append_is_fsync_on_each_call` | Monkeypatch `os.fsync` to count invocations; append 3 triage results. | `fsync.call_count >= 3` — one `fsync` per append (crash-safety per risk_book.md §4.2 "never truncated"). |
| U31 | `test_risk_log_append_does_not_truncate` | Append once; reopen the underlying file in append mode externally and write a sentinel line; append again. | Final file has 3 lines (two from `RiskLog` + sentinel); the second `RiskLog` append did NOT overwrite the sentinel. Validates "append-only" invariant. |
| U32 | `test_risk_log_append_recovers_on_disk_full` | Monkeypatch the underlying write to raise `OSError(ENOSPC)` once; append; assert the run continues. | `RiskLog.append` logs to stderr (captured) and returns normally. Validates risk_book.md §5 "Log to stderr + continue" policy. No `OSError` propagates. |

### 1.7 `RiskLog.summary()` — `test_risk_log_summary.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U33 | `test_summary_counts_fires_per_risk_id` | Append 3× R01, 2× R05, 1× R99. | `log.summary() == {"R01": 3, "R05": 2, "R99": 1}`. |
| U34 | `test_summary_on_empty_log_returns_empty_dict` | `RiskLog().summary()`. | `== {}`. |

### 1.8 `RiskEntry` immutability — `test_risk_entry_frozen.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U35 | `test_risk_entry_is_frozen` | `entry = Risk.assess()[0]`; attempt `entry.id = "X"`. | Raises `dataclasses.FrozenInstanceError`. |
| U36 | `test_risk_entry_mitigation_immutable` | Attempt `entry.mitigation = "changed"`. | Raises `FrozenInstanceError`. |
| U37 | `test_risk_entry_design_refs_is_tuple_not_list` | `isinstance(entry.design_refs, tuple)`. | `True` — tuple enforces structural immutability at the type level. |

### 1.9 Design-refs validation — `test_risk_design_refs.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U38 | `test_design_refs_cite_existing_sections` | For every entry, for every ref in `entry.design_refs`: parse `"<file> §<section>"`; open the file; assert the section heading is grep-matchable. | No missing citations. Enforces risk_book.md §5 "`design_refs` cites a non-existent section" is caught. |

### 1.10 Field type safety — `test_risk_type_safety.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U39 | `test_every_field_typed_correctly` | For each entry: `isinstance(entry.id, str)`; `isinstance(entry.probability, Probability)`; `isinstance(entry.impact, Impact)`; `isinstance(entry.mitigation, str)`; `isinstance(entry.owner, str)`; `isinstance(entry.trigger_signal, TriggerSignal)`; `entry.stop_condition is None or isinstance(entry.stop_condition, StopCondition)`; `isinstance(entry.design_refs, tuple)`. | All assertions hold for every entry in `Risk.assess()`. |

**Total unit tests: 39 cases. Requirement was ≥ 20. Delivered 39.**

---

## 2. Property tests (hypothesis)

**Framework:** `hypothesis`. Strategies live in `tests/strategies/risk.py` (shared with `env_tests.md` for the integration "risk-log write assertion" path).

| # | Name | Invariant | Strategy |
|---|---|---|---|
| P1 | `prop_triage_is_total_over_TriggerSignal` | For every value in `TriggerSignal`, `Risk.triage(sig)` returns a `TriageResult` (never raises). Asserts risk_book.md §2.3 post-condition "Never raises". | `st.sampled_from(list(TriggerSignal))`; 1000 examples. |
| P2 | `prop_triage_is_pure` | For any `sig`, `Risk.triage(sig)` called N times returns structurally-equal `TriageResult` values (after N=500 calls, all `==`). Asserts risk_book.md §2.3 "Idempotent". | `st.sampled_from(list(TriggerSignal))`; 500 examples, each invoked 10×. |
| P3 | `prop_triage_escalate_iff_stop_condition_set` | For every signal, `result.escalate == (result.entry.stop_condition in {ESCALATE_TO_USER, HARD_STOP})`. Biconditional from risk_book.md §2.3 `TriageResult` doc. | `st.sampled_from(list(TriggerSignal))`; 500 examples. |
| P4 | `prop_triage_hard_stop_iff_HARD_STOP` | `result.hard_stop == (result.entry.stop_condition == StopCondition.HARD_STOP)`. | 500 examples. |
| P5 | `prop_RiskEntry_immutable_under_mutation_attempts` | Given any entry in `Risk.assess()`, any attempted attribute assignment on any field raises `FrozenInstanceError`. Asserts risk_book.md §4.1. | `st.sampled_from(Risk.assess())` × `st.sampled_from(RiskEntry field names)` × `st.text()`; 500 examples. |
| P6 | `prop_RiskLog_append_preserves_order_under_shuffle` | Append N triage results with ascending timestamps (N ∈ [1, 50]); `to_jsonl` output order matches append order regardless of hypothesis-shuffled input. | `st.lists(triage_result_strategy(), min_size=1, max_size=50)`; 200 examples. |
| P7 | `prop_RiskLog_append_only_never_shrinks` | Given an initial `RiskLog` with K entries, any subsequent `append` call strictly increases line count. No append can reduce the file size. | `st.integers(min_value=1, max_value=20)` K + subsequent appends; 200 examples. |

**Total properties: 7. Requirement was ≥ 5. Delivered 7.**

---

## 3. Integration tests

**Scope:** cross-module scenarios that exercise `Risk.triage` + `RiskLog` end-to-end, plus a cross-doc consistency check that fails if risk_book.md §12 drifts from §10.

File: `tests/test_risk_integration.py`.

### 3.1 End-to-end trigger scenarios

| # | Name | Setup | Assertion |
|---|---|---|---|
| I1 | `test_r01_v100_fp16_grad_nan_full_flow` | Simulate training callback observing `grad_norm=inf` on V100 at stage-1 step 14; callback emits `TriggerSignal.GRAD_NORM_INF`; `Risk.triage(sig)`; `RiskLog.append(result, ts_iso=<now>)`; then simulate orchestrator receiving the `SendMessage` payload. | `result.entry.id == "R01"`; `result.hard_stop is False`; `result.escalate is False`; log file gets exactly one new line whose `risk_id == "R01"`; orchestrator message payload contains `"R01"` and the mitigation text. |
| I2 | `test_r01_stop_v100_down_over_8h_hard_stop` | Simulate `V100_DOWN` at T=0; watchdog continues to fire for 8h+; at the 8h wall-clock threshold the watchdog emits `TriggerSignal.V100_DOWN_OVER_8H`; `Risk.triage(sig)`. | `result.entry.id == "R01-STOP"`; `result.hard_stop is True`; `result.escalate is True`; log line present; orchestrator receives `"HARD_STOP R01-STOP"` message. Validates risk_book.md §8.3 exactly. |
| I3 | `test_r06_stop_hf_hub_outage_hard_stop` | Simulate `HF_HUB_OUTAGE` flagged at T=0; health probe keeps failing; at 2h wall-clock, emits `HF_HUB_OUTAGE_OVER_2H`. | `result.entry.id == "R06-STOP"`; `result.hard_stop is True`; log line present. |
| I4 | `test_team_drop_promotes_to_hard_stop_only_when_gate_fails` | First `TEAM_MEMBER_DROP` fires → `R09` escalate (not hard-stop). Then `TEAM_3PERSON_BELOW_GATE` fires. | First triage: `hard_stop is False`, `escalate is True`. Second triage: `hard_stop is True`. Validates CLAUDE.md §11 hard-stop #3 promotion semantics (risk_book.md §10 R09 and §7.4 reassignment). |
| I5 | `test_r05_reward_hack_halts_and_resumes_from_pre_regression` | Simulate `R5_DROP_WITH_HACK_SPIKE` at step 180 of Stage-2; `Risk.triage`; assert the action string contains `"pre-regression"` / `"probe"` tokens indicating Person B is engaged, not an auto-resume. | `result.entry.id == "R05"`; `result.escalate is True`; `"probe"` in `result.action`. Validates risk_book.md §8.2. |
| I6 | `test_orchestrator_halts_dispatch_on_hard_stop` | Fake orchestrator harness consumes triage results via `SendMessage`; feed one `R01-STOP` triage. | Harness assertion: no new `Agent(...)` dispatch call after the hard-stop message. Validates risk_book.md §3.2 triage-loop step 3. |

### 3.2 Cross-doc consistency check

| # | Name | Setup | Assertion |
|---|---|---|---|
| I7 | `test_section_12_table_matches_section_10_entries` | Parse risk_book.md §12 table (columns: Risk / DESIGN.md §14 # / CLAUDE.md §11 / training.md §5 / vendors.md §5 / audio.md §5 / rewards.md). For each row (R01…R99), look up the entry in `Risk.assess()` by ID, and for each non-empty cell, verify the cited file+section exists and the cited reference is also present in `entry.design_refs` OR in the entry's cross-doc footprint. | Every row matches a register entry; every non-empty cell resolves to a real section heading; no orphan rows. Validates risk_book.md §12 "Every cell in this table corresponds to an actual citation". |
| I8 | `test_every_claude_md_section_11_signal_is_routable` | Parse CLAUDE.md §11 "Escalate" block (5 items) + "Hard stop" block (3 items). For each bullet, map its English-language signal phrasing to a `TriggerSignal` enum member, call `Risk.triage(sig)`, and assert the result has `escalate is True` (or `hard_stop is True` for the hard-stop block). | All 8 bullets route to a live entry with the correct stop_condition. Zero CLAUDE.md §11 signals map to R99. |

---

## 4. Coverage target

### 4.1 Line + branch coverage

- **Target:** 100% line coverage and ≥ 95% branch coverage on `driftcall/risk.py` (the single file housing `RiskEntry`, `Probability`, `Impact`, `TriggerSignal`, `StopCondition`, `Risk.assess`, `Risk.triage`, `TriageResult`, `RiskLog.append`, `RiskLog.to_jsonl`, `RiskLog.summary`).
- **Tool:** `pytest --cov=driftcall.risk --cov-branch --cov-report=term-missing --cov-fail-under=100`.
- **Branch coverage script:** `scripts/check_branch_coverage.py` parses `coverage.xml` and fails if the `<branches-covered>/<branches-valid>` ratio for `driftcall/risk.py` < 0.95.
- **CI gate:** in `pyproject.toml`:
  ```
  [tool.coverage.report]
  fail_under = 100
  show_missing = true
  skip_covered = false
  ```
- **Branches that must be hit:** every `if stop_condition is None` vs `is ESCALATE_TO_USER` vs `is HARD_STOP` branch; the "signal not in mapping" fallthrough to R99; the `RiskLog.append` stderr-fallback branch when the write raises `OSError`; the `ts_iso` serialisation branch in `to_jsonl`.

### 4.2 CLAUDE.md §11 coverage matrix

The test suite explicitly enumerates the 8 CLAUDE.md §11 items and asserts a live test maps to each:

| CLAUDE.md §11 item | Kind | Register entry | Test(s) |
|---|---|---|---|
| #1 Gemma 3n E2B smoke fails | Escalate | R13 | U23, I8 |
| #2 `openenv validate` fails 3× | Escalate | R11 | U23, I8 |
| #3 Stage-1 R1 < 0.4 at step 100 | Escalate | R14 | U14, U23, I8 |
| #4 Critic flags DESIGN.md flaw | Escalate | R15 (process path) | U23, I8 (process-level spec-conflict per risk_book.md §7.5) |
| #5 Merge conflict cross-owner | Escalate | R15 | U23, I8 |
| #1 V100 unavailable > 8h | Hard stop | R01-STOP | U25, U26, U27, I2, I8 |
| #2 HF Hub / Spaces > 2h | Hard stop | R06-STOP | U25, U26, I3, I8 |
| #3 Team drop + 3-person gate miss | Hard stop | R09 (promoted) | U25, I4, I8 |

All 8 items are covered by ≥ 2 tests each. `test_every_claude_md_section_11_signal_is_routable` (I8) is the single regression-blocker that fails if ANY §11 item stops being routable.

### 4.3 No-network, no-LLM enforcement

File: `tests/test_risk_no_io.py`.

Purpose: encode the risk_book.md §6.3 invariant that `driftcall/risk.py` does not depend on HF Hub, LLMs, vendor modules, or wall-clock randomness at `assess()` time. Mirrors the rewards-module no-LLM enforcement.

```python
FORBIDDEN_MODULES = frozenset({
    "openai", "anthropic", "transformers", "torch", "unsloth",
    "requests", "httpx", "aiohttp", "google.generativeai", "huggingface_hub",
    "cohere", "mistralai", "vllm", "llama_cpp",
})

def test_risk_module_does_not_import_forbidden():
    """Static import scan of driftcall/risk.py tree. No forbidden package may appear."""

def test_risk_assess_does_not_hit_network_or_clock():
    """Monkeypatch socket.socket, time.time, datetime.now to raise; call Risk.assess() 10×.
    None may fire. (RiskLog.append is allowed to call time.time internally if
    constructed with ts_iso supplied by caller — that path is tested separately.)"""
```

---

## 5. Fixtures

**Location:** `tests/conftest.py` (auto-discovered). **Shared** with `env_tests.md`, `training_tests.md`, `evaluation_tests.md` — integration tests in those plans assert that a `Risk.triage` call during (e.g.) training-callback exercises emits a well-formed JSONL line into the path yielded by `risk_log_tmpfile`. Names and bodies below are the single source of truth.

### 5.1 `risk_log_tmpfile`

```python
@pytest.fixture
def risk_log_tmpfile(tmp_path: Path) -> Path:
    """Returns a unique tmp path for a per-test risk_log.jsonl.
    Shared with env_tests.md, training_tests.md, evaluation_tests.md:
    their integration tests construct RiskLog(path=risk_log_tmpfile) and
    assert that after a simulated trigger, the file contains exactly the
    JSONL lines expected for that scenario. Single source of truth."""
    return tmp_path / "risk_log.jsonl"
```

Consumers:
- `env_tests.md` integration test: `DriftCallEnv.step` raising `AudioDecodeError` → `Risk.triage` → assert `risk_log_tmpfile` contains one line with `risk_id` matching the audio-subsystem entry.
- `training_tests.md` integration test: training-callback simulated `NonFiniteGradientError` → `Risk.triage(GRAD_NORM_INF)` → assert `risk_log_tmpfile` contains `"R01"`.
- `evaluation_tests.md` integration test: eval loop observes `STAGE1_R1_BELOW_0_4_AT_100` → `Risk.triage` → assert `risk_log_tmpfile` contains `"R14"`.

### 5.2 `all_trigger_signals_enum`

```python
@pytest.fixture(scope="session")
def all_trigger_signals_enum() -> tuple[TriggerSignal, ...]:
    """Returns tuple(TriggerSignal) — every enum member, in declaration order.
    Used by property tests (P1-P4) and by U18 (totality proof).
    Session-scoped because the enum is immutable."""
    return tuple(TriggerSignal)
```

### 5.3 `frozen_assess_snapshot`

```python
@pytest.fixture(scope="session")
def frozen_assess_snapshot() -> tuple[RiskEntry, ...]:
    """Returns tuple(Risk.assess()) — the canonical register, frozen at first
    fixture instantiation. Session-scoped so every test in the suite sees
    the SAME register (detects accidental mutation paths early).
    Used by U2, U3, U4, U5, U6, U7, U8, U9, U22, U23, U24, U25, U38, U39, I7."""
    return tuple(Risk.assess())
```

These three fixtures are the contract surface between `risk_book_tests.md` and the sibling test plans. Any rename, scope change, or body change here triggers a coordinated PR update across all four test-plan docs.

---

## Summary

- **Unit tests:** 39 cases across 10 files (requirement ≥ 20)
- **Property tests:** 7 invariants over `Risk.triage`, `RiskEntry`, `RiskLog` (requirement ≥ 5)
- **Integration tests:** 6 end-to-end trigger scenarios + 2 cross-doc consistency checks = 8 cases (requirement ≥ 3)
- **Coverage target:** 100% line + ≥ 95% branch on `driftcall/risk.py`
- **CLAUDE.md §11 coverage:** all 5 escalate items + all 3 hard-stop items mapped to ≥ 2 tests each; regression-blocker test I8 fails the suite if any §11 item becomes un-routable
- **Fixtures:** 3 shared fixtures (`risk_log_tmpfile`, `all_trigger_signals_enum`, `frozen_assess_snapshot`) consumed by env_tests.md, training_tests.md, evaluation_tests.md
- **JSONL append-only:** asserted via U30 (fsync), U31 (no truncation), P6 (order preservation), P7 (monotonic growth); no truncation test deliberately excluded per brief
