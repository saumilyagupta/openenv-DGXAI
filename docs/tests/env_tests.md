# env_tests.md — Test Plan for `driftcall/env.py`

**Module under test:** `driftcall/env.py` (class `DriftCallEnv`)
**Design doc:** `DRIFTCALL/docs/modules/env.md` (final sealed, 9-section spec)
**Owner:** Person B (Rewards & Tests); reviewed by Person A (Environment)
**Implements test coverage for:** DESIGN.md §4 (OpenEnv Interface), §4.2–4.5 (reset/step/budget), §6.2 (drift trigger), §7 (reward invariants), §9.4 (audio boundary), §11.1 (one env per session)
**Framework:** `pytest` + `hypothesis` (+ `pytest-cov`)
**Coverage tool:** `pytest --cov=driftcall.env --cov-branch --cov-report=term-missing`
**Status:** Test plan — pre-critic-gate
**Last updated:** 2026-04-24
**Training path constraint:** All tests are **CUDA-free** (text-only). Audio-boundary tests use in-process stub engines — no Kokoro / Whisper model loads, no network, no disk writes.

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `driftcall/env.py`. Every behavior clause in `env.md §2–§3`, every error mode `E1–E12` in `env.md §5`, every edge case in `env.md §7`, and every worked example in `env.md §8` has at least one dedicated test. Fixtures are **shared** with `docs/tests/deploy_env_space_tests.md` and reuse factories already defined in `models_tests.md`, `vendors_tests.md`, `drift_injector_tests.md`, `task_generator_tests.md`, and `rewards_tests.md` — single source of truth in `tests/conftest.py`.

**Test count target:** ≥ 25 unit + ≥ 5 property + 4 integration = **34 cases minimum**; inventory below sums to **45** (35 unit + 6 property + 4 integration).

---

## 0. Scope & Contract

Covered (public surface of `DriftCallEnv` + `EnvConfig.from_mapping`):

- `DriftCallEnv.__init__(config)` — config validation, unknown-key rejection, mutually-exclusive fields
- `reset(seed)` — deterministic trajectory, curriculum_stage derivation, language_weights propagation, `audio_boundary_enabled` toggle invokes `tts_engine.synthesize`
- `step(action)` — full pipeline ordering per env.md §2.3: (1a pure `_validate_action` → 1b caller handles repeated failures → 2 turn increment → 3 drift fold → 4 side-channel emit → 5 dispatch → 6 `dataclasses.replace` record → 7 terminal check → 8 `compute_rewards` once → 9 observation)
- `state()` — frozen reference return (no deepcopy), `E2` when unready
- `close()` — idempotent, `E3` afterwards, does NOT free shared audio singletons (env.md §9 open question 7)
- `episode()`, `rewards()`, `done()` — terminal-only gating, memoized return
- All 12 typed exceptions in `driftcall.env.errors` rooted at `DriftCallEnvError`

Not covered here (covered elsewhere, referenced only):
- Vendor dispatch internals → `vendors_tests.md`
- Drift pattern catalogue → `drift_injector_tests.md`
- Reward arithmetic → `rewards_tests.md`
- Sim-caller responder body → resolved via env.md §9 Q1 at critic gate; this plan only asserts the responder is deterministic `(seed, turn)`-keyed.

---

## 1. Unit tests (≥ 25 cases — inventory: 35)

All unit tests live in `tests/test_env/`. Layout:

```
tests/test_env/
  __init__.py
  test_init_config_validation.py
  test_reset.py
  test_step_ordering.py
  test_step_validation_purity.py
  test_state_accessor.py
  test_close_idempotent.py
  test_terminal_accessors.py
  test_audio_boundary_toggle.py
  test_error_taxonomy.py
```

### 1.1 `__init__` + `EnvConfig.from_mapping` — config validation (9 cases)

**Scope:** E1 `InvalidConfigError` on every malformed-config branch. `__init__` performs no I/O.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_init_default_config_ok` | `DriftCallEnv()` (no arg) | Succeeds. `env._config.curriculum_stage == 1`. `env._config.language_weights == {"en":0.4,"hinglish":0.4,"hi":0.1,"ta":0.05,"kn":0.05}`. `env._config.audio_boundary_enabled is False`. `env._state is None`. |
| U2 | `test_init_rejects_unknown_key` | `DriftCallEnv({"curriculum_stage":1, "frobnicate":True})` | Raises `InvalidConfigError`; message contains `"frobnicate"` and the full allowed-key list. |
| U3 | `test_init_rejects_invalid_stage` | Parametrized: `0, 4, -1, "1", 1.0, None` | Raises `InvalidConfigError` with `"curriculum_stage"`. |
| U4 | `test_init_rejects_weights_wrong_sum` | `language_weights={"en":0.5,"hinglish":0.4}` (sum=0.9) | Raises `InvalidConfigError`; message cites `"sum"`. |
| U5 | `test_init_rejects_weights_negative` | `language_weights={"en":0.6,"hinglish":0.5,"hi":-0.1}` | Raises `InvalidConfigError`; cites `"negative"`. |
| U6 | `test_init_rejects_audio_enabled_missing_tts` | `audio_boundary_enabled=True, tts_engine=None, asr_engine=<stub>` | Raises `InvalidConfigError`; cites `"tts_engine"`. |
| U7 | `test_init_rejects_audio_disabled_with_tts` | `audio_boundary_enabled=False, tts_engine=<stub>` | Raises `InvalidConfigError` ("tts_engine must be None when audio_boundary_enabled is False" — env.md §7.5). |
| U8 | `test_init_is_pure_no_io` | Patch `builtins.open`, `socket.socket`, and `os.urandom` to raise. `DriftCallEnv({"curriculum_stage":2})`. | Succeeds without invoking any patched callable. Asserts env.md §2.1 "no I/O, no model load, no network call". |
| U9 | `test_init_stores_frozen_config_copy` | Pass a mutable `weights` dict; mutate it after construction. | `env._config.language_weights` unchanged. `EnvConfig` instance has `__dataclass_params__.frozen is True`. |

### 1.2 `reset()` — trajectory setup (8 cases)

| # | Name | Setup | Assertion |
|---|---|---|---|
| U10 | `test_reset_stage1_sets_max_turns_8` | `env=DriftCallEnv({"curriculum_stage":1}); obs=env.reset(seed=1)` | `env._state.max_turns == 8`. `obs.budget_remaining == 8`. `obs.turn == 0`. |
| U11 | `test_reset_stage2_sets_max_turns_12` | stage 2 | `max_turns == 12`; `budget_remaining == 12`. |
| U12 | `test_reset_stage3_sets_max_turns_16` | stage 3 | `max_turns == 16`; `budget_remaining == 16`. |
| U13 | `test_reset_populates_curriculum_stage_on_state` | stage 2 | `env._state.stage == 2` (or equivalent attribute; matches env.md §4.3 `stage` field piped into `Episode`). |
| U14 | `test_reset_passes_language_weights_to_task_generator` | Monkeypatch `task_generator.generate` to record args. `reset(seed=7)` with custom weights. | Recorded `language_weights` argument is **byte-identical** to `env._config.language_weights` (not merely equal-by-value). |
| U15 | `test_reset_same_seed_same_goal_and_schedule` | `env.reset(seed=42)` twice (construct two envs) | `obs_a.goal == obs_b.goal`; `env_a._state.drift_schedule == env_b._state.drift_schedule`; `env_a._state.vendor_states == env_b._state.vendor_states`. |
| U16 | `test_reset_none_seed_populates_from_urandom` | `reset(seed=None)` | `env._seed` is an `int`. Two calls produce different `_seed` with high probability (assert inequality across 3 calls — tolerates 1-in-2^64 flake). |
| U17 | `test_reset_audio_boundary_enabled_invokes_tts_synthesize` | Stub `tts_engine` with a recording synthesize. `audio_boundary_enabled=True`. `reset(seed=11)`. | Stub recorded **exactly one** call with args `(goal.seed_utterance, goal.language)`. `obs.last_transcript == obs.goal.seed_utterance` (canonical source unchanged — env.md §3.7 clause 1). |

### 1.3 `step()` — pipeline ordering (7 cases)

Every case instruments the env by monkeypatching private helpers (`_validate_action`, `_fire_drifts`, `_dispatch`, `_record_action`, `_check_terminal`) to append their names to a shared `call_log` list, proving the order.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U18 | `test_step_validates_before_any_mutation` | Valid stage-1 env after `reset`. Issue a valid TOOL_CALL. | `call_log == ["_validate_action", "_fire_drifts", "_emit_side_channel", "_dispatch", "_record_action", "_check_terminal", "_build_observation"]` — this is the env.md §2.3 order. |
| U19 | `test_step_increments_turn_after_validate_before_dispatch` | Valid TOOL_CALL. | `obs.turn == 1` post-step. Turn counter bump occurs between `_validate_action` and `_fire_drifts` (per env.md §2.3 step 2). Instrumented via snapshot of `self._state.turn` inside stubbed `_fire_drifts`. |
| U20 | `test_step_fires_drifts_before_dispatch` | Scripted scheduler fires `airline.price_rename` at turn 1. Agent action: `TOOL_CALL airline.search` at turn 1. | `obs.tool_results[-1].schema_version == "v2"` (tool saw post-drift schema). `obs.drift_log[-1].pattern_id == "airline.price_rename"`. |
| U21 | `test_step_records_action_via_dataclasses_replace` | Valid TOOL_CALL. | `prev_state = env._state; env.step(a); next_state = env._state`. Assert `prev_state is not next_state`, `id(prev_state.actions) != id(next_state.actions)`, `next_state.actions == prev_state.actions + (a,)`. |
| U22 | `test_step_checks_terminal_after_record` | Stage-1 env; issue 8 benign SPEAK actions (budget=8). | 8th step: `env.done() is True`. `env.episode().terminated_by == "TIMEOUT"`. Turn counter = 8. |
| U23 | `test_step_submit_calls_compute_rewards_exactly_once` | Monkeypatch `rewards.compute_rewards` with a recorder. Issue TOOL_CALL then SUBMIT. | Recorder called **once**. `env.rewards()` returns the exact object the recorder produced. A second call to `env.rewards()` returns the **same identity** (memoized — env.md §3.6). |
| U24 | `test_step_abort_forces_r1_zero` | `reset(seed=1)`; `step(ABORT)`. | `env.episode().terminated_by == "ABORT"`. `env.rewards().r1 == 0.0`. R2…R5 still computed (non-None). |

### 1.4 `_validate_action` purity & `InvalidActionError` (4 cases)

These cases pin env.md §3.5 / E4 behavior: `_validate_action` raises **before** any mutation; env remains valid for a subsequent `step()`.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U25 | `test_invalid_action_raises_no_state_mutation` | Valid stage-1 env. Snapshot `prev_state = env._state`. Call `env.step(DriftCallAction(action_type=TOOL_CALL, tool_name="airline.search"))` with `tool_args=None` (required dict). | Raises `InvalidActionError`. `env._state is prev_state`. `env._state.turn == prev_state.turn`. `len(env._state.actions) == len(prev_state.actions)`. `env._state.done is False`. No Rewards cached (`env._rewards is None`). |
| U26 | `test_env_valid_after_invalid_action` | U25's env, then issue a **valid** TOOL_CALL. | Succeeds. `env._state.turn == 1`. Observation returned normally. Proves env is still steppable. |
| U27 | `test_invalid_action_no_drift_fired_no_terminal_marker` | Scripted scheduler places drift at turn 1. Attempt invalid action. | Raises `InvalidActionError`. `env._state.drift_fired == ()`. `env.done() is False`. The drift did NOT fire (drift firing is inside step 3, after validate). |
| U28 | `test_oversize_rationale_raises_invalid_action` | `DriftCallAction(action_type=SUBMIT, confidence=0.5, rationale="x"*201)` | Raises `InvalidActionError` with `"rationale"`. State unchanged (repeat U25's state-preservation asserts). |

### 1.5 `state()` — frozen reference (2 cases)

| # | Name | Setup | Assertion |
|---|---|---|---|
| U29 | `test_state_returns_frozen_reference` | Post-reset env. | `env.state() is env._state`. `env.state().__dataclass_params__.frozen is True`. Attempting `env.state().turn = 99` raises `dataclasses.FrozenInstanceError`. |
| U30 | `test_state_unready_raises_e2` | Fresh `DriftCallEnv()` without reset. | `env.state()` raises `EnvNotReadyError`. `env.done() is False` (not an error — env.md §7.1). |

### 1.6 `close()` — idempotency (2 cases)

| # | Name | Setup | Assertion |
|---|---|---|---|
| U31 | `test_close_idempotent` | `env.close(); env.close(); env.close()` | No exception; `env._closed is True` after first call and stays True. |
| U32 | `test_close_does_not_free_shared_audio_engines` | Build env with `audio_boundary_enabled=True` and stub TTS/ASR engines. `env.close()`. | `env._closed is True`; `env._state is None`; the stub engines expose no `close()` method at all (`assert not hasattr(tts_stub, "close")` and same for `asr_stub`) — env.md §9 Q7: engines are process-global singletons, and audio.md §2.1–2.2 define no `close()` on `TTSEngine`/`ASREngine`. |

### 1.7 Terminal-only accessors + error taxonomy (3 cases)

| # | Name | Setup | Assertion |
|---|---|---|---|
| U33 | `test_episode_before_terminal_raises_e6` | Post-reset, mid-episode. | `env.episode()` raises `EpisodeNotTerminalError`. Same for `env.rewards()`. `env.done() is False`. |
| U34 | `test_double_submit_raises_e5` | Submit, then attempt another step. | Second `step(...)` raises `EpisodeAlreadyTerminalError` (E5 — env.md §7.2). `env.done()` still True. Rewards object identity preserved. |
| U35 | `test_all_12_errors_derive_from_driftcallenverror` | Introspect `driftcall.env.errors`. | The set `{InvalidConfigError, EnvNotReadyError, EnvClosedError, InvalidActionError, EpisodeAlreadyTerminalError, EpisodeNotTerminalError, ConcurrentStepError, UnknownDomainError, UnknownToolError, DriftInjectionError, RewardComputationError, AudioPipelineError}` each subclass `DriftCallEnvError` which subclasses `Exception`. Count is exactly 12. |

---

## 2. Property tests (≥ 5 — inventory: 6)

Written with `hypothesis`. Strategies live in `tests/test_env/strategies.py` (shared with `test_rewards` where applicable).

| # | Name | Property | Strategy |
|---|---|---|---|
| P1 | `test_step_is_pure_per_call` | For a fresh env `e1` and `e2` constructed with the same config and `reset(seed=s)`, given the same action sequence, every `step()` return is **equal** and the post-step states are **equal**. Same `(state, action) → (state', observation)`. | Seeds in `integers(0, 2**31-1)`; action sequences built from a `DriftCallAction` strategy over valid types; stage in `sampled_from([1,2,3])`. ≥ 200 examples. |
| P2 | `test_validation_failure_preserves_pre_step_state` | For any env in a steppable state and any `DriftCallAction` that fails `_validate_action`: state after the raised `InvalidActionError` equals state before (by identity — `env._state is prev`). | Mixed-validity action strategy; hypothesis `assume()` filters to invalid ones. |
| P3 | `test_turn_counter_monotone_non_decreasing` | Across any legal step sequence, `env._state.turn` is monotone non-decreasing; it **strictly increases** on every non-raising `step()` and is **unchanged** on every raised `InvalidActionError`. | Random action sequences up to length 20; assume `stage=3` to permit budget 16. |
| P4 | `test_frozen_state_identity_changes_on_transition` | After every successful `step()`, `prev_state is not next_state` and `id(prev_state.actions) != id(next_state.actions)` whenever `len(next.actions) > len(prev.actions)`. (env.md §3.8 invariant.) | As P1. |
| P5 | `test_rewards_memoized_identity` | After termination, `env.rewards() is env.rewards()` (identity, not just equality) across 10 calls. Same for `env.episode()`. | Parametrized over `terminated_by ∈ {"SUBMIT","ABORT","TIMEOUT"}`. |
| P6 | `test_available_tools_fixed_for_episode` | The set `obs.available_tools` is equal across every observation in an episode, regardless of drifts fired. (env.md §3.4 clause 4.) | Random schedules over stage 2/3; ≥ 50 episodes. |

---

## 3. Integration tests (4 cases)

Live in `tests/test_env/test_e2e.py`. These are **full episode traces** matching env.md §8 examples. All dependencies are real (real `task_generator`, real `drift_injector`, real vendors, real `rewards.compute_rewards`) — only the audio engines are stubbed in I4.

| # | Name | Maps to | Scenario |
|---|---|---|---|
| I1 | `test_episode_stage1_airline_happy_submit` | env.md §8.1 | `DriftCallEnv({"curriculum_stage":1})`; `reset(seed=42)`. Replay the 5-turn script: `airline.search` → 3 more tool calls → `SUBMIT(confidence=0.9)`. Assertions: `env.done() is True`; `env.episode().terminated_by == "SUBMIT"`; `env.episode().turns_used == 5`; `obs.drift_log == ()`; `env.rewards().r1 == 1.0`; `env.rewards().r2 == 0.5` (stage-1 neutral); `env.rewards().reward` in `[0.85, 1.0]`. |
| I2 | `test_episode_stage2_drift_detect_adapt` | env.md §8.2 | `stage=2; seed=7`. Scripted sequence through turn 6 terminating in SUBMIT. Drift `airline.price_rename` fires turn 3. Agent SPEAK at turn 4 mentions `"total_fare_inr"`. Assertions: `obs.drift_log[0].pattern_id == "airline.price_rename"`; `obs.drift_log[0].turn == 3`; `obs.tool_results[-2].response` references `"total_fare_inr"` (not `"price"`); `env.rewards().r1 == 1.0`; `env.rewards().r2 == 1.0`; `env.rewards().reward ≈ 0.90 ± 0.05`. |
| I3 | `test_episode_stage3_compound_drift_timeout` | env.md §8.3 | `stage=3; seed=2026`. Script designed to consume all 16 turns. Two drifts fire (airline turn 3, payment turn 9). Assertions: `env.done() is True`; `env.episode().terminated_by == "TIMEOUT"`; `env.episode().turns_used == 16`; `env.rewards().r1 == 0.0`; `env.rewards().r2 in {0.5, 1.0}`; `env.rewards().reward < 0.3`. |
| I4 | `test_episode_audio_boundary_enabled_stubs` | env.md §8.4 | `audio_boundary_enabled=True`, `tts_engine=StubTTS()`, `asr_engine=StubWhisper()` (contracts in §5 — signatures match `audio.md §2.1–2.2`). Stubs are in-process, CUDA-free, deterministic: `synthesize(text, language_code, voice_pack=None, *, seed=0, sample_rate_hz=16000) → f"WAV[{text}:{language_code}:{seed}:{sample_rate_hz}]".encode("utf-8")`; `transcribe(audio_bytes, language_hint, *, beam_size=1, vad_filter=True, max_duration_s=30.0) → TranscriptResult(text=<scripted>, language_detected="hinglish", confidence=0.82, duration_s=1.250)`. Episode: `reset(seed=11)` → `CLARIFY` → `TOOL_CALL` → `SUBMIT`. Assertions: stub TTS synthesize called on `reset` **and** on every CLARIFY/SPEAK side-channel emission; `obs.last_transcript` after CLARIFY equals the stubbed ASR text; `obs.last_confidence == 0.82`; reward computation is **100% textual** — no TTS bytes reach `compute_rewards` (verified by asserting `episode.actions` and `episode.tool_results` contain no `bytes` objects). |

All integration tests reuse fixtures:
- `goal_airline`, `goal_restaurant` — from `drift_injector_tests.md §5.2` (session-scoped `GoalSpec` instances)
- `airline_v1`, `airline_v2`, `payment_v2` — from `vendors_tests.md §5.1` (per-domain aliases over `vendor_states_v{1,2,3}`; `payment_v2` is the post-`auth_scope_bump` state)
- `drift_patterns_fixture` — from `drift_injector_tests.md §5.1` (authoritative 20-pattern catalogue; individual events + compound schedules used by I2/I3 are defined locally in §5 below as `drift_event_airline_price_rename_turn3`, `drift_event_payment_auth_turn9`, and `schedule_stage3_compound`, because drift_injector_tests.md only ships the catalogue, not pre-composed schedules)
- `episode_happy_airline`, `episode_timeout` — from `rewards_tests.md §5` (§5.1 and §5.4 respectively)
- `valid_tool_call_action`, `valid_submit_action`, `valid_observation_reset` — from `models_tests.md §5.4` (the factory/instance fixtures used to assemble per-step action sequences)

No integration test touches the network. No test loads a real Kokoro/Whisper model.

---

## 4. Coverage target

**100% line coverage** and **≥ 95% branch coverage** on `driftcall/env.py` under `pytest --cov=driftcall.env --cov-branch --cov-report=term-missing`.

### 4.1 Error-mode coverage matrix (every E1–E12 raised at least once)

| Code | Exception | Raised by which test |
|---|---|---|
| E1 | `InvalidConfigError` | U2 (unknown key), U3 (bad stage), U4 (weights sum), U5 (negative weight), U6 (missing TTS), U7 (forbidden TTS). Also raised from U4.3 reset if scripted scheduler produces turn > max_turns — covered by a dedicated test `test_reset_scripted_bad_schedule_raises_e1`. |
| E2 | `EnvNotReadyError` | U30 (`state()`), plus `test_step_before_reset_raises_e2`, `test_episode_before_reset_raises_e2`. |
| E3 | `EnvClosedError` | `test_step_after_close_raises_e3`, `test_reset_after_close_raises_e3`. |
| E4 | `InvalidActionError` | U25, U26, U27, U28, plus per-ActionType parametrized cases: missing `tool_name` on TOOL_CALL, message len 0 and len 2001 on SPEAK, NUL byte in message on CLARIFY, missing `confidence` on SUBMIT, forbidden `tool_name` on ABORT. |
| E5 | `EpisodeAlreadyTerminalError` | U34 (double SUBMIT). |
| E6 | `EpisodeNotTerminalError` | U33. |
| E7 | `ConcurrentStepError` | `test_reentrant_step_raises_e7` — stub a vendor `dispatch` that re-invokes `env.step(other_action)`; assert E7 raised on the inner call; assert outer state unchanged. |
| E8 | `UnknownDomainError` | `test_probe_schema_unknown_domain_raises_e8` — PROBE_SCHEMA with `tool_name="spaceship"`. |
| E9 | `UnknownToolError` | `test_tool_call_unknown_tool_raises_e9` — `tool_name="airline.teleport"`. |
| E10 | `DriftInjectionError` | `test_drift_fold_error_propagates_e10` — scripted scheduler yields event with unknown `pattern_id`; env must not swallow. |
| E11 | `RewardComputationError` | `test_reward_compute_error_propagates_e11` — monkeypatch `rewards.compute_rewards` to raise; env must surface. |
| E12 | `AudioPipelineError` | `test_audio_pipeline_error_on_clarify` — stub ASR that raises on 2nd transcribe; assert E12 surfaces from `step(CLARIFY)`; episode does NOT terminate (env.md §5 E12 note). Second test: `test_audio_pipeline_error_on_reset_is_e1_class` — stub TTS that raises on `reset`; the env is unready afterwards per env.md §5 E12. |

Total dedicated error-mode tests: 12 exceptions × ≥ 1 = 12 minimum; inventory covers 18 error-mode paths.

### 4.2 Line/branch targets

- `DriftCallEnv.__init__` — 100% line; 100% branch (both `config is None` and `config is dict` branches hit in U1, U2).
- `EnvConfig.from_mapping` — 100% line; 100% branch (all 7 raise branches covered by U2–U7 + reset-bad-schedule).
- `reset` — 100% line; step 7b audio branch covered by U17 (True) and U10 (False).
- `step` — 100% line; all 6 ActionType dispatch branches (TOOL_CALL / SPEAK / CLARIFY / PROBE_SCHEMA / SUBMIT / ABORT) each have ≥ 1 unit test + integration coverage; drift-fold-empty vs non-empty both covered (I1 empty, I2 non-empty); terminal vs non-terminal both covered (U22 TIMEOUT, U23 SUBMIT, I1/I2/I3 mix).
- `state`, `close`, `episode`, `rewards`, `done` — 100% line; all raise/early-return branches covered.
- `_validate_action` — 100% line; every row of env.md §3.1 Table is parametrized (per-ActionType forbidden-field matrix).
- `_build_observation` — 100% line; `last_transcript` branches for turn 0 vs mid-episode vs audio-enabled all covered (U17, I4, I1).

Branch coverage < 95% is a **hard CI fail**.

---

## 5. Fixtures

All fixtures defined in `tests/conftest.py` under the `env_*` namespace. Shared with `docs/tests/deploy_env_space_tests.md` (same names, same content).

| Name | Scope | Purpose | Reuses |
|---|---|---|---|
| `env_stage1_airline` | function | `DriftCallEnv({"curriculum_stage":1})` already `reset(seed=42)`, goal forced to airline via scripted `task_generator` monkeypatch when hermetic goal needed. Provides `(env, obs0)` tuple. | `goal_airline` from `drift_injector_tests.md §5.2`; `airline_v1` from `vendors_tests.md §5.1`. |
| `env_stage2_restaurant_drift` | function | Stage-2 env `reset(seed=7)` with restaurant goal, scripted scheduler that fires `restaurant.items_shape_bump` at turn 3. Returns `(env, obs0, drift_event)`. | `goal_restaurant` from `drift_injector_tests.md §5.2`; `drift_event_restaurant_items_shape_bump_turn3` defined below. |
| `env_stage3_compound` | function | Stage-3 env `reset(seed=2026)`, scripted scheduler with compound drift (airline turn 3 + payment turn 9). Used by I3. | `schedule_stage3_compound` defined below; reuses `drift_patterns_fixture` catalogue from `drift_injector_tests.md §5.1`. |
| `env_audio_enabled` | function | Stage-1 env with `audio_boundary_enabled=True`, `tts_engine=StubTTS()`, `asr_engine=StubWhisper()`. Stubs are pure Python, CUDA-free, deterministic. Returns `(env, tts_stub, asr_stub)` for assertions on call counts. | `StubTTS`, `StubWhisper` classes defined in `tests/stubs/audio_stubs.py`. |
| `env_config_invalid_key` | function | `{"curriculum_stage":1, "frobnicate":True}` — a single malformed config dict reused across U2 and any critic-requested smoke test. | — |

Stub engine contracts (pinned here for cross-doc consistency with `audio_tests.md`; signatures match `docs/modules/audio.md §2.1` and `§2.2` exactly):

```python
from driftcall.audio.asr_whisper import TranscriptResult
from driftcall.audio.tts_kokoro import VoicePack

class StubTTS:
    """In-process TTS double. Matches audio.md §2.1 `TTSEngine.synthesize` signature."""
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, VoicePack | None, int, int]] = []

    def synthesize(
        self,
        text: str,
        language_code: str,
        voice_pack: VoicePack | None = None,
        *,
        seed: int = 0,
        sample_rate_hz: int = 16000,
    ) -> bytes:
        self.calls.append((text, language_code, voice_pack, seed, sample_rate_hz))
        return f"WAV[{text}:{language_code}:{seed}:{sample_rate_hz}]".encode("utf-8")

class StubWhisper:
    """In-process ASR double. Matches audio.md §2.2 `ASREngine.transcribe` signature
    and the 4-field `TranscriptResult` contract (text, language_detected, confidence, duration_s)."""
    def __init__(self, scripted: dict[int, str] | None = None) -> None:
        self.calls: list[bytes] = []
        self._scripted = scripted or {}

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: str | None,
        *,
        beam_size: int = 1,
        vad_filter: bool = True,
        max_duration_s: float = 30.0,
    ) -> TranscriptResult:
        self.calls.append(audio_bytes)
        turn = len(self.calls)
        return TranscriptResult(
            text=self._scripted.get(turn, "shaam ko, 7 baje"),
            language_detected="hinglish",
            confidence=0.82,
            duration_s=1.250,
        )
```

Neither stub exposes a `.close()` method: `audio.md §2.1–2.2` defines no such method on `TTSEngine`/`ASREngine`, and the engines are process-global singletons (env.md §9 Q7) — U32 asserts `env.close()` does NOT invoke anything engine-side, so the stubs simply must not carry a `close()` attribute at all (U32's "call count is 0" is upgraded to `not hasattr(stub, "close")` to match the real contract).

### 5.1 Locally-defined drift events and schedules (not shipped by `drift_injector_tests.md`)

`drift_injector_tests.md §5.1` publishes the 20-pattern catalogue (`drift_patterns_fixture`) but does NOT pre-compose per-test `DriftEvent` instances or full `DriftSchedule` objects — those are composed locally here because scheduling is an env-side concern. All three fixtures below are session-scoped and import `drift_patterns_fixture` to look up the authoritative pattern record.

```python
from driftcall.models import DriftEvent
from driftcall.drift_injector import DriftSchedule

@pytest.fixture(scope="session")
def drift_event_airline_price_rename_turn3(drift_patterns_fixture) -> DriftEvent:
    """Used by I2. Pattern id asserted byte-identical to drift_patterns_fixture entry."""
    pattern = next(p for p in drift_patterns_fixture if p.id == "airline.price_rename")
    return DriftEvent(
        turn=3,
        drift_type=pattern.drift_type,       # "schema"
        domain=pattern.domain,               # "airline"
        description=pattern.description,
        from_version=pattern.from_version,   # "v1"
        to_version=pattern.to_version,       # "v2"
    )

@pytest.fixture(scope="session")
def drift_event_restaurant_items_shape_bump_turn3(drift_patterns_fixture) -> DriftEvent:
    """Used by env_stage2_restaurant_drift. `restaurant.items_shape_bump` is the
    canonical restaurant schema drift per drift_injector.md §4.4 (items gain required `modifiers`)."""
    pattern = next(p for p in drift_patterns_fixture if p.id == "restaurant.items_shape_bump")
    return DriftEvent(
        turn=3,
        drift_type=pattern.drift_type,
        domain=pattern.domain,
        description=pattern.description,
        from_version=pattern.from_version,
        to_version=pattern.to_version,
    )

@pytest.fixture(scope="session")
def drift_event_payment_auth_turn9(drift_patterns_fixture) -> DriftEvent:
    """Used by I3. Pattern id `payment.auth_scope_upgrade` (Auth axis, drift_injector.md §4.4)."""
    pattern = next(p for p in drift_patterns_fixture if p.id == "payment.auth_scope_upgrade")
    return DriftEvent(
        turn=9,
        drift_type=pattern.drift_type,       # "auth"
        domain=pattern.domain,               # "payment"
        description=pattern.description,
        from_version=pattern.from_version,
        to_version=pattern.to_version,
    )

@pytest.fixture(scope="session")
def schedule_stage3_compound(
    drift_event_airline_price_rename_turn3,
    drift_event_payment_auth_turn9,
) -> DriftSchedule:
    """Used by I3. Two drifts, one per domain, matching env.md §8.3 worked example."""
    return DriftSchedule(events=(
        drift_event_airline_price_rename_turn3,
        drift_event_payment_auth_turn9,
    ))
```

These three `DriftEvent`s plus one `DriftSchedule` are the only fixtures defined *in* `env_tests.md`; everything else is imported from the sibling test plans cited in §3 above.

**Fixture immutability rule:** if any field of any fixture changes here, the matching fixture in `deploy_env_space_tests.md §5` **must** be updated in the same commit — they share a single `conftest.py` definition. CI guards this via a grep-based pre-commit hook (`scripts/check_fixture_parity.sh`).
