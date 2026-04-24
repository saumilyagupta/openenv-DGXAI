# step_10_env — DriftCallEnv

Implements `docs/modules/env.md` and `DESIGN.md §4`.

## Public surface

| Symbol | Kind | Notes |
|---|---|---|
| `DriftCallEnv` | class | OpenEnv-compliant RL environment. Single-session, single-episode-at-a-time. |
| `EnvConfig` | frozen dataclass | Validated config snapshot. Built via `EnvConfig.from_mapping(...)`. |
| `Episode` | frozen dataclass | Terminal-only snapshot fed to `cells.step_08_rewards.compute_rewards`. |
| `DriftScheduler` | Protocol | `(stage, seed, goal) -> tuple[DriftEvent, ...]`. Default: `drift_injector.build_schedule`. |
| `TTSEngine` / `ASREngine` | Protocols | Audio boundary contracts (env.md §2.1). |
| `DriftCallEnvError` and 12 subclasses | exceptions | E1..E12 typed taxonomy. |

## Wiring

```
reset(seed)
  └── task_generator.generate(seed, stage, language_weights)
  └── per-domain vendor.initial_state(seed, goal)        # airline, cab, restaurant, hotel, payment
  └── scheduler(stage, seed, goal)                       # default = drift_injector.build_schedule
  └── audio_boundary_enabled? tts_engine.synthesize(seed_utterance, language)
  └── DriftCallObservation(turn=0, ...)

step(action, *, force_drift_pattern=None)
  1a. _validate_action(action)            # pure, raises InvalidActionError BEFORE mutation
  1b. force_drift_pattern resolved        # unknown -> InvalidActionError
  2.  turn += 1                            # via dataclasses.replace
  3.  drift fold:                          # forced pattern OR scheduled pending drifts
        - sort by (turn asc, pattern_id asc)
        - apply via drift_injector.apply_drift
  4.  side-channel emit pass               # vendor.emit_side_channel_if_pending per domain
  5.  dispatch:
        TOOL_CALL    -> vendor.dispatch(...) and merge any pending notice into ToolResult
        SPEAK/CLARIFY-> no state change
        PROBE_SCHEMA -> vendor.describe_schema(state, version), wrapped as ToolResult
        SUBMIT       -> terminate("SUBMIT")
        ABORT        -> terminate("ABORT")
  6.  record action (and ToolResult, if any) via dataclasses.replace
  7.  if turn >= max_turns -> terminate("TIMEOUT")
  8.  if terminal -> build Episode + step_08_rewards.compute_rewards (memoized)
  9.  return DriftCallObservation
```

## Termination

`terminated_by ∈ {SUBMIT, ABORT, TIMEOUT, ANTI_HACK}`. Reward layer reads `terminated_by` to force `r1=0` for ABORT/TIMEOUT/ANTI_HACK. `Episode` and `Rewards` are write-once; `episode()`/`rewards()` return memoized identities.

## Determinism contract

Same `(config, seed)` ⇒ byte-identical `goal`, `drift_schedule`, and initial `vendor_states`. The only non-deterministic field is `episode_id` (uuid4), which is purely an audit handle (env.md §9 Q5).

## Error taxonomy (E1–E12)

All extend `DriftCallEnvError(Exception)`:

| # | Class | When |
|---|---|---|
| E1 | `InvalidConfigError` | unknown key, bad weights, missing audio engine, etc. |
| E2 | `EnvNotReadyError` | step/state/episode/rewards before reset |
| E3 | `EnvClosedError` | reset/step after close |
| E4 | `InvalidActionError` | per-`ActionType` field-matrix violation; force_drift_pattern unknown |
| E5 | `EpisodeAlreadyTerminalError` | step after termination |
| E6 | `EpisodeNotTerminalError` | episode/rewards before termination |
| E7 | `ConcurrentStepError` | reentrant step |
| E8 | `UnknownDomainError` | PROBE_SCHEMA on unregistered domain |
| E9 | `UnknownToolError` | TOOL_CALL with tool_name not in available_tools |
| E10 | `DriftInjectionError` | drift fold failure (propagated from drift_injector) |
| E11 | `RewardComputationError` | compute_rewards failure |
| E12 | `AudioPipelineError` | TTS/ASR engine raised at boundary |

Validation in `_validate_action` is strictly pure: raises before any state mutation, so the env remains valid for a subsequent `step()`.

## Audio boundary

`audio_boundary_enabled=True` requires both `tts_engine` and `asr_engine`. On `reset()` the env calls `tts_engine.synthesize(goal.seed_utterance, goal.language)`; the canonical `last_transcript` remains the textual `seed_utterance`. The audio pipeline never feeds bytes back into reward computation.

## Out of scope

- LLM judging — never. The env is the judge.
- Concurrency — single-session by contract; no locks, no asyncio.
- Disk/network I/O at `__init__` — strictly forbidden.
