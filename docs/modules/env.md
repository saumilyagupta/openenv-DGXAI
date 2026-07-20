# `driftcall/env.py` — DriftCallEnv integration class

**Design doc.** No code is written against this spec until ≥2 fresh critics return `NOTHING_FURTHER`.
**Implements DESIGN.md §4 (OpenEnv Interface) in full.**
**Sections per CLAUDE.md §3.1:** Purpose · Interface · Behavior spec · Data structures · Error modes · Dependencies · Edge cases · Examples · Open questions.

---

## 1. Purpose

`DriftCallEnv` is the **one public surface** that the OpenEnv runner, the training loop, the FastAPI shim in `app.py`, and the Gradio demo all call. It is the sole integration point that composes every other module into an RL-compliant episode:

- **`task_generator.generate`** — samples a seeded `GoalSpec` at `reset()` (task_generator.md §2 generate()).
- **`drift_injector.build_schedule` / `apply_drift`** — pre-computes the episode's drift timetable and fires each event at the start of its scheduled turn.
- **`vendors/*`** (airline, cab, restaurant, hotel, payment) — dispatches `TOOL_CALL` actions and evolves per-vendor frozen `VendorState` records (vendors.md §2 dispatch()).
- **`models`** — supplies every frozen dataclass crossing the boundary (`DriftCallAction`, `DriftCallObservation`, `DriftCallState`, `ToolResult`, `DriftEvent`, `GoalSpec`, `Episode`).
- **`rewards.compute_rewards`** — called **exactly once** at termination to produce the final `Rewards` scalar consumed by GRPO.
- **`audio/*`** — *optional* boundary-only pipeline (ASR on user utterance entry, TTS on `SPEAK` replies); disabled during training, enabled behind a config flag on the deployed env Space (DESIGN.md §9.4).

The env is the **judge** (DESIGN.md §0, §7). It holds the authoritative `DriftCallState`, enforces the budget, triggers drifts deterministically, dispatches tools, detects termination, and computes rewards server-side — no LLM judge anywhere. The agent observes through `DriftCallObservation` and acts through `DriftCallAction`; everything else is env-internal.

This module has **zero domain logic of its own**. Every vendor rule, every drift mutation, every reward clause is owned by the downstream module. `env.py` is plumbing: validate → drift → dispatch → record → terminate → observe.

Cross-references: DESIGN.md §3.1 (high-level diagram), §4.1–4.5 (dataclasses, reset, step, termination, budget), §6.2 (drift trigger logic), §7 (reward invariants), §9.4 (audio training/deploy split).

---

## 2. Interface

```python
from __future__ import annotations

from typing import Any, Literal, Protocol

from driftcall.models import (
    DriftCallAction,
    DriftCallObservation,
    DriftCallState,
    Episode,
    GoalSpec,
    ToolResult,
    DriftEvent,
    ActionType,
)
from driftcall.rewards import Rewards


class DriftCallEnv:
    """
    OpenEnv-compliant RL environment for DriftCall.

    One instance == one episode lifecycle controller. Reuse across episodes
    via reset(); do not share a single instance across concurrent episodes
    (not thread-safe; see §3.9 / Error mode E7).
    """

    # -- construction --------------------------------------------------------
    def __init__(self, config: dict[str, Any] | None = None) -> None: ...

    # -- OpenEnv primitives --------------------------------------------------
    def reset(self, seed: int | None = None) -> DriftCallObservation: ...

    def step(
        self,
        action: DriftCallAction,
        *,
        force_drift_pattern: str | None = None,
    ) -> DriftCallObservation: ...

    def state(self) -> DriftCallState: ...

    def close(self) -> None: ...

    # -- terminal-only accessors (see §3.6) ---------------------------------
    def episode(self) -> Episode: ...

    def rewards(self) -> Rewards: ...

    def done(self) -> bool: ...
```

### 2.1 `__init__(self, config)`

**Config contract** (frozen-copy stored on the instance as `self._config: EnvConfig`, a frozen dataclass — §4.1):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `curriculum_stage` | `Literal[1, 2, 3]` | `1` | Drift frequency (DESIGN.md §4.5, §6.2). |
| `language_weights` | `dict[LanguageCode, float]` | `{"en": 0.4, "hinglish": 0.4, "hi": 0.1, "ta": 0.05, "kn": 0.05}` | Non-negative; must sum to `1.0 ± 1e-6`. Passed to `task_generator.generate`. |
| `audio_boundary_enabled` | `bool` | `False` | If True, ASR/TTS runs at the env boundary (see §3.7). **MUST be False during GRPO training** (DESIGN.md §9.4). |
| `max_turns_override` | `int \| None` | `None` | Only for tests. If None, derived from stage via §3.2 Table A. |
| `scheduler` | `DriftScheduler` | `drift_injector.build_schedule` | Injection seam for scripted schedules in tests (drift_injector.md §2). |
| `tts_engine` | `TTSEngine \| None` | `None` | Required iff `audio_boundary_enabled`; otherwise must be None. |
| `asr_engine` | `ASREngine \| None` | `None` | Same as above. |

**Postconditions:** `__init__` performs **no I/O, no model load, no network call**. All validation is pure (raises `InvalidConfigError` on malformed config). The env is **not ready for `step()`** — callers MUST call `reset()` first.

### 2.2 `reset(seed)`

```python
def reset(self, seed: int | None = None) -> DriftCallObservation:
    """
    Begin a new episode.

    Signature takes `seed` ONLY (DESIGN.md §4.2 — locked 2026-04-24). Config
    (curriculum_stage, language_weights, audio_boundary_enabled, engines) is
    bound at `__init__` and is immutable for the env instance's lifetime.
    Callers that need a different stage or language mix MUST construct a new
    `DriftCallEnv`.

    If seed is None, one is drawn from os.urandom(8) and stored. Same seed ⇒
    byte-identical GoalSpec, drift_schedule, and initial vendor_states
    (DESIGN.md §4.2).

    Steps (in this order):
        1. episode_id = uuid4() (purely for audit/log correlation; NOT seeded).
        2. goal = task_generator.generate(seed, stage, language_weights).
        3. vendor_states = {d: vendors[d].initial_state(seed, goal) for d in DOMAINS}.
        4. schema_versions = {d: "v1" for d in DOMAINS}.
        5. drift_schedule = self._config.scheduler(stage, seed, goal).
        6. Build fresh DriftCallState (turn=0, max_turns per §3.2, done=False,
           actions=(), drift_fired=()).
        7. Build DriftCallObservation for turn 0 via _build_observation (§3.4).
        7b. If `self._config.audio_boundary_enabled` is True, invoke `tts_engine.synthesize(goal.seed_utterance, goal.language)` and emit audio on the side channel per §3.7. Canonical `last_transcript` remains `goal.seed_utterance` — audio is a boundary artifact, not the source of truth.
        8. Return observation; store state on self._state.

    Raises: InvalidLanguageWeightError, InvalidStageError from task_generator
            (propagated; see §5 E1). On any failure the env remains unready
            (self._state stays None).
    """
```

### 2.3 `step(action)`

```python
def step(
    self,
    action: DriftCallAction,
    *,
    force_drift_pattern: str | None = None,
) -> DriftCallObservation:
    """
    Advance one turn. Full pipeline (DESIGN.md §4.3):

        0. Preconditions: self._state is not None; self._state.done is False.
        1a. Validate action via `_validate_action` (pure — no state mutation). On
            validation failure, raise `InvalidActionError` immediately. No turn
            counter increment. No action stored. No termination marker yet.
        1b. If the caller catches the `InvalidActionError` but continues to invoke
            step() with another malformed action, repeated failures are tracked by
            an outer anti-hack handler in the caller. The env itself treats each
            malformed step() as a pure exception-raising operation with zero side
            effects.
        2. turn_current = self._state.turn + 1.
        3. Fire drifts for turn_current (see §3.3 firing spec):
           - If `force_drift_pattern is not None`, inject that single pattern AT
             THIS TURN, overriding the pre-computed `drift_schedule`. Any
             schedule-driven drifts that would have fired on turn_current are
             SUPPRESSED for this step — judge intent wins over RNG. The forced
             pattern must be a valid `DRIFT_PATTERN_IDS` literal; unknown ids
             raise `InvalidActionError` before any state mutation.
           - If `force_drift_pattern is None`, fire all drifts from
             `drift_schedule` whose turn == turn_current, folding `apply_drift`
             in (turn asc, pattern_id asc) order (drift_injector.md §3.1). Each
             application returns a new frozen state; the fold produces the
             post-drift state.
        4. For every vendor whose state just mutated (i.e., whose domain appeared
           in at least one pending drift), call vendors[d].emit_side_channel_if_pending
           and stash any (notice, new_vendor_state) for attachment to the first
           subsequent ToolResult on that domain (§3.3).
        5. Dispatch the action:
              TOOL_CALL    → vendors[d].dispatch(...) → (ToolResult, new_vendor_state)
              SPEAK/CLARIFY→ no state change; pure log (§3.4)
              PROBE_SCHEMA → vendors[d].describe_schema(current_schema_version)
                             packaged as a ToolResult with status="ok"
              SUBMIT       → terminate (see §3.5); compute rewards
              ABORT        → terminate with R1=0 (see §3.5)
        6. Append action (and any resulting ToolResult) to self._state via
           dataclasses.replace — state is rebuilt frozen each step; no in-place
           mutation (§3.8).
        7. Budget check: if new turn >= max_turns and not already terminal,
           terminate with terminated_by="TIMEOUT", R1 forced to 0.
        8. If terminal, build Episode (§4.3) and call rewards.compute_rewards
           EXACTLY ONCE. Cache the Rewards on self._rewards. Set state.done=True.
        9. Build and return the new observation.
    """
```

### 2.4 `state()`

```python
def state(self) -> DriftCallState:
    """
    Return the current frozen state. Required by OpenEnv validator.
    Returns the frozen `self._state` reference directly (no deepcopy needed — dataclass is `frozen=True` and every field is a frozen object, tuple, or primitive; external mutation is impossible by construction).
    Raises EnvNotReadyError if called before reset().
    """
```

### 2.5 `close()`

```python
def close(self) -> None:
    """
    Release any resources (audio engines if loaded, nothing else). Idempotent.
    After close(), calls to reset/step raise EnvClosedError.
    """
```

### 2.6 `episode()` / `rewards()` / `done()` (terminal-only)

```python
def episode(self) -> Episode:
    """Return the finalized Episode. Raises EpisodeNotTerminalError if not done."""

def rewards(self) -> Rewards:
    """Return the cached Rewards. Raises EpisodeNotTerminalError if not done."""

def done(self) -> bool:
    """True iff the current episode has terminated. False before reset()."""
```

`episode()` and `rewards()` are memoized — they return the cached objects, never recompute.

---

## 3. Behavior spec

### 3.1 Action validation (`_validate_action`)

For every `DriftCallAction`, enforce the per-`ActionType` required-field matrix. Any violation raises `InvalidActionError` and triggers the anti-hack termination path (§3.5) — validation failures are agent failures, not env bugs.

| `action_type` | Required | Forbidden | Extra constraints |
|---|---|---|---|
| `TOOL_CALL` | `tool_name`, `tool_args` (may be `{}`) | `message`, `confidence` | `tool_name` ∈ `self._available_tools()`; `tool_args` JSON-serializable dict. |
| `SPEAK` | `message` (len ∈ [1, 2000]) | `tool_name`, `tool_args`, `confidence` | Valid UTF-8; no NUL bytes. |
| `CLARIFY` | `message` (len ∈ [1, 2000]) | `tool_name`, `tool_args`, `confidence` | Same as SPEAK. |
| `PROBE_SCHEMA` | `tool_name` (domain, e.g. `"airline"`) | `tool_args`, `message`, `confidence` | Domain must exist in `self._state.vendor_states`. |
| `SUBMIT` | `confidence` ∈ `[0.0, 1.0]` | `tool_name`, `tool_args` | `message` optional (final user-facing reply); `rationale` optional ≤ 200 chars. |
| `ABORT` | — | all of `tool_name`, `tool_args`, `confidence` | `message` optional. |

`rationale`, if present, is always bounded to 200 chars (matches models.md §3.5). Longer ⇒ `InvalidActionError`.

### 3.2 Budget & max_turns

Derived in `reset()` from `curriculum_stage` (Table A, DESIGN.md §4.5):

| Stage | `max_turns` |
|---|---|
| 1 | 8 |
| 2 | 12 |
| 3 | 16 |

Override only via `config.max_turns_override` (tests only).

`DriftCallObservation.budget_remaining = max_turns - turn`. When `turn >= max_turns` after a non-terminal action, termination fires (§3.5).

### 3.3 Drift firing order and side channels

Within a single `step()`, drift application occurs **before** the agent's action is dispatched (drift_injector.md §3.1). This has three consequences the env MUST enforce:

1. **Deterministic fold order.** If two drifts are scheduled for the same turn (possible in Stage 3 only), they apply in `(turn asc, pattern_id asc)` order. This is **env-enforced** — drift_injector does not sort.
2. **`force_drift_pattern` override (manual drift firing).** When `step()` is called with `force_drift_pattern=<pattern_id>`:
   - The forced pattern is resolved against `drift_injector.DRIFT_PATTERN_IDS`; unknown ids raise `InvalidActionError` BEFORE any state mutation.
   - Exactly ONE `DriftEvent` is synthesized for `turn_current` (not read from `drift_schedule`) and applied via `apply_drift`.
   - Any schedule-driven drifts that would have fired on `turn_current` are SUPPRESSED for this step and do NOT carry over to a later turn — judge intent overrides RNG (see deploy_demo_space.md §3.8, §7.3).
   - The synthesized event is appended to `state.drift_fired` with `from_version/to_version` taken from the pattern's catalogue entry; `drift_log` on the returned observation reflects it exactly like any scheduled drift. The source (manual vs scheduled) is NOT encoded in `DriftEvent`; the demo's trace panel decorates the row with `actor="drift", action_or_event="manual:<pattern_id>"` out-of-band (deploy_demo_space.md §3.8).
3. **Side-channel notices** (drift_injector §3.4 `side_channel_notice_append`): the vendor stashes the notice on `VendorState.side_channel_notice`. At the **top of the next `step()`** (not the same one that fired the drift), the env calls `emit_side_channel_if_pending` per mutated domain and, **when the agent's action dispatches a tool on that domain**, attaches the returned notice to `ToolResult.response["_notice"]`. One-shot: the same notice never appears on two ToolResults. If no tool on that domain is called, the notice remains pending until one is (or episode ends — it is then recorded on the final Episode.vendor_states_final and the reward pipeline may read it via R2 detection hints).

### 3.4 Observation builder (`_build_observation`)

A pure helper that projects `DriftCallState` → `DriftCallObservation` for the agent. Rules:

1. `turn`, `goal`, `budget_remaining = max_turns - turn` come straight from state.
2. `tool_results` = **full** episode history, not just the latest (DESIGN.md §4.1). Immutable tuple.
3. `drift_log` = `state.drift_fired` as-is. **The env does NOT hide unfired drifts from the agent**; `drift_schedule` is NEVER exposed (that would leak future information).
4. `available_tools` = union of every vendor's `TOOLS` constant whose domain is relevant to the goal, PLUS `"payment.charge"` and `"payment.refund"` (payment is cross-cutting). The set is **fixed for the episode** (DESIGN.md §4.1). Drift may change field shapes but never adds/removes a tool name.
5. `last_transcript` / `last_lang` / `last_confidence`:
   - **Turn 0** (reset): = `(goal.seed_utterance, goal.language, 1.0)` — the user's opening utterance, deterministic.
   - **After a `SPEAK`/`CLARIFY` agent action**: no new user utterance exists; the fields carry forward from the previous turn (they describe the **user's** latest message, not the agent's).
   - **When audio pipeline is enabled** (§3.7): after the env's sim-caller replies to a `CLARIFY`, `last_transcript/last_lang/last_confidence` are updated from the ASR output.

### 3.5 Termination

Termination sets `state.done = True` and invokes `rewards.compute_rewards` exactly once. `terminated_by` is one of:

| Value | Trigger |
|---|---|
| `"SUBMIT"` | Agent issued `SUBMIT`. Reward reflects full R1…R5. |
| `"ABORT"` | Agent issued `ABORT`. R1 forced to 0; R2…R5 still computed for diagnostics. |
| `"TIMEOUT"` | `turn >= max_turns` after a non-terminal action. R1 forced to 0. |
| `"ANTI_HACK"` | Triggered by ≥3 consecutive validation failures (tracked by an outer anti-hack handler in the caller, not the env itself) OR by state-corruption attempts detected by R5 (DESIGN.md §4.4, §7.1 R5). A single `InvalidActionError` raise from `_validate_action` is pure — it does NOT terminate the episode, does NOT record the action, and does NOT increment the turn counter. State-corruption-path ANTI_HACK termination records no ToolResult. |

Termination is **atomic within a `step()`**: either the step commits a terminal transition (and rewards are computed) or it raises before any state change. There is no partial termination.

### 3.6 Terminal-only accessors

`episode()` / `rewards()` are valid only after `done() == True`. `Episode` (§4.3) is built by snapshotting the final state and flattening it into the shape rewards.md §2.4 expects. Reward computation is **memoized** — two calls to `rewards()` return the same object (not just equal).

### 3.7 Optional audio boundary

`audio_boundary_enabled == False` (default, training path) means the env is **purely textual**: `DriftCallObservation.last_transcript` carries the goal's `seed_utterance` on turn 0 and a deterministic simulated user reply whenever the agent `CLARIFY`s (drawn from a scripted responder keyed on (seed, turn); see audio.md §3.1 non-training path).

`audio_boundary_enabled == True` (deployed env Space path) means:

1. On `reset()`, the env uses `tts_engine.synthesize(goal.seed_utterance, goal.language)` to produce the opening user audio for display/demo, but the **canonical `last_transcript` is still the `GoalSpec.seed_utterance`** — TTS is a sim-caller convenience, not a new source of truth.
2. On `CLARIFY`, the env's sim-caller composes a text reply, runs `tts_engine.synthesize`, then `asr_engine.transcribe`, and uses the ASR `TranscriptResult.text/language/confidence` to populate the next observation's `last_*` fields. This intentionally round-trips through audio to mirror production noise (DESIGN.md §9.4).
3. On `SPEAK`, the env optionally synthesizes TTS for the agent's reply — this is emitted on a side channel (e.g., FastAPI SSE) but **does not** enter the state.

The audio pipeline is a **post-observation wrapper**. The env NEVER feeds TTS bytes back into the RL loop; reward computation is 100% textual (DESIGN.md §7). Audio disabled ⇒ no `import driftcall.audio` anywhere in the training call graph (audio.md §6.3 import firewall).

### 3.8 Immutability & state evolution

Every `DriftCallState` transition goes through `dataclasses.replace`. The env holds **one** `self._state` reference; it is reassigned to a new frozen object on every transition. No vendor state, no drift log, no action tuple is ever mutated in place. The `tests/test_env.py` property test asserts `id(prev_state.actions) != id(next_state.actions)` whenever `len(next.actions) > len(prev.actions)`.

The env's own instance state fields:
- `self._config: EnvConfig` (frozen, set at `__init__`)
- `self._state: DriftCallState | None` (reassigned each step)
- `self._rewards: Rewards | None` (set once at termination)
- `self._episode: Episode | None` (set once at termination)
- `self._closed: bool`
- `self._seed: int | None`
- `self._episode_id: str | None`

No other mutable attributes.

### 3.9 Concurrency

A `DriftCallEnv` instance is **single-session**. The FastAPI shim in `app.py` creates one instance per session (pooled by episode_id). Sharing one instance across threads is undefined behavior (no locks, state reassignment is not atomic). This matches DESIGN.md §11.1 (env Space spawns one env per request).

---

## 4. Data structures

### 4.1 `EnvConfig` (frozen, env-internal)

```python
@dataclass(frozen=True)
class EnvConfig:
    curriculum_stage: Literal[1, 2, 3]
    language_weights: dict[LanguageCode, float]      # key-copied to dict in __init__
    audio_boundary_enabled: bool
    max_turns_override: int | None
    scheduler: DriftScheduler                         # drift_injector.DriftScheduler
    tts_engine: TTSEngine | None                      # audio.TTSEngine
    asr_engine: ASREngine | None                      # audio.ASREngine
```

Construction is an explicit `EnvConfig.from_mapping(config_dict) -> EnvConfig` classmethod that raises `InvalidConfigError` on any key not in the table above, on type mismatches, or on mutually-exclusive fields (e.g., `audio_boundary_enabled=True` with `tts_engine=None`). The dict fed into `__init__` is **never stored directly** — it is validated and frozen.

### 4.2 Internal session state

The env holds exactly one live `DriftCallState` reference (`self._state`). Every field inside is frozen or a tuple of frozen objects. `self._side_channel_pending: dict[str, str]` caches pending notices keyed by domain (pure dict, module-private, reassigned on each emit/consume) — see §3.3.

There is **no rolling log buffer** separate from state; `state.actions`, `state.drift_fired`, and the per-turn `tool_results` tuple in the observation are the only record.

### 4.3 `Episode` construction (terminal-only)

At termination the env synthesizes the `Episode` object (rewards.md §2.4) by:

```python
Episode(
    episode_id       = self._episode_id,
    goal             = self._state.goal,
    actions          = self._state.actions,
    tool_results     = observation_tool_results_at_termination,  # full tuple
    drift_log        = self._state.drift_fired,
    vendor_states_final   = {d: dataclasses.asdict(vs)
                             for d, vs in self._state.vendor_states.items()},
    schema_versions_final = dict(self._state.schema_versions),
    max_turns        = self._state.max_turns,
    turns_used       = len(self._state.actions),
    terminated_by    = <one of "SUBMIT" | "ABORT" | "TIMEOUT" | "ANTI_HACK">,
    stage            = self._config.curriculum_stage,
)
```

`vendor_states_final` is a **snapshot dict** (not the frozen `VendorState` dataclass), produced by `dataclasses.asdict` — this matches rewards.md §2.4's `dict[str, dict[str, Any]]`.

### 4.4 Frozen invariants (checked by property tests)

- `DriftCallEnv.__dataclass_params__` — not a dataclass; instance attrs are the only mutation surface.
- Every `self._state` transition produces a new object: `prev is not next`.
- `self._rewards` and `self._episode` are write-once.

---

## 5. Error modes

Every error is a typed exception from `driftcall.env.errors`. All extend `DriftCallEnvError` (which extends `Exception`). No bare `raise` anywhere.

| # | Exception | Raised from | Condition |
|---|---|---|---|
| E1 | `InvalidConfigError` | `__init__` / `EnvConfig.from_mapping` | Unknown key, wrong type, weights don't sum to 1, `audio_boundary_enabled=True` with missing TTS/ASR engine. |
| E2 | `EnvNotReadyError` | `step`, `state`, `episode`, `rewards` | Called before `reset()`. |
| E3 | `EnvClosedError` | `reset`, `step` | Called after `close()`. |
| E4 | `InvalidActionError` | `step._validate_action` | Any §3.1 validation failure. **Pure exception** — raised BEFORE any state mutation: no turn increment, no action stored, no termination marker set. The env remains in its pre-step state and is still valid for a subsequent `step()` call. Repeated malformed actions (≥3 consecutive) are the concern of an outer anti-hack handler in the caller, which may then terminate the episode with `terminated_by="ANTI_HACK"` via a separate mechanism (see §3.5). |
| E5 | `EpisodeAlreadyTerminalError` | `step` | `self._state.done == True` when `step` called. |
| E6 | `EpisodeNotTerminalError` | `episode`, `rewards` | Called before termination. |
| E7 | `ConcurrentStepError` | `step` | Reentrant call detected via a per-instance `_step_in_progress` flag — single-threaded usage is a hard invariant (§3.9). |
| E8 | `UnknownDomainError` | `step` (PROBE_SCHEMA), `_build_observation` | Agent named a vendor domain not present in state. |
| E9 | `UnknownToolError` | `step` (TOOL_CALL) | `tool_name` not in `self._available_tools()`. Treated as anti-hack (same path as E4). |
| E10 | `DriftInjectionError` | `step` (during fold) | Propagated from `drift_injector.apply_drift`; not caught — indicates an internal invariant violation, not an agent error. |
| E11 | `RewardComputationError` | `step` (termination) | Propagated from `rewards.compute_rewards` if Episode is malformed; this is a bug in env, never silently swallowed. |
| E12 | `AudioPipelineError` | `step` / `reset` | Only when `audio_boundary_enabled=True` and the engine raises. Surfaced as-is; episode does NOT terminate unless audio fails on `reset()` (then `E1`-style reset failure). |

**Never caught, never wrapped:** any exception not in the table escapes as-is — silent failure is a design-doc violation (CLAUDE.md §0, DESIGN.md §7).

---

## 6. Dependencies

### 6.1 Upstream (imports from)

- `driftcall.models` — all frozen dataclasses and `ActionType`.
- `driftcall.task_generator` — `generate(seed, stage, language_weights)` (task_generator.md §2 generate).
- `driftcall.drift_injector` — `build_schedule`, `apply_drift`, `DriftScheduler` protocol, list of drift exceptions.
- `driftcall.vendors.{airline,cab,restaurant,hotel,payment}` — the 5 vendor modules, each exposing `dispatch`, `initial_state`, `apply_schema_mutation`, `describe_schema`, `emit_side_channel_if_pending`, `TOOLS` (vendors.md §2.1 six-method interface).
- `driftcall.rewards` — `compute_rewards(episode) -> Rewards`. **Imported at module top; NOT lazy.**
- `driftcall.audio.tts_kokoro`, `driftcall.audio.asr_whisper` — **imported lazily**, guarded by `if self._config.audio_boundary_enabled:` to preserve the training-path import firewall (audio.md §6.3).
- `dataclasses`, `uuid`, `os` (for `os.urandom`), `typing` — stdlib.

### 6.2 Downstream (consumed by)

- `driftcall/app.py` — FastAPI OpenEnv server; owns the env pool (one instance per session).
- `training/train_grpo.py` — via `openenv.client.EnvClient` pointed at a local uvicorn process.
- `training/eval_baseline.py`, `training/eval_final.py` — same path.
- `demo/app_gradio.py` — for interactive demos.
- `tests/test_env.py`, `tests/test_e2e.py` — direct instantiation.

### 6.3 Prohibited dependencies

- **No LLM / HTTP calls.** The env composes deterministic, in-process modules only.
- **No direct disk I/O** beyond what `task_generator.load_templates` and `drift_injector.list_patterns` already own.
- **No `threading`, no `asyncio`** — the env is synchronous; concurrency is the caller's concern.
- **No mutable global state.** All config is per-instance.

---

## 7. Edge cases

### 7.1 Action issued before `reset()`

```python
env = DriftCallEnv(config)
env.step(some_action)   # raises E2 EnvNotReadyError
```

Same for `state()`, `episode()`, `rewards()`. `done()` returns `False` (not an error — "no episode" implies "not done").

### 7.2 Double `SUBMIT` (agent emits SUBMIT twice)

First `SUBMIT` terminates normally. Second `step` raises `E5 EpisodeAlreadyTerminalError` — the env does NOT silently ignore the second action. Callers MUST check `done()` between steps. This is intentional: in training the rollout loop always stops at `done=True`; anything else is a bug upstream.

### 7.3 `ABORT` mid-turn (after some tool calls)

Legal. `ABORT` always terminates with `terminated_by="ABORT"`, `R1=0.0` (forced by rewards.md §3.2). R2…R5 are still computed over the partial episode — the training signal reflects what the agent *did* do before giving up. `confidence` on the ABORT action is ignored (models.md §3.5 specifies confidence is for SUBMIT only; forbidden on ABORT per §3.1 Table).

### 7.4 Drift scheduled on a turn that is also terminal (TIMEOUT race)

Pre-condition: `drift_schedule` placement rules (drift_injector §3.2) exclude the last 3 turns for stage 2 and the last 3 for stage 3, so this shouldn't arise from `build_schedule`. But a scripted test scheduler could inject one.

**Resolution:** drift fires FIRST (step 3 in §2.3), then the agent's action is dispatched, then the budget check runs. So a drift on `turn == max_turns - 1` with the agent issuing `TOOL_CALL` that turn: the tool sees the new schema, the ToolResult is recorded, then budget check fires TIMEOUT. The drift is recorded in `drift_fired` and the Episode.

If a scripted scheduler places a drift on `turn > max_turns`, `build_schedule`'s own guard (drift_injector.md §5) raises `DriftScheduleError` at `reset()`. The env surfaces this as an `E1`-class reset failure.

### 7.5 Audio pipeline disabled but config passes `tts_engine`

`EnvConfig.from_mapping` raises `E1 InvalidConfigError` ("tts_engine must be None when audio_boundary_enabled is False"). Fail loud at construction; never silently drop.

### 7.6 Same tool called twice in a row with identical args

Legal. Vendors are pure; same `(vendor_state, args, seed)` ⇒ identical `ToolResult`. Both results appear in the history tuple. The env does not de-duplicate — R4 (format compliance, rewards.md §3.5) may penalize for wasted turns, but that is the reward layer's call, not the env's.

### 7.7 `PROBE_SCHEMA` on a domain with no prior interaction

Legal. `describe_schema(vendor_state, current_version)` is defined for any `(vendor, v1|v2|v3)` pair. PROBE_SCHEMA produces a `ToolResult` with `status="ok"`, `tool_name="probe:<domain>"`, `response=describe_schema(...)`, `schema_version=current`, `latency_ms=0` (probes are instantaneous). It counts against the budget like any other action (DESIGN.md §4.3).

### 7.8 `SUBMIT` with out-of-range `confidence`

Caught by `_validate_action`: `confidence` must be `float` in `[0.0, 1.0]`. Outside ⇒ `E4 InvalidActionError` → anti-hack termination.

### 7.9 Seed collision across concurrent episodes

Not an env concern — each `DriftCallEnv` instance is a single episode and holds its own `self._seed`. Two instances created with the same seed produce identical trajectories by design (reproducibility). Callers that want distinct episodes must supply distinct seeds.

### 7.10 Vendor `dispatch` returns a `ToolResult` for a drift-renamed field

Vendors' `dispatch` already handles this — the vendor reads `schema_versions[domain]` and returns the schema's current field names. The env does not post-process `ToolResult.response`; it is passed verbatim to the observation so the agent sees the actual post-drift shape. R2's detection hints (drift_injector `DriftPattern.detection_hints`) are the contract for the reward layer.

### 7.11 Closed env still queried

After `close()`, `reset` and `step` raise `E3 EnvClosedError`. `done()` returns whatever it returned before close (sticky). `state()`, `episode()`, `rewards()` return the cached terminal objects if termination happened before close — callers can still audit a finished episode post-close.

---

## 8. Examples

### 8.1 Stage-1 happy path (no drift, SUBMIT with full credit)

Stage 1: English airline booking, no drift, agent completes in 5 turns.

```python
from driftcall.env import DriftCallEnv
from driftcall.models import DriftCallAction, ActionType

env = DriftCallEnv({"curriculum_stage": 1})           # audio disabled (default)
obs = env.reset(seed=42)
# obs.turn == 0
# obs.goal.domain == "airline"            (example; seed-dependent)
# obs.goal.language == "en"               (seed+weights dependent)
# obs.drift_log == ()                     (stage 1)
# obs.budget_remaining == 8               (stage 1)
# obs.last_transcript == obs.goal.seed_utterance

obs = env.step(DriftCallAction(
    action_type=ActionType.TOOL_CALL,
    tool_name="airline.search",
    tool_args={"from_": "HYD", "to": "BLR", "date": "2026-04-25"},
))
# obs.turn == 1
# len(obs.tool_results) == 1, tool_results[0].status == "ok"
# obs.drift_log == ()                     (still none)

# ... 3 more tool calls to book + confirm ...

obs = env.step(DriftCallAction(
    action_type=ActionType.SUBMIT,
    confidence=0.9,
    message="Booking confirmed, PNR X3K9Z, total ₹7200.",
))
# env.done() == True
# env.rewards().r1 == 1.0          (goal satisfied)
# env.rewards().r2 == 0.5          (stage 1 → R2 neutral)
# env.rewards().reward ∈ [0.85, 1.0]   (quality minus tiny Brier)
# env.episode().terminated_by == "SUBMIT"
# env.episode().turns_used == 5
```

### 8.2 Stage-2 drift detected and adapted

Stage 2 airline with `price_rename` drift at turn 3. Agent detects the drift on turn 4 via a `SPEAK` that mentions the rename, then adapts.

```python
env = DriftCallEnv({"curriculum_stage": 2})
obs = env.reset(seed=7)
# obs.goal.domain == "airline"
# state.drift_schedule has length 1; e.g. turn=3, pattern_id="airline.price_rename"

# Turn 1: search (pre-drift) — returns schema v1 with "price" field
obs = env.step(DriftCallAction(action_type=ActionType.TOOL_CALL,
                               tool_name="airline.search",
                               tool_args={"from_": "HYD", "to": "BLR", "date": "2026-04-25"}))

# Turn 2: agent picks a flight (still v1)
# Turn 3: DRIFT FIRES at start of step — schema moves to v2
#         agent's turn-3 action sees the new schema already.
obs = env.step(DriftCallAction(action_type=ActionType.TOOL_CALL,
                               tool_name="airline.search",
                               tool_args={"from_": "HYD", "to": "BLR", "date": "2026-04-25"}))
# obs.drift_log == (DriftEvent(turn=3, drift_type="schema", domain="airline",
#                              pattern_id="airline.price_rename", from_version="v1",
#                              to_version="v2", description="price renamed to total_fare_inr"),)
# obs.tool_results[-1].response uses "total_fare_inr" not "price"

# Turn 4: agent announces the rename (R2 detection window = [3, 5], mention hits)
obs = env.step(DriftCallAction(action_type=ActionType.SPEAK,
                               message="Note: the price field was renamed to total_fare_inr."))

# Turn 5: adapted tool_args — book with new field awareness
obs = env.step(DriftCallAction(action_type=ActionType.TOOL_CALL,
                               tool_name="airline.book",
                               tool_args={"flight_id": "6E-2345", "payment_token": "tok_v1"}))

# Turn 6: submit
obs = env.step(DriftCallAction(action_type=ActionType.SUBMIT, confidence=0.8,
                               message="Booked with updated pricing."))
# env.rewards().r1 == 1.0
# env.rewards().r2 == 1.0          (drift detected within 2-turn window, mention in SPEAK)
# env.rewards().reward ≈ 0.90
```

### 8.3 Stage-3 compound drift, TIMEOUT

Stage 3, two drifts: airline policy at turn 3, payment auth at turn 9. Agent handles the first but runs out of budget on the second.

```python
env = DriftCallEnv({"curriculum_stage": 3})
obs = env.reset(seed=2026)
# state.drift_schedule has length 2
# obs.budget_remaining == 16

# ... turns 1-8 consumed detecting and adapting to first drift ...
# turn 9: payment auth drift fires; old token_v1 returns 401 on payment.charge
# turns 10-15: agent probing and retrying, accumulating 6 actions
# turn 16: budget exhausted → TIMEOUT

# Final step (turn 16): agent attempted another payment.charge, TIMEOUT fires
# after action dispatch because turn == max_turns.

assert env.done()
assert env.episode().terminated_by == "TIMEOUT"
assert env.rewards().r1 == 0.0            # TIMEOUT forces R1=0
assert env.rewards().r2 in (0.5, 1.0)     # one of two drifts may still have been detected
assert env.rewards().reward < 0.3         # quality dominated by r3, r4; brier small
```

### 8.4 Audio-boundary-enabled deployed Space

```python
from driftcall.audio.tts_kokoro import get_tts_engine
from driftcall.audio.asr_whisper import get_asr_engine

env = DriftCallEnv({
    "curriculum_stage": 1,
    "audio_boundary_enabled": True,
    "tts_engine": get_tts_engine(),
    "asr_engine": get_asr_engine(),
})
obs = env.reset(seed=11)
# obs.last_transcript is still the textual seed_utterance (canonical source);
# the TTS audio is emitted on a side channel for UI rendering.

# Agent asks a clarifying question
obs = env.step(DriftCallAction(action_type=ActionType.CLARIFY,
                               message="When do you want to depart, morning or evening?"))
# env synthesizes a sim-caller reply ("shaam ko, 7 baje"), TTS→ASR round-trips,
# and obs.last_transcript/last_lang/last_confidence now reflect the ASR result.
# last_confidence may be < 1.0 due to whisper uncertainty.
```

---

## 9. Open questions

1. **Sim-caller responder for `CLARIFY` in text-only mode.** This doc specifies "deterministic scripted responder keyed on (seed, turn)" but the responder is not yet specified anywhere. Where does it live? Options: (a) an internal helper in `env.py`, (b) a new `driftcall.simcaller` module, (c) a field on the template in `task_generator`. *Proposal:* keep it in `env.py` as a private helper; it is env-internal plumbing and does not need a module boundary. To be resolved at critic gate.

2. **`PROBE_SCHEMA` tool_name vs domain.** DESIGN.md §4.1 shows `tool_name` for `PROBE_SCHEMA`, but semantically the action names a *domain*, not a tool. This doc uses the convention "PROBE_SCHEMA's `tool_name` holds the domain string (e.g., `'airline'`)". Alternative: introduce an explicit `domain` field on `DriftCallAction`. Pick one and reflect in `models.md`. *Proposal:* keep `tool_name` carrying the domain, add a models.md clarifying note. Cheaper than a schema change.

3. **Side-channel notice attachment ordering.** If two drifts on the same domain both stash notices (stage 3 edge case), does the agent see both concatenated, or only the first, on the next matching tool call? This doc says "one-shot per call, carry-over until consumed"; need a concrete concatenation rule if both are pending. *Proposal:* concat with `\n---\n` separator, emit all pending notices at once, clear together. Resolve with drift_injector critic.

4. **`env.episode().vendor_states_final` field-stability across refactors.** Rewards consumes `vendor_states_final` as a raw `dict[str, dict[str, Any]]`. If a vendor later adds/removes a VendorState field, the Episode's shape changes. Do we need a versioned snapshot schema? *Proposal:* No; rewards reads only documented fields and uses `.get(..., default)` everywhere. Confirm with rewards critic.

5. **Deterministic episode_id.** Currently `uuid4()` — *non-deterministic*. This is intentional for audit uniqueness but means two `reset(seed=42)` calls produce different `episode_id`s. Is that acceptable to DESIGN.md §4.2's "Deterministic for reproducibility"? *Proposal:* Yes — the *trajectory* is deterministic, the audit ID is not. Same stance as rewards.md (episode_id is opaque). Flag for gate-D cross-doc review.

6. **Thread-safety guard strength.** `E7 ConcurrentStepError` requires a per-instance `_step_in_progress` bool. This catches re-entrancy but not two threads colliding outside the GIL (same bool is not atomic). Do we need `threading.Lock`? *Proposal:* No — §3.9 declares single-threaded contract; the bool is diagnostic, not an invariant. Note in code comment; do not add a lock (adds import, slows step).

7. **Audio engine lifecycle on `close()`.** TTS/ASR engines are singletons (audio.md §2.1 `get_tts_engine` / §2.2 `get_asr_engine`). If multiple envs share them, `close()` must NOT free the engine. *Proposal:* `close()` releases only per-env state (drops `self._state`, `self._rewards`, `self._episode`) and sets `self._closed=True`. Audio engines are process-global and outlive the env. Confirm at critic gate.
