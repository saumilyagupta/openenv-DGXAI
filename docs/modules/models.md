# models.md — DriftCall Core Dataclasses

**Owner:** Person A (Environment)
**Implements:** DESIGN.md §4.1 (Dataclasses), §4.2 (reset), §4.3 (step), §4.4 (termination), §7.1 (Reward inputs)
**Status:** DRAFT — pending ≥ 2 fresh critic rounds

---

## 1. Purpose

`driftcall/models.py` is the single home for every immutable data type that crosses a module boundary in the DriftCall environment. It defines the on-the-wire contract between the agent, the env core, the vendor mocks, the drift injector, the reward suite, and the FastAPI surface. Nothing in this module does work — it only declares shape.

The module serves five consumers simultaneously:

1. **The agent loop** — builds `DriftCallAction` instances and receives `DriftCallObservation`.
2. **The env core** (`env.py`) — consumes actions, emits observations, owns `DriftCallState`.
3. **The vendor mocks** (`vendors/*.py`) — return `ToolResult`.
4. **The drift injector** (`drift_injector.py`) — emits `DriftEvent` and mutates vendor schemas.
5. **The reward suite** (`rewards.py`) — reads episode trails (actions + tool_results + drift_log) to compute R1–R5.

Because all dataclasses are `frozen=True`, every state transition produces a new object. This matches the project-wide immutability rule (CLAUDE.md §7) and makes episode replay trivially deterministic.

---

## 2. Interface

Every declaration below is the *exact* target signature. No additional fields may be introduced without a DESIGN.md update first.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ActionType(str, Enum):
    TOOL_CALL = "tool_call"
    SPEAK = "speak"
    CLARIFY = "clarify"
    PROBE_SCHEMA = "probe_schema"
    SUBMIT = "submit"
    ABORT = "abort"


@dataclass(frozen=True)
class DriftCallAction:
    action_type: ActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    message: str | None = None
    confidence: float | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: Literal["ok", "schema_error", "policy_error", "auth_error", "timeout"]
    response: dict[str, Any]
    schema_version: str
    latency_ms: int


@dataclass(frozen=True)
class DriftEvent:
    turn: int
    drift_type: Literal["schema", "policy", "tnc", "pricing", "auth"]
    domain: str
    description: str
    from_version: str
    to_version: str
    pattern_id: str                     # registry key — matches drift_injector catalogue


@dataclass(frozen=True)
class GoalSpec:
    domain: str
    intent: str
    slots: dict[str, Any]
    constraints: dict[str, Any]
    language: Literal["hi", "ta", "kn", "en", "hinglish"]
    seed_utterance: str


@dataclass(frozen=True)
class DriftCallObservation:
    turn: int
    goal: GoalSpec
    last_transcript: str
    last_lang: str
    last_confidence: float
    tool_results: tuple[ToolResult, ...]
    drift_log: tuple[DriftEvent, ...]
    budget_remaining: int
    available_tools: tuple[str, ...]


@dataclass(frozen=True)
class DriftCallState:
    episode_id: str
    goal: GoalSpec
    vendor_states: dict[str, dict[str, Any]]
    schema_versions: dict[str, str]
    drift_schedule: tuple[DriftEvent, ...]
    drift_fired: tuple[DriftEvent, ...]
    turn: int
    max_turns: int
    actions: tuple[DriftCallAction, ...]
    done: bool
```

Exports (module `__all__`):

```python
__all__ = [
    "ActionType",
    "DriftCallAction",
    "ToolResult",
    "DriftEvent",
    "GoalSpec",
    "DriftCallObservation",
    "DriftCallState",
]
```

No factory helpers, no `from_dict` / `to_dict` methods — serialization lives in `env.py` and `app.py` (see §6). This module is pure shape.

---

## 3. Behavior spec

### 3.1 Immutability guarantees

- Every class is `@dataclass(frozen=True)`. Attribute assignment after construction raises `dataclasses.FrozenInstanceError`.
- Sequence-typed fields use `tuple[...]` (not `list[...]`). This is both semantic ("append means new object") and structural (tuples hash cleanly).
- Dict-typed fields (`tool_args`, `response`, `slots`, `constraints`, `vendor_states`, `schema_versions`) are conventionally immutable — the module does not deep-freeze them, and consumers MUST NOT mutate them in place. The env core always constructs new dicts when state evolves (see §3.3). Consumers that need to defensively copy should use `copy.deepcopy` at read time.
- The decision to leave dicts unfrozen (vs. using `types.MappingProxyType`) is pragmatic: JSON round-trips through FastAPI / TRL / `openenv` tooling produce plain `dict`, and wrapping them would force an unwrap on every serialization.

### 3.2 Equality and hashing

- `frozen=True` combined with the default `eq=True` means `__hash__` is auto-generated using the tuple of fields.
- **A dataclass with a `dict` field is not hashable** because `dict` is not hashable. Concretely:
  - `ActionType` — hashable (Enum).
  - `DriftCallAction` — **not hashable** (`tool_args` is `dict`).
  - `ToolResult` — **not hashable** (`response` is `dict`).
  - `DriftEvent` — hashable (all fields are primitive strings/ints).
  - `GoalSpec` — **not hashable** (`slots`, `constraints` are dicts).
  - `DriftCallObservation` — **not hashable** (contains `GoalSpec` + tuples of unhashable `ToolResult`).
  - `DriftCallState` — **not hashable** (contains dicts and unhashable nested types).
- Equality comparison still works for all of them (it compares field-by-field, and dict equality is value-based). Callers that need a hash key should compute one explicitly, e.g. via `hash(episode_id)` or a canonical JSON dump.

### 3.3 State evolution pattern

Mutation is forbidden; the env core always uses `dataclasses.replace` to emit new states:

```python
new_state = replace(state, turn=state.turn + 1, actions=state.actions + (action,))
```

For dict fields, the env core builds a fresh dict:

```python
new_versions = {**state.schema_versions, "airline": "v2"}
new_state = replace(state, schema_versions=new_versions)
```

This is a convention enforced by code review + tests — the `models.py` module itself cannot prevent an ill-behaved caller from `state.vendor_states["airline"].update(...)`. See §5 for the enforcement strategy.

### 3.4 Serialization semantics

- `models.py` does not define serializers. `env.py` uses `dataclasses.asdict` for outbound JSON (FastAPI boundary), which handles nested tuples/dicts correctly. `Enum` values serialize to their `.value` (`"tool_call"`, etc.) because `ActionType` inherits from `str`.
- Deserialization (inbound `DriftCallAction` from HTTP) happens in `app.py` with a pydantic model that mirrors the fields — pydantic validates types, then constructs the frozen dataclass. `models.py` is not involved.
- Required invariant for any serializer: the JSON round-trip `action → json → action` must produce an equal object (`==` returns True) for every valid action.

### 3.5 Field-level invariants

| Class | Field | Invariant | Enforced by |
|---|---|---|---|
| `DriftCallAction` | `action_type == TOOL_CALL` | `tool_name` and `tool_args` both non-None | `env.step` validation |
| `DriftCallAction` | `action_type == SPEAK` or `CLARIFY` | `message` non-None | `env.step` validation |
| `DriftCallAction` | `action_type == SUBMIT` | `confidence` in `[0.0, 1.0]` | `env.step` validation + reward pipeline |
| `DriftCallAction` | `action_type == PROBE_SCHEMA` | `tool_name` ∈ `{"airline", "cab", "restaurant", "hotel", "payment"}` (bare domain name — NOT `domain.verb` format); `tool_args` may be `None` or `{}`; `confidence` must be `None` (probe is not a submit) | `env.step` validation |
| `DriftCallAction` | `rationale` | len ≤ 200 chars if non-None | `env.step` validation (R4 penalizes excess) |
| `ToolResult` | `status == "ok"` | `response` non-empty dict | vendor mock contract |
| `ToolResult` | `schema_version` | matches `/^v\d+$/` | vendor mock contract |
| `ToolResult` | `latency_ms` | non-negative int | vendor mock contract |
| `DriftEvent` | `turn` | `1 ≤ turn ≤ max_turns - 1` (drifts fire strictly before the final turn so the agent has ≥ 1 post-drift turn to react; consistent with DESIGN.md §6.2 stage-2/3 schedules) | drift injector |
| `DriftEvent` | `from_version != to_version` | always | drift injector + drift pattern library |
| `GoalSpec` | `language` | one of the 5 Literals | task generator |
| `DriftCallObservation` | `budget_remaining` | `max_turns - turn`, `≥ 0` | env observation builder |
| `DriftCallState` | `len(actions) == turn` | always (one action per turn) | env.step |
| `DriftCallState` | `drift_fired ⊆ drift_schedule` | always (subset, order preserved) | drift injector |
| `DriftCallState` | `done == True` → no further step allowed | enforced by env | `env.step` raises on terminated state |

`models.py` itself does NOT enforce any of these — they are contracts validated in `env.py`, `vendors/*.py`, and `drift_injector.py`. `models.py` is purely declarative.

---

## 4. Data structures

Every field is documented: name, type, semantic meaning, constraints, and who writes it.

### 4.1 `ActionType` (Enum)

`str`-mixed Enum so instances serialize naturally to JSON strings.

| Value | Meaning | Required companion fields on `DriftCallAction` |
|---|---|---|
| `TOOL_CALL` | Invoke a mock vendor tool | `tool_name`, `tool_args` |
| `SPEAK` | Emit a text reply to the user (TTS at deploy boundary) | `message` |
| `CLARIFY` | Ask a clarifying question back to the user | `message` |
| `PROBE_SCHEMA` | Ask the env for the current schema snapshot of a domain | `tool_name` = domain (e.g., `"airline"`) |
| `SUBMIT` | Declare the task complete and attach `confidence` | `confidence` |
| `ABORT` | Explicit failure — terminate episode with R1=0 | none |

### 4.2 `DriftCallAction`

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `action_type` | `ActionType` | Which action variant is being taken | Required; no default | Agent |
| `tool_name` | `str \| None` | Fully-qualified tool id (`"airline.search"`) or domain for `PROBE_SCHEMA` | `None` unless `action_type ∈ {TOOL_CALL, PROBE_SCHEMA}`; format `domain.verb` for tool calls | Agent |
| `tool_args` | `dict[str, Any] \| None` | Keyword arguments for the tool call | `None` unless `action_type == TOOL_CALL`; JSON-serializable; no nested callables | Agent |
| `message` | `str \| None` | Utterance text (user-language-mirrored) | Non-None when `action_type ∈ {SPEAK, CLARIFY}`; Unicode (Devanagari / Tamil / Kannada welcome); max ~4 KB | Agent |
| `confidence` | `float \| None` | Self-assessed probability of task success | Required when `action_type == SUBMIT`; `0.0 ≤ c ≤ 1.0`; feeds Brier term in §7.2 | Agent |
| `rationale` | `str \| None` | Optional chain-of-thought / reasoning | Max 200 chars enforced at step-time; excess → R4 penalty | Agent |

### 4.3 `ToolResult`

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `tool_name` | `str` | Echoes the invoked tool | Must equal the triggering `DriftCallAction.tool_name` | Vendor mock |
| `status` | Literal (5-value) | Outcome classification — drives R2/R5 detection signals | Exactly one of `"ok"`, `"schema_error"`, `"policy_error"`, `"auth_error"`, `"timeout"` | Vendor mock |
| `response` | `dict[str, Any]` | Raw response body; shape depends on tool + current schema version | JSON-serializable; on non-ok status, contains `error_code` key | Vendor mock |
| `schema_version` | `str` | Version stamp of the schema used to serialize `response` | Matches `^v\d+$` — currently `"v1"`, `"v2"`, `"v3"` | Vendor mock |
| `latency_ms` | `int` | Simulated latency (deterministic per seed) | `≥ 0`; typical 50–400 ms; `timeout` status → 5000+ | Vendor mock |

### 4.4 `DriftEvent`

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `turn` | `int` | Turn at which the drift fires (start-of-turn, before action evaluation, DESIGN.md §6.2) | `1 ≤ turn ≤ max_turns - 1` | Drift injector |
| `drift_type` | Literal (5-value) | Taxonomy — DESIGN.md §6.1 | Exactly one of `"schema"`, `"policy"`, `"tnc"`, `"pricing"`, `"auth"` | Drift injector |
| `domain` | `str` | Target vendor | One of `"airline"`, `"cab"`, `"restaurant"`, `"hotel"`, `"payment"` | Drift injector |
| `description` | `str` | Human-readable, used by R2 keyword match | Non-empty; ≤ 256 chars; includes drifted field name where applicable | Drift injector |
| `from_version` | `str` | Schema version before drift | Matches `^v\d+$`; must differ from `to_version` | Drift injector |
| `to_version` | `str` | Schema version after drift | Matches `^v\d+$` | Drift injector |

### 4.5 `GoalSpec`

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `domain` | `str` | Primary vendor domain for this goal | One of the 4 consumer domains (`payment` is transversal, not a goal domain) | Task generator |
| `intent` | `str` | Intent id (e.g., `"book_flight"`, `"order_food"`) | From a closed set defined in `task_generator.md` | Task generator |
| `slots` | `dict[str, Any]` | Parsed required + optional slots (`{"from": "HYD", "to": "BLR", "when": "2026-04-30"}`) | All values JSON-serializable primitives; keys are string slot names | Task generator |
| `constraints` | `dict[str, Any]` | Budget / time window / dietary / etc. | Keys drawn from constraint vocabulary documented in `rewards.md`; values JSON primitives | Task generator |
| `language` | Literal (5-value) | Target language for the brief and expected reply mirror | Exactly one of `"hi"`, `"ta"`, `"kn"`, `"en"`, `"hinglish"` | Task generator |
| `seed_utterance` | `str` | The raw user utterance (text-form even in training) | Non-empty Unicode; no PII | Task generator |

### 4.6 `DriftCallObservation`

The agent-facing view. Must never leak internal state beyond what DESIGN.md §4.3 defines.

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `turn` | `int` | Current turn (0 at reset, incremented at step start) | `0 ≤ turn ≤ max_turns` | Env observation builder |
| `goal` | `GoalSpec` | Immutable goal for the episode — copied by ref from state | Identical `GoalSpec` across all observations in one episode | Env observation builder |
| `last_transcript` | `str` | Most recent user utterance in text form (post-ASR in deploy, as-authored in training) | Empty string on turn 0 if no prior utterance | Env observation builder |
| `last_lang` | `str` | Language detected from last utterance | One of the 5 language literals or `""` on turn 0 | Env observation builder |
| `last_confidence` | `float` | ASR confidence for `last_transcript` (1.0 in training) | `0.0 ≤ c ≤ 1.0` | Env observation builder |
| `tool_results` | `tuple[ToolResult, ...]` | Full history of tool results this episode | Order = chronological; length grows by ≤ 1 per turn | Env observation builder |
| `drift_log` | `tuple[DriftEvent, ...]` | Drifts that HAVE fired (subset of schedule, order preserved) | Monotonically growing across turns | Env observation builder |
| `budget_remaining` | `int` | `max_turns - turn` | `≥ 0` | Env observation builder |
| `available_tools` | `tuple[str, ...]` | Fully-qualified tool ids the agent may call this turn | Stable within an episode (auth-drifted tools still listed but will return `auth_error`) | Env observation builder |

### 4.7 `DriftCallState`

Env-internal authoritative state. The agent never sees this directly.

| Field | Type | Semantic | Constraint | Writer |
|---|---|---|---|---|
| `episode_id` | `str` | Opaque unique id (e.g., `"ep_000123"` or a UUID4) | Unique per session; stable across one episode | Env (at reset) |
| `goal` | `GoalSpec` | Same object as observation.goal | Never changes within an episode | Env (at reset) |
| `vendor_states` | `dict[str, dict[str, Any]]` | Mutable mock DBs keyed by domain (`{"airline": {...}, ...}`) | Top-level keys = 5 domains; inner shape defined per-vendor in `vendors.md` | Env + vendor mocks (via `replace`) |
| `schema_versions` | `dict[str, str]` | Current schema version per domain | Keys = 5 domains; values `^v\d+$`; monotonically advanced by drift injector | Drift injector |
| `drift_schedule` | `tuple[DriftEvent, ...]` | Pre-computed schedule sampled at reset (DESIGN.md §6.2) | Sorted by `turn` ascending; length 0 / 1 / 2 for curriculum stages 1 / 2 / 3 | Drift injector (at reset) |
| `drift_fired` | `tuple[DriftEvent, ...]` | Drifts that have already fired | Prefix of `drift_schedule` by turn order | Drift injector |
| `turn` | `int` | Current turn | `0 ≤ turn ≤ max_turns`; starts at 0, increments in `step` | Env |
| `max_turns` | `int` | Turn budget (8 / 12 / 16 per curriculum stage, DESIGN.md §4.5) | `> 0`; stable across episode | Env (at reset) |
| `actions` | `tuple[DriftCallAction, ...]` | Full agent action history | `len == turn` invariant | Env |
| `done` | `bool` | Terminal flag | `False` until SUBMIT / ABORT / timeout / R5 corruption | Env |

---

## 5. Error modes

`models.py` itself has effectively zero runtime error surface — it only declares frozen dataclasses. Errors arise in the following situations:

| Situation | Exception | Where raised |
|---|---|---|
| Caller assigns to a frozen field, e.g. `action.tool_name = "x"` | `dataclasses.FrozenInstanceError` | Python runtime (automatic) |
| Caller constructs `DriftCallAction` with wrong-typed `action_type` (not an `ActionType`) | `TypeError` at construction if type-checked; otherwise mypy catches it | Python runtime / mypy |
| Caller constructs a dataclass missing a required field | `TypeError: __init__() missing N required positional arguments` | Python runtime (automatic) |
| Caller passes `language` outside the 5-value Literal | Not enforced at runtime by Python — accepted. mypy `--strict` catches it. `env.step` validates and raises `ValueError` for HTTP callers. | env.py / app.py validation layer |
| Caller passes an unhashable object (e.g., set) inside `tool_args` | No error at construction; later `dataclasses.asdict` or `json.dumps` raises `TypeError` | Serialization layer (env.py / app.py) |
| Vendor mock returns a non-JSON-serializable value (e.g., `set`, `bytes`, a custom class instance) inside `ToolResult.response` | No error at `ToolResult` construction; `TypeError` raised later at the FastAPI/JSON serialization boundary (`json.dumps` in `env.py` / `app.py`). **Mitigation:** every vendor test (`tests/test_vendors.py`) must assert JSON round-trip safety (`json.loads(json.dumps(result.response))`) for every `ToolResult.response` the vendor can produce. | Serialization layer (env.py / app.py); detection NOT at construction time |
| Caller attempts to hash `DriftCallAction` / `ToolResult` / `GoalSpec` / `DriftCallObservation` / `DriftCallState` | `TypeError: unhashable type` (because of nested `dict`) | Python runtime when the hash is attempted |
| Caller mutates a `dict` field in place (e.g., `state.vendor_states["airline"]["x"] = 1`) | **No exception** — this is a convention violation. Enforcement is review + tests (a property test snapshots `state.vendor_states` before each step and diffs after; any unintended mutation fails the test). | Enforced by `tests/test_env.py` property tests, not by `models.py` |

**Partial-data behavior:** no dataclass in this module supports "partial" construction. Every required field must be provided, or `__init__` raises. Optional fields on `DriftCallAction` default to `None`. There is no `from_partial_dict` helper — callers build actions explicitly.

---

## 6. Dependencies

### 6.1 Stdlib only

`models.py` imports only:

- `__future__.annotations` — PEP 563 deferred-evaluation annotations (mandatory per CLAUDE.md §4.2)
- `dataclasses.dataclass`, `dataclasses.field` — for frozen dataclass declarations
- `enum.Enum` — for `ActionType`
- `typing.Any`, `typing.Literal` — for dict and enumerated-string fields

**No third-party imports.** No pydantic, no attrs, no msgspec. This keeps `models.py` importable inside the Unsloth training loop, the FastAPI server, and the reward suite with zero install cost.

### 6.2 Downstream consumers (who imports `models.py`)

| Consumer module | Uses |
|---|---|
| `driftcall/env.py` | All 7 classes + `ActionType`. Central composition point. |
| `driftcall/rewards.py` | `DriftCallState`, `DriftCallAction`, `DriftEvent`, `ToolResult`, `ActionType`, `GoalSpec` — reads the episode trail to compute R1–R5. |
| `driftcall/drift_injector.py` | `DriftEvent`, `DriftCallState` — emits events, returns new state. |
| `driftcall/vendors/*.py` (5 files) | `ToolResult`. Each vendor returns `ToolResult` from its tool handlers. |
| `driftcall/task_generator.py` | `GoalSpec` — procedurally samples goals. |
| `driftcall/audio/asr_whisper.py` | None directly — it produces raw transcript+lang+confidence which `env.py` embeds into `DriftCallObservation`. |
| `driftcall/audio/tts_kokoro.py` | `DriftCallAction` (reads `.message` field for TTS synthesis). |
| `app.py` (FastAPI) | All 7 classes for request/response (de)serialization via companion pydantic models. |
| `training/train_grpo.py` | `DriftCallObservation`, `DriftCallAction`, `ActionType` — builds prompts + parses completions. |
| `training/eval_baseline.py`, `training/eval_final.py` | Same as training. |
| `demo/app_gradio.py` | `DriftCallObservation`, `DriftCallAction`, `ActionType` — drives the Gradio trace panel. |
| `tests/test_*.py` | All 7 classes for fixture construction. |

### 6.3 Upstream dependencies of `models.py`

**None.** `models.py` is a leaf — the graph flows outward from it. This is deliberate: making it dependency-free means it can be imported in the trainer process without dragging in FastAPI, whisper, or vendor mocks.

---

## 7. Edge cases

Numbered edge cases with expected behavior. These are the cases the test plan (`docs/tests/models_tests.md`) must cover.

1. **Empty `tool_results` / `drift_log` at turn 0.** `DriftCallObservation` constructed at `reset()` has `tool_results=()` and `drift_log=()`. Both must type-check as empty tuples, not `None`. Tests must assert `isinstance(obs.tool_results, tuple)` and `len(obs.tool_results) == 0`. Downstream code iterating with `for r in obs.tool_results:` works correctly on empty.

2. **`None` vs empty string on `last_transcript` / `last_lang`.** At turn 0, before any user utterance has been processed, `last_transcript=""` and `last_lang=""` (empty strings), NOT `None`. This keeps the field non-nullable (typing simpler for the agent) and makes `len(last_transcript) == 0` a clean "no-utterance" check. `last_confidence=1.0` at turn 0 (treated as "authored", perfect-ASR placeholder).

3. **Unicode slots in `GoalSpec.seed_utterance` and `DriftCallAction.message`.** Hindi (Devanagari), Tamil, Kannada, and mixed Hinglish strings must round-trip through `dataclasses.asdict` → `json.dumps(ensure_ascii=False)` → `json.loads` → construction unchanged. Tests must cover at least: `"मुझे कल दिल्ली जाना है"`, `"{when} அன்று விமானம்"`, `"{when} inda {to} ge"`, `"Bhai Friday ko Bangalore jaana hai"`.

4. **`tool_args` with nested dicts and lists.** Agents may pass `{"filters": {"class": ["economy", "premium"], "max_stops": 1}}`. Nested structures must survive JSON round-trip. Non-JSON-serializable values (e.g., `set`, `datetime`) are rejected by `env.step` validation, not by the dataclass itself — this edge case is about documenting that `models.py` does NOT validate, so callers must.

5. **Large `drift_log` history.** Stage 3 episodes allow up to 2 drifts, so `drift_log` length is `≤ 2` in practice. However, `DriftCallState.actions` can grow to `max_turns = 16` entries. The observation builder must NOT truncate; full history is always included because R1–R5 need it at submit time. Serializer must handle a 16-action tuple without pathological blowup (sanity check: < 64 KB JSON per typical observation).

6. **`DriftCallAction` with `action_type=SUBMIT` and `confidence=None`.** Construction succeeds (no runtime check). `env.step` validation must reject with `ValueError("SUBMIT requires confidence")`. Documenting this here because the dataclass's loose default (`confidence: float | None = None`) could mislead a reader into thinking SUBMIT without confidence is valid.

7. **`ToolResult.status != "ok"` with an empty `response` dict.** Permitted. Vendor mocks returning `schema_error` / `policy_error` / `auth_error` / `timeout` MUST still populate `response` with at least `{"error_code": "<CODE>"}` so R2 can keyword-match. Empty `response={}` is allowed by the dataclass but is a vendor-contract bug caught in `tests/test_vendors.py`.

8. **`DriftEvent.from_version == to_version`.** The Drift Injector MUST NOT emit such events; the drift pattern library rejects them at load. `models.py` does not enforce — but see §3.5 invariants. Tested by `drift_injector` tests, not here.

9. **Constructing `DriftCallState` with `len(actions) != turn`.** `models.py` accepts this silently. It is a severe env-core bug if it happens; tested by a property assertion in `tests/test_env.py` on every step. Documented here so critics know the invariant's home.

10. **`available_tools` on an `auth_error`-drifted tool.** Still listed. Agents should attempt calls to discover the auth drift; removing the tool would leak the drift. This is a critical design decision — documented in the constraint column of §4.6 and repeated here for visibility.

---

## 8. Examples

### 8.1 Constructing a valid `TOOL_CALL` action

```python
from __future__ import annotations

from driftcall.models import ActionType, DriftCallAction

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
    rationale="User asked for cheapest evening flight under 8000",
)

assert action.action_type is ActionType.TOOL_CALL
assert action.tool_name == "airline.search"
assert action.confidence is None  # not a SUBMIT
```

### 8.2 Constructing a `ToolResult` after a successful search

```python
from driftcall.models import ToolResult

result = ToolResult(
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
            }
        ]
    },
    schema_version="v1",
    latency_ms=142,
)

assert result.status == "ok"
assert result.response["results"][0]["price"] == 7200
```

### 8.3 Constructing a complete `DriftCallObservation` at turn 0 (reset)

```python
from driftcall.models import DriftCallObservation, GoalSpec

goal = GoalSpec(
    domain="airline",
    intent="book_flight",
    slots={"from": "HYD", "to": "BLR", "when": "2026-04-25"},
    constraints={"budget_inr": 8000, "time_window": "evening"},
    language="hinglish",
    seed_utterance="Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad",
)

obs = DriftCallObservation(
    turn=0,
    goal=goal,
    last_transcript="",
    last_lang="",
    last_confidence=1.0,
    tool_results=(),
    drift_log=(),
    budget_remaining=12,      # Stage 2: max_turns = 12
    available_tools=(
        "airline.search",
        "airline.book",
        "airline.cancel",
        "airline.get_booking",
        "payment.charge",
    ),
)

assert obs.turn == 0
assert obs.goal.language == "hinglish"
assert len(obs.tool_results) == 0
```

### 8.4 Constructing a `DriftEvent` and appending it to state via `replace`

```python
from dataclasses import replace

from driftcall.models import DriftCallState, DriftEvent

drift = DriftEvent(
    turn=4,
    drift_type="schema",
    domain="airline",
    description="field 'price' renamed to 'total_fare_inr'; 'currency' removed",
    from_version="v1",
    to_version="v2",
)

new_state = replace(
    state,
    drift_fired=state.drift_fired + (drift,),
    schema_versions={**state.schema_versions, "airline": "v2"},
)

assert drift in new_state.drift_fired
assert new_state.schema_versions["airline"] == "v2"
# Original state untouched:
assert drift not in state.drift_fired
```

---

## 9. Open questions

None — spec is complete. Every dataclass, field, type, and invariant is locked against DESIGN.md §4.1. No ambiguity remains that would block downstream modules (`env.md`, `rewards.md`, `vendors.md`, `drift_injector.md`, `task_generator.md`) from referencing this doc as a stable contract.
