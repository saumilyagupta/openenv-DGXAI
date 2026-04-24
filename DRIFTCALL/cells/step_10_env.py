"""Cell 10 — DriftCallEnv integration class.

Implements ``docs/modules/env.md`` and DESIGN.md §4. ``DriftCallEnv`` is the
single public surface that composes models, vendors, drift_injector,
task_generator, rewards, and the optional audio boundary into an
OpenEnv-compliant RL environment.

Hard rules (env.md §3.8, CLAUDE.md §0):
- All public dataclasses are frozen.
- State transitions go through ``dataclasses.replace``; no in-place mutation.
- Validation is pure: ``InvalidActionError`` raises BEFORE any state mutation.
- Rewards are computed exactly once at termination and memoized.
- No LLM judge anywhere; no network/disk I/O at ``__init__``.
"""

from __future__ import annotations

import os
import struct
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftCallObservation,
    DriftCallState,
    DriftEvent,
    GoalSpec,
    ToolResult,
)
from cells.step_05_vendors import TOOLS as VENDOR_TOOLS
from cells.step_05_vendors import VENDOR_REGISTRY
from cells.step_06_drift_injector import (
    DriftCatalogueError,
    DriftDomainMismatchError,
    DriftReapplicationError,
    DriftScheduleConflictError,
    UnknownDriftPatternError,
    apply_drift,
    build_schedule,
    list_patterns,
)
from cells.step_07_task_generator import (
    InvalidLanguageWeightError,
    InvalidStageError,
)
from cells.step_07_task_generator import (
    generate as task_generate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# rewards is imported lazily inside _compute_rewards to keep the env importable
# even before step_08_rewards.py lands; failures surface as RewardComputationError.

_DEFAULT_LANGUAGE_WEIGHTS: dict[str, float] = {
    "en": 0.4,
    "hinglish": 0.4,
    "hi": 0.1,
    "ta": 0.05,
    "kn": 0.05,
}

_LANGUAGE_CODES: frozenset[str] = frozenset({"hi", "ta", "kn", "en", "hinglish"})

_STAGE_MAX_TURNS: dict[int, int] = {1: 8, 2: 12, 3: 16}

_VENDOR_DOMAINS: tuple[str, ...] = ("airline", "cab", "restaurant", "hotel", "payment")

_TERMINATED_VALUES: frozenset[str] = frozenset({"SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"})

_NOW_IST: datetime = datetime(2026, 4, 25, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


# ---------------------------------------------------------------------------
# Error taxonomy (env.md §5)
# ---------------------------------------------------------------------------


class DriftCallEnvError(Exception):
    """Root for every typed env error (env.md §5)."""


class InvalidConfigError(DriftCallEnvError):
    """E1 — malformed config dict."""


class EnvNotReadyError(DriftCallEnvError):
    """E2 — operation issued before reset()."""


class EnvClosedError(DriftCallEnvError):
    """E3 — operation issued after close()."""


class InvalidActionError(DriftCallEnvError):
    """E4 — action fails the per-ActionType field matrix."""


class EpisodeAlreadyTerminalError(DriftCallEnvError):
    """E5 — step() called after termination."""


class EpisodeNotTerminalError(DriftCallEnvError):
    """E6 — episode()/rewards() called before termination."""


class ConcurrentStepError(DriftCallEnvError):
    """E7 — reentrant step() detected."""


class UnknownDomainError(DriftCallEnvError):
    """E8 — PROBE_SCHEMA on a domain that is not registered."""


class UnknownToolError(DriftCallEnvError):
    """E9 — TOOL_CALL with a tool_name not in available_tools()."""


class DriftInjectionError(DriftCallEnvError):
    """E10 — drift fold raised; surfaced as-is."""


class RewardComputationError(DriftCallEnvError):
    """E11 — compute_rewards raised; surfaced as-is."""


class AudioPipelineError(DriftCallEnvError):
    """E12 — TTS/ASR engine raised on a step()/reset() boundary."""


_ALL_ERROR_CLASSES: tuple[type[DriftCallEnvError], ...] = (
    InvalidConfigError,
    EnvNotReadyError,
    EnvClosedError,
    InvalidActionError,
    EpisodeAlreadyTerminalError,
    EpisodeNotTerminalError,
    ConcurrentStepError,
    UnknownDomainError,
    UnknownToolError,
    DriftInjectionError,
    RewardComputationError,
    AudioPipelineError,
)


# ---------------------------------------------------------------------------
# Protocols (env.md §2.1)
# ---------------------------------------------------------------------------


class DriftScheduler(Protocol):
    def __call__(
        self, stage: int, episode_seed: int, goal: GoalSpec
    ) -> tuple[DriftEvent, ...]: ...


class TTSEngine(Protocol):
    def synthesize(
        self,
        text: str,
        language_code: str,
        voice_pack: Any | None = None,
        *,
        seed: int = 0,
        sample_rate_hz: int = 16000,
    ) -> bytes: ...


class ASREngine(Protocol):
    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: str | None,
        *,
        beam_size: int = 1,
        vad_filter: bool = True,
        max_duration_s: float = 30.0,
    ) -> Any: ...


def _default_scheduler(
    stage: int, episode_seed: int, goal: GoalSpec
) -> tuple[DriftEvent, ...]:
    return build_schedule(stage, episode_seed, goal)


# ---------------------------------------------------------------------------
# Episode (env.md §4.3) — built at termination, fed to rewards.compute_rewards.
# Matches the Episode shape consumed by step_08_rewards (kw fields).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    episode_id: str
    goal: GoalSpec
    actions: tuple[DriftCallAction, ...]
    action_turns: tuple[int, ...]
    tool_results: tuple[ToolResult, ...]
    tool_result_turns: tuple[int, ...]
    drift_log: tuple[DriftEvent, ...]
    vendor_states_final: dict[str, dict[str, Any]]
    schema_versions_final: dict[str, str]
    max_turns: int
    turns_used: int
    terminated_by: Literal["SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"]
    stage: Literal[1, 2, 3]
    drift_pattern_overrides: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnvConfig (env.md §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    curriculum_stage: Literal[1, 2, 3]
    language_weights: dict[str, float]
    audio_boundary_enabled: bool
    max_turns_override: int | None
    scheduler: DriftScheduler
    tts_engine: TTSEngine | None
    asr_engine: ASREngine | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> EnvConfig:
        allowed = {
            "curriculum_stage",
            "language_weights",
            "audio_boundary_enabled",
            "max_turns_override",
            "scheduler",
            "tts_engine",
            "asr_engine",
        }
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise InvalidConfigError(
                f"config must be a dict or None, got {type(raw).__name__}"
            )

        unknown = set(raw.keys()) - allowed
        if unknown:
            raise InvalidConfigError(
                f"unknown config key(s): {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}"
            )

        stage_raw = raw.get("curriculum_stage", 1)
        if isinstance(stage_raw, bool) or not isinstance(stage_raw, int):
            raise InvalidConfigError(
                f"curriculum_stage must be int in {{1,2,3}}, got "
                f"{type(stage_raw).__name__}"
            )
        if stage_raw not in (1, 2, 3):
            raise InvalidConfigError(
                f"curriculum_stage must be 1, 2, or 3; got {stage_raw!r}"
            )
        stage = cast("Literal[1, 2, 3]", stage_raw)

        weights_raw = raw.get("language_weights", _DEFAULT_LANGUAGE_WEIGHTS)
        if not isinstance(weights_raw, dict) or not weights_raw:
            raise InvalidConfigError(
                "language_weights must be a non-empty dict"
            )
        for k, v in weights_raw.items():
            if k not in _LANGUAGE_CODES:
                raise InvalidConfigError(
                    f"language_weights: unknown language {k!r}; "
                    f"allowed: {sorted(_LANGUAGE_CODES)}"
                )
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidConfigError(
                    f"language_weights[{k!r}] must be numeric, got "
                    f"{type(v).__name__}"
                )
            if v < 0:
                raise InvalidConfigError(
                    f"language_weights[{k!r}]={v} is negative"
                )
        total = sum(float(v) for v in weights_raw.values())
        if abs(total - 1.0) > 1e-6:
            raise InvalidConfigError(
                f"language_weights sum {total!r} not within 1.0 ± 1e-6"
            )
        # Frozen copy.
        weights: dict[str, float] = {k: float(v) for k, v in weights_raw.items()}

        audio_enabled_raw = raw.get("audio_boundary_enabled", False)
        if not isinstance(audio_enabled_raw, bool):
            raise InvalidConfigError(
                f"audio_boundary_enabled must be bool, got "
                f"{type(audio_enabled_raw).__name__}"
            )
        audio_enabled = audio_enabled_raw

        max_turns_override = raw.get("max_turns_override")
        if max_turns_override is not None:
            if isinstance(max_turns_override, bool) or not isinstance(
                max_turns_override, int
            ):
                raise InvalidConfigError(
                    f"max_turns_override must be int or None, got "
                    f"{type(max_turns_override).__name__}"
                )
            if max_turns_override < 1:
                raise InvalidConfigError(
                    f"max_turns_override must be >= 1, got {max_turns_override}"
                )

        scheduler = raw.get("scheduler", _default_scheduler)
        if not callable(scheduler):
            raise InvalidConfigError("scheduler must be callable")

        tts_engine = raw.get("tts_engine")
        asr_engine = raw.get("asr_engine")

        if audio_enabled:
            if tts_engine is None:
                raise InvalidConfigError(
                    "tts_engine is required when audio_boundary_enabled is True"
                )
            if asr_engine is None:
                raise InvalidConfigError(
                    "asr_engine is required when audio_boundary_enabled is True"
                )
        else:
            if tts_engine is not None:
                raise InvalidConfigError(
                    "tts_engine must be None when audio_boundary_enabled is False"
                )
            if asr_engine is not None:
                raise InvalidConfigError(
                    "asr_engine must be None when audio_boundary_enabled is False"
                )

        return cls(
            curriculum_stage=stage,
            language_weights=weights,
            audio_boundary_enabled=audio_enabled,
            max_turns_override=max_turns_override,
            scheduler=cast("DriftScheduler", scheduler),
            tts_engine=cast("TTSEngine | None", tts_engine),
            asr_engine=cast("ASREngine | None", asr_engine),
        )


# ---------------------------------------------------------------------------
# DriftCallEnv
# ---------------------------------------------------------------------------


def _make_seed_from_urandom() -> int:
    raw = os.urandom(8)
    (value,) = struct.unpack("<Q", raw)
    return int(value)


def _vendor_state_to_dict(state: Any) -> dict[str, Any]:
    """Coerce a frozen vendor dataclass (or already-dict) into a plain dict."""
    if isinstance(state, dict):
        return dict(state)
    # All vendor states are frozen dataclasses.
    import dataclasses as _dc

    if _dc.is_dataclass(state) and not isinstance(state, type):
        return _dc.asdict(state)
    # Defensive: best-effort fallback.
    return {"_raw": repr(state)}


class DriftCallEnv:
    """OpenEnv-compliant RL environment for DriftCall (env.md §1)."""

    # -- construction --------------------------------------------------------

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: EnvConfig = EnvConfig.from_mapping(config)
        self._state: DriftCallState | None = None
        self._rewards: Any | None = None
        self._episode: Episode | None = None
        self._closed: bool = False
        self._seed: int | None = None
        self._episode_id: str | None = None
        # Pending side-channel notices keyed by domain (env.md §3.3).
        self._side_channel_pending: dict[str, str] = {}
        # Per-vendor-state cache (frozen dataclass or dict). Kept on the env
        # because DriftCallState.vendor_states is a dict[str, dict] for
        # compatibility with the design dataclass.
        self._vendor_state_objects: dict[str, Any] = {}
        # Re-entrancy guard (E7).
        self._step_in_progress: bool = False

    # -- internal helpers ----------------------------------------------------

    @property
    def _max_turns(self) -> int:
        if self._config.max_turns_override is not None:
            return int(self._config.max_turns_override)
        return _STAGE_MAX_TURNS[self._config.curriculum_stage]

    def _available_tools(self) -> tuple[str, ...]:
        return VENDOR_TOOLS

    def _ensure_ready_for_step(self) -> None:
        if self._closed:
            raise EnvClosedError("env is closed")
        if self._state is None:
            raise EnvNotReadyError("reset() must be called before step()")
        if self._state.done:
            raise EpisodeAlreadyTerminalError(
                f"episode already terminated (terminated_by={self._terminated_by()})"
            )

    def _terminated_by(self) -> str | None:
        return self._episode.terminated_by if self._episode is not None else None

    # -- OpenEnv primitives --------------------------------------------------

    def reset(self, seed: int | None = None) -> DriftCallObservation:
        if self._closed:
            raise EnvClosedError("env is closed")

        if seed is None:
            seed = _make_seed_from_urandom()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidActionError(
                f"seed must be int or None, got {type(seed).__name__}"
            )

        self._seed = int(seed)
        # Reset memoization; legacy state is dropped before any propagatable
        # exception can leak (env.md §2.2 docstring).
        self._state = None
        self._rewards = None
        self._episode = None
        self._side_channel_pending = {}
        self._vendor_state_objects = {}
        self._episode_id = None

        try:
            goal = task_generate(
                self._seed,
                self._config.curriculum_stage,
                cast("dict[Any, float]", self._config.language_weights),
            )
        except (InvalidLanguageWeightError, InvalidStageError) as exc:
            # E1-class reset failure (env.md §2.2 raises clause).
            raise InvalidConfigError(str(exc)) from exc

        # Initial per-domain vendor state objects (frozen dataclasses).
        vendor_state_objects: dict[str, Any] = {}
        vendor_states_dict: dict[str, dict[str, Any]] = {}
        for domain in _VENDOR_DOMAINS:
            ns = VENDOR_REGISTRY[domain]
            vs = ns.initial_state(self._seed, goal)
            vendor_state_objects[domain] = vs
            vendor_states_dict[domain] = _vendor_state_to_dict(vs)

        schema_versions = {d: "v1" for d in _VENDOR_DOMAINS}

        try:
            schedule = self._config.scheduler(
                self._config.curriculum_stage, self._seed, goal
            )
        except (
            DriftScheduleConflictError,
            DriftCatalogueError,
            UnknownDriftPatternError,
            DriftDomainMismatchError,
        ) as exc:
            # Bad scheduler at reset is an E1 (env.md §7.4).
            raise InvalidConfigError(f"scheduler failure: {exc}") from exc

        self._episode_id = uuid.uuid4().hex

        max_turns = self._max_turns
        new_state = DriftCallState(
            episode_id=self._episode_id,
            goal=goal,
            vendor_states=vendor_states_dict,
            schema_versions=schema_versions,
            drift_schedule=tuple(schedule),
            drift_fired=(),
            turn=0,
            max_turns=max_turns,
            actions=(),
            done=False,
        )
        self._state = new_state
        self._vendor_state_objects = vendor_state_objects

        if self._config.audio_boundary_enabled:
            tts = self._config.tts_engine
            assert tts is not None  # validated in EnvConfig
            try:
                tts.synthesize(goal.seed_utterance, goal.language)
            except Exception as exc:  # noqa: BLE001 — surface as E12-class
                # Audio failure on reset leaves env unready (env.md §5 E12).
                self._state = None
                self._vendor_state_objects = {}
                self._episode_id = None
                raise AudioPipelineError(f"TTS reset failure: {exc}") from exc

        return self._build_observation()

    def step(
        self,
        action: DriftCallAction,
        *,
        force_drift_pattern: str | None = None,
    ) -> DriftCallObservation:
        # 1a. Pure validation — must raise before any state mutation.
        self._ensure_ready_for_step()
        self._validate_action(action)
        if force_drift_pattern is not None:
            valid_ids = {p.id for p in list_patterns()}
            if force_drift_pattern not in valid_ids:
                raise InvalidActionError(
                    f"force_drift_pattern {force_drift_pattern!r} not a known "
                    f"pattern_id"
                )

        if self._step_in_progress:
            raise ConcurrentStepError("reentrant step() detected")
        self._step_in_progress = True
        try:
            return self._step_inner(action, force_drift_pattern)
        finally:
            self._step_in_progress = False

    def _step_inner(
        self,
        action: DriftCallAction,
        force_drift_pattern: str | None,
    ) -> DriftCallObservation:
        assert self._state is not None  # ensured above
        # 2. Increment turn counter.
        turn_current = self._state.turn + 1
        self._state = replace(self._state, turn=turn_current)

        # 3. Fire drifts for this turn.
        self._fire_drifts(turn_current, force_drift_pattern)

        # 4. Side-channel emit pass — refresh pending notices for any vendor
        # whose state just mutated.
        self._emit_side_channel()

        # 5. Dispatch action.
        new_tool_result, terminate, terminated_by = self._dispatch(action)

        # 6. Record action (and ToolResult, if any) via dataclasses.replace.
        new_actions = self._state.actions + (action,)
        if new_tool_result is not None:
            # Tool result history lives on the state's vendor history; here we
            # rely on the running observation history we will rebuild in §3.4.
            self._tool_results = self._tool_results + (new_tool_result,)
            self._tool_result_turns = self._tool_result_turns + (turn_current,)
        self._action_turns = self._action_turns + (turn_current,)
        self._state = replace(self._state, actions=new_actions)

        # 7. Budget check — only if action did not already terminate.
        if not terminate and turn_current >= self._state.max_turns:
            terminate = True
            terminated_by = "TIMEOUT"

        # 8. If terminal, build Episode + compute rewards.
        if terminate:
            assert terminated_by is not None
            self._terminate(terminated_by)

        # 9. Build observation.
        return self._build_observation()

    def state(self) -> DriftCallState:
        if self._state is None:
            raise EnvNotReadyError("reset() must be called before state()")
        return self._state

    def close(self) -> None:
        # Idempotent.
        self._closed = True
        # Per env.md §9 Q7: never invoke close on shared audio engines.
        # Only drop per-env state.
        self._side_channel_pending = {}
        self._vendor_state_objects = {}
        # Note: we keep self._state, self._rewards, self._episode so post-close
        # audits still work (env.md §7.11).

    def episode(self) -> Episode:
        if self._episode is None:
            raise EpisodeNotTerminalError("episode is not terminal")
        return self._episode

    def rewards(self) -> Any:
        if self._rewards is None:
            raise EpisodeNotTerminalError("episode is not terminal")
        return self._rewards

    def done(self) -> bool:
        if self._state is None:
            return False
        return bool(self._state.done)

    # -- validation ----------------------------------------------------------

    def _validate_action(self, action: DriftCallAction) -> None:
        if not isinstance(action, DriftCallAction):
            raise InvalidActionError(
                f"action must be DriftCallAction, got {type(action).__name__}"
            )
        atype = action.action_type
        if not isinstance(atype, ActionType):
            raise InvalidActionError(
                f"action_type must be ActionType, got {type(atype).__name__}"
            )

        # rationale length cap (env.md §3.1).
        if action.rationale is not None and len(action.rationale) > 200:
            raise InvalidActionError(
                f"rationale length {len(action.rationale)} exceeds 200"
            )

        if atype == ActionType.TOOL_CALL:
            if not action.tool_name or not isinstance(action.tool_name, str):
                raise InvalidActionError("TOOL_CALL requires non-empty tool_name")
            if action.tool_args is None or not isinstance(action.tool_args, dict):
                raise InvalidActionError(
                    "TOOL_CALL requires tool_args dict (may be empty)"
                )
            if action.message is not None or action.confidence is not None:
                raise InvalidActionError(
                    "TOOL_CALL forbids message/confidence"
                )
            if action.tool_name not in self._available_tools():
                raise UnknownToolError(
                    f"tool_name {action.tool_name!r} not in available_tools()"
                )
            # JSON-serializability (shallow check: must be dict; values arbitrary).
            return

        if atype == ActionType.SPEAK or atype == ActionType.CLARIFY:
            if not isinstance(action.message, str):
                raise InvalidActionError(
                    f"{atype.value} requires str message"
                )
            if not (1 <= len(action.message) <= 2000):
                raise InvalidActionError(
                    f"{atype.value} message length must be in [1, 2000], "
                    f"got {len(action.message)}"
                )
            if "\x00" in action.message:
                raise InvalidActionError(
                    f"{atype.value} message contains NUL byte"
                )
            if (
                action.tool_name is not None
                or action.tool_args is not None
                or action.confidence is not None
            ):
                raise InvalidActionError(
                    f"{atype.value} forbids tool_name/tool_args/confidence"
                )
            return

        if atype == ActionType.PROBE_SCHEMA:
            if not action.tool_name or not isinstance(action.tool_name, str):
                raise InvalidActionError(
                    "PROBE_SCHEMA requires tool_name (domain string)"
                )
            if (
                action.tool_args is not None
                or action.message is not None
                or action.confidence is not None
            ):
                raise InvalidActionError(
                    "PROBE_SCHEMA forbids tool_args/message/confidence"
                )
            assert self._state is not None
            if action.tool_name not in self._state.vendor_states:
                raise UnknownDomainError(
                    f"PROBE_SCHEMA: domain {action.tool_name!r} not registered"
                )
            return

        if atype == ActionType.SUBMIT:
            if action.confidence is None or not isinstance(
                action.confidence, (int, float)
            ) or isinstance(action.confidence, bool):
                raise InvalidActionError("SUBMIT requires float confidence")
            conf = float(action.confidence)
            if not (0.0 <= conf <= 1.0):
                raise InvalidActionError(
                    f"SUBMIT confidence {conf!r} outside [0.0, 1.0]"
                )
            if action.tool_name is not None or action.tool_args is not None:
                raise InvalidActionError(
                    "SUBMIT forbids tool_name/tool_args"
                )
            if action.message is not None and not isinstance(action.message, str):
                raise InvalidActionError("SUBMIT message must be str if present")
            return

        if atype == ActionType.ABORT:
            if (
                action.tool_name is not None
                or action.tool_args is not None
                or action.confidence is not None
            ):
                raise InvalidActionError(
                    "ABORT forbids tool_name/tool_args/confidence"
                )
            return

        # Unreachable — all six ActionType members handled above.
        raise InvalidActionError(f"unhandled action_type {atype!r}")

    # -- drift firing --------------------------------------------------------

    def _fire_drifts(self, turn_current: int, force_pattern: str | None) -> None:
        assert self._state is not None
        if force_pattern is not None:
            patterns_by_id = {p.id: p for p in list_patterns()}
            pattern = patterns_by_id[force_pattern]
            if pattern.domain not in self._state.vendor_states:
                raise DriftInjectionError(
                    f"force_drift_pattern {force_pattern!r}: domain "
                    f"{pattern.domain!r} not registered"
                )
            event = DriftEvent(
                turn=turn_current,
                drift_type=pattern.drift_type,
                domain=pattern.domain,
                description=pattern.description,
                from_version=pattern.from_version,
                to_version=pattern.to_version,
                pattern_id=pattern.id,
            )
            try:
                self._state = apply_drift(self._state, event)
            except (
                UnknownDriftPatternError,
                DriftDomainMismatchError,
                DriftReapplicationError,
            ) as exc:
                raise DriftInjectionError(str(exc)) from exc
            return

        # Schedule-driven fold.
        pending = tuple(
            e for e in self._state.drift_schedule
            if e.turn == turn_current and e not in self._state.drift_fired
        )
        if not pending:
            return
        ordered = tuple(sorted(pending, key=lambda e: (e.turn, e.pattern_id)))
        for event in ordered:
            try:
                self._state = apply_drift(self._state, event)
            except (
                UnknownDriftPatternError,
                DriftDomainMismatchError,
                DriftReapplicationError,
            ) as exc:
                raise DriftInjectionError(str(exc)) from exc

    def _emit_side_channel(self) -> None:
        """Refresh pending side-channel notices per env.md §3.3 clause 3."""
        assert self._state is not None
        new_pending = dict(self._side_channel_pending)
        for domain in _VENDOR_DOMAINS:
            ns = VENDOR_REGISTRY[domain]
            vs_obj = self._vendor_state_objects.get(domain)
            if vs_obj is None:
                continue
            try:
                notice, new_state = ns.emit_side_channel_if_pending(vs_obj)
            except Exception as exc:  # noqa: BLE001 — defensive
                raise DriftInjectionError(
                    f"side-channel emit failed for {domain}: {exc}"
                ) from exc
            if notice is not None:
                existing = new_pending.get(domain)
                merged = (
                    f"{existing}\n---\n{notice}" if existing else notice
                )
                new_pending[domain] = merged
            self._vendor_state_objects[domain] = new_state
        self._side_channel_pending = new_pending

    # -- dispatch ------------------------------------------------------------

    @property
    def _tool_results(self) -> tuple[ToolResult, ...]:
        return getattr(self, "_tool_results_internal", ())

    @_tool_results.setter
    def _tool_results(self, value: tuple[ToolResult, ...]) -> None:
        self._tool_results_internal = value

    @property
    def _tool_result_turns(self) -> tuple[int, ...]:
        return getattr(self, "_tool_result_turns_internal", ())

    @_tool_result_turns.setter
    def _tool_result_turns(self, value: tuple[int, ...]) -> None:
        self._tool_result_turns_internal = value

    @property
    def _action_turns(self) -> tuple[int, ...]:
        return getattr(self, "_action_turns_internal", ())

    @_action_turns.setter
    def _action_turns(self, value: tuple[int, ...]) -> None:
        self._action_turns_internal = value

    def _dispatch(
        self, action: DriftCallAction
    ) -> tuple[ToolResult | None, bool, str | None]:
        """Return (tool_result, terminate?, terminated_by?)."""
        assert self._state is not None
        atype = action.action_type

        if atype == ActionType.SUBMIT:
            return None, True, "SUBMIT"
        if atype == ActionType.ABORT:
            return None, True, "ABORT"
        if atype == ActionType.SPEAK or atype == ActionType.CLARIFY:
            return None, False, None

        if atype == ActionType.PROBE_SCHEMA:
            assert action.tool_name is not None
            domain = action.tool_name
            ns = VENDOR_REGISTRY[domain]
            vs_obj = self._vendor_state_objects[domain]
            schema_version = self._state.schema_versions[domain]
            schema = ns.describe_schema(vs_obj, schema_version)
            tr = ToolResult(
                tool_name=f"probe:{domain}",
                status="ok",
                response=dict(schema),
                schema_version=schema_version,
                latency_ms=0,
            )
            return tr, False, None

        if atype == ActionType.TOOL_CALL:
            assert action.tool_name is not None and action.tool_args is not None
            tool_name = action.tool_name
            domain = tool_name.split(".", 1)[0]
            if domain not in self._state.vendor_states:
                raise UnknownDomainError(
                    f"tool {tool_name!r} targets unknown domain {domain!r}"
                )
            ns = VENDOR_REGISTRY[domain]
            vs_obj = self._vendor_state_objects[domain]
            schema_version = self._state.schema_versions[domain]
            try:
                if domain == "payment":
                    tr, new_vs = ns.dispatch(
                        tool_name,
                        action.tool_args,
                        vs_obj,
                        schema_version,
                        self._seed,
                        _NOW_IST,
                    )
                    payment_state = new_vs
                else:
                    payment_state = self._vendor_state_objects.get("payment")
                    tr, new_vs, payment_state = ns.dispatch(
                        tool_name,
                        action.tool_args,
                        vs_obj,
                        schema_version,
                        self._seed,
                        _NOW_IST,
                        payment_state,
                    )
            except ValueError as exc:
                # Unknown tool inside a known domain → treat as anti-hack.
                raise UnknownToolError(str(exc)) from exc

            self._vendor_state_objects[domain] = new_vs
            if payment_state is not None:
                self._vendor_state_objects["payment"] = payment_state

            # Refresh state.vendor_states snapshot.
            new_vendor_states = dict(self._state.vendor_states)
            new_vendor_states[domain] = _vendor_state_to_dict(new_vs)
            if domain != "payment" and payment_state is not None:
                new_vendor_states["payment"] = _vendor_state_to_dict(payment_state)
            self._state = replace(self._state, vendor_states=new_vendor_states)

            # Attach pending side-channel notice (one-shot per domain).
            notice = self._side_channel_pending.pop(domain, None)
            if notice is not None:
                merged_response = dict(tr.response)
                merged_response["_notice"] = notice
                tr = ToolResult(
                    tool_name=tr.tool_name,
                    status=tr.status,
                    response=merged_response,
                    schema_version=tr.schema_version,
                    latency_ms=tr.latency_ms,
                )
            return tr, False, None

        # Unreachable.
        raise InvalidActionError(f"unhandled action_type {atype!r}")

    # -- termination ---------------------------------------------------------

    def _terminate(self, terminated_by: str) -> None:
        assert self._state is not None
        if terminated_by not in _TERMINATED_VALUES:
            raise RewardComputationError(
                f"unknown terminated_by sentinel {terminated_by!r}"
            )
        self._state = replace(self._state, done=True)
        episode = Episode(
            episode_id=self._state.episode_id,
            goal=self._state.goal,
            actions=self._state.actions,
            action_turns=self._action_turns,
            tool_results=self._tool_results,
            tool_result_turns=self._tool_result_turns,
            drift_log=self._state.drift_fired,
            vendor_states_final={
                d: _vendor_state_to_dict(self._vendor_state_objects[d])
                for d in _VENDOR_DOMAINS
            },
            schema_versions_final=dict(self._state.schema_versions),
            max_turns=self._state.max_turns,
            turns_used=len(self._state.actions),
            terminated_by=cast(
                "Literal['SUBMIT','ABORT','TIMEOUT','ANTI_HACK']", terminated_by
            ),
            stage=self._config.curriculum_stage,
        )
        self._episode = episode
        self._rewards = self._compute_rewards(episode)

    @staticmethod
    def _compute_rewards(episode: Episode) -> Any:
        import importlib

        try:
            mod = importlib.import_module("cells.step_08_rewards")
        except ImportError as exc:
            raise RewardComputationError(
                f"rewards module unavailable: {exc}"
            ) from exc
        compute = getattr(mod, "compute_rewards", None)
        if compute is None:
            raise RewardComputationError(
                "cells.step_08_rewards has no compute_rewards"
            )
        try:
            return compute(episode)
        except Exception as exc:
            raise RewardComputationError(str(exc)) from exc

    # -- observation builder -------------------------------------------------

    def _build_observation(self) -> DriftCallObservation:
        assert self._state is not None
        st = self._state
        if st.turn == 0:
            last_transcript = st.goal.seed_utterance
            last_lang = st.goal.language
            last_confidence = 1.0
        else:
            last_transcript = st.goal.seed_utterance
            last_lang = st.goal.language
            last_confidence = 1.0

        return DriftCallObservation(
            turn=st.turn,
            goal=st.goal,
            last_transcript=last_transcript,
            last_lang=last_lang,
            last_confidence=last_confidence,
            tool_results=self._tool_results,
            drift_log=st.drift_fired,
            budget_remaining=max(0, st.max_turns - st.turn),
            available_tools=self._available_tools(),
        )


__all__ = [
    "ASREngine",
    "AudioPipelineError",
    "ConcurrentStepError",
    "DriftCallEnv",
    "DriftCallEnvError",
    "DriftInjectionError",
    "DriftScheduler",
    "EnvClosedError",
    "EnvConfig",
    "EnvNotReadyError",
    "Episode",
    "EpisodeAlreadyTerminalError",
    "EpisodeNotTerminalError",
    "InvalidActionError",
    "InvalidConfigError",
    "RewardComputationError",
    "TTSEngine",
    "UnknownDomainError",
    "UnknownToolError",
]
