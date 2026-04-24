"""Cell 11 — DriftCallEnv smoke episode.

End-to-end smoke test that boots ``DriftCallEnv`` (cell 10) with a Stage-1
English airline configuration, runs one short episode, and prints the
resulting reward breakdown. Mirrors ``DESIGN.md`` §16.A.2 and
``docs/modules/env.md`` §8.1.

The cell exposes two public callables:

* :func:`run_smoke_episode` — pure helper that returns a :class:`SmokeResult`
  containing the (terminated) env, observation, and rewards. Useful from
  tests.
* :func:`main` — notebook-cell entry point; prints a small summary table to
  stdout and returns the same :class:`SmokeResult`.

The cell never imports ``torch``, audio engines, or any LLM stack — it is
text-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cells.step_04_models import (
    ActionType,
    DriftCallAction,
    DriftCallObservation,
)
from cells.step_10_env import DriftCallEnv

if TYPE_CHECKING:  # pragma: no cover — typing only
    from cells.step_08_rewards import Rewards


SMOKE_SEED: int = 42
SMOKE_CONFIDENCE: float = 0.8


@dataclass(frozen=True)
class SmokeResult:
    """Container returned by :func:`run_smoke_episode`."""

    env: DriftCallEnv
    final_observation: DriftCallObservation
    rewards: Rewards


def _build_env() -> DriftCallEnv:
    """Construct the canonical Stage-1, English-only, no-audio env."""
    return DriftCallEnv(
        config={
            "curriculum_stage": 1,
            "language_weights": {"en": 1.0},
            "audio_boundary_enabled": False,
        },
    )


def _pick_search_tool(obs: DriftCallObservation) -> str:
    """Return the first ``<domain>.search``-style tool exposed for the goal."""
    domain = obs.goal.domain
    for tool in obs.available_tools:
        if tool == f"{domain}.search":
            return tool
    # Fall back to any tool in the domain if no explicit search action exists.
    for tool in obs.available_tools:
        if tool.startswith(f"{domain}."):
            return tool
    raise RuntimeError(f"no tools available for domain {domain!r}")


def _pick_book_tool(obs: DriftCallObservation) -> str | None:
    """Return the first ``<domain>.book``/``<domain>.order``/etc. tool, if any."""
    domain = obs.goal.domain
    for verb in ("book", "order", "reserve", "create"):
        candidate = f"{domain}.{verb}"
        if candidate in obs.available_tools:
            return candidate
    return None


def run_smoke_episode(seed: int = SMOKE_SEED) -> SmokeResult:
    """Run a single Stage-1 airline-style episode and return the rewards.

    Action sequence:

    1. ``TOOL_CALL`` to the domain's ``search`` endpoint (no args — vendors
       are tolerant of empty args at v1).
    2. ``TOOL_CALL`` to the domain's ``book``/``order`` endpoint, if exposed.
    3. ``SUBMIT`` with ``confidence=0.8``.
    """
    env = _build_env()
    obs = env.reset(seed=seed)

    obs = env.step(
        DriftCallAction(
            action_type=ActionType.TOOL_CALL,
            tool_name=_pick_search_tool(obs),
            tool_args={},
            rationale="smoke: discover candidates",
        ),
    )

    book_tool = _pick_book_tool(obs)
    if book_tool is not None and not env.done():
        obs = env.step(
            DriftCallAction(
                action_type=ActionType.TOOL_CALL,
                tool_name=book_tool,
                tool_args={},
                rationale="smoke: commit booking",
            ),
        )

    if not env.done():
        obs = env.step(
            DriftCallAction(
                action_type=ActionType.SUBMIT,
                confidence=SMOKE_CONFIDENCE,
                message="smoke episode complete",
                rationale="smoke: terminate",
            ),
        )

    rewards = env.rewards()
    return SmokeResult(env=env, final_observation=obs, rewards=rewards)


def _format_summary(result: SmokeResult) -> str:
    r = result.rewards
    ep = result.env.episode()
    lines = [
        "=== DriftCall smoke episode ===",
        f"  episode_id    : {ep.episode_id}",
        f"  domain        : {ep.goal.domain}",
        f"  language      : {ep.goal.language}",
        f"  terminated_by : {ep.terminated_by}",
        f"  turns_used    : {ep.turns_used} / {ep.max_turns}",
        "  --- rewards ---",
        f"  r1 (task)        : {r.r1:.3f}",
        f"  r2 (drift)       : {r.r2:.3f}",
        f"  r3 (constraints) : {r.r3:.3f}",
        f"  r4 (format)      : {r.r4:.3f}",
        f"  r5 (anti-hack)   : {r.r5:.3f}",
        f"  reward (final)   : {r.reward:.3f}",
    ]
    return "\n".join(lines)


def main() -> SmokeResult:
    """Run the smoke episode and print a summary table to stdout."""
    result = run_smoke_episode()
    print(_format_summary(result))
    return result


__all__ = [
    "SMOKE_CONFIDENCE",
    "SMOKE_SEED",
    "SmokeResult",
    "main",
    "run_smoke_episode",
]
