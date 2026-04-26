"""Tests for cells/step_11_smoke_env.py.

The smoke cell boots ``DriftCallEnv`` with a Stage-1 airline configuration,
runs one episode (search -> book -> submit), computes rewards, and prints a
human-readable summary table to stdout. These tests verify:

* the cell module imports cleanly,
* its public helpers (``run_smoke_episode`` and ``main``) have the expected
  shape,
* a real Stage-1 episode boots, advances, terminates, and produces in-bounds
  rewards,
* every reward field falls within its documented interval (R1..R4 in
  ``[0, 1]``, R5 in ``[-1, 0]``, ``reward`` in ``[0, 1]``),
* running the cell as a script exits 0 and emits the expected stdout markers.

If ``cells.step_08_rewards`` or ``cells.step_10_env`` have not yet landed
(parallel coders), the integration tests skip gracefully — the smoke cell is
still importable and statically inspectable on its own.
"""

from __future__ import annotations

import importlib
import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_CELL_PATH = REPO_ROOT / "cells" / "step_11_smoke_env.py"
SMOKE_CELL_MD_PATH = REPO_ROOT / "cells" / "step_11_smoke_env.md"


# ---------------------------------------------------------------------------
# Module-level invariants (no DriftCallEnv needed)
# ---------------------------------------------------------------------------


def test_smoke_cell_file_exists() -> None:
    assert SMOKE_CELL_PATH.is_file(), f"missing cell at {SMOKE_CELL_PATH}"


def test_smoke_cell_markdown_preamble_exists() -> None:
    assert SMOKE_CELL_MD_PATH.is_file(), f"missing markdown preamble at {SMOKE_CELL_MD_PATH}"
    text = SMOKE_CELL_MD_PATH.read_text(encoding="utf-8")
    assert text.strip(), "markdown preamble is empty"


def test_smoke_cell_uses_future_annotations() -> None:
    text = SMOKE_CELL_PATH.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in text


def test_smoke_cell_has_no_main_guard() -> None:
    """Cells per CLAUDE.md §2 are top-level only — no ``if __name__ == '__main__'``."""
    text = SMOKE_CELL_PATH.read_text(encoding="utf-8")
    assert '__main__' not in text or 'if __name__' not in text, (
        "cell must not contain an `if __name__ == '__main__':` block; "
        "expose a top-level `main()` function instead"
    )


def test_smoke_cell_imports_cleanly() -> None:
    """Importing the cell must not crash even before the env/rewards land.

    If the import-time dependencies are absent the cell is allowed to skip
    via ``pytest.importorskip`` semantics, but bare module load must not
    raise unrelated errors.
    """
    try:
        mod = importlib.import_module("cells.step_11_smoke_env")
    except ModuleNotFoundError as exc:
        # Tolerated only when the missing module is one of the parallel deps.
        missing = exc.name or ""
        if missing in {"cells.step_08_rewards", "cells.step_10_env"}:
            pytest.skip(f"parallel dep not yet implemented: {missing}")
        raise
    assert mod is not None


def test_smoke_cell_exposes_public_api() -> None:
    mod = pytest.importorskip("cells.step_11_smoke_env")
    assert hasattr(mod, "run_smoke_episode"), "missing run_smoke_episode()"
    assert hasattr(mod, "main"), "missing main()"
    assert callable(mod.run_smoke_episode)
    assert callable(mod.main)


def test_smoke_cell_no_forbidden_imports() -> None:
    """The smoke cell is text-only — must not pull in heavy deps."""
    text = SMOKE_CELL_PATH.read_text(encoding="utf-8")
    forbidden = ("torch", "transformers", "unsloth", "kokoro", "faster_whisper")
    for needle in forbidden:
        # Allow them only if mentioned in comments / docstrings (very rare).
        # Be conservative — flag any `import <pkg>` or `from <pkg>`.
        bad_import = re.search(rf"^\s*(?:from|import)\s+{re.escape(needle)}\b", text, re.M)
        assert bad_import is None, f"forbidden import found: {needle}"


# ---------------------------------------------------------------------------
# Behavioural smoke tests (require step_08 + step_10)
# ---------------------------------------------------------------------------


def _import_smoke_or_skip() -> Any:
    pytest.importorskip("cells.step_08_rewards")
    pytest.importorskip("cells.step_10_env")
    return pytest.importorskip("cells.step_11_smoke_env")


def test_env_constructs_for_stage1() -> None:
    smoke = _import_smoke_or_skip()
    env_mod = importlib.import_module("cells.step_10_env")
    DriftCallEnv = env_mod.DriftCallEnv

    env = DriftCallEnv(
        config={
            "curriculum_stage": 1,
            "language_weights": {"en": 1.0},
            "audio_boundary_enabled": False,
        },
    )
    assert env is not None
    # touch helper to keep coverage on the cell file even if it inlines build
    assert callable(smoke.run_smoke_episode)


def test_reset_shape_seed_42() -> None:
    _import_smoke_or_skip()
    env_mod = importlib.import_module("cells.step_10_env")
    DriftCallEnv = env_mod.DriftCallEnv

    env = DriftCallEnv(
        config={
            "curriculum_stage": 1,
            "language_weights": {"en": 1.0},
            "audio_boundary_enabled": False,
        },
    )
    obs = env.reset(seed=42)
    assert obs.turn == 0
    assert len(obs.tool_results) == 0
    assert len(obs.drift_log) == 0
    assert obs.goal.domain in {"airline", "cab", "restaurant", "hotel", "payment"}
    # budget_remaining at turn 0 == max_turns. Stage 1 -> 8 (env.md §3.2).
    assert obs.budget_remaining == 8


def test_step_single_action_increments_turn() -> None:
    _import_smoke_or_skip()
    from cells.step_04_models import ActionType, DriftCallAction
    env_mod = importlib.import_module("cells.step_10_env")
    DriftCallEnv = env_mod.DriftCallEnv

    env = DriftCallEnv(
        config={
            "curriculum_stage": 1,
            "language_weights": {"en": 1.0},
            "audio_boundary_enabled": False,
        },
    )
    obs0 = env.reset(seed=42)
    assert obs0.turn == 0

    domain = obs0.goal.domain
    # Pick the first available tool for that domain (search-style if present).
    candidates = [t for t in obs0.available_tools if t.startswith(f"{domain}.")]
    assert candidates, f"no available tools for domain {domain}"
    tool = candidates[0]

    obs1 = env.step(
        DriftCallAction(
            action_type=ActionType.TOOL_CALL,
            tool_name=tool,
            tool_args={},
            rationale="smoke probe",
        ),
    )
    assert obs1.turn == 1
    assert len(obs1.tool_results) >= 1


def test_run_smoke_episode_returns_rewards() -> None:
    smoke = _import_smoke_or_skip()
    result = smoke.run_smoke_episode(seed=42)
    # Result is a tuple (env, rewards) per the cell's contract.
    assert hasattr(result, "rewards") or isinstance(result, tuple), (
        "run_smoke_episode must return either a SmokeResult-like object or "
        "(env, rewards)"
    )
    rewards = result.rewards if hasattr(result, "rewards") else result[-1]
    # All five components in their documented intervals.
    assert 0.0 <= rewards.r1 <= 1.0
    assert 0.0 <= rewards.r2 <= 1.0
    assert 0.0 <= rewards.r3 <= 1.0
    assert 0.0 <= rewards.r4 <= 1.0
    assert -1.0 <= rewards.r5 <= 0.0
    assert 0.0 <= rewards.reward <= 1.0


def test_full_episode_roundtrip_terminates() -> None:
    smoke = _import_smoke_or_skip()
    result = smoke.run_smoke_episode(seed=42)
    env = result.env if hasattr(result, "env") else result[0]
    assert env.done() is True
    episode = env.episode()
    assert episode.terminated_by in {"SUBMIT", "ABORT", "TIMEOUT", "ANTI_HACK"}
    assert episode.turns_used >= 1


def test_run_smoke_episode_prints_summary_table() -> None:
    smoke = _import_smoke_or_skip()
    buf = io.StringIO()
    with redirect_stdout(buf):
        smoke.main()
    out = buf.getvalue()
    # Look for stable markers from the summary table.
    assert "DriftCall smoke episode" in out or "Smoke episode" in out, (
        f"summary banner missing; got:\n{out}"
    )
    assert "reward" in out.lower()
    assert "r1" in out.lower()


# ---------------------------------------------------------------------------
# Integration: run the cell as a script and verify it exits 0
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cell_runs_as_script_and_exits_zero() -> None:
    """Execute the cell with ``python3 -m cells.step_11_smoke_env`` semantics.

    Cells per CLAUDE.md §2 are top-level only, so we run them by importing
    the module and calling ``main()`` from a subprocess shim. This catches
    accidents like blocking input(), unguarded network I/O, or import-time
    side effects.
    """
    pytest.importorskip("cells.step_08_rewards")
    pytest.importorskip("cells.step_10_env")
    pytest.importorskip("cells.step_11_smoke_env")

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from cells import step_11_smoke_env\n"
        "step_11_smoke_env.main()\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"cell exited {completed.returncode}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "reward" in completed.stdout.lower()
