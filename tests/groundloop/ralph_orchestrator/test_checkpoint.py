from __future__ import annotations

from pathlib import Path

from groundloop.ralph_orchestrator.checkpoint import load_checkpoint, save_checkpoint
from groundloop.ralph_orchestrator.models import RunResult


def _result() -> RunResult:
    return RunResult(
        run_id="r1", spec="s", started_at="t0", ended_at="t1",
        final_score=1.0, final_files={"main.py": "x = 1\n"},
        iterations=(), terminated_by="target_hit",
    )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    out = save_checkpoint(_result(), tmp_path)
    loaded = load_checkpoint(out)
    assert loaded.run_id == "r1"
    assert loaded.final_files == {"main.py": "x = 1\n"}


def test_checkpoint_atomic_write(tmp_path: Path) -> None:
    save_checkpoint(_result(), tmp_path)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
