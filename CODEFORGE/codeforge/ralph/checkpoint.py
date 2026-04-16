from __future__ import annotations

import os
from pathlib import Path

from codeforge.ralph.models import RunResult


def save_checkpoint(run: RunResult, checkpoint_dir: Path) -> Path:
    """Atomically write a RunResult to ``checkpoint_dir/run_{run_id}.json``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_dir / f"run_{run.run_id}.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final


def load_checkpoint(checkpoint_path: Path) -> RunResult:
    """Load a RunResult from a checkpoint JSON file."""
    return RunResult.model_validate_json(
        checkpoint_path.read_text(encoding="utf-8"),
    )
