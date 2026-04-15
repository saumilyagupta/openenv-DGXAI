from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundloop.audit_reporter.cli import main
from groundloop.ralph_orchestrator.models import Iteration, RunResult


def test_cli_reads_run_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = RunResult(
        run_id="rx", spec="s", started_at="t", ended_at="t",
        final_score=0.5, final_files={},
        iterations=(
            Iteration(
                index=0, cited_node_ids=("n1",), rationale="r",
                proposed_files={}, sandbox_score_before=0.0,
                sandbox_score_after=0.5, kept=True, reason="score_improved",
            ),
        ),
        terminated_by="max_iters",
    )
    p = tmp_path / "run.json"
    p.write_text(run.model_dump_json(), encoding="utf-8")
    rc = main(["--run-json", str(p)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["run_id"] == "rx"
    assert payload["iterations_total"] == 1
