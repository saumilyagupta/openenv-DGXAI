from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundloop.ralph_orchestrator.cli import main


def test_cli_runs_with_stub(
    spec_path: Path,
    tiny_corpus_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    initial = tmp_path / "main.py"
    initial.write_text("from __future__ import annotations\n\n\ndef greet(n: str) -> str:\n    return 'hi'\n")
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tiny_corpus_path),
        "--initial-file", f"main.py={initial}",
        "--max-iters", "1",
        "--target-score", "1.1",
        "--synthesizer", "stub",
        "--format", "json",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "run_id" in data
    assert "terminated_by" in data


def test_cli_missing_corpus(tmp_path: Path, spec_path: Path) -> None:
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--initial-file", f"main.py={tmp_path / 'whatever.py'}",
    ])
    assert rc == 1


def test_cli_missing_initial_file(tmp_path: Path, spec_path: Path, tiny_corpus_path: Path) -> None:
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tiny_corpus_path),
        "--initial-file", f"main.py={tmp_path / 'missing.py'}",
    ])
    assert rc == 1
