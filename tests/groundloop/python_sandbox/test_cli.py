from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundloop.python_sandbox.cli import main


def test_cli_json_output(clean_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(clean_project), "--tool", "imports", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "composite_score" in data


def test_cli_missing_dir(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "nonexistent"), "--format", "json"])
    assert rc == 1
