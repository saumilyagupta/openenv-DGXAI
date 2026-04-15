from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundloop.lib_grounder.cli import main


def test_cli_code_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["-c", "import os", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["groundedness"] == 1.0


def test_cli_code_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["-c", "import nonexistent_pkg_zzz_987"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "groundedness=" in out
    assert "ungrounded" in out


def test_cli_requires_source() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_file_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "sample.py"
    p.write_text("import os\n", encoding="utf-8")
    rc = main(["-f", str(p), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["groundedness"] == 1.0
