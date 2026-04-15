from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.interrogator.cli import main


def test_cli_brief_only(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--brief", "build a REST API"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("\n") >= 5


def test_cli_brief_with_corpus(
    tiny_corpus_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--brief", "build a python api", "--corpus", str(tiny_corpus_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "?" in out
