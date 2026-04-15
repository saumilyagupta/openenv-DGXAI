from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.skills_scraper.cli import main


def test_cli_on_fixtures(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(fixtures_dir / "**" / "SKILL.md"),
        "--output", str(out),
    ])
    assert exit_code == 0
    assert out.exists()


def test_cli_zero_nodes_nonzero_exit(tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(tmp_path / "nonexistent" / "*.md"),
        "--output", str(out),
    ])
    assert exit_code == 1
