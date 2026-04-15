from __future__ import annotations

from pathlib import Path

from groundloop.skills_scraper.discovery import walk_sources
from groundloop.skills_scraper.models import SourceRoot


def test_walk_finds_all_fixture_skills(fixtures_dir: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    results = list(walk_sources(sources))
    assert len(results) == 6


def test_walk_yields_path_and_source_root(fixtures_dir: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    for path, root in walk_sources(sources):
        assert path.is_file()
        assert root.label == "fake"


def test_walk_expands_tilde(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".x").mkdir()
    (tmp_path / ".x" / "SKILL.md").write_text("")
    sources = [SourceRoot(label="home", glob="~/.x/SKILL.md")]
    results = list(walk_sources(sources))
    assert len(results) == 1


def test_walk_skips_missing_dirs() -> None:
    sources = [SourceRoot(label="missing", glob="/nonexistent/**/SKILL.md")]
    assert list(walk_sources(sources)) == []
