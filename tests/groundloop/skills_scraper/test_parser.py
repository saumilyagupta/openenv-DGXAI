from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.skills_scraper.parser import ParseError, parse_skill


def test_parse_normal(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["normal"])
    assert parsed.frontmatter["name"] == "python-testing"
    assert "Fixtures" in parsed.body
    assert parsed.mtime > 0


def test_parse_no_frontmatter(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["no_frontmatter"])
    assert parsed.frontmatter == {}
    assert "Bare Skill" in parsed.body


def test_parse_malformed_raises(fixture_paths: dict[str, Path]) -> None:
    with pytest.raises(ParseError):
        parse_skill(fixture_paths["malformed"])


def test_parse_single_section(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["single_section"])
    assert parsed.frontmatter["name"] == "flat-skill"
    assert parsed.body.strip().startswith("Just one blob")
