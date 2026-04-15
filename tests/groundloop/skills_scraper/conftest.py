from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "fake_skills"


@pytest.fixture
def fixture_paths(fixtures_dir: Path) -> dict[str, Path]:
    return {
        "normal": fixtures_dir / "normal" / "SKILL.md",
        "no_frontmatter": fixtures_dir / "no_frontmatter" / "SKILL.md",
        "malformed": fixtures_dir / "malformed_yaml" / "SKILL.md",
        "single_section": fixtures_dir / "single_section" / "SKILL.md",
        "dup_a": fixtures_dir / "dup_a" / "coding-standards" / "SKILL.md",
        "dup_b": fixtures_dir / "dup_b" / "coding-standards" / "SKILL.md",
    }
