from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.skills_scraper.models import (
    ParsedSkill,
    ScrapeError,
    ScrapeManifest,
    SectionChunk,
    SkillNode,
    SourceRoot,
)


def test_source_root_basic():
    root = SourceRoot(label="user-skills", glob="~/.claude/skills/*/SKILL.md")
    assert root.label == "user-skills"
    assert root.glob == "~/.claude/skills/*/SKILL.md"


def test_skill_node_is_frozen():
    node = SkillNode(
        id="abc123",
        skill_name="python-testing",
        skill_description="Testing patterns",
        skill_type="flexible",
        section_path=("Fixtures",),
        section_title="Fixtures",
        section_body="Use pytest fixtures.",
        source_path="/tmp/x/SKILL.md",
        source_root="user-skills",
        tags=("domain:python", "phase:test"),
        trigger_hints="Testing patterns",
        mtime=1.0,
        body_hash="deadbeef",
        alias_sources=(),
    )
    with pytest.raises(ValidationError):
        node.skill_name = "other"  # type: ignore[misc]


def test_skill_node_rejects_missing_required():
    with pytest.raises(ValidationError):
        SkillNode()  # type: ignore[call-arg]


def test_skill_type_accepts_none():
    node = SkillNode(
        id="x",
        skill_name="n",
        skill_description="",
        skill_type=None,
        section_path=(),
        section_title="",
        section_body="",
        source_path="",
        source_root="",
        tags=(),
        trigger_hints="",
        mtime=0.0,
        body_hash="",
        alias_sources=(),
    )
    assert node.skill_type is None


def test_scrape_manifest_shape():
    m = ScrapeManifest(
        generated_at="2026-04-15T00:00:00Z",
        sources=[SourceRoot(label="a", glob="b")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=5,
        errors=[],
        corpus_sha256="abc",
    )
    assert m.total_nodes == 5


def test_scrape_error_carries_context():
    err = ScrapeError(path="/tmp/x", stage="parse", reason="bad yaml")
    assert err.stage == "parse"


def test_parsed_skill_defaults():
    p = ParsedSkill(frontmatter={}, body="", mtime=0.0)
    assert p.frontmatter == {}


def test_section_chunk_shape():
    c = SectionChunk(section_path=("A", "B"), section_body="hello")
    assert c.section_path == ("A", "B")
