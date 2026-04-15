from __future__ import annotations

from pathlib import Path

from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def test_pipeline_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    result = run_scraper(sources=sources, output=out)
    assert result.scraped_files >= 4  # normal + no_frontmatter + single + (dup dedup'd)
    assert result.skipped_files >= 1  # malformed
    assert out.exists()
    assert (tmp_path / "corpus.manifest.json").exists()


def test_pipeline_single_section_uses_skill_name(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    run_scraper(sources=sources, output=out)
    import json
    nodes = [json.loads(ln) for ln in out.read_text().splitlines()]
    flat = [n for n in nodes if n["skill_name"] == "flat-skill"]
    assert len(flat) == 1
    assert flat[0]["section_path"] == ["flat-skill"]
    assert flat[0]["section_title"] == "flat-skill"


def test_pipeline_deduplicates_identical_bodies(fixtures_dir: Path, tmp_path: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    run_scraper(sources=sources, output=out)
    import json
    nodes = [json.loads(ln) for ln in out.read_text().splitlines()]
    coding_standards = [n for n in nodes if n["skill_name"] == "coding-standards"]

    # dup_a and dup_b have identical bodies → each (skill, path, body_hash)
    # key must collapse to a single node whose alias_sources records the
    # second path. Collect one such node unconditionally — the assertion
    # must fail if dedup regresses.
    nodes_with_aliases = [n for n in coding_standards if n["alias_sources"]]
    assert nodes_with_aliases, (
        "dedup regression: expected at least one coding-standards node "
        "with a populated alias_sources from dup_a/dup_b collapse"
    )
    for n in nodes_with_aliases:
        assert len(n["alias_sources"]) >= 1
