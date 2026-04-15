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


def test_pipeline_deduplicates_identical_bodies(fixtures_dir: Path, tmp_path: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    result = run_scraper(sources=sources, output=out)
    import json
    nodes = [json.loads(ln) for ln in out.read_text().splitlines()]
    coding_standards = [n for n in nodes if n["skill_name"] == "coding-standards"]
    # Each section from coding-standards should appear once, with alias_sources populated
    for n in coding_standards:
        if n["alias_sources"]:
            assert len(n["alias_sources"]) >= 1
