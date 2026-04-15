from __future__ import annotations

import json
from pathlib import Path

from groundloop.skills_scraper.cli import main


def test_e2e_cli_against_fixtures(fixtures_dir: Path, tmp_path: Path) -> None:
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "sources:\n"
        f"  - label: fake\n"
        f"    glob: {fixtures_dir}/**/SKILL.md\n"
    )
    out = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "corpus.manifest.json"

    exit_code = main([
        "--sources", str(sources_yaml),
        "--output", str(out),
    ])

    assert exit_code == 0
    assert out.exists(), "corpus JSONL must be written"
    assert manifest.exists(), "manifest JSON must be written"

    nodes = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert nodes, "at least one node expected from fixtures"

    # dup_a and dup_b have identical coding-standards content -> at least one
    # node must carry an alias_sources entry.
    assert any(
        n["alias_sources"] for n in nodes if n["skill_name"] == "coding-standards"
    ), "dedup must populate alias_sources for identical-body duplicates"

    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["total_nodes"] == len(nodes)
    assert len(manifest_data["corpus_sha256"]) == 64
