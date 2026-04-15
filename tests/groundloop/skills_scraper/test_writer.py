from __future__ import annotations

import json
from pathlib import Path

from groundloop.skills_scraper.models import ScrapeError, ScrapeManifest, SkillNode, SourceRoot
from groundloop.skills_scraper.writer import manifest_for, write_corpus, write_manifest


def _node(nid: str, body: str = "body") -> SkillNode:
    return SkillNode(
        id=nid,
        skill_name="s",
        skill_description="d",
        skill_type=None,
        section_path=(),
        section_title="",
        section_body=body,
        source_path="/p",
        source_root="r",
        tags=(),
        trigger_hints="",
        mtime=0.0,
        body_hash="h",
        alias_sources=(),
    )


def test_write_corpus_sorted_jsonl(tmp_path: Path):
    out = tmp_path / "c.jsonl"
    nodes = [_node("zzz"), _node("aaa"), _node("mmm")]
    path = write_corpus(nodes, out)
    lines = path.read_text().splitlines()
    ids = [json.loads(ln)["id"] for ln in lines]
    assert ids == ["aaa", "mmm", "zzz"]


def test_write_manifest(tmp_path: Path):
    m = ScrapeManifest(
        generated_at="2026-04-15T00:00:00Z",
        sources=[SourceRoot(label="x", glob="y")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=1,
        errors=[ScrapeError(path="/p", stage="parse", reason="r")],
        corpus_sha256="abc",
    )
    out = tmp_path / "m.json"
    write_manifest(m, out)
    parsed = json.loads(out.read_text())
    assert parsed["total_nodes"] == 1
    assert parsed["errors"][0]["stage"] == "parse"


def test_manifest_for_computes_hash(tmp_path: Path):
    out = tmp_path / "c.jsonl"
    out.write_text('{"id":"x"}\n')
    m = manifest_for(
        corpus_path=out,
        sources=[SourceRoot(label="a", glob="b")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=1,
        errors=[],
    )
    assert len(m.corpus_sha256) == 64
