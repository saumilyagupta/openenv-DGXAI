from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from groundloop.skills_scraper.models import (
    ScrapeError,
    ScrapeManifest,
    SkillNode,
    SourceRoot,
)


def write_corpus(nodes: Iterable[SkillNode], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_nodes = sorted(nodes, key=lambda n: n.id)
    with out_path.open("w", encoding="utf-8") as f:
        for node in sorted_nodes:
            f.write(node.model_dump_json())
            f.write("\n")
    return out_path


def write_manifest(manifest: ScrapeManifest, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def manifest_for(
    *,
    corpus_path: Path,
    sources: list[SourceRoot],
    scraped_files: int,
    skipped_files: int,
    total_nodes: int,
    errors: list[ScrapeError],
) -> ScrapeManifest:
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    return ScrapeManifest(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sources=sources,
        scraped_files=scraped_files,
        skipped_files=skipped_files,
        total_nodes=total_nodes,
        errors=errors,
        corpus_sha256=digest,
    )
