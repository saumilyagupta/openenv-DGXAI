from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_corpus(nodes: Iterable[dict[str, Any]], out_path: Path) -> Path:
    """Write nodes as sorted JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_nodes = sorted(nodes, key=lambda n: str(n["id"]))
    with out_path.open("w", encoding="utf-8") as f:
        for node in sorted_nodes:
            f.write(json.dumps(node, default=str))
            f.write("\n")
    return out_path


def compute_corpus_sha256(corpus_path: Path) -> str:
    """SHA256 hex digest of the corpus file."""
    return hashlib.sha256(corpus_path.read_bytes()).hexdigest()


def write_manifest(
    *,
    corpus_path: Path,
    sources: list[dict[str, str]],
    scraped_files: int,
    skipped_files: int,
    total_nodes: int,
    errors: list[dict[str, str]],
) -> Path:
    """Write a manifest JSON alongside the corpus."""
    digest = compute_corpus_sha256(corpus_path)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": sources,
        "scraped_files": scraped_files,
        "skipped_files": skipped_files,
        "total_nodes": total_nodes,
        "errors": errors,
        "corpus_sha256": digest,
    }
    manifest_path = corpus_path.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest_path
