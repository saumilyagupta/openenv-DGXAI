from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeforge.scraper.chunker import chunk_body
from codeforge.scraper.discovery import SourceRoot, walk_sources
from codeforge.scraper.parser import ParseError, parse_skill
from codeforge.scraper.tagger import infer_tags
from codeforge.scraper.writer import write_corpus, write_manifest

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrapeResult:
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[dict[str, str]]
    corpus_path: Path
    manifest_path: Path


def _node_id(source_path: str, section_path: tuple[str, ...]) -> str:
    raw = source_path + "#" + "/".join(section_path)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def scrape_single_skill(path: Path) -> list[dict[str, Any]]:
    """Scrape a single SKILL.md file and return a list of node dicts."""
    parsed = parse_skill(path)
    body = parsed.body.strip()
    if not body:
        return []

    fm = parsed.frontmatter
    name_val = fm.get("name")
    skill_name = str(name_val) if name_val else path.parent.name
    desc_val = fm.get("description", "")
    skill_desc = str(desc_val) if desc_val else ""
    type_val = fm.get("type")
    skill_type: str | None = (
        str(type_val) if type_val in ("flexible", "rigid") else None
    )
    triggers = skill_desc

    raw_chunks = chunk_body(body)
    if not raw_chunks:
        return []

    chunks = [
        c.model_copy(update={"section_path": (skill_name,)})
        if c.section_path == ()
        else c
        for c in raw_chunks
    ]

    nodes: list[dict[str, Any]] = []
    for chunk in chunks:
        tags = list(
            infer_tags(
                skill_name,
                chunk.section_path[-1] if chunk.section_path else "",
                chunk.section_body,
            )
        )
        node: dict[str, Any] = {
            "id": _node_id(str(path), chunk.section_path),
            "skill_name": skill_name,
            "skill_description": skill_desc,
            "skill_type": skill_type,
            "section_path": list(chunk.section_path),
            "section_title": chunk.section_path[-1] if chunk.section_path else "",
            "section_body": chunk.section_body,
            "source_path": str(path),
            "source_root": "",
            "tags": tags,
            "trigger_hints": triggers,
            "mtime": parsed.mtime,
            "body_hash": _body_hash(chunk.section_body),
            "alias_sources": [],
        }
        nodes.append(node)
    return nodes


def _dedupe(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate nodes by (skill_name, section_path, body_hash)."""
    index: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
    for n in nodes:
        key = (n["skill_name"], tuple(n["section_path"]), n["body_hash"])
        existing = index.get(key)
        if existing is None:
            index[key] = n
        else:
            aliases = list(existing.get("alias_sources", []))
            aliases.append(n["source_path"])
            existing["alias_sources"] = aliases
    return list(index.values())


def run_scraper(
    *, sources: list[SourceRoot], output: Path
) -> ScrapeResult:
    """Full pipeline: discover -> parse -> chunk -> tag -> deduplicate -> write."""
    nodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scraped = 0
    skipped = 0

    for path, root in walk_sources(sources):
        try:
            file_nodes = scrape_single_skill(path)
        except ParseError as e:
            errors.append({"path": str(path), "stage": "parse", "reason": str(e)})
            skipped += 1
            continue

        if not file_nodes:
            skipped += 1
            continue

        for n in file_nodes:
            n["source_root"] = root.label
        nodes.extend(file_nodes)
        scraped += 1

    deduped = _dedupe(nodes)
    corpus_path = write_corpus(deduped, output)
    manifest_path = write_manifest(
        corpus_path=corpus_path,
        sources=[{"label": s.label, "glob": s.glob} for s in sources],
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
    )

    return ScrapeResult(
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
        corpus_path=corpus_path,
        manifest_path=manifest_path,
    )
