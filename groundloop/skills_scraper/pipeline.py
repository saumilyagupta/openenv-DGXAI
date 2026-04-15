from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from groundloop.skills_scraper.chunker import chunk_body
from groundloop.skills_scraper.discovery import walk_sources
from groundloop.skills_scraper.models import (
    ScrapeError,
    SkillNode,
    SourceRoot,
)
from groundloop.skills_scraper.parser import ParseError, parse_skill
from groundloop.skills_scraper.tagger import infer_tags
from groundloop.skills_scraper.writer import manifest_for, write_corpus, write_manifest

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrapeResult:
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[ScrapeError]
    corpus_path: Path
    manifest_path: Path


def _node_id(source_path: str, section_path: tuple[str, ...]) -> str:
    raw = source_path + "#" + "/".join(section_path)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def run_scraper(*, sources: list[SourceRoot], output: Path) -> ScrapeResult:
    nodes: list[SkillNode] = []
    errors: list[ScrapeError] = []
    scraped = 0
    skipped = 0

    for path, root in walk_sources(sources):
        try:
            parsed = parse_skill(path)
        except ParseError as e:
            errors.append(ScrapeError(path=str(path), stage="parse", reason=str(e)))
            skipped += 1
            continue

        body = parsed.body.strip()
        if not body:
            skipped += 1
            continue

        fm = parsed.frontmatter
        name_val = fm.get("name")
        skill_name = str(name_val) if name_val else path.parent.name
        desc_val = fm.get("description", "")
        skill_desc = str(desc_val) if desc_val else ""
        type_val = fm.get("type")
        skill_type: Literal["flexible", "rigid"] | None = (
            type_val if type_val in ("flexible", "rigid") else None
        )
        triggers = skill_desc

        raw_chunks = chunk_body(body)
        if not raw_chunks:
            skipped += 1
            continue

        # Spec §4.3: if no H2/H3 at all, emit a single chunk with
        # section_path=(skill_name,) — chunker returns () in that case.
        chunks = [
            c.model_copy(update={"section_path": (skill_name,)})
            if c.section_path == ()
            else c
            for c in raw_chunks
        ]

        for chunk in chunks:
            tags = tuple(
                infer_tags(
                    skill_name,
                    chunk.section_path[-1] if chunk.section_path else "",
                    chunk.section_body,
                )
            )
            node = SkillNode(
                id=_node_id(str(path), chunk.section_path),
                skill_name=skill_name,
                skill_description=skill_desc,
                skill_type=skill_type,
                section_path=chunk.section_path,
                section_title=chunk.section_path[-1] if chunk.section_path else "",
                section_body=chunk.section_body,
                source_path=str(path),
                source_root=root.label,
                tags=tags,
                trigger_hints=triggers,
                mtime=parsed.mtime,
                body_hash=_body_hash(chunk.section_body),
                alias_sources=(),
            )
            nodes.append(node)
        scraped += 1

    deduped = _dedupe(nodes)
    corpus_path = write_corpus(deduped, output)
    manifest = manifest_for(
        corpus_path=corpus_path,
        sources=sources,
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
    )
    manifest_path = output.with_suffix(".manifest.json")
    write_manifest(manifest, manifest_path)

    return ScrapeResult(
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
        corpus_path=corpus_path,
        manifest_path=manifest_path,
    )


def _dedupe(nodes: list[SkillNode]) -> list[SkillNode]:
    index: dict[tuple[str, tuple[str, ...], str], SkillNode] = {}
    for n in nodes:
        key = (n.skill_name, n.section_path, n.body_hash)
        existing = index.get(key)
        if existing is None:
            index[key] = n
        else:
            aliases = (*existing.alias_sources, n.source_path)
            index[key] = existing.model_copy(update={"alias_sources": aliases})
    return list(index.values())
