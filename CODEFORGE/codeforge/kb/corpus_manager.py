from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from codeforge.scraper.discovery import SourceRoot, walk_sources
from codeforge.scraper.parser import ParseError, parse_skill
from codeforge.scraper.pipeline import _dedupe, scrape_single_skill

_log = logging.getLogger(__name__)


class SkillCorpusManager:
    """Manages the skill corpus: add, remove, refresh, save/load JSONL."""

    def __init__(self, *, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._nodes: list[dict[str, Any]] = []

    @property
    def nodes(self) -> list[dict[str, Any]]:
        """Read-only access to the current node list."""
        return list(self._nodes)

    def load(self) -> None:
        """Load existing corpus from JSONL file."""
        if not self._corpus_path.is_file():
            self._nodes = []
            return
        nodes: list[dict[str, Any]] = []
        with self._corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nodes.append(json.loads(line))
        self._nodes = nodes

    def add_skill(self, path: Path) -> int:
        """Scrape a single SKILL.md, append nodes to corpus. Returns count added."""
        new_nodes = scrape_single_skill(path)
        if not new_nodes:
            return 0
        combined = list(self._nodes) + new_nodes
        deduped = _dedupe(combined)
        added = len(deduped) - len(self._nodes)
        self._nodes = deduped
        return max(added, 0)

    def remove_skill(self, skill_name: str) -> int:
        """Remove all nodes for skill_name. Returns count removed."""
        before = len(self._nodes)
        self._nodes = [n for n in self._nodes if n["skill_name"] != skill_name]
        return before - len(self._nodes)

    def refresh(
        self, *, sources: list[SourceRoot] | None = None
    ) -> dict[str, int]:
        """Diff disk sources vs corpus by mtime/body_hash.

        Returns {added, removed, unchanged}.
        """
        if sources is None:
            sources = []

        # Build a map of current corpus keyed by source_path
        existing_by_path: dict[str, list[dict[str, Any]]] = {}
        for node in self._nodes:
            sp = node.get("source_path", "")
            existing_by_path.setdefault(sp, []).append(node)

        # Discover what's on disk
        disk_paths: set[str] = set()
        disk_nodes: list[dict[str, Any]] = []
        for path, root in walk_sources(sources):
            disk_paths.add(str(path))
            try:
                parse_skill(path)
            except ParseError:
                continue
            file_nodes = scrape_single_skill(path)
            for n in file_nodes:
                n["source_root"] = root.label
            disk_nodes.extend(file_nodes)

        # Nodes from sources not on disk anymore -> removed
        removed_count = 0
        kept: list[dict[str, Any]] = []
        for node in self._nodes:
            sp = node.get("source_path", "")
            if sp in disk_paths:
                # Check if body_hash or mtime changed
                pass  # handled below
            elif sources:
                # source_path no longer on disk via any source root
                removed_count += 1
                continue
            kept.append(node)

        # Determine unchanged vs changed via body_hash comparison
        existing_hashes: set[tuple[str, str]] = set()
        for node in kept:
            existing_hashes.add(
                (node.get("source_path", ""), node.get("body_hash", ""))
            )

        # Add new/changed nodes from disk
        added_count = 0
        for dn in disk_nodes:
            key = (dn.get("source_path", ""), dn.get("body_hash", ""))
            if key not in existing_hashes:
                added_count += 1

        # Replace nodes from disk-paths with fresh scraped versions
        non_disk = [n for n in kept if n.get("source_path", "") not in disk_paths]
        combined = non_disk + disk_nodes
        deduped = _dedupe(combined)
        unchanged = len(self._nodes) - removed_count - added_count
        if unchanged < 0:
            unchanged = 0

        self._nodes = deduped
        return {"added": added_count, "removed": removed_count, "unchanged": unchanged}

    def save(self) -> None:
        """Write corpus back to JSONL."""
        self._corpus_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_nodes = sorted(self._nodes, key=lambda n: str(n["id"]))
        with self._corpus_path.open("w", encoding="utf-8") as f:
            for node in sorted_nodes:
                f.write(json.dumps(node, default=str))
                f.write("\n")

    def node_count(self) -> int:
        """Return current number of nodes in the corpus."""
        return len(self._nodes)
