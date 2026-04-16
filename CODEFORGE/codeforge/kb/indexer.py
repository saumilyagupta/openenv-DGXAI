from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from codeforge.kb.models import Cluster, ClusterManifest, SearchResult
from codeforge.kb.tokenizer import tokenize


class SkillsIndex:
    """BM25-backed full-text search over the skill documentation corpus."""

    def __init__(self, *, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._nodes: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_sha256: str = ""
        self._cluster_manifest: ClusterManifest | None = None
        self._node_to_cluster: dict[str, Cluster] = {}

    def build(self) -> None:
        """Load JSONL corpus and build BM25 index."""
        if not self._corpus_path.is_file():
            msg = f"corpus missing: {self._corpus_path}"
            raise FileNotFoundError(msg)
        self._corpus_sha256 = hashlib.sha256(
            self._corpus_path.read_bytes(),
        ).hexdigest()
        self._nodes = []
        self._tokenized = []
        with self._corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                node: dict[str, Any] = json.loads(line)
                self._nodes.append(node)
                self._tokenized.append(
                    tokenize(str(node.get("section_body", ""))),
                )
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        required_tags: set[str] | None = None,
    ) -> list[SearchResult]:
        """BM25 search over corpus, optionally filtered by tags."""
        q_tokens = tokenize(query)
        if not q_tokens or self._bm25 is None or not self._nodes:
            return []
        candidates: list[int] = list(range(len(self._nodes)))
        if required_tags:
            candidates = [
                i
                for i in candidates
                if required_tags.issubset(set(self._nodes[i].get("tags", [])))
            ]
            if not candidates:
                return []
        all_scores = self._bm25.get_scores(q_tokens)
        scored = [(i, float(all_scores[i])) for i in candidates]
        scored.sort(key=lambda pair: (-pair[1], self._nodes[pair[0]]["id"]))
        top = scored[:top_k]
        results: list[SearchResult] = []
        for rank, (i, score) in enumerate(top, start=1):
            node = self._nodes[i]
            cluster = self._node_to_cluster.get(node["id"])
            results.append(
                SearchResult(
                    node_id=node["id"],
                    skill_name=node["skill_name"],
                    section_path=tuple(node["section_path"]),
                    section_body=node["section_body"],
                    tags=tuple(node["tags"]),
                    source_path=node["source_path"],
                    score=score,
                    rank=rank,
                    cluster_id=cluster.cluster_id if cluster else None,
                ),
            )
        return results

    def attach_cluster_manifest(self, manifest: ClusterManifest) -> None:
        """Wire cluster assignments to nodes for search enrichment."""
        self._cluster_manifest = manifest
        self._node_to_cluster = {
            nid: cluster
            for cluster in manifest.clusters
            for nid in cluster.member_node_ids
        }

    def cluster_by_label(self, label: str) -> Cluster | None:
        """Look up a cluster by its label string."""
        if self._cluster_manifest is None:
            return None
        for c in self._cluster_manifest.clusters:
            if c.label == label:
                return c
        return None

    def nodes_in_cluster(
        self,
        cluster_label: str,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """Return corpus nodes belonging to the named cluster."""
        cluster = self.cluster_by_label(cluster_label)
        if cluster is None:
            return []
        member_ids = set(cluster.member_node_ids)
        results: list[SearchResult] = []
        for node in self._nodes:
            if node["id"] not in member_ids:
                continue
            results.append(
                SearchResult(
                    node_id=node["id"],
                    skill_name=node["skill_name"],
                    section_path=tuple(node["section_path"]),
                    section_body=node["section_body"],
                    tags=tuple(node["tags"]),
                    source_path=node["source_path"],
                    score=0.0,
                    rank=len(results) + 1,
                    cluster_id=cluster.cluster_id,
                ),
            )
            if len(results) >= top_k:
                break
        return results

    def stats(self) -> dict[str, int | float]:
        """Return index statistics: node_count, vocab_size, avg_doc_len."""
        if not self._tokenized:
            return {"node_count": 0, "vocab_size": 0, "avg_doc_len": 0.0}
        vocab: set[str] = set()
        total_len = 0
        for toks in self._tokenized:
            vocab.update(toks)
            total_len += len(toks)
        return {
            "node_count": len(self._nodes),
            "vocab_size": len(vocab),
            "avg_doc_len": total_len / len(self._tokenized),
        }

    def all_cluster_labels(self) -> list[str]:
        """Return all cluster labels (for MCP discovery tool)."""
        if self._cluster_manifest is None:
            return []
        return [c.label for c in self._cluster_manifest.clusters]

    def all_tags(self) -> set[str]:
        """Return all unique tags across the corpus (for MCP discovery tool)."""
        tags: set[str] = set()
        for node in self._nodes:
            for t in node.get("tags", []):
                tags.add(str(t))
        return tags
