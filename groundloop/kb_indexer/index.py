from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from groundloop.kb_indexer.cache import load_cache, save_cache
from groundloop.kb_indexer.models import SearchResult
from groundloop.kb_indexer.tokenizer import tokenize


class SkillsIndex:
    def __init__(self, *, corpus_path: Path, cache_path: Path | None = None) -> None:
        self._corpus_path = corpus_path
        self._cache_path = cache_path
        self._nodes: list[dict] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_sha256: str = ""

    def build(self) -> None:
        if not self._corpus_path.is_file():
            msg = f"corpus missing: {self._corpus_path}"
            raise FileNotFoundError(msg)
        self._corpus_sha256 = hashlib.sha256(self._corpus_path.read_bytes()).hexdigest()
        self._nodes = []
        self._tokenized = []
        with self._corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                node = json.loads(line)
                self._nodes.append(node)
                self._tokenized.append(tokenize(node.get("section_body", "")))
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    def save(self) -> None:
        if self._cache_path is None:
            msg = "cache_path not set"
            raise ValueError(msg)
        state = {
            "tokenized": self._tokenized,
            "nodes": self._nodes,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_cache(state, self._cache_path, self._corpus_sha256)

    @classmethod
    def load(cls, *, corpus_path: Path, cache_path: Path) -> "SkillsIndex | None":
        if not corpus_path.is_file():
            return None
        sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        payload = load_cache(cache_path, sha)
        if payload is None:
            return None
        idx = cls(corpus_path=corpus_path, cache_path=cache_path)
        idx._nodes = payload["nodes"]
        idx._tokenized = payload["tokenized"]
        idx._corpus_sha256 = sha
        if idx._tokenized:
            idx._bm25 = BM25Okapi(idx._tokenized)
        return idx

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        required_tags: set[str] | None = None,
    ) -> list[SearchResult]:
        q_tokens = tokenize(query)
        if not q_tokens or self._bm25 is None or not self._nodes:
            return []
        candidates: list[int] = list(range(len(self._nodes)))
        if required_tags:
            candidates = [
                i for i in candidates
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
                )
            )
        return results

    def stats(self) -> dict[str, int | float]:
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
