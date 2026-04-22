from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

# Lightweight tokenizer — matches CodeForge behavior closely enough for retrieval.
import re

_TOK = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text or "")]


@dataclass(frozen=True)
class Snippet:
    node_id: str
    skill_name: str
    section_title: str
    body: str
    score: float


class KBRetriever:
    """BM25 retriever over CodeForge skills corpus. Pure-Python, no HTTP."""

    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._nodes: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def build(self) -> None:
        if not self._corpus_path.is_file():
            raise FileNotFoundError(f"corpus missing: {self._corpus_path}")
        tokenized: list[list[str]] = []
        with self._corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                node = json.loads(line)
                self._nodes.append(node)
                tokenized.append(tokenize(str(node.get("section_body", ""))))
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, *, top_k: int = 3, snippet_chars: int = 400) -> list[Snippet]:
        q = tokenize(query)
        if not q or self._bm25 is None or not self._nodes:
            return []
        scores = self._bm25.get_scores(q)
        scored = sorted(
            range(len(self._nodes)),
            key=lambda i: (-float(scores[i]), self._nodes[i]["id"]),
        )[:top_k]
        out: list[Snippet] = []
        for i in scored:
            n = self._nodes[i]
            body = str(n.get("section_body", "")).strip()
            if len(body) > snippet_chars:
                body = body[:snippet_chars].rstrip() + "..."
            title_parts = n.get("section_path") or [n.get("section_title", "")]
            title = " / ".join(str(p) for p in title_parts if p)
            out.append(
                Snippet(
                    node_id=str(n["id"]),
                    skill_name=str(n.get("skill_name", "")),
                    section_title=title,
                    body=body,
                    score=float(scores[i]),
                )
            )
        return out
