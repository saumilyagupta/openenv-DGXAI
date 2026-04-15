from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from groundloop.kb_indexer.models import Cluster, ClusterManifest
from groundloop.kb_indexer.tokenizer import tokenize

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "are", "was",
    "not", "but", "use", "can", "all", "one", "from", "when",
    "which", "have", "any", "should", "would", "must", "will",
    "your", "you", "our", "its", "their", "them", "they",
})


def _filter_tokens(tokens: list[str], min_length: int = 3) -> list[str]:
    return [t for t in tokens if len(t) >= min_length and t not in _STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return len(inter) / len(union)


def _connected_components(
    node_ids: list[str], adj: dict[str, set[str]],
) -> list[set[str]]:
    seen: set[str] = set()
    comps: list[set[str]] = []
    for start in node_ids:
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            nid = stack.pop()
            if nid in comp:
                continue
            comp.add(nid)
            seen.add(nid)
            for nbr in adj.get(nid, ()):
                if nbr not in comp:
                    stack.append(nbr)
        comps.append(comp)
    return comps


def _dominant_domain(tags_list: Iterable[tuple[str, ...]]) -> str:
    counts: Counter[str] = Counter()
    for tags in tags_list:
        for t in tags:
            if t.startswith("domain:"):
                counts[t.split(":", 1)[1]] += 1
    if not counts:
        return "general"
    return counts.most_common(1)[0][0]


def _label_cluster(
    nodes_data: list[dict[str, Any]],
) -> tuple[str, tuple[str, ...], str]:
    all_tokens: Counter[str] = Counter()
    tag_lists: list[tuple[str, ...]] = []
    for nd in nodes_data:
        all_tokens.update(nd["tokens"])
        tag_lists.append(tuple(nd.get("tags", ())))
    top3 = [t for t, _ in all_tokens.most_common(3)]
    dominant = _dominant_domain(tag_lists)
    label = f"{dominant}_" + "_".join(top3) if top3 else dominant
    return label, tuple(top3), dominant


def build_clusters(
    nodes: list[dict[str, Any]],
    *,
    jaccard_threshold: float = 0.15,
    min_token_length: int = 3,
    corpus_sha256: str = "",
    generated_at: str | None = None,
) -> ClusterManifest:
    node_ids = [n["id"] for n in nodes]
    token_sets: dict[str, set[str]] = {
        n["id"]: set(
            _filter_tokens(tokenize(n.get("section_body", "")), min_token_length),
        )
        for n in nodes
    }
    adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
    # O(n²) pair scan over corpus.
    # Acceptable up to ~5k nodes (~12M comparisons).
    # For larger corpora switch to MinHash/LSH (datasketch) or semantic embeddings.
    for i, a in enumerate(node_ids):
        for b in node_ids[i + 1:]:
            sim = _jaccard(token_sets[a], token_sets[b])
            if sim >= jaccard_threshold:
                adj[a].add(b)
                adj[b].add(a)

    components = _connected_components(node_ids, adj)
    node_by_id = {n["id"]: n for n in nodes}

    clusters: list[Cluster] = []
    singletons = 0
    for comp in components:
        member_ids = sorted(comp)
        nodes_data = [
            {
                "tokens": _filter_tokens(
                    tokenize(node_by_id[nid].get("section_body", "")),
                    min_token_length,
                ),
                "tags": tuple(node_by_id[nid].get("tags", ())),
            }
            for nid in member_ids
        ]
        label, top_tokens, dominant = _label_cluster(nodes_data)
        cluster_id = hashlib.sha256(
            "|".join(member_ids).encode(),
        ).hexdigest()[:12]
        if len(member_ids) == 1:
            singletons += 1
        clusters.append(Cluster(
            cluster_id=cluster_id,
            label=label,
            dominant_domain=dominant,
            top_tokens=top_tokens,
            node_count=len(member_ids),
            member_node_ids=tuple(member_ids),
        ))
    clusters.sort(key=lambda c: (-c.node_count, c.cluster_id))

    stamp = (
        generated_at
        if generated_at is not None
        else datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    return ClusterManifest(
        generated_at=stamp,
        corpus_sha256=corpus_sha256,
        jaccard_threshold=jaccard_threshold,
        total_clusters=len(clusters),
        total_nodes_clustered=len(node_ids),
        singletons=singletons,
        clusters=tuple(clusters),
    )


def save_manifest(manifest: ClusterManifest, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_manifest(path: Path) -> ClusterManifest:
    return ClusterManifest.model_validate_json(path.read_text(encoding="utf-8"))
