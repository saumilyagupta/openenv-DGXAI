# CodeForge M1 — Graphify Clustering Plan

> Use superpowers:subagent-driven-development.

**Goal:** Build connected-component clusters over the 1006-node skill corpus; emit `cluster_manifest.json`; wire into `SkillsIndex`.

**Spec:** `docs/superpowers/specs/2026-04-15-codeforge-m1-graphify-clustering.md`.

---

## Task 1: Extend models with `Cluster`, `ClusterManifest`, and `SearchResult.cluster_id`

**Files:** `groundloop/kb_indexer/models.py`, `tests/groundloop/kb_indexer/test_models.py` (extend).

- [ ] Add failing tests:
```python
import pytest
from pydantic import ValidationError

from groundloop.kb_indexer.models import Cluster, ClusterManifest, SearchResult


def test_cluster_frozen():
    c = Cluster(
        cluster_id="abc", label="python_testing",
        dominant_domain="python", top_tokens=("pytest",),
        node_count=3, member_node_ids=("n1", "n2", "n3"),
    )
    with pytest.raises(ValidationError):
        c.label = "other"  # type: ignore[misc]


def test_cluster_manifest_shape():
    m = ClusterManifest(
        generated_at="t", corpus_sha256="sha",
        jaccard_threshold=0.15, total_clusters=1,
        total_nodes_clustered=3, singletons=0,
        clusters=(Cluster(
            cluster_id="c1", label="l", dominant_domain="python",
            top_tokens=(), node_count=3, member_node_ids=("n1","n2","n3"),
        ),),
    )
    assert m.total_clusters == 1


def test_search_result_cluster_id_defaults_none():
    r = SearchResult(
        node_id="n1", skill_name="s", section_path=("x",),
        section_body="b", tags=("domain:python",), source_path="/p",
        score=1.0, rank=1,
    )
    assert r.cluster_id is None
```

- [ ] Implement (extend existing `models.py`):
```python
# add new classes at module end:

class Cluster(BaseModel):
    model_config = ConfigDict(frozen=True)
    cluster_id: str
    label: str
    dominant_domain: str
    top_tokens: tuple[str, ...]
    node_count: int
    member_node_ids: tuple[str, ...]


class ClusterManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: str
    corpus_sha256: str
    jaccard_threshold: float
    total_clusters: int
    total_nodes_clustered: int
    singletons: int
    clusters: tuple[Cluster, ...]
```

Also add `cluster_id: str | None = None` to `SearchResult`. Since `SearchResult` is frozen, this is a default value added to the definition.

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m1): add Cluster + ClusterManifest models; SearchResult.cluster_id`

---

## Task 2: `cluster.py` — Jaccard + connected components + labeling

**Files:** `groundloop/kb_indexer/cluster.py` (new), `tests/groundloop/kb_indexer/test_cluster.py` (new).

- [ ] Write failing tests:
```python
from __future__ import annotations

from groundloop.kb_indexer.cluster import (
    _connected_components,
    _jaccard,
    _label_cluster,
    build_clusters,
)


def test_jaccard_identity():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3  # |∩|=1, |∪|=3


def test_jaccard_empty_both():
    assert _jaccard(set(), set()) == 0.0


def test_connected_components_simple():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}, "d": set()}
    comps = _connected_components(["a", "b", "c", "d"], adj)
    assert len(comps) == 2
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 3]


def test_label_cluster_uses_top_tokens_and_domain():
    nodes_data = [
        {"tokens": ["pytest", "fixture", "test"], "domain": "python"},
        {"tokens": ["pytest", "coverage", "test"], "domain": "python"},
    ]
    label, top_tokens, dominant = _label_cluster(nodes_data)
    assert dominant == "python"
    assert "pytest" in top_tokens
    assert label.startswith("python_")


def test_build_clusters_on_tiny_fixture(tiny_corpus_path):
    # 5 nodes: python_testing x2, api-design, security-review, backend-patterns
    import json
    nodes = [json.loads(ln) for ln in tiny_corpus_path.read_text().splitlines() if ln.strip()]
    manifest = build_clusters(nodes, jaccard_threshold=0.1)
    assert manifest.total_nodes_clustered == len(nodes)
    assert manifest.total_clusters >= 1
    # All node IDs accounted for (in exactly one cluster)
    seen: set[str] = set()
    for c in manifest.clusters:
        for nid in c.member_node_ids:
            assert nid not in seen, f"node {nid} in multiple clusters"
            seen.add(nid)
    assert len(seen) == len(nodes)
```

- [ ] Implement `cluster.py`:
```python
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

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
    nodes_data: list[dict],
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
    nodes: list[dict],
    *,
    jaccard_threshold: float = 0.15,
    min_token_length: int = 3,
    corpus_sha256: str = "",
) -> ClusterManifest:
    node_ids = [n["id"] for n in nodes]
    token_sets: dict[str, set[str]] = {
        n["id"]: set(_filter_tokens(tokenize(n.get("section_body", "")), min_token_length))
        for n in nodes
    }
    # Build adjacency
    adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
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
                "tags": node_by_id[nid].get("tags", ()),
            }
            for nid in member_ids
        ]
        label, top_tokens, dominant = _label_cluster(nodes_data)
        cluster_id = hashlib.sha256("|".join(member_ids).encode()).hexdigest()[:12]
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

    return ClusterManifest(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
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
```

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m1): Jaccard clustering + connected components`

---

## Task 3: SkillsIndex cluster helpers

**Files:** `groundloop/kb_indexer/index.py` (extend), `tests/groundloop/kb_indexer/test_index.py` (extend).

- [ ] Write failing tests (add to existing test file):
```python
def test_attach_cluster_manifest_and_lookup(tiny_corpus_path, tmp_path):
    import json
    from groundloop.kb_indexer.cluster import build_clusters
    from groundloop.kb_indexer.index import SkillsIndex

    nodes = [json.loads(ln) for ln in tiny_corpus_path.read_text().splitlines() if ln.strip()]
    manifest = build_clusters(nodes, jaccard_threshold=0.1)

    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=tmp_path / "c.pkl")
    idx.build()
    idx.attach_cluster_manifest(manifest)

    # Every node has a cluster_id
    for n in nodes:
        assert idx.cluster_id_for(n["id"]) is not None


def test_nodes_in_cluster_returns_search_results(tiny_corpus_path, tmp_path):
    import json
    from groundloop.kb_indexer.cluster import build_clusters
    from groundloop.kb_indexer.index import SkillsIndex

    nodes = [json.loads(ln) for ln in tiny_corpus_path.read_text().splitlines() if ln.strip()]
    manifest = build_clusters(nodes, jaccard_threshold=0.1)

    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=tmp_path / "c.pkl")
    idx.build()
    idx.attach_cluster_manifest(manifest)

    # Grab the first cluster's label and query
    label = manifest.clusters[0].label
    results = idx.nodes_in_cluster(label, top_k=10)
    assert len(results) >= 1
    assert all(r.cluster_id == manifest.clusters[0].cluster_id for r in results)
```

- [ ] Extend `SkillsIndex` in `groundloop/kb_indexer/index.py`:
```python
# Add to class body:

def __init__(self, *, corpus_path: Path, cache_path: Path | None = None) -> None:
    # existing ...
    self._cluster_manifest: ClusterManifest | None = None
    self._node_to_cluster: dict[str, Cluster] = {}

def attach_cluster_manifest(self, manifest: ClusterManifest) -> None:
    self._cluster_manifest = manifest
    self._node_to_cluster = {
        nid: cluster
        for cluster in manifest.clusters
        for nid in cluster.member_node_ids
    }

def cluster_id_for(self, node_id: str) -> str | None:
    c = self._node_to_cluster.get(node_id)
    return c.cluster_id if c else None

def cluster_by_label(self, label: str) -> Cluster | None:
    if self._cluster_manifest is None:
        return None
    for c in self._cluster_manifest.clusters:
        if c.label == label:
            return c
    return None

def nodes_in_cluster(self, cluster_label: str, top_k: int = 50) -> list[SearchResult]:
    cluster = self.cluster_by_label(cluster_label)
    if cluster is None:
        return []
    member_ids = set(cluster.member_node_ids)
    results: list[SearchResult] = []
    for rank, node in enumerate(self._nodes, start=1):
        if node["id"] not in member_ids:
            continue
        results.append(SearchResult(
            node_id=node["id"],
            skill_name=node["skill_name"],
            section_path=tuple(node["section_path"]),
            section_body=node["section_body"],
            tags=tuple(node["tags"]),
            source_path=node["source_path"],
            score=0.0,
            rank=len(results) + 1,
            cluster_id=cluster.cluster_id,
        ))
        if len(results) >= top_k:
            break
    return results
```

Also update the existing `search` method to populate `cluster_id` on each result when a manifest is attached:
```python
# Inside search(), after building SearchResult, if self._node_to_cluster has the id:
cluster_id = self._node_to_cluster[node["id"]].cluster_id if node["id"] in self._node_to_cluster else None
# and pass cluster_id=cluster_id to SearchResult(...).
```

Imports: add `from groundloop.kb_indexer.models import Cluster, ClusterManifest, SearchResult` to the top of `index.py` (SearchResult already imported).

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m1): SkillsIndex cluster attach/lookup/nodes_in_cluster`

---

## Task 4: CLI `cluster` subcommand

**Files:** `groundloop/kb_indexer/cli.py` (extend), `tests/groundloop/kb_indexer/test_cli.py` (extend).

- [ ] Write failing test:
```python
def test_cli_cluster(tiny_corpus_path, tmp_path, capsys):
    manifest_path = tmp_path / "cluster_manifest.json"
    rc = main([
        "cluster",
        "--corpus", str(tiny_corpus_path),
        "--manifest", str(manifest_path),
        "--threshold", "0.1",
    ])
    assert rc == 0
    assert manifest_path.exists()
    import json
    m = json.loads(manifest_path.read_text())
    assert m["total_nodes_clustered"] >= 1
    assert m["corpus_sha256"]


def test_cli_cluster_missing_corpus(tmp_path):
    rc = main([
        "cluster",
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--manifest", str(tmp_path / "m.json"),
    ])
    assert rc == 1
```

- [ ] Extend `cli.py`:
```python
# Add new subcommand in _parse():
cluster = sub.add_parser("cluster", help="Build cluster manifest from corpus")
_add_common(cluster)
cluster.add_argument("--manifest", type=Path, default=Path("groundloop/kb/cluster_manifest.json"))
cluster.add_argument("--threshold", type=float, default=0.15)

# Add new command handler:

def _cmd_cluster(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    import hashlib
    import json as _json
    nodes = [
        _json.loads(ln)
        for ln in args.corpus.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    corpus_sha256 = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    from groundloop.kb_indexer.cluster import build_clusters, save_manifest
    manifest = build_clusters(
        nodes, jaccard_threshold=args.threshold, corpus_sha256=corpus_sha256,
    )
    save_manifest(manifest, args.manifest)
    print(f"built: {manifest.total_clusters} clusters, "
          f"{manifest.total_nodes_clustered} nodes, "
          f"{manifest.singletons} singletons, "
          f"threshold={args.threshold}")
    return 0

# In main(), route:
if args.command == "cluster":
    return _cmd_cluster(args)
```

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m1): CLI cluster subcommand`

---

## Task 5: Build on real corpus + update README

- [ ] Run: `python3 -m groundloop.kb_indexer cluster` — should complete in <10s on 1006-node corpus. Record output (num clusters, num singletons) for the README.
- [ ] Verify determinism: run twice, diff the `cluster_manifest.json` files — must be byte-identical.
- [ ] Append a short `### Cluster Manifest` subsection to `README.md` (under the existing `## GroundLoop Skills Scraper` or `### KB Indexer` block) describing:
  - What the cluster manifest is.
  - How to build: `python3 -m groundloop.kb_indexer cluster`.
  - The number of clusters generated on the current corpus (from the live run).
- [ ] Run full test suite + ruff + mypy on `groundloop/kb_indexer/`. Expect all clean.
- [ ] Commit: `docs(codeforge-m1): README entry + cluster manifest smoke-tested on real corpus`

---

## Self-Review

- ✅ Every spec §5 API has an implementing task.
- ✅ Spec §7 acceptance criteria 1–7 covered: (1) real-corpus smoke Task 5, (2,3) build_clusters + sha stability tested Task 2, (4) partition check Task 2, (5) nodes_in_cluster Task 3, (6) lint final Task 5, (7) coverage ≥85%.
- ✅ No placeholders.
- ✅ Type consistency: Cluster fields are identical across models.py / cluster.py / index.py / cli.py.
