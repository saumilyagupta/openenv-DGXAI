# CodeForge M1 — Graphify Clustering

**Date:** 2026-04-15
**Module:** M1 of the fully-wired CodeForge (per CLAUDE.md §2).
**Depends on:** `groundloop/kb_indexer/` (shipped).
**Consumed by:** M3 (`query_cluster` action), M6 (AuditLedger cluster-coverage metric).

---

## 1. Purpose

Turn the flat 1006-node Layer B corpus into a graph with **connected-component communities**. Each node gets a `cluster_id`; each cluster gets a human-readable `label`. Agents will query clusters (via M3) and the audit will track which communities an episode touched.

This is Pillar 3 of CodeForge's philosophical lineage (graphify-style structured KB).

## 2. Scope

**In scope:**

- `groundloop/kb_indexer/cluster.py` — new module. Pure-Python clustering over tokenized skill nodes.
- `cluster_manifest.json` — new output artifact alongside `skills_corpus.jsonl` in `groundloop/kb/`.
- Extend `SearchResult` (models.py) with `cluster_id: str | None = None`.
- Extend `SkillsIndex` with `cluster_id_for(node_id) -> str | None` and `nodes_in_cluster(cluster_label) -> list[SearchResult]`.
- CLI subcommand: `python -m groundloop.kb_indexer cluster` builds the manifest from the current corpus + saved index cache.
- Tests.

**Out of scope:**

- LLM-based labeling (we use deterministic top-tokens).
- Hierarchical clustering / dendrograms.
- Vector embeddings / cosine similarity (Jaccard-on-tokens is sufficient for the existing corpus).
- Live re-clustering on corpus change — only runs when CLI invoked.

## 3. Algorithm

1. **Tokenize** every node body with the existing `groundloop.kb_indexer.tokenizer.tokenize`.
2. **Filter**: drop tokens of length <3 and stopwords (tiny hardcoded list). This is to prevent "the", "and" from dominating similarity.
3. **Similarity**: pairwise Jaccard over token sets. Threshold `JACCARD_THRESHOLD = 0.15` → edge.
4. **Graph build**: 1006 nodes, edges where similarity ≥ threshold. Store as adjacency dict.
5. **Connected components** via BFS.
6. **Label a cluster** by:
   - Top-3 most frequent tokens across its member nodes (excluding stopwords).
   - Dominant `domain` tag (mode of member tag lists).
   - Label string: `f"{domain}_{top3_tokens_joined_by_underscore}"` (e.g. `python_testing_pytest_fixtures`).
7. **Cluster ID**: `sha256(sorted(member_node_ids))[:12]` — deterministic across runs.

## 4. Output

### `cluster_manifest.json`

```json
{
  "generated_at": "2026-04-15T11:30:00+00:00",
  "corpus_sha256": "<same as skills_corpus.manifest.json>",
  "jaccard_threshold": 0.15,
  "total_clusters": 42,
  "total_nodes_clustered": 1006,
  "singletons": 34,
  "clusters": [
    {
      "cluster_id": "abc123def456",
      "label": "python_testing_pytest_fixtures",
      "dominant_domain": "python",
      "top_tokens": ["pytest", "fixture", "test"],
      "node_count": 47,
      "member_node_ids": ["node_id_1", "node_id_2", ...]
    },
    ...
  ]
}
```

## 5. API surface

### 5.1 `cluster.py` (new module)

```python
def build_clusters(
    nodes: list[SkillNode],
    *,
    jaccard_threshold: float = 0.15,
    min_token_length: int = 3,
) -> ClusterManifest: ...

def save_manifest(manifest: ClusterManifest, out_path: Path) -> Path: ...

def load_manifest(path: Path) -> ClusterManifest: ...
```

### 5.2 `models.py` additions

```python
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


class SearchResult(BaseModel):
    # existing fields unchanged
    ...
    cluster_id: str | None = None   # NEW — populated when index is cluster-aware
```

### 5.3 `SkillsIndex` additions

```python
def attach_cluster_manifest(self, manifest: ClusterManifest) -> None: ...
def cluster_id_for(self, node_id: str) -> str | None: ...
def cluster_by_label(self, label: str) -> Cluster | None: ...
def nodes_in_cluster(self, cluster_label: str, top_k: int = 50) -> list[SearchResult]: ...
```

### 5.4 CLI

```
python -m groundloop.kb_indexer cluster
    [--corpus groundloop/kb/skills_corpus.jsonl]
    [--cache groundloop/kb/skills_index.pkl]
    [--manifest groundloop/kb/cluster_manifest.json]
    [--threshold 0.15]
```

Builds the manifest. Exit 0 on success, 1 on missing corpus.

## 6. Testing

Coverage target: **85%**.

- `test_cluster_jaccard.py` — Jaccard on simple token sets (identity=1.0, disjoint=0.0, partial).
- `test_cluster_build.py` — tiny synthetic corpus (5 nodes), verify component count + labels.
- `test_cluster_manifest_roundtrip.py` — save + load, sha256 stable.
- `test_cluster_on_real_corpus.py` — build on `tests/groundloop/kb_indexer/fixtures/tiny_corpus.jsonl` (5 nodes), assert deterministic output across 2 runs.
- `test_skills_index_cluster_helpers.py` — `cluster_id_for`, `nodes_in_cluster`.
- `test_cli_cluster.py` — CLI exit codes + manifest emission.

## 7. Acceptance

1. `python3 -m groundloop.kb_indexer cluster` completes in <10s on the 1006-node real corpus.
2. `cluster_manifest.json` parses back into `ClusterManifest` and carries `corpus_sha256` matching `skills_corpus.manifest.json`.
3. Running twice on unchanged corpus → byte-identical manifest.
4. Every node appears in exactly one cluster (could be singleton).
5. `SkillsIndex.nodes_in_cluster("python_testing_...")` returns at least one node with `cluster_id` set.
6. `ruff check groundloop/kb_indexer/` + `mypy --strict groundloop/kb_indexer/` clean.
7. Coverage ≥ 85% on the new code.

## 8. Deliverables

- `groundloop/kb_indexer/cluster.py` (new).
- `groundloop/kb_indexer/models.py` — `Cluster`, `ClusterManifest` added; `SearchResult.cluster_id` added.
- `groundloop/kb_indexer/index.py` — `attach_cluster_manifest`, `cluster_id_for`, `cluster_by_label`, `nodes_in_cluster`.
- `groundloop/kb_indexer/cli.py` — new `cluster` subcommand.
- Tests.
- `groundloop/kb/cluster_manifest.json` (generated artifact; gitignored).
- No new dependencies.
