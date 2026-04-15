# GroundLoop KB-Indexer — Design Spec

**Date:** 2026-04-15
**Sub-project:** #3 of 8 in the GroundLoop MCP decomposition
**Depends on:** skills-scraper (#2, shipped) — consumes its JSONL corpus.
**Consumed by:** ralph-orchestrator (#7), lib-grounder (#4), interrogator (#5)

---

## 1. Purpose

Index the skill-node corpus produced by skills-scraper and expose a fast, deterministic search API. Every downstream GroundLoop component queries this index to retrieve the most relevant reasoning-policy sections for its current decision.

## 2. Scope

**In scope:**

- Load the scraper's JSONL corpus + manifest.
- Build a BM25 index over section bodies.
- Persist the index to disk, cache-invalidate via corpus sha256.
- Expose a `search(query, top_k, required_tags)` API.
- CLI: `build`, `search`, `stats` subcommands.

**Out of scope:**

- Vector / dense embeddings (future phase).
- Query rewriting / LLM reranking.
- Indexing anything other than SkillNode (lib-grounder will carry its own index).
- Network / remote corpora.

## 3. Architecture

```
groundloop/kb_indexer/
  __init__.py           # re-exports SkillsIndex, SearchResult
  __main__.py           # entrypoint
  tokenizer.py          # tokenize(text) -> list[str]
  index.py              # SkillsIndex: build, save, load, search
  models.py             # SearchResult (Pydantic v2 frozen)
  cache.py              # load/save pickle with sha256 invalidation
  cli.py                # build/search/stats subcommands
```

File size: each module ≤ 200 lines.

## 4. Component Contracts

### 4.1 `tokenizer.py`

```python
def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
```

Deterministic. Same output for same input every run.

### 4.2 `models.py`

```python
class SearchResult(BaseModel, frozen=True):
    node_id: str             # SkillNode.id
    skill_name: str
    section_path: tuple[str, ...]
    section_body: str
    tags: tuple[str, ...]
    source_path: str
    score: float             # BM25 raw score
    rank: int                # 1-based rank in result set
```

### 4.3 `index.py`

```python
class SkillsIndex:
    def __init__(self, corpus_path: Path, cache_path: Path | None = None): ...
    def build(self) -> None: ...                                 # tokenize + BM25
    def save(self) -> None: ...                                   # pickle to cache
    @classmethod
    def load(cls, corpus_path: Path, cache_path: Path) -> "SkillsIndex": ...
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        required_tags: set[str] | None = None,
    ) -> list[SearchResult]: ...
    def stats(self) -> dict[str, int | float]: ...               # node_count, vocab_size, avg_doc_len
```

Search algorithm:
1. Tokenize query.
2. If `required_tags` set, pre-filter candidate node indices to those whose tags ⊇ required_tags.
3. Compute BM25 scores over candidates only (pass filtered subset to BM25Okapi).
4. Sort by `(-score, node_id)` for deterministic tie-breaking.
5. Take top_k. Attach 1-based rank. Return `list[SearchResult]`.

### 4.4 `cache.py`

```python
def save_cache(index_state: dict, cache_path: Path, corpus_sha256: str) -> None: ...
def load_cache(cache_path: Path, expected_sha256: str) -> dict | None: ...
```

Cache layout (pickle): `{"sha256": str, "tokenized": list[list[str]], "node_index": list[dict], "built_at": str}`. `load_cache` returns `None` if sha256 mismatch or file missing.

### 4.5 `cli.py`

```
python -m groundloop.kb_indexer build
    [--corpus groundloop/kb/skills_corpus.jsonl]
    [--cache groundloop/kb/skills_index.pkl]
    [--force]

python -m groundloop.kb_indexer search <query>
    [--top-k 5]
    [--tag tag1] [--tag tag2]
    [--format json|text]

python -m groundloop.kb_indexer stats
```

`build` exits 0 on success, 1 on missing corpus. `search` exits 0 even on empty results. `stats` exits 0 always.

## 5. Data Flow

```
skills_corpus.jsonl + manifest.sha256
    └─> SkillsIndex.build()
         ├─> for each node: tokenize(body) → tokens[]
         ├─> BM25Okapi(list[tokens])
         └─> cache.save_cache(..., sha256)

search(query, top_k, required_tags)
    ├─> tokenize(query)
    ├─> filter_candidates(required_tags)
    ├─> bm25.get_scores(tokens) restricted to candidates
    ├─> sort by (-score, node_id)
    └─> top_k → [SearchResult, ...]
```

## 6. Error Handling

| Condition | Action |
|---|---|
| Corpus missing | CLI exit 1 with message |
| Cache stale (sha256 mismatch) | Silent rebuild |
| Cache corrupt | Silent rebuild, WARN log |
| Empty corpus | Build succeeds, `search` returns `[]` |
| `required_tags` with no matches | Return `[]` |
| Query produces zero tokens | Return `[]` |

## 7. Testing

Coverage target: **90%**.

Test files:
- `test_tokenizer.py` — deterministic splits, edge cases (unicode, punctuation, empty).
- `test_models.py` — SearchResult frozen, rank field behavior.
- `test_cache.py` — save/load roundtrip, sha256 invalidation, corrupt file handling.
- `test_index.py` — build, search top-k ranking, tag filter correctness, deterministic tie-break, stats.
- `test_cli.py` — build/search/stats exit codes + output parsing.
- `test_e2e.py` — against a fixture corpus (5-10 nodes), round-trip build → save → reload → search.

## 8. Acceptance Criteria

1. `python -m groundloop.kb_indexer build` completes in < 3 seconds on the 1006-node real corpus.
2. First `search "pytest fixtures"` returns a result set with at least one `skill_name="python-testing"` node in top 5.
3. `search "api design"` with `--tag domain:api` returns only `domain:api` nodes.
4. Re-running `build` without `--force` uses the cache (stdout: "cache hit").
5. Search latency < 10ms on the 1006-node corpus (warm cache).
6. `ruff check` clean, `mypy --strict` clean.
7. 90% line coverage on `groundloop/kb_indexer/`.
8. Two consecutive `search <same-query>` calls return byte-identical ordered results.

## 9. Dependencies

Already present: `rank_bm25==0.2.2`, `pydantic>=2`, `pytest`, `pytest-cov`.

Nothing new to add.

## 10. Deliverables

1. `groundloop/kb_indexer/` package with 6 modules.
2. `tests/groundloop/kb_indexer/` with fixtures + test suite.
3. `groundloop/kb/skills_index.pkl` (generated, gitignored).
4. README addendum explaining `build`, `search`, `stats` usage.
