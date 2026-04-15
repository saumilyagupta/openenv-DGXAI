# GroundLoop KB-Indexer Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a deterministic BM25 index over the SkillNode corpus produced by skills-scraper, with tag-filter queries, persistence, and a CLI.

**Architecture:** 6-module package (`tokenizer`, `models`, `cache`, `index`, `cli`, `__main__`). BM25 via `rank_bm25` (already pinned). Cache as pickle with corpus sha256 for invalidation.

**Tech Stack:** Python 3.11+, Pydantic v2, `rank_bm25`, pickle, pytest.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-kb-indexer-design.md`.

---

## File Structure

```
groundloop/kb_indexer/
  __init__.py
  __main__.py
  tokenizer.py
  models.py
  cache.py
  index.py
  cli.py
tests/groundloop/kb_indexer/
  __init__.py
  conftest.py
  fixtures/
    tiny_corpus.jsonl          # 5 synthetic SkillNodes for fast tests
    tiny_manifest.json
  test_tokenizer.py
  test_models.py
  test_cache.py
  test_index.py
  test_cli.py
  test_e2e.py
```

---

## Task 1: Scaffold

**Files:** `groundloop/kb_indexer/__init__.py`, `tests/groundloop/kb_indexer/__init__.py`, `tests/groundloop/kb_indexer/conftest.py`

- [ ] Create dirs and empty `__init__.py` files.
- [ ] Create `conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_corpus_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "tiny_corpus.jsonl"
```

- [ ] Commit: `chore: scaffold groundloop/kb_indexer package`

---

## Task 2: Tokenizer

**Files:** `groundloop/kb_indexer/tokenizer.py`, `tests/groundloop/kb_indexer/test_tokenizer.py`

- [ ] Write failing tests:

```python
from groundloop.kb_indexer.tokenizer import tokenize


def test_tokenize_lowercase():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_splits_on_punctuation():
    assert tokenize("foo-bar,baz.qux") == ["foo", "bar", "baz", "qux"]


def test_tokenize_drops_empty_and_numbers_kept():
    assert tokenize("pytest 3.11") == ["pytest", "3", "11"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_only_punctuation():
    assert tokenize("!!!---???") == []


def test_tokenize_unicode():
    assert tokenize("café résumé") == ["café", "résumé"]


def test_tokenize_deterministic():
    a = tokenize("The quick brown fox")
    b = tokenize("The quick brown fox")
    assert a == b
```

- [ ] Run: expect FAIL.
- [ ] Implement:

```python
from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if t]
```

- [ ] Run: expect PASS (7 tests).
- [ ] Commit: `feat(kb-indexer): unicode-aware tokenizer`

---

## Task 3: SearchResult model

**Files:** `groundloop/kb_indexer/models.py`, `tests/groundloop/kb_indexer/test_models.py`

- [ ] Write failing tests:

```python
import pytest
from pydantic import ValidationError

from groundloop.kb_indexer.models import SearchResult


def test_search_result_frozen():
    r = SearchResult(
        node_id="abc",
        skill_name="python-testing",
        section_path=("Fixtures",),
        section_body="body",
        tags=("domain:python",),
        source_path="/p",
        score=1.23,
        rank=1,
    )
    with pytest.raises(ValidationError):
        r.rank = 2  # type: ignore[misc]


def test_search_result_requires_fields():
    with pytest.raises(ValidationError):
        SearchResult()  # type: ignore[call-arg]
```

- [ ] Run: expect FAIL.
- [ ] Implement:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    node_id: str
    skill_name: str
    section_path: tuple[str, ...]
    section_body: str
    tags: tuple[str, ...]
    source_path: str
    score: float
    rank: int
```

- [ ] Run: expect PASS.
- [ ] Commit: `feat(kb-indexer): SearchResult model`

---

## Task 4: Fixture corpus

**Files:** `tests/groundloop/kb_indexer/fixtures/tiny_corpus.jsonl`, `tiny_manifest.json`

- [ ] Create `tiny_corpus.jsonl` with 5 synthetic SkillNodes (one line each, JSON) — exactly this content:

```
{"id":"a0000000000000a1","skill_name":"python-testing","skill_description":"pytest","skill_type":"flexible","section_path":["Fixtures"],"section_title":"Fixtures","section_body":"Use pytest fixtures for setup and teardown. Scope can be function module or session.","source_path":"/fake/python-testing/SKILL.md","source_root":"user-skills","tags":["domain:python","phase:test"],"trigger_hints":"pytest","mtime":0.0,"body_hash":"h1","alias_sources":[]}
{"id":"b0000000000000b2","skill_name":"api-design","skill_description":"apis","skill_type":null,"section_path":["Endpoints"],"section_title":"Endpoints","section_body":"Design REST endpoints with clear resource naming and proper HTTP verbs.","source_path":"/fake/api-design/SKILL.md","source_root":"user-skills","tags":["domain:api","phase:plan"],"trigger_hints":"api","mtime":0.0,"body_hash":"h2","alias_sources":[]}
{"id":"c0000000000000c3","skill_name":"security-review","skill_description":"security","skill_type":"rigid","section_path":["Checklist"],"section_title":"Checklist","section_body":"Check authentication authorization input validation secrets and injection risks.","source_path":"/fake/security-review/SKILL.md","source_root":"user-skills","tags":["domain:security","phase:review"],"trigger_hints":"security","mtime":0.0,"body_hash":"h3","alias_sources":[]}
{"id":"d0000000000000d4","skill_name":"python-testing","skill_description":"pytest","skill_type":"flexible","section_path":["Parametrization"],"section_title":"Parametrization","section_body":"Use pytest mark parametrize to run a test with multiple inputs.","source_path":"/fake/python-testing/SKILL.md","source_root":"user-skills","tags":["domain:python","phase:test"],"trigger_hints":"pytest","mtime":0.0,"body_hash":"h4","alias_sources":[]}
{"id":"e0000000000000e5","skill_name":"backend-patterns","skill_description":"backend","skill_type":"flexible","section_path":["Layers"],"section_title":"Layers","section_body":"Separate routes from services from repositories to enable testing and reuse.","source_path":"/fake/backend-patterns/SKILL.md","source_root":"user-skills","tags":["domain:backend","phase:plan"],"trigger_hints":"backend","mtime":0.0,"body_hash":"h5","alias_sources":[]}
```

- [ ] Create `tiny_manifest.json`:

```json
{
  "generated_at": "2026-04-15T00:00:00+00:00",
  "sources": [{"label": "fake", "glob": "/fake/**/SKILL.md"}],
  "scraped_files": 4,
  "skipped_files": 0,
  "total_nodes": 5,
  "errors": [],
  "corpus_sha256": "will_be_set_in_test"
}
```

- [ ] Commit: `test(kb-indexer): tiny fixture corpus`

---

## Task 5: Cache module

**Files:** `groundloop/kb_indexer/cache.py`, `tests/groundloop/kb_indexer/test_cache.py`

- [ ] Write failing tests:

```python
from pathlib import Path

from groundloop.kb_indexer.cache import load_cache, save_cache


def test_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "c.pkl"
    state = {"tokenized": [["a", "b"]], "node_index": [{"id": "x"}], "built_at": "2026-04-15T00:00:00+00:00"}
    save_cache(state, cache_path, "sha1")
    loaded = load_cache(cache_path, "sha1")
    assert loaded is not None
    assert loaded["tokenized"] == [["a", "b"]]


def test_cache_sha_mismatch_returns_none(tmp_path: Path):
    cache_path = tmp_path / "c.pkl"
    save_cache({"tokenized": [], "node_index": [], "built_at": "t"}, cache_path, "sha1")
    assert load_cache(cache_path, "other-sha") is None


def test_cache_missing_returns_none(tmp_path: Path):
    assert load_cache(tmp_path / "none.pkl", "sha") is None


def test_cache_corrupt_returns_none(tmp_path: Path):
    cache_path = tmp_path / "c.pkl"
    cache_path.write_bytes(b"not a pickle")
    assert load_cache(cache_path, "sha") is None
```

- [ ] Run: expect FAIL.
- [ ] Implement:

```python
from __future__ import annotations

import logging
import pickle
from pathlib import Path

_log = logging.getLogger(__name__)


def save_cache(state: dict, cache_path: Path, corpus_sha256: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sha256": corpus_sha256, **state}
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(cache_path: Path, expected_sha256: str) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        _log.warning("cache: corrupt or unreadable %s: %s", cache_path, e)
        return None
    if payload.get("sha256") != expected_sha256:
        return None
    return payload
```

- [ ] Run: expect PASS.
- [ ] Commit: `feat(kb-indexer): pickle cache with sha256 invalidation`

---

## Task 6: SkillsIndex core

**Files:** `groundloop/kb_indexer/index.py`, `tests/groundloop/kb_indexer/test_index.py`

- [ ] Write failing tests:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from groundloop.kb_indexer.index import SkillsIndex


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def built_index(tiny_corpus_path: Path, tmp_path: Path) -> SkillsIndex:
    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=tmp_path / "c.pkl")
    idx.build()
    return idx


def test_build_populates_nodes(built_index: SkillsIndex) -> None:
    assert built_index.stats()["node_count"] == 5


def test_search_pytest_fixtures(built_index: SkillsIndex) -> None:
    results = built_index.search("pytest fixtures")
    assert results[0].skill_name == "python-testing"
    assert results[0].rank == 1


def test_search_tag_filter(built_index: SkillsIndex) -> None:
    results = built_index.search("testing", required_tags={"domain:security"})
    assert all("domain:security" in r.tags for r in results)


def test_search_empty_query_returns_empty(built_index: SkillsIndex) -> None:
    assert built_index.search("") == []


def test_search_deterministic_ordering(built_index: SkillsIndex) -> None:
    a = built_index.search("python testing")
    b = built_index.search("python testing")
    assert [r.node_id for r in a] == [r.node_id for r in b]


def test_save_and_load_cache(tiny_corpus_path: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "c.pkl"
    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=cache_path)
    idx.build()
    idx.save()
    # Cache now exists and sha matches
    expected = _sha256_of(tiny_corpus_path)
    loaded = SkillsIndex.load(corpus_path=tiny_corpus_path, cache_path=cache_path)
    assert loaded is not None
    assert loaded.stats()["node_count"] == 5
    assert loaded._corpus_sha256 == expected


def test_load_returns_none_when_corpus_changed(tiny_corpus_path: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "c.pkl"
    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=cache_path)
    idx.build()
    idx.save()
    # Simulate corpus change by writing a bogus cache sha
    import pickle
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    payload["sha256"] = "DIFFERENT"
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)
    assert SkillsIndex.load(corpus_path=tiny_corpus_path, cache_path=cache_path) is None


def test_tag_filter_no_matches_returns_empty(built_index: SkillsIndex) -> None:
    assert built_index.search("python", required_tags={"domain:nonexistent"}) == []
```

- [ ] Run: expect FAIL.
- [ ] Implement:

```python
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
        candidates = range(len(self._nodes))
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
```

- [ ] Run: expect PASS (8 tests).
- [ ] Commit: `feat(kb-indexer): SkillsIndex build/save/load/search`

---

## Task 7: CLI

**Files:** `groundloop/kb_indexer/cli.py`, `groundloop/kb_indexer/__main__.py`, `tests/groundloop/kb_indexer/test_cli.py`

- [ ] Write failing tests:

```python
from __future__ import annotations

import json
from pathlib import Path

from groundloop.kb_indexer.cli import main


def test_cli_build(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    rc = main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    assert rc == 0
    assert cache.exists()


def test_cli_build_missing_corpus_exits_1(tmp_path: Path) -> None:
    rc = main(["build", "--corpus", str(tmp_path / "none.jsonl"), "--cache", str(tmp_path / "c.pkl")])
    assert rc == 1


def test_cli_search_json(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main([
        "search", "pytest fixtures",
        "--corpus", str(tiny_corpus_path),
        "--cache", str(cache),
        "--format", "json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) >= 1
    assert data[0]["skill_name"] == "python-testing"


def test_cli_stats(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main(["stats", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "node_count" in out


def test_cli_search_with_tag_filter(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main([
        "search", "testing",
        "--corpus", str(tiny_corpus_path),
        "--cache", str(cache),
        "--tag", "domain:security",
        "--format", "json",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    for item in data:
        assert "domain:security" in item["tags"]
```

- [ ] Run: expect FAIL.
- [ ] Implement `cli.py`:

```python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex

_DEFAULT_CORPUS = Path("groundloop/kb/skills_corpus.jsonl")
_DEFAULT_CACHE = Path("groundloop/kb/skills_index.pkl")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.kb_indexer")
    sub = p.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build + persist the index")
    _add_common(build)
    build.add_argument("--force", action="store_true")

    search = sub.add_parser("search", help="Search the index")
    _add_common(search)
    search.add_argument("query", type=str)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--format", choices=("text", "json"), default="text")

    stats = sub.add_parser("stats", help="Show index stats")
    _add_common(stats)

    return p.parse_args(argv)


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    cached = None if args.force else SkillsIndex.load(corpus_path=args.corpus, cache_path=args.cache)
    if cached is not None:
        print("cache hit")
        return 0
    idx = SkillsIndex(corpus_path=args.corpus, cache_path=args.cache)
    idx.build()
    idx.save()
    s = idx.stats()
    print(f"built: {s['node_count']} nodes, {s['vocab_size']} vocab, avg {s['avg_doc_len']:.1f} toks/doc")
    return 0


def _load_or_build(args: argparse.Namespace) -> SkillsIndex | None:
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return None
    idx = SkillsIndex.load(corpus_path=args.corpus, cache_path=args.cache)
    if idx is None:
        idx = SkillsIndex(corpus_path=args.corpus, cache_path=args.cache)
        idx.build()
        idx.save()
    return idx


def _cmd_search(args: argparse.Namespace) -> int:
    idx = _load_or_build(args)
    if idx is None:
        return 1
    tags = set(args.tag) if args.tag else None
    results = idx.search(args.query, top_k=args.top_k, required_tags=tags)
    if args.format == "json":
        payload = [r.model_dump() for r in results]
        # tuples serialize as lists; fix for clarity
        for p in payload:
            p["section_path"] = list(p["section_path"])
            p["tags"] = list(p["tags"])
        print(json.dumps(payload))
    else:
        for r in results:
            print(f"[{r.rank}] score={r.score:.3f} {r.skill_name}/{'/'.join(r.section_path)}")
            print(f"    tags={','.join(r.tags)}")
            print(f"    {r.section_body[:140]}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    idx = _load_or_build(args)
    if idx is None:
        return 1
    s = idx.stats()
    print(json.dumps(s))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "stats":
        return _cmd_stats(args)
    return 1
```

- [ ] `__main__.py`:

```python
from groundloop.kb_indexer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Run: expect PASS (5 tests).
- [ ] Commit: `feat(kb-indexer): CLI with build/search/stats subcommands`

---

## Task 8: E2E + public API

**Files:** `tests/groundloop/kb_indexer/test_e2e.py`, `groundloop/kb_indexer/__init__.py`

- [ ] Write e2e test:

```python
from __future__ import annotations

from pathlib import Path

from groundloop.kb_indexer import SearchResult, SkillsIndex


def test_e2e_build_save_reload_search(tiny_corpus_path: Path, tmp_path: Path) -> None:
    cache = tmp_path / "idx.pkl"
    idx = SkillsIndex(corpus_path=tiny_corpus_path, cache_path=cache)
    idx.build()
    idx.save()
    reloaded = SkillsIndex.load(corpus_path=tiny_corpus_path, cache_path=cache)
    assert reloaded is not None
    r1 = idx.search("pytest fixtures")
    r2 = reloaded.search("pytest fixtures")
    assert [r.node_id for r in r1] == [r.node_id for r in r2]
    assert all(isinstance(r, SearchResult) for r in r2)
```

- [ ] Populate `__init__.py`:

```python
from __future__ import annotations

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.kb_indexer.models import SearchResult

__all__ = ["SearchResult", "SkillsIndex"]
```

- [ ] Run full suite with coverage:

```
python3 -m pytest tests/groundloop/kb_indexer/ -v --cov=groundloop.kb_indexer --cov-report=term
ruff check groundloop/kb_indexer/
mypy --strict groundloop/kb_indexer/
```

Expected: all pass, coverage ≥ 90%, ruff + mypy clean.

- [ ] Commit: `feat(kb-indexer): public API + e2e test`

---

## Task 9: Real-system smoke test + README

**Files:** `README.md`

- [ ] Build against real corpus:

```
python3 -m groundloop.kb_indexer build
```

Expected: `built: 1006 nodes, ~XX vocab, avg ~YY toks/doc`.

- [ ] Run sample search:

```
python3 -m groundloop.kb_indexer search "pytest fixtures" --top-k 3
python3 -m groundloop.kb_indexer search "api design" --tag domain:api --top-k 3
```

Expected: sensible results; at least one result's `skill_name` matches `python-testing` / `api-design` / similar.

- [ ] Append a `### KB Indexer` subsection to the existing `## GroundLoop Skills Scraper` README block explaining the three commands.

- [ ] Commit: `docs: README entry for kb-indexer; smoke-tested on real corpus`

---

## Self-Review

- ✅ Every spec §4 component has an implementing task.
- ✅ Acceptance criteria §8.1–§8.8 covered by CLI smoke (§8.1, §8.4), search test (§8.2, §8.3), deterministic-ordering test (§8.8), coverage + lint tasks (§8.6, §8.7), e2e test (§8.5).
- ✅ No placeholders.
- ✅ Type consistency: `SearchResult` fields identical in models.py (Task 3), index.py (Task 6), and CLI JSON output (Task 7).
