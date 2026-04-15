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
    import pickle
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    payload["sha256"] = "DIFFERENT"
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)
    assert SkillsIndex.load(corpus_path=tiny_corpus_path, cache_path=cache_path) is None


def test_tag_filter_no_matches_returns_empty(built_index: SkillsIndex) -> None:
    assert built_index.search("python", required_tags={"domain:nonexistent"}) == []
