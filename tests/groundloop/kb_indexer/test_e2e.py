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
