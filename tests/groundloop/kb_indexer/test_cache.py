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
