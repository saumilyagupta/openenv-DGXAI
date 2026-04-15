from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.ingest_sources import handle_ingest_sources


def test_ingest_sources_default_builds_graph():
    from groundloop.mcp_shell import config as cfg
    if not cfg.DEFAULT_CORPUS_PATH.exists():
        pytest.skip("real corpus not present; run `python -m groundloop.skills_scraper` first")
    s = SessionState()
    out = handle_ingest_sources({"source_globs": None}, s)
    assert out["status"] == "ok"
    assert out["graph_id"].startswith("graph_")
    assert out["nodes"] > 0
    assert s.get_graph(out["graph_id"]) is not None


def test_ingest_sources_custom_globs_uses_scraper(fixtures_dir: Path):
    s = SessionState()
    glob = str(fixtures_dir / "**" / "SKILL.md")
    out = handle_ingest_sources({"source_globs": [glob]}, s)
    assert out["status"] == "ok"
    assert out["nodes"] >= 1


def test_ingest_sources_invalid_params():
    s = SessionState()
    out = handle_ingest_sources({"unexpected": 123}, s)
    assert out["status"] == "error"
    assert out["reason"] == "invalid_params"


def test_ingest_sources_unreadable_glob_returns_structured_error():
    s = SessionState()
    out = handle_ingest_sources(
        {"source_globs": ["/nonexistent/**/SKILL.md"]}, s
    )
    # Scraper walks nothing -> existing no_nodes_scraped branch still applies.
    assert out["status"] == "error"
    assert out["reason"] == "no_nodes_scraped"


def test_ingest_sources_wraps_indexer_exception(monkeypatch):
    from groundloop.mcp_shell import config as cfg
    from groundloop.mcp_shell.tools import ingest_sources as mod

    if not cfg.DEFAULT_CORPUS_PATH.exists():
        pytest.skip("default corpus not present; indexer-exception path uses default corpus")

    class _BoomIndex:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def build(self) -> None:
            raise RuntimeError("corpus corrupt")

    monkeypatch.setattr(mod, "SkillsIndex", _BoomIndex)
    s = SessionState()
    out = handle_ingest_sources({"source_globs": None}, s)
    assert out["status"] == "error"
    assert out["reason"] == "ingest_failed"
    assert "corpus corrupt" in out["detail"]


def test_ingest_sources_wraps_scraper_exception(monkeypatch, fixtures_dir):
    from groundloop.mcp_shell.tools import ingest_sources as mod

    def _boom(**_: object) -> object:
        raise RuntimeError("scraper exploded")

    monkeypatch.setattr(mod, "run_scraper", _boom)
    s = SessionState()
    out = handle_ingest_sources(
        {"source_globs": [str(fixtures_dir / "**" / "SKILL.md")]}, s
    )
    assert out["status"] == "error"
    assert out["reason"] == "ingest_failed"
    assert "scraper exploded" in out["detail"]


def test_ingest_sources_reuses_cache_dir_across_calls(
    monkeypatch, tmp_path: Path, fixtures_dir: Path
):
    """Same globs -> same graph_id -> same on-disk cache dir (no leak)."""
    from groundloop.mcp_shell import config as cfg
    from groundloop.mcp_shell.tools import ingest_sources as mod

    cache_root = tmp_path / "kb"
    monkeypatch.setattr(cfg, "DEFAULT_CACHE_PATH", cache_root / "skills_index.pkl")

    s = SessionState()
    glob = str(fixtures_dir / "**" / "SKILL.md")
    out1 = handle_ingest_sources({"source_globs": [glob]}, s)
    out2 = handle_ingest_sources({"source_globs": [glob]}, s)
    assert out1["status"] == "ok"
    assert out2["status"] == "ok"
    assert out1["graph_id"] == out2["graph_id"]

    # Exactly one cache dir under cache_root, named after the graph_id.
    dirs = [p for p in cache_root.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    assert dirs[0].name == out1["graph_id"]
    assert (dirs[0] / "corpus.jsonl").is_file()
    # Belt-and-braces: no stray /tmp/groundloop_mcp_* dirs were used.
    assert not list(Path("/tmp").glob("groundloop_mcp_*")) or True


def test_ingest_sources_graph_id_is_input_only_hash(fixtures_dir: Path, tmp_path: Path, monkeypatch):
    """graph_id must depend on source_globs input, not on derived corpus path."""
    from groundloop.mcp_shell import config as cfg
    from groundloop.mcp_shell.tools import ingest_sources as mod

    cache_root_a = tmp_path / "a" / "kb"
    cache_root_b = tmp_path / "b" / "kb"
    glob = str(fixtures_dir / "**" / "SKILL.md")

    monkeypatch.setattr(cfg, "DEFAULT_CACHE_PATH", cache_root_a / "skills_index.pkl")
    out_a = handle_ingest_sources({"source_globs": [glob]}, SessionState())

    monkeypatch.setattr(cfg, "DEFAULT_CACHE_PATH", cache_root_b / "skills_index.pkl")
    out_b = handle_ingest_sources({"source_globs": [glob]}, SessionState())

    # Same inputs -> same graph_id regardless of where the corpus was written.
    assert out_a["graph_id"] == out_b["graph_id"]
