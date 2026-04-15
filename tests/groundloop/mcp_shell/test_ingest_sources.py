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
