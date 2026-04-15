from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent.parent / "skills_scraper" / "fixtures" / "fake_skills"


@pytest.fixture
def tiny_corpus_path() -> Path:
    return Path(__file__).parent.parent / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"


@pytest.fixture(autouse=True)
def _isolate_mcp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect mcp_shell's DEFAULT_CACHE_PATH to tmp_path so custom-globs
    tests don't write graph dirs into the real `groundloop/kb/` tree."""
    from groundloop.mcp_shell import config as cfg

    monkeypatch.setattr(
        cfg, "DEFAULT_CACHE_PATH", tmp_path / "kb" / "skills_index.pkl"
    )
