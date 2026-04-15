from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent.parent / "skills_scraper" / "fixtures" / "fake_skills"


@pytest.fixture
def tiny_corpus_path() -> Path:
    return Path(__file__).parent.parent / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
