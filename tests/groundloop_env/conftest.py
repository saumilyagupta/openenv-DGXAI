from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def tiny_corpus_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "groundloop" / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
