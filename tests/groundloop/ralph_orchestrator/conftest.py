from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def spec_path() -> Path:
    return Path(__file__).parent / "fixtures" / "spec_simple.txt"


@pytest.fixture
def initial_files(tmp_path: Path) -> dict[str, str]:
    src = Path(__file__).parent / "fixtures" / "initial_files" / "main.py"
    return {"main.py": src.read_text(encoding="utf-8")}


@pytest.fixture
def tiny_corpus_path() -> Path:
    return Path(__file__).parent.parent / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
