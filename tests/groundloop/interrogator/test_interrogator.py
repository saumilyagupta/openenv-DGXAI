from __future__ import annotations

from pathlib import Path

from groundloop.interrogator.interrogator import Interrogator
from groundloop.kb_indexer.index import SkillsIndex


def test_interrogator_returns_five_questions(tiny_corpus_path: Path) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    r = Interrogator(idx).generate("build a python api")
    assert len(r.questions) == 5
    assert len(r.cited_node_ids) >= 1


def test_interrogator_deterministic(tiny_corpus_path: Path) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    i = Interrogator(idx)
    assert i.generate("x") == i.generate("x")


def test_interrogator_no_index_still_returns_questions() -> None:
    r = Interrogator(None).generate("something")
    assert len(r.questions) == 5
    assert r.cited_node_ids == ()


def test_interrogator_empty_brief() -> None:
    r = Interrogator(None).generate("")
    assert "the task" in r.questions[0]
