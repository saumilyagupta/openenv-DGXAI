from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from codeforge.interrogator import InterrogationResult, Interrogator


# ---------------------------------------------------------------------------
# Fixtures — duck-typed stub that matches SkillsIndex.search() interface
# ---------------------------------------------------------------------------

_CORPUS_NODES = [
    {
        "id": "test_001",
        "skill_name": "python-testing",
        "section_path": ["Testing", "Fixtures"],
        "section_body": (
            "Use pytest fixtures for setup and teardown. "
            "Fixtures can be parameterized."
        ),
        "tags": ["domain:testing", "phase:implementation"],
        "source_path": "/test/skills/SKILL.md",
    },
    {
        "id": "test_002",
        "skill_name": "python-patterns",
        "section_path": ["Patterns", "Error Handling"],
        "section_body": (
            "Handle errors explicitly at every level. "
            "Provide user-friendly error messages."
        ),
        "tags": ["domain:patterns", "phase:implementation"],
        "source_path": "/test/skills/patterns/SKILL.md",
    },
    {
        "id": "test_003",
        "skill_name": "security-review",
        "section_path": ["Security", "Input Validation"],
        "section_body": (
            "Validate all user input before processing. "
            "Use schema-based validation where available."
        ),
        "tags": ["domain:security", "phase:review"],
        "source_path": "/test/skills/security/SKILL.md",
    },
]


@dataclass(frozen=True)
class _FakeSearchResult:
    """Minimal duck-typed SearchResult for testing without the KB module."""

    node_id: str
    skill_name: str
    section_path: tuple[str, ...]
    section_body: str
    tags: tuple[str, ...]
    source_path: str
    score: float
    rank: int
    cluster_id: str | None = None


class _FakeIndex:
    """Duck-typed stand-in for SkillsIndex — returns canned search results."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        required_tags: set[str] | None = None,
    ) -> list[_FakeSearchResult]:
        results: list[_FakeSearchResult] = []
        for rank, node in enumerate(self._nodes[:top_k], start=1):
            results.append(
                _FakeSearchResult(
                    node_id=node["id"],
                    skill_name=node["skill_name"],
                    section_path=tuple(node["section_path"]),
                    section_body=node["section_body"],
                    tags=tuple(node["tags"]),
                    source_path=node["source_path"],
                    score=1.0 / rank,
                    rank=rank,
                )
            )
        return results


@pytest.fixture()
def fake_index() -> _FakeIndex:
    return _FakeIndex(_CORPUS_NODES)


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    """Write a small JSONL corpus and return its path."""
    p = tmp_path / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for node in _CORPUS_NODES:
            f.write(json.dumps(node) + "\n")
    return p


def _kb_available() -> bool:
    try:
        from codeforge.kb.indexer import SkillsIndex  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Tests — InterrogationResult model
# ---------------------------------------------------------------------------


class TestInterrogationResultModel:
    def test_frozen(self) -> None:
        result = InterrogationResult(
            questions=("q1",),
            cited_node_ids=("n1",),
        )
        with pytest.raises(Exception):  # noqa: B017
            result.questions = ("q2",)  # type: ignore[misc]

    def test_fields(self) -> None:
        result = InterrogationResult(
            questions=("a", "b"),
            cited_node_ids=("x",),
        )
        assert result.questions == ("a", "b")
        assert result.cited_node_ids == ("x",)


# ---------------------------------------------------------------------------
# Tests — Interrogator with None index
# ---------------------------------------------------------------------------


class TestInterrogatorNoIndex:
    def test_generate_returns_questions(self) -> None:
        interr = Interrogator(index=None)
        result = interr.generate("implement greet(name)")
        assert isinstance(result, InterrogationResult)
        assert len(result.questions) == 5

    def test_generate_empty_cited_ids(self) -> None:
        interr = Interrogator(index=None)
        result = interr.generate("implement greet(name)")
        assert result.cited_node_ids == ()

    def test_questions_contain_brief(self) -> None:
        interr = Interrogator(index=None)
        result = interr.generate("implement greet(name)")
        assert any("implement greet(name)" in q for q in result.questions)

    def test_fallback_skill_names(self) -> None:
        interr = Interrogator(index=None)
        result = interr.generate("do something")
        assert any("the skill library" in q for q in result.questions)


# ---------------------------------------------------------------------------
# Tests — Interrogator with a duck-typed index stub
# ---------------------------------------------------------------------------


class TestInterrogatorWithFakeIndex:
    def test_generate_returns_five_questions(self, fake_index: _FakeIndex) -> None:
        interr = Interrogator(index=fake_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures for testing")
        assert len(result.questions) == 5

    def test_cited_node_ids_from_search(self, fake_index: _FakeIndex) -> None:
        interr = Interrogator(index=fake_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures for testing")
        assert len(result.cited_node_ids) > 0
        corpus_ids = {n["id"] for n in _CORPUS_NODES}
        for nid in result.cited_node_ids:
            assert nid in corpus_ids

    def test_cited_ids_max_two(self, fake_index: _FakeIndex) -> None:
        interr = Interrogator(index=fake_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures")
        assert len(result.cited_node_ids) <= 2

    def test_questions_reference_skill_names(self, fake_index: _FakeIndex) -> None:
        interr = Interrogator(index=fake_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures for testing")
        all_text = " ".join(result.questions)
        corpus_skill_names = {n["skill_name"] for n in _CORPUS_NODES}
        assert any(sn in all_text for sn in corpus_skill_names)

    def test_single_node_index(self) -> None:
        """When only one search result, skill_name2 falls back to skill_name."""
        idx = _FakeIndex([_CORPUS_NODES[0]])
        interr = Interrogator(index=idx)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures")
        assert len(result.questions) == 5
        assert len(result.cited_node_ids) == 1
        # Both template slots should use the same skill name
        all_text = " ".join(result.questions)
        assert all_text.count("python-testing") >= 2


# ---------------------------------------------------------------------------
# Tests — Interrogator with real SkillsIndex (skipped if M3 not built yet)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kb_available(), reason="codeforge.kb.indexer not available yet (M3 in progress)")
class TestInterrogatorWithRealIndex:
    @pytest.fixture()
    def real_index(self, corpus_path: Path) -> object:
        from codeforge.kb.indexer import SkillsIndex
        idx = SkillsIndex(corpus_path=corpus_path)
        idx.build()
        return idx

    def test_generate_returns_five_questions(self, real_index: object) -> None:
        interr = Interrogator(index=real_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures for testing")
        assert len(result.questions) == 5

    def test_cited_node_ids_from_real_search(self, real_index: object) -> None:
        interr = Interrogator(index=real_index)  # type: ignore[arg-type]
        result = interr.generate("pytest fixtures for testing")
        assert len(result.cited_node_ids) > 0
        corpus_ids = {n["id"] for n in _CORPUS_NODES}
        for nid in result.cited_node_ids:
            assert nid in corpus_ids


# ---------------------------------------------------------------------------
# Tests — brief_head truncation
# ---------------------------------------------------------------------------


class TestBriefHeadTruncation:
    def test_long_brief_truncated(self) -> None:
        interr = Interrogator(index=None)
        long_brief = "x" * 200
        result = interr.generate(long_brief)
        for q in result.questions:
            assert long_brief not in q
        assert any("x" * 80 in q for q in result.questions)

    def test_empty_brief_uses_fallback(self) -> None:
        interr = Interrogator(index=None)
        result = interr.generate("   ")
        assert any("the task" in q for q in result.questions)

    def test_exact_80_chars(self) -> None:
        interr = Interrogator(index=None)
        brief = "a" * 80
        result = interr.generate(brief)
        assert any(brief in q for q in result.questions)
