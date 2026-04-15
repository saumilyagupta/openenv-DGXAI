from __future__ import annotations

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def _cit(node_id: str = "n1", body: str = "some body", rank: int = 1, score: float = 1.0) -> SearchResult:
    return SearchResult(
        node_id=node_id, skill_name="python-testing", section_path=("Fixtures",),
        section_body=body, tags=("domain:python",), source_path="/x", score=score, rank=rank,
    )


def test_stub_no_citations_is_noop() -> None:
    s = StubSynthesizer()
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1"}, citations=[], iteration=0)
    assert out.proposed_files == {"main.py": "x = 1"}
    assert out.cited_node_ids == ()


def test_stub_appends_comment_when_no_code_block() -> None:
    s = StubSynthesizer()
    cit = _cit(body="Discussion about fixtures, no code blocks here.")
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1"}, citations=[cit], iteration=0)
    assert "consulted: python-testing" in out.proposed_files["main.py"]
    assert "n1" in out.cited_node_ids


def test_stub_extracts_fenced_python() -> None:
    body = "Example:\n```python\ndef demo() -> int:\n    return 42\n```\n"
    s = StubSynthesizer()
    cit = _cit(body=body)
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1\n"}, citations=[cit], iteration=0)
    assert "demo" in out.proposed_files["main.py"]
    assert "_from_python_testing_1" in out.proposed_files["main.py"]


def test_stub_deterministic() -> None:
    body = "```python\ndef demo(): pass\n```"
    s = StubSynthesizer()
    cit = _cit(body=body)
    a = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    b = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    assert a.proposed_files == b.proposed_files
    assert a.rationale == b.rationale


def test_stub_idempotent_on_already_applied_code() -> None:
    body = "```python\ndef demo(): pass\n```"
    s = StubSynthesizer()
    cit = _cit(body=body)
    once = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    twice = s.synthesize(spec="x", current_files=once.proposed_files, citations=[cit], iteration=1)
    assert twice.proposed_files == once.proposed_files
