from __future__ import annotations

import pytest

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator import openai_synthesizer as mod


def _cit() -> SearchResult:
    return SearchResult(
        node_id="n1", skill_name="python-testing", section_path=("Fixtures",),
        section_body="Use pytest fixtures.", tags=("domain:python",),
        source_path="/x", score=1.0, rank=1,
    )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = self._Chat(content)

    class _Chat:
        def __init__(self, content: str) -> None:
            self.completions = _FakeClient._Completions(content)

    class _Completions:
        def __init__(self, content: str) -> None:
            self._content = content

        def create(self, **_: object) -> object:
            class _Msg:
                def __init__(self, c: str) -> None:
                    self.content = c
            class _Choice:
                def __init__(self, c: str) -> None:
                    self.message = _Msg(c)
            class _Resp:
                def __init__(self, c: str) -> None:
                    self.choices = [_Choice(c)]
            return _Resp(self._content)


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mod.OpenAISynthesizer()


def test_openai_parses_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    content = '{"proposed_files": {"main.py": "x = 2"}, "rationale": "ok", "cited_node_ids": ["n1"]}'
    monkeypatch.setattr(mod, "OpenAI", lambda **kw: _FakeClient(content))
    s = mod.OpenAISynthesizer()
    out = s.synthesize(spec="s", current_files={"main.py": "x = 1"}, citations=[_cit()], iteration=0)
    assert out.proposed_files == {"main.py": "x = 2"}
    assert out.cited_node_ids == ("n1",)


def test_openai_parse_error_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(mod, "OpenAI", lambda **kw: _FakeClient("not json"))
    s = mod.OpenAISynthesizer()
    current = {"main.py": "x = 1"}
    out = s.synthesize(spec="s", current_files=current, citations=[_cit()], iteration=0)
    assert out.proposed_files == current
    assert out.rationale == "parse_error"
    assert out.cited_node_ids == ()
