from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from codeforge.ralph.models import SynthesisResult
from codeforge.ralph.synthesizer import LLMSynthesizer, StubSynthesizer, Synthesizer

if TYPE_CHECKING:
    from codeforge.kb.models import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_citation(
    *,
    node_id: str = "node_42",
    skill_name: str = "python-patterns",
    section_path: tuple[str, ...] = ("Idioms", "Type Hints"),
    section_body: str = "Use type annotations on all function signatures.",
    score: float = 12.5,
    rank: int = 0,
) -> SearchResult:
    from codeforge.kb.models import SearchResult as SR

    return SR(
        node_id=node_id,
        skill_name=skill_name,
        section_path=section_path,
        section_body=section_body,
        tags=("python",),
        source_path="/skills/python-patterns/SKILL.md",
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_llm_synthesizer_implements_protocol(self) -> None:
        """LLMSynthesizer structurally satisfies the Synthesizer protocol."""
        synth = LLMSynthesizer(api_key="test-key")
        assert hasattr(synth, "synthesize")
        # Verify the signature matches: keyword-only with correct names
        import inspect

        sig = inspect.signature(synth.synthesize)
        param_names = list(sig.parameters.keys())
        assert "spec" in param_names
        assert "current_files" in param_names
        assert "citations" in param_names
        assert "iteration" in param_names


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_spec(self) -> None:
        synth = LLMSynthesizer(api_key="k")
        prompt = synth._build_prompt(
            "implement greet(name)",
            {"main.py": "def greet(name): pass"},
            [],
            0,
        )
        assert "implement greet(name)" in prompt

    def test_includes_file_contents(self) -> None:
        synth = LLMSynthesizer(api_key="k")
        prompt = synth._build_prompt(
            "spec",
            {"main.py": "def greet(name): pass", "util.py": "import os"},
            [],
            0,
        )
        assert "def greet(name): pass" in prompt
        assert "import os" in prompt
        assert "main.py" in prompt
        assert "util.py" in prompt

    def test_includes_citations(self) -> None:
        cit = _make_citation(
            skill_name="python-patterns",
            section_body="Use type annotations.",
        )
        synth = LLMSynthesizer(api_key="k")
        prompt = synth._build_prompt("spec", {}, [cit], 0)
        assert "python-patterns" in prompt
        assert "Use type annotations." in prompt

    def test_includes_iteration(self) -> None:
        synth = LLMSynthesizer(api_key="k")
        prompt = synth._build_prompt("spec", {}, [], 3)
        assert "3" in prompt


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_single_file_block(self) -> None:
        response = (
            "Here is the improved code:\n"
            "# filename: main.py\n"
            "```python\n"
            "def greet(name: str) -> str:\n"
            "    return f'Hello, {name}!'\n"
            "```\n"
            "Rationale: Added type hints."
        )
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [])
        assert "main.py" in result.proposed_files
        assert "def greet(name: str) -> str:" in result.proposed_files["main.py"]
        assert result.rationale != ""

    def test_multiple_file_blocks(self) -> None:
        response = (
            "## main.py\n"
            "```python\n"
            "from util import helper\n"
            "def main() -> None: helper()\n"
            "```\n"
            "## util.py\n"
            "```python\n"
            "def helper() -> None: pass\n"
            "```\n"
            "Done."
        )
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [])
        assert "main.py" in result.proposed_files
        assert "util.py" in result.proposed_files
        assert "from util import helper" in result.proposed_files["main.py"]
        assert "def helper() -> None: pass" in result.proposed_files["util.py"]

    def test_no_code_blocks_returns_empty_files(self) -> None:
        response = "I don't know how to help with this."
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [])
        assert result.proposed_files == {}
        assert "no parseable code" in result.rationale.lower() or result.rationale != ""

    def test_cited_node_ids_from_citations(self) -> None:
        cit1 = _make_citation(node_id="node_1", skill_name="alpha")
        cit2 = _make_citation(node_id="node_2", skill_name="beta")
        response = (
            "# filename: main.py\n"
            "```python\n"
            "x = 1\n"
            "```\n"
        )
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [cit1, cit2])
        # All citation node_ids that were provided should be in the result
        assert "node_1" in result.cited_node_ids
        assert "node_2" in result.cited_node_ids

    def test_hash_filename_header(self) -> None:
        """Supports '# filename: xxx.py' header style."""
        response = (
            "# filename: core.py\n"
            "```python\n"
            "x = 42\n"
            "```\n"
        )
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [])
        assert "core.py" in result.proposed_files
        assert "x = 42" in result.proposed_files["core.py"]

    def test_double_hash_header(self) -> None:
        """Supports '## xxx.py' header style."""
        response = (
            "## data.py\n"
            "```python\n"
            "y = 99\n"
            "```\n"
        )
        synth = LLMSynthesizer(api_key="k")
        result = synth._parse_response(response, [])
        assert "data.py" in result.proposed_files


# ---------------------------------------------------------------------------
# _call_llm + synthesize
# ---------------------------------------------------------------------------

class TestCallLLM:
    def test_raises_without_api_key(self) -> None:
        synth = LLMSynthesizer(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            synth._call_llm("hello")

    def test_no_api_key_default(self) -> None:
        """Constructor with no key and no env var results in empty key."""
        with patch.dict("os.environ", {}, clear=True):
            synth = LLMSynthesizer()
            assert synth._api_key == ""

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-123"}):
            synth = LLMSynthesizer()
            assert synth._api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            synth = LLMSynthesizer(api_key="explicit-key")
            assert synth._api_key == "explicit-key"

    def test_call_llm_uses_anthropic_sdk(self) -> None:
        synth = LLMSynthesizer(api_key="test-key", model="claude-test")
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="response text")]
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = synth._call_llm("test prompt")

        assert result == "response text"
        mock_client.messages.create.assert_called_once_with(
            model="claude-test",
            max_tokens=4096,
            messages=[{"role": "user", "content": "test prompt"}],
        )

    def test_call_llm_falls_back_to_httpx(self) -> None:
        synth = LLMSynthesizer(api_key="test-key", model="claude-test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"text": "httpx response"}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("sys.modules", {"anthropic": None}):
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                result = synth._call_llm("test prompt")

        assert result == "httpx response"
        mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# Full synthesize flow
# ---------------------------------------------------------------------------

class TestSynthesize:
    def test_synthesize_raises_without_key(self) -> None:
        synth = LLMSynthesizer(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            synth.synthesize(
                spec="test",
                current_files={"main.py": "pass"},
                citations=[],
                iteration=0,
            )

    def test_synthesize_with_mocked_api(self) -> None:
        synth = LLMSynthesizer(api_key="test-key")
        mock_response = (
            "Here's improved code:\n"
            "# filename: main.py\n"
            "```python\n"
            "def greet(name: str) -> str:\n"
            "    return f'Hello, {name}!'\n"
            "```\n"
            "Rationale: Added type hints."
        )
        with patch.object(synth, "_call_llm", return_value=mock_response):
            result = synth.synthesize(
                spec="implement greet",
                current_files={"main.py": "def greet(name): pass"},
                citations=[],
                iteration=0,
            )
        assert isinstance(result, SynthesisResult)
        assert "main.py" in result.proposed_files
        assert "def greet(name: str) -> str:" in result.proposed_files["main.py"]
        assert result.rationale != ""

    def test_synthesize_with_citations(self) -> None:
        cit = _make_citation(node_id="node_99")
        synth = LLMSynthesizer(api_key="test-key")
        mock_response = (
            "# filename: main.py\n"
            "```python\n"
            "x = 1\n"
            "```\n"
        )
        with patch.object(synth, "_call_llm", return_value=mock_response):
            result = synth.synthesize(
                spec="do stuff",
                current_files={},
                citations=[cit],
                iteration=1,
            )
        assert "node_99" in result.cited_node_ids

    def test_synthesize_unparseable_returns_current_files(self) -> None:
        synth = LLMSynthesizer(api_key="test-key")
        mock_response = "I don't understand the task."
        with patch.object(synth, "_call_llm", return_value=mock_response):
            result = synth.synthesize(
                spec="do stuff",
                current_files={"main.py": "original"},
                citations=[],
                iteration=0,
            )
        # Falls back to current files when no code blocks parsed
        assert result.proposed_files == {"main.py": "original"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# StubSynthesizer with-citations branches (improves file coverage)
# ---------------------------------------------------------------------------

class TestStubSynthesizerCitations:
    def test_with_code_block_citation(self) -> None:
        """StubSynthesizer applies a fenced code block from the top citation."""
        cit = _make_citation(
            skill_name="test-skill",
            section_body="Example:\n```python\nx = 42\n```\n",
            rank=0,
        )
        synth = StubSynthesizer()
        result = synth.synthesize(
            spec="implement something",
            current_files={"main.py": ""},
            citations=[cit],
            iteration=0,
        )
        assert "_from_test_skill_0" in result.proposed_files["main.py"]
        assert "Applied suggestion" in result.rationale
        assert cit.node_id in result.cited_node_ids

    def test_already_applied_code_block(self) -> None:
        """StubSynthesizer detects wrapper already exists."""
        cit = _make_citation(
            skill_name="test-skill",
            section_body="Example:\n```python\nx = 42\n```\n",
            rank=0,
        )
        synth = StubSynthesizer()
        # Pre-populate main.py with the wrapper name
        result = synth.synthesize(
            spec="test",
            current_files={"main.py": "def _from_test_skill_0(): pass"},
            citations=[cit],
            iteration=0,
        )
        assert "Already applied" in result.rationale

    def test_no_code_block_in_citation(self) -> None:
        """StubSynthesizer handles citation with no fenced code block."""
        cit = _make_citation(
            skill_name="guidance",
            section_body="Just some text, no code block.",
            rank=0,
        )
        synth = StubSynthesizer()
        result = synth.synthesize(
            spec="test",
            current_files={"main.py": ""},
            citations=[cit],
            iteration=0,
        )
        assert "Consulted" in result.rationale
        assert "# consulted:" in result.proposed_files["main.py"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_default_model(self) -> None:
        synth = LLMSynthesizer(api_key="k")
        assert synth._model == "claude-sonnet-4-20250514"

    def test_custom_model(self) -> None:
        synth = LLMSynthesizer(api_key="k", model="gpt-4")
        assert synth._model == "gpt-4"

    def test_default_max_tokens(self) -> None:
        synth = LLMSynthesizer(api_key="k")
        assert synth._max_tokens == 4096

    def test_custom_max_tokens(self) -> None:
        synth = LLMSynthesizer(api_key="k", max_tokens=8192)
        assert synth._max_tokens == 8192
