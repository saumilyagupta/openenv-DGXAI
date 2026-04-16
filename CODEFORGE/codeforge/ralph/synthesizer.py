from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from codeforge.ralph.models import SynthesisResult

if TYPE_CHECKING:
    from codeforge.kb.models import SearchResult

_FENCED_PY_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# Matches '# filename: foo.py' or '## foo.py' immediately before a fenced block
_FILENAME_HEADER_RE = re.compile(
    r"(?:^#{1,2}\s+(?:filename:\s*)?(\S+\.py)\s*$)",
    re.MULTILINE,
)


class Synthesizer(Protocol):
    """Abstract synthesizer interface for Ralph loop."""

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult: ...


class StubSynthesizer:
    """Deterministic, KB-grounded stub. No LLM. Used in tests and as default."""

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        del spec, iteration  # inputs retained for protocol; stub ignores

        if not citations:
            return SynthesisResult(
                proposed_files=dict(current_files),
                rationale="no_citations",
                cited_node_ids=(),
            )

        top = citations[0]
        main = current_files.get("main.py", "")
        blocks = _FENCED_PY_RE.findall(top.section_body)

        wrapper_name = f"_from_{top.skill_name.replace('-', '_')}_{top.rank}"
        if blocks and wrapper_name not in main:
            body = "\n".join(
                f"    {ln}" if ln.strip() else "" for ln in blocks[0].splitlines()
            )
            snippet = f"\n\ndef {wrapper_name}() -> None:\n{body}\n"
            new_main = main + snippet
            rationale = (
                f"Applied suggestion from "
                f"{top.skill_name}/{'/'.join(top.section_path)}"
            )
        elif blocks:
            new_main = main
            rationale = (
                f"Already applied "
                f"{top.skill_name}/{'/'.join(top.section_path)}"
            )
        else:
            comment = f"# consulted: {top.skill_name}/{'/'.join(top.section_path)}\n"
            new_main = main + (comment if comment not in main else "")
            rationale = (
                f"Consulted "
                f"{top.skill_name}/{'/'.join(top.section_path)} (no code block)"
            )

        new_files = {**current_files, "main.py": new_main}
        return SynthesisResult(
            proposed_files=new_files,
            rationale=rationale,
            cited_node_ids=(top.node_id,),
        )


class LLMSynthesizer:
    """Calls any LLM to produce improved code given spec + current files + citations.

    Provider-agnostic: works with Ollama, OpenAI, Anthropic, or any
    OpenAI-compatible API. Set *provider* to choose the backend:

    - ``"openai"`` — OpenAI / OpenAI-compatible (default). Works with Ollama,
      LM Studio, vLLM, Together, Groq, etc. Set *base_url* for local models.
    - ``"anthropic"`` — Anthropic Claude API.
    - ``"ollama"`` — Shortcut for Ollama (sets base_url to localhost:11434).

    Examples::

        # Ollama (local, no API key needed)
        LLMSynthesizer(provider="ollama", model="llama3")

        # OpenAI
        LLMSynthesizer(provider="openai", model="gpt-4o")

        # Anthropic
        LLMSynthesizer(provider="anthropic", model="claude-sonnet-4-20250514")

        # Any OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.)
        LLMSynthesizer(
            provider="openai",
            base_url="http://localhost:8000/v1",
            model="my-local-model",
        )
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._provider: str = provider.lower()
        self._max_tokens: int = max_tokens

        if self._provider == "ollama":
            self._base_url = base_url or "http://localhost:11434/v1"
            self._api_key = api_key or "ollama"  # Ollama ignores this
            self._model = model or "llama3"
        elif self._provider == "anthropic":
            self._base_url = base_url or "https://api.anthropic.com"
            self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._model = model or "claude-sonnet-4-20250514"
        else:  # openai or any compatible
            self._base_url = base_url or "https://api.openai.com/v1"
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self._model = model or "gpt-4o"

    # ------------------------------------------------------------------
    # Public API (satisfies Synthesizer protocol)
    # ------------------------------------------------------------------

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        """Build prompt, call LLM, parse response into *SynthesisResult*."""
        prompt = self._build_prompt(spec, current_files, citations, iteration)
        response_text = self._call_llm(prompt)
        result = self._parse_response(response_text, citations)

        # If no code blocks were parsed, fall back to current files unchanged.
        if not result.proposed_files:
            return SynthesisResult(
                proposed_files=dict(current_files),
                rationale=result.rationale or "No parseable code blocks in LLM response",
                cited_node_ids=result.cited_node_ids,
            )
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> str:
        """Build the synthesis prompt with spec, files, citations, and iteration."""
        parts: list[str] = [
            "You are a Python code synthesis assistant.",
            f"Iteration: {iteration}",
            "",
            "## Task Specification",
            spec,
        ]

        if current_files:
            parts.append("")
            parts.append("## Current Files")
            for fname, content in current_files.items():
                parts.append(f"\n### {fname}")
                parts.append(f"```python\n{content}\n```")

        if citations:
            parts.append("")
            parts.append("## Skill Corpus Citations")
            for cit in citations:
                parts.append(
                    f"\n### {cit.skill_name} / {'/'.join(cit.section_path)}"
                    f" (score={cit.score:.1f})"
                )
                parts.append(cit.section_body)

        parts.append("")
        parts.append("## Instructions")
        parts.append(
            "Produce improved Python files. For EACH file, emit a header "
            "`# filename: <name>.py` followed by a fenced python code block. "
            "After all files, write a short rationale explaining your changes."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM API call (provider-agnostic)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider.

        Supports three backends:
        - **openai / ollama**: OpenAI-compatible ``/chat/completions`` endpoint.
          Works with Ollama, LM Studio, vLLM, Together, Groq, OpenAI, etc.
        - **anthropic**: Anthropic ``/v1/messages`` endpoint.
        """
        if self._provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai_compatible(prompt)

    def _call_openai_compatible(self, prompt: str) -> str:
        """OpenAI-compatible API (works with Ollama, LM Studio, vLLM, etc.)."""
        import httpx

        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key and self._api_key != "ollama":
            headers["Authorization"] = f"Bearer {self._api_key}"

        resp = httpx.post(
            url,
            headers=headers,
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    def _call_anthropic(self, prompt: str) -> str:
        """Anthropic Claude API."""
        if not self._api_key:
            msg = (
                "ANTHROPIC_API_KEY not set. For local models, use "
                "provider='ollama' or provider='openai' with base_url."
            )
            raise ValueError(msg)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)
            message = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            block = message.content[0]
            return str(getattr(block, "text", ""))
        except ImportError:
            import httpx

            resp = httpx.post(
                f"{self._base_url.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data: dict[str, object] = resp.json()
            content = data["content"]
            assert isinstance(content, list)
            first = content[0]
            assert isinstance(first, dict)
            return str(first["text"])

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        text: str,
        citations: Sequence[SearchResult],
    ) -> SynthesisResult:
        """Parse LLM response: extract fenced code blocks with filename headers."""
        proposed_files: dict[str, str] = {}

        # Strategy: find all filename headers and pair each with the next
        # fenced python block.
        header_positions: list[tuple[int, str]] = [
            (m.start(), m.group(1))
            for m in _FILENAME_HEADER_RE.finditer(text)
        ]

        code_blocks: list[tuple[int, str]] = [
            (m.start(), m.group(1))
            for m in _FENCED_PY_RE.finditer(text)
        ]

        if header_positions and code_blocks:
            for hdr_pos, filename in header_positions:
                # Find the first code block that follows this header
                for blk_pos, code in code_blocks:
                    if blk_pos > hdr_pos:
                        proposed_files[filename] = code
                        break

        # Extract rationale from non-code, non-header text
        rationale_text = text
        for _, code in code_blocks:
            rationale_text = rationale_text.replace(f"```python\n{code}\n```", "")
        for m in _FILENAME_HEADER_RE.finditer(rationale_text):
            rationale_text = rationale_text.replace(m.group(0), "")
        rationale = rationale_text.strip()
        # Collapse to a single line for storage
        rationale = " ".join(rationale.split())

        if not proposed_files:
            rationale = rationale or "No parseable code blocks in LLM response"

        cited_node_ids = tuple(c.node_id for c in citations)

        return SynthesisResult(
            proposed_files=proposed_files,
            rationale=rationale,
            cited_node_ids=cited_node_ids,
        )
