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
    """Calls an LLM to produce improved code given spec + current files + citations.

    Uses the Anthropic Python SDK (``anthropic``) or falls back to ``httpx``
    for raw API calls when the SDK is unavailable.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        self._api_key: str = api_key if api_key is not None else os.environ.get(
            "ANTHROPIC_API_KEY", "",
        )
        self._model: str = model
        self._max_tokens: int = max_tokens

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
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the Claude API.  Falls back to *httpx* if the SDK is absent."""
        if not self._api_key:
            msg = "ANTHROPIC_API_KEY not set — LLMSynthesizer requires an API key"
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
                "https://api.anthropic.com/v1/messages",
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
