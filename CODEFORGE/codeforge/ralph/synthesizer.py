from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from codeforge.ralph.models import SynthesisResult

if TYPE_CHECKING:
    from codeforge.kb.models import SearchResult

_FENCED_PY_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


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
