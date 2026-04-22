from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BASE_INSTRUCTIONS = (
    "You are an expert Python programmer. Solve the problem with MINIMAL code — "
    "no docstrings, no comments, no explanation."
)

OUTPUT_FMT = (
    "Output format: a single fenced Python code block. Start with ```python and end "
    "with ```. Nothing else outside the block."
)


@dataclass(frozen=True)
class SnippetLike:
    skill_name: str
    section_title: str
    body: str


def _format_citations(citations: list[dict[str, Any]], snippet_chars: int = 400) -> str:
    if not citations:
        return ""
    blocks: list[str] = []
    for i, c in enumerate(citations, start=1):
        name = c.get("skill_name") or c.get("id") or "?"
        sec = c.get("section_path") or [c.get("section_title", "")]
        title = " / ".join(str(p) for p in sec if p)
        body = str(c.get("section_body", "")).strip()
        if len(body) > snippet_chars:
            body = body[:snippet_chars].rstrip() + "..."
        header = f"[{i}] {name} — {title}".strip(" —")
        blocks.append(f"{header}\n{body}")
    return "CodeForge KB citations (research material, may or may not apply):\n\n" + "\n\n".join(blocks)


def _format_questions(questions: list[str]) -> str:
    if not questions:
        return ""
    lines = [f"- {q}" for q in questions[:5]]
    return "CodeForge interrogator Socratic prompts (apply only if relevant):\n" + "\n".join(lines)


def build_prompt(
    mode: str,
    text: str,
    test_list: list[str],
    *,
    citations: list[dict[str, Any]] | None = None,
    questions: list[str] | None = None,
    snippet_chars: int = 400,
) -> str:
    text = (text or "").strip()
    sample_test = test_list[0] if test_list else ""

    parts: list[str] = [BASE_INSTRUCTIONS, ""]

    if mode == "with_mcp":
        cit_block = _format_citations(citations or [], snippet_chars=snippet_chars)
        q_block = _format_questions(questions or [])
        if cit_block:
            parts.extend([cit_block, ""])
        if q_block:
            parts.extend([q_block, ""])

    parts.extend(["Problem:", text, ""])
    if sample_test:
        parts.extend(["Your solution must satisfy this test:", sample_test, ""])
    parts.append(OUTPUT_FMT)
    return "\n".join(parts)
