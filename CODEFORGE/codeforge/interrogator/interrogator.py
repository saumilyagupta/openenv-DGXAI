from __future__ import annotations

from typing import TYPE_CHECKING

from codeforge.interrogator.models import InterrogationResult

if TYPE_CHECKING:
    from codeforge.kb.indexer import SkillsIndex

_TEMPLATES = (
    "What is the exact success criterion for '{brief_head}'?",
    "Have you considered the guidance from {skill_name}: '{section_title}'?",
    "Which of these assumptions is most load-bearing: success metric, inputs, failure modes?",
    "What is the single hardest edge case for '{brief_head}'?",
    "Have you consulted {skill_name2} for the patterns it recommends?",
)


class Interrogator:
    """Generates Socratic questions that cite real skill corpus nodes."""

    def __init__(self, index: SkillsIndex | None) -> None:
        self._index = index

    def generate(self, brief: str, *, top_k: int = 5) -> InterrogationResult:
        brief_head = brief.strip()[:80] or "the task"
        results = (
            self._index.search(brief, top_k=top_k)
            if self._index is not None
            else []
        )
        cited_ids = tuple(r.node_id for r in results[:2])
        first = results[0] if results else None
        second = results[1] if len(results) > 1 else first

        skill_name = first.skill_name if first else "the skill library"
        section_title = (
            "/".join(first.section_path) if first else "the relevant section"
        )
        skill_name2 = second.skill_name if second else skill_name

        questions = tuple(
            t.format(
                brief_head=brief_head,
                skill_name=skill_name,
                section_title=section_title,
                skill_name2=skill_name2,
            )
            for t in _TEMPLATES
        )
        return InterrogationResult(questions=questions, cited_node_ids=cited_ids)
