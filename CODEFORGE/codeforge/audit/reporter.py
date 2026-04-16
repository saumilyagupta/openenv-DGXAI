from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from codeforge.audit.models import AuditReport

if TYPE_CHECKING:
    from codeforge.ralph.models import RunResult


class AuditReporter:
    """Builds an AuditReport from a Ralph RunResult."""

    @staticmethod
    def build(run: RunResult, hallucination_rate: float = 0.0) -> AuditReport:
        cites: Counter[str] = Counter()
        kept = 0
        regressed = 0
        plateau = 0
        trajectory: list[float] = []

        for it in run.iterations:
            cites.update(it.cited_node_ids)
            trajectory.append(it.sandbox_score_after)
            if it.reason == "score_improved":
                kept += 1
            elif it.reason == "score_regressed":
                regressed += 1
            elif it.reason == "score_plateau":
                plateau += 1

        summary = (
            f"run={run.run_id} iters={len(run.iterations)} "
            f"final={run.final_score:.3f} terminated_by={run.terminated_by}"
        )
        skill_citations = tuple(
            sorted(cites.items(), key=lambda kv: (-kv[1], kv[0]))
        )

        return AuditReport(
            run_id=run.run_id,
            summary=summary,
            iterations_total=len(run.iterations),
            iterations_kept=kept,
            iterations_regressed=regressed,
            iterations_plateau=plateau,
            skill_citations=skill_citations,
            score_trajectory=tuple(trajectory),
            final_score=run.final_score,
            terminated_by=run.terminated_by,
            hallucination_rate=hallucination_rate,
        )
