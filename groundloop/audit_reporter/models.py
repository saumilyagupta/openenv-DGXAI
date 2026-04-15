from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    summary: str
    iterations_total: int
    iterations_kept: int
    iterations_regressed: int
    iterations_plateau: int
    skill_citations: tuple[tuple[str, int], ...]
    score_trajectory: tuple[float, ...]
    final_score: float
    terminated_by: str
    hallucination_rate: float
