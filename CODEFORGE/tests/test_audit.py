from __future__ import annotations

import pytest

from codeforge.audit import AuditLedger, AuditReport, AuditReporter
from codeforge.models import AuditEntry
from codeforge.ralph.models import Iteration, RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    step: int = 0,
    action: str = "submit",
    reward: float = 0.5,
    quality: float = 0.6,
    skill_ids: tuple[str, ...] = (),
    clusters: tuple[str, ...] = (),
    brier: float | None = None,
    confidence: float | None = None,
) -> AuditEntry:
    return AuditEntry(
        step_index=step,
        action_type=action,
        cited_skill_ids=skill_ids,
        cited_clusters=clusters,
        grounding_report=None,
        reward=reward,
        brier_penalty=brier,
        confidence_declared=confidence,
        quality=quality,
    )


def _make_iteration(
    index: int,
    score_before: float,
    score_after: float,
    reason: str = "score_improved",
    cited: tuple[str, ...] = (),
) -> Iteration:
    return Iteration(
        index=index,
        cited_node_ids=cited,
        rationale="test",
        proposed_files={"main.py": "pass"},
        sandbox_score_before=score_before,
        sandbox_score_after=score_after,
        kept=reason == "score_improved",
        reason=reason,  # type: ignore[arg-type]
    )


def _make_run_result(
    iterations: tuple[Iteration, ...] | None = None,
    final_score: float = 0.85,
    terminated_by: str = "target_hit",
) -> RunResult:
    if iterations is None:
        iterations = (
            _make_iteration(0, 0.3, 0.5, "score_improved", ("skill-a", "skill-b")),
            _make_iteration(1, 0.5, 0.4, "score_regressed", ("skill-a",)),
            _make_iteration(2, 0.5, 0.5, "score_plateau", ("skill-c",)),
            _make_iteration(3, 0.5, 0.85, "score_improved", ("skill-a", "skill-c")),
        )
    return RunResult(
        run_id="run-001",
        spec="implement greet(name)",
        started_at="2026-04-16T10:00:00Z",
        ended_at="2026-04-16T10:05:00Z",
        final_score=final_score,
        final_files={"main.py": "def greet(name): ..."},
        iterations=iterations,
        terminated_by=terminated_by,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# AuditLedger tests
# ---------------------------------------------------------------------------

class TestAuditLedger:
    def test_starts_empty(self) -> None:
        ledger = AuditLedger()
        assert ledger.entries() == ()
        assert ledger.step_count() == 0

    def test_append_adds_entries(self) -> None:
        ledger = AuditLedger()
        entry = _make_entry(step=0)
        ledger.append(entry)
        assert ledger.step_count() == 1
        assert ledger.entries()[0] is entry

    def test_entries_returns_tuple(self) -> None:
        ledger = AuditLedger()
        ledger.append(_make_entry(step=0))
        ledger.append(_make_entry(step=1))
        result = ledger.entries()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_total_reward_sums_correctly(self) -> None:
        ledger = AuditLedger()
        ledger.append(_make_entry(step=0, reward=0.3))
        ledger.append(_make_entry(step=1, reward=0.5))
        ledger.append(_make_entry(step=2, reward=0.0))
        assert ledger.total_reward() == pytest.approx(0.8)

    def test_citation_count_aggregates(self) -> None:
        ledger = AuditLedger()
        ledger.append(_make_entry(step=0, skill_ids=("a", "b")))
        ledger.append(_make_entry(step=1, skill_ids=("a", "c")))
        ledger.append(_make_entry(step=2, skill_ids=("a",)))
        counts = ledger.citation_count()
        assert counts["a"] == 3
        assert counts["b"] == 1
        assert counts["c"] == 1

    def test_step_count(self) -> None:
        ledger = AuditLedger()
        for i in range(5):
            ledger.append(_make_entry(step=i))
        assert ledger.step_count() == 5

    def test_serialize_returns_dict(self) -> None:
        ledger = AuditLedger()
        ledger.append(_make_entry(step=0, reward=0.5, skill_ids=("x",)))
        serialized = ledger.serialize()
        assert isinstance(serialized, dict)
        assert "entries" in serialized
        assert "total_reward" in serialized
        assert "step_count" in serialized
        assert serialized["step_count"] == 1
        assert serialized["total_reward"] == pytest.approx(0.5)

    def test_empty_ledger_total_reward_is_zero(self) -> None:
        ledger = AuditLedger()
        assert ledger.total_reward() == 0.0

    def test_empty_ledger_citation_count_is_empty(self) -> None:
        ledger = AuditLedger()
        assert ledger.citation_count() == {}


# ---------------------------------------------------------------------------
# AuditReporter tests
# ---------------------------------------------------------------------------

class TestAuditReporter:
    def test_build_returns_audit_report(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert isinstance(report, AuditReport)

    def test_iteration_counts(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert report.iterations_total == 4
        assert report.iterations_kept == 2
        assert report.iterations_regressed == 1
        assert report.iterations_plateau == 1

    def test_skill_citations_sorted_by_frequency(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        # skill-a cited 3 times (iters 0,1,3), skill-c 2 times (iters 2,3), skill-b 1 time (iter 0)
        assert report.skill_citations[0] == ("skill-a", 3)
        assert report.skill_citations[1] == ("skill-c", 2)
        assert report.skill_citations[2] == ("skill-b", 1)

    def test_score_trajectory(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert report.score_trajectory == (0.5, 0.4, 0.5, 0.85)

    def test_final_score(self) -> None:
        run = _make_run_result(final_score=0.92)
        report = AuditReporter.build(run, hallucination_rate=0.0)
        assert report.final_score == pytest.approx(0.92)

    def test_hallucination_rate_passed_through(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run, hallucination_rate=0.15)
        assert report.hallucination_rate == pytest.approx(0.15)

    def test_terminated_by(self) -> None:
        run = _make_run_result(terminated_by="max_iters")
        report = AuditReporter.build(run)
        assert report.terminated_by == "max_iters"

    def test_run_id(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert report.run_id == "run-001"

    def test_summary_contains_key_info(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert "run-001" in report.summary
        assert "4" in report.summary  # iterations count
        assert "target_hit" in report.summary


# ---------------------------------------------------------------------------
# AuditReport tests
# ---------------------------------------------------------------------------

class TestAuditReport:
    def test_frozen(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        with pytest.raises(Exception):  # noqa: B017
            report.run_id = "modified"  # type: ignore[misc]

    def test_default_hallucination_rate(self) -> None:
        run = _make_run_result()
        report = AuditReporter.build(run)
        assert report.hallucination_rate == 0.0

    def test_empty_iterations(self) -> None:
        run = _make_run_result(iterations=(), final_score=0.0)
        report = AuditReporter.build(run)
        assert report.iterations_total == 0
        assert report.iterations_kept == 0
        assert report.skill_citations == ()
        assert report.score_trajectory == ()
