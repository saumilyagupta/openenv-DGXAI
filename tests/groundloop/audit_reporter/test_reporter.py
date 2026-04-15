from __future__ import annotations

from groundloop.audit_reporter.reporter import AuditReporter
from groundloop.ralph_orchestrator.models import Iteration, RunResult


def _mk_run(iters: list[tuple[str, float, tuple[str, ...]]]) -> RunResult:
    it_objs = []
    for i, (reason, score, cites) in enumerate(iters):
        it_objs.append(
            Iteration(
                index=i, cited_node_ids=cites, rationale="r",
                proposed_files={}, sandbox_score_before=0.0,
                sandbox_score_after=score, kept=reason == "score_improved",
                reason=reason,  # type: ignore[arg-type]
            )
        )
    return RunResult(
        run_id="r1", spec="s", started_at="t", ended_at="t",
        final_score=iters[-1][1] if iters else 0.0,
        final_files={}, iterations=tuple(it_objs),
        terminated_by="max_iters",
    )


def test_audit_report_counts_reasons() -> None:
    run = _mk_run(
        [
            ("score_improved", 0.5, ("n1",)),
            ("score_regressed", 0.4, ("n2",)),
            ("score_plateau", 0.4, ("n1",)),
        ]
    )
    r = AuditReporter.build(run)
    assert r.iterations_total == 3
    assert r.iterations_kept == 1
    assert r.iterations_regressed == 1
    assert r.iterations_plateau == 1


def test_audit_report_aggregates_citations() -> None:
    run = _mk_run(
        [
            ("score_improved", 0.5, ("n1", "n2")),
            ("score_improved", 0.7, ("n1",)),
        ]
    )
    r = AuditReporter.build(run)
    d = dict(r.skill_citations)
    assert d["n1"] == 2
    assert d["n2"] == 1


def test_audit_report_trajectory() -> None:
    run = _mk_run(
        [
            ("score_improved", 0.3, ()),
            ("score_improved", 0.7, ()),
        ]
    )
    r = AuditReporter.build(run)
    assert r.score_trajectory == (0.3, 0.7)


def test_audit_report_empty_iterations() -> None:
    run = _mk_run([])
    r = AuditReporter.build(run)
    assert r.iterations_total == 0
    assert r.skill_citations == ()
    assert r.score_trajectory == ()


def test_audit_report_summary_contains_run_id() -> None:
    run = _mk_run([("score_improved", 0.9, ())])
    r = AuditReporter.build(run)
    assert "r1" in r.summary
    assert "max_iters" in r.summary
