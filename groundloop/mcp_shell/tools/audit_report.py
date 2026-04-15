from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundloop.audit_reporter.reporter import AuditReporter
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AuditReportInput


def handle_audit_report(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = AuditReportInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    run = session.get_run_result(inp.run_id)
    if run is None:
        return {"status": "error", "reason": "unknown_run_id", "detail": inp.run_id}
    report = AuditReporter.build(run)
    return {
        "status": "ok",
        "run_id": run.run_id,
        "report": report.model_dump(),
        "metrics": session.metrics_snapshot(),
    }
