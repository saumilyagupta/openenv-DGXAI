from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AuditReportInput


def handle_audit_report(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = AuditReportInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    record = session.get_run(inp.run_id)
    if record is None:
        return {"status": "error", "reason": "unknown_run_id", "detail": inp.run_id}
    summary = (
        f"run={record.run_id} spec={record.spec[:60]!r} "
        f"status={record.status} iters={record.iterations}"
    )
    return {
        "status": "ok",
        "run_id": record.run_id,
        "summary": summary,
        "run_status": record.status,
        "iterations": record.iterations,
        "notes": list(record.notes),
        "metrics": session.metrics_snapshot(),
    }
