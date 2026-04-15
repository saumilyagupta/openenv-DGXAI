from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundloop.audit_reporter.reporter import AuditReporter
from groundloop.ralph_orchestrator.models import RunResult


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="groundloop.audit_reporter")
    p.add_argument("--run-json", required=True, type=Path)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    run = RunResult.model_validate_json(args.run_json.read_text(encoding="utf-8"))
    report = AuditReporter.build(run)
    print(report.model_dump_json())
    return 0
