from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop.python_sandbox.tools import DEFAULT_TOOLS


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.python_sandbox")
    p.add_argument("project_dir", type=Path)
    p.add_argument("--tool", action="append", default=None)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if not args.project_dir.is_dir():
        print(f"ERROR: project_dir not found: {args.project_dir}", file=sys.stderr)
        return 1
    tools = tuple(args.tool) if args.tool else DEFAULT_TOOLS
    result = run_sandbox(
        project_dir=args.project_dir, tools=tools, timeout_per_tool=args.timeout,
    )
    if args.format == "json":
        print(result.model_dump_json())
    else:
        print(f"composite_score={result.composite_score:.3f}")
        for name, p in result.parsed.items():
            print(f"  {name}: ok={p.ok} count={p.count}")
        if result.imports.unresolved:
            print(f"  unresolved imports: {list(result.imports.unresolved)}")
    return 0
