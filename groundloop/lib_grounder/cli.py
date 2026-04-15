from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundloop.lib_grounder.grounder import ground


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="groundloop.lib_grounder")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-c", "--code", type=str)
    src.add_argument("-f", "--file", type=Path)
    p.add_argument("--format", choices=("text", "json"), default="text")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    source = args.code if args.code is not None else args.file.read_text(encoding="utf-8")
    report = ground(source)
    if args.format == "json":
        print(report.model_dump_json())
    else:
        print(f"groundedness={report.groundedness:.3f}")
        for s in report.ungrounded:
            name = f"{s.module}.{s.attr}" if s.attr else s.module
            print(f"  ungrounded {s.kind} at line {s.line}: {name}")
    return 0
