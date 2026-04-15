from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groundloop.skills_scraper.config import DEFAULT_OUTPUT, DEFAULT_SOURCES
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="groundloop.skills_scraper")
    parser.add_argument(
        "--sources",
        help="Glob pattern for SKILL.md files (overrides defaults). Repeatable.",
        action="append",
        default=None,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.sources:
        sources = [
            SourceRoot(label=f"cli-{i}", glob=g) for i, g in enumerate(args.sources)
        ]
    else:
        sources = DEFAULT_SOURCES

    output = Path(args.output).expanduser()
    result = run_scraper(sources=sources, output=output)

    print(
        f"scraped={result.scraped_files} files, "
        f"skipped={result.skipped_files}, "
        f"nodes={result.total_nodes}, "
        f"errors={len(result.errors)}"
    )

    if result.total_nodes == 0:
        return 1
    return 0
