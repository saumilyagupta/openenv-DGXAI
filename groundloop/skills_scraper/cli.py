from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from groundloop.skills_scraper.config import DEFAULT_OUTPUT, DEFAULT_SOURCES
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="groundloop.skills_scraper")
    parser.add_argument(
        "--sources",
        help="Either the literal 'default' (use DEFAULT_SOURCES) or a path to "
             "a YAML file with schema: {sources: [{label, glob}, ...]}.",
        default="default",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _load_sources_yaml(path: Path) -> list[SourceRoot]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError(
            f"invalid sources file {path}: expected top-level 'sources' key"
        )
    entries = data["sources"]
    if not isinstance(entries, list):
        raise ValueError(f"invalid sources file {path}: 'sources' must be a list")
    return [SourceRoot(**entry) for entry in entries]


def _resolve_sources(arg: str) -> list[SourceRoot]:
    if arg == "default":
        return DEFAULT_SOURCES
    path = Path(arg).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"--sources must be 'default' or an existing YAML file; got {arg!r}"
        )
    return _load_sources_yaml(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        sources = _resolve_sources(args.sources)
    except (FileNotFoundError, ValueError, yaml.YAMLError, ValidationError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

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
