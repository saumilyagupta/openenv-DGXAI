from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex

_DEFAULT_CORPUS = Path("groundloop/kb/skills_corpus.jsonl")
_DEFAULT_CACHE = Path("groundloop/kb/skills_index.pkl")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.kb_indexer")
    sub = p.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build + persist the index")
    _add_common(build)
    build.add_argument("--force", action="store_true")

    search = sub.add_parser("search", help="Search the index")
    _add_common(search)
    search.add_argument("query", type=str)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--format", choices=("text", "json"), default="text")

    stats = sub.add_parser("stats", help="Show index stats")
    _add_common(stats)

    return p.parse_args(argv)


def _cmd_build(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    cached = None if args.force else SkillsIndex.load(corpus_path=args.corpus, cache_path=args.cache)
    if cached is not None:
        print("cache hit")
        return 0
    idx = SkillsIndex(corpus_path=args.corpus, cache_path=args.cache)
    idx.build()
    idx.save()
    s = idx.stats()
    print(f"built: {s['node_count']} nodes, {s['vocab_size']} vocab, avg {s['avg_doc_len']:.1f} toks/doc")
    return 0


def _load_or_build(args: argparse.Namespace) -> SkillsIndex | None:
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return None
    idx = SkillsIndex.load(corpus_path=args.corpus, cache_path=args.cache)
    if idx is None:
        idx = SkillsIndex(corpus_path=args.corpus, cache_path=args.cache)
        idx.build()
        idx.save()
    return idx


def _cmd_search(args: argparse.Namespace) -> int:
    idx = _load_or_build(args)
    if idx is None:
        return 1
    tags = set(args.tag) if args.tag else None
    results = idx.search(args.query, top_k=args.top_k, required_tags=tags)
    if args.format == "json":
        payload = [r.model_dump() for r in results]
        for p in payload:
            p["section_path"] = list(p["section_path"])
            p["tags"] = list(p["tags"])
        print(json.dumps(payload))
    else:
        for r in results:
            print(f"[{r.rank}] score={r.score:.3f} {r.skill_name}/{'/'.join(r.section_path)}")
            print(f"    tags={','.join(r.tags)}")
            print(f"    {r.section_body[:140]}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    idx = _load_or_build(args)
    if idx is None:
        return 1
    s = idx.stats()
    print(json.dumps(s))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "stats":
        return _cmd_stats(args)
    return 1
