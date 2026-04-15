from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundloop.interrogator.interrogator import Interrogator
from groundloop.kb_indexer.index import SkillsIndex


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="groundloop.interrogator")
    p.add_argument("--brief", required=True, type=str)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    index: SkillsIndex | None = None
    if args.corpus is not None:
        index = SkillsIndex(corpus_path=args.corpus)
        index.build()

    result = Interrogator(index).generate(args.brief, top_k=args.top_k)
    for q in result.questions:
        print(q)
    if result.cited_node_ids:
        print(f"cited: {', '.join(result.cited_node_ids)}")
    return 0
