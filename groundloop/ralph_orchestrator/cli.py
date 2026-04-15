from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer
from groundloop.ralph_orchestrator.synthesizer import Synthesizer


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.ralph_orchestrator")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the Ralph loop")
    run.add_argument("spec_file", type=Path)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--initial-file", action="append", required=True,
                     help="Repeatable: relname=path/on/disk")
    run.add_argument("--max-iters", type=int, default=5)
    run.add_argument("--target-score", type=float, default=0.95)
    run.add_argument("--checkpoint-dir", type=Path, default=None)
    run.add_argument("--synthesizer", choices=("stub", "openai"), default="stub")
    run.add_argument("--format", choices=("text", "json"), default="text")
    return p.parse_args(argv)


def _build_synth(kind: str) -> Synthesizer:
    if kind == "openai":
        from groundloop.ralph_orchestrator.openai_synthesizer import OpenAISynthesizer
        return OpenAISynthesizer()
    return StubSynthesizer()


def _load_initial(pairs: list[str]) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            return None
        name, path = pair.split("=", 1)
        p = Path(path)
        if not p.is_file():
            return None
        out[name] = p.read_text(encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    initial = _load_initial(args.initial_file)
    if initial is None:
        print("ERROR: one or more --initial-file paths missing or malformed", file=sys.stderr)
        return 1
    spec = args.spec_file.read_text(encoding="utf-8")

    idx = SkillsIndex(corpus_path=args.corpus)
    idx.build()

    try:
        synth = _build_synth(args.synthesizer)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    cfg = LoopConfig(max_iters=args.max_iters, target_score=args.target_score)
    result = run_loop(
        spec=spec, initial_files=initial, index=idx, synthesizer=synth,
        config=cfg, checkpoint_dir=args.checkpoint_dir,
    )
    if args.format == "json":
        print(result.model_dump_json())
    else:
        print(
            f"run_id={result.run_id} terminated_by={result.terminated_by} "
            f"final_score={result.final_score:.3f} iters={len(result.iterations)}"
        )
    return 0
