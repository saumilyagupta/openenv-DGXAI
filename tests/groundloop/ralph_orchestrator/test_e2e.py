from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def test_e2e_stub_tiny_corpus(tiny_corpus_path: Path, initial_files: Mapping[str, str]) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    synth = StubSynthesizer()
    result = run_loop(
        spec="Build a greet function using pytest patterns",
        initial_files=initial_files, index=idx, synthesizer=synth,
        config=LoopConfig(max_iters=2, target_score=1.1),
    )
    assert len(result.iterations) >= 1
    seen = {nid for it in result.iterations for nid in it.cited_node_ids}
    assert seen
    assert result.final_score >= 0.0
