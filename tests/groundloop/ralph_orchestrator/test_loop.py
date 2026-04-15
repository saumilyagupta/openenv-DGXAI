from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig, SynthesisResult


class _NoopSynth:
    def synthesize(self, *, spec, current_files, citations, iteration):  # type: ignore[no-untyped-def]
        return SynthesisResult(
            proposed_files=dict(current_files), rationale="noop", cited_node_ids=(),
        )


class _ImproverSynth:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, *, spec, current_files, citations, iteration):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            new = {**current_files, "main.py": "def ok() -> int:\n    return 1\n"}
        else:
            new = dict(current_files)
        return SynthesisResult(proposed_files=new, rationale="improve", cited_node_ids=())


class _RegressorSynth:
    def synthesize(self, *, spec, current_files, citations, iteration):  # type: ignore[no-untyped-def]
        bad = "import nonexistent_zzz_{i}\n".format(i=iteration)
        return SynthesisResult(
            proposed_files={**current_files, "main.py": bad},
            rationale="regress",
            cited_node_ids=(),
        )


def test_loop_terminates_on_max_iters(tiny_corpus_path: Path, initial_files: Mapping[str, str]) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_NoopSynth(), config=LoopConfig(max_iters=3, target_score=1.1),
    )
    assert result.terminated_by == "max_iters"
    assert len(result.iterations) == 3
    assert all(it.reason == "score_plateau" for it in result.iterations)


def test_loop_target_hit_early(tiny_corpus_path: Path) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    perfect = {"main.py": "from __future__ import annotations\n\n\ndef ok() -> int:\n    return 1\n"}
    result = run_loop(
        spec="x", initial_files=perfect, index=idx,
        synthesizer=_NoopSynth(), config=LoopConfig(max_iters=3, target_score=0.95),
    )
    assert result.terminated_by == "target_hit"
    assert result.iterations == ()


def test_loop_stuck_after_three_regressions(tiny_corpus_path: Path, initial_files: Mapping[str, str]) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_RegressorSynth(), config=LoopConfig(max_iters=10, target_score=1.1),
    )
    assert result.terminated_by == "stuck"
    assert len(result.iterations) == 3
    assert all(it.kept is False for it in result.iterations)


def test_loop_writes_checkpoint(tmp_path: Path, tiny_corpus_path: Path, initial_files: Mapping[str, str]) -> None:
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_NoopSynth(),
        config=LoopConfig(max_iters=1, target_score=1.1),
        checkpoint_dir=tmp_path,
    )
    assert (tmp_path / f"run_{result.run_id}.json").exists()
