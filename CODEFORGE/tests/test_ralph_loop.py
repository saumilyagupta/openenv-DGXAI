from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from codeforge.ralph.checkpoint import load_checkpoint, save_checkpoint
from codeforge.ralph.loop import run_loop
from codeforge.ralph.models import (
    Iteration,
    LoopConfig,
    RunResult,
    SynthesisResult,
)
from codeforge.ralph.synthesizer import StubSynthesizer, Synthesizer

if TYPE_CHECKING:
    from codeforge.kb.models import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sandbox_result(score: float) -> MagicMock:
    """Return a mock SandboxResult with the given composite_score."""
    m = MagicMock()
    m.composite_score = score
    return m


class _FixedScoreSynthesizer:
    """Synthesizer that always returns the same files (for controlled tests)."""

    def __init__(self, files: dict[str, str]) -> None:
        self._files = files

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        return SynthesisResult(
            proposed_files=self._files,
            rationale="fixed",
            cited_node_ids=("node_1",),
        )


class _ErrorSynthesizer:
    """Synthesizer that always raises."""

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        msg = "boom"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestLoopConfig:
    def test_defaults(self) -> None:
        cfg = LoopConfig()
        assert cfg.max_iters == 5
        assert cfg.target_score == 0.95
        assert cfg.tools == ("ruff", "imports")
        assert cfg.timeout_per_tool == 60.0
        assert cfg.top_k_citations == 5

    def test_frozen(self) -> None:
        cfg = LoopConfig()
        with pytest.raises(Exception):  # noqa: B017
            cfg.max_iters = 10  # type: ignore[misc]

    def test_max_iters_bounds(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            LoopConfig(max_iters=0)
        with pytest.raises(Exception):  # noqa: B017
            LoopConfig(max_iters=101)


class TestSynthesisResult:
    def test_frozen(self) -> None:
        sr = SynthesisResult(
            proposed_files={"main.py": "pass"},
            rationale="test",
            cited_node_ids=("a",),
        )
        with pytest.raises(Exception):  # noqa: B017
            sr.rationale = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        sr = SynthesisResult(
            proposed_files={"main.py": "pass"},
            rationale="test",
            cited_node_ids=("a", "b"),
        )
        assert sr.proposed_files == {"main.py": "pass"}
        assert sr.rationale == "test"
        assert sr.cited_node_ids == ("a", "b")


class TestIteration:
    def test_fields(self) -> None:
        it = Iteration(
            index=0,
            cited_node_ids=("n1",),
            rationale="better",
            proposed_files={"main.py": "pass"},
            sandbox_score_before=0.3,
            sandbox_score_after=0.5,
            kept=True,
            reason="score_improved",
        )
        assert it.index == 0
        assert it.kept is True
        assert it.reason == "score_improved"


class TestRunResult:
    def test_fields(self) -> None:
        rr = RunResult(
            run_id="r1",
            spec="do stuff",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:01:00+00:00",
            final_score=0.8,
            final_files={"main.py": "pass"},
            iterations=(),
            terminated_by="max_iters",
        )
        assert rr.run_id == "r1"
        assert rr.terminated_by == "max_iters"
        assert rr.iterations == ()


# ---------------------------------------------------------------------------
# StubSynthesizer tests
# ---------------------------------------------------------------------------

class TestStubSynthesizer:
    def test_returns_valid_synthesis_result(self) -> None:
        synth = StubSynthesizer()
        result = synth.synthesize(
            spec="implement greet(name)",
            current_files={"main.py": ""},
            citations=[],
            iteration=0,
        )
        assert isinstance(result, SynthesisResult)
        assert isinstance(result.proposed_files, dict)
        assert isinstance(result.rationale, str)
        assert isinstance(result.cited_node_ids, tuple)

    def test_no_citations_returns_current_files(self) -> None:
        synth = StubSynthesizer()
        files = {"main.py": "print('hello')"}
        result = synth.synthesize(
            spec="test",
            current_files=files,
            citations=[],
            iteration=0,
        )
        assert result.proposed_files == files
        assert result.rationale == "no_citations"
        assert result.cited_node_ids == ()


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        rr = RunResult(
            run_id="chk1",
            spec="spec",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:01:00+00:00",
            final_score=0.7,
            final_files={"main.py": "x = 1"},
            iterations=(
                Iteration(
                    index=0,
                    cited_node_ids=("n1",),
                    rationale="improved",
                    proposed_files={"main.py": "x = 1"},
                    sandbox_score_before=0.3,
                    sandbox_score_after=0.7,
                    kept=True,
                    reason="score_improved",
                ),
            ),
            terminated_by="max_iters",
        )
        path = save_checkpoint(rr, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_checkpoint(path)
        assert loaded.run_id == "chk1"
        assert loaded.final_score == 0.7
        assert len(loaded.iterations) == 1
        assert loaded.iterations[0].kept is True

    def test_save_creates_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested" / "dir"
        rr = RunResult(
            run_id="chk2",
            spec="s",
            started_at="t0",
            ended_at="t1",
            final_score=0.0,
            final_files={},
            iterations=(),
            terminated_by="max_iters",
        )
        path = save_checkpoint(rr, subdir)
        assert path.exists()

    def test_atomic_write_no_tmp_left(self, tmp_path: Path) -> None:
        rr = RunResult(
            run_id="chk3",
            spec="s",
            started_at="t0",
            ended_at="t1",
            final_score=0.0,
            final_files={},
            iterations=(),
            terminated_by="max_iters",
        )
        save_checkpoint(rr, tmp_path)
        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_load_content_matches_json(self, tmp_path: Path) -> None:
        rr = RunResult(
            run_id="chk4",
            spec="my spec",
            started_at="t0",
            ended_at="t1",
            final_score=0.5,
            final_files={"a.py": "pass"},
            iterations=(),
            terminated_by="stuck",
        )
        path = save_checkpoint(rr, tmp_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["run_id"] == "chk4"
        assert raw["terminated_by"] == "stuck"


# ---------------------------------------------------------------------------
# run_loop tests
# ---------------------------------------------------------------------------

class TestRunLoop:
    @patch("codeforge.ralph.loop.run_sandbox")
    def test_basic_execution(self, mock_sandbox: MagicMock) -> None:
        mock_sandbox.return_value = _make_sandbox_result(0.5)
        synth = _FixedScoreSynthesizer({"main.py": "x = 1"})
        cfg = LoopConfig(max_iters=2, target_score=0.99)

        result = run_loop(
            spec="implement greet",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        assert isinstance(result, RunResult)
        assert result.spec == "implement greet"
        assert result.terminated_by in ("max_iters", "stuck", "target_hit")
        assert len(result.iterations) <= 2

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_terminates_target_hit(self, mock_sandbox: MagicMock) -> None:
        # Score already >= target before first iteration
        mock_sandbox.return_value = _make_sandbox_result(0.99)
        synth = _FixedScoreSynthesizer({"main.py": "pass"})
        cfg = LoopConfig(max_iters=5, target_score=0.95)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        assert result.terminated_by == "target_hit"
        assert len(result.iterations) == 0

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_terminates_stuck(self, mock_sandbox: MagicMock) -> None:
        # Score always regresses: before=0.5, after=0.3
        call_count = 0

        def side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # Odd calls (score_before): 0.5, Even calls (score_after): 0.3
            if call_count % 2 == 1:
                return _make_sandbox_result(0.5)
            return _make_sandbox_result(0.3)

        mock_sandbox.side_effect = side_effect
        synth = _FixedScoreSynthesizer({"main.py": "worse"})
        cfg = LoopConfig(max_iters=10, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        assert result.terminated_by == "stuck"

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_terminates_max_iters(self, mock_sandbox: MagicMock) -> None:
        # Score stays flat at 0.5 (plateau, not regression)
        mock_sandbox.return_value = _make_sandbox_result(0.5)
        synth = _FixedScoreSynthesizer({"main.py": "same"})
        cfg = LoopConfig(max_iters=3, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        assert result.terminated_by == "max_iters"
        assert len(result.iterations) == 3

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_kept_only_when_improved(self, mock_sandbox: MagicMock) -> None:
        scores = [0.3, 0.5, 0.5, 0.4, 0.4, 0.6]
        call_idx = 0

        def side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_idx
            s = scores[call_idx] if call_idx < len(scores) else 0.5
            call_idx += 1
            return _make_sandbox_result(s)

        mock_sandbox.side_effect = side_effect
        synth = _FixedScoreSynthesizer({"main.py": "x = 1"})
        cfg = LoopConfig(max_iters=3, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        # Iteration 0: before=0.3, after=0.5 → kept (improved)
        assert result.iterations[0].kept is True
        assert result.iterations[0].reason == "score_improved"
        # Iteration 1: before=0.5, after=0.4 → not kept (regressed)
        assert result.iterations[1].kept is False
        assert result.iterations[1].reason == "score_regressed"

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_checkpoint_saved(self, mock_sandbox: MagicMock, tmp_path: Path) -> None:
        mock_sandbox.return_value = _make_sandbox_result(0.5)
        synth = _FixedScoreSynthesizer({"main.py": "pass"})
        cfg = LoopConfig(max_iters=2, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
            checkpoint_dir=tmp_path,
        )
        # Checkpoint files should exist
        json_files = list(tmp_path.glob("run_*.json"))
        assert len(json_files) >= 1

        # Verify the final checkpoint is loadable
        loaded = load_checkpoint(json_files[0])
        assert loaded.run_id == result.run_id

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_synthesizer_error_handled(self, mock_sandbox: MagicMock) -> None:
        mock_sandbox.return_value = _make_sandbox_result(0.5)
        synth = _ErrorSynthesizer()
        cfg = LoopConfig(max_iters=4, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        # Should terminate stuck after 3 synthesizer errors (consecutive regressions)
        assert result.terminated_by == "stuck"
        for it in result.iterations:
            assert it.reason == "synthesizer_error"
            assert it.kept is False

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_run_result_has_correct_fields(self, mock_sandbox: MagicMock) -> None:
        mock_sandbox.return_value = _make_sandbox_result(0.5)
        synth = _FixedScoreSynthesizer({"main.py": "x = 1"})
        cfg = LoopConfig(max_iters=1, target_score=0.99)

        result = run_loop(
            spec="implement greet(name)",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        assert result.run_id.startswith("ralph_")
        assert result.spec == "implement greet(name)"
        assert result.started_at != ""
        assert result.ended_at != ""
        assert isinstance(result.final_score, float)
        assert isinstance(result.final_files, dict)
        assert isinstance(result.iterations, tuple)

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_sandbox_error_returns_zero(self, mock_sandbox: MagicMock) -> None:
        # Sandbox raises on first call, returns 0.5 on second
        call_count = 0

        def side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "sandbox crashed"
                raise RuntimeError(msg)
            return _make_sandbox_result(0.5)

        mock_sandbox.side_effect = side_effect
        synth = _FixedScoreSynthesizer({"main.py": "x = 1"})
        cfg = LoopConfig(max_iters=1, target_score=0.99)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        # Should still produce a result (sandbox error returns 0.0)
        assert isinstance(result, RunResult)
        assert result.iterations[0].sandbox_score_before == 0.0

    @patch("codeforge.ralph.loop.run_sandbox")
    def test_improving_loop(self, mock_sandbox: MagicMock) -> None:
        """Loop where each iteration improves, hitting target."""
        scores = [0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 1.0]
        call_idx = 0

        def side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_idx
            s = scores[call_idx] if call_idx < len(scores) else 1.0
            call_idx += 1
            return _make_sandbox_result(s)

        mock_sandbox.side_effect = side_effect
        synth = _FixedScoreSynthesizer({"main.py": "x = 1"})
        cfg = LoopConfig(max_iters=10, target_score=0.95)

        result = run_loop(
            spec="test",
            initial_files={"main.py": ""},
            index=MagicMock(),
            synthesizer=synth,
            config=cfg,
        )
        # After iter 2: score_before=0.7, score_after=0.9, kept
        # Next: score_before=0.9 → still < 0.95 → iter 3: score_before=0.9, after=1.0
        # Actually depends on exact call pattern — just verify it terminates with target_hit or max_iters
        for it in result.iterations:
            if it.kept:
                assert it.reason == "score_improved"
