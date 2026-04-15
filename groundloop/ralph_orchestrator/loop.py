from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop.ralph_orchestrator.checkpoint import save_checkpoint
from groundloop.ralph_orchestrator.models import (
    Iteration,
    IterationReason,
    LoopConfig,
    RunResult,
    TerminationReason,
)
from groundloop.ralph_orchestrator.synthesizer import Synthesizer

_log = logging.getLogger(__name__)
_STUCK_THRESHOLD = 3


def _score_files(files: Mapping[str, str], config: LoopConfig) -> float:
    try:
        result = run_sandbox(
            files=dict(files), tools=config.tools, timeout_per_tool=config.timeout_per_tool,
        )
    except Exception as e:  # noqa: BLE001 - sandbox errors must not kill the loop
        _log.exception("sandbox error: %s", e)
        return 0.0
    return result.composite_score


def run_loop(
    *,
    spec: str,
    initial_files: Mapping[str, str],
    index: SkillsIndex,
    synthesizer: Synthesizer,
    config: LoopConfig | None = None,
    checkpoint_dir: Path | None = None,
) -> RunResult:
    cfg = config or LoopConfig()
    run_id = f"ralph_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    current: dict[str, str] = dict(initial_files)
    iterations: list[Iteration] = []
    consecutive_regressions = 0
    terminated_by: TerminationReason = "max_iters"

    for i in range(cfg.max_iters):
        score_before = _score_files(current, cfg)
        if score_before >= cfg.target_score:
            terminated_by = "target_hit"
            break

        citations = index.search(spec, top_k=cfg.top_k_citations)

        synth_reason: IterationReason | None = None
        try:
            synth = synthesizer.synthesize(
                spec=spec, current_files=current, citations=citations, iteration=i,
            )
        except Exception as e:  # noqa: BLE001 - synthesizer errors must not kill the loop
            _log.exception("synthesizer error: %s", e)
            synth = None
            synth_reason = "synthesizer_error"

        if synth is None:
            iterations.append(
                Iteration(
                    index=i, cited_node_ids=(), rationale="synth_error",
                    proposed_files=current, sandbox_score_before=score_before,
                    sandbox_score_after=score_before, kept=False,
                    reason=synth_reason or "synthesizer_error",
                )
            )
            consecutive_regressions += 1
        else:
            score_after = _score_files(synth.proposed_files, cfg)
            reason: IterationReason
            if score_after > score_before:
                kept = True
                reason = "score_improved"
                consecutive_regressions = 0
                current = dict(synth.proposed_files)
            elif score_after < score_before:
                kept = False
                reason = "score_regressed"
                consecutive_regressions += 1
            else:
                kept = False
                reason = "score_plateau"
                consecutive_regressions = 0
            iterations.append(
                Iteration(
                    index=i,
                    cited_node_ids=synth.cited_node_ids,
                    rationale=synth.rationale,
                    proposed_files=synth.proposed_files,
                    sandbox_score_before=score_before,
                    sandbox_score_after=score_after,
                    kept=kept,
                    reason=reason,
                )
            )

        if checkpoint_dir is not None:
            try:
                save_checkpoint(
                    RunResult(
                        run_id=run_id, spec=spec, started_at=started_at,
                        ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        final_score=iterations[-1].sandbox_score_after,
                        final_files=current, iterations=tuple(iterations),
                        terminated_by="in_progress",
                    ),
                    checkpoint_dir,
                )
            except OSError as e:
                _log.warning("checkpoint write failed: %s", e)

        if consecutive_regressions >= _STUCK_THRESHOLD:
            terminated_by = "stuck"
            break

    final_score = (
        iterations[-1].sandbox_score_after if iterations else _score_files(current, cfg)
    )
    result = RunResult(
        run_id=run_id, spec=spec, started_at=started_at,
        ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
        final_score=final_score, final_files=current,
        iterations=tuple(iterations), terminated_by=terminated_by,
    )
    if checkpoint_dir is not None:
        try:
            save_checkpoint(result, checkpoint_dir)
        except OSError as e:
            _log.warning("final checkpoint write failed: %s", e)
    return result
