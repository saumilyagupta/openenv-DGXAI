"""Tests for cells/step_20_probe.py.

Covers evaluation.md §2.1 (probe_reward_hacking), §2.3 (render_probe_report_md),
§3.1, §3.6, §3.8, §4.4, §4.5, §5 ProbeOnBaseModelError /
ProbeInsufficientSamplesError. Stubbed training.eval — zero CUDA.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cells.step_18_eval_baseline import (
    DriftDetectionLatency,
    EvalBudgetExceededError,
    EvalReport,
    EvaluationError,
)
from cells.step_20_probe import (
    BUDGET_PROBE_SECONDS,
    EXPLOIT_CLASSES,
    ProbeExploitClassSummary,
    ProbeHit,
    ProbeInsufficientSamplesError,
    ProbeOnBaseModelError,
    ProbeReport,
    probe_reward_hacking,
    render_probe_report_md,
    scan_episode_for_exploits,
    serialize_probe_report,
)


@dataclass(frozen=True)
class _BriefRow:
    episode_id: str


@dataclass(frozen=True)
class _RewardsLike:
    breakdown: dict[str, Any]


def _fake_briefs(n: int = 260) -> list[_BriefRow]:
    return [_BriefRow(episode_id=f"s2_ep_{i:08d}") for i in range(n)]


def _empty_latency() -> DriftDetectionLatency:
    nan = float("nan")
    return DriftDetectionLatency(nan, nan, nan, nan, nan, nan, 0)


def _empty_report(model_path: str = "/ckpt") -> EvalReport:
    return EvalReport(
        model_path=model_path,
        n_episodes=200,
        reward_mean_ci=(0.5, 0.4, 0.6),
        r1_mean_ci=(0.5, 0.4, 0.6),
        r2_mean_ci=(0.5, 0.4, 0.6),
        r3_mean_ci=(0.5, 0.4, 0.6),
        r4_mean_ci=(0.5, 0.4, 0.6),
        r5_mean_ci=(0.0, 0.0, 0.0),
        brier_mean=0.1,
        floor_applied_rate=0.0,
        hallucinated_field_rate=0.0,
        reward_hacking_offenses={},
        drift_detection_latency=_empty_latency(),
        per_language=(),
        breakdown={},
    )


def _rewards_dict_no_offenses(n: int, prefix: str) -> dict[str, _RewardsLike]:
    return {
        f"{prefix}_{i:08d}": _RewardsLike(breakdown={"anti_hack": {"offenses": []}})
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Guards — evaluation.md §5
# ---------------------------------------------------------------------------


def test_probe_on_base_model_raises() -> None:
    briefs = _fake_briefs()
    with pytest.raises(ProbeOnBaseModelError):
        probe_reward_hacking(
            "base", 200, training_eval=MagicMock(), briefs=briefs,
        )


def test_probe_rejects_non_path_checkpoint() -> None:
    briefs = _fake_briefs()
    with pytest.raises(EvaluationError, match="pathlib.Path|got str"):
        probe_reward_hacking(
            "/some/string", 200, training_eval=MagicMock(), briefs=briefs,  # type: ignore[arg-type]
        )


def test_probe_insufficient_samples_raises() -> None:
    briefs = _fake_briefs()
    with pytest.raises(ProbeInsufficientSamplesError, match="n < 50"):
        probe_reward_hacking(
            Path("/ckpt"), 49, training_eval=MagicMock(), briefs=briefs,
        )


def test_probe_rejects_too_few_briefs() -> None:
    briefs = _fake_briefs(60)  # Need 250 for episodes=200
    with pytest.raises(EvaluationError, match=">= 250"):
        probe_reward_hacking(
            Path("/ckpt"), 200, training_eval=MagicMock(), briefs=briefs,
        )


# ---------------------------------------------------------------------------
# Episode-slice selection — evaluation.md §3.1
# ---------------------------------------------------------------------------


def test_probe_reads_rows_50_to_250_disjoint_from_paired_50() -> None:
    briefs = _fake_briefs(260)
    rewards = _rewards_dict_no_offenses(200, "s2_ep")

    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,  # type: ignore[arg-type]
    )

    paired_ids = {b.episode_id for b in briefs[:50]}
    probe_ids = {b.episode_id for b in briefs[50:250]}
    assert paired_ids.isdisjoint(probe_ids)
    # Probe report only references probe slice.
    referenced = {hit.episode_id for hit in report.raw_hits}
    assert referenced.issubset(probe_ids)


# ---------------------------------------------------------------------------
# Scanner mechanics — evaluation.md §3.6
# ---------------------------------------------------------------------------


def test_scan_episode_returns_empty_for_no_offenses() -> None:
    rw = _RewardsLike(breakdown={"anti_hack": {"offenses": []}})
    assert scan_episode_for_exploits("s2_ep_00000001", rw) == []


def test_scan_episode_handles_missing_breakdown() -> None:
    rw = _RewardsLike(breakdown={})
    assert scan_episode_for_exploits("s2_ep_00000001", rw) == []


def test_scan_episode_normalizes_alias_codes() -> None:
    """rewards.md emits 'bare_drift_assertion' but evaluation.md §3.6 normalizes
    that onto 'bare_drift_claim'.
    """
    rw = _RewardsLike(breakdown={
        "anti_hack": {
            "offenses": [
                {"code": "bare_drift_assertion", "turn": 6, "evidence": "drift mentioned"},
            ],
        },
    })
    hits = scan_episode_for_exploits("s2_ep_00000001", rw)
    assert len(hits) == 1
    assert hits[0].exploit_class == "bare_drift_claim"


def test_probe_emits_all_5_known_classes_when_zero_offenses() -> None:
    briefs = _fake_briefs(260)
    rewards = _rewards_dict_no_offenses(200, "s2_ep")
    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,  # type: ignore[arg-type]
    )
    classes_in_report = {row.exploit_class for row in report.per_class}
    for cls in EXPLOIT_CLASSES:
        assert cls in classes_in_report
    for row in report.per_class:
        if row.exploit_class in EXPLOIT_CLASSES:
            assert row.count == 0
            assert row.example_episode_id is None


def test_probe_counts_offenses_and_records_first_example() -> None:
    briefs = _fake_briefs(260)
    rewards: dict[str, Any] = _rewards_dict_no_offenses(200, "s2_ep")
    # Inject offenses into specific episodes.
    rewards["s2_ep_00000050"] = _RewardsLike(
        breakdown={
            "anti_hack": {
                "offenses": [
                    {"code": "hallucinated_field", "turn": 3, "evidence": "total_fare_inr"},
                ],
            },
        },
    )
    rewards["s2_ep_00000099"] = _RewardsLike(
        breakdown={
            "anti_hack": {
                "offenses": [
                    {"code": "hallucinated_field", "turn": 5, "evidence": "booking_ref"},
                ],
            },
        },
    )
    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,
    )
    halluc = next(r for r in report.per_class if r.exploit_class == "hallucinated_field")
    assert halluc.count == 2
    assert halluc.example_episode_id == "s2_ep_00000050"
    assert report.total_hits == 2


# ---------------------------------------------------------------------------
# Novel-class discovery — evaluation.md §3.6 edge case 5
# ---------------------------------------------------------------------------


def test_probe_surfaces_novel_class_with_threshold_1() -> None:
    briefs = _fake_briefs(260)
    rewards: dict[str, Any] = _rewards_dict_no_offenses(200, "s2_ep")
    rewards["s2_ep_00000131"] = _RewardsLike(
        breakdown={
            "anti_hack": {
                "offenses": [
                    {"code": "zero_width_evasion", "turn": 4, "evidence": "ZWJ in message"},
                ],
            },
        },
    )
    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,
    )
    assert report.novel_classes == ("zero_width_evasion",)
    novel_row = next(
        r for r in report.per_class if r.exploit_class == "zero_width_evasion"
    )
    assert novel_row.count == 1


# ---------------------------------------------------------------------------
# Markdown writeup — evaluation.md §2.3, §4.5
# ---------------------------------------------------------------------------


def test_render_probe_report_md_matches_template(tmp_path: Path) -> None:
    briefs = _fake_briefs(260)
    rewards = _rewards_dict_no_offenses(200, "s2_ep")
    report = probe_reward_hacking(
        Path("/ckpt/stage3_final"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,  # type: ignore[arg-type]
        git_sha="a9f2e7b",
        timestamp_ist="2026-04-26T18:42:00+05:30",
    )
    out_path = tmp_path / "probe_report.md"
    render_probe_report_md(report, out_path)
    text = out_path.read_text(encoding="utf-8")

    # Header lines + summary table presence.
    assert "DriftCall — Reward-Hacking Probe Report" in text
    assert "**Model:** `/ckpt/stage3_final`" in text
    assert "**Git SHA:** `a9f2e7b`" in text
    assert "val/briefs.jsonl rows [50:250]" in text

    # All 5 known classes appear as level-3 sections.
    section_count = text.count("### ")
    assert section_count >= 5

    # No-exploit reports must say so.
    assert "**Total offenses:** 0" in text
    assert "**Novel exploit classes:** none" in text
    assert "No LLM-as-judge" in text


def test_render_probe_report_md_flags_unknown_exploit_class(tmp_path: Path) -> None:
    briefs = _fake_briefs(260)
    rewards: dict[str, Any] = _rewards_dict_no_offenses(200, "s2_ep")
    rewards["s2_ep_00000101"] = _RewardsLike(
        breakdown={
            "anti_hack": {
                "offenses": [
                    {"code": "zero_width_evasion", "turn": 1, "evidence": "ZWJ"},
                ],
            },
        },
    )
    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,
    )
    out_path = tmp_path / "probe_report.md"
    render_probe_report_md(report, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "UNKNOWN EXPLOIT CLASS" in text
    assert "zero_width_evasion" in text


def test_probe_report_json_round_trip() -> None:
    briefs = _fake_briefs(260)
    rewards = _rewards_dict_no_offenses(200, "s2_ep")
    report = probe_reward_hacking(
        Path("/ckpt"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,  # type: ignore[arg-type]
    )
    serialized = serialize_probe_report(report)
    parsed = json.loads(serialized)
    assert parsed["n_episodes"] == 200
    assert parsed["total_hits"] == 0
    assert parsed["model_path"] == "/ckpt"
    # canonical serialization is sorted + compact
    serialized_again = serialize_probe_report(report)
    assert serialized == serialized_again


# ---------------------------------------------------------------------------
# Wall-clock budget — evaluation.md §3.8
# ---------------------------------------------------------------------------


def test_probe_budget_60_min_exceeded_raises() -> None:
    briefs = _fake_briefs(260)
    rewards = _rewards_dict_no_offenses(200, "s2_ep")
    fake_clock = iter([0.0, BUDGET_PROBE_SECONDS + 1.0])
    with pytest.raises(EvalBudgetExceededError, match="60 min"):
        probe_reward_hacking(
            Path("/ckpt"),
            200,
            training_eval=MagicMock(return_value=_empty_report()),
            briefs=briefs,
            rewards_by_episode=rewards,  # type: ignore[arg-type]
            monotonic=lambda: next(fake_clock),
        )


# ---------------------------------------------------------------------------
# No LLM-as-judge static AST scan
# ---------------------------------------------------------------------------


def test_no_llm_judge_imports_in_probe_module() -> None:
    src = Path("cells/step_20_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in ("openai", "anthropic", "vertexai"):
                    assert forbidden not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for forbidden in ("openai", "anthropic", "vertexai"):
                assert forbidden not in node.module


# ---------------------------------------------------------------------------
# Integration test — 200-episode probe → markdown
# ---------------------------------------------------------------------------


def test_integration_probe_200_episodes_produces_markdown_and_json(tmp_path: Path) -> None:
    """End-to-end: 200 episodes, mixed offenses, full markdown + JSON artefacts."""
    briefs = _fake_briefs(260)
    rewards: dict[str, Any] = _rewards_dict_no_offenses(200, "s2_ep")
    rewards["s2_ep_00000117"] = _RewardsLike(
        breakdown={"anti_hack": {"offenses": [
            {"code": "hallucinated_field", "turn": 5, "evidence": "total_fare_inr"},
        ]}},
    )
    rewards["s2_ep_00000130"] = _RewardsLike(
        breakdown={"anti_hack": {"offenses": [
            {"code": "hallucinated_field", "turn": 7, "evidence": "booking_ref"},
        ]}},
    )
    # Place the bare-drift offense in the probe slice (index >= 50).
    rewards["s2_ep_00000149"] = _RewardsLike(
        breakdown={"anti_hack": {"offenses": [
            {"code": "bare_drift_assertion", "turn": 6, "evidence": "drift mentioned"},
        ]}},
    )

    report = probe_reward_hacking(
        Path("/abs/path/checkpoints/stage3_final"),
        200,
        training_eval=MagicMock(return_value=_empty_report()),
        briefs=briefs,
        rewards_by_episode=rewards,
        git_sha="a9f2e7b",
        timestamp_ist="2026-04-26T18:42:00+05:30",
    )

    md_path = render_probe_report_md(report, tmp_path / "probe_report.md")
    text = md_path.read_text(encoding="utf-8")
    assert text.count("### ") >= 5
    assert "**Total offenses:** 3" in text
    assert "Novel exploit classes:** none" in text

    # JSON round-trip determinism.
    a = serialize_probe_report(report)
    b = serialize_probe_report(report)
    assert a == b
    assert isinstance(json.loads(a), dict)
