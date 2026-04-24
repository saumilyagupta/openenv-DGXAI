"""Cell 22 — Markdown summary table (baseline → final → Δ).

Renders the markdown table that drives DESIGN.md §15 pitch 2:00–2:40
"before/after" slide. Per evaluation.md §3.3, §3.4, §3.5:

- Per-reward baseline mean + 95% CI → final mean + 95% CI → paired Δ.
- Per-language breakdown table (n_episodes, reward_mean, R1..R5 means).
- Drift-detection latency before/after row.

Hard rules:
- No LLM-as-judge; static AST scan via ``_NO_LLM_JUDGE_FORBIDDEN_IMPORTS``.
- Every numeric cell rounds to 3 decimals.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cells.step_18_eval_baseline import EvalReport, PerLanguageReport


__all__ = [
    "format_per_language_table",
    "format_per_reward_table",
    "print_summary_table",
]


_NO_LLM_JUDGE_FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {"openai", "anthropic", "vertexai", "google.generativeai", "cohere"},
)

_REWARD_KEYS: tuple[str, ...] = ("reward", "r1", "r2", "r3", "r4", "r5")


def _fmt_ci(triple: tuple[float, float, float]) -> str:
    mean, lo, hi = triple
    if math.isnan(mean):
        return "NaN"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_paired(triple: tuple[float, float, float] | None) -> str:
    if triple is None:
        return "—"
    mean, lo, hi = triple
    if math.isnan(mean):
        return "NaN"
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def format_per_reward_table(baseline: EvalReport, final: EvalReport) -> str:
    """Markdown table: per-reward baseline mean+CI → final mean+CI → Δ with CI."""
    paired_block = final.breakdown.get("paired_ci", {})
    if not isinstance(paired_block, dict):
        paired_block = {}

    lines: list[str] = []
    lines.append("| Reward | Baseline mean [95% CI] | Final mean [95% CI] | Δ paired [95% CI] |")
    lines.append("|--------|------------------------|---------------------|-------------------|")
    for key in _REWARD_KEYS:
        base_ci = getattr(baseline, f"{key}_mean_ci")
        final_ci = getattr(final, f"{key}_mean_ci")
        paired = paired_block.get(key)
        lines.append(
            f"| {key.upper():6s} | {_fmt_ci(base_ci):22s} | "
            f"{_fmt_ci(final_ci):19s} | {_fmt_paired(paired):17s} |",
        )
    return "\n".join(lines)


def _fmt_lang_cell(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.3f}"


def _per_lang_lookup(report: EvalReport) -> dict[str, PerLanguageReport]:
    return {pl.language: pl for pl in report.per_language}


def format_per_language_table(baseline: EvalReport, final: EvalReport) -> str:
    """Markdown table: per-language reward_mean baseline → final."""
    base_lookup = _per_lang_lookup(baseline)
    final_lookup = _per_lang_lookup(final)
    languages = sorted(set(base_lookup) | set(final_lookup))

    lines: list[str] = []
    lines.append(
        "| Language | n_episodes | Baseline reward_mean | Final reward_mean | Δ reward_mean |",
    )
    lines.append(
        "|----------|------------|----------------------|-------------------|---------------|",
    )
    for lang in languages:
        b = base_lookup.get(lang)
        f = final_lookup.get(lang)
        n = max(b.n_episodes if b else 0, f.n_episodes if f else 0)
        b_mean = b.reward_mean if b else float("nan")
        f_mean = f.reward_mean if f else float("nan")
        if math.isnan(b_mean) or math.isnan(f_mean):
            delta_str = "—"
        else:
            delta = f_mean - b_mean
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.3f}"
        lines.append(
            f"| {lang:8s} | {n:10d} | {_fmt_lang_cell(b_mean):20s} | "
            f"{_fmt_lang_cell(f_mean):17s} | {delta_str:13s} |",
        )
    return "\n".join(lines)


def _fmt_latency(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.2f}"


def format_drift_latency_table(baseline: EvalReport, final: EvalReport) -> str:
    """Markdown table: drift-detection latency p50/p95 baseline vs final."""
    bl = baseline.drift_detection_latency
    fl = final.drift_detection_latency
    lines: list[str] = []
    lines.append("| Stage | Baseline p50 | Baseline p95 | Final p50 | Final p95 | Undetected |")
    lines.append("|-------|--------------|--------------|-----------|-----------|------------|")
    lines.append(
        f"| Stage 2 | {_fmt_latency(bl.stage2_median):12s} | "
        f"{_fmt_latency(bl.stage2_p95):12s} | "
        f"{_fmt_latency(fl.stage2_median):9s} | "
        f"{_fmt_latency(fl.stage2_p95):9s} | "
        f"{fl.undetected_count:10d} |",
    )
    lines.append(
        f"| Stage 3 | {_fmt_latency(bl.stage3_median):12s} | "
        f"{_fmt_latency(bl.stage3_p95):12s} | "
        f"{_fmt_latency(fl.stage3_median):9s} | "
        f"{_fmt_latency(fl.stage3_p95):9s} | "
        f"{bl.undetected_count:10d} |",
    )
    return "\n".join(lines)


def print_summary_table(baseline: EvalReport, final: EvalReport) -> str:
    """Top-level entry point — emit the full multi-section markdown summary."""
    sections: list[str] = []
    sections.append("# DriftCall — Baseline → Final summary")
    sections.append("")
    sections.append(f"**Baseline model:** `{baseline.model_path}`")
    sections.append(f"**Final model:** `{final.model_path}`")
    sections.append(f"**Episodes:** baseline {baseline.n_episodes}, final {final.n_episodes}")
    sections.append("")
    sections.append("## Per-reward (mean + 95% CI)")
    sections.append("")
    sections.append(format_per_reward_table(baseline, final))
    sections.append("")
    sections.append("## Per-language breakdown")
    sections.append("")
    sections.append(format_per_language_table(baseline, final))
    sections.append("")
    sections.append("## Drift-detection latency")
    sections.append("")
    sections.append(format_drift_latency_table(baseline, final))
    sections.append("")

    # Reward-hacking offenses summary (DESIGN.md §15 pitch).
    sections.append("## Reward-hacking offenses (final vs baseline)")
    sections.append("")
    sections.append("| Class | Baseline | Final |")
    sections.append("|-------|----------|-------|")
    keys = sorted(set(baseline.reward_hacking_offenses) | set(final.reward_hacking_offenses))
    for key in keys:
        b_count = baseline.reward_hacking_offenses.get(key, 0)
        f_count = final.reward_hacking_offenses.get(key, 0)
        sections.append(f"| {key:22s} | {b_count:8d} | {f_count:5d} |")
    sections.append("")
    return "\n".join(sections)
