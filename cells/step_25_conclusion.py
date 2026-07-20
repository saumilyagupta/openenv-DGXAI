"""Cell 25 — Final conclusion cell.

Prints final eval metrics, HF Hub links, pitch summary, and the closing line.
Implements DESIGN.md §13 (Deliverables) + §15 (pitch close) + the canonical
asset table in ``docs/modules/pitch_demo.md`` §2.3.

Pure-print module. No I/O beyond stdout. Tests run :func:`render_conclusion`
and assert that every section header and locked metric appears.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

# ---------------------------------------------------------------------------
# Locked metrics (DESIGN.md §15, pitch_demo.md §3.4 Section 3, §3.1 Beat 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalMetric:
    """One row of the final eval table."""

    name: str
    before: str
    after: str
    delta: str


FINAL_METRICS: tuple[FinalMetric, ...] = (
    FinalMetric(name="Task completion (R1)", before="18%", after="64%", delta="+46pp"),
    FinalMetric(name="Drift detection (R2)", before="8%", after="71%", delta="+63pp"),
    FinalMetric(name="Adaptation latency", before="4.2 turns", after="1.6 turns", delta="-2.6"),
    FinalMetric(name="Anti-hack penalty (R5)", before="0.0", after="0.02", delta="≈ 0"),
    FinalMetric(name="Format compliance (R4)", before="0.41", after="0.92", delta="+0.51"),
)


# ---------------------------------------------------------------------------
# HF Hub links (pitch_demo.md §2.3, DESIGN.md §11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubLink:
    """One row of the HF link table."""

    label: str
    url: str


HUB_LINKS: tuple[HubLink, ...] = (
    HubLink(
        label="Model (LoRA)",
        url="https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora",
    ),
    HubLink(
        label="Dataset",
        url="https://huggingface.co/datasets/driftcall/driftcall-indic-briefs",
    ),
    HubLink(
        label="Env Space",
        url="https://huggingface.co/spaces/driftcall/driftcall-env",
    ),
    HubLink(
        label="Demo Space",
        url="https://huggingface.co/spaces/driftcall/driftcall-demo",
    ),
)


# ---------------------------------------------------------------------------
# Pitch summary (DESIGN.md §15 Beat 5)
# ---------------------------------------------------------------------------


PITCH_SUMMARY: tuple[str, ...] = (
    "Zero voice OpenEnv environments existed before this.",
    "Zero schema-drift environments. Zero Indic environments.",
    "DriftCall is all three in one — Gemma 3n E2B + GRPO + Kokoro + faster-whisper,",
    "200,000 procedural episodes, 5 deterministic rewards, 20 drift patterns,",
    "trained in 14 hours on a single V100.",
)


CLOSING_LINE: str = "Built in 48h, Apache 2.0, see DESIGN.md"


# ---------------------------------------------------------------------------
# Section headers — exactly the strings tests assert against
# ---------------------------------------------------------------------------


HEADER_METRICS: str = "Final eval metrics"
HEADER_LINKS: str = "Hugging Face Hub"
HEADER_PITCH: str = "Pitch summary"
HEADER_CLOSE: str = "Closing"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_metrics_table(metrics: tuple[FinalMetric, ...]) -> str:
    """Plain-text table; no external table lib, deterministic columns."""

    headers = ("Metric", "Before", "After", "Delta")
    rows: list[tuple[str, ...]] = [headers]
    for m in metrics:
        rows.append((m.name, m.before, m.after, m.delta))
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    lines: list[str] = []
    for idx, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append(sep)
    return "\n".join(lines)


def _format_links(links: tuple[HubLink, ...]) -> str:
    width = max(len(link.label) for link in links)
    return "\n".join(f"  {link.label.ljust(width)}  {link.url}" for link in links)


def render_conclusion(
    *,
    metrics: tuple[FinalMetric, ...] = FINAL_METRICS,
    links: tuple[HubLink, ...] = HUB_LINKS,
    pitch: tuple[str, ...] = PITCH_SUMMARY,
    closing: str = CLOSING_LINE,
) -> str:
    """Render the conclusion text (no I/O). Used by ``main`` and tests."""

    parts: list[str] = []
    parts.append(f"=== {HEADER_METRICS} ===")
    parts.append(_format_metrics_table(metrics))
    parts.append("")
    parts.append(f"=== {HEADER_LINKS} ===")
    parts.append(_format_links(links))
    parts.append("")
    parts.append(f"=== {HEADER_PITCH} ===")
    parts.extend(pitch)
    parts.append("")
    parts.append(f"=== {HEADER_CLOSE} ===")
    parts.append(closing)
    return "\n".join(parts)


def main(stream: TextIO | None = None) -> None:
    """Print the conclusion. Defaults to ``sys.stdout``; tests pass StringIO."""

    target = stream if stream is not None else sys.stdout
    target.write(render_conclusion())
    target.write("\n")


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()


__all__ = [
    "CLOSING_LINE",
    "FINAL_METRICS",
    "FinalMetric",
    "HEADER_CLOSE",
    "HEADER_LINKS",
    "HEADER_METRICS",
    "HEADER_PITCH",
    "HUB_LINKS",
    "HubLink",
    "PITCH_SUMMARY",
    "main",
    "render_conclusion",
]
