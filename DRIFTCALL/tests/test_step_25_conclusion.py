"""Tests for cells/step_25_conclusion.py.

Smoke + content tests for the final-cell renderer. Covers:
  - main() runs and writes to a caller-supplied StringIO.
  - Every section header appears.
  - Every locked metric (R1, R2, latency) appears with both before+after.
  - Every HF Hub URL appears.
  - The closing line appears verbatim.
  - Frozen dataclasses cannot be mutated.
  - No pragmas in the source.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest

from cells import step_25_conclusion as conclusion
from cells.step_25_conclusion import (
    CLOSING_LINE,
    FINAL_METRICS,
    HEADER_CLOSE,
    HEADER_LINKS,
    HEADER_METRICS,
    HEADER_PITCH,
    HUB_LINKS,
    PITCH_SUMMARY,
    FinalMetric,
    HubLink,
    main,
    render_conclusion,
)

# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_final_metric_is_frozen(self) -> None:
        m = FinalMetric(name="x", before="1", after="2", delta="+1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.name = "y"

    def test_hub_link_is_frozen(self) -> None:
        link = HubLink(label="x", url="https://x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            link.url = "https://y"

    def test_final_metrics_tuple_count_and_types(self) -> None:
        assert len(FINAL_METRICS) == 5
        for m in FINAL_METRICS:
            assert isinstance(m, FinalMetric)

    def test_hub_links_count_and_types(self) -> None:
        assert len(HUB_LINKS) == 4
        for link in HUB_LINKS:
            assert isinstance(link, HubLink)


# ---------------------------------------------------------------------------
# Locked content
# ---------------------------------------------------------------------------


class TestLockedMetrics:
    def test_r1_metric_present(self) -> None:
        names = {m.name for m in FINAL_METRICS}
        assert "Task completion (R1)" in names

    def test_r2_metric_present(self) -> None:
        names = {m.name for m in FINAL_METRICS}
        assert "Drift detection (R2)" in names

    def test_latency_metric_present(self) -> None:
        names = {m.name for m in FINAL_METRICS}
        assert "Adaptation latency" in names

    def test_r1_before_after_match_design_15(self) -> None:
        r1 = next(m for m in FINAL_METRICS if m.name == "Task completion (R1)")
        assert r1.before == "18%"
        assert r1.after == "64%"

    def test_r2_before_after_match_design_15(self) -> None:
        r2 = next(m for m in FINAL_METRICS if m.name == "Drift detection (R2)")
        assert r2.before == "8%"
        assert r2.after == "71%"

    def test_latency_before_after_match_design_15(self) -> None:
        lat = next(m for m in FINAL_METRICS if m.name == "Adaptation latency")
        assert lat.before == "4.2 turns"
        assert lat.after == "1.6 turns"


class TestLockedLinks:
    def test_lora_repo_url(self) -> None:
        urls = {link.url for link in HUB_LINKS}
        assert "https://huggingface.co/DGXAI/gemma-4-e2b-driftcall-lora" in urls

    def test_dataset_repo_url(self) -> None:
        urls = {link.url for link in HUB_LINKS}
        assert "https://huggingface.co/datasets/driftcall/driftcall-indic-briefs" in urls

    def test_env_space_url(self) -> None:
        urls = {link.url for link in HUB_LINKS}
        assert "https://huggingface.co/spaces/driftcall/driftcall-env" in urls

    def test_demo_space_url(self) -> None:
        urls = {link.url for link in HUB_LINKS}
        assert "https://huggingface.co/spaces/driftcall/driftcall-demo" in urls


class TestLockedClosing:
    def test_closing_verbatim(self) -> None:
        assert CLOSING_LINE == "Built in 48h, Apache 2.0, see DESIGN.md"

    def test_pitch_summary_nonempty(self) -> None:
        assert len(PITCH_SUMMARY) >= 3
        for line in PITCH_SUMMARY:
            assert isinstance(line, str)
            assert line.strip() != ""


# ---------------------------------------------------------------------------
# render_conclusion
# ---------------------------------------------------------------------------


class TestRenderConclusion:
    def test_returns_string(self) -> None:
        text = render_conclusion()
        assert isinstance(text, str)
        assert text != ""

    def test_contains_metrics_header(self) -> None:
        assert HEADER_METRICS in render_conclusion()

    def test_contains_links_header(self) -> None:
        assert HEADER_LINKS in render_conclusion()

    def test_contains_pitch_header(self) -> None:
        assert HEADER_PITCH in render_conclusion()

    def test_contains_close_header(self) -> None:
        assert HEADER_CLOSE in render_conclusion()

    def test_contains_closing_line(self) -> None:
        assert CLOSING_LINE in render_conclusion()

    def test_contains_all_metric_names(self) -> None:
        text = render_conclusion()
        for m in FINAL_METRICS:
            assert m.name in text

    def test_contains_all_metric_before_after(self) -> None:
        text = render_conclusion()
        for m in FINAL_METRICS:
            assert m.before in text
            assert m.after in text

    def test_contains_all_link_urls(self) -> None:
        text = render_conclusion()
        for link in HUB_LINKS:
            assert link.url in text

    def test_contains_all_pitch_lines(self) -> None:
        text = render_conclusion()
        for line in PITCH_SUMMARY:
            assert line in text

    def test_render_is_pure_for_default_args(self) -> None:
        a = render_conclusion()
        b = render_conclusion()
        assert a == b

    def test_render_with_overrides(self) -> None:
        custom_metrics = (FinalMetric(name="custom", before="0", after="1", delta="+1"),)
        text = render_conclusion(metrics=custom_metrics)
        assert "custom" in text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_to_provided_stream(self) -> None:
        buf = io.StringIO()
        main(stream=buf)
        text = buf.getvalue()
        assert HEADER_METRICS in text
        assert HEADER_CLOSE in text
        assert CLOSING_LINE in text

    def test_main_terminates_with_newline(self) -> None:
        buf = io.StringIO()
        main(stream=buf)
        assert buf.getvalue().endswith("\n")

    def test_main_uses_stdout_when_no_stream(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main()
        captured = capsys.readouterr()
        assert HEADER_METRICS in captured.out
        assert CLOSING_LINE in captured.out


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_render_conclusion_callable(self) -> None:
        assert callable(render_conclusion)

    def test_main_callable(self) -> None:
        assert callable(main)

    def test_module_no_pragma_violations(self) -> None:
        text = Path(conclusion.__file__).read_text(encoding="utf-8")
        # Pragmas are forbidden in cell sources (typing escape hatches).
        # An idiomatic `pragma: no cover` on the __main__ guard is fine.
        forbidden_marker_a = "type" + ": " + "ignore"
        forbidden_marker_b = "# " + "noqa"
        assert forbidden_marker_a not in text
        assert forbidden_marker_b not in text
