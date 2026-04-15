from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.mcp_shell.tools.schemas import (
    AuditReportInput,
    AutonomousBuildInput,
    GroundCheckInput,
    IngestSourcesInput,
    InterrogateInput,
)


def test_interrogate_requires_brief():
    with pytest.raises(ValidationError):
        InterrogateInput()  # type: ignore[call-arg]
    InterrogateInput(brief="hi")


def test_ingest_sources_allows_null_globs():
    assert IngestSourcesInput(source_globs=None).source_globs is None


def test_ground_check_defaults():
    i = GroundCheckInput(claim="x", graph_id="g")
    assert i.top_k == 5
    assert i.required_tags == []


def test_ground_check_top_k_positive():
    with pytest.raises(ValidationError):
        GroundCheckInput(claim="x", graph_id="g", top_k=0)


def test_autonomous_build_defaults():
    a = AutonomousBuildInput(spec="s", graph_id="g")
    assert a.max_iters == 3


def test_audit_report_requires_run_id():
    with pytest.raises(ValidationError):
        AuditReportInput()  # type: ignore[call-arg]
    AuditReportInput(run_id="r1")
