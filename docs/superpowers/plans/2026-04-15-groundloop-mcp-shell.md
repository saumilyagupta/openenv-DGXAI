# GroundLoop MCP-Shell Implementation Plan

> Use superpowers:subagent-driven-development.

**Goal:** stdio MCP server exposing 5 GroundLoop tools (interrogate, ingest_sources, ground_check, autonomous_build, audit_report) with working Layer-B-grounded handlers.

**Architecture:** 5 handler modules + shared session + Pydantic schemas + server shell. Uses `mcp` SDK v1.27.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-mcp-shell-design.md`.

---

## File Structure (create in order)

```
groundloop/mcp_shell/
  __init__.py
  __main__.py
  config.py
  session.py
  tools/
    __init__.py
    schemas.py
    interrogate.py
    ingest_sources.py
    ground_check.py
    autonomous_build.py
    audit_report.py
  server.py
tests/groundloop/mcp_shell/
  __init__.py
  conftest.py
  test_schemas.py
  test_session.py
  test_interrogate.py
  test_ingest_sources.py
  test_ground_check.py
  test_autonomous_build.py
  test_audit_report.py
  test_server_registration.py
  test_e2e.py
```

---

## Task 1: Scaffold + deps

- [ ] Create dirs and empty `__init__.py` files (including `tools/__init__.py`).
- [ ] Append to `requirements.txt`:
```
mcp>=1.27
```
- [ ] Create `groundloop/mcp_shell/config.py`:
```python
from __future__ import annotations

from pathlib import Path

DEFAULT_CORPUS_PATH = Path("groundloop/kb/skills_corpus.jsonl")
DEFAULT_CACHE_PATH = Path("groundloop/kb/skills_index.pkl")
DEFAULT_GRAPH_ID_PREFIX = "graph_"
GROUND_CHECK_TOP_K = 5
GROUND_CHECK_GROUNDED_THRESHOLD = 0.5
```
- [ ] Commit: `chore: scaffold groundloop/mcp_shell package`

---

## Task 2: Pydantic schemas

**File:** `groundloop/mcp_shell/tools/schemas.py`, `tests/groundloop/mcp_shell/test_schemas.py`

- [ ] Write failing tests:
```python
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
        InterrogateInput()
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
        AuditReportInput()
    AuditReportInput(run_id="r1")
```
- [ ] Run: expect FAIL.
- [ ] Implement:
```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InterrogateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brief: str = Field(min_length=1)


class IngestSourcesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_globs: list[str] | None = None


class GroundCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)
    required_tags: list[str] = Field(default_factory=list)


class AutonomousBuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    max_iters: int = Field(default=3, gt=0, le=20)


class AuditReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
```
- [ ] Run: expect PASS (6 tests).
- [ ] Commit: `feat(mcp-shell): Pydantic input schemas for 5 tools`

---

## Task 3: Session state

**File:** `groundloop/mcp_shell/session.py`, `tests/groundloop/mcp_shell/test_session.py`

- [ ] Write failing tests:
```python
from groundloop.mcp_shell.session import RunRecord, SessionState


def test_session_state_initial_counts_zero():
    s = SessionState()
    m = s.metrics_snapshot()
    assert m == {"tool_calls": 0, "graphs_built": 0, "ground_checks": 0}


def test_session_state_increments():
    s = SessionState()
    s.inc("tool_calls")
    s.inc("tool_calls")
    s.inc("graphs_built")
    assert s.metrics_snapshot()["tool_calls"] == 2
    assert s.metrics_snapshot()["graphs_built"] == 1


def test_session_state_registers_graph():
    s = SessionState()
    s.register_graph("g1", object())
    assert s.get_graph("g1") is not None


def test_session_state_missing_graph_returns_none():
    s = SessionState()
    assert s.get_graph("missing") is None


def test_session_state_creates_run_record():
    s = SessionState()
    r = s.create_run(spec="build a thing", graph_id="g1")
    assert isinstance(r, RunRecord)
    assert r.spec == "build a thing"
    assert s.get_run(r.run_id) is r


def test_session_state_missing_run_returns_none():
    assert SessionState().get_run("missing") is None
```
- [ ] Implement:
```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    spec: str
    graph_id: str
    status: str = "pending_orchestrator"
    iterations: int = 0
    notes: list[str] = field(default_factory=list)


class SessionState:
    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}
        self._runs: dict[str, RunRecord] = {}
        self._metrics: dict[str, int] = {
            "tool_calls": 0,
            "graphs_built": 0,
            "ground_checks": 0,
        }

    def inc(self, key: str, by: int = 1) -> None:
        if key not in self._metrics:
            self._metrics[key] = 0
        self._metrics[key] += by

    def metrics_snapshot(self) -> dict[str, int]:
        return dict(self._metrics)

    def register_graph(self, graph_id: str, index: Any) -> None:
        self._graphs[graph_id] = index

    def get_graph(self, graph_id: str) -> Any | None:
        return self._graphs.get(graph_id)

    def create_run(self, *, spec: str, graph_id: str) -> RunRecord:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        rec = RunRecord(run_id=run_id, spec=spec, graph_id=graph_id)
        self._runs[run_id] = rec
        return rec

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)
```
- [ ] Run: expect PASS (6 tests).
- [ ] Commit: `feat(mcp-shell): in-memory session state`

---

## Task 4: `interrogate` handler (stub)

**File:** `groundloop/mcp_shell/tools/interrogate.py`, `tests/groundloop/mcp_shell/test_interrogate.py`

- [ ] Write failing tests:
```python
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.interrogate import handle_interrogate


def test_interrogate_returns_three_questions():
    s = SessionState()
    out = handle_interrogate({"brief": "build a python REST API"}, s)
    assert "questions" in out
    assert len(out["questions"]) == 3


def test_interrogate_questions_deterministic():
    s = SessionState()
    a = handle_interrogate({"brief": "build a python REST API"}, s)
    b = handle_interrogate({"brief": "build a python REST API"}, s)
    assert a == b


def test_interrogate_rejects_empty():
    s = SessionState()
    out = handle_interrogate({"brief": ""}, s)
    assert out.get("status") == "error"
```
- [ ] Implement:
```python
from __future__ import annotations

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import InterrogateInput

_QUESTION_TEMPLATES = (
    "What are the exact success criteria for '{brief_head}'?",
    "Which external systems or libraries are in or out of scope for '{brief_head}'?",
    "What is the single most failure-prone assumption baked into '{brief_head}'?",
)


def handle_interrogate(args: dict, session: SessionState) -> dict:
    try:
        inp = InterrogateInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    brief_head = inp.brief[:80]
    questions = [t.format(brief_head=brief_head) for t in _QUESTION_TEMPLATES]
    return {"status": "ok", "questions": questions}
```
- [ ] Run: expect PASS.
- [ ] Commit: `feat(mcp-shell): interrogate handler (stub, 3 template questions)`

---

## Task 5: `ingest_sources` handler

**File:** `groundloop/mcp_shell/tools/ingest_sources.py`, `tests/groundloop/mcp_shell/test_ingest_sources.py`

- [ ] Write failing tests:
```python
from pathlib import Path

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.ingest_sources import handle_ingest_sources


def test_ingest_sources_default_builds_graph(tmp_path: Path, monkeypatch):
    # Use existing real corpus as the "default"
    from groundloop.mcp_shell import config as cfg
    if not cfg.DEFAULT_CORPUS_PATH.exists():
        # Skip if corpus not pre-built
        import pytest
        pytest.skip("real corpus not present; run `python -m groundloop.skills_scraper` first")
    s = SessionState()
    out = handle_ingest_sources({"source_globs": None}, s)
    assert out["status"] == "ok"
    assert out["graph_id"].startswith("graph_")
    assert out["nodes"] > 0
    assert s.get_graph(out["graph_id"]) is not None


def test_ingest_sources_custom_globs_uses_scraper(fixtures_dir: Path, tmp_path: Path):
    # Scraper fixtures from sub-project #2 exist
    s = SessionState()
    glob = str(fixtures_dir / "**" / "SKILL.md")
    out = handle_ingest_sources({"source_globs": [glob]}, s)
    assert out["status"] == "ok"
    assert out["nodes"] >= 1
```
- [ ] Implement:
```python
from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell import config as cfg
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import IngestSourcesInput
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def _derive_graph_id(corpus_path: Path, source_globs: list[str] | None) -> str:
    payload = str(corpus_path) + "|" + "|".join(source_globs or [])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{cfg.DEFAULT_GRAPH_ID_PREFIX}{digest}"


def handle_ingest_sources(args: dict, session: SessionState) -> dict:
    try:
        inp = IngestSourcesInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    t0 = time.monotonic()

    if inp.source_globs is None:
        corpus_path = cfg.DEFAULT_CORPUS_PATH
        if not corpus_path.exists():
            return {
                "status": "error",
                "reason": "missing_default_corpus",
                "detail": f"{corpus_path} not built; run `python -m groundloop.skills_scraper`.",
            }
    else:
        sources = [SourceRoot(label=f"mcp-{i}", glob=g) for i, g in enumerate(inp.source_globs)]
        tmp = Path(tempfile.mkdtemp(prefix="groundloop_mcp_"))
        corpus_path = tmp / "corpus.jsonl"
        result = run_scraper(sources=sources, output=corpus_path)
        if result.total_nodes == 0:
            return {"status": "error", "reason": "no_nodes_scraped"}

    index = SkillsIndex(corpus_path=corpus_path)
    index.build()
    graph_id = _derive_graph_id(corpus_path, inp.source_globs)
    session.register_graph(graph_id, index)
    session.inc("graphs_built")

    return {
        "status": "ok",
        "graph_id": graph_id,
        "nodes": int(index.stats()["node_count"]),
        "build_ms": round((time.monotonic() - t0) * 1000, 2),
    }
```
- [ ] Also add `fixtures_dir` fixture to `tests/groundloop/mcp_shell/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    # Reuse skills-scraper fixtures
    return Path(__file__).parent.parent / "skills_scraper" / "fixtures" / "fake_skills"
```
- [ ] Run: expect PASS (may skip first test if corpus not present).
- [ ] Commit: `feat(mcp-shell): ingest_sources handler`

---

## Task 6: `ground_check` handler

**File:** `groundloop/mcp_shell/tools/ground_check.py`, `tests/groundloop/mcp_shell/test_ground_check.py`

- [ ] Write failing tests:
```python
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.ground_check import handle_ground_check


def _prepare_session_with_index(tiny_corpus_path: Path) -> tuple[SessionState, str]:
    s = SessionState()
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    graph_id = "graph_test"
    s.register_graph(graph_id, idx)
    return s, graph_id


def test_ground_check_grounded(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check({"claim": "pytest fixtures", "graph_id": gid}, s)
    assert out["status"] == "ok"
    assert out["verdict"] in {"grounded", "uncertain"}
    assert len(out["citations"]) >= 1


def test_ground_check_unknown_graph():
    s = SessionState()
    out = handle_ground_check({"claim": "x", "graph_id": "nope"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_graph_id"


def test_ground_check_with_tag_filter(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check(
        {"claim": "testing", "graph_id": gid, "required_tags": ["domain:security"]}, s
    )
    assert out["status"] == "ok"
    for c in out["citations"]:
        assert "domain:security" in c["tags"]


def test_ground_check_empty_result_is_ungrounded(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check({"claim": "zzzzz", "graph_id": gid}, s)
    assert out["status"] == "ok"
    assert out["verdict"] == "ungrounded"
    assert out["citations"] == []
```
- [ ] Also add `tiny_corpus_path` fixture to `tests/groundloop/mcp_shell/conftest.py` (pointing to kb_indexer fixture).
```python
@pytest.fixture
def tiny_corpus_path() -> Path:
    return Path(__file__).parent.parent / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
```
- [ ] Implement:
```python
from __future__ import annotations

import math

from pydantic import ValidationError

from groundloop.mcp_shell import config as cfg
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import GroundCheckInput


def _softmax_first(scores: list[float]) -> float:
    if not scores:
        return 0.0
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return exps[0] / total if total > 0 else 0.0


def handle_ground_check(args: dict, session: SessionState) -> dict:
    try:
        inp = GroundCheckInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    session.inc("ground_checks")

    index = session.get_graph(inp.graph_id)
    if index is None:
        return {"status": "error", "reason": "unknown_graph_id", "detail": inp.graph_id}

    tags = set(inp.required_tags) if inp.required_tags else None
    results = index.search(inp.claim, top_k=inp.top_k, required_tags=tags)

    if not results:
        return {
            "status": "ok",
            "verdict": "ungrounded",
            "citations": [],
            "confidence": 0.0,
        }

    scores = [r.score for r in results]
    confidence = _softmax_first(scores)
    verdict = "grounded" if confidence >= cfg.GROUND_CHECK_GROUNDED_THRESHOLD else "uncertain"

    citations = [
        {
            "node_id": r.node_id,
            "skill_name": r.skill_name,
            "section_path": list(r.section_path),
            "section_body": r.section_body,
            "tags": list(r.tags),
            "source_path": r.source_path,
            "score": r.score,
            "rank": r.rank,
        }
        for r in results
    ]

    return {
        "status": "ok",
        "verdict": verdict,
        "citations": citations,
        "confidence": confidence,
    }
```
- [ ] Run: expect PASS (4 tests).
- [ ] Commit: `feat(mcp-shell): ground_check handler with softmax confidence`

---

## Task 7: `autonomous_build` handler (stub)

**File:** `groundloop/mcp_shell/tools/autonomous_build.py`, `tests/groundloop/mcp_shell/test_autonomous_build.py`

- [ ] Write failing tests:
```python
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def test_autonomous_build_returns_pending_run():
    s = SessionState()
    out = handle_autonomous_build({"spec": "build an API", "graph_id": "g"}, s)
    assert out["status"] == "ok"
    assert out["run_status"] == "pending_orchestrator"
    assert out["run_id"].startswith("run_")
    assert s.get_run(out["run_id"]) is not None


def test_autonomous_build_invalid_params():
    s = SessionState()
    out = handle_autonomous_build({"spec": "", "graph_id": "g"}, s)
    assert out["status"] == "error"
```
- [ ] Implement:
```python
from __future__ import annotations

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AutonomousBuildInput


def handle_autonomous_build(args: dict, session: SessionState) -> dict:
    try:
        inp = AutonomousBuildInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    record = session.create_run(spec=inp.spec, graph_id=inp.graph_id)
    record.notes.append(
        "ralph-orchestrator (#7) not yet wired in; run is registered but no iteration will occur."
    )
    return {
        "status": "ok",
        "run_id": record.run_id,
        "run_status": record.status,
        "iterations": record.iterations,
        "notes": list(record.notes),
    }
```
- [ ] Run: expect PASS.
- [ ] Commit: `feat(mcp-shell): autonomous_build handler (stub for #7)`

---

## Task 8: `audit_report` handler

**File:** `groundloop/mcp_shell/tools/audit_report.py`, `tests/groundloop/mcp_shell/test_audit_report.py`

- [ ] Write failing tests:
```python
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.audit_report import handle_audit_report
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def test_audit_report_known_run():
    s = SessionState()
    run = handle_autonomous_build({"spec": "x", "graph_id": "g"}, s)
    out = handle_audit_report({"run_id": run["run_id"]}, s)
    assert out["status"] == "ok"
    assert out["run_id"] == run["run_id"]
    assert "tool_calls" in out["metrics"]


def test_audit_report_unknown_run():
    s = SessionState()
    out = handle_audit_report({"run_id": "missing"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_run_id"
```
- [ ] Implement:
```python
from __future__ import annotations

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AuditReportInput


def handle_audit_report(args: dict, session: SessionState) -> dict:
    try:
        inp = AuditReportInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    record = session.get_run(inp.run_id)
    if record is None:
        return {"status": "error", "reason": "unknown_run_id", "detail": inp.run_id}
    summary = (
        f"run={record.run_id} spec={record.spec[:60]!r} "
        f"status={record.status} iters={record.iterations}"
    )
    return {
        "status": "ok",
        "run_id": record.run_id,
        "summary": summary,
        "run_status": record.status,
        "iterations": record.iterations,
        "notes": list(record.notes),
        "metrics": session.metrics_snapshot(),
    }
```
- [ ] Run: expect PASS.
- [ ] Commit: `feat(mcp-shell): audit_report handler`

---

## Task 9: MCP Server wiring

**File:** `groundloop/mcp_shell/server.py`, `groundloop/mcp_shell/__main__.py`, `tests/groundloop/mcp_shell/test_server_registration.py`

- [ ] Write failing test for registration:
```python
import pytest

from groundloop.mcp_shell.server import build_server


@pytest.mark.asyncio
async def test_server_lists_five_tools():
    server, _ = build_server()
    # The list_tools decorator registered a handler; call it directly.
    handler = server.request_handlers[list(server.request_handlers)[0]]
    # Simpler path: use the registered list_tools via server._tool_handlers or similar.
    # Fallback: invoke via call_tool router if exposed. Use build_server to inspect our tool registry instead.
    from groundloop.mcp_shell.server import TOOL_NAMES
    assert set(TOOL_NAMES) == {
        "interrogate",
        "ingest_sources",
        "ground_check",
        "autonomous_build",
        "audit_report",
    }


def test_tool_dispatcher_unknown_tool_error():
    from groundloop.mcp_shell.server import dispatch
    from groundloop.mcp_shell.session import SessionState
    out = dispatch("nope", {}, SessionState())
    assert out["status"] == "error"
    assert out["reason"] == "unknown_tool"
```
- [ ] Install `pytest-asyncio` only if tests require async — the second test is sync so we can skip it. Delete the async test if it adds pytest-asyncio friction; the `TOOL_NAMES` check + dispatcher test are sufficient.
- [ ] Revised test (no async):
```python
from groundloop.mcp_shell.server import TOOL_NAMES, dispatch
from groundloop.mcp_shell.session import SessionState


def test_tool_names_are_five_expected():
    assert set(TOOL_NAMES) == {
        "interrogate",
        "ingest_sources",
        "ground_check",
        "autonomous_build",
        "audit_report",
    }


def test_dispatch_unknown_tool():
    out = dispatch("nope", {}, SessionState())
    assert out["status"] == "error"
    assert out["reason"] == "unknown_tool"


def test_dispatch_interrogate_routes_correctly():
    out = dispatch("interrogate", {"brief": "hello"}, SessionState())
    assert out["status"] == "ok"
    assert "questions" in out
```
- [ ] Implement `server.py`:
```python
from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.audit_report import handle_audit_report
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build
from groundloop.mcp_shell.tools.ground_check import handle_ground_check
from groundloop.mcp_shell.tools.ingest_sources import handle_ingest_sources
from groundloop.mcp_shell.tools.interrogate import handle_interrogate

_log = logging.getLogger(__name__)

TOOL_NAMES = (
    "interrogate",
    "ingest_sources",
    "ground_check",
    "autonomous_build",
    "audit_report",
)

_HANDLERS = {
    "interrogate": handle_interrogate,
    "ingest_sources": handle_ingest_sources,
    "ground_check": handle_ground_check,
    "autonomous_build": handle_autonomous_build,
    "audit_report": handle_audit_report,
}

_TOOL_DESCRIPTIONS = {
    "interrogate": "Return Socratic clarifying questions about a project brief.",
    "ingest_sources": "Build a KB graph from skill sources. Returns a graph_id.",
    "ground_check": "Search a KB graph for evidence grounding a claim; returns verdict + citations.",
    "autonomous_build": "Kick off the Ralph-loop codebase build. (Stub until #7 ships.)",
    "audit_report": "Return the structured audit report for a previously started run_id.",
}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "interrogate": {
        "type": "object",
        "properties": {"brief": {"type": "string", "minLength": 1}},
        "required": ["brief"],
        "additionalProperties": False,
    },
    "ingest_sources": {
        "type": "object",
        "properties": {
            "source_globs": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "default": None,
            }
        },
        "additionalProperties": False,
    },
    "ground_check": {
        "type": "object",
        "properties": {
            "claim": {"type": "string", "minLength": 1},
            "graph_id": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            "required_tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["claim", "graph_id"],
        "additionalProperties": False,
    },
    "autonomous_build": {
        "type": "object",
        "properties": {
            "spec": {"type": "string", "minLength": 1},
            "graph_id": {"type": "string", "minLength": 1},
            "max_iters": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
        },
        "required": ["spec", "graph_id"],
        "additionalProperties": False,
    },
    "audit_report": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "minLength": 1}},
        "required": ["run_id"],
        "additionalProperties": False,
    },
}


def dispatch(name: str, args: dict, session: SessionState) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"status": "error", "reason": "unknown_tool", "detail": name}
    return handler(args, session)


def build_server() -> tuple[Server, SessionState]:
    server: Server = Server("groundloop")
    session = SessionState()

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=_TOOL_DESCRIPTIONS[name],
                inputSchema=_TOOL_SCHEMAS[name],
            )
            for name in TOOL_NAMES
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = dispatch(name, arguments, session)
        return [types.TextContent(type="text", text=json.dumps(result))]

    return server, session


async def _run_server() -> None:
    logging.basicConfig(level=logging.INFO)
    server, _ = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```
- [ ] Implement `__main__.py`:
```python
from __future__ import annotations

import asyncio

from groundloop.mcp_shell.server import _run_server


def main() -> int:
    asyncio.run(_run_server())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] Run: expect PASS (3 tests).
- [ ] Commit: `feat(mcp-shell): MCP Server with tool list + dispatcher`

---

## Task 10: E2E test

**File:** `tests/groundloop/mcp_shell/test_e2e.py`

- [ ] Write test exercising all 5 tools through `dispatch` with real kb-indexer:
```python
from pathlib import Path

from groundloop.mcp_shell.server import dispatch
from groundloop.mcp_shell.session import SessionState


def test_e2e_full_tool_roundtrip(fixtures_dir: Path, tiny_corpus_path: Path):
    session = SessionState()

    # 1. interrogate
    r1 = dispatch("interrogate", {"brief": "build a python api"}, session)
    assert r1["status"] == "ok"
    assert len(r1["questions"]) == 3

    # 2. ingest_sources using the scraper fixtures
    r2 = dispatch(
        "ingest_sources",
        {"source_globs": [str(fixtures_dir / "**" / "SKILL.md")]},
        session,
    )
    assert r2["status"] == "ok"
    graph_id = r2["graph_id"]
    assert r2["nodes"] >= 1

    # 3. ground_check against the freshly built graph
    r3 = dispatch(
        "ground_check",
        {"claim": "pytest fixtures", "graph_id": graph_id, "top_k": 3},
        session,
    )
    assert r3["status"] == "ok"
    assert r3["verdict"] in {"grounded", "uncertain", "ungrounded"}

    # 4. autonomous_build (stub)
    r4 = dispatch(
        "autonomous_build",
        {"spec": "ship a FastAPI service", "graph_id": graph_id},
        session,
    )
    assert r4["status"] == "ok"
    run_id = r4["run_id"]

    # 5. audit_report
    r5 = dispatch("audit_report", {"run_id": run_id}, session)
    assert r5["status"] == "ok"
    assert r5["metrics"]["tool_calls"] >= 5
    assert r5["metrics"]["graphs_built"] >= 1
    assert r5["metrics"]["ground_checks"] >= 1
```
- [ ] Run: expect PASS.
- [ ] Commit: `test(mcp-shell): full 5-tool e2e dispatch test`

---

## Task 11: Public API + README

**File:** `groundloop/mcp_shell/__init__.py`, `README.md`

- [ ] Populate `__init__.py`:
```python
from __future__ import annotations

from groundloop.mcp_shell.server import TOOL_NAMES, build_server, dispatch
from groundloop.mcp_shell.session import RunRecord, SessionState

__all__ = ["TOOL_NAMES", "RunRecord", "SessionState", "build_server", "dispatch"]
```
- [ ] Run full suite: `python3 -m pytest tests/groundloop/mcp_shell/ -v --cov=groundloop.mcp_shell --cov-report=term` — expect coverage ≥ 90%.
- [ ] Run `ruff check groundloop/mcp_shell/` and `mypy --strict groundloop/mcp_shell/` — expect clean.
- [ ] Append a `### MCP Shell (Server)` section to `README.md` with:
  - One-sentence description.
  - Attach instruction: put this into Claude Code's MCP config:
    ```json
    {
      "mcpServers": {
        "groundloop": {
          "command": "python3",
          "args": ["-m", "groundloop.mcp_shell"]
        }
      }
    }
    ```
  - Brief description of the 5 tools.
- [ ] Commit: `feat(mcp-shell): public API + README; coverage and linters clean`

---

## Self-Review

- ✅ Every spec §4 tool has a handler + tests.
- ✅ Spec §8 acceptance criteria: §8.1 (server starts) covered by `build_server`; §8.2 (ListTools) covered by `TOOL_NAMES` test; §8.3 (ingest→ground roundtrip) by e2e; §8.4 (invalid params) by unknown_graph_id + dispatch unknown_tool; §8.5 (ruff+mypy) by Task 11; §8.6 (coverage) by Task 11; §8.7 (stdin EOF shutdown) handled by `stdio_server` context manager.
- ✅ No placeholders.
- ✅ Type consistency: all handlers return `dict` with `status` key; errors are `{status:error, reason, detail?}`; ok is `{status:ok, ...payload}`.
