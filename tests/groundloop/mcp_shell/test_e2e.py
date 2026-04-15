from __future__ import annotations

from pathlib import Path

from groundloop.mcp_shell.server import dispatch
from groundloop.mcp_shell.session import SessionState


def test_e2e_full_tool_roundtrip(fixtures_dir: Path) -> None:
    session = SessionState()

    # 1. interrogate
    r1 = dispatch("interrogate", {"brief": "build a python api"}, session)
    assert r1["status"] == "ok"
    assert len(r1["questions"]) == 5

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

    # 4. autonomous_build runs the real loop
    r4 = dispatch(
        "autonomous_build",
        {"spec": "ship a FastAPI service", "graph_id": graph_id, "max_iters": 1},
        session,
    )
    assert r4["status"] == "ok"
    run_id = r4["run_id"]
    assert run_id.startswith("ralph_")

    # 5. audit_report references the build's run_id
    r5 = dispatch("audit_report", {"run_id": run_id}, session)
    assert r5["status"] == "ok"
    assert r5["run_id"] == run_id
    assert r5["report"]["run_id"] == run_id
    assert r5["metrics"]["tool_calls"] >= 5
    assert r5["metrics"]["graphs_built"] >= 1
    assert r5["metrics"]["ground_checks"] >= 1
