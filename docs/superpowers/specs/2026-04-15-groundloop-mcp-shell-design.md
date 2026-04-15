# GroundLoop MCP-Shell — Design Spec

**Date:** 2026-04-15
**Sub-project:** #1 of 8
**Depends on:** skills-scraper (#2, shipped), kb-indexer (#3, shipped)
**Consumed by:** all other sub-projects (they register into this shell)

---

## 1. Purpose

The stdio MCP server that external clients (Claude Code, Cursor, Codex) attach to. Registers the 5 GroundLoop tools. Owns session state: `graph_id` → built `SkillsIndex` mapping.

Each tool ships with a **working initial handler** that integrates what's already built (skills-scraper + kb-indexer). Other sub-projects will extend these handlers as they ship.

## 2. Scope

**In scope:**

- stdio MCP server using the official `mcp` Python SDK (v1.27).
- 5 tools registered with JSON Schema input validation:
  - `interrogate(brief: str) -> {questions: [str]}`
  - `ingest_sources(source_globs: [str]|null) -> {graph_id: str, nodes: int}`
  - `ground_check(claim: str, graph_id: str, top_k: int = 5, required_tags: [str] = []) -> {verdict, citations, confidence}`
  - `autonomous_build(spec: str, graph_id: str, max_iters: int = 3) -> {status, iterations, notes}`
  - `audit_report(run_id: str) -> {summary, metrics}`
- In-memory session state keyed by `graph_id`.
- Graceful shutdown; proper JSON-RPC error responses.
- Entrypoint: `python -m groundloop.mcp_shell`.

**Out of scope (handled by later sub-projects):**

- `autonomous_build`'s actual Ralph loop (→ #7 ralph-orchestrator).
- `interrogate`'s Socratic-question generation (→ #5 interrogator) — shell ships a deterministic stub returning 3 template questions.
- `ground_check`'s full Layer A symbol verification (→ #4 lib-grounder) — shell ships Layer B grounding (skills KB) only for now.
- `audit_report`'s full metric tracking (→ #8 audit-reporter) — shell ships a session-stub version.

Stub handlers MUST return well-formed structured responses so clients can integration-test the full loop immediately.

## 3. Architecture

```
groundloop/mcp_shell/
  __init__.py
  __main__.py         # entrypoint
  server.py           # MCP Server instance + lifecycle
  session.py          # SessionState: graph_id → SkillsIndex cache
  tools/
    __init__.py
    schemas.py        # Pydantic v2 input/output models for all 5 tools
    interrogate.py    # handler (stub: 3 template questions)
    ingest_sources.py # handler (calls skills-scraper + kb-indexer)
    ground_check.py   # handler (kb-indexer search + confidence)
    autonomous_build.py # handler (stub: returns status=pending with notes)
    audit_report.py   # handler (stub: returns session stats)
  config.py           # constants (DEFAULT_CORPUS_PATH, DEFAULT_CACHE_PATH)
tests/groundloop/mcp_shell/
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

Each module ≤ 200 lines.

## 4. Tool contracts

All tools validate input with Pydantic v2 schemas. All output is JSON-serializable dicts.

### 4.1 `interrogate`

Input: `{ "brief": str }`
Output: `{ "questions": [str] }` — 3 deterministic template questions derived from brief length + keyword heuristics. Stub for #5.

### 4.2 `ingest_sources`

Input: `{ "source_globs": list[str] | null }` — if null, uses the scraper's `DEFAULT_SOURCES`.
Behavior: invokes `groundloop.skills_scraper.pipeline.run_scraper(...)`, then `SkillsIndex(corpus_path).build()`, caches it in session under a new `graph_id` (sha256 of corpus path + sources).
Output: `{ "graph_id": str, "nodes": int, "build_ms": float }`

### 4.3 `ground_check`

Input: `{ "claim": str, "graph_id": str, "top_k": int = 5, "required_tags": list[str] = [] }`
Behavior: look up SkillsIndex by `graph_id`, run `search`. Derive a confidence score as `softmax(top_k scores)[0]` — so a sharp single match = high confidence, diffuse results = low.
Output: `{ "verdict": "grounded"|"uncertain"|"ungrounded", "citations": list[SearchResult], "confidence": float }` — verdict thresholds: `grounded` if `confidence ≥ 0.5` and `top score > 0`, `ungrounded` if no citations, `uncertain` otherwise.

### 4.4 `autonomous_build`

Input: `{ "spec": str, "graph_id": str, "max_iters": int = 3 }`
Behavior: stub for #7. Returns a structured placeholder including a `run_id` that `audit_report` will recognize.
Output: `{ "run_id": str, "status": "pending_orchestrator", "iterations": 0, "notes": str }`

### 4.5 `audit_report`

Input: `{ "run_id": str }`
Behavior: looks up run_id in session; returns session-level stats (graphs built, tool calls made).
Output: `{ "run_id": str, "summary": str, "metrics": { "tool_calls": int, "graphs_built": int, "ground_checks": int } }`

## 5. Session state

```python
class SessionState:
    graphs: dict[str, SkillsIndex]    # graph_id -> index
    runs: dict[str, RunRecord]        # run_id -> stub record
    metrics: MetricsCounter            # tool_calls, ground_checks, etc.
```

Per-process, in-memory. New server = fresh session.

## 6. Error handling

- Pydantic validation failure → JSON-RPC invalid_params error.
- Missing `graph_id` in session → JSON-RPC internal_error with structured reason `"unknown_graph_id"`.
- Scraper/indexer failures → wrapped, surfaced through tool response with `status: "error"`.
- Server-level exceptions → logged; MCP SDK handles JSON-RPC response framing.

## 7. Testing

Coverage target: **90%**.

Unit tests per handler (mock SkillsIndex where needed). Schema round-trip tests. Server-registration test verifies all 5 tools appear with correct input schemas. E2E test spawns the server in-process, sends a `ListTools` + a `CallTool` for each of the 5 tools, asserts well-formed responses.

## 8. Acceptance criteria

1. `python -m groundloop.mcp_shell` starts without error (use a test harness that signals SIGTERM after a successful `initialize`).
2. `ListTools` returns exactly 5 tools with matching names.
3. Round-trip `ingest_sources` → `ground_check("pytest fixtures", graph_id)` returns `verdict="grounded"` with a `python-testing` citation.
4. Invalid input (e.g., `ground_check(graph_id="nonexistent")`) returns a structured error, not a crash.
5. `ruff check` + `mypy --strict` clean on `groundloop/mcp_shell/`.
6. Coverage ≥ 90%.
7. Server shuts down cleanly on stdin EOF.

## 9. Dependencies

Already installed: `mcp==1.27.0`, `pydantic>=2`, everything from #2 and #3. Add to `requirements.txt`:

```
mcp>=1.27
```

## 10. Deliverables

Package, test suite, gitignored runtime state, README subsection with `"attach GroundLoop as an MCP server"` instructions.
