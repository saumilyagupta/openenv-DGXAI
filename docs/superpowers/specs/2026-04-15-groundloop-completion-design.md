# GroundLoop Completion — Design Spec

**Date:** 2026-04-15
**Sub-project:** combined #4 (lib-grounder) + #5 (interrogator) + #8 (audit-reporter) + final mcp-shell wiring
**Depends on:** all prior sub-projects (skills-scraper, kb-indexer, mcp-shell, python-sandbox, ralph-orchestrator)

---

## 1. Purpose

Ship the three remaining modules as a coordinated final pass, and wire them into the existing mcp-shell stubs to complete the end-to-end GroundLoop MCP server.

## 2. Scope

**In scope:**

1. **`groundloop/lib_grounder/`** (Layer A) — AST-level Python symbol grounding. Verify every `import x`, `from x import y`, and `module.attr` reference in a source string against the running Python environment. Output: `GroundingReport` with lists of grounded / ungrounded symbols.
2. **`groundloop/interrogator/`** — Deterministic Socratic question generator upgrade. Replaces the mcp-shell stub. Uses the kb-indexer to pull the top skill citations relevant to the brief, and generates 3–5 questions targeting the load-bearing assumptions.
3. **`groundloop/audit_reporter/`** — Generates structured audit reports from a `RunResult`. Computes metrics: total iterations, improvement trajectory, skill citation frequencies, hallucination rate (ungrounded claims), final composite score. Text + JSON output.
4. **Wire all three into `groundloop/mcp_shell/tools/*.py`**:
   - `interrogate` → delegate to `Interrogator.generate(brief, index)`.
   - `autonomous_build` → call `run_loop(...)` with `StubSynthesizer` by default, or `OpenAISynthesizer` when `OPENAI_API_KEY` is set. Save the `RunResult` keyed by `run_id` into session state.
   - `audit_report` → call `AuditReporter.build(run_id, session)` returning structured metrics.
   - `ground_check` → optionally augment the Layer B (skills KB) result with Layer A (lib-grounder) when the claim looks like code — detect a code claim by presence of ``` fences or `import ` / `def ` / `class ` keywords. Merge the reports.

**Out of scope:**

- Parallel / branching Ralph experiments (Phase 2).
- Persistent run storage across MCP restarts.
- Prompt caching for OpenAISynthesizer (future).

## 3. Module Contracts

### 3.1 `lib_grounder/`

```
groundloop/lib_grounder/
  __init__.py
  models.py        # GroundingReport, Symbol
  grounder.py      # ground(source: str) -> GroundingReport
  cli.py
```

`Symbol`: `{module: str, attr: str | None, kind: "import" | "attribute", resolved: bool, line: int}`.
`GroundingReport`: `{total_symbols: int, grounded: tuple[Symbol, ...], ungrounded: tuple[Symbol, ...], groundedness: float}` (groundedness = grounded / total, 1.0 if total==0).

Algorithm:
1. Parse source with `ast.parse`.
2. Walk tree, collect `Import`, `ImportFrom`, `Attribute` nodes.
3. For imports, verify via `importlib.util.find_spec`.
4. For `Attribute` nodes where the value is a `Name` matching an import, try `hasattr(module, attr)` after `importlib.import_module(module)` — wrapped in try/except so grounder never raises.
5. Return `GroundingReport`.

### 3.2 `interrogator/`

```
groundloop/interrogator/
  __init__.py
  models.py        # InterrogationResult (questions, cited_node_ids)
  interrogator.py  # Interrogator.generate(brief, index) -> InterrogationResult
  cli.py
```

Algorithm:
1. Query the skills index with the brief, top_k=5.
2. For each citation, extract the `section_title` and any imperative-mood sentence.
3. Template each question as one of:
   - "What is the exact success criterion for ...?"
   - "Have you considered the constraint from <skill_name>: '<section_title>'?"
   - "Which of these assumptions is most likely to be wrong: <list>?"
4. Return 3–5 deterministic questions + the `cited_node_ids` used.

### 3.3 `audit_reporter/`

```
groundloop/audit_reporter/
  __init__.py
  models.py        # AuditReport
  reporter.py      # AuditReporter.build(run: RunResult) -> AuditReport
  cli.py
```

`AuditReport`:
```python
class AuditReport(BaseModel, frozen=True):
    run_id: str
    summary: str                          # human-readable one-liner
    iterations_total: int
    iterations_kept: int
    iterations_regressed: int
    iterations_plateau: int
    skill_citations: tuple[tuple[str, int], ...]   # (node_id, count)
    score_trajectory: tuple[float, ...]           # sandbox_score_after per iter
    final_score: float
    terminated_by: str
    hallucination_rate: float                     # grounded/ungrounded if lib-grounder attached
```

### 3.4 mcp-shell wiring

Each existing handler module in `groundloop/mcp_shell/tools/` is upgraded:
- `interrogate.py` — wraps `Interrogator` instead of the template stub. Fallback to template if no `graph_id` is provided, because interrogate can be called pre-ingest.
- `autonomous_build.py` — executes `run_loop` synchronously (the MCP call is long-lived), stores `RunResult` in `session.runs` keyed by `run_id`. Returns iteration count + final_score + run_id.
- `audit_report.py` — calls `AuditReporter.build` using the stored `RunResult`. Returns the AuditReport as dict.
- `ground_check.py` — detects code claims, calls `lib_grounder.ground(...)` on detected fenced code, merges with existing KB-citation verdict. Confidence recomputed as `softmax(top_k) * groundedness`.

Session state extension:
```python
class SessionState:
    ...
    runs: dict[str, RunResult]  # replaces the RunRecord-only dict; now stores full RunResult after autonomous_build completes
```

Need to adjust `RunRecord` → `RunResult` or keep both; implementation detail.

## 4. Testing

Each module gets its own test suite (≥85% coverage). Wiring tests live in `tests/groundloop/mcp_shell/` as additions to the existing e2e.

Fixture reuse:
- tiny_corpus from kb_indexer for interrogator.
- mock RunResult for audit_reporter.
- synthetic Python source strings for lib_grounder (clean + broken).

## 5. Acceptance Criteria

1. `python -m groundloop.lib_grounder -c "import os"` returns groundedness=1.0.
2. `python -m groundloop.lib_grounder -c "import nonexistent_zzz_pkg"` returns groundedness<1.0 with `nonexistent_zzz_pkg` in ungrounded.
3. `Interrogator(index).generate("build a python api")` returns ≥3 deterministic questions and ≥1 cited_node_id.
4. `AuditReporter.build(sample_run_result)` returns a report whose `iterations_total == len(run.iterations)` and whose `score_trajectory` equals the per-iteration scores.
5. Full mcp-shell e2e: `interrogate → ingest_sources → ground_check(code_claim) → autonomous_build → audit_report` all succeed and the audit report references the run_id from autonomous_build.
6. `ruff check` + `mypy --strict` clean on all four new/updated packages.
7. Overall test suite across ALL groundloop/ packages passes with ≥85% coverage.

## 6. Dependencies

None new. Uses stdlib + pydantic + already-shipped internal packages.

## 7. Deliverables

Three packages + updated mcp-shell handlers + README section "End-to-end GroundLoop".
