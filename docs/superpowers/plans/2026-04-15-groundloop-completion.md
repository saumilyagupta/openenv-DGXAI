# GroundLoop Completion Implementation Plan

> Use superpowers:subagent-driven-development.

**Goal:** Ship lib-grounder (#4), interrogator (#5), audit-reporter (#8), then wire all three into mcp-shell to complete end-to-end GroundLoop.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-completion-design.md`.

---

## Task 1: lib-grounder — scaffold + models + grounder + CLI + tests

**Files:** `groundloop/lib_grounder/{__init__.py,__main__.py,models.py,grounder.py,cli.py}`, `tests/groundloop/lib_grounder/{__init__.py,conftest.py,test_models.py,test_grounder.py,test_cli.py}`.

- [ ] Scaffold dirs.
- [ ] `models.py` — Pydantic v2 frozen:
  ```python
  from __future__ import annotations

  from typing import Literal

  from pydantic import BaseModel, ConfigDict


  class Symbol(BaseModel):
      model_config = ConfigDict(frozen=True)
      module: str
      attr: str | None
      kind: Literal["import", "attribute"]
      resolved: bool
      line: int


  class GroundingReport(BaseModel):
      model_config = ConfigDict(frozen=True)
      total_symbols: int
      grounded: tuple[Symbol, ...]
      ungrounded: tuple[Symbol, ...]
      groundedness: float
  ```
- [ ] `grounder.py` — `ground(source: str) -> GroundingReport`:
  ```python
  from __future__ import annotations

  import ast
  import importlib
  import importlib.util
  import logging

  from groundloop.lib_grounder.models import GroundingReport, Symbol

  _log = logging.getLogger(__name__)


  def _module_spec(name: str) -> bool:
      try:
          return importlib.util.find_spec(name) is not None
      except (ImportError, ValueError):
          return False


  def _has_attr(module: str, attr: str) -> bool:
      try:
          mod = importlib.import_module(module)
      except Exception:  # noqa: BLE001 - defensive; any import-time failure = unresolved
          return False
      return hasattr(mod, attr)


  def ground(source: str) -> GroundingReport:
      try:
          tree = ast.parse(source)
      except SyntaxError:
          return GroundingReport(total_symbols=0, grounded=(), ungrounded=(), groundedness=1.0)

      symbols: list[Symbol] = []
      import_to_module: dict[str, str] = {}

      for node in ast.walk(tree):
          if isinstance(node, ast.Import):
              for alias in node.names:
                  pkg = alias.name.split(".")[0]
                  resolved = _module_spec(pkg)
                  symbols.append(Symbol(module=alias.name, attr=None, kind="import",
                                        resolved=resolved, line=node.lineno))
                  import_to_module[alias.asname or pkg] = alias.name
          elif isinstance(node, ast.ImportFrom):
              if node.level != 0 or node.module is None:
                  continue
              resolved_mod = _module_spec(node.module)
              for alias in node.names:
                  attr_resolved = resolved_mod and _has_attr(node.module, alias.name)
                  symbols.append(Symbol(module=node.module, attr=alias.name, kind="import",
                                        resolved=attr_resolved, line=node.lineno))

      for node in ast.walk(tree):
          if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
              base = node.value.id
              mod_name = import_to_module.get(base)
              if mod_name is None:
                  continue
              top = mod_name.split(".")[0]
              resolved = _has_attr(top, node.attr)
              symbols.append(Symbol(module=mod_name, attr=node.attr, kind="attribute",
                                    resolved=resolved, line=node.lineno))

      grounded = tuple(s for s in symbols if s.resolved)
      ungrounded = tuple(s for s in symbols if not s.resolved)
      total = len(symbols)
      groundedness = 1.0 if total == 0 else len(grounded) / total
      return GroundingReport(
          total_symbols=total, grounded=grounded, ungrounded=ungrounded,
          groundedness=groundedness,
      )
  ```
- [ ] `cli.py`:
  ```python
  from __future__ import annotations

  import argparse
  import json
  import sys
  from pathlib import Path

  from groundloop.lib_grounder.grounder import ground


  def main(argv: list[str] | None = None) -> int:
      p = argparse.ArgumentParser(prog="groundloop.lib_grounder")
      src = p.add_mutually_exclusive_group(required=True)
      src.add_argument("-c", "--code", type=str)
      src.add_argument("-f", "--file", type=Path)
      p.add_argument("--format", choices=("text", "json"), default="text")
      args = p.parse_args(argv or sys.argv[1:])
      source = args.code if args.code is not None else args.file.read_text(encoding="utf-8")
      report = ground(source)
      if args.format == "json":
          print(report.model_dump_json())
      else:
          print(f"groundedness={report.groundedness:.3f}")
          for s in report.ungrounded:
              name = f"{s.module}.{s.attr}" if s.attr else s.module
              print(f"  ungrounded {s.kind} at line {s.line}: {name}")
      return 0
  ```
- [ ] `__main__.py` — standard `raise SystemExit(main())`.
- [ ] `__init__.py` re-exports `GroundingReport`, `Symbol`, `ground`.
- [ ] Tests:
  ```python
  # test_grounder.py
  from groundloop.lib_grounder.grounder import ground


  def test_clean_import_resolves():
      r = ground("import os\n")
      assert r.groundedness == 1.0
      assert len(r.grounded) == 1


  def test_bogus_package_unresolved():
      r = ground("import nonexistent_zzz_pkg_987\n")
      assert r.groundedness < 1.0
      assert any(s.module == "nonexistent_zzz_pkg_987" for s in r.ungrounded)


  def test_from_import_attribute():
      r = ground("from os import getcwd\n")
      assert r.groundedness == 1.0


  def test_from_import_bogus_attr():
      r = ground("from os import not_a_real_attr_zzz\n")
      assert r.groundedness < 1.0


  def test_attribute_access_resolved():
      r = ground("import os\nos.getcwd\n")
      # Both import and attribute grounded
      assert r.groundedness == 1.0


  def test_attribute_access_hallucinated():
      r = ground("import os\nos.totally_fake_method_xyz\n")
      assert r.groundedness < 1.0


  def test_empty_source():
      r = ground("")
      assert r.groundedness == 1.0
      assert r.total_symbols == 0


  def test_syntax_error_returns_empty():
      r = ground("def (")
      assert r.total_symbols == 0
      assert r.groundedness == 1.0


  def test_relative_import_skipped():
      r = ground("from . import foo\n")
      assert r.total_symbols == 0
  ```
  Plus a tiny `test_cli.py` with two cases: `--code "import os"` and missing input (--code '').
- [ ] Run: `pytest tests/groundloop/lib_grounder/ -v --cov=groundloop.lib_grounder` — expect all pass, ≥85% coverage.
- [ ] Run: `ruff check groundloop/lib_grounder/`, `mypy --strict groundloop/lib_grounder/` — expect clean.
- [ ] Smoke: `python3 -m groundloop.lib_grounder -c "import os" --format json` — expect `"groundedness": 1.0`.
- [ ] Commit: `feat(lib-grounder): AST-level Python symbol grounding (sub-project #4)`

---

## Task 2: interrogator — scaffold + generator + CLI + tests

**Files:** `groundloop/interrogator/{__init__.py,__main__.py,models.py,interrogator.py,cli.py}`, tests.

- [ ] `models.py`:
  ```python
  from __future__ import annotations

  from pydantic import BaseModel, ConfigDict


  class InterrogationResult(BaseModel):
      model_config = ConfigDict(frozen=True)
      questions: tuple[str, ...]
      cited_node_ids: tuple[str, ...]
  ```
- [ ] `interrogator.py`:
  ```python
  from __future__ import annotations

  from groundloop.interrogator.models import InterrogationResult
  from groundloop.kb_indexer.index import SkillsIndex

  _TEMPLATES = (
      "What is the exact success criterion for '{brief_head}'?",
      "Have you considered the guidance from {skill_name}: '{section_title}'?",
      "Which of these assumptions is most load-bearing: success metric, inputs, failure modes?",
      "What is the single hardest edge case for '{brief_head}'?",
      "Have you consulted {skill_name2} for the patterns it recommends?",
  )


  class Interrogator:
      def __init__(self, index: SkillsIndex | None) -> None:
          self._index = index

      def generate(self, brief: str, *, top_k: int = 5) -> InterrogationResult:
          brief_head = brief.strip()[:80] or "the task"
          results = self._index.search(brief, top_k=top_k) if self._index is not None else []
          cited_ids = tuple(r.node_id for r in results[:2])
          first = results[0] if results else None
          second = results[1] if len(results) > 1 else first

          skill_name = first.skill_name if first else "the skill library"
          section_title = "/".join(first.section_path) if first else "the relevant section"
          skill_name2 = second.skill_name if second else skill_name

          questions = tuple(
              t.format(
                  brief_head=brief_head,
                  skill_name=skill_name,
                  section_title=section_title,
                  skill_name2=skill_name2,
              )
              for t in _TEMPLATES
          )
          return InterrogationResult(questions=questions, cited_node_ids=cited_ids)
  ```
- [ ] `cli.py` — accept `--brief <text>` + `--corpus <path>`, print questions.
- [ ] `__main__.py`, `__init__.py`.
- [ ] Tests:
  ```python
  # test_interrogator.py
  from pathlib import Path

  from groundloop.interrogator.interrogator import Interrogator
  from groundloop.kb_indexer.index import SkillsIndex


  def test_interrogator_returns_five_questions(tiny_corpus_path: Path):
      idx = SkillsIndex(corpus_path=tiny_corpus_path)
      idx.build()
      r = Interrogator(idx).generate("build a python api")
      assert len(r.questions) == 5
      assert len(r.cited_node_ids) >= 1


  def test_interrogator_deterministic(tiny_corpus_path: Path):
      idx = SkillsIndex(corpus_path=tiny_corpus_path)
      idx.build()
      i = Interrogator(idx)
      assert i.generate("x") == i.generate("x")


  def test_interrogator_no_index_still_returns_questions():
      r = Interrogator(None).generate("something")
      assert len(r.questions) == 5
      assert r.cited_node_ids == ()


  def test_interrogator_empty_brief():
      r = Interrogator(None).generate("")
      assert "the task" in r.questions[0]
  ```
  (Add `tiny_corpus_path` fixture in `tests/groundloop/interrogator/conftest.py` pointing to kb_indexer fixture.)
- [ ] Run pytest/ruff/mypy — expect clean + ≥85% cov.
- [ ] Commit: `feat(interrogator): deterministic Socratic question generator (sub-project #5)`

---

## Task 3: audit-reporter — scaffold + reporter + tests

**Files:** `groundloop/audit_reporter/{__init__.py,__main__.py,models.py,reporter.py,cli.py}`, tests.

- [ ] `models.py`:
  ```python
  from __future__ import annotations

  from pydantic import BaseModel, ConfigDict


  class AuditReport(BaseModel):
      model_config = ConfigDict(frozen=True)
      run_id: str
      summary: str
      iterations_total: int
      iterations_kept: int
      iterations_regressed: int
      iterations_plateau: int
      skill_citations: tuple[tuple[str, int], ...]
      score_trajectory: tuple[float, ...]
      final_score: float
      terminated_by: str
      hallucination_rate: float
  ```
- [ ] `reporter.py`:
  ```python
  from __future__ import annotations

  from collections import Counter

  from groundloop.audit_reporter.models import AuditReport
  from groundloop.ralph_orchestrator.models import RunResult


  class AuditReporter:
      @staticmethod
      def build(run: RunResult, *, hallucination_rate: float = 0.0) -> AuditReport:
          cites: Counter[str] = Counter()
          kept = 0
          regressed = 0
          plateau = 0
          trajectory: list[float] = []
          for it in run.iterations:
              cites.update(it.cited_node_ids)
              trajectory.append(it.sandbox_score_after)
              if it.reason == "score_improved":
                  kept += 1
              elif it.reason == "score_regressed":
                  regressed += 1
              elif it.reason == "score_plateau":
                  plateau += 1

          summary = (
              f"run={run.run_id} iters={len(run.iterations)} "
              f"final={run.final_score:.3f} terminated_by={run.terminated_by}"
          )
          skill_citations = tuple(sorted(cites.items(), key=lambda kv: (-kv[1], kv[0])))
          return AuditReport(
              run_id=run.run_id, summary=summary,
              iterations_total=len(run.iterations), iterations_kept=kept,
              iterations_regressed=regressed, iterations_plateau=plateau,
              skill_citations=skill_citations, score_trajectory=tuple(trajectory),
              final_score=run.final_score, terminated_by=run.terminated_by,
              hallucination_rate=hallucination_rate,
          )
  ```
- [ ] `cli.py` — accept `--run-json <path>`, load RunResult, emit AuditReport.
- [ ] Tests:
  ```python
  # test_reporter.py
  from groundloop.audit_reporter.reporter import AuditReporter
  from groundloop.ralph_orchestrator.models import Iteration, RunResult


  def _mk_run(iters: list[tuple[str, float, tuple[str, ...]]]) -> RunResult:
      """iters: list of (reason, score_after, cited_ids)"""
      it_objs = []
      for i, (reason, score, cites) in enumerate(iters):
          it_objs.append(Iteration(
              index=i, cited_node_ids=cites, rationale="r",
              proposed_files={}, sandbox_score_before=0.0,
              sandbox_score_after=score, kept=reason == "score_improved",
              reason=reason,
          ))
      return RunResult(
          run_id="r1", spec="s", started_at="t", ended_at="t",
          final_score=iters[-1][1] if iters else 0.0,
          final_files={}, iterations=tuple(it_objs),
          terminated_by="max_iters",
      )


  def test_audit_report_counts_reasons():
      run = _mk_run([
          ("score_improved", 0.5, ("n1",)),
          ("score_regressed", 0.4, ("n2",)),
          ("score_plateau", 0.4, ("n1",)),
      ])
      r = AuditReporter.build(run)
      assert r.iterations_total == 3
      assert r.iterations_kept == 1
      assert r.iterations_regressed == 1
      assert r.iterations_plateau == 1


  def test_audit_report_aggregates_citations():
      run = _mk_run([
          ("score_improved", 0.5, ("n1", "n2")),
          ("score_improved", 0.7, ("n1",)),
      ])
      r = AuditReporter.build(run)
      # n1 cited twice, n2 once
      d = dict(r.skill_citations)
      assert d["n1"] == 2
      assert d["n2"] == 1


  def test_audit_report_trajectory():
      run = _mk_run([
          ("score_improved", 0.3, ()),
          ("score_improved", 0.7, ()),
      ])
      r = AuditReporter.build(run)
      assert r.score_trajectory == (0.3, 0.7)


  def test_audit_report_empty_iterations():
      run = _mk_run([])
      r = AuditReporter.build(run)
      assert r.iterations_total == 0
      assert r.skill_citations == ()
      assert r.score_trajectory == ()
  ```
- [ ] Run pytest/ruff/mypy — clean + ≥85% cov.
- [ ] Commit: `feat(audit-reporter): structured audit report from RunResult (sub-project #8)`

---

## Task 4: Wire mcp-shell handlers to the new modules

**Files:**
- `groundloop/mcp_shell/tools/interrogate.py` (rewrite)
- `groundloop/mcp_shell/tools/autonomous_build.py` (rewrite)
- `groundloop/mcp_shell/tools/audit_report.py` (rewrite)
- `groundloop/mcp_shell/tools/ground_check.py` (augment)
- `groundloop/mcp_shell/session.py` (extend — store `RunResult` alongside `RunRecord` or replace)
- Tests updated.

### 4a. interrogate handler

Replace stub with real Interrogator. Handler reads optional `graph_id` from `args` — if present and valid, pass the index; otherwise call `Interrogator(None)`.

Update `InterrogateInput` schema to accept optional `graph_id` (None default).

Update existing tests + add a new test asserting 5 questions when graph_id provided.

### 4b. autonomous_build handler

Replace stub with:
```python
from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def handle_autonomous_build(args: dict, session: SessionState) -> dict:
    try:
        inp = AutonomousBuildInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")

    index = session.get_graph(inp.graph_id)
    if index is None:
        return {"status": "error", "reason": "unknown_graph_id", "detail": inp.graph_id}

    initial_files = {"main.py": "from __future__ import annotations\n"}

    synth = StubSynthesizer()
    config = LoopConfig(max_iters=inp.max_iters, target_score=0.95)

    try:
        result = run_loop(
            spec=inp.spec, initial_files=initial_files, index=index,
            synthesizer=synth, config=config,
        )
    except Exception as e:  # noqa: BLE001 - must return structured error
        return {"status": "error", "reason": "loop_failed", "detail": str(e)}

    session.register_run(result)

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_status": result.terminated_by,
        "iterations": len(result.iterations),
        "final_score": result.final_score,
        "final_files": result.final_files,
    }
```

Add `session.register_run(result)` method — stores by `result.run_id`.

### 4c. audit_report handler

Replace stub:
```python
from groundloop.audit_reporter.reporter import AuditReporter


def handle_audit_report(args: dict, session: SessionState) -> dict:
    try:
        inp = AuditReportInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    run = session.get_run_result(inp.run_id)
    if run is None:
        return {"status": "error", "reason": "unknown_run_id", "detail": inp.run_id}
    report = AuditReporter.build(run)
    return {"status": "ok", "report": report.model_dump()}
```

Session gets `get_run_result(run_id) -> RunResult | None` method.

### 4d. ground_check augment

When `inp.claim` contains ```` ``` ```` fenced code OR `import ` / `def ` / `class ` tokens:
1. Extract all fenced Python code blocks (reuse the regex from ralph's stub_synthesizer).
2. Run `lib_grounder.ground(<concatenated code>)` → GroundingReport.
3. Include `layer_a` dict in response: `{"total_symbols": ..., "groundedness": ..., "ungrounded": [...]}`.
4. Adjust final `confidence` = `softmax * groundingreport.groundedness` (only if Layer A ran).

### 4e. Session upgrade

Extend `SessionState`:
```python
self._run_results: dict[str, RunResult] = {}

def register_run(self, result: RunResult) -> None:
    self._run_results[result.run_id] = result

def get_run_result(self, run_id: str) -> RunResult | None:
    return self._run_results.get(run_id)
```

Keep `create_run` / `get_run` for the stubbed `RunRecord` (backward compat).

- [ ] Update tests in `tests/groundloop/mcp_shell/`:
  - `test_interrogate.py` — 5 questions path.
  - `test_autonomous_build.py` — executes loop, returns run_id, iterations count.
  - `test_audit_report.py` — fetches AuditReport for a completed run.
  - `test_ground_check.py` — code-claim path exercises lib-grounder.
  - `test_e2e.py` — full 5-tool chain ending with audit_report that references the run_id from autonomous_build.
- [ ] Run entire groundloop test suite: `pytest tests/groundloop/ -v --cov=groundloop --cov-report=term` — expect ≥85% coverage overall, all pass.
- [ ] Run `ruff check groundloop/` + `mypy --strict groundloop/` — expect clean.
- [ ] Commit: `feat(mcp-shell): wire interrogator, ralph-orchestrator, audit-reporter, and lib-grounder into real handlers`

---

## Task 5: README + final smoke

- [ ] Append a top-level `## End-to-End GroundLoop` section to README with a short demo walkthrough:
  1. Install deps.
  2. `python3 -m groundloop.skills_scraper` to build Layer B corpus.
  3. `python3 -m groundloop.mcp_shell` to start the MCP server.
  4. From a client, call `ingest_sources(null)` → `ground_check("code claim")` → `autonomous_build(spec, graph_id)` → `audit_report(run_id)`.
- [ ] Run final cross-package verification:
  ```
  python3 -m pytest tests/groundloop/ --cov=groundloop --cov-report=term -q
  ruff check groundloop/
  mypy --strict groundloop/
  ```
  Expect ≥85% overall coverage, zero failures, zero lint/type issues.
- [ ] End-to-end smoke via mcp-shell dispatch:
  ```
  python3 -c "
  from groundloop.mcp_shell.server import dispatch
  from groundloop.mcp_shell.session import SessionState
  s = SessionState()
  print('INTERROGATE:', dispatch('interrogate', {'brief': 'build a REST API'}, s))
  r = dispatch('ingest_sources', {'source_globs': None}, s)
  gid = r['graph_id']
  print('INGEST:', gid)
  gc = dispatch('ground_check', {'claim': '\`\`\`python\nimport os\nos.getcwd()\n\`\`\`', 'graph_id': gid}, s)
  print('GROUND:', gc['verdict'], 'layer_a=', gc.get('layer_a'))
  ab = dispatch('autonomous_build', {'spec': 'build greet(name) function', 'graph_id': gid, 'max_iters': 1}, s)
  print('BUILD:', ab['run_id'], ab['run_status'], ab['iterations'])
  ar = dispatch('audit_report', {'run_id': ab['run_id']}, s)
  print('AUDIT:', ar['status'], 'iters=', ar['report']['iterations_total'])
  "
  ```
  Expected: all 5 calls return `status=ok` (or well-formed verdicts). Audit report's `run_id` matches the build.
- [ ] Commit: `docs: end-to-end GroundLoop README + cross-package verification`

---

## Self-Review

- ✅ All four new modules (lib_grounder, interrogator, audit_reporter) have tests + CLI.
- ✅ Existing mcp_shell handlers rewritten to call real implementations.
- ✅ SessionState extended with `register_run`/`get_run_result`.
- ✅ Full cross-package coverage target met.
- ✅ Spec §5 acceptance criteria 1–7 covered.
- ✅ No placeholders.
