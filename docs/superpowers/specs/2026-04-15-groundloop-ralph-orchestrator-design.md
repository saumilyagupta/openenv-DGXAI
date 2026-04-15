# GroundLoop Ralph-Orchestrator — Design Spec

**Date:** 2026-04-15
**Sub-project:** #7 of 8 — the core Ralph loop
**Depends on:** skills-scraper (#2), kb-indexer (#3), python-sandbox (#6), mcp-shell (#1, for later wiring)
**Consumed by:** mcp-shell's `autonomous_build` tool (wired in sub-project #8/post-ralph)

---

## 1. Purpose

Run the autonomous iteration loop: given a spec and a KB graph, iteratively synthesize Python code, verify it with the sandbox, score it, and keep or discard changes until an exit criterion fires. Analogous to Karpathy's Autoresearch, but for codebase construction (metric = `composite_score` from the sandbox).

## 2. Scope

**In scope:**

- `run_loop(spec, index, *, max_iters, synthesizer, target_score) -> RunResult`
- A pluggable `Synthesizer` Protocol so code generation is injectable.
- A built-in `StubSynthesizer` that performs deterministic, skill-grounded file edits (no LLM, testable).
- An optional `OpenAISynthesizer` that calls an OpenAI-compatible chat API via the installed `openai` lib (disabled unless `OPENAI_API_KEY` is set).
- Per-iteration audit record: which KB nodes were cited, which files changed, which score deltas resulted, whether changes were kept or rolled back.
- Checkpoint/resume via JSON on disk.
- CLI: `python -m groundloop.ralph_orchestrator run <spec> --graph <corpus_path> ...`.

**Out of scope:**

- Parallel / branching experiments (Phase 2).
- MCP wiring of `autonomous_build` — deferred to a tiny post-ralph integration step.
- Full Layer-A (library/symbol) grounding — separate sub-project #4.
- Interrogator pre-flight — sub-project #5.

## 3. Architecture

```
groundloop/ralph_orchestrator/
  __init__.py
  __main__.py
  models.py            # Iteration, RunResult, SynthesisResult, LoopConfig (all Pydantic v2 frozen)
  synthesizer.py       # Synthesizer Protocol
  stub_synthesizer.py  # deterministic KB-grounded stub
  openai_synthesizer.py# optional real-LLM implementation
  loop.py              # run_loop core
  checkpoint.py        # save/load JSON per run
  cli.py
tests/groundloop/ralph_orchestrator/
  fixtures/
    spec_simple.txt
    initial_files/
      main.py          # intentionally trivial starting code
  conftest.py
  test_models.py
  test_stub_synthesizer.py
  test_openai_synthesizer.py   # skipped if no key
  test_loop.py
  test_checkpoint.py
  test_cli.py
  test_e2e.py
```

Each module ≤ 200 lines.

## 4. Component Contracts

### 4.1 `synthesizer.py` — `Synthesizer` protocol

```python
class Synthesizer(Protocol):
    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],     # filename → content
        citations: Sequence[SearchResult],    # kb-indexer results
        iteration: int,
    ) -> SynthesisResult: ...
```

`SynthesisResult`:
```python
class SynthesisResult(BaseModel, frozen=True):
    proposed_files: dict[str, str]             # full replacement set
    rationale: str                             # short reason for changes
    cited_node_ids: tuple[str, ...]            # subset of citations actually used
```

### 4.2 `stub_synthesizer.py` — `StubSynthesizer`

Deterministic, no external deps. Algorithm:

1. Take the highest-scored citation.
2. Extract any fenced Python code blocks from `citation.section_body`.
3. If any block is found and not already present in `current_files["main.py"]`, append it to `main.py` inside a new function `_from_<skill>_<rank>()`.
4. If no code blocks are found, append a one-line comment `# consulted: <skill>/<section>` to `main.py` — simple but deterministic signal.
5. Rationale = `"Applied suggestion from <skill_name>/<section_path>"`.

This gives the loop a concrete, testable iteration signal without needing an LLM. The score will typically rise then plateau — which is the realistic loop dynamic we want to demonstrate.

### 4.3 `openai_synthesizer.py` — `OpenAISynthesizer`

Uses the already-installed `openai` lib. Reads `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL_NAME` from env. Builds a chat completion with:
- system: "You generate production Python code. Use only the provided skill citations. Return a JSON object `{proposed_files: {filename: content}, rationale: str, cited_node_ids: [str]}`."
- user: spec + current files + formatted citations.

Parses the JSON response; on parse error returns `SynthesisResult(proposed_files=current_files, rationale="parse_error", cited_node_ids=())` — effectively a no-op iteration that will be discarded by the keep-if-better gate.

Skipped entirely if `OPENAI_API_KEY` is unset (constructor raises `RuntimeError`).

### 4.4 `models.py`

```python
class LoopConfig(BaseModel, frozen=True):
    max_iters: int = 5
    target_score: float = 0.95
    tools: tuple[str, ...] = ("ruff", "imports")
    timeout_per_tool: float = 60.0
    top_k_citations: int = 5

class Iteration(BaseModel, frozen=True):
    index: int                                # 0-based iteration number
    cited_node_ids: tuple[str, ...]
    rationale: str
    proposed_files: dict[str, str]
    sandbox_score_before: float
    sandbox_score_after: float
    kept: bool
    reason: str                               # "score_improved" | "score_regressed" | "target_hit"

class RunResult(BaseModel, frozen=True):
    run_id: str
    spec: str
    started_at: str
    ended_at: str
    final_score: float
    final_files: dict[str, str]
    iterations: tuple[Iteration, ...]
    terminated_by: str                        # "target_hit" | "max_iters" | "stuck"
```

### 4.5 `loop.py`

```python
def run_loop(
    *,
    spec: str,
    initial_files: Mapping[str, str],
    index: SkillsIndex,
    synthesizer: Synthesizer,
    config: LoopConfig = LoopConfig(),
    checkpoint_dir: Path | None = None,
) -> RunResult: ...
```

Per-iteration:

1. Score current files → `score_before`.
2. If `score_before >= target_score`: terminate `target_hit`.
3. Query `index.search(spec, top_k=config.top_k_citations)` → citations.
4. `synthesizer.synthesize(...)` → SynthesisResult.
5. Score proposed_files → `score_after`.
6. If `score_after > score_before`: keep (new current files = proposed). `reason="score_improved"`.
7. Else: revert. `reason="score_regressed"`.
8. Record Iteration.
9. If `checkpoint_dir`: write `checkpoint_dir/run_<id>.json` after each iteration.
10. If 3 consecutive `score_regressed` iterations: terminate `stuck`.
11. After `max_iters`: terminate `max_iters`.

### 4.6 `checkpoint.py`

```python
def save_checkpoint(run: RunResult, checkpoint_dir: Path) -> Path: ...
def load_checkpoint(checkpoint_path: Path) -> RunResult: ...
```

JSON layout = `RunResult.model_dump()`. Crash-safe: write to tempfile, rename atomically.

### 4.7 `cli.py`

```
python -m groundloop.ralph_orchestrator run <spec-file>
    --corpus <path>
    [--initial-file main.py=<path>]  [repeatable]
    [--max-iters 5]
    [--target-score 0.95]
    [--checkpoint-dir <path>]
    [--synthesizer stub|openai]
    [--format json|text]
```

Exit 0 on success (even if `target_score` not hit). Exit 1 on missing corpus or initial-file paths.

## 5. Error Handling

| Condition | Action |
|---|---|
| Missing initial-file path | CLI exit 1 |
| Sandbox raises | Treated as score=0.0 for that iteration; recorded as `reason="sandbox_error"`, kept=False |
| Synthesizer raises | Same — score=0.0, `rationale="synthesizer_error: <msg>"` |
| Index search returns `[]` | Empty citations passed to synthesizer; synth can still emit changes but with `cited_node_ids=()` |
| Checkpoint write fails | Log WARN, continue (don't fail the run over checkpoint I/O) |
| `OPENAI_API_KEY` unset but `--synthesizer openai` | CLI exit 1 with clear message |

## 6. Testing

Coverage target: **85%**. LLM path (`openai_synthesizer.py`) skipped unless `OPENAI_API_KEY` is set — tests mock `openai.OpenAI` via monkeypatch.

Fixtures:
- `fixtures/spec_simple.txt` — one-line spec.
- `fixtures/initial_files/main.py` — trivial starting file.
- Reuse `tests/groundloop/kb_indexer/fixtures/tiny_corpus.jsonl` for the index.

Tests:
- `test_models.py` — frozen, required fields.
- `test_stub_synthesizer.py` — fenced-code extraction, deterministic output across calls, citation tracking.
- `test_openai_synthesizer.py` — monkeypatched client, JSON parse success + failure paths.
- `test_loop.py` — loop terminates on target hit, max_iters, stuck (3 consecutive regressions); keep/revert semantics; sandbox error handled.
- `test_checkpoint.py` — save/load roundtrip; atomic rename.
- `test_cli.py` — argparse, missing-corpus exit 1, json vs text format.
- `test_e2e.py` — end-to-end with stub synthesizer, tiny corpus, initial file; asserts `RunResult` has ≥ 1 iteration and `final_score >= 0`.

## 7. Acceptance Criteria

1. `run_loop(spec, initial_files, index, StubSynthesizer())` completes in <5s on tiny corpus.
2. Stub synthesizer is deterministic: same inputs → byte-identical `SynthesisResult.proposed_files`.
3. Loop honors `target_score` early-exit: if initial files already score ≥ target, terminate with 0 iterations.
4. Loop honors `max_iters` hard cap.
5. Loop detects `stuck` after 3 consecutive regressions.
6. Checkpoint file is valid JSON that round-trips back into a `RunResult`.
7. `ruff check` + `mypy --strict` clean.
8. Coverage ≥ 85%.
9. E2E test produces a `RunResult` with ≥ 1 `Iteration` record and a cited KB node_id traceable back to the tiny corpus.

## 8. Dependencies

Already installed: everything from #2/#3/#6 plus `openai>=1.0.0` (for the optional synthesizer). No new deps.

## 9. Deliverables

Package, tests, fixtures, CLI, README subsection explaining stub vs openai synthesizers and env vars.
