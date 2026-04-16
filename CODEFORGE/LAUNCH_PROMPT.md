# CodeForge Launch Prompt

Copy everything below the line and paste it as the first message in a new Claude Code session opened at `/home/krrish/Desktop/Project/openenv-DGXAI`.

---

You are starting a fresh CodeForge build session. Your job is to build CodeForge **from scratch** inside the `CODEFORGE/` directory as a standalone, production-grade product.

## The Situation

The `CODEFORGE/` directory already contains:
- `SYSTEM_DESIGN.md` — the authoritative spec (1,942 lines, 20 sections, 8 critics reviewed). This is the blueprint.
- `graphify/` — cloned from github.com/safishamsi/graphify (27K stars). Reference for AST-based knowledge graphs. Use the ARCHITECTURE PATTERNS only — do NOT use its Claude API extraction (breaks determinism).
- `ralph/` — cloned from github.com/snarktank/ralph (17K stars). Reference for PRD→subtask decomposition and autonomous iteration loops. Use the CONCEPTS only — do NOT import the bash scripts.
- `everything-claude-code/` — cloned from github.com/affaan-m/everything-claude-code (157K stars). 306 unique SKILL.md files. Use as the **skill corpus source** — scrape `skills/*/SKILL.md` (exact format match with YAML frontmatter + markdown sections).

The parent repo (`/home/krrish/Desktop/Project/openenv-DGXAI`) has existing `groundloop/` and `groundloop_env/` packages with 258 passing tests. These are REFERENCE IMPLEMENTATIONS — read them to understand how the sandbox, grounder, KB indexer, Ralph loop, interrogator, and audit reporter work. But you are building a NEW, CLEAN implementation in `CODEFORGE/codeforge/` that fixes all known bugs and includes all features from day one.

## Step 1: Read the source-of-truth documents

Before writing ANY code, read these completely:

1. `CLAUDE.md` — the execution plan (phases, checklists, workflow, parallel strategy, what NOT to do)
2. `CODEFORGE/SYSTEM_DESIGN.md` — the authoritative spec (every module, exact target code, reward system, MCP design, known bugs and their fixes)

Read CLAUDE.md §0.1 — it mandates reading SYSTEM_DESIGN.md before every module.

## Step 2: Read the reference implementations

Read the existing code in the parent repo to understand HOW things work — but do not modify these files. You are building fresh.

Key files to read for understanding (not modification):
- `groundloop/python_sandbox/sandbox.py` — how the sandbox writes files and runs tools
- `groundloop/python_sandbox/metric.py` — the composite score formula (has double-counting bug — your version fixes it)
- `groundloop/lib_grounder/grounder.py` — how AST grounding works (has 3 bugs — your version fixes them)
- `groundloop_env/grader.py` — the reward function (has floor exploit — your version fixes it)
- `groundloop_env/environment.py` — how action routing works (only 2 actions — yours has all 6+)
- `groundloop_env/tasks.py` — task definitions (reuse as-is)
- `groundloop/kb_indexer/index.py` — BM25 search (reuse pattern, add corpus manager)
- `groundloop/kb_indexer/cluster.py` — Jaccard clustering (reuse pattern)
- `groundloop/ralph_orchestrator/loop.py` — Ralph retry loop (reuse pattern, add real synthesizer)
- `groundloop/interrogator/interrogator.py` — Socratic questions (reuse pattern)
- `groundloop/audit_reporter/reporter.py` — audit reports (reuse pattern)
- `groundloop/skills_scraper/pipeline.py` — how skills are scraped (reuse, point at ECC)
- `models.py` — current data models (yours will have all 6 action types from day one)
- `groundloop_env/app.py` — FastAPI app (yours adds session isolation)

Also read the three cloned repos:
- `CODEFORGE/graphify/graphify/` — study the AST extraction and graph query patterns
- `CODEFORGE/ralph/ralph.sh` and `CODEFORGE/ralph/prompt.md` — study the PRD decomposition loop
- `CODEFORGE/everything-claude-code/skills/` — verify the SKILL.md format matches our scraper expectations

## Step 3: Build the CODEFORGE/codeforge/ package from scratch

Create this structure inside `CODEFORGE/`:

```
CODEFORGE/
├── SYSTEM_DESIGN.md          (exists — the spec)
├── LAUNCH_PROMPT.md           (exists — this file)
├── graphify/                  (exists — reference repo)
├── ralph/                     (exists — reference repo)
├── everything-claude-code/    (exists — corpus source)
├── codeforge/                 (BUILD THIS — the product)
│   ├── __init__.py
│   ├── models.py              ← All data models (6 action types, full observation, AuditEntry)
│   ├── grader.py              ← FIXED reward function (floor=0.50) — SYSTEM_DESIGN §4.8.1
│   ├── grounder.py            ← FIXED AST grounding (SyntaxError→0.0, zero→0.5, full path) — §4.8.3
│   ├── shaping.py             ← Citation shaping rewards — §4.8.4
│   ├── tasks.py               ← Task definitions (3 levels)
│   ├── observation.py         ← Observation builder
│   ├── environment.py         ← CodeForgeEnvironment (all 6 actions, validation, session-ready)
│   ├── app.py                 ← FastAPI server with session isolation — §15
│   ├── mcp_server.py          ← MCP server (tools + resources + prompts) — §9
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── sandbox.py         ← run_sandbox() — writes files, runs real tools
│   │   ├── runner.py          ← subprocess tool execution with timeouts
│   │   ├── tools.py           ← Tool registry (ruff, mypy, pytest, imports)
│   │   ├── imports.py         ← Import scanning
│   │   ├── metric.py          ← FIXED composite_score (no double-counting, tools filter) — §4.8.2
│   │   └── models.py          ← ToolResult, ParsedResult, ImportReport, SandboxResult
│   ├── kb/
│   │   ├── __init__.py
│   │   ├── indexer.py         ← SkillsIndex with BM25 search
│   │   ├── cluster.py         ← Jaccard clustering + connected components
│   │   ├── code_graph.py      ← KB2: AST knowledge graph (ast + networkx) — §4.3.2
│   │   ├── corpus_manager.py  ← add/remove/refresh skills — §4.3.1
│   │   ├── tokenizer.py       ← Text tokenization
│   │   └── models.py          ← SearchResult, Cluster, ClusterManifest
│   ├── ralph/
│   │   ├── __init__.py
│   │   ├── loop.py            ← Score-gated retry loop
│   │   ├── synthesizer.py     ← Synthesizer Protocol + LLMSynthesizer — §4.4
│   │   ├── planner.py         ← Task decomposition into subtasks — §4.4
│   │   ├── checkpoint.py      ← Disk persistence
│   │   └── models.py          ← LoopConfig, SynthesisResult, Iteration, RunResult
│   ├── interrogator/
│   │   ├── __init__.py
│   │   ├── interrogator.py    ← Socratic question generation
│   │   └── models.py          ← InterrogationResult
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── ledger.py          ← AuditLedger (per-episode, per-step append) — §12
│   │   ├── reporter.py        ← AuditReporter (build reports from runs)
│   │   └── models.py          ← AuditEntry, AuditReport
│   └── scraper/
│       ├── __init__.py
│       ├── pipeline.py        ← End-to-end scrape pipeline
│       ├── discovery.py       ← File discovery via globs
│       ├── parser.py          ← YAML frontmatter + markdown parsing
│       ├── chunker.py         ← Section-level chunking
│       ├── tagger.py          ← Domain/topic tagging
│       └── writer.py          ← JSONL serialization
├── tests/                     (BUILD THIS — comprehensive tests)
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_grader.py
│   ├── test_grounder.py
│   ├── test_sandbox.py
│   ├── test_indexer.py
│   ├── test_cluster.py
│   ├── test_code_graph.py
│   ├── test_environment.py
│   ├── test_ralph_loop.py
│   ├── test_interrogator.py
│   ├── test_audit.py
│   ├── test_corpus_manager.py
│   ├── test_app.py
│   ├── test_mcp_server.py
│   └── test_e2e.py            ← Full 6-action episode test
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── openenv.yaml
└── inference.py               ← Baseline agent using all 6 actions
```

## Step 4: Build phase by phase

Follow CLAUDE.md §4 phases. Each phase has a checklist. Tick checkboxes as you complete them.

**Phase 0: Foundation (models + grader + grounder + scoring)**
Build the core that everything depends on. All bug fixes from SYSTEM_DESIGN §19 are baked in from day one — these are not "fixes" to apply, they are the CORRECT implementation.
- `codeforge/models.py` — all 6 action types, full observation schema, AuditEntry
- `codeforge/grader.py` — reward function with floor=0.50 (SYSTEM_DESIGN §4.8.1)
- `codeforge/grounder.py` — AST grounding with all 3 fixes (§4.8.3)
- `codeforge/sandbox/` — full sandbox with fixed `composite_score` (§4.8.2)
- `codeforge/tasks.py` — 3 task levels
- Tests for all of the above

**Phase 1: Actions + Environment (M3–M8)**
Wire all 6 actions. Use the parallel strategy from CLAUDE.md §6:
- Spawn M3 (query_cluster) + M4 (interrogate) in parallel — independent handlers
- Then M5 (run_ralph) + M6 (AuditLedger)
- Then M7 (observation builder) + M8 (integration test)

**Phase 2: MCP + Deployment (M9)**
MCP server with tools, resources, prompts, session isolation, auth. Dockerfile. openenv.yaml.
- Spawn MCP implementation + Docker config in parallel

**Phase 3: Intelligence (M10–M13)**
- M10 (KB2 code graph) + M11 (ECC corpus) + M12 (LLM synthesizer) — all parallel
- M13 (planner + incremental scoring) — after M12

## Step 5: Quality enforcement

For EVERY module, before moving on:
1. `python3 -m pytest tests/ -v` → all pass
2. Coverage >= 85% on the package under work
3. `ruff check codeforge/` → clean
4. `mypy --strict codeforge/` → clean
5. Dispatch a fresh critic agent — report NOTHING_FURTHER or NEEDS_CHANGES
6. If NEEDS_CHANGES: fix, re-run gates, dispatch ANOTHER fresh critic (never reuse)
7. Loop until NOTHING_FURTHER

## Key rules

- **SYSTEM_DESIGN.md is the spec.** Read the section before implementing. Follow it exactly. Update it if reality diverges.
- **No stubs, no fakes, no bandaids.** If you don't know a library API, search docs or say so. Never fabricate.
- **`from __future__ import annotations`** on every Python file.
- **TDD strictly.** Test first (RED), implement (GREEN), refactor.
- **Use worktree isolation** (`isolation: "worktree"`) for parallel implementer agents.
- **Fresh critic every round.** Never reuse a critic agent.
- **Max 3 parallel agents.** More than that loses coordination.
- **The 3 reference repos are for READING, not importing.** Study their patterns, build your own clean implementation.
- **ECC `skills/*/SKILL.md` is for SCRAPING into the corpus.** Do not scrape duplicate dirs (.agents/, .cursor/, translations).
- **The environment grades the LLM, not itself.** Every reward traces to real sandbox output (subprocess execution of ruff/mypy/pytest), real AST grounding (importlib.find_spec), and real skill corpus citations. No mocks in production code.

## What success looks like

- `CODEFORGE/codeforge/` is a complete, standalone Python package
- All tests pass, coverage >= 85%
- MCP server exposes all tools, resources, prompts
- Docker image builds and runs, corpus baked in
- `openenv validate` passes
- CLAUDE.md checklists are all green
- SYSTEM_DESIGN.md reflects the final state of the code
- Any LLM connecting via MCP is FORCED to produce real, verified, grounded Python code — because the environment is the judge

Start now. Read the docs first.
