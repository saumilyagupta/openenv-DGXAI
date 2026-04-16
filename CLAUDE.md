---
description:
alwaysApply: true
---

# CLAUDE.md — CodeForge: Build It, Ship It, No Faking It

**Last updated:** 2026-04-16
**System design:** `CODEFORGE/SYSTEM_DESIGN.md` (1,941 lines, 20 sections, 8 critics reviewed)
**Reference repos:** `CODEFORGE/graphify/`, `CODEFORGE/ralph/`, `CODEFORGE/everything-claude-code/`

---

## 0. Prime Directive

Build CodeForge as a production-grade MCP-exposed RL environment that **forces any LLM to write real, verified, grounded Python code**. No stubs. No fakes. No bandaids. No hallucinated APIs. If you don't know a library, read its docs or say so — never make it up.

The environment is the judge, not the LLM. Every reward traces to real sandbox output, real AST grounding, and real skill corpus citations.

**Stop conditions:**
1. Genuinely destructive action (force push, hard reset, deleting user data, touching files outside this repo)
2. All phase checklists green

Everything else: proceed without asking.

### 0.1 The System Design Doc Is The Source of Truth

**`CODEFORGE/SYSTEM_DESIGN.md` is the authoritative specification. Every session MUST:**

1. **Read it first.** Before writing any code, read the relevant section of `CODEFORGE/SYSTEM_DESIGN.md`. The exact target code for every module is in there — grader (§4.8.1), scoring (§4.8.2), grounder (§4.8.3), shaping (§4.8.4), Ralph reward (§4.8.5), MCP tools (§9.1), KB2 code graph (§4.3.2), session isolation (§15), error handling (§17).

2. **Follow it exactly.** If SYSTEM_DESIGN.md says `_UNCERTAIN_FLOOR = 0.50`, the code says `_UNCERTAIN_FLOOR = 0.50`. If SYSTEM_DESIGN.md says `groundedness = 0.5 if total == 0`, the code says that. No creative reinterpretation.

3. **Update it when reality changes.** If during implementation you discover that a spec in SYSTEM_DESIGN.md is wrong, incomplete, or needs adjustment — update the doc FIRST, then implement. The doc must always reflect the current truth, not a stale snapshot. Commit doc changes with `docs(codeforge): update SYSTEM_DESIGN.md — <what changed and why>`.

4. **Never contradict it silently.** If your code diverges from the doc, that is a bug — either in the code or in the doc. Fix one or the other. Never leave them out of sync.

5. **Reference it in commit messages.** When implementing a module, cite the section: `feat(codeforge-m3): query_cluster action (SYSTEM_DESIGN §4.3.1, §5.2)`

**The relationship:** This CLAUDE.md tells you WHAT to do and in WHAT ORDER. SYSTEM_DESIGN.md tells you HOW everything works, the exact target code, the known bugs, and the architectural decisions. Both must stay in sync. Both must be read before any major decision.

---

## 1. What CodeForge Is

An OpenEnv-compliant RL environment where an LLM agent receives a natural-language brief ("implement greet(name)") and must produce working Python code through iterative actions. Exposed via:

- **REST API** (`/reset`, `/step`, `/state`) — OpenEnv judge compliance
- **MCP Server** (tools + resources + prompts) — how engineers/LLMs actually use it

**The invariant:** Every reward-earning action must trace to (a) a sandbox-verified programmatic signal, (b) a Layer-A grounded symbol, and (c) a Layer-B skill citation — recorded as a `(reward, evidence, policy)` triple.

---

## 2. Architecture (read `CODEFORGE/SYSTEM_DESIGN.md` §2 for the full diagram)

```
LLM Agent ──▶ MCP Server ──▶ FastAPI ──▶ CodeForgeEnvironment
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                    Python Sandbox       AST Grounder        KB Indexer
                    (ruff,mypy,pytest)   (import resolve)    (BM25+clusters)
                          │                   │                   │
                          └───────┬───────────┘                   │
                                  ▼                               │
                            Grader (reward)                       │
                            quality = 0.6*sandbox + 0.4*ground    │
                            + Brier calibration penalty           │
                                                                  │
                    Ralph Loop ◀──────────────────────────────────┘
                    (synthesize → score → keep if better)
                                  │
                            Audit Ledger
                            (every step recorded)
```

---

## 3. The 3 Layers of Reward (never confuse them)

| Layer | File (in `CODEFORGE/codeforge/`) | Input | Output | Notes |
|---|---|---|---|---|
| **Reward Function** | `grader.py` | sandbox_score, groundedness, confidence | reward (0-1) | Floor=0.50 from day one. UNCHANGED across all phases. |
| **Scoring Pipeline** | `sandbox/metric.py` | files, tools | composite_score (0-1) | Penalty-only (no double-counting). Has `tools` filter for subtask scoring. |
| **Grounding Check** | `grounder.py` | source code | groundedness (0-1) | SyntaxError→0.0, zero symbols→0.5, full module path. Correct from day one. |

Building from scratch means these are CORRECT from the start — not patched old code. The reward function does not change across phases. It takes two floats in, puts one float out.

---

## 4. Phases

### Phase 0: Foundation — Models + Reward System + Sandbox (build from scratch in `CODEFORGE/codeforge/`)

Build the core that everything else depends on. All critical bugs from SYSTEM_DESIGN §19 are the CORRECT implementation from day one — not patches on old code.

The existing `groundloop/` and `groundloop_env/` in the parent repo are REFERENCE IMPLEMENTATIONS. Read them to understand how things work. Do NOT modify them. Build clean in `CODEFORGE/codeforge/`.

| # | What to build | Target file | Spec |
|---|---|---|---|
| P0-1 | Data models (all 6 action types, full observation, AuditEntry) | `codeforge/models.py` | SYSTEM_DESIGN §8 |
| P0-2 | Reward function (floor=0.50, correct from day one) | `codeforge/grader.py` | SYSTEM_DESIGN §4.8.1 |
| P0-3 | AST grounder (SyntaxError→0.0, zero→0.5, full path resolution) | `codeforge/grounder.py` | SYSTEM_DESIGN §4.8.3 |
| P0-4 | Sandbox (run real tools, fixed composite_score with tools filter) | `codeforge/sandbox/` | SYSTEM_DESIGN §4.1, §4.8.2 |
| P0-5 | Task definitions (3 levels) | `codeforge/tasks.py` | SYSTEM_DESIGN §11 |
| P0-6 | Tests for all of the above | `CODEFORGE/tests/` | Coverage >= 85% |

**Quality gate:** All tests pass. `ruff check codeforge/` clean. `mypy --strict codeforge/` clean.

- [x] **P0** Foundation built, tested, green

### Phase 1: All 6 Actions + Environment (build from scratch in `CODEFORGE/codeforge/`)

Build the environment with ALL 6 actions wired from day one. Read the reference implementations in `groundloop/` and `groundloop_env/` for patterns, but write clean code in `CODEFORGE/codeforge/`.

| Module | What to build | Target files | Spec |
|---|---|---|---|
| **M3** | KB indexer (BM25 + clustering) + `query_kb` + `query_cluster` handlers | `codeforge/kb/indexer.py`, `codeforge/kb/cluster.py`, `codeforge/kb/models.py`, `codeforge/kb/tokenizer.py` | SYSTEM_DESIGN §4.3.1 |
| **M4** | Interrogator + `interrogate` handler | `codeforge/interrogator/` | SYSTEM_DESIGN §4.5 |
| **M5** | Ralph loop + `run_ralph` handler + variable budget cost | `codeforge/ralph/` | SYSTEM_DESIGN §4.4, §4.8.5 |
| **M6** | AuditLedger + AuditReporter + `get_audit` handler | `codeforge/audit/` | SYSTEM_DESIGN §12, §4.6 |
| **M7** | Environment (all 6 actions, validation, file limits, filename allowlist) + observation builder | `codeforge/environment.py`, `codeforge/observation.py` | SYSTEM_DESIGN §4.9, §17 |
| **M8** | FastAPI app (with session isolation) + full 6-action integration test + inference baseline | `codeforge/app.py`, `CODEFORGE/tests/test_e2e.py`, `CODEFORGE/inference.py` | SYSTEM_DESIGN §3.1, §15 |

**Action surface after Phase 1:**

| Action | Cost | Reward | What it does |
|---|---|---|---|
| `query_kb` | 1 | 0.0 | BM25 search over skill corpus |
| `query_cluster` | 1 | 0.0 | Browse cluster members by label |
| `interrogate` | 1 | 0.0 | Socratic questions citing skill nodes |
| `run_ralph` | N (max_iters) | `calibrated_reward(final, confidence=0.75) - 0.05*wasted` | Autonomous improve loop |
| `submit` | 1 | `calibrated_reward(sandbox, groundedness, confidence)` | Grade code via sandbox+grounder+Brier |
| `get_audit` | 0 | 0.0 | Read-only audit trail |

**Reward formula (after P0 fixes):**
```
quality   = 0.6 * sandbox_composite_score + 0.4 * groundedness
brier     = min((confidence - quality)², 0.5)   # if confidence provided
reward    = quality * (1 - brier)
reward    = max(reward, 0.50) if confidence < 0.3 AND quality < 0.5   # uncertain floor
reward    = clamp(0.0, 1.0), round to 3 decimals
```

**Quality gate per module:** tests pass, coverage >= 85%, ruff clean, mypy clean, fresh critic returns NOTHING_FURTHER.

- [x] **M3** KB subsystem (indexer + cluster + tokenizer + models) in `codeforge/kb/`
- [x] **M4** Interrogator in `codeforge/interrogator/`
- [x] **M5** Ralph loop (loop + synthesizer protocol + stub + checkpoint) in `codeforge/ralph/`
- [x] **M6** Audit (ledger + reporter) in `codeforge/audit/`
- [x] **M7** Environment + observation builder in `codeforge/environment.py` + `codeforge/observation.py`
- [x] **M8** FastAPI app + session isolation + e2e test + inference baseline

### Phase 2: MCP Server + Deployment (M9)

Expose the environment as an MCP server with tools, resources, and prompts. Deploy to HF Space.

**MCP tools (8):** `codeforge_reset`, `codeforge_query_kb`, `codeforge_query_cluster`, `codeforge_interrogate`, `codeforge_run_ralph`, `codeforge_submit`, `codeforge_get_audit`, `codeforge_state`

**MCP discovery tools (2, zero budget):** `codeforge_list_clusters`, `codeforge_list_tags`

**MCP resources:** `codeforge://corpus/stats`, `codeforge://corpus/node/{id}`, `codeforge://tasks`, `codeforge://audit/{episode_id}`

**MCP prompts:** `codeforge_system` (rules + constraints), `codeforge_task_brief` (dynamic per-reset)

**Session isolation:** Session-keyed environment pool, not a global singleton. Max 10 concurrent sessions, 1hr TTL. SSE transport requires bearer token auth.

**Deployment:**
- Docker: FastAPI on 7860, MCP SSE on 7861, ruff/mypy/pytest installed in image
- HF Space: `krrishchoudhary109/code-forge`, Docker SDK
- Corpus baked into image

See `SYSTEM_DESIGN.md §9` for full tool schemas, `§13` for deployment, `§14-15` for security + session isolation.

- [x] **M9** MCP server functional (tools + resources + prompts)
- [x] Corpus baked into Docker image
- [x] `openenv validate` passes
- [ ] HF Space deployed
- [ ] Live baseline on deployed Space

### Phase 3: Intelligence Layer (M10–M13)

Post-submission enhancements. Two knowledge bases, real LLM synthesizer, task decomposition.

**M10 — Code Knowledge Graph (KB2):**
Build `codeforge/kb/code_graph.py`. `ast` stdlib + `networkx.DiGraph`. ~80 lines. No new deps. Extract module→imports→module, function→calls→function, class→inherits→class edges from the agent's `current_files`. New action: `query_code_graph` (0 budget). Reduces token usage by providing structural answers instead of dumping all file contents. See `SYSTEM_DESIGN.md §4.3.2`.

**M11 — ECC Corpus Integration:**
Clone `everything-claude-code` (already in `CODEFORGE/`). Add one `SourceRoot` glob pointing at `CODEFORGE/everything-claude-code/skills/*/SKILL.md`. Re-scrape. Corpus grows from ~1,006 to ~2,500+ nodes. Build `SkillCorpusManager` with add/remove/refresh (~100 lines). See `SYSTEM_DESIGN.md §4.3.1`.

**M12 — LLM Synthesizer:**
Implement `LLMSynthesizer(Synthesizer)` that wraps Claude/GPT API. Takes spec + current_files + citations + iteration, returns proposed_files + rationale + cited_node_ids. ~150 lines. The `Synthesizer` Protocol already exists — just implement it. See `SYSTEM_DESIGN.md §4.4 Honest Gaps`.

**M13 — Task Planner + Incremental Scoring:**
`Planner` decomposes specs into ordered subtasks. Each subtask has `target_files` and `tools`. Ralph runs on each subtask with subtask-scoped scoring (`composite_score(result, tools=subtask.tools)`). The `tools` filter on `composite_score()` is the only scoring pipeline change needed (3 lines). See `SYSTEM_DESIGN.md §4.8.2`.

**Shaping rewards (Phase 3+):**
After successful submit, retroactive +0.01 per prior query whose cited skills appear in submitted code, max 0.05. See `SYSTEM_DESIGN.md §4.8.4`.

- [x] **M10** Code Knowledge Graph (KB2) shipped
- [x] **M11** ECC corpus integrated, SkillCorpusManager working
- [x] **M12** LLM Synthesizer passing real tests
- [x] **M13** Planner + incremental scoring working

---

## 5. Per-Module Workflow (rigid, every module, every phase)

```
0. Read CODEFORGE/SYSTEM_DESIGN.md — find the section for this module, read the exact spec
1. Implement (TDD: test first RED, implement GREEN, refactor)
   - python3 -m pytest tests/ -v → all pass
   - python3 -m pytest tests/<module>/ --cov=<pkg> → coverage >= 85%
   - ruff check <pkg>/ → clean
   - mypy --strict <pkg>/ → clean
3. Dispatch fresh critic (Agent, never reuse critics)
   - Reports: NOTHING_FURTHER | NEEDS_CHANGES
4. If NEEDS_CHANGES: fix, re-run gates, dispatch ANOTHER fresh critic
5. Loop until NOTHING_FURTHER
6. Commit: feat(codeforge-mN): <what> (cite SYSTEM_DESIGN section)
7. If implementation diverged from SYSTEM_DESIGN.md: update the doc, commit separately
8. Tick checkbox in this file
9. Move to next module
```

**Hard rules:**
- `from __future__ import annotations` on every file
- Every implementer task = one commit, conventional format
- Fresh critic every round — never reuse
- Test count must not drop between commits
- Coverage must not drop below 85%
- Zero `ruff check` or `mypy --strict` warnings introduced
- No push until phase checklist is green

---

## 6. Parallel Execution Strategy

**Speed rule: if two modules don't share state, they run in parallel.**

The orchestrator (you, the main Claude session) does NOT implement modules directly. It spawns teammates via `Agent` with `isolation: "worktree"` for independent work, and uses `SendMessage` to coordinate. The orchestrator's job is: dispatch, monitor, review, merge.

### 6.1 Parallelism Map — What Can Run Together

All work happens in `CODEFORGE/codeforge/`. Reference code in `groundloop/` is READ-ONLY.

```
Phase 0: Foundation (all new files — high parallelism)
├── Agent A (worktree): models.py + grader.py + tasks.py       ← data models + reward function
├── Agent B (worktree): grounder.py                            ← AST grounding (independent module)
├── Agent C (worktree): sandbox/ (all files)                   ← sandbox pipeline (independent module)
│   (ALL THREE are independent — different files, no shared imports yet)
├── Wait for all → merge → run full test suite
└── Done

Phase 1: Actions + Environment (some parallel, some sequential)
├── Agent A (worktree): M3 kb/ (indexer, cluster, tokenizer, models)   ← KB subsystem
├── Agent B (worktree): M4 interrogator/                               ← interrogator subsystem
├── Agent C (worktree): M5 ralph/ (loop, synthesizer, checkpoint, models) ← ralph subsystem
│   (M3, M4, M5 are INDEPENDENT subsystems — different directories, no shared state)
├── Wait for all → merge
├── Agent D (worktree): M6 audit/ (ledger, reporter, models)          ← depends on ralph models
│   (SEQUENTIAL after M5 — uses ralph RunResult type)
├── Wait → merge
├── Agent E: M7 environment.py + observation.py                        ← imports ALL subsystems above
│   (SEQUENTIAL — this is the glue that wires everything together)
├── Agent F: M8 app.py + test_e2e.py + inference.py                   ← depends on environment
└── Done

Phase 2: MCP + Deployment
├── Agent A (worktree): M9 mcp_server.py (tools + resources + prompts) ← new file
├── Agent B (worktree): Dockerfile + pyproject.toml + openenv.yaml     ← deployment files
│   (INDEPENDENT — MCP server and Docker config don't share files)
├── Wait for both → merge
├── openenv validate + HF deploy                                       ← sequential, needs HF token
└── Done

Phase 3: Intelligence Layer
├── Agent A (worktree): M10 kb/code_graph.py                          ← new file in kb/
├── Agent B (worktree): M11 scraper/ + kb/corpus_manager.py           ← new files
├── Agent C (worktree): M12 ralph/synthesizer.py (LLMSynthesizer)     ← extends existing file
│   (ALL THREE are independent — different files/dirs)
├── Wait for all → merge
├── Agent D: M13 ralph/planner.py + sandbox/metric.py tools filter    ← depends on M12
└── Done
```

### 6.2 How To Spawn Parallel Teammates

**For independent subsystems (e.g., M3 KB + M4 Interrogator + M5 Ralph — all in Phase 1):**
```
Send a SINGLE message with MULTIPLE Agent tool calls:

Agent(
  name="m3-kb-subsystem",
  isolation: "worktree",
  prompt: "Build the KB subsystem from scratch in CODEFORGE/codeforge/kb/.
    Read CODEFORGE/SYSTEM_DESIGN.md §4.3.1 first.
    Read groundloop/kb_indexer/index.py and groundloop/kb_indexer/cluster.py as REFERENCE — do not modify them.
    Create: indexer.py (BM25 search), cluster.py (Jaccard + connected components), tokenizer.py, models.py (SearchResult, Cluster, ClusterManifest).
    Write tests in CODEFORGE/tests/test_indexer.py and test_cluster.py.
    TDD: test first RED, implement GREEN.
    Quality gates: pytest pass, coverage >= 85%, ruff clean, mypy --strict clean.
    from __future__ import annotations on every file.
    When done, report files created and test results.",
  run_in_background: true
)

Agent(
  name="m4-interrogator",
  isolation: "worktree",
  prompt: "Build the interrogator from scratch in CODEFORGE/codeforge/interrogator/.
    Read CODEFORGE/SYSTEM_DESIGN.md §4.5 first.
    Read groundloop/interrogator/interrogator.py as REFERENCE — do not modify it.
    Create: interrogator.py, models.py.
    Write tests in CODEFORGE/tests/test_interrogator.py.
    TDD: test first RED, implement GREEN.
    Quality gates: pytest pass, coverage >= 85%, ruff clean, mypy --strict clean.
    from __future__ import annotations on every file.
    When done, report files created and test results.",
  run_in_background: true
)

Agent(
  name="m5-ralph",
  isolation: "worktree",
  prompt: "Build the Ralph orchestrator from scratch in CODEFORGE/codeforge/ralph/.
    Read CODEFORGE/SYSTEM_DESIGN.md §4.4 and §4.8.5 first.
    Read groundloop/ralph_orchestrator/loop.py as REFERENCE — do not modify it.
    Also read CODEFORGE/ralph/ralph.sh and CODEFORGE/ralph/prompt.md for decomposition patterns.
    Create: loop.py, synthesizer.py (Protocol + StubSynthesizer), checkpoint.py, models.py.
    Write tests in CODEFORGE/tests/test_ralph_loop.py.
    TDD: test first RED, implement GREEN.
    Quality gates: pytest pass, coverage >= 85%, ruff clean, mypy --strict clean.
    from __future__ import annotations on every file.
    When done, report files created and test results.",
  run_in_background: true
)
```

All three run in isolated git worktrees. They create files in different directories. Zero conflict risk.

**For critic reviews (always parallel with next implementation):**
```
After M3 finishes, spawn critic in background while M6 (audit) starts:

Agent(
  name="m3-critic-1",
  prompt: "Review the KB subsystem in CODEFORGE/codeforge/kb/ against CODEFORGE/SYSTEM_DESIGN.md §4.3.1.
    Check: does search() match the spec? Are cluster labels correct? Is BM25 scoring right?
    Compare against the reference implementation in groundloop/kb_indexer/ — is anything missing?
    Report NOTHING_FURTHER or NEEDS_CHANGES with specific file:line citations.",
  run_in_background: true
)
```

### 6.3 Rules for Parallel Execution

1. **Never parallelize modules that touch the same file.** M7 (schema updates) touches `models.py` and `observation_builder.py` — it runs alone after M3-M6.
2. **Worktree isolation is mandatory for implementers.** Use `isolation: "worktree"` so each agent has its own copy of the repo. Merge results back to main after review.
3. **Critics run in the MAIN worktree.** They only read, never write. Safe to run concurrently.
4. **If a merge conflict happens:** the orchestrator resolves it manually — never let an agent force-resolve.
5. **Max 3 parallel agents at once.** More than that and context management becomes the bottleneck.
6. **Every parallel agent gets the same briefing prefix:**
   ```
   "You are implementing CodeForge module [X].
    MANDATORY: Read CODEFORGE/SYSTEM_DESIGN.md §[section] FIRST.
    Follow it exactly. If the spec is wrong, report the discrepancy — do not silently diverge.
    TDD: test first (RED), implement (GREEN), refactor.
    Quality gates: pytest pass, coverage >= 85%, ruff clean, mypy --strict clean.
    from __future__ import annotations on every file."
   ```

### 6.4 Estimated Speedup

| Phase | Sequential | Parallel (3 agents) | Speedup |
|---|---|---|---|
| P0 (bug fixes) | 6 tasks serial | 2 parallel batches | ~2x |
| P1 (M3-M8) | 6 modules serial | M3+M4 parallel, M5 overlap, M6 serial, M7+M8 serial | ~1.5x |
| P2 (M9 + deploy) | 2 tasks serial | MCP + Docker parallel | ~2x |
| P3 (M10-M13) | 4 modules serial | M10+M11+M12 parallel, M13 serial | ~2.5x |

Total: ~40% faster than fully sequential, with higher quality because critics review in parallel with implementation of the next independent module.

---

## 7. Build vs Integrate Decisions

| Component | Decision | Rationale |
|---|---|---|
| Skills corpus from ECC | **INTEGRATE** | Clone repo, add one glob, zero code changes, doubles corpus |
| Skill add/remove/refresh | **BUILD** (~100 lines) | Specific to our JSONL format, mtime+body_hash fields exist |
| AST code graph (KB2) | **BUILD** (~80 lines) | ast stdlib + networkx. Graphify needs Claude API calls — breaks determinism |
| Ralph retry loop | **KEEP OURS** | snarktank/ralph is bash, not Python. Our ralph_orchestrator is already clean |
| LLM Synthesizer | **BUILD** (~150 lines) | No clean drop-in. Synthesizer Protocol already defined |
| Task Planner | **BUILD** (~200 lines) | SWE-agent too coupled. Our tasks are simple enough |
| BM25 search | **KEEP** (rank_bm25) | Adequate for 2,500 nodes |

---

## 8. Reference Repos (`CODEFORGE/`)

| Repo | Stars | What we use | What we skip |
|---|---|---|---|
| `graphify/` | 27K | Architecture pattern for KB2 (ast + graph + communities). Query interface concepts (BFS/DFS/path). | Claude API extraction (breaks determinism). Tree-sitter (overkill for Python-only). Leiden algorithm (Jaccard CC is enough). |
| `ralph/` | 17K | PRD→user-story decomposition pattern for M13 planner. Loop-with-fresh-context-per-iteration concept. | Bash script (not importable). Git branch management. Amp/Claude CLI integration. |
| `everything-claude-code/` | 157K | 306 unique SKILL.md files as corpus source for M11. Exact format match (YAML frontmatter + markdown). | Platform-specific duplicates (.agents/, .cursor/, translations). Non-skill files. |

---

## 9. Commands

All commands run from the repo root (`/home/krrish/Desktop/Project/openenv-DGXAI`).

| Purpose | Command |
|---|---|
| **CODEFORGE tests** | `python3 -m pytest CODEFORGE/tests/ -v` |
| **CODEFORGE coverage** | `python3 -m pytest CODEFORGE/tests/ --cov=CODEFORGE/codeforge --cov-report=term` |
| **CODEFORGE lint** | `ruff check CODEFORGE/codeforge/` |
| **CODEFORGE types** | `mypy --strict CODEFORGE/codeforge/` |
| **CODEFORGE server** | `uvicorn CODEFORGE.codeforge.app:app --host 0.0.0.0 --port 7860` |
| **CODEFORGE baseline** | `python3 CODEFORGE/inference.py` |
| **CODEFORGE Docker** | `docker build -t code-forge CODEFORGE/ && docker run -p 7860:7860 code-forge` |
| Old tests (reference) | `python3 -m pytest tests/ -v` |
| Deploy to HF | `git push space main` |

---

## 10. Tasks (3 levels, unchanged)

| Level | Task ID | Budget | Target | Tools |
|---|---|---|---|---|
| easy | `greet_single_file` | 4 | 0.90 | ruff, imports, mypy |
| medium | `greet_with_tests` | 6 | 0.80 | ruff, imports, mypy, pytest |
| hard | `multi_file_module` | 10 | 0.70 | ruff, imports, mypy, pytest |

---

## 11. Round-1 EpistemicNav (DO NOT TOUCH)

Round 1 shipped 2026-04-08. Left untouched under `server/`, `data/`, and git history for judge verification. Round-1 models (`EpistemicAction`, `EpistemicObservation`, `ActionType`, `EvidenceSnippet`) coexist in `models.py` — do not modify them.

Verify: `git diff d252064..HEAD -- server/ | wc -l` should be 0.

---

## 12. Sanity Checks (before every commit)

1. All CODEFORGE tests green: `python3 -m pytest CODEFORGE/tests/ -v`
2. Old tests still green (don't break the reference): `python3 -m pytest tests/ -v`
3. Audit invariant: every reward traces to (sandbox, grounding, skill) triple
4. Round-1 `server/` untouched: `git diff d252064..HEAD -- server/ | wc -l` = 0
5. No hardcoded secrets
6. `ruff check CODEFORGE/codeforge/` clean, `mypy --strict CODEFORGE/codeforge/` clean
7. SYSTEM_DESIGN.md matches what the code actually does

---

## 13. What NOT To Do

- Do NOT use Graphify's Claude API extraction — grading must be LLM-free and deterministic
- Do NOT import snarktank/ralph bash scripts — our Python ralph_orchestrator is the foundation
- Do NOT scrape everything-claude-code's duplicate directories (.agents/, .cursor/, translations) — only `skills/*/SKILL.md`
- Do NOT modify `server/`, `data/`, or Round-1 models in `models.py`
- Do NOT add the uncertain floor value to MCP tool descriptions (leaks the exploit)
- Do NOT mock the sandbox or grounder in integration tests — run real tools
- Do NOT add features beyond what the current phase requires
- Do NOT push until the phase checklist is green

---

## 14. Skills (for the orchestrator)

| Skill | When |
|---|---|
| `/tdd`, `/python-testing` | Every module implementation |
| `/python-review` | Quality reviewer prompt |
| `/security-review` | M6 audit ledger, M9 MCP server |
| `/superpowers:systematic-debugging` | Test suite regressions |
| `/superpowers:verification-before-completion` | Before ticking any checkbox |
| `/huggingface-skills:hf-cli` | HF Space deployment |

Do NOT invoke `/superpowers:brainstorming` — design is locked in `CODEFORGE/SYSTEM_DESIGN.md`.

---

**This file IS the execution plan. `CODEFORGE/SYSTEM_DESIGN.md` IS the spec. Read both before any major decision. Tick checkboxes as you complete them. Work phase by phase. Critic every module. Fix what critics find. Move on only when NOTHING_FURTHER.**
