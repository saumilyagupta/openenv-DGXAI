---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Active Project: GroundLoop MCP (Offline Hackathon, Round 2)

**Status:** Design phase (as of 2026-04-15). Round 1 (EpistemicNav) shipped 2026-04-08 — see archive section below.

### One-Line Pitch

An **stdio MCP server** that takes a product brief and autonomously ships a **grounded, audit-trailed, production-grade Python codebase** — zero human intervention between the brief and the finished repo.

### Philosophical Lineage

GroundLoop fuses four ideas into one tool:

1. **Karpathy's Autoresearch (Mar 2026, ~630 LoC)** — an autonomous agent that edits a training script, runs short experiments, checks a metric, keeps/discards, and repeats. GroundLoop applies the same loop to **codebase construction** instead of hyperparameter search.
2. **EpistemicNav (Round 1)** — Brier-calibrated confidence. Every load-bearing generation step emits `{claim, confidence, evidence_ids}`; low-confidence outputs are blocked from advancing.
3. **`/graphify`** — input → knowledge graph → clustered communities → audit report. The project brief + ingested docs become a graph; every generated artifact must cite a graph node.
4. **Ralph loop** — persistent, self-driving iteration: plan → build → audit → patch → repeat, until exit criteria hit.

### Target MCP Tool Surface (draft)

| Tool | Purpose |
|---|---|
| `interrogate(brief)` | Socratic question batch. Forces the LLM to ask, not assume, until spec ambiguity drops below threshold. |
| `ingest_sources(urls, files)` | Builds graph KB + BM25/vector index over Python docs (Context7, readthedocs, repo READMEs). Returns `graph_id`. |
| `ground_check(claim, graph_id)` | Returns `{verdict, brier_conf, citations[], uncertain_reason}`. Refuses ungrounded claims. |
| `autonomous_build(spec, graph_id, exit_criteria)` | Runs the Ralph loop. Each iteration gated by `ground_check` before code is written/modified. |
| `audit_report(run_id)` | Hallucination rate, ungrounded claims, calibration curve, graph coverage, test/coverage deltas per iteration. |

### Two-Layer Grounding Architecture

GroundLoop grounds on **two independent knowledge bases**:

**Layer A — Library/API Ground Truth ("does this symbol exist?")**

Python-only constraint shrinks the hallucination surface. Every factual code claim is verifiable.

| Check | Verification |
|---|---|
| **Import graph** | Every import resolves in a pinned `requirements.txt`; no phantom packages. |
| **Symbol graph** | Every `lib.foo.bar()` call verified via AST + installed-package introspection. |
| **Version graph** | Method signatures match the pinned version (catches version-drift hallucinations). |
| **Doc citations** | Each non-trivial API usage cites a Context7/readthedocs node. |
| **Test groundedness** | Tests assert against real behavior, not LLM-imagined behavior. |

**Layer B — Reasoning Policy KB ("how should I think?")**

The scraped Claude Code **skills corpus** becomes GroundLoop's distilled "how to reason" policy. 91 SKILL.md files already on disk (as of 2026-04-15), covering tdd-workflow, security-review, python-patterns, backend-patterns, api-design, verification-loop, systematic-debugging, etc. — every skill is a distilled decision scaffold.

**Scrape sources (priority order):**

1. `~/.claude/skills/` — 57 files, primary.
2. `~/.claude/.agents/skills/` — 24 files, superpowers skills.
3. `~/.claude/.cursor/skills/` — 6 files.
4. `~/.claude/plugins/marketplaces/` — 4 files (startup-skill etc.).

**Scrape pipeline:**

- Walk each source, parse YAML frontmatter (`name`, `description`, `triggers`) + body.
- Chunk body by markdown headers; embed each chunk as a graph node.
- Tag nodes by domain (python, security, testing, api, frontend, etc.) and by phase (plan, build, review, verify).
- Index with BM25 (reuse `server/retriever.py` from EpistemicNav) + optional vector index for semantic lookup.

**Runtime use inside the Ralph loop:**

Before each autonomous decision, GroundLoop queries the policy KB for the matching reasoning scaffold:

| Decision point | KB query → policy node |
|---|---|
| "Start a new feature" | `tdd-workflow`, `plan`, `superpowers:writing-plans` |
| "Design an API" | `api-design`, `backend-patterns` |
| "Handle secrets" | `security-review`, `security-scan` |
| "Write tests" | `python-testing`, `tdd-workflow`, `verification-loop` |
| "Something broke" | `superpowers:systematic-debugging`, `build-fix` |
| "About to claim done" | `superpowers:verification-before-completion` |

This converts free-form LLM judgement into a **policy-grounded reasoning graph** — the loop's next action is selected against the best-matching skill node, and the audit report records which skill each decision cited. Hallucinated reasoning ("let me just skip tests here") is caught because the selected skill node contradicts it.

### Exit Criteria for the Autonomous Loop

1. All generated code compiles + typechecks (`ruff`, `mypy`).
2. Test suite passes with ≥80% coverage (`pytest --cov`).
3. Every architectural decision traces to a KB graph node (citations).
4. Hallucination audit: 0 ungrounded imports, 0 invented methods, 0 fabricated params.
5. Security scan clean (`pip-audit`, `bandit`, no secrets).
6. Brier-calibrated confidence ≥ threshold on all load-bearing claims.

### Composite Metric (Analogue to Autoresearch's `val_bpb`)

```
score = (groundedness × test_pass_rate × coverage) − (hallucination_count × penalty)
```

Goal: strictly increasing across iterations until exit criteria hit, OR iteration budget exhausted.

### Demo Showpiece

> Judge gives a one-line brief. User pastes it into Claude Code (which has GroundLoop MCP attached). GroundLoop runs overnight. Next morning: a working, tested, grounded Python service (e.g. FastAPI + Postgres + pytest + OpenTelemetry) with a full audit trail showing every decision cited to a KB doc node. **Zero human intervention between paste and result.**

### Hackathon Constraints (Unknown — TBD)

Duration, team size, hardware (DGX access hinted by repo name `openenv-DGXAI`), judging rubric, demo format. Clarify before committing to scope caps.

### Review Workflow (post-implementation, per slice)

Critic and verifier are spawned as a **team via `TeamCreate`** — never via plain `Agent`. Rules:

1. **Team name:** `groundloop-review`. Created once.
2. **Fresh critic per round:** each round spawns a brand-new teammate (`fresh-critic-1`, `fresh-critic-2`, …). A critic is **never reused** — re-prompting the same teammate produces stale critique because they already own their prior conclusions. One teammate = one critique. Then archive and spawn a new one for the next round.
3. **Persistent verifier:** one teammate named `verifier` (spawned once, reused across rounds via `SendMessage`). Its job: run the actual grounding checks, tests, and exit-criteria metrics on the current slice. Verifier carries context across rounds so it can measure deltas.
4. **Loop:** build slice → fresh critic critiques → verifier measures → apply fixes → spawn next fresh critic → repeat until the latest critic returns "nothing to improve" AND verifier confirms all exit criteria pass.
5. **No self-critique from the orchestrator.** The orchestrator never critiques its own design — that critique is stale by definition.
6. **Shutdown:** graceful `shutdown_request` to each teammate when the slice is done, then `TeamDelete` only after all members are idle.

This workflow applies to **each implementation slice of GroundLoop itself**, not to the autonomous loop GroundLoop runs on user briefs (that loop is the product's own internal Ralph loop).

### Design-Phase Open Questions

- [ ] Loop concurrency model: single-agent sequential vs. Autoresearch-style parallel experiments with "keep best"?
- [ ] KB ingestion: live Context7 MCP federation vs. pre-built offline snapshot for Layer A?
- [ ] Sandbox: local venv, Docker, or `firecracker`/`e2b`-style microVM for safety?
- [ ] Resume/checkpoint: can the loop survive a mid-run crash and continue?
- [ ] Output repo scaffold: opinionated (FastAPI template) or synthesized per-brief?
- [ ] Skills-KB refresh cadence: scrape once at install vs. watch `~/.claude/skills/` for changes?
- [ ] Skill-node granularity: one node per SKILL.md vs. one node per markdown section?
- [ ] Conflict resolution: when two skill nodes contradict (e.g., language-specific override of a common rule), which wins? (Prior memory: language-specific rules take precedence — carry that forward.)

---

## Archive: EpistemicNav (Round 1 — Shipped 2026-04-08)

Round 1 deliverable for the Meta PyTorch OpenEnv Hackathon x Scaler SST 2026. An OpenEnv-compliant RL environment that trains LLM agents to reason accurately under uncertainty, rewarding calibrated confidence (Brier score) rather than just correct answers. **Submitted and cleared round 1.** Code lives in this repo — reused as the epistemic core for GroundLoop.

### Core Components (reusable for GroundLoop)

*   **`server/environment.py`** — `EpistemicNavEnvironment(Environment)`: claims, evidence gathering via BM25, budget tracking, step/reset/state loop.
*   **`server/app.py`** — FastAPI app via `openenv.create_app()` exposing `/step`, `/reset`, `/state` on `0.0.0.0:7860`.
*   **`server/grader.py`** — Brier score reward. Penalises overconfidence on wrong and underconfidence on right. `verdict="uncertain"` on genuinely uncertain claims → min 0.70 reward.
*   **`server/retriever.py`** — BM25Okapi wrapper over `data/evidence.json`. Pure Python, <10ms/query, no GPU. **Reuse target for GroundLoop KB retrieval.**
*   **`models.py`** — Pydantic v2: `EpistemicAction` (QUERY/COMMIT), `EpistemicObservation`, `EvidenceSnippet`.
*   **`client.py`** — `EpistemicEnv(GenericEnvClient)`: HTTP client.
*   **`inference.py`** — Baseline LLM agent over OpenAI-compatible API. <20 min on 2vCPU/8GB.

### Data

*   **`data/claims.json`** — 400 claims (200 easy, 150 medium, 50 hard).
*   **`data/evidence.json`** — 2000 snippets across 15+ domains.

### Three Tasks

1. **easy (single_hop):** Single-hop factual claim. Reward ceiling ~0.98.
2. **medium (multi_hop):** Multi-hop, 3–4 evidence pieces. Reward ceiling ~0.88.
3. **hard (contradictory):** Contradictory evidence. Correct answer is `"uncertain"`. Reward floor 0.70.

### Round 1 Development Guidelines (still authoritative for any EpistemicNav edits)

*   **OpenEnv Spec Strictness:** All API endpoints strictly adhere to OpenEnv spec and `problem_statement_and_guidelines.md`.
*   **Anti-Reward Hacking:** Rewards tied to Brier calibration. Query steps return 0.0 reward. No negative rewards ([0,1] range).
*   **Determinism:** Grader returns reproducible scores in [0.0, 1.0].
*   **No External Calls from Env:** All evidence pre-cached. No web requests from the server.
*   **Containerization:** Docker on HF Spaces. `0.0.0.0:7860`. Image ~280MB (python:3.11-slim).

### Round 1 Common Commands

*   **Local Server:** `uvicorn server.app:app --host 0.0.0.0 --port 7860`
*   **Docker Build:** `docker build -t epistemic-nav .` then `docker run -p 7860:7860 epistemic-nav`
*   **Baseline Agent:** `python inference.py` (requires API keys in `.env`)
*   **Deploy:** `git push space main`

---

## Skill Usage Guide

Invoke via `/skill-name`. The guide applies to both GroundLoop (active) and any EpistemicNav maintenance work.

### Phase 1: Planning

| Skill | When to Use |
|---|---|
| `/superpowers:brainstorming` | **Next step for GroundLoop.** Before any creative/architectural decision. |
| `/superpowers:writing-plans` | After brainstorming converges — lock the multi-step implementation spec before code. |
| `/plan` | Before each implementation chunk — step-by-step plan, risks identified. |

### Phase 2: Research & Knowledge Base

| Skill | When to Use |
|---|---|
| `/deep-research` | Research MCP patterns, Autoresearch internals, graph-KB implementations, existing grounded-codegen tools. |
| `/exa-search` | Broader web research when GitHub/docs are insufficient. |
| `/graphify` | Convert research + docs into a knowledge graph (directly reused as GroundLoop's KB backbone). |

### Phase 3: Python Development

| Skill | When to Use |
|---|---|
| `/python-patterns` | Writing Python — idiomatic patterns, type hints, Pydantic v2, dataclasses. |
| `/coding-standards` | Code quality, naming, file organization. |
| `/backend-patterns` | If GroundLoop exposes an HTTP MCP variant or the generated demo project is a backend. |
| `/tdd` | Before implementing any new feature. RED → GREEN → IMPROVE. Target 80%+ coverage. |
| `/python-testing` | pytest strategies, fixtures, mocking, parametrization. |
| `/superpowers:test-driven-development` | Before writing implementation code for any feature or bugfix. |
| `/claude-api` | If GroundLoop's internal loop calls Claude API directly (prompt caching, tool use, batch). |

### Phase 4: MCP-Specific (GroundLoop)

| Skill | When to Use |
|---|---|
| *(MCP builder skill TBD — check `/find-skills` for MCP server scaffolding)* | Bootstrapping the stdio MCP server, tool schemas, manifest. |
| `/plugin:context7:context7__query-docs` | Live library-doc grounding during codegen. |
| `/pinecone:quickstart` | Optional vector KB backend (BM25 is the default per EpistemicNav reuse). |

### Phase 5: Code Review & Quality

| Skill | When to Use |
|---|---|
| `/python-review` | After writing Python — PEP 8, type hints, security, idioms. |
| `/code-review` | After each chunk — quality, security, maintainability. |
| `/simplify` | Review changed code for reuse, quality, efficiency. |
| `/security-scan` | Before commits — leaked keys, config issues, injection risks. |
| `/security-review` | When handling user input, secrets, API endpoints. |

### Phase 6: Build & Verification

| Skill | When to Use |
|---|---|
| `/build-fix` | When build or MCP server startup fails. |
| `/superpowers:verification-before-completion` | Before claiming any task done. Run verification, confirm output. |
| `/superpowers:systematic-debugging` | Bugs, test failures, unexpected loop behavior. |
| `/verify` | Full verification loop before shipping. |

### Phase 7: Documentation & Finish

| Skill | When to Use |
|---|---|
| `/update-docs` | Update README with MCP tool surface, setup, demo instructions. |
| `/superpowers:finishing-a-development-branch` | When implementation complete and tests pass. |
| `/superpowers:requesting-code-review` | Before final submission. |

### Utility Skills

| Skill | When to Use |
|---|---|
| `/aside` | Quick side question without losing context. |
| `/superpowers:dispatching-parallel-agents` | 2+ independent tasks (e.g., build grader + build retriever simultaneously). |
| `/save-session` | Save state before ending work. |
| `/resume-session` | Start of new session — load prior context. |

### Skills NOT Relevant

- Vercel / deployment-platform skills (stdio MCP is local; hosted variant would target HF Space if ever).
- Frontend/React/Next.js/Figma skills (no UI).
- Django / Spring Boot / Go / Kotlin / Java skills (Python-only project).
- Heavy database skills (KB is flat JSON + BM25, optional vector DB).

---

## Round 1 Submission Checklist (Historical — all shipped)

- [x] `claims.json` — 400 claims, balanced distribution (200 easy, 150 medium, 50 hard)
- [x] `evidence.json` — 2000 snippets, 15+ domains
- [x] `models.py` — typed Pydantic v2 Action + Observation
- [x] `server/environment.py` — step(), reset(), state() implemented
- [x] `server/grader.py` — Brier score, scores in [0.0, 1.0]
- [x] `server/retriever.py` — BM25, top-k, <10ms per query
- [x] `Dockerfile` builds locally
- [x] `openenv.yaml` valid
- [x] HF Space deployed
- [x] `inference.py` in root — uses API_BASE_URL, MODEL_NAME, HF_TOKEN
- [x] `inference.py` runtime <20 min on 2vCPU / 8GB
- [x] `README.md` — action space, observation space, setup, task descriptions
- [x] Pre-submission validation script passed

## Round 2 Submission Checklist (GroundLoop — TBD)

- [ ] Brainstorm complete → spec locked
- [ ] Written plan approved
- [ ] MCP server scaffold (stdio transport, tool manifest)
- [ ] `interrogate`, `ingest_sources`, `ground_check`, `autonomous_build`, `audit_report` tools implemented
- [ ] KB graph builder (reuses EpistemicNav BM25 retriever + Context7 for live Python docs)
- [ ] Composite metric tracker + audit-report generator
- [ ] Ralph loop orchestrator with checkpoint/resume
- [ ] Python sandbox (venv or Docker) for generated-code verification
- [ ] Test suite ≥80% coverage on GroundLoop itself
- [ ] Demo brief + recorded overnight run showing a working output repo
- [ ] README + audit-report sample
