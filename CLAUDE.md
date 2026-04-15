---
description: 
alwaysApply: true
---

# CLAUDE.md — CodeForge OpenEnv: Autonomous Completion Playbook

This file is the **operating manual** for driving CodeForge from its current MVP state to fully-wired completion in one autonomous run. It is the authoritative spec, sequence, and quality gate. Any future session reading this should pick up and execute without asking.

---

## 0. The Prime Directive

**Complete CodeForge non-stop.** No approval gates. No clarifying questions once a module is picked. Every decision is taken by the orchestrator; every module is gated by a fresh-critic team — critics are the quality floor, not the human.

Stop conditions, in order of precedence:

1. A genuinely destructive action is about to happen (force push, hard reset, deleting user data, touching files outside this repo).
2. The entire CodeForge completion checklist (§7) is green.

Everything else is proceed-without-asking.

---

## 1. The Mission: Fully-Wired CodeForge

CodeForge is an OpenEnv-compliant RL environment where the agent iteratively builds a Python codebase from a natural-language brief. The MVP (shipped 2026-04-15) exposes `query_kb` + `submit`. The completion goal is the fully-wired version below.

### The One-Sentence Invariant

> **Every reward-earning action in CodeForge must trace to (a) a sandbox-verified programmatic signal, (b) a Layer-A grounded symbol, and (c) a Layer-B skill citation — recorded in the audit trail as a `(reward, evidence, policy)` triple.**

### 4 Pillars → 4 Integrations

1. **Karpathy Autoresearch** → `run_ralph` action: agent requests N autonomous iterations (plan → synthesize → sandbox-score → keep-if-better), returns best trajectory.
2. **EpistemicNav Brier calibration** → agent declares `confidence ∈ [0,1]` on submit; `reward = quality * (1 - min(brier, 0.5))`. "Uncertain" is first-class (floor 0.70 if confidence<0.3 AND quality<0.5).
3. **Graphify clustering** → `query_cluster` action: 1006 skill nodes → Jaccard-edge graph → connected-component communities → agent retrieves by `cluster_label`.
4. **Ralph persistence** → per-step `AuditLedger` + checkpoint-on-disk; `get_audit` returns the full `(reward, evidence, policy)` trail, resumable across env restarts.

### 6 Actions (target surface)

| Action | Cost | Reward | Wires |
|---|---|---|---|
| `query_kb` | 1 budget | 0.0 | Layer B flat BM25 |
| `query_cluster` | 1 budget | 0.0 | Pillar 3 — graph community |
| `interrogate` | 1 budget | 0.0 | Socratic front-loading |
| `run_ralph` | N budget | Best iter's calibrated reward (minus 0.05/iter wasted) | Pillar 1 + 4 |
| `submit` (w/ confidence) | 1 budget | `calibrated_reward` | Pillar 2 |
| `get_audit` | 0 budget | 0.0 | Pillar 4 — read-only ledger |

### Reward formula (target)

```
quality  = 0.6 * sandbox_composite_score + 0.4 * grounding_score
brier    = (confidence - quality)² if confidence provided else 0.0
penalty  = min(brier, 0.5)
reward   = quality * (1 - penalty)
reward   = 0.70 if (confidence < 0.3 AND quality < 0.5) else reward   # "uncertain" floor
reward   = round(max(0.0, min(1.0, reward)), 3)
```

Deterministic, clamped to `[0, 1]`, no negatives.

### Observation & Action schemas (target)

```python
class CodeForgeAction(Action):
    action_type: Literal["query_kb", "query_cluster", "interrogate",
                         "run_ralph", "submit", "get_audit"]
    # existing
    claim: str | None = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    files: dict[str, str] | None = None
    # new
    cluster_label: str | None = None              # query_cluster
    max_iters: int = 3                            # run_ralph
    confidence: float | None = None               # submit (triggers Brier)
    target_run_id: str | None = None              # get_audit

class CodeForgeObservation(Observation):
    # existing
    episode_id: str
    task_id: str
    task_level: str
    task_brief: str
    initial_files: dict[str, str]
    current_files: dict[str, str]
    budget_remaining: int
    previous_score: float
    last_citations: tuple[dict, ...]
    last_grounding: dict | None
    is_done: bool
    last_reward: float
    # new
    last_cluster_hits: tuple[str, ...] = ()
    last_interrogation_questions: tuple[str, ...] = ()
    last_ralph_run_id: str | None = None
    last_ralph_iterations: tuple[dict, ...] = ()
    cumulative_audit_summary: dict = {}
```

### AuditLedger (per-episode)

```python
@dataclass(frozen=True)
class AuditEntry:
    step_index: int
    action_type: str
    cited_skill_ids: tuple[str, ...]
    cited_clusters: tuple[str, ...]
    grounding_report: dict | None
    reward: float
    brier_penalty: float | None
    confidence_declared: float | None
    quality: float
```

Every `step()` appends one entry. `get_audit` returns the ledger.

---

## 2. Execution Sequence (the 8 modules)

Build in this order. Each module is a complete slice (code + tests + review loop). Do NOT reorder — later modules depend on earlier ones.

| # | Module | Est. time | Depends on |
|---|---|---|---|
| M1 | **Graphify clustering** — `groundloop/kb_indexer/cluster.py`, `cluster_manifest.json`, extend `SearchResult` with `cluster_id`. | ~2h | kb_indexer (shipped) |
| M2 | **Brier-calibrated reward** — extend `groundloop_env/grader.py`, add `confidence` field to `CodeForgeAction`, update schema. | ~30m | models.py |
| M3 | **`query_cluster` action** — env handler reading from cluster manifest. | ~30m | M1 |
| M4 | **`interrogate` action** — env handler wrapping `groundloop/interrogator/`. | ~30m | kb_indexer (shipped) |
| M5 | **`run_ralph` action** — env handler wrapping `groundloop/ralph_orchestrator/run_loop`, with budget accounting + reward plumbing. | ~2h | M2, ralph_orchestrator (shipped) |
| M6 | **`AuditLedger` + `get_audit` action** — per-step ledger append, `groundloop/audit_reporter` integration for summaries. | ~1h | M2, M5 |
| M7 | **Observation/Action schema updates** — add all new fields; update `observation_builder.py` to populate them. | ~20m | M1–M6 |
| M8 | **Full integration test + baseline re-score + README update** — one episode exercising all 6 actions; regenerate baselines table. | ~1h | all |

**Total: ~8 hours of implementer time.**

After M8: the submission-blocker fixes (§7) — Docker corpus bake, HF Space, openenv validate.

---

## 3. Per-Module Workflow (rigid)

Every module goes through this exact sequence. Do NOT skip any step.

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Write spec:  docs/superpowers/specs/YYYY-MM-DD-codeforge-Mn.md   │
│ 2. Write plan:  docs/superpowers/plans/YYYY-MM-DD-codeforge-Mn.md   │
│ 3. Commit spec + plan. Message: "docs: Mn <name> spec + plan"       │
│ 4. TeamCreate team_name = "codeforge-mN-<slug>"                     │
│ 5. Agent(name="implementer", team_name=...) — runs all plan tasks   │
│ 6. Wait for STATUS: DONE report                                     │
│ 7. Agent(name="spec-reviewer-1", team_name=..., FRESH) — verifies   │
│    against spec. Reports VERDICT: PASS | FAIL.                      │
│ 8. If FAIL: SendMessage implementer with fix list. Goto 6.          │
│ 9. Agent(name="quality-reviewer-1", team_name=..., FRESH) — verifies │
│    code quality. Reports NOTHING_FURTHER | APPROVED_WITH_CONCERNS | │
│    NEEDS_CHANGES.                                                    │
│ 10. If not NOTHING_FURTHER: SendMessage implementer with fixes.      │
│     Spawn fresh-critic-N+1. Goto 10 until NOTHING_FURTHER.           │
│ 11. TeamDelete (force with `rm -rf ~/.claude/teams/<name>` if stuck) │
│ 12. Git log: confirm Mn commits exist.                              │
│ 13. Update §7 checklist in THIS file.                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Hard rules (non-negotiable)

- **Fresh critic every round.** Never reuse a spec-reviewer or quality-reviewer. `fresh-critic-N+1` each time. Critics know their prior conclusions — reusing them produces stale critique.
- **Implementer is persistent** — reuse via `SendMessage` for fix rounds.
- **TeamCreate, not raw Agent**, for all critic/implementer spawns. Enforces the review pattern.
- **TeamDelete directly** when convergence hit; don't wait for shutdown_request handshake (it doesn't always work). If stuck, `rm -rf ~/.claude/teams/<name> ~/.claude/tasks/<name>` then call TeamDelete.
- **Write to disk as you go** — specs and plans persist in `docs/superpowers/` for traceability. Don't keep design in memory only.
- **Every implementer task = one commit.** Conventional commits: `feat(codeforge-mN): ...`, `fix(codeforge-mN): ...`, `test(codeforge-mN): ...`, `docs(codeforge-mN): ...`.
- **TDD strictly.** Test first, confirm RED, implement, confirm GREEN. No exceptions.
- **`from __future__ import annotations`** on every module.
- **No push, no merge until §7 is green.** All work lives on local `main`.

### Red flags the orchestrator must catch

- Implementer silent for >5 minutes → SendMessage status check.
- Critic returns "looks good" with no specific citations → dispatch another fresh critic (suspect hallucinated review).
- Test count drops between commits → regression; stop and investigate.
- Coverage drops below 85% on the package under work → block commit until recovered.
- Any `ruff check` or `mypy --strict` warning introduced → block commit until clean.

---

## 4. The Shared Substrate (already shipped, reuse only)

Do NOT modify these packages during CodeForge completion. Import and consume only.

| Package | What it provides |
|---|---|
| `groundloop/skills_scraper/` | 1006-node Layer B corpus. `run_scraper(sources, output)` → JSONL + manifest. |
| `groundloop/kb_indexer/` | `SkillsIndex.build/save/load/search`. BM25 over skill corpus. **M1 extends this with clustering — adding a new file, not modifying existing.** |
| `groundloop/python_sandbox/` | `run_sandbox(files, tools)` → `SandboxResult` with `composite_score`. Programmatic code grader. |
| `groundloop/lib_grounder/` | `ground(source)` → `GroundingReport{groundedness}`. AST symbol check. |
| `groundloop/ralph_orchestrator/` | `run_loop(spec, files, index, synthesizer, config)` → `RunResult`. The Ralph loop. **M5 wraps this.** |
| `groundloop/interrogator/` | `Interrogator(index).generate(brief)` → 5 questions + cited node IDs. **M4 wraps this.** |
| `groundloop/audit_reporter/` | `AuditReporter.build(run)` → `AuditReport`. **M6 integrates this.** |

Round-1 env (`server/`, `data/`) is **untouched** — judge needs it for round-1 verification.

---

## 5. Environment Behavior Reference

### `reset(task_level)`

Pick task by level. Initialize: `budget_remaining = task.max_budget`, `current_files = task.initial_files`, `previous_score = 0.0`, `is_done = False`, fresh `AuditLedger`. Return observation.

### `step(action)` — routing table

| action_type | Budget | Response |
|---|---|---|
| `query_kb` | `-=1` | `SkillsIndex.search(claim, top_k, required_tags)` → `last_citations` populated, reward=0, audit entry logged |
| `query_cluster` | `-=1` | lookup cluster_label in manifest, return top-k nodes in cluster → `last_cluster_hits` populated, reward=0, audit entry logged |
| `interrogate` | `-=1` | `Interrogator(index).generate(task_brief)` → `last_interrogation_questions` populated, reward=0, audit entry logged |
| `run_ralph` | `-=max_iters` | `ralph_orchestrator.run_loop(...)` → `last_ralph_run_id`, `last_ralph_iterations`, reward = `calibrated_reward(final_score, confidence=0.75) - 0.05 * wasted_iters`, audit entry logged, checkpoint persisted |
| `submit` | `-=1` | `run_sandbox + ground` → compute quality → Brier penalty if `confidence` given → `last_reward = calibrated_reward`, audit entry logged. If reward ≥ task.target_score: `is_done = True` |
| `get_audit` | 0 | read-only, returns serialized `AuditLedger` of target_run_id (default = current episode), no budget change, no audit entry |

Done detection: `is_done = True` when `budget_remaining <= 0` OR submit reward ≥ target.

### Tasks (unchanged)

1. **easy `greet_single_file`** — budget 4, target 0.90, tools `ruff+imports+mypy`.
2. **medium `greet_with_tests`** — budget 6, target 0.80, tools `ruff+imports+mypy+pytest`.
3. **hard `multi_file_module`** — budget 10, target 0.70, tools `ruff+imports+mypy+pytest`.

---

## 6. Quality Gates (applied to every module)

Before closing a module:

- [ ] `python3 -m pytest tests/ -v` → all pass
- [ ] `python3 -m pytest tests/<module>/ --cov=<pkg>` → coverage ≥ 85%
- [ ] `ruff check <pkg>/` → clean
- [ ] `mypy --strict <pkg>/` → clean (openenv-core stub-missing errors are acceptable; application errors are NOT)
- [ ] No new deps added to `requirements.txt` (everything needed is already there)
- [ ] Spec self-review: grep plan for TBD/TODO/implement-later placeholders — zero tolerance
- [ ] Fresh spec-reviewer returned VERDICT: PASS
- [ ] Fresh quality-reviewer returned NOTHING_FURTHER
- [ ] Commits follow convention `<type>(codeforge-mN): <what>`

If any gate fails: dispatch a fresh implementer-fix-round with the specific gate failure. Loop until all gates pass.

---

## 7. Submission Completion Checklist

Tick items as they land. Last 4 are submission-blocking.

### Code milestones

- [x] **M1** Graphify clustering shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M2** Brier-calibrated reward shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M3** `query_cluster` action shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M4** `interrogate` action shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M5** `run_ralph` action shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M6** `AuditLedger` + `get_audit` action shipped, tested, reviewed, NOTHING_FURTHER
- [ ] **M7** Observation/Action schemas updated across all tests
- [ ] **M8** Full 6-action integration test + baselines regenerated + README updated

### Submission blockers

- [ ] **Corpus baked into Docker image** — commit a frozen snapshot of `groundloop/kb/skills_corpus.jsonl` so the Space has a non-empty KB.
- [ ] **`openenv validate`** passes (install the CLI if needed; fix violations).
- [ ] **HF Space deployed** — `git push space main` (requires user's HF token; if not available at automation time, leave stubbed and flag for user).
- [ ] **Live baseline on deployed Space** — `API_BASE_URL=<space-url> python3 inference.py` produces a baseline table; record in README.

### Already done (MVP snapshot)

- [x] OpenEnv env compliant (reset/step/state, FastAPI, openenv.yaml)
- [x] 3 tasks defined + graders working
- [x] Dockerfile builds + runs locally, endpoints verified
- [x] 222/222 tests pass, 93% coverage on groundloop_env, 91% overall
- [x] `ruff check` clean, `mypy --strict` clean (modulo upstream stubs)
- [x] Round 1 EpistemicNav preserved in `server/`

---

## 8. Commands (use frequently)

| Purpose | Command |
|---|---|
| Local server | `uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860` |
| Baseline agent | `python3 inference.py` |
| Docker build | `docker build -t code-forge . && docker run -p 7860:7860 code-forge` |
| Build KB corpus | `python3 -m groundloop.skills_scraper` |
| Query KB | `python3 -m groundloop.kb_indexer search "pytest fixtures" --top-k 3` |
| Full test suite | `python3 -m pytest tests/ -v` |
| Coverage | `python3 -m pytest tests/<m>/ --cov=<pkg> --cov-report=term` |
| Lint | `ruff check <pkg>/` |
| Types | `mypy --strict <pkg>/` |
| Live smoke (new actions) | `curl -s -X POST http://localhost:7860/step -H 'content-type: application/json' -d '{"action":{"action_type":"query_cluster","cluster_label":"python_testing_pytest"}}'` |
| Deploy to HF | `git push space main` |

---

## 9. Skill Usage (for the orchestrator running this playbook)

| Skill | When |
|---|---|
| `/plan` | Before each module's spec + plan |
| `/superpowers:writing-plans` | If a module's plan needs deeper structure |
| `/tdd`, `/python-testing` | Inside every implementer dispatch |
| `/python-review` | Inform the quality-reviewer's prompt |
| `/security-review` | For M6 audit ledger (handles user-submitted code paths) |
| `/graphify` | Optional inspiration for M1 clustering |
| `/superpowers:systematic-debugging` | When a test suite regresses during a module |
| `/superpowers:verification-before-completion` | Before ticking a §7 checkbox |
| `/huggingface-skills:hf-cli` | For the HF Space submission step |
| `/update-docs` | For M8's README update |

Do NOT invoke `/superpowers:brainstorming` — design is already locked in this file.

Skills NOT relevant: Vercel, Frontend/React/Figma, Django/Spring/Go/Kotlin/Java, heavy-database.

---

## 10. Archive: Round 1 — EpistemicNav (shipped 2026-04-08)

Round 1: OpenEnv RL environment for Brier-calibrated reasoning under uncertainty. Submitted and passed. **Left untouched** under `server/` + `data/` + `openenv.yaml` git history for judge verification. Round-1 `models.py` types coexist with CodeForge types.

Do NOT modify `server/`, `data/`, round-1 `inference.py` portions that live in git history (`d252064`), round-1 portions of `models.py`.

---

## 11. Sanity Invariants (self-check before each action)

Before each commit/TeamDelete/checkbox-tick, confirm:

1. Does the env still respond correctly to all 3 tasks? (`curl -X POST /reset {"task_level":"easy"} → /step {"action":{"action_type":"submit", ...}}` → reward > 0.8)
2. Are all existing tests still green?
3. Is the audit invariant preserved — can you still trace every submitted reward back to a `(sandbox, grounding, skill)` triple?
4. Is `round 1 server/` untouched? (`git diff d252064..HEAD -- server/ | wc -l` should be 0)

If any answer is "no" or "unknown": halt, investigate, restore, then continue.

---

**This file IS the plan. Re-read it before any major decision. Update §7 checkboxes as you tick them. Nothing else should stop you until all boxes are green.**
