---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Active Project: CodeForge OpenEnv (Round 2, Offline Hackathon)

**Status:** Shipped locally (2026-04-15). Round 1 (EpistemicNav) shipped 2026-04-08 — see archive. The MCP framing was pivoted to OpenEnv on 2026-04-15 because the hackathon mandates OpenEnv compliance (step/reset/state, openenv.yaml, Dockerfile, HF Space).

### One-Line Pitch

An **OpenEnv-compliant RL environment** where the agent iteratively builds a Python codebase from a natural-language brief. Rewards combine programmatic code quality (`ruff`/`mypy`/`pytest`/`imports`) with AST-level symbol grounding. Same philosophical thread as Karpathy's Autoresearch (iterate toward a metric) and round-1 EpistemicNav (grounded, calibrated reasoning).

### Philosophical Lineage (4 pillars)

All four ideas are now implemented and callable inside the env:

1. **Karpathy's Autoresearch (Mar 2026, ~630 LoC)** — autonomous iterate-till-metric-converges loop. Implemented as `groundloop/ralph_orchestrator/run_loop`. Agent can opt into iteration via repeated `SUBMIT` actions; the env's internal score is the equivalent of Autoresearch's `val_bpb`.
2. **EpistemicNav (Round 1)** — Brier-calibrated confidence + "uncertain" verdict as a first-class outcome. Round-1 grader lives in `server/grader.py`; CodeForge reward uses a related composite `0.6*sandbox + 0.4*grounding` with clamping to `[0.0, 1.0]`.
3. **`/graphify`-inspired grounding** — input → knowledge base. CodeForge ships BM25-indexed section-level grounding over 1006 skill nodes (Layer B). Community clustering is deferred Phase 2.
4. **Ralph loop** — persistent, self-driving iteration with checkpoint/resume. Lives in `groundloop/ralph_orchestrator/`; used by `inference.py` baseline agent.

### Env Action/Observation Surface

| Action type | Purpose |
|---|---|
| `query_kb` | Search the skill corpus. Returns citations (top-k section bodies with `skill_name`, `section_path`, score, tags). Reward 0.0, costs 1 budget. |
| `submit` | Send `{files: {path: content}}`. Env runs `ruff`/`mypy`/`pytest`/`imports` via `groundloop.python_sandbox`, computes `groundedness` via `groundloop.lib_grounder`, emits reward = `0.6 * sandbox_composite_score + 0.4 * groundedness`. Costs 1 budget. |

Observation fields: `{episode_id, task_id, task_level, task_brief, initial_files, current_files, budget_remaining, previous_score, last_citations, last_grounding, is_done, last_reward}`.

### Three Tasks

1. **easy / `greet_single_file`** (budget 4, target 0.90, tools `ruff+imports+mypy`) — implement `greet(name: str) -> str`. Baseline: **1.000**.
2. **medium / `greet_with_tests`** (budget 6, target 0.80, tools `ruff+imports+mypy+pytest`) — add pytest + None-guard. Baseline: **0.920**.
3. **hard / `multi_file_module`** (budget 10, target 0.70, tools `ruff+imports+mypy+pytest`) — split into `main.py + core.py + test_core.py`, mypy-strict clean. Baseline: **0.840**.

### Two-Layer Grounding Architecture

**Layer A — Library/API Ground Truth ("does this symbol exist?")**

Implemented in `groundloop/lib_grounder/`:

| Check | Verification |
|---|---|
| Import graph | `importlib.util.find_spec` on every top-level `import x`. |
| Symbol graph | `hasattr(module, attr)` on every `from x import y` and `x.attr` reference. |
| AST-based | Pure `ast.parse` walk — no code execution. |
| Output | `GroundingReport{total, grounded, ungrounded, groundedness ∈ [0,1]}`. |

Context7 live-doc federation is deferred Phase 2.

**Layer B — Reasoning Policy KB ("how should I think?")**

The scraped Claude Code skills corpus = 1006 section-level nodes extracted from 93 `SKILL.md` files on disk (as of 2026-04-15).

Pipeline (`groundloop/skills_scraper/`):

1. Walks 4 source roots: `~/.claude/skills/`, `~/.claude/.agents/skills/`, `~/.claude/.cursor/skills/`, `~/.claude/plugins/marketplaces/`.
2. Parses YAML frontmatter + markdown body per SKILL.md.
3. Chunks body on H2/H3 boundaries (H4+ folds into parent H3).
4. Deterministic rule-based tags per node: `domain:{python,js,go,kotlin,security,frontend,backend,data,api,mcp,devops,general}`, `phase:{plan,build,test,review,deploy,debug,docs}`.
5. Emits JSONL corpus + sha256-stable manifest.

Index (`groundloop/kb_indexer/`):

- BM25Okapi over section bodies, tokenized with word-boundary regex.
- Cache invalidation via corpus sha256.
- Query API: `search(query, top_k, required_tags) -> list[SearchResult]`.
- Deterministic tie-breaking on `(-score, node_id)`.

Example decision-point → skill-node routing:

| Decision point | KB query → policy node |
|---|---|
| "Start a new feature" | `tdd-workflow`, `plan`, `superpowers:writing-plans` |
| "Design an API" | `api-design`, `backend-patterns` |
| "Handle secrets" | `security-review`, `security-scan` |
| "Write tests" | `python-testing`, `tdd-workflow` |
| "Something broke" | `superpowers:systematic-debugging` |

### Reward Formula

```
reward = 0.6 * sandbox_composite_score + 0.4 * grounding_score
# round to 3 dp; clamp to [0.0, 1.0]
```

Sandbox composite is defined by `groundloop.python_sandbox.metric.composite_score`:

```
tool_pass_rate                       = (# tools with ok=True) / (# tools run)
imports_penalty                      = min(1.0, len(unresolved) * 0.1)
ruff_penalty                         = min(ruff.count, 20) / 40
mypy_penalty                         = min(mypy.count, 20) / 40
pytest_penalty                       = 0.5 if pytest exit != 0 else 0.0
raw                                  = tool_pass_rate - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
composite_score                      = max(0.0, min(1.0, raw))
```

Deterministic: same inputs → same reward.

### Exit Criteria (per task)

- Reward ≥ `task.target_score` → `is_done=True`.
- Budget exhausted → `is_done=True`.

Agent can also opt into cross-step iteration by repeating `submit` with revisions until target hit.

### Target Architecture — Fully-Wired CodeForge (Design Vision)

This section is the **architectural blueprint** for the fully-integrated CodeForge. The current env ships the MVP (KB + grounding); this design wires in the remaining pillars so every claim in the philosophical lineage is operationally live.

#### The One-Sentence Invariant

> **Every reward-earning action in CodeForge must trace to (a) a sandbox-verified programmatic signal, (b) a Layer-A grounded symbol, and (c) a Layer-B skill citation — recorded in the audit trail as a `(reward, evidence, policy)` triple.**

Without (a) you have hallucination. Without (b) you have version-drift. Without (c) you have unprincipled code. All three are enforced by construction.

#### Four Pillars, Four Integrations

```
┌────────────────────── CodeForgeEnvironment.step(action) ───────────────────────┐
│                                                                                │
│   Pillar 1: Autoresearch  ───►  run_ralph action                               │
│   ─────────────────────         • Agent requests N autonomous iterations       │
│   (iterate-till-metric)         • Each iter: re-query KB → synthesize →        │
│                                   sandbox-score → keep-if-better → log         │
│                                 • Returns best iteration's files + trajectory  │
│                                                                                │
│   Pillar 2: EpistemicNav  ───►  Brier-calibrated reward on submit              │
│   ──────────────────────        • Agent declares `confidence ∈ [0,1]` on       │
│   (calibrated confidence)         submission — "how sure am I this builds?"    │
│                                 • Reward = quality × (1 − BrierPenalty)        │
│                                 • Overconfident bad code punished harder than  │
│                                   honestly-uncertain bad code (round-1 rule)   │
│                                                                                │
│   Pillar 3: Graphify      ───►  query_cluster action + cluster-aware KB        │
│   ───────────────────           • 1006 skill nodes → graph communities via     │
│   (graph + communities)           token-overlap clustering at build time       │
│                                 • Agent can query "give me everything in the   │
│                                   testing community" not just flat BM25        │
│                                 • Audit tracks graph_coverage = unique         │
│                                   clusters visited / total clusters            │
│                                                                                │
│   Pillar 4: Ralph Loop    ───►  Persistent run_id + checkpoint/resume          │
│   ─────────────────────         • Each submit writes a Ralph checkpoint        │
│   (self-driving + resume)       • Agent can `get_audit` on any past run_id     │
│                                 • Survives env restart (checkpoint on disk)    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### Expanded Action Surface

The agent gets **6 action types**, each with a clear reward contract:

| Action | Cost | Reward | Wires pillar |
|---|---|---|---|
| `query_kb` | 1 budget | 0.0 | Layer B flat search |
| `query_cluster` | 1 budget | 0.0 | **Pillar 3** — graph community retrieval |
| `interrogate` | 1 budget | 0.0 | **Deferred wiring**: Socratic front-loading — agent gets 3-5 clarifying questions about the task brief, answering them guides synthesis. Anti-premature-submission signal. |
| `run_ralph` | N budget (N = requested iters) | Best iteration's calibrated reward | **Pillar 1 + 4** — autonomous iterate-till-converge, with checkpoint |
| `submit` | 1 budget | `calibrated_reward` | **Pillar 2** — agent declares `confidence`, Brier penalty applied |
| `get_audit` | 0 budget (read-only) | 0.0 | **Pillar 4** — returns full `(reward, evidence, policy)` trail for the episode |

#### Calibrated Reward Formula (Pillar 2 + everything)

```
quality  = 0.6 * sandbox_composite_score + 0.4 * grounding_score         # current formula
brier    = (confidence - actual_quality)^2                               # EpistemicNav round-1 calibration
penalty  = min(brier, 0.5)                                               # cap at 50%, no negative
final    = quality * (1 - penalty)
final    = round(max(0.0, min(1.0, final)), 3)                           # clamp + determinism
```

Special case (round-1 rule): if `confidence < 0.3` AND `quality < 0.5`, reward floor = 0.70 — "I don't know" is a valid, rewarded answer.

#### Layer-B KB: Skills as Operational Policy

The 1006 skill nodes are not just retrievable — they're **the authoritative source of correct-thinking patterns**. Every skill is a distilled decision scaffold.

**Graph construction (build time, one-shot):**

1. `skills_corpus.jsonl` → embedding tokenization per section body.
2. Pairwise Jaccard similarity on token sets — threshold 0.2 → edge.
3. Connected-component clustering → `cluster_id` per node (typically 30-80 communities).
4. Label each cluster by its top-3 tokens + most-cited `domain` tag (e.g., cluster `test_pytest_coverage`, `security_input_validation`, `backend_api_design`).
5. Persist `cluster_manifest.json` alongside `skills_corpus.jsonl` with `{cluster_id: {label, node_count, member_node_ids}}`.

**Runtime use:**

- `query_kb(claim)` → flat BM25 as today.
- `query_cluster(cluster_label)` → returns all nodes in that cluster (bounded by top-k).
- Audit records `graph_coverage = visited_clusters / total_clusters` per episode — diverse-coverage agents score higher in post-hoc analysis (not in reward, but reported).

**Example cluster labels (derived from 1006 nodes):**

- `python_testing_pytest` (pytest patterns, fixtures, parametrization, coverage)
- `security_authz_secrets` (auth flows, secret handling, OWASP)
- `backend_api_rest_fastapi` (endpoint design, error handling, envelopes)
- `tdd_workflow_red_green` (tdd cycle, test-first discipline)
- `debugging_systematic` (bisect, logs, minimal repros)
- `docs_readme_spec` (documentation patterns)
- `mcp_server_stdio` (MCP protocol, tool manifest)
- `claude_api_prompt_caching` (Claude/Anthropic API patterns)
- `deployment_docker_hf_space` (containerization, HF deploy)
- `golang_patterns`, `kotlin_patterns`, `django_patterns`, … (language-specific pods)

Cluster labels are **emergent, not hand-coded** — the graph discovers coherent domains from the skill corpus automatically.

#### Layer-A KB: Symbol Ground Truth

Already shipped via `lib_grounder`. Target extension:

- Attach `GroundingReport` to every submit.
- Introduce **negative grounding**: every `hasattr(mod, "fabricated_method")` miss → contributes to `hallucination_count` in audit.
- Future Phase 3: Context7 live-doc integration — currently deferred; AST-only grounding is already discriminative (catches invented imports, invented method names).

#### Ralph Loop as a Step Action (Pillar 1 + 4)

When the agent issues `run_ralph(spec=obs.task_brief, max_iters=5)`:

1. Env calls `ralph_orchestrator.run_loop(spec, current_files, index, synthesizer, config)`.
2. Inside the loop, each iteration:
   - Calls KB (cited nodes logged).
   - Synthesizes new files (StubSynthesizer by default; OpenAISynthesizer if `OPENAI_API_KEY` set).
   - Runs sandbox for the score.
   - Keeps if better, reverts if worse.
3. Returns `RunResult` to the env.
4. Env sets `current_files = result.final_files` and surfaces per-iteration deltas in `last_iterations` field of observation.
5. Reward for the `run_ralph` action = `calibrated_reward(final_score, implicit_confidence=0.75)` minus 0.05 per wasted iteration (anti-thrash).
6. Checkpoint written to disk keyed by `run_id`; survives env restart.

This makes CodeForge's env **its own Autoresearch**: the agent can request autonomous sub-iteration and be rewarded by the result quality, not just a single submit's luck.

#### Skill-Cited Audit Trail (Pillar 4)

Every action is appended to an `AuditLedger` in the episode:

```python
@dataclass(frozen=True)
class AuditEntry:
    step_index: int                    # which env step
    action_type: str                   # query_kb / submit / run_ralph / ...
    cited_skill_ids: tuple[str, ...]   # which KB nodes were consulted
    cited_clusters: tuple[str, ...]    # which graph communities touched
    grounding_report: dict | None      # Layer A symbol check on any code change
    reward: float
    brier_penalty: float | None        # if submit
    confidence_declared: float | None  # if submit
    quality: float                     # raw quality before calibration
```

`get_audit` returns the ledger. This is the literal materialization of the invariant: every row is a `(reward, evidence, policy)` triple.

Post-episode analytics (baseline scoring, judging metrics):

- **Grounded-submit rate**: `sum(1 for e in ledger if e.action=='submit' and e.grounding.groundedness >= 0.95) / total_submits`
- **Skill-citation diversity**: unique skill_ids cited / total available
- **Cluster coverage**: unique cluster_ids visited / total clusters
- **Calibration error**: mean Brier penalty across submits (lower = better-calibrated agent)
- **Iteration efficiency**: `final_score / actions_spent`

#### Target Observation Schema (Post-Integration)

```python
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
    # new — full wiring
    last_cluster_hits: tuple[str, ...]          # from query_cluster
    last_interrogation_questions: tuple[str, ...]  # from interrogate
    last_ralph_run_id: str | None                # from run_ralph
    last_ralph_iterations: tuple[dict, ...]      # per-iter trajectory
    cumulative_audit_summary: dict               # running totals
```

#### Target Action Schema (Post-Integration)

```python
class CodeForgeAction(Action):
    action_type: Literal[
        "query_kb", "query_cluster", "interrogate",
        "run_ralph", "submit", "get_audit",
    ]
    # existing
    claim: str | None = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    files: dict[str, str] | None = None
    # new
    cluster_label: str | None = None             # query_cluster
    max_iters: int = 3                           # run_ralph
    confidence: float | None = None              # submit (triggers Brier calibration)
    target_run_id: str | None = None             # get_audit (defaults to current episode)
```

#### Implementation Order (when we build this)

1. **Graphify clustering** (~1-2h): add `groundloop/kb_indexer/cluster.py`, emit `cluster_manifest.json` alongside corpus. Extend `SearchResult` with `cluster_id`.
2. **Brier calibration** (~30m): add `confidence` field to `CodeForgeAction`; extend `grader.compute_reward` with Brier penalty; preserve round-1 "uncertain-is-ok" floor.
3. **`run_ralph` action** (~1-2h): env wraps `ralph_orchestrator.run_loop`; checkpoint via existing `save_checkpoint`; reward plumbed through calibrated formula.
4. **`interrogate` action** (~30m): thin wrapper over `groundloop.interrogator.Interrogator.generate`.
5. **`query_cluster` action** (~30m): lookup by cluster_label in manifest; return SearchResult subset.
6. **`get_audit` + AuditLedger** (~1h): accumulate entries per-step; serialize on demand; integrate `groundloop.audit_reporter.AuditReporter` for summaries.
7. **Observation schema update** (~20m): add new fields; default them when action doesn't touch them.
8. **Test coverage** (~1h): each new action gets ≥3 tests; end-to-end integration test runs all 6 actions in one episode.

Total: **~6-7 hours** to fully-wired CodeForge.

#### Why This Design Matters

Each pillar alone is table stakes:

- Autoresearch alone = fancy iteration, no epistemic honesty.
- Brier alone = calibrated but blind to code.
- Graphify alone = organized KB, no action.
- Ralph alone = loops, no grounding.

**Together they form a defensible architecture**: the agent must cite, commit, admit uncertainty, and iterate — each gated by the others. That's what round 1's EpistemicNav proved for simple factual claims; CodeForge scales the same contract to full Python codebases.

---

### Package Map

```
groundloop_env/                   # the OpenEnv env (NEW, active)
  environment.py                  # CodeForgeEnvironment(Environment)
  app.py                          # FastAPI via openenv.create_app
  tasks.py                        # 3 task definitions
  grader.py                       # reward = 0.6*sandbox + 0.4*grounding
  observation_builder.py          # CodeForgeObservation assembly
  requirements.txt
models.py                         # CodeForgeAction + CodeForgeObservation (also keeps round-1 types)
openenv.yaml                      # name: code-forge, 3 tasks, reward_range [0,1]
Dockerfile                        # python:3.11-slim, uvicorn on :7860
inference.py                      # baseline agent (HTTP client, hand-coded stub solutions)

groundloop/                       # utility primitives, all shipped
  skills_scraper/                 # #2 — KB corpus builder
  kb_indexer/                     # #3 — BM25 retriever
  python_sandbox/                 # #6 — the grader; runs ruff/mypy/pytest/imports
  lib_grounder/                   # #4 — Layer A symbol grounding
  ralph_orchestrator/             # #7 — autonomous loop (StubSynthesizer + OpenAISynthesizer)
  interrogator/                   # #5 — Socratic question generator (available, not yet wired into env)
  audit_reporter/                 # #8 — RunResult → AuditReport (available, not yet wired into env)

server/                           # round-1 EpistemicNav (UNTOUCHED for judge verification)
data/                             # round-1 claims + evidence (UNTOUCHED)
docs/superpowers/specs/*.md       # 6 design specs (skills-scraper, kb-indexer, mcp-shell [archived], python-sandbox, ralph-orchestrator, codeforge-env)
docs/superpowers/plans/*.md       # corresponding implementation plans
```

### Review Workflow (per sub-project slice)

Critic and verifier spawned as a **team via `TeamCreate`** — never via plain `Agent`. Rules:

1. **One team per sub-project.** Cleanup after convergence.
2. **Fresh critic per round:** new teammate (`spec-reviewer-1`, `spec-reviewer-2`, `quality-reviewer-1`, ...) every round. Never reused — stale by definition.
3. **Persistent implementer:** one teammate named `implementer`, reused across fix rounds via `SendMessage`.
4. **Loop:** spec-review → fixes → quality-review → fixes → fresh critic again → until `NOTHING_FURTHER` + all acceptance criteria pass.
5. **No self-critique from the orchestrator.**
6. **Shutdown:** call `TeamDelete` directly when convergence hit (no need for handshake dance — the `shutdown_request` message is optional, `TeamDelete` is authoritative).

### Development Guidelines (authoritative for CodeForge)

- **OpenEnv Spec Strictness:** every env endpoint conforms to `openenv.core.env_server.http_server.create_app(factory, Action, Observation)`. `/reset`, `/step`, `/state` are SDK-wired; `/tasks` + `/` are custom.
- **Anti-Reward Hacking:** reward in `[0.0, 1.0]`. `query_kb` returns 0.0. No negative rewards.
- **Determinism:** same files → same reward. Sandbox, grounder, tokenizer all deterministic.
- **Sandbox isolation:** `groundloop/python_sandbox/` uses `subprocess.run` with `shell=False`, path-traversal guard on `files` dict (`is_relative_to(tmp_root_resolved)`), `tempfile.mkdtemp` + `shutil.rmtree` in finally.
- **Python-only generated code.** Shrinks hallucination surface; makes grounding verifiable.
- **Containerization:** `python:3.11-slim`, binds `0.0.0.0:7860`. Corpus can be baked via `ENV GROUNDLOOP_CORPUS_PATH` or built at runtime.

### Common Commands (CodeForge)

- **Local server:** `uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860`
- **Baseline agent:** `python3 inference.py` (starts local server first, or set `API_BASE_URL`)
- **Docker:** `docker build -t code-forge . && docker run -p 7860:7860 code-forge`
- **Build KB corpus:** `python3 -m groundloop.skills_scraper`
- **Query KB from CLI:** `python3 -m groundloop.kb_indexer search "pytest fixtures" --top-k 3`
- **Deploy to HF Space:** `git push space main` (requires `space` remote + HF write token)

### What's Shipped vs Deferred

| Component | Shipped? | Wired into env? | Phase |
|---|---|---|---|
| `groundloop_env/` OpenEnv env | ✅ | ✅ | 1 |
| `skills_scraper` (KB Layer B) | ✅ | ✅ via `kb_indexer` | 1 |
| `kb_indexer` BM25 search | ✅ | ✅ via `query_kb` action | 1 |
| `python_sandbox` grader | ✅ | ✅ via `submit` action | 1 |
| `lib_grounder` (Layer A AST) | ✅ | ✅ attached to `submit` reward | 1 |
| `ralph_orchestrator` | ✅ | ⏳ standalone only; wire as `run_ralph` action | **2** |
| `interrogator` | ✅ | ⏳ wire as `interrogate` action | **2** |
| `audit_reporter` | ✅ | ⏳ wire as `get_audit` + `AuditLedger` per-step | **2** |
| Graphify community clustering | ❌ | — (build in `kb_indexer/cluster.py`) | **2** |
| Brier-calibrated confidence on reward | ❌ | — (extend `grader.compute_reward`) | **2** |
| Context7 live doc federation | ❌ | — | 3 |
| Multi-language generated code | ❌ | — | 3 |
| Policy-gradient RL baseline | ❌ | — | 3 |

### Round-2 Submission Checklist

- [x] `openenv.yaml` updated to `name: code-forge`, 3 tasks, reward_range [0,1]
- [x] `groundloop_env/environment.py` implements reset/step/state
- [x] `groundloop_env/grader.py` — reward 0.6*sandbox + 0.4*grounding, clamped
- [x] `models.py` — `CodeForgeAction` + `CodeForgeObservation` Pydantic v2
- [x] Dockerfile builds and runs (verified locally)
- [x] `inference.py` in root — HTTP client, 3-task baseline
- [x] README updated with Round-2 section, tasks table, baseline scores, reward formula
- [x] 222/222 tests pass, 93% coverage on `groundloop_env/`, 91% overall
- [x] `ruff check groundloop_env/` clean
- [x] Two spec-review rounds + one quality-review round converged to PASS / NOTHING_FURTHER
- [ ] **HF Space deployed** — requires `git push space main` (needs HF token); blocked on credentials.
- [ ] `openenv validate` — not yet run; check if CLI is installed on target box.
- [ ] Bake frozen skills corpus into Docker image so the Space has a non-empty KB (commit a snapshot of `groundloop/kb/skills_corpus.jsonl`).
- [ ] **Offline hackathon brief** — unknown until event. Be ready to adapt.

---

## Archive: EpistemicNav (Round 1 — Shipped 2026-04-08)

Round 1 deliverable. OpenEnv RL environment training LLM agents to reason accurately under uncertainty, rewarding calibrated confidence via Brier score. **Submitted and cleared round 1.** `server/` + `data/` left untouched for judge reference.

### Core Components

*   **`server/environment.py`** — `EpistemicNavEnvironment(Environment)`: claims, evidence gathering via BM25, budget tracking, step/reset/state loop.
*   **`server/app.py`** — FastAPI app via `openenv.create_app()` exposing `/step`, `/reset`, `/state` on `0.0.0.0:7860`.
*   **`server/grader.py`** — Brier score reward. Penalises overconfidence on wrong and underconfidence on right. `verdict="uncertain"` on genuinely uncertain claims → min 0.70 reward.
*   **`server/retriever.py`** — BM25Okapi wrapper over `data/evidence.json`. Pure Python, <10ms/query, no GPU.
*   **`models.py` (round-1 portion)** — Pydantic v2: `EpistemicAction` (QUERY/COMMIT), `EpistemicObservation`, `EvidenceSnippet`. Coexists with new `CodeForgeAction`/`CodeForgeObservation`.
*   **`client.py`** — `EpistemicEnv(GenericEnvClient)`: HTTP client.
*   **Round-1 `inference.py`** — replaced by CodeForge's version; baseline logic archived in git history.

### Round-1 Data

*   **`data/claims.json`** — 400 claims (200 easy, 150 medium, 50 hard).
*   **`data/evidence.json`** — 2000 snippets across 15+ domains.

### Round-1 Tasks

1. **easy (single_hop):** Single-hop factual claim. Reward ceiling ~0.98.
2. **medium (multi_hop):** Multi-hop, 3–4 evidence pieces. Reward ceiling ~0.88.
3. **hard (contradictory):** Contradictory evidence. Correct answer is `"uncertain"`. Reward floor 0.70.

### Round-1 Active Env

`server/` is code-complete but only one env is exposed via `openenv.yaml`. The active YAML points to `code-forge` (round 2). To temporarily re-expose round 1, swap `openenv.yaml` to the archived contents in git history `d252064:openenv.yaml`, or use a parallel YAML (not done).

---

## Skill Usage Guide

Invoke via `/skill-name`. Applies to both CodeForge (active) and any EpistemicNav maintenance work.

### Phase 1: Planning

| Skill | When to Use |
|---|---|
| `/superpowers:brainstorming` | Before any creative/architectural decision. |
| `/superpowers:writing-plans` | After brainstorming converges — lock the multi-step implementation spec before code. |
| `/plan` | Before each implementation chunk — step-by-step plan, risks identified. |

### Phase 2: Research & Knowledge Base

| Skill | When to Use |
|---|---|
| `/deep-research` | Research OpenEnv patterns, Autoresearch internals, existing grounded-codegen tools. |
| `/exa-search` | Broader web research when GitHub/docs insufficient. |
| `/graphify` | Convert research + docs into a knowledge graph; optional advanced Layer B enhancement. |

### Phase 3: Python Development

| Skill | When to Use |
|---|---|
| `/python-patterns` | Idiomatic Python — type hints, Pydantic v2, dataclasses. |
| `/coding-standards` | Code quality, naming, file organization. |
| `/backend-patterns` | FastAPI endpoint design, API response format. |
| `/tdd` | Before implementing any new feature. RED → GREEN → IMPROVE. ≥85% coverage target. |
| `/python-testing` | pytest strategies, fixtures, mocking, parametrization. |
| `/superpowers:test-driven-development` | Before writing implementation code. |

### Phase 4: OpenEnv-Specific

| Skill | When to Use |
|---|---|
| `/plugin:context7:context7__query-docs` | Optional live library-doc grounding (future Phase 2 Layer A enhancement). |
| `/pinecone:quickstart` | Optional vector-KB backend (BM25 is the default, sufficient for Layer B). |

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
| `/build-fix` | When build or env startup fails. |
| `/superpowers:verification-before-completion` | Before claiming any task done. |
| `/superpowers:systematic-debugging` | Bugs, test failures, unexpected behavior. |
| `/verify` | Full verification loop before shipping. |

### Phase 7: Documentation & Finish

| Skill | When to Use |
|---|---|
| `/update-docs` | Update README with env description, action/obs spaces, baseline scores. |
| `/superpowers:finishing-a-development-branch` | When implementation complete and tests pass. |
| `/superpowers:requesting-code-review` | Before final submission. |
| `/huggingface-skills:hf-cli` | HF Space push + token setup. |

### Utility Skills

| Skill | When to Use |
|---|---|
| `/aside` | Quick side question without losing context. |
| `/superpowers:dispatching-parallel-agents` | 2+ independent tasks. |
| `/save-session` | Save state before ending work. |
| `/resume-session` | Start of new session — load prior context. |

### Skills NOT Relevant

- Vercel / other cloud deployment platforms (target is HF Space).
- Frontend/React/Next.js/Figma (no UI).
- Django / Spring Boot / Go / Kotlin / Java (Python-only).
- Heavy database skills (no DB — flat JSONL + BM25).

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
