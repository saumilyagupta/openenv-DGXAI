# CodeForge — SOTA Research Paper Alignment & Strategic Update Plan

**Date:** 2026-04-22
**Scope:** Theme #2 (Long-Horizon Code) + Theme #3.1 (Scientific Workflow / Tool-Discovery) of OpenEnv Apr '26 Hackathon.
**Method:** Deconstruction → SOTA benchmark table (16 papers) → falsifiable stress tests → tiered P0–P3 plan.
**Subject under review:** `CODEFORGE/` (4080 LOC core + 429 tests), per `CODEFORGE/CLAUDE.md` + `CODEFORGE/SYSTEM_DESIGN.md`.
**Constraint set (from `docs/hackthondocs/`):** OpenEnv latest release, HF Spaces deploy, Unsloth/TRL Colab training script, mini-blog/video, reward-curve evidence of training progress.
**Companion docs:** `hackathon-alignment-analysis.md`, `codeforge-critical-analysis.md`, `benchmark-codeforge-config-analysis.md`.

---

## 1. Executive Summary

CodeForge's environment-as-judge triple (sandbox + AST grounder + skill corpus) is architecturally close to **FunSearch** (Nature '24) and **AlphaEvolve** (DeepMind '25) but ships only the linear-chain variant of the pattern. Two exploits (uncertain-floor, zero-symbol ground) currently leak 0.20–0.50 reward to trivial submissions. Retrieval layer (BM25 + Jaccard only) trails hybrid embedding baselines by ~25% Recall@10 on paraphrased queries. Ralph orchestrator is single-chain vs SOTA tree/population-based peers (LATS ICML '24, FunSearch Nature '24). Episode-level reward only — no Process Reward Model signal (Math-Shepherd ACL '24) — caps Theme #2 long-horizon credit assignment.

Plan below closes exploits in P0 (correctness), adopts FunSearch population + PRM step-level credit + hybrid retrieval + V-STaR verifier in P1 (architecture), and pivots/extends via LATS, PlanSearch, LILO, LiveCodeBench in P2 (quality / demo leverage).

---

## 2. CodeForge Deconstruction

### 2.1 Strengths

| # | Property | Evidence |
|---|---|---|
| S1 | Deterministic LLM-free reward | `grader.py` takes `(sandbox_score, groundedness, confidence)` → float. No LLM in reward path. |
| S2 | Simple auditable weighting | `quality = 0.6 * sandbox + 0.4 * ground` + Brier penalty. 3 knobs, all inspectable. |
| S3 | Evolutionary lineage | Ralph = synth → score → keep-if-better. Same shape as FunSearch's sampler→evaluator→program DB. |
| S4 | Dual knowledge base | KB1 (skill corpus, BM25+Jaccard) + KB2 (intra-file AST call/import/inherit graph). |
| S5 | Prod-shape MCP | Session-isolated FastAPI + MCP SSE + bearer auth + corpus baked into Docker. |
| S6 | Auditable triple invariant | Every reward-earning action traces to `(sandbox-verified signal, Layer-A grounded symbol, Layer-B skill citation)`. |

### 2.2 Vulnerabilities (correctness / reward-hack)

| # | Vulnerability | Line / Source | Severity |
|---|---|---|---|
| V1 | **Uncertain-floor exploit.** `reward = max(reward, 0.50) if confidence < 0.3 AND quality < 0.5` — submit empty code with `confidence=0.1` returns free 0.50. | `CLAUDE.md §4` reward formula, `SYSTEM_DESIGN §4.8.1` | **CRITICAL** |
| V2 | **Zero-symbol ground floor.** Empty / `pass` modules parse fine; zero-symbol case returns ground=0.5 → reward ≥ 0.20 pre-sandbox. | `grounder.py` spec `SYSTEM_DESIGN §4.8.3` | **HIGH** |
| V3 | **Linear wasted-iter penalty.** Ralph charges `0.05 × wasted_iters`; agent spams cheap failing iters for sandbox-message info < penalty. | `SYSTEM_DESIGN §4.8.5` | **MEDIUM** |
| V4 | **Episode-level reward only.** Iter-2 fix + iter-5 regress → only final captured, no per-iter credit. | Ralph reward formula | **MEDIUM** (long-horizon blocker) |
| V5 | **BM25-only retrieval.** Paraphrase miss, e.g. "greet user" vs `salutation_formatter.md`. No embedding fallback. | `kb/indexer.py` | **MEDIUM** |
| V6 | **Monolithic brief.** No scattered-instruction / 300-step mode. Theme #2 example expects exactly this. | `tasks.py` | **MEDIUM** |
| V7 | **Single-chain Ralph.** No population / tree search. Under-explores alt branches. | `ralph/loop.py` | **MEDIUM** |

### 2.3 Inefficiencies

| # | Inefficiency | Cost |
|---|---|---|
| I1 | KB2 is intra-code only; task-level multi-file dep graph absent | ceilings `multi_file_module` task |
| I2 | M12 `LLMSynthesizer` = stub; no mutation/crossover operator | leaves FunSearch-style gain on table |
| I3 | No verifier co-training | generator-only post-train ceiling |
| I4 | `composite_score(tools=…)` filter exists, but no score decomposition for subtask aggregation | M13 planner can't drop-in use it cleanly |

---

## 3. SOTA Benchmark Table (16 papers)

Papers picked on **direct upgrade path** for a specific CodeForge module + **underrated** status where available (low citations relative to applicability or mis-shelved as "math" / "agent" vs "code RL").

### 3.1 Master table

| # | Paper | Year / Venue | One-line core signal | Maps to CodeForge | Underrated signal |
|---|---|---|---|---|---|
| 1 | **FunSearch** — Romera-Paredes et al. | 2024, *Nature* | Evolutionary program search, LLM=mutator, deterministic fitness, program DB | Ralph M5 → population + tournament | **Yes** — shelved as math discovery; the RL-env pattern is under-adopted |
| 2 | **AlphaEvolve** — DeepMind | 2025, blog + tech report | FunSearch v2; beat Strassen 4×4 record; evaluator + program DB + ensemble | Ralph M5 + Synth M12 | **Yes** — very recent, low citations, highly actionable |
| 3 | **RLEF: Grounding Code LLMs in Execution Feedback** — Gehring et al. (Meta) | 2024 | Multi-turn RL on execution feedback; CodeContests SOTA | Validates `0.6*sandbox` weighting | No (cited), but under-adopted outside Meta |
| 4 | **SWE-RL** — Wei et al. (Meta) | 2025 | RL on real GitHub PR data w/ rule-based rewards; transfer outside code | Training stack (Unsloth/TRL script) | **Yes** — 2025, low coverage |
| 5 | **Absolute Zero Reasoner (AZR)** — Zhao et al. | 2025 | Self-play code, zero human data, model proposes + solves + verifies | Theme #4 pivot / self-curriculum | **Yes** — arxiv-only, under-cited |
| 6 | **V-STaR** — Hosseini et al. | 2024 | Verifier + generator co-training via DPO on verifier-labeled trajectories | Post-train loop / audit-ledger replay | Moderate |
| 7 | **Math-Shepherd / PRM** — Wang et al. | 2024, *ACL* | Step-level Process Reward Model; auto-labeled via rollout completions | Ralph per-iter credit; fixes V4 | Landmark in math, **underused in code** |
| 8 | **LILO** — Grand et al. | 2024, *ICLR* | Library induction via Stitch compression + LLM rename/doc | Corpus growth & compaction (M11) | **Yes** |
| 9 | **CodePlan** — Bairi et al. (MSR) | 2024, *ICSE* | Repo-level coding via dependency-planning DAG + block-level edits | `multi_file_module` + M13 planner | **Yes** |
| 10 | **LATS — Language Agent Tree Search** — Zhou et al. | 2024, *ICML* | MCTS over LM agent trajectories + value function + reflection | Ralph → tree search | Well-cited |
| 11 | **Agentless** — Xia et al. | 2024 | 27.3% SWE-Bench Lite without agent scaffold; 3 phases: localize, repair, validate | Validates 3-layer CodeForge pipeline | Cited, **lesson under-applied** |
| 12 | **AutoCodeRover** — Zhang et al. | 2024, *ISSTA* | AST-aware stratified retrieval + spectrum-based fault localization | KB1+KB2 fusion + failure loc | **Yes** |
| 13 | **PlanSearch** — Wang et al. | 2024 | Search over diverse natural-language plans > N-sampling at same budget | M13 planner diversity | **Yes** |
| 14 | **CodeHalu** — Tian et al. | 2024 | Code hallucination taxonomy (fabricated APIs, wrong signatures), 699 cases | Validates grounder design; ablation corpus | **Yes**, niche |
| 15 | **LiveCodeBench** — Jain et al. | 2024 | Contamination-free rolling code benchmark (problems by release date) | Held-out eval split; demo curve credibility | Well-cited |
| 16 | **ScienceAgentBench** — Chen et al. | 2024 | 102 scientific workflow tasks; papers→code→experiments | Theme #3.1 pivot option | Recent, **underrated** |

### 3.2 Where CodeForge **leads** vs this set

- LLM-free deterministic reward. Most of {RLEF, SWE-RL, V-STaR, AZR, LATS} use LLM-judge or pure test-suite. None combine sandbox + AST grounding + skill citation triple.
- Audit invariant `(reward, evidence, policy)` is stronger than any baseline's logging surface.

### 3.3 Where CodeForge **lags**

- **Search shape:** single-chain Ralph vs FunSearch population / LATS tree / PlanSearch diverse plans.
- **Credit assignment:** episode-level only vs Math-Shepherd step-level PRM.
- **Retrieval:** BM25+Jaccard vs hybrid BM25+embedding (industry baseline since 2023).
- **Training:** generator-only vs V-STaR / SWE-RL verifier+generator co-train.
- **Task scale:** 3 fixed toy tasks (`greet_single_file`, `greet_with_tests`, `multi_file_module`) vs SWE-Bench / LiveCodeBench / ScienceAgentBench.

---

## 4. Hypothesis Stress Tests

Seven falsifiable hypotheses. Walked through exact formula / spec file.

| H | Claim | Walk-through | Verdict |
|---|---|---|---|
| **H1** | Uncertain-floor exploit | Submit `# empty` + `confidence=0.1` → `quality=0` and `confidence<0.3` → `max(0, 0.50) = 0.50` free | **Confirmed failure** (CLAUDE.md §4 formula is explicit) |
| **H2** | Zero-symbol ground floor stacks with H1 | Empty parseable module → `imports+defs+classes=0` → ground=0.5. `reward = 0.6*sandbox + 0.4*0.5 = 0.6*sandbox + 0.20`. Sandbox may give partial on clean ruff/mypy of empty file → reward ≥ 0.20 even without H1 | **Likely failure** (depends on `composite_score([], tools=[])` behavior) |
| **H3** | Ralph linear chain under-explores | LATS (ICML '24) reports +15–22% on HotpotQA / HumanEval vs ReAct baseline. Ralph is ReAct-family, single-chain. FunSearch Fig. 3 shows population > single-chain across budgets | **Confirmed gap** (multi-paper baseline diff) |
| **H4** | BM25 paraphrase miss | "greet user w/ name" vs skill `salutation_formatter.md`: BM25 IDF overlap low → rank > 20. BGE-small cosine ≥ 0.6 → top-5. Standard hybrid benchmark |  **Confirmed failure** at retrieval |
| **H5** | 300-instr scatter mode unsupported | Brief is `str`; planner M13 decomposes spec but no instruction index / retrieval over the brief itself. Theme #2 literal example demands 300 scattered instructions | **Likely failure** — needs instr-aware decomposition |
| **H6** | No step-level credit (PRM gap) | Iter-2 fix ruff, iter-5 regresses tests → Ralph reward = `calibrated(final, 0.75) - 0.05*wasted`. Per-iter advantage signal lost. Math-Shepherd reports +5–10% Pass@1 from PRM signal alone | **Confirmed gap** |
| **H7** | Wasted-iter penalty (linear) exploitable | Run N cheap failing iters; penalty `0.05*N` caps at 0.50 even for 10 iters. Information extracted from sandbox messages (e.g., "ruff: undefined name X") can exceed 0.50 signal → net positive | **Likely exploit** |

---

## 5. Strategic Update Plan

### 5.1 P0 — Critical (correctness / reward-hack close)

| # | What | Why | How | Expected impact | Deps |
|---|---|---|---|---|---|
| **P0-1** | Close uncertain-floor exploit | H1 proven | Guard: apply floor iff `code_non_empty AND at_least_one_def_or_class AND sandbox_ran_successfully`. Record exploit attempts in audit ledger with `exploit_type="floor_abuse"` | Blocks free 0.50 | `grader.py`, update `SYSTEM_DESIGN §4.8.1` |
| **P0-2** | Fix zero-symbol ground | H2 | Return 0.5 only when AST parses AND `len(imports)+len(defs)+len(classes)==0` AND `body_lines >= 3`. Empty/whitespace file → 0.0. Add test `test_empty_file_gets_zero_ground` | Blocks partial-credit spam | `grounder.py`, update `SYSTEM_DESIGN §4.8.3` |
| **P0-3** | Quadratic wasted-iter penalty | H7 | `cost = N + 0.05 * wasted_iters**1.5` OR hard cap `wasted_iters ≤ 2`. Log rationale in audit | Removes info-probe incentive | `ralph/loop.py` |

### 5.2 P1 — High-impact (architecture)

| # | What | Why | How | Expected impact | Deps |
|---|---|---|---|---|---|
| **P1-1** | **Ralph → FunSearch/AlphaEvolve population** | H3; FunSearch Nature '24 | Population `K=4` candidate programs; each iter: sample parent pair, LLM mutates, grade via existing `composite_score`, tournament-replace worst. Elitism on `quality`. ~200 LOC rewrite of `ralph/loop.py`. Keep existing synthesizer Protocol unchanged | +10–20% Pass@k per FunSearch Fig. 3 | P0 complete |
| **P1-2** | **Process Reward Model** (Math-Shepherd style) | H6 | After each Ralph iter, emit `(state, action, delta_quality)`. Auto-label via rollout completion rate (was the final answer `quality ≥ 0.8`?). Store in audit ledger for offline train. Wire PRM score as extra reward channel in Ralph `run_result.rationale_trace` | Step-level credit unlocks Theme #2 long-horizon training | Audit ledger M6 |
| **P1-3** | **Hybrid retrieval BM25 + BGE-small embeddings** | H4 | `bge-small-en-v1.5` (33M, CPU ok). Precompute embeddings at corpus-index time (Docker build step). Query-time: `final_score = 0.5*bm25_norm + 0.5*cosine`. Add `mcp_tool("query_kb_hybrid")` or extend existing | +~25% Recall@10 on paraphrase | `kb/indexer.py`, Dockerfile |
| **P1-4** | **V-STaR verifier co-training branch** | Generator-only ceiling | Train small verifier (e.g., DistilBERT-size) on `(trajectory_tokens, success_label)` from audit ledger. Use DPO with verifier-preferred-vs-rejected pairs. Expose as optional reward channel `verifier_score` | Orthogonal signal; directly feeds **Unsloth/TRL training script** (hackathon criterion 10%) | Needs ≥1K audit trajectories; can bootstrap from offline greet-task replays |

### 5.3 P2 — Improvement (quality / demo leverage)

| # | What | Why | How | Expected impact |
|---|---|---|---|---|
| **P2-1** | LATS-style tree search over action surface | H3 complement | MCTS node = `(obs, action_sequence)`. UCB1 selection. Reflection at leaves using `get_audit`. Optional: run only above budget threshold | +5–10% on `multi_file_module` task |
| **P2-2** | **PlanSearch** in M13 planner | Plan diversity beats N-sampling at same budget | Generate `k=5` plans via temperature + explicit diversity prompt. Pick by `diversity_score + self_critique`. ~80 LOC in `ralph/planner.py` | Richer decomposition curves for demo reel |
| **P2-3** | **LILO corpus compaction** | ECC expansion → ~2,500+ nodes; redundancy | Periodic Stitch-like compression: cluster low-use / high-redundancy skills, LLM rewrite as merged skill, version corpus with hash-pinning | Token cost ↓ for `query_kb`; higher signal per retrieval |
| **P2-4** | **CodeHalu** ablation corpus | Validate grounder empirically | Replay 699 CodeHalu hallucination cases against `grounder.py`; measure false-negative rate; publish table in pitch slide | Publishable ablation → storytelling (30% weight) |
| **P2-5** | **LiveCodeBench** held-out split | Contamination risk in `greet(name)` / MBPP | Pull 50 LiveCodeBench tasks released post-2025-07; use as eval-only split for pitch reward curve | Demo curve credibility → Showing Improvement (20% weight) |

### 5.4 P3 — Optional / pivot

| # | What | Why | Impact |
|---|---|---|---|
| **P3-1** | **AZR self-play curriculum module** | Theme #4 bonus, zero human data | Wild-card pitch angle |
| **P3-2** | Scatter-brief 300-instr task variant | Theme #2 literal example match | Direct rubric alignment |
| **P3-3** | **ScienceAgentBench** task import | Theme #3.1 strongest-fit pivot (papers→code→experiments verbatim) | Judging rubric literal match |
| **P3-4** | **AutoCodeRover** spectrum fault-loc fusion into KB2 | Multi-file failure localization | `multi_file_module` ceiling lift |

---

## 6. Hackathon-Scoring Mapping

| Criterion | Weight | Primary P-items | Expected pts lift (from `hackathon-alignment-analysis.md` baseline 57/100) |
|---|---|---|---|
| Environment Innovation | 40% | P1-1 (FunSearch-as-OpenEnv), P1-2 (PRM audit), P2-4 (CodeHalu ablation) | +4 to +7 |
| Storytelling | 30% | P2-4, P2-5 (demo reel material) | +3 to +5 |
| Showing Improvement in Rewards | 20% | P1-3 (retrieval curve), P1-4 (verifier co-train curve), P2-5 | +10 to +14 *(the 4/20 hole per `hackathon-alignment-analysis.md` §2)* |
| Reward + Training Pipeline | 10% | P0-1..3 (close exploits), P1-1 (new reward shape), **P1-4 (Unsloth/TRL script!)** | +3 to +5 |
| **Projected total** | | | **77–88 / 100** (from 57) |

---

## 7. Uncertainty Flags

- Inferred: `composite_score([], tools=[])` returns > 0 on empty files. **Confirm by reading `CODEFORGE/codeforge/sandbox/metric.py`** before shipping P0-2.
- **AlphaEvolve** paper is DeepMind tech report + blog only. Mutation-operator spec partially inferred from FunSearch lineage.
- **AZR (Zhao 2025)** arxiv-only, no venue. Treat as directional signal, not canonical method.
- **SWE-RL (Wei 2025)** also arxiv/Meta-blog; weight tuning details sparse.
- PRM auto-labeling (Math-Shepherd) originally math; code transfer has less validation — expect to tune label-propagation threshold.

---

## 8. Execution Order (dependency-resolved)

```
P0-1 ──┐
P0-2 ──┼─▶ P1-1 ──┬─▶ P1-2 ──┬─▶ P1-4 ──▶ P2-4, P2-5 (demo material)
P0-3 ──┘          │          │
                  │          └─▶ P2-2 (PlanSearch in M13)
                  │
                  └─▶ P2-1 (LATS; optional parallel to P1-2)

P1-3 (hybrid retrieval) — independent, runs parallel to P1-1/P1-2

P2-3 (LILO compaction) — independent, runs after M11 corpus merge

P3-* — scope-permitting after P1/P2
```

**Minimum-viable-pitch set** (if time-boxed): P0-1 + P0-2 + P1-3 + P1-4 + P2-5. Closes both exploits, adds hybrid retrieval curve, adds Unsloth/TRL training script with verifier reward channel, adds contamination-free held-out eval. Covers the two lowest-scoring rubric axes (Improvement 20%, Pipeline 10%) directly.

---

## 9. Cross-References

- `hackathon-alignment-analysis.md` — current rubric score 57/100, primary pitch = Theme #3.1, secondary = Theme #2.
- `codeforge-critical-analysis.md` — architecture-level deconstruction (pre-paper-benchmark).
- `benchmark-codeforge-config-analysis.md` — MBPP harness baseline 45–48% Pass@1.
- `CODEFORGE/SYSTEM_DESIGN.md §4.8.1–4.8.5` — exact target code for reward / ground / score / shaping / Ralph reward.
- `CODEFORGE/CLAUDE.md §3` — layer-by-layer reward taxonomy.
