# CodeForge vs OpenEnv Hackathon Guide — Critical Alignment Analysis

**Date:** 2026-04-22
**Scope:** `docs/hackthondocs/[External] Apr '26 OpenEnv Hackathon Themes.pdf` + judging rubric therein (participant help guide PDF was password-locked; inferred gaps from themes + `problem_statement_and_guidelines.md`).
**Method:** per-axis numeric scoring, gap severity × effort × ROI, iteration-cost projection.
**Subject:** `CODEFORGE/` (RL env, 4080 LOC core + 429 tests) + `benckmark-codeforge/` (MBPP harness, 974 samples).

---

## 1. Theme Alignment Matrix (0–10 per theme)

| Theme | Expected Outcome | CodeForge Match | Score | Why |
|---|---|---|---|---|
| T1 Multi-Agent | cooperation / negotiation / theory-of-mind | single-agent vs environment; only secondary "actor" is Socratic interrogator | **1/10** | no peers, no incentives, no partial observability of another mind |
| T2 Long-Horizon | 300-scattered-instructions / codebase refactor / sparse rewards | has sparse rewards + Ralph decomposition + audit memory | **5/10** | shape aligns, scale does not — budget ≤ 10, task brief two sentences, `greet(name)` is not long-horizon |
| **T3.1 Professional Tasks** | real tool/API interaction, causal world model, multi-step workflow | ruff + mypy + pytest + importlib, KB2 code graph, session isolation, 6-action surface | **8/10** | **best fit.** "env is judge" ≡ "model does real hard work instead of shortcuts" verbatim |
| T3.2 Personalized | personal assistant, emails, dinner conflicts | none | **1/10** | orthogonal |
| T4 Self-Improvement | curriculum gen, self-challenge, adaptive difficulty | Ralph keep-if-better = shallow self-play only | **3/10** | no task generator, no difficulty escalator, 3 fixed tasks |
| T5 Wild Card | novel + value-add | "env as judge, no hallucination" framing + Brier calibration | **6/10** | pitchable wildcard fallback |

**Primary pitch target = T3.1 Professional Tasks. Secondary = T2 Long-Horizon.**

### Sub-theme bonus scan (none close enough to chase)

| Sub-theme | Sponsor | Fit | Notes |
|---|---|---|---|
| Scalable Oversight | Fleet AI | 3/10 | audit ledger monitors the agent's own actions, not other agents |
| Multi-Actor Environments | Halluminate | 2/10 | interrogator is the only second actor, not managed multiples |
| Enterprise non-code workflows | Scale AI | 0/10 | CodeForge is code-only |
| Capped/uncapped token-scaled rewards | Mercor | 4/10 | Brier already caps at 1.0, budget caps actions; partial fit |
| Multi-App enterprise RL | Scaler AI Labs | 1/10 | single-app (code sandbox) |
| Consumer schema drift | Patronus AI | 1/10 | no drift in schemas |
| Simulated SMEs | Snorkel AI | 3/10 | interrogator asks Socratic Qs from a corpus, not changing preferences |

---

## 2. Judging Rubric — Current State (100 pts)

| Criterion | Weight | Current pts | Evidence |
|---|---|---|---|
| Environment Innovation | 40 | **30** | Brier + AST grounder + citation shaping + "env judges" invariant is genuinely novel. Downside: `greet(name)` is trivial — judges will see toy tasks first. MBPP baseline 45–48% proves env is non-saturated. |
| Storytelling | 30 | **18** | `README.md` + 2,040-line `SYSTEM_DESIGN.md` = strong written story. No video, no mini-blog, no 3-min pitch deck built. Two demo scripts exist (`demo_without_mcp.py`, `demo_agent_with_mcp.py`) but no narrated walkthrough. |
| Showing Improvement in Rewards | 20 | **4** | **Largest hole.** Benchmark shows inference pass@1, not training reward curve. RL "lift" (+3 pp, 42→45) is test-time refinement, not trained improvement. Judges want "before weights / after weights" curve. None exist. |
| Reward + Training Pipeline | 10 | **5** | Reward math itself is 5/5 (calibrated, floored, composite). Training pipeline side = **0/5** (no Unsloth, no TRL, no GRPO/PPO script, no Colab notebook anywhere in repo — `grep unsloth\|trl\|GRPO\|PPO` returns 0 code files). |
| **TOTAL** | **100** | **57** | |

---

## 3. Minimum-Requirements Gate (any FAIL = disqualification)

| Minimum | Status | Evidence |
|---|---|---|
| OpenEnv latest release | **PASS** | `openenv.yaml` declares env/action/observation classes, FastAPI `/reset` `/step` `/state` live in `CODEFORGE/codeforge/app.py` |
| Minimal training script (Unsloth or TRL Colab) | **FAIL** | 0 hits for `unsloth`, `from trl`, `GRPOTrainer`, `PPOTrainer` across `*.py`; 0 `.ipynb` files in repo |
| Mini-blog (HF) or <2 min video | **FAIL** | nothing in repo, nothing referenced |
| HF Space hosted | **FAIL** | `CLAUDE.md` §M9 checklist shows `[ ] HF Space deployed` still unchecked; Docker image builds locally but not pushed to `huggingface.co/spaces/krrishchoudhary109/code-forge` |

**3 of 4 minimums FAIL ⇒ current state = disqualified regardless of score.** Fix these before anything else.

---

## 4. Gap Severity × Effort × ROI

| Gap | Severity | Effort | ΔScore possible | Priority |
|---|---|---|---|---|
| TRL/Unsloth GRPO Colab script | FATAL (min req) | 1 day | +0 rubric, eligibility unlock | P0 |
| HF Space deploy | FATAL (min req) | 4 h | +0 rubric, eligibility unlock | P0 |
| Mini-blog or 2-min video | FATAL (min req) | 3 h | +5 (storytelling) | P0 |
| Reward curve during training (train-eval plot) | HIGH | 1 day after P0-1 | +12 (improvement-in-rewards box) | P1 |
| Replace `greet(name)` with non-trivial tasks | HIGH | 2 days | +6 (innovation box) | P1 |
| Explicit long-horizon variant (multi-file, 10+ subtasks) | MED | 3 days | +4 (theme fit → T2 visibility) | P2 |
| Pitch deck + narrative | MED | 1 day | +4 (storytelling) | P2 |

---

## 5. Multi-Iteration Score Projection

Baseline = 57/100 but **DQ via min-req fail**. Convert to eligible first.

| Iter | Deliverable | Effort | Cumulative score | Eligible? |
|---|---|---|---|---|
| 0 | current repo | — | 57 | **NO** (3 min-req fails) |
| 1 | Unsloth GRPO Colab on CodeForge + HF Space push + 2-min Loom | 2 days | 62 | **YES** |
| 2 | training reward curve (100 steps, 1.5B qwen, GRPO) plotted | 1 day | 74 | YES |
| 3 | realistic tasks (replace `greet` with 5 MBPP-style sourced from corpus) | 2 days | 80 | YES |
| 4 | explicit T2 long-horizon variant (10-subtask refactor task) | 3 days | 84 | YES |
| 5 | polished pitch deck + rehearsed 3-min demo | 1 day | 88 | YES |

**Best realistic terminal score ≈ 85–90/100 in ~9 working days.** Diminishing returns past iter 5.

---

## 6. Critical Mis-Alignment: Benchmark Work vs Hackathon

Current 974-sample MBPP benchmark is **off the critical path**.

- Benchmark measures: `pass@1(model + MCP)` — static inference on an external dataset.
- Hackathon measures: "did your ENV make the AGENT better?" — reward-curve-under-training.
- Delta: you are proving model-picker, judges want environment-as-trainer.

The MBPP work is valuable but speaks to theme 3.1 indirectly. Convert by:

1. Keep 200-sample MBPP as **held-out eval**.
2. Use CodeForge's 3 tasks (plus new ones) as **training distribution**.
3. Run GRPO on the training distribution, show reward curve.
4. Report pre-vs-post MBPP pass@1 as the "agent got better" proof.

That single reframe moves the 20% improvement-in-rewards box from **4 → 17**.

---

## 7. Theme Fit Deep-Dive — Why T3.1 Beats T2

| Signal | T2 Long-Horizon | T3.1 Professional |
|---|---|---|
| Example env list mentions codebase refactor | yes | no |
| Example env list mentions "scientific workflow loops (papers → code → experiments)" | no | **yes** — maps directly to corpus → code → sandbox |
| Example env list mentions "tool-discovery benchmarks" | no | **yes** |
| Expected outcome | "beyond context memory limits" | "capturing nuances of partially observable world" |
| CodeForge reality check | budget cap 10, two-sentence brief | partial obs (hidden tests), 2,648-node corpus as world, tool orchestration |

T3.1 is a truer fit. Pitch as T3.1 with T2 as backup framing.

---

## 8. Honest Roasts

1. **"Long-horizon" claim is aspirational.** Budget = 10 and a one-function brief is not long-horizon. Fix by adding a task with ≥ 50 budget and 5+ files.
2. **Innovation is real but hidden behind triviality.** A judge seeing `greet(name)` will pattern-match "toy" before reading the Brier math. Lead the pitch with MBPP lift + training curve, not `greet`.
3. **Zero RL training code exists.** The "RL environment" is currently an "RL-shaped evaluation environment." Single biggest credibility gap.
4. **Audit ledger unused in pitch.** Real differentiator (traceability, Fleet AI Scalable Oversight bonus candidate) but not surfaced anywhere.
5. **Self-play (T4) is close but not wired.** Ralph already keeps-if-better. Add a task generator that mutates MBPP problems → near-free T4 bonus consideration.

---

## 9. Recommended Execution Order (next 9 working days)

```
Day 1-2:  Unsloth GRPO Colab + HF Space deploy + 2-min Loom   [unlock eligibility]
Day 3:    Reward curve plot over 100 training steps            [+12]
Day 4-5:  Replace greet tasks with 5 non-trivial tasks         [+6]
Day 6-8:  Long-horizon task variant (10-subtask refactor)      [+4]
Day 9:    Pitch deck + rehearsal                               [+4]
```

---

## 10. Evidence Appendix (files referenced)

| File | Role | Key fact |
|---|---|---|
| `CODEFORGE/openenv.yaml` | env manifest | 3 tasks (easy/medium/hard), 6 actions, port 7860 |
| `CODEFORGE/README.md` | external story | 170 lines, covers architecture + 10 cheat-prevention mappings |
| `CODEFORGE/SYSTEM_DESIGN.md` | internal spec | 2,040 lines, authoritative |
| `CODEFORGE/codeforge/grader.py` | reward | 39 LOC, Brier + floor |
| `CODEFORGE/codeforge/tasks.py` | task defs | 137 LOC, includes hidden test suites |
| `CODEFORGE/codeforge/environment.py` | env core | 515 LOC, all 6 actions wired |
| `CODEFORGE/codeforge/mcp_server.py` | MCP surface | 848 LOC, 10 tools |
| `CODEFORGE/tests/` | coverage | 20 test files, 429 tests per README |
| `CODEFORGE/inference.py` | baseline | REST client demo, not a training loop |
| `benckmark-codeforge/results/report.md` | benchmark | MBPP 974; plain 45.6%, MCP-raw-BM25 44.9%, RL final@3 45% on 200-subset |
| `CLAUDE.md` | execution plan | §M9 checklist: `[ ] HF Space deployed` still open |
| (missing) | — | no `.ipynb`, no `unsloth`/`trl` imports, no HF Space URL live |

---

## 11. One-Line Verdict

Strong core RL env (**57/100** on merit) but **currently disqualified** on 3 of 4 hackathon minimum requirements; pitch as **Theme 3.1 Professional Tasks**; single highest-ROI unlock is a **TRL/Unsloth Colab GRPO training script** that produces an actual reward curve — without it, the 20% improvement-in-rewards bucket stays near-zero.
