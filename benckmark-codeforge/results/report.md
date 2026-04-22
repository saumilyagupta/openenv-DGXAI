# benckmark-codeforge — Final Report

**Model:** `qwen2.5-coder:1.5b` (Q4_K_M, 986 MB, 4096 ctx) hosted on Ollama
**Endpoint:** `http://172.22.2.151:7021/api/generate`
**Dataset:** MBPP full config, all splits concatenated — **974 samples**
**Dataset SHA256:** `616cbea1e7b2123d6889382cccacb6ba29481fd873d2e716fddcaf1e7f3e20de`
**Run date:** 2026-04-22
**Judge:** local Python subprocess sandbox, 3 hidden asserts per task, 15s wall timeout

---

## 1. Modes compared

| Mode | N | Description |
|---|---|---|
| **without_mcp** | 974 | One-shot: `problem + sample_test` → qwen → sandbox |
| **with_mcp (one-shot, raw-BM25 v1)** | 974 | One-shot with raw BM25 retrieval over CodeForge corpus — *legacy, see §6* |
| **with_mcp (real MCP server, one-shot, partial)** | 53 | Uses `CodeForgeMCPServer` — `reset` + `query_kb` + `interrogate`; run aborted at 53 samples |
| **rl_agent (real MCP + refinement)** | 200 | Multi-iter: MCP context → gen → sandbox → on-fail re-query KB with error signature → regenerate up to `max_iters=3` |

---

## 2. Headline results

### All-974 one-shot (no refinement)

| Mode | pass@1 | Mean latency (s) | Mean eval_count |
|---|---|---|---|
| without_mcp | **0.4559** (444/974) | 11.44 | 37.4 |
| with_mcp (raw-BM25 legacy) | 0.4487 (437/974) | 22.83 | 40.9 |

Lift raw-BM25 vs baseline: **−0.72 pp**, McNemar χ²=0.39, p=0.534 (not significant).

### Real-MCP-server partial (53 samples)

| Mode | pass@1 |
|---|---|
| with_mcp (real CodeForgeMCPServer, one-shot) | **0.5472** (29/53) |

*(partial run; sample is non-random — first 53 task IDs processed)*

### RL agent on 200-sample subset (max_iters=3, real MCP + refinement)

| Metric | Value |
|---|---|
| first-try pass@1 (iter 0 only) | **0.4200** (84/200) |
| final pass@k (any iter passes) | **0.4500** (90/200) |
| Refinement lift | **+3.00 pp** (6 tasks flipped fail→pass via retry) |
| Mean iters to pass | 1.09 |
| Mean best quality (frac asserts) | 0.52 |
| Mean best reward (Brier-calibrated, conf=0.7) | 0.48 |
| Iter distribution | 84 passed at iter 0, 4 at iter 1, 2 at iter 2, 110 never passed |

### Head-to-head on same 200 task IDs

| Mode on matched 200 IDs | pass |
|---|---|
| without_mcp (one-shot, no MCP) | **95/200 = 47.5%** |
| RL with MCP — first-try | 84/200 = 42.0% |
| RL with MCP — final@k (3 iters) | 90/200 = 45.0% |

---

## 3. Reward model (RL mode)

Per `SYSTEM_DESIGN.md §4.8.1`:

```
quality     = fraction of hidden asserts passed (0..1)
confidence  = 0.7 (declared by config)
brier       = min((confidence - quality)^2, 0.5)
reward      = quality * (1 - brier)    # clamped [0, 1]
```

Average reward across 200 RL episodes: **0.476**.

## 4. Per-iteration pass counts (RL)

| Iter | Passed this iter | Cumulative |
|---|---|---|
| 0 (first try) | 84 | 84 |
| 1 (first refine) | 4 | 88 |
| 2 (second refine) | 2 | 90 |

Refinement efficacy dropped sharply: iter-1 fixes 4 tasks, iter-2 only 2 more. Typical: model rotates between equally-wrong implementations rather than converging.

## 5. Failure breakdown per iter (RL, 200 tasks)

| Reason | iter 0 | iter 1 | iter 2 |
|---|---|---|---|
| pass | 84 | 4 | 2 |
| assertion_error | 96 | 95 | 93 |
| name_error | 12 | 11 | 11 |
| runtime_error | 7 | 5 | 5 |
| timeout | 1 | 1 | 1 |

`assertion_error` dominates and barely moves across iterations: the model produces syntactically-valid code that executes but returns wrong values. Error-feedback prompting causes model to flip logic, not to reason about semantics. With a 1.5B model, error traces are consumed as surface pattern-match rather than root-cause diagnosis.

## 6. Why "with_mcp" legacy 974 result is low-confidence

The **974-sample with_mcp run** used a raw BM25 retriever (`scripts/kb_retriever.py`, now removed from the critical path) that bypassed `CodeForgeMCPServer`. That short-circuited the full MCP contract (no `reset`, no `interrogate`, no session budget). It is therefore only a measure of **naive RAG** with the CodeForge corpus — not of the MCP pipeline.

The 53-sample real-MCP partial and the 200-sample RL runs use `CodeForgeMCPServer.handle_tool(...)` directly, matching the `demo_agent_with_mcp.py` pattern: `reset` → `query_kb` → `interrogate`. Those numbers are representative.

## 7. Key observations

1. **MCP context hurts first-try accuracy on MBPP** (−5.5 pp vs. plain qwen on matched 200 IDs). The CodeForge skill corpus (Claude hooks/agents/workflow rules) is topically misaligned with MBPP algorithm tasks; injected citations act as distraction tokens for a small 1.5B model. The interrogator's Socratic questions are generated against CodeForge's fixed task (`multi_file_module`), so they are also off-topic.
2. **Refinement recovers some lost ground** (+3 pp within RL arm) but does not surpass plain single-shot. Gains plateau after iter 1.
3. **Absolute failure mode is semantic, not surface-level.** `assertion_error` is 48% of attempts at every iter. Model writes runnable Python but misunderstands problem intent. Error feedback does not fix this class of bug reliably at 1.5B scale.
4. **Corpus alignment is the lever, not pipeline complexity.** An MBPP-aligned corpus (algorithm cookbook, stdlib recipes) would likely flip MCP from distracting to useful. Larger base model would also help — qwen 1.5B lacks capacity to integrate retrieved context.
5. **Latency cost of MCP is meaningful.** Raw-BM25 one-shot mean latency was 22.8 s vs. 11.4 s for plain gen — 2× slowdown from longer prompts. Real MCP adds ~5–10 s of `reset + query_kb + interrogate` on top.

## 8. Cost

| Mode | Gens | Wall time | Mean latency |
|---|---|---|---|
| without_mcp (974) | 974 | ~3h20m | 11.4 s |
| with_mcp raw-BM25 (974) | 974 | ~6h | 22.8 s |
| with_mcp real (53 partial) | 53 | ~15 min | — |
| rl_agent (200, avg 1.58 gens/task) | ~316 | ~65 min | — |

## 9. Artifacts

- Per-task traces: `benckmark-codeforge/results/raw/{without_mcp, with_mcp, rl}/{task_id}.json`
- Aggregate: `benckmark-codeforge/results/metrics_without_mcp.json`, `metrics_with_mcp.json`, `metrics_rl.json`
- Comparison CSV: `benckmark-codeforge/results/comparison.csv`
- Run logs: `benckmark-codeforge/results/logs/run.log`, `rl_run.log`
- Live dashboard: `python scripts/monitor_server.py` → http://127.0.0.1:7870

## 10. Reproduction

```bash
cd benckmark-codeforge
pip install -r requirements.txt
# One-shot baseline (no MCP)
PYTHONPATH="../CODEFORGE" python scripts/run_benchmark.py --config config.yaml --modes without_mcp
# One-shot with real MCP server
PYTHONPATH="../CODEFORGE" python scripts/run_benchmark.py --config config.yaml --modes with_mcp
# RL agent (iterative)
PYTHONPATH="../CODEFORGE" python scripts/rl_agent.py --config config.yaml --limit 200 --max-iters 3 --concurrency 2
# Aggregate + regen report
python scripts/report_gen.py
```

## 11. Recommendations for follow-up

1. **Retrieval corpus:** replace CodeForge skill corpus with a Python-algorithm corpus (Hacker Rank writeups, RosettaCode, stdlib docs) for MBPP-like benchmarks.
2. **Scale model:** rerun with qwen2.5-coder:7B or 14B. MCP context is more likely to help at higher parameter counts where integration capacity exists.
3. **Reward shaping:** add the SYSTEM_DESIGN §4.8.4 retroactive shaping — +0.01 per prior KB query whose cited skills appear in the submitted code, capped at +0.05. Would differentiate "retrieval I actually used" from "retrieval I ignored".
4. **Calibrated confidence:** let the model declare confidence per iteration; use it in Brier penalty instead of fixed 0.7.
5. **Task-aware interrogator:** current interrogator questions are tied to CodeForge's fixed tasks. For MBPP-style eval, generate Socratic questions grounded in the MBPP problem text itself.
