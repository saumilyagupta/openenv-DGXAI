# 09 — Results

Full benchmark ran on 2026-04-22 against **qwen2.5-coder:1.5b** (Q4_K_M, 986 MB) hosted on Ollama at `http://172.22.2.151:7021`.

**Authoritative report:** `benckmark-codeforge/results/report.md`
**Dataset SHA256:** `616cbea1e7b2123d6889382cccacb6ba29481fd873d2e716fddcaf1e7f3e20de`

---

## Headline — all three experiments

### A. 974-sample baselines (one-shot)

| Mode | pass@1 | Passed / N | Mean latency |
|---|---|---|---|
| **without_mcp** | **0.4559** | 444 / 974 | 11.4 s |
| with_mcp (raw BM25 retrieval, legacy) | 0.4487 | 437 / 974 | 22.8 s |

McNemar χ² = 0.39, p = 0.534 → not significant. RAG over CodeForge skills corpus did not move the needle on MBPP.

### B. Real CodeForgeMCPServer partial (53 samples, one-shot)

| Mode | pass@1 |
|---|---|
| with_mcp (real MCP server: `reset` + `query_kb` + `interrogate`) | **0.5472** (29/53) |

Partial run — stopped at 53. Sample is non-random (first-IDs-first).

### C. RL agent (200 samples, max_iters=3)

Full `demo_agent_with_mcp.py` loop: `reset → query_kb → interrogate → gen → sandbox → on-fail re-query + refine → repeat`.

| Metric | Value |
|---|---|
| first-try pass@1 (iter 0) | 0.4200 (84/200) |
| final pass@k (3 iters) | 0.4500 (90/200) |
| Refinement lift | **+3.00 pp** |
| Mean iters to pass | 1.09 |
| Mean best quality (frac asserts) | 0.52 |
| Mean best reward (Brier, conf=0.7) | 0.48 |

### Head-to-head on same 200 task IDs

| Arm | pass |
|---|---|
| plain qwen (no MCP, one-shot) | 95/200 = **47.5%** |
| qwen + MCP (first try, RL iter 0) | 84/200 = 42.0% |
| qwen + MCP + refinement (RL 3 iters) | 90/200 = 45.0% |

MCP hurts first-try by 5.5 pp. Refinement claws back 3 pp. Plain qwen still wins.

---

## Per-iteration pass counts (RL)

| Iter | Passed this iter | Cumulative |
|---|---|---|
| 0 | 84 | 84 |
| 1 | 4 | 88 |
| 2 | 2 | 90 |

Gains plateau after iter 1.

## Failure taxonomy (RL, per iter)

| Reason | iter 0 | iter 1 | iter 2 |
|---|---|---|---|
| pass            | 84 | 4 | 2 |
| assertion_error | 96 | 95 | 93 |
| name_error      | 12 | 11 | 11 |
| runtime_error   | 7  | 5  | 5  |
| timeout         | 1  | 1  | 1  |

`assertion_error` dominates at every iter — code runs but returns wrong output. Error-feedback prompting does not fix semantic understanding at 1.5B scale.

## Reward model

Per `SYSTEM_DESIGN.md §4.8.1`:
```
quality   = fraction of hidden asserts passed
confidence = 0.7 (fixed)
brier     = min((confidence - quality)^2, 0.5)
reward    = quality * (1 - brier)
```

## Observations

1. **MCP context hurt absolute accuracy** on small model (−5.5 pp vs plain qwen).
2. **Corpus misalignment** is the root cause — CodeForge skills corpus is Claude-Code workflow material, MBPP is basic Python algorithms. Citations distract rather than guide.
3. **Interrogator off-topic** — generates questions against CodeForge's fixed `multi_file_module` task, not the MBPP problem at hand.
4. **Refinement plateau** — model rotates between equally-wrong implementations instead of converging. Small model can't integrate error traces into semantic correction reliably.
5. **Latency tax** — MCP adds 2× wall time (11→23 s) without accuracy return on this setup.

## What would actually move the number

1. Python-algorithm corpus (RosettaCode, stdlib docs, algorithm cookbook) instead of CodeForge Claude-skills corpus
2. Larger base model (7B+) with capacity to integrate retrieved context
3. Task-aware interrogator grounded in MBPP problem, not CodeForge fixed task
4. SYSTEM_DESIGN §4.8.4 retroactive shaping rewards to promote citations that actually appear in submitted code
5. Model-declared confidence per iteration for proper Brier calibration

## Artifacts

- `benckmark-codeforge/results/report.md`
- `benckmark-codeforge/results/metrics_without_mcp.json`
- `benckmark-codeforge/results/metrics_with_mcp.json` (legacy)
- `benckmark-codeforge/results/metrics_rl.json`
- `benckmark-codeforge/results/comparison.csv`
- `benckmark-codeforge/results/raw/{without_mcp, with_mcp, rl}/{task_id}.json`
- `benckmark-codeforge/results/logs/run.log`, `rl_run.log`
