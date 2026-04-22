# benckmark-codeforge Config vs CODEFORGE — Critical Analysis

**Analyst:** Principal Research Scientist & Lead Systems Architect
**Date:** 2026-04-22
**Scope:** Verify whether `benckmark-codeforge/config.yaml` (+ its harness code) is configured to extract the **best, substantively valid** signal from the `CODEFORGE/` environment on MBPP.
**Verdict (TL;DR):** **NO.** The harness runs, but the configuration under-utilizes CodeForge by ~60% of its intended signal, mismatches task semantics, and the published MBPP results (`metrics_rl.json`, `metrics_with_mcp.json`) are **not statistically valid** (n=10 out of 974). Several fixes are P0/P1 blockers before any comparison is publishable.

---

## 1. System Snapshot

| Surface | What it is |
|---|---|
| `CODEFORGE/codeforge/` | RL env: 3 toy tasks (greet easy/medium/hard), budget {4,6,10}, 8 MCP tools, BM25+Jaccard KB over 2,212 nodes from ECC SKILL.md, reward = 0.6·sandbox + 0.4·grounding with Brier penalty and uncertain-floor 0.50 |
| `benckmark-codeforge/` | MBPP (974 problems) harness. Calls `qwen2.5-coder:1.5b` via Ollama at `172.22.2.151:7021`. 2 modes: `without_mcp`, `with_mcp`. Optional `rl_agent.py` with KB re-query on failure. Embeds `CodeForgeMCPServer` directly (no SSE). Local sandbox is the authoritative grader |

**Claim the harness makes:** "Benchmark CodeForge's MCP signal on MBPP." **Reality:** it benchmarks **two of eight** MCP tools (`query_kb`, `interrogate`), pipes their text into a plain prompt, and grades with its **own** pass/fail — bypassing the CodeForge reward entirely.

---

## 2. Deconstruction

### 2.1 Strengths (what is right)

1. **Resume + deterministic decoding.** `temperature=0.0`, `resume: true`, per-task JSON artifacts, content-hash digest of dataset (`dataset_sha256`). Re-runs are idempotent and auditable.
2. **Code extraction is robust.** `sandbox_runner.extract_code` handles closed fence, unclosed (truncated) fence, and raw non-fenced output via `_try_parse_prefix`. Right call for small OSS models.
3. **Stderr classifier is useful.** `assertion_error` / `name_error` / `import_error` / `runtime_error` / `timeout` → good signal for failure-mode analysis.
4. **Embedded MCP avoids transport flake.** Direct `handle_tool` calls mean the benchmark doesn't die on SSE hiccups. Trade-off: it does **not** validate the real deployment path.
5. **Thread-lock around `handle_tool`.** Correct — `CodeForgeMCPServer` mutates a shared session dict without internal synchronization (`mcp_client.py:47`).
6. **Per-assert quality in `rl_agent.count_passed_asserts`.** Partial credit is sane for RL-style reward instead of binary pass/fail.

### 2.2 Vulnerabilities (what is wrong)

| # | File:Line | Problem | Severity |
|---|---|---|---|
| V1 | `config.yaml:37` `mcp.task_level: hard` | `hard` task is a **multi-file `greet` module** (CODEFORGE/tasks.py:107). Its `brief`, `initial_files`, and `hidden_tests` are **ignored** by the MBPP harness. You only want `hard`'s budget=10, so you pay a full `reset` for a task context you discard. | P1 |
| V2 | `mcp_client.gather_context` | Creates a **new session per MBPP task** → 974 sessions. `CODEFORGE` session pool is **capped at 10** with 1h TTL (SYSTEM_DESIGN §15). Likely hits eviction/rejection silently — no error path logged. | P0 |
| V3 | `rl_agent.compute_reward` vs `CODEFORGE/codeforge/grader.py` | Benchmark reward = `quality·(1−brier)` where `quality = asserts_ok/total`. **CodeForge reward = (0.6·sandbox + 0.4·grounding)·(1−brier)`. Grounding (0.4 weight) is **never computed**. The paper/report therefore does **not** measure CodeForge's reward — only a hand-rolled proxy. | P0 |
| V4 | `rl_agent.py:270` `confidence: 0.7` (hardcoded) | Makes Brier penalty a deterministic function of quality. Uncertain floor (`conf<0.3 ∧ quality<0.5 → 0.50`) is **dead code**. Loses a key CodeForge calibration dimension. | P1 |
| V5 | `mcp_server.py:64` description says `"1006 skill nodes"` but `skills_corpus.manifest.json` shows `total_nodes: 2212` | Stale tool description. LLM is told the corpus is half its actual size — affects its query planning heuristics. | P2 |
| V6 | `config.yaml:24` `concurrency: 2` + `mcp_client._lock` | MCP is serialized via global lock → effective MCP concurrency = 1. Generation runs at 2. Cold MCP build (first task) blocks one worker. Throughput pinned by MCP, not LLM. | P2 |
| V7 | `rl_agent.py:306-313` `requery_kb_on_refine` uses `prev_stderr[:200]` as claim | Failure signature is the **last 200 chars** of stderr (a Python traceback tail = file paths + `AssertionError`). BM25 over traceback noise retrieves near-garbage. Needs a structured failure-claim builder (exception type + key identifiers). | P1 |
| V8 | `run_benchmark.py` / `rl_agent.py` — only 2/8 MCP tools used | Unused: `query_cluster`, `query_code_graph` (KB2), `get_audit`, `list_clusters`, `list_tags`, `submit`. Code-graph (KB2) is specifically designed to answer **"what does this module expose?"** — directly relevant for MBPP's stdlib-heavy problems. | P1 |
| V9 | `results/metrics_no_test.json` / `metrics_with_test.json` | n=5, model=`gemma4` (different model), `no_code_block` = 4/5 (80%). Obsolete from a prior run. Leaving them in `results/` pollutes the comparison table produced by `report_gen.py`. | P2 |
| V10 | `results/metrics_with_mcp.json` / `metrics_rl.json` | **n=10** of 974. Statistical noise: ±15% 95% CI on `pass@1=0.5`. Yet `metrics_without_mcp.json` has n=974. **Direct comparison is invalid.** | P0 |
| V11 | `config.yaml:5` `num_predict: 1024` + `with_mcp` prompt with `top_k=5, snippet_chars=400` + refine prompt carrying `prev_code + stderr[-800:] + stdout[-300:]` | qwen2.5-coder:1.5b context = 32K but attention weakens past 2-4K on a 1.5B model. Worst-case prompt ≈ 2.5-3K tokens. Quality likely dips on refine iterations. No prompt-token logging to verify. | P1 |
| V12 | `sandbox_runner.run_sandbox` — no CPU-time ulimit | `cpu_timeout_seconds: 10` in config is **not wired** to the subprocess. Only `wall_timeout_seconds=15` is used. `config.yaml:20` is dead config. | P2 |
| V13 | Hash check on resume | `resume: true` re-reads old `raw/<mode>/<tid>.json` without validating corpus SHA or model identity. Swapping model or re-scraping corpus silently reuses stale results. | P1 |
| V14 | No `code_graph` index built from MBPP problem imports | KB2 (`CODEFORGE/codeforge/kb/code_graph.py`) could answer structural questions about stdlib/third-party call patterns, but benchmark never invokes it. Same dev-cost as `query_kb`. | P2 |

### 2.3 Inefficiencies

- **Triple sandbox invocation per iter.** `run_sandbox` → then `count_passed_asserts` loops sandbox again per-assert → then (on refine) sandbox again. For MBPP's 3-assert tasks this is ~4× the needed subprocess spawn cost. Batch by running the full script once and parsing per-assert outcome from a tiny pytest-style reporter.
- **No prompt caching.** Ollama supports `cache_prompt`-style reuse for shared prefixes (model, instructions, KB citations). All `with_mcp` prompts share `_INSTR + _FMT + citations`. You re-tokenize every call.
- **Citations re-queried identically per task.** `initial_citations` is computed from `task["text"]`; for the >80% of MBPP tasks that are "sort a list", "sum two numbers", "find the nth X" — the top-k returned from 2,212-node ECC SKILL.md corpus is **near-random** (ECC has almost no MBPP-style algorithms content). Expensive miss.

---

## 3. SOTA Benchmarking

| Dimension | This harness | SOTA / standard practice (2024-26) | Gap |
|---|---|---|---|
| MBPP evaluation protocol | pass@1, greedy, n=974 (without_mcp) / n=10 (others) | pass@1 and pass@10 with `T=0.2-0.8` sampling, full 974 on every mode (Austin et al. 2021; OpenAI HumanEval/MBPP+); EvalPlus uses sanitized+augmented tests (Liu et al. 2023) | Missing: pass@k, MBPP+ augmented tests, sampling-based CI |
| RAG for code | Flat BM25 over 2,212 skill SKILL.md sections, top_k=5, 400-char snippets | Hybrid dense+lexical (BM25+bge-code or SPLADE), rerank with cross-encoder, CodeRAG-Bench (Wang 2024), Self-RAG (Asai 2023) | ~2 generations behind |
| Iterative refinement | 3-5 fixed iters, stderr-based re-query | Self-Debug (Chen 2023) with self-critique, Reflexion (Shinn 2023) with verbal learning traces, LDB (Zhong 2024) line-by-line debugging | No self-critique, no verbal memory across tasks |
| Reward model | `quality·(1−brier)` with hardcoded `conf=0.7` | Process reward models (PRM, Lightman 2023), outcome reward + unit-test shaping (AlphaCode-2, CodeRL) | Missing: stepwise PRM, no confidence elicitation from the model itself |
| MCP usage | 2/8 tools, direct Python embedding | Real MCP over SSE/stdio with schema validation (MCP spec 2024-11), multi-tool plans (ReAct, Toolformer) | Bypasses the actual MCP protocol the env was built to expose |
| Grounding signal | Not used | AST-level grounding (CodeForge's own `grounder.py`), symbol-level attribution | **The env's own grounding signal is ignored.** |
| Calibration | Brier with fixed confidence | Model-elicited confidence (verbalized / logit-based), ECE + Brier (Kadavath 2022) | Misses the calibration benchmark CodeForge is explicitly designed to support |
| Baseline strength | Single 1.5B model | Also report: non-CoT base, greedy CoT, GPT-4o / Claude 3.5 reference ceiling | No ceiling; no ablation |

**Where it leads:** the **embedded MCP + BM25 corpus + per-task JSON artifacts + stderr-classifier** stack is cleaner than most public MBPP harnesses. If the holes below are fixed, this is publishable scaffolding.

---

## 4. Hypotheses & Stress Tests

| H | Hypothesis | Test walk-through | Verdict |
|---|---|---|---|
| H1 | Session-cap OOM: after ~10-20 MBPP tasks, `codeforge_reset` either fails or silently evicts, and `gather_context` returns empty citations for downstream tasks. | `CodeForgeMCPServer` (SYSTEM_DESIGN §15) enforces `MAX_SESSIONS=10, TTL=1h`. Benchmark creates 1 session/task, never closes. At task 11, the pool evicts LRU. `gather_context` catches any `Exception` and returns `MCPContext(citations=[], questions=[], error=...)`. The harness logs `err=...` but **does not abort**. → With `n=974` and no-error check, citations silently zero out after task ≥ 11. **Confirmed failure.** | Confirmed |
| H2 | KB retrieval on ECC corpus is near-random for MBPP queries because ECC is a **process/skills** corpus (auth, deploy, security, design) not an **algorithms** corpus. | Corpus manifest shows source glob = `CODEFORGE/everything-claude-code/skills/*/SKILL.md` (all 183 skills). Spot queries ("write function that sums a list", "reverse a string") against the corpus via `kb_retriever` will return `python-testing`, `python-patterns`, `coding-standards` — generic hygiene, not MBPP-relevant algorithms. Hence `with_mcp` showing pass@1=0.50 vs `without_mcp`=0.4559 on **different n** is not a real MCP uplift — it's noise. | Likely failure (confirm by running with_mcp on full 974) |
| H3 | Refine-prompt regression: on iter ≥ 2, prompt size pushes past attention sweet-spot for qwen2.5-coder:1.5b, and second-attempt quality is ≤ first-attempt quality for most tasks. | `build_refine_prompt` = instr + citations (~1.5K chars) + problem + sample_test + prev_code + stderr[-800:] + stdout[-300:]. At ~2.5-3K tokens on a 1.5B model, second-pass correctness should degrade relative to a fresh re-roll. `metrics_rl.json` shows `iter_distribution: {"3": 5, "1": 5}` and `mean_iters_to_pass=1.0` — i.e., no task was rescued by refinement in the n=10 sample. | Likely failure at scale |
| H4 | The `hard` task_level wastes budget. Using `easy` (budget=4) would yield identical KB retrieval signal at 60% the MCP overhead and free up budget for `query_code_graph`. | `codeforge_reset` with `task_level=easy` returns budget=4; benchmark uses 2 units (query_kb + interrogate) per task, leaving 2 unused regardless. Switching to `easy` cuts nothing of substance and frees the `hard`-only budget headroom for an extra KB2 call. | Confirmed (design-level) |
| H5 | Obsolete `metrics_no_test.json` / `metrics_with_test.json` poison `comparison.csv`. | `report_gen.py` globs `metrics_*.json`. Two stale runs (gemma4, n=5) join 1 real run (qwen, n=974). Any downstream table mixing these is silently wrong. | Confirmed |

---

## 5. Strategic Update Plan

### P0 — Correctness blockers

#### P0.1 — Run `with_mcp` and `rl_agent` on the **full 974** MBPP set

- **Why:** V10. Current comparisons are statistically meaningless (n=10).
- **How:**
  ```bash
  rm benckmark-codeforge/results/metrics_with_mcp.json benckmark-codeforge/results/metrics_rl.json
  # fresh sessions, full run
  python scripts/run_benchmark.py --config config.yaml --modes with_mcp
  python scripts/rl_agent.py --config config.yaml
  ```
- **Expected impact:** replaces n=10 noise with publishable n=974. Budget: ~3-4 hrs at concurrency=2, ~12 MBPP/min.
- **Dependency:** P0.2 first (else sessions silently break after task 11).

#### P0.2 — Close CodeForge sessions after each MBPP task

- **Why:** V2 / H1. Sessions leak; env pool evicts; citations silently go to zero.
- **How:** Add session-release path to `CodeForgeMCPServer` (new tool `codeforge_close` **or** direct `_sessions.pop(sid)` via an accessor). Call at the end of `mcp_client.gather_context`.
  ```python
  # mcp_client.py, after interrogate:
  with self._lock:
      server.release_session(sid)   # add this method
  ```
- **Expected impact:** unlocks full-974 `with_mcp` run; eliminates silent error-path MCPContext returns.

#### P0.3 — Either compute CodeForge's real reward, or re-label the metric

- **Why:** V3. The published numbers are **not CodeForge reward** — they're per-assert pass rate.
- **How (option A, faithful):** import `from codeforge.grader import compute_reward` and `from codeforge.grounder import compute_groundedness`. Feed `sandbox_score = asserts_ok/total` and `groundedness = compute_groundedness(ext.code)`. Replace the ad-hoc `compute_reward` in `rl_agent.py`.
- **How (option B, honest):** rename `metrics_rl.json` fields `mean_best_reward` → `mean_asserts_pass_fraction_brier_adjusted` and add a disclaimer in `docs/benchmark/09-results.md`.
- **Expected impact:** either (A) an honest end-to-end CodeForge benchmark, or (B) no longer mis-labeling a proxy as "CodeForge reward." Pick A.

---

### P1 — High-impact config + harness changes

#### P1.1 — Drop `task_level: hard` → `easy`; stop ignoring the brief

- **Why:** V1 / H4. You use 2/10 budget units; `easy` gives 4, which is enough for reset + query_kb + interrogate + one extra tool.
- **How:** `config.yaml`:
  ```yaml
  mcp:
    task_level: easy   # was hard
    top_k: 5
  ```
- **Expected impact:** identical retrieval, cleaner semantics, budget available for P1.4 (`query_code_graph`).

#### P1.2 — Structured failure-claim builder for `requery_kb_on_refine`

- **Why:** V7. Current claim = `problem + " | failure: " + stderr[-200:]`; BM25 over traceback noise is near-random.
- **How:** Parse `stderr` for exception class + symbol; build claim like `"{problem} | {ExceptionClass} on {symbol_hint}"`. Strip file paths, temp dirs, line numbers.
  ```python
  def failure_claim(problem: str, stderr: str) -> str:
      exc = re.search(r"^\s*([A-Z][A-Za-z]+Error):\s*(.+?)$", stderr, re.M)
      if not exc: return problem
      return f"{problem} | {exc.group(1)}: {exc.group(2)[:80]}"
  ```
- **Expected impact:** refinement citations become signal not noise. Likely recovers 3-8% of currently-stuck `assertion_error` tasks (435/974 = 44.7% of failures).

#### P1.3 — Add pass@k with sampling

- **Why:** SOTA table. Single-sample `T=0.0` is a lower bound and doesn't reflect realistic serving.
- **How:**
  ```yaml
  model:
    temperature: 0.2     # new: sampling run
    num_predict: 1024
  runner:
    samples_per_task: 10  # new
  ```
  In `run_benchmark.py` add a `pass_at_k` summarizer (Chen 2021 estimator).
- **Expected impact:** `pass@10` typically > `pass@1` by 15-25% on MBPP for 1.5B coders. Removes unfair comparison vs SOTA tables.

#### P1.4 — Wire `query_code_graph` (KB2) for MBPP

- **Why:** V8. KB2 answers structural questions deterministically without an LLM call. For MBPP's stdlib-heavy problems, "what does `itertools` expose?" is cheap and accurate.
- **How:** Add a step between `query_kb` and `interrogate` in `mcp_client.gather_context`: build a graph over the **submitted code** each iter; on refine, call `query_code_graph` with the failing symbol from P1.2's structured claim.
- **Expected impact:** +2-5% pass@1 on tasks failing via `NameError` / `ImportError` (currently 45 + ?? of 974). Low implementation cost (~40 LOC).

#### P1.5 — Add corpus + model digest to resume gate

- **Why:** V13.
- **How:** record `dataset_sha256 + corpus_sha256 + model_name` in each result JSON; on resume, skip load if any mismatch.
- **Expected impact:** eliminates silent stale-result contamination across env changes.

#### P1.6 — Delete obsolete result files + regen report

- **Why:** V9.
- **How:** `rm results/metrics_no_test.json results/metrics_with_test.json results/raw/no_test results/raw/with_test`. Update `report_gen.py` to glob only `{without_mcp, with_mcp, rl}`.
- **Expected impact:** `comparison.csv` / `report.md` no longer mix gemma4 (n=5) with qwen (n=974).

---

### P2 — Improvements

| ID | What | Why | Expected impact |
|---|---|---|---|
| P2.1 | Batch sandbox: one subprocess per iter using a tiny pytest-collector (`-q --tb=line`) that reports per-assert pass/fail | V3.3 — currently ~4× subprocess spawn cost per iter | ~30% wall-time reduction on full 974 run |
| P2.2 | Ollama prompt cache on instr+citations prefix | Inefficiency §2.3 | Lower TTFT; no correctness change |
| P2.3 | Bump `concurrency` to 4 **after** P0.2 (sessions close cleanly) | V6 unblocks | ~2× throughput |
| P2.4 | Fix stale `"1006 skill nodes"` string in `mcp_server.py:64` to `"2212 skill nodes"` (or load from manifest) | V5 | Truthful tool description; minor LLM planning benefit |
| P2.5 | Wire `sandbox.cpu_timeout_seconds` via `resource.setrlimit(RLIMIT_CPU, ...)` (Linux) or drop from config | V12 — config is lying | Honest config |
| P2.6 | Log `prompt_tokens` and `completion_tokens` per task (Ollama returns `prompt_eval_count`) | V11 — diagnose refine degradation | Enables token-budget analysis |

---

### P3 — Optional / research

- **P3.1** Replace BM25 with hybrid BM25 + small code embedder (e.g. `jina-code-v2-small`); rerank top-50 → top-5.
- **P3.2** Self-critique iter (ask the model to find the bug before re-writing) — Self-Debug / Reflexion style.
- **P3.3** Add MBPP+ (EvalPlus) sanitized+augmented tests instead of/alongside raw MBPP.
- **P3.4** Run one ceiling-baseline with a frontier model (Claude 3.5 / GPT-4o) to calibrate how much of the 45.6% pass@1 is model-ceiling vs env-support.
- **P3.5** Real MCP protocol runner (SSE + schema validation) as a separate mode `with_mcp_sse` to validate production path.

---

## 6. Final Answer

**Is `benckmark-codeforge` configured for the best results on `CODEFORGE`?** No. Before any published comparison:

1. Fix session leak (P0.2) — without it, anything past task 11 silently has no citations.
2. Run `with_mcp` and `rl_agent` on all 974 tasks (P0.1) — n=10 is not a result.
3. Either use CodeForge's real reward or stop calling it CodeForge's reward (P0.3).

After those three, flip `task_level → easy` (P1.1), fix the failure-claim builder (P1.2), and run one pass@k sweep (P1.3). That gets you to a defensible benchmark.

**Until P0.1–P0.3 land, treat the `with_mcp` and `rl_agent` numbers as smoke-test only, not evidence.**

---

## Appendix A — Evidence Pointers

| Claim | File:Line |
|---|---|
| CodeForge task budgets | `CODEFORGE/codeforge/tasks.py:83,104,126` |
| CodeForge reward formula | `CODEFORGE/codeforge/grader.py:11-39` |
| Corpus size = 2212 | `CODEFORGE/codeforge/kb/skills_corpus.manifest.json:11` |
| Stale "1006 nodes" description | `CODEFORGE/codeforge/mcp_server.py:64` |
| Session cap (MAX_SESSIONS=10, TTL 1h) | SYSTEM_DESIGN §15 |
| Benchmark config | `benckmark-codeforge/config.yaml` |
| Benchmark reward divergence | `benckmark-codeforge/scripts/rl_agent.py:205-214` |
| MCP session creation per task | `benckmark-codeforge/scripts/mcp_client.py:57-90` (no release) |
| n=10 metrics | `benckmark-codeforge/results/metrics_with_mcp.json`, `metrics_rl.json` |
| n=974 metrics | `benckmark-codeforge/results/metrics_without_mcp.json` |
| Obsolete gemma4 runs | `benckmark-codeforge/results/metrics_no_test.json`, `metrics_with_test.json` |
