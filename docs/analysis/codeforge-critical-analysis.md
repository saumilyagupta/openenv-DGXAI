# CodeForge — Critical Analysis, SOTA Benchmark, RL-Signal Critique

**Date:** 2026-04-21
**Scope:** Full CodeForge repo (`CODEFORGE/`) — architecture, exploit surface, SOTA positioning, RL training signal
**Analyst role:** Principal Research Scientist + Lead Systems Architect
**Spec reference:** `CODEFORGE/SYSTEM_DESIGN.md` (1,942 lines, authoritative)

---

## 0. Executive Summary

CodeForge is an OpenEnv-compliant RL environment that attempts to force LLM agents to write real, verified, grounded Python code by making the **environment the judge, not the LLM**. The core invariant — every reward must trace to (sandbox, grounding, citation) — is architecturally sound and correctly implemented in the grading primitives (`grader.py`, `grounder.py`, `sandbox/metric.py`).

**What is genuinely novel:**

1. AST-grounding against live Python runtime (`importlib.find_spec` + `hasattr`) to catch hallucinated APIs before test execution.
2. Brier-calibrated reward on the submit action — unique in code-gen RL envs.
3. MCP-native surface (10 tools + resources + prompts) — makes the env usable as agentic infra, not just a benchmark.
4. Audit ledger with per-step evidence triples for deterministic replay.

**What blocks production / training:**

1. **REST API loses session state every request** — `create_app(_get_or_create_env, ...)` factory returns a new environment per call. The `_sessions` pool defined in `app.py` is dead code on the REST path. Only MCP sessions actually work.
2. **Sandbox has no real isolation.** `subprocess.run(..., env=os.environ)` inside a tmpdir. No seccomp, no network namespace, no env scrubbing. Agent-submitted `main.py` executes during pytest collection and can exfiltrate `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and any other parent-process secret.
3. **Brier calibration incentive is inverted.** The uncertain floor (0.50) triggers only when `confidence is not None and < 0.3`. Omitting confidence on good code yields higher expected reward than declaring high confidence on good code. Training will teach agents to *omit* confidence.
4. **Reward code is duplicated** between `grader.compute_reward()` and `environment._handle_submit` (inline Brier math). Split-brain drift risk.
5. **Task corpus is 3 toy tasks.** Cannot train a generalist. Will overfit to `greet` after ~200 rollouts. SOTA peers have 1k–10k real tasks (SWE-bench, BigCodeBench, R2E, APPS).

The core reward architecture is ahead of SOTA in at least two dimensions (grounding, calibration). The delivery surface, sandbox hardening, and task-corpus scale lag SOTA by 1–3 orders of magnitude.

---

## 1. Phase 1 Intake — Materials Reviewed

Files read for this analysis:

| File | Purpose |
|---|---|
| `CODEFORGE/README.md` | Product intro, reward formula, anti-cheat matrix |
| `CODEFORGE/pyproject.toml` | Deps: fastapi, pydantic, rank_bm25, networkx, mcp |
| `CODEFORGE/codeforge/grader.py` | Reward function — Brier + uncertain floor |
| `CODEFORGE/codeforge/grounder.py` | AST grounding — SyntaxError→0.0, zero-symbol→0.5, full-path resolve |
| `CODEFORGE/codeforge/sandbox/metric.py` | Composite score — penalty-only |
| `CODEFORGE/codeforge/sandbox/sandbox.py` | Sandbox orchestration — tmp dir, tool loop |
| `CODEFORGE/codeforge/sandbox/runner.py` | `subprocess.run` wrapper |
| `CODEFORGE/codeforge/sandbox/tools.py` | Tool registry — ruff/mypy/pytest/imports |
| `CODEFORGE/codeforge/environment.py` | 6-action env, file validation, Ralph+submit handlers |
| `CODEFORGE/codeforge/shaping.py` | Citation-comment shaping bonus (+0.01/match, cap 0.05) |
| `CODEFORGE/codeforge/app.py` | FastAPI app + session pool (REST side) |
| `CODEFORGE/codeforge/mcp_server.py` | MCP server — 10 tools, embedded env |
| `CODEFORGE/codeforge/kb/indexer.py` | BM25 search + cluster lookup |
| `CODEFORGE/codeforge/ralph/loop.py` | Score-gated retry loop |
| `CODEFORGE/codeforge/ralph/synthesizer.py` | Stub + LLM synthesizer |
| `CODEFORGE/codeforge/tasks.py` | 3 tasks + hidden correctness tests |

Scope: all three requested depths — (a) architectural + exploit review, (b) SOTA benchmark, (c) RL training-signal critique.

---

## 2. Phase 2a — Deconstruction

### 2a.1 Strengths

| # | Strength | Evidence |
|---|---|---|
| S1 | Env-as-judge separation | `grader.py`, `grounder.py`, `sandbox/` entirely LLM-free. Grading determinism preserved. |
| S2 | AST grounding against live runtime | `grounder.py:53-64` — `importlib.import_module(mod)` + `hasattr(mod, attr)` on full module path. Catches hallucinated APIs before tests. Novel. |
| S3 | Brier calibration on submit | `grader.py:28` — `reward = quality * (1 - min((c-q)², 0.5))`. Forces honest self-assessment. |
| S4 | Penalty-only composite | `metric.py:39` — `raw = 1.0 - imports - ruff - mypy - pytest`. No double-count with pass-rate. |
| S5 | Dual surface (REST + MCP) | Unique positioning. REST = OpenEnv judge path; MCP = agentic-infra path. |
| S6 | Hidden tests bake-in | `tasks.py:25-68`, `environment.py:370` — agent-invisible tests merged at sandbox time. Closes "clean garbage" exploit. |
| S7 | Audit ledger with evidence triple | `(reward, brier, quality, cited_skills, grounding_report)` per step. Replay + verify. |
| S8 | Filename allowlist + size caps | `environment.py:32-58` — `^[a-z][a-z0-9_]*\.py$` regex, forbidden list, 10-file/50KB/200KB caps. Hardens against `conftest.py` hijack, mass-file DoS. |
| S9 | Shaping cap | `shaping.py:36` — 0.05 hard cap prevents reward explosion. |
| S10 | Frozen corpus baked into image | `kb/skills_corpus.jsonl` — deterministic across agent runs. |

### 2a.2 Vulnerabilities

Severity: **CRIT** (blocks production) > **HIGH** (correctness / exploit) > **MED** (edge case / drift risk) > **LOW** (polish).

| # | Sev | File:Line | Issue | Reproduction / Evidence |
|---|---|---|---|---|
| V1 | **CRIT** | `app.py:88` | REST API has no session persistence. `create_app(_get_or_create_env, ...)` uses a factory that returns a **new** `CodeForgeEnvironment()` per call. The `_sessions` / `_session_access` / `create_session` / `get_session` machinery (lines 32–82) is dead code on the OpenEnv HTTP path. After `/reset`, the `/step` request hits a fresh env with no episode. | Agent calls `/reset` → `{episode_id, budget:4}`. Agent calls `/step` with submit action → new env, `self._task is None`, returns `_error_obs("No active episode — call reset() first")`. |
| V2 | **CRIT** | `runner.py:36` | Sandbox subprocess inherits full parent environment including `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROUNDLOOP_CORPUS_PATH`, any `*_TOKEN`. `env = None if env_overrides is None else {**_os_env(), **env_overrides}`. When `env_overrides is None` (default path), `subprocess.run` uses parent env verbatim. | Agent submits `main.py: import os; print(os.environ.get("ANTHROPIC_API_KEY", ""))`. Pytest collects → imports `main` → module body runs at import time → key hits `stdout` captured in `ToolResult` → surfaces via `parsed["pytest"].details["tail"]` in observation. |
| V3 | **CRIT** | `runner.py:29` | No sandbox isolation. Plain `subprocess.run(argv, cwd=tmp, ...)`. No seccomp, no network namespace, no mount namespace, no resource limits. Docker is the only boundary. | Agent submits `main.py: import socket; socket.socket().connect(("attacker",80))` or `import os; os.system("rm -rf ~")`. Both run during pytest's import phase. Container isolates host but not other sessions / training jobs sharing the container. |
| V4 | **HIGH** | `grader.py:32-37` | Uncertain floor applies **only when `confidence is not None`**. Omission branch skips the floor check entirely. Creates asymmetry: honest-uncertain submit with bad code floored at 0.50, but omitted-confidence submit of mediocre code computed normally. | See H1 in §3. `(q=0.8, c=None→0.5)` → brier=0.09 → reward=0.728. `(q=0.8, c=0.85)` → brier=0.0025 → reward=0.798. So honest=better, but vs `(q=0.8, c=0.5)` → reward=0.728 — omission ties calibrated 0.5. And vs `(q=0.3, c=0.2)` → floor=0.50, beats `(q=0.3, c=None→0.5)` → brier=0.04 → reward=0.288. Agent learns: **omit on good code, declare 0.2 on bad**. |
| V5 | **HIGH** | `metric.py:30-36` + `sandbox.py:69-78` | Missing-tool silent bypass. Unavailable tool produces `ParsedResult(ok=False, count=0, details={"unavailable": True})`. `composite_score` then computes `ruff_penalty = min(0, 20)/40 = 0`. If Docker image loses `mypy` on PATH, composite jumps ~0.5 free. | Set `PATH=/usr/bin` stripping `mypy`. Submit garbage code. `ruff_penalty=0` (no ruff), `mypy_penalty=0` (unavailable), `pytest_penalty=0.5` → raw=0.5. Versus agent-unaware-of-drift: thinks all tools ran. Container drift silently inflates reward. |
| V6 | **HIGH** | `shaping.py:6,32` | Citation-comment shaping is trivially forgeable. Agent runs 5 `query_kb("anything")` calls, then submits code with 5 `# cited: python-testing` / `# cited: coding-standards` comments. `citation_shaping_bonus` matches `skill_name` against `_CITATION_RE`, adds +0.01/match up to 0.05. No verification that skill is actually used, no dedup on same skill, no semantic tie to the code. | `prior_citations=[{skill_name:"python-testing"},{skill_name:"coding-standards"},...]`. Submit `main.py: # cited: python-testing\n# cited: coding-standards\n# cited: foo\n# cited: bar\n# cited: baz\ndef greet(...): ...`. Matches=5 → +0.05 reward free. |
| V7 | **HIGH** | `grader.py:27` vs `environment.py:394-397` | Two sites compute Brier + quality. `compute_reward` and inline in `_handle_submit`. Currently numerically equivalent; no protection against drift. One change in one file creates silent divergence between audit (`self._last_brier_penalty`, `self._last_quality`) and actual returned reward. | Grep for `effective_conf` and `brier` — both files. Any refactor to one without the other creates audit lie. |
| V8 | **HIGH** | `environment.py:388` | `concatenated = "\n".join(action.files.values())` — concatenates all files with one newline. No file separators. Grounding report line numbers are meaningless (refer to concat offset, not source file). AST may parse fused garbage or misattribute symbols. | Files `a.py` (ends without newline) + `b.py` (starts with `def foo`) fuse: `...last_liney_of_a\ndef foo(...)`. Parses. Symbols attributed to wrong file in `Symbol.line`. Debug UX broken; false-negative grounding on edge-case multi-file submits. |
| V9 | **MED** | `environment.py:294-298` | Budget charged **before** handler executes. `run_ralph(max_iters=10)` debits 10, Ralph may exit on `target_hit` after 2 iters, or raise after 0. Budget already gone. | `run_ralph(max_iters=10)` with stuck synthesizer → Ralph breaks at iter=3 (`stuck`), 7 iters unused, 10 budget consumed. User paid for unran work. |
| V10 | **MED** | `environment.py:211-218` | `_last_reward`, `_last_citations` etc. reset at step start, populated by handler. If handler raises mid-computation, audit entry shows `reward=0.0` with no distinction between "computed zero" and "never computed." | Handler throws after partial state set → audit says reward=0, brier=None → ambiguous. |
| V11 | **MED** | `grounder.py:144-187` | Attribute-chain resolution walks bottom-up, reverses, builds `full_mod = mod_name + "." + ".".join(chain)`, falls back to `mod_name` if `find_spec(full_mod)` fails. On non-module attributes this produces false ungroundeds. | `import sys; sys.path_hooks.append(...)`. chain=["path_hooks"], full_mod="sys.path_hooks", `find_spec("sys.path_hooks")` returns None (it's a list, not a module), check_mod falls back to "sys", `hasattr(sys, "append")` = False → marked ungrounded. But the code is valid. |
| V12 | **MED** | `loop.py:112-115,147` | `_STUCK_THRESHOLD = 3` only counts `score_regressed`. `score_plateau` resets `consecutive_regressions=0`. Plateau-forever loops run full `max_iters`. | Stub synthesizer + achievable=0.70, target=0.90. Every iter plateaus at 0.70. Never trips stuck. Burns max_iters budget. Waste penalty (0.05/iter) applies only to regressions, not plateaus → silent budget drain with no reward penalty signal. |
| V13 | **MED** | `indexer.py:47,92-99` | `attach_cluster_manifest` mutates `_node_to_cluster` after `build()`. No immutability guard, no versioning. Single-session use is safe today, but pattern invites drift. | Refactor risk. |
| V14 | **MED** | `environment.py:276-277` | Episode done on `budget_remaining <= 0`. If submit hits target it also sets `_is_done=True` (line 419). Double-done paths. Consistent today, but edge case: `run_ralph` doesn't check target post-run. | Ralph ends at 0.92 with target=0.90 → ralph_reward assigned, `_is_done` not set. Agent wastes submit to close episode. |
| V15 | **LOW** | `environment.py:386` | `local_modules` derived from filename stems only. `numpy.py` submitted → `local_modules={"numpy"}` → all `from numpy import ...` treated as local → grounding bypassed entirely. | H10 in §3. Fixable via shadow-filename blocklist. |
| V16 | **LOW** | `synthesizer.py:245,294` | `httpx.post(..., timeout=120.0)` — sync. Blocks env thread. 10 concurrent sessions × 120s worst case = thread pool exhausted. | FastAPI sync-route starvation under load. |
| V17 | **LOW** | `mcp_server.py:63` | Tool description says "1006 skill nodes" — stale. README says 2,648. Drift between doc and data. | Cosmetic but signals doc-code sync discipline gap. |
| V18 | **LOW** | `app.py:77` | `min(_session_access, key=_session_access.get)` — typing cast required (`# type: ignore[arg-type]`). Works but ugly; replace with explicit key lambda. | Minor. |

### 2a.3 Inefficiencies

| # | File | Issue | Fix |
|---|---|---|---|
| I1 | `indexer.py:69-72` | `BM25Okapi.get_scores` scores full corpus then filters by tags. At N=2,648 it's fine; breaks at 10k+. | Pre-filter by tag then score subset. |
| I2 | `environment.py:121-141` | Corpus JSONL read twice per `_ensure_index` — once by `SkillsIndex.build`, once inline for clusters. | Pass nodes list from build to cluster construction. |
| I3 | `environment.py:388-389` | Full AST re-walk on every submit. Cacheable by file-content hash. | `functools.lru_cache` on hashed file contents. |
| I4 | `sandbox.py:41-52,98-100` | `tempfile.mkdtemp` + `shutil.rmtree` per submit. At 1000 req/s = significant disk churn. | Pooled tmp roots with clean-between-use. |
| I5 | `runner.py:29` | Tools run sequentially. ruff/mypy/imports are independent; pytest depends only on imports passing. | `ThreadPoolExecutor` for parallel; ~3x wallclock savings. |
| I6 | `app.py:31-82` | `threading.Lock` around whole session table. | Async-native `asyncio.Lock` or per-session lock. |
| I7 | `sandbox/tools.py:23` | `mypy --no-incremental` disables cache. Slow on every run. | `dmypy` daemon mode — 5-10x faster submit cycles. |

---

## 3. Phase 3 — Hypotheses + Stress Tests

| H | Hypothesis | Test | Verdict | Failure Location |
|---|---|---|---|---|
| H1 | Agent maximizes reward by **omitting confidence on good code + declaring 0.2 on bad** | Compute paths: A `(q=0.8, c=None→0.5)` → brier=0.09 → r=0.728. B `(q=0.8, c=0.85)` → r=0.798. C `(q=0.2, c=0.2)` → quality=0.2, confidence<0.3 AND quality<0.5 → floor → r=0.50. D `(q=0.2, c=None→0.5)` → brier=0.09 → r=0.182. Optimal: B on good, C on bad. Since agent doesn't know q a priori: if any chance of being bad, set c=0.2 to get floor. If confident good, c=None avoids penalty of overestimation. | **Confirmed failure.** Floor-only-when-not-None asymmetry creates exploitable two-branch policy. Training signal trains the wrong behavior. | `grader.py:32-37` |
| H2 | REST-API agent **loses all episode state** between `/reset` and `/step` | `app.py:88` passes `_get_or_create_env` as factory. `openenv-core`'s `create_app` either (a) calls factory per request → fresh env per step, or (b) holds singleton → all concurrent users share state = session bleed. The `_sessions` pool exists but no REST route references it. | **Confirmed failure** both branches. Factory-per-call → "no active episode". Singleton → user-to-user state bleed. REST path is unusable for production. | `app.py:88` + dead code at `app.py:38-82` |
| H3 | Submitted module body executes at pytest import time with full parent env | Submit `main.py: import os; open("/tmp/exfil","w").write(str(dict(os.environ)))`. Pytest's collection imports modules under the tmp dir. Module body runs before any `def test_*` is discovered. `runner.py:36` passes parent env. Written file persists in tmpdir until `shutil.rmtree` but stdout still captured. | **Confirmed failure.** Secrets exfil + arbitrary FS / network in sandbox process. Docker container is the only boundary; shared-training-host scenario is broken. | `runner.py:29-37` + `sandbox.py:52` |
| H4 | Ralph loop loops forever on plateau | Stub synthesizer + target=0.90, stable plateau=0.70. Iter 0: before=0.70, after=0.70 → plateau, `consecutive_regressions=0`. Iter 1: same. ... until `max_iters`. | **Confirmed.** No termination. Full budget drained. Waste penalty only tracks regressions. | `loop.py:112-115` |
| H5 | RL signal is too sparse on easy task (budget=4) | 4-step horizon: 3 info-gathering actions (0 reward) + 1 submit (calibrated reward). For REINFORCE/GRPO: return at step 0,1,2 = 0; return at step 3 = reward. Advantage collapses to submit. Credit cannot flow to earlier actions without value function or shaping. At G=8 group GRPO, variance dominated by Brier stochasticity. | **Likely failure for plain policy-gradient.** Requires value function, potential-based shaping, or trajectory-level advantage attribution. Current `shaping.py` (0.01/forged-citation, 0.05 cap) insufficient. | `environment.py` budget model + `shaping.py` |
| H6 | Sandbox wallclock dominates training throughput | Per submit: ruff 0.1s + imports 0.1s + mypy 2-5s (cold, strict, no-incremental) + pytest 0.5-2s = **~4-8s/submit sync**. Ralph 10 iters × 4s = 40s. 100k rollouts × 10s avg = 278 hours = 11.5 days single-threaded. Parallel tools → ~1.5-3s/submit. dmypy → mypy 0.2s → ~1s/submit total. | **Likely bottleneck.** Current design cannot finish a meaningful training run in reasonable wallclock on DGX without P1-4 (parallel) + P2-2 (dmypy). | `runner.py`, `sandbox.py`, `tools.py` |
| H7 | Citation shaping trivially farmable | Agent does 5× `query_kb("pytest")`. Each returns top-1 `{skill_name: "python-testing", ...}`. Deduped in `cited_skills` set. Submit with `# cited: python-testing\n# cited: coding-standards\n# cited: python-patterns\n# cited: foo\n# cited: bar\n`. `citation_shaping_bonus` matches comments against `cited_skills` set via regex. 5 matches × 0.01 = 0.05. | **Confirmed.** Exploit costs ~5 budget on hard task (max_budget=10). Free +0.05 reward margin. | `shaping.py:23-36` |
| H8 | Missing tool = free pass | Strip `mypy` from PATH via Dockerfile regression. `is_available("mypy")=False`. `parsed["mypy"]=ParsedResult(ok=False, count=0, details={"unavailable":True})`. `composite_score` computes `mypy_penalty = min(0,20)/40 = 0`. All else equal, agent's composite rises by up to 0.5. | **Confirmed.** Container drift silently inflates scores with no loud failure. | `metric.py:32-34` + `sandbox.py:69-78` |
| H9 | Grounder catches `from magic_ai import solve` | `import_module("magic_ai")` raises ModuleNotFoundError. `_has_attr` returns False via `except Exception`. Symbol marked `resolved=False`. Groundedness drops. Pytest also fails on ImportError. Double-penalty. | **Passes.** Core value prop works as designed. | `grounder.py:108-132` |
| H10 | Shadow-filename bypasses grounding | Agent submits `numpy.py` (real stdlib/PyPI name). Filename regex `^[a-z][a-z0-9_]*\.py$` allows it (lowercase start, alphanum+underscore). `local_modules={"numpy"}` → every `from numpy import x` treated as local → `resolved=True`. Groundedness inflated. Pytest may hit collision: tmp dir is CWD, `numpy.py` shadows real numpy, any test importing numpy gets the agent's stub. May fail differently than expected. | **Confirmed partial exploit.** Grounding bypassed. Pytest behavior depends on whether tests actually use numpy. Fix: blocklist stdlib + top-1000 PyPI names. | `environment.py:385,32` |
| H11 | Concurrent sessions race on `_sessions` dict | `app.py:31` `_lock = threading.Lock()`. All reads/writes go through lock. FastAPI is async; sync lock in async handler blocks event loop. 10 concurrent sessions × 120s httpx calls = worker starvation. | **Confirmed in async FastAPI.** Mixing `threading.Lock` with async routes is anti-pattern. Async lock or thread-pool executor needed. | `app.py:31-82` |
| H12 | Synthesizer parse-fail reverts to current files silently | `LLMSynthesizer._parse_response` returns empty `proposed_files` if LLM output has no parseable fenced blocks. `synthesize()` falls back to `current_files` unchanged. Ralph loop's `score_after == score_before` → plateau → `consecutive_regressions=0` (does not count toward stuck). Silent no-op iteration burns budget. | **Confirmed.** Non-parseable LLM responses cost budget without progress signal. Should penalize explicit parse failure as regression. | `synthesizer.py:164-171` + `loop.py:103-115` |

---

## 4. Phase 2b — SOTA Benchmarking

### 4.1 Comparative Matrix

| Dimension | CodeForge | SOTA / Peer | Gap |
|---|---|---|---|
| Task corpus size | **3** | SWE-bench 2,294 · BigCodeBench 1,140 · HumanEval+ 164 · R2E thousands · APPS 10,000 | **2–3 orders behind** |
| Task realism | Toy (`greet`) | SWE-bench — real GitHub issues with real PRs | Not competitive for real code |
| Action surface | 6 actions, whole-file submit | SWE-agent ACI: `open_file`, `goto`, `edit_replace`, `scroll_up/down`, `find_file`, shell, submit | **No incremental editing** |
| Sandbox isolation | `subprocess.run`, parent env passthrough | Sandbox Fusion (ByteDance): seccomp + cgroup + namespace · e2b: Firecracker microVM · Daytona: containerized per-exec | **Weak; Docker is only wall** |
| Language coverage | Python only | Aider bench 10+ · SWE-bench Multilingual Java/JS/Go/Py/Rust · LiveCodeBench Py-only but larger | **Single-language** |
| Retrieval | BM25 only (rank_bm25) | Hybrid BM25 + bge/E5/cohere-embed + reranker · RAG SOTA 2024 | **No semantic; paraphrase queries miss** |
| Grading novelty | AST-ground + Brier | Test-pass only (most benchmarks) · R2E equivalence check | **Ahead** |
| RL framework integration | OpenEnv-native REST | Verifiers (willccbb) → TRL/GRPO direct · verl gym adapter · openrlhf async sampler | **No trainer adapter** |
| Audit trace | Full ledger with evidence triple | SWE-agent trajectories · LiveCodeBench traces | Comparable |
| Confidence calibration | Brier on submit | None in SWE-bench/HumanEval/BigCodeBench · Kadavath 2022 for factual QA · ECE in LM-Eval-Harness | **Ahead in code-gen** |
| Curriculum / procedural gen | Hand-written 3 tasks | APPS (auto-scraped) · R2E (LLM-extracted + test-synthesized) | **No gen pipeline** |
| Tool ecosystem | ruff+mypy+pytest+imports | Sandbox Fusion: 15+ langs + formatters + security scanners + runtime monitors | Narrow |
| Concurrency ceiling | 10 sessions, threading | SWE-bench Lite parallel harness (Modal): 500+ concurrent | Low |
| Reward shape | `0.6*sandbox + 0.4*ground, Brier, floor, citation shaping (0.05 cap)` | SWE-bench: binary pass/fail · BigCodeBench: unit-test pass-rate · R2E: equivalence score · LiveCodeBench: pass@k | Richer than most, but mostly flat gradient between 0.2 and 0.8 |
| Deployment surface | REST + MCP + Docker | Modal / Fly / HF Spaces / Replit sandboxes / e2b | Comparable once HF deploy lands |

### 4.2 Where CodeForge Leads

1. **AST grounding against live runtime is a genuine contribution.** Not present in SWE-bench, HumanEval, BigCodeBench, R2E, LiveCodeBench, APPS, or MBPP. Closes the "fabricate before tests" failure mode explicitly.
2. **Brier-calibrated reward in code-gen RL is novel.** Calibration work exists in factual QA (Kadavath et al. 2022) and ECE in LM-Eval-Harness. Not applied as a reward signal in code-gen RL envs prior.
3. **MCP-native RL env.** Unique. Verifiers, openrlhf, verl are gym-native. CodeForge's MCP surface makes it directly usable as agentic infra for any MCP-aware LLM client without a bespoke adapter.
4. **Evidence-triple audit ledger.** Every reward traces to (sandbox, ground, citation). Stronger than per-step trajectory logs in SWE-agent which don't tie rewards to grounded symbols.

### 4.3 Where CodeForge Lags

1. **Task diversity.** 3 toy tasks. Any agent overfits trivially after ~200 rollouts. No real code-gen benchmark claim possible without 1k+.
2. **Editing primitives.** Whole-file submit vs SWE-agent's ACI primitives. Roughly 2x sample-efficiency loss per SWE-agent paper's own ablation.
3. **Sandbox hardening.** Production deployments (e2b, Daytona, Sandbox Fusion) use VMs or seccomp-cgroup combos. CodeForge is plain `subprocess.run`.
4. **Retrieval quality.** BM25 + Jaccard clustering is 2020-era. MTEB 2024 data shows hybrid retrieval + reranker gains recall@5 by 10-20pp over BM25 on technical docs.
5. **Trainer integration.** No path to verl/TRL/openrlhf today. OpenEnv REST is inference-grade, not training-grade (latency + session bug V1).

---

## 5. RL Training-Signal Critique

### 5.1 Reward Surface Shape

- `quality = 0.6*sandbox + 0.4*groundedness` — both ∈ [0,1].
- `brier_penalty = min((c-q)², 0.5)` — ∈ [0, 0.5].
- `reward = quality * (1 - brier)` — ∈ [0, 1].
- Uncertain floor: `reward = max(reward, 0.50)` iff `c<0.3 and q<0.5 and c is not None`.
- Shaping: `+0.01/citation-comment`, cap 0.05.

**Sandbox granularity:**
- ruff: `min(n, 20)/40` — each violation = 0.025, capped at 0.5.
- mypy: same.
- pytest: binary 0 or 0.5.
- imports: 0.1 per unresolved, capped at 1.0.

**Implication:** single-edit rarely shifts reward by > 0.05. Plateau regions vast. Pytest-flip (binary 0.5) is the dominant gradient signal; everything else is a sloped approach. REINFORCE/GRPO will see near-zero advantage for 80% of mid-quality submissions.

### 5.2 Reward Sparsity

- Easy task: budget 4, target 0.90. 3 info-gathering + 1 submit. Episode is a bandit, not a trajectory.
- Medium: budget 6, target 0.80.
- Hard: budget 10, target 0.70.

Per-step rewards: `query_kb` = 0, `query_cluster` = 0, `interrogate` = 0, `get_audit` = 0, `submit` = calibrated, `run_ralph` = calibrated (per-iter cost aggregated into one reward).

Per-step-reward distribution over an episode: mostly zero, one positive at end. Classic sparse-reward RL.

### 5.3 Exploration

- `StubSynthesizer` is deterministic — Ralph loop is greedy hill-climb with no stochastic exploration.
- `LLMSynthesizer` gets exploration from temperature, not from env.
- No epsilon-greedy, Boltzmann, entropy bonus, or noise injection anywhere.
- `query_kb` always returns top-k deterministically.

### 5.4 Training Throughput Estimate

With current synchronous sandbox:
- submit wallclock: ~4–8s (see H6)
- Average episode: ~3–6 actions → ~5–15s per episode
- 100k episodes ≈ **140–420 hours** single-worker
- 8-worker parallel (DGX H100 host) ≈ **18–50 hours** wallclock for one run

With P1-4 (parallel tools) + P2-2 (dmypy):
- submit wallclock: ~0.5–1.5s
- 100k episodes ≈ **14–40 hours** single-worker
- 8-worker: **2–5 hours**

Sandbox latency is the top optimization target for any DGX training run.

### 5.5 Credit Assignment

- 10-step episode with 9 query actions + 1 submit.
- Only submit has reward.
- GRPO with group-size G: all G trajectories share same prompt. If all G submit reasonable code, advantages come from Brier stochasticity + sandbox noise, not from policy.
- No value function in current design. Credit flows to submit action only.
- Potential-based shaping via `φ(s) = top_bm25_score / max_score` would preserve optimality (Ng 1999) and provide dense signal. Not implemented.

### 5.6 Curriculum

- 3 fixed tasks. No procedural generation. No difficulty scheduling.
- Training on 3 tasks → policy memorizes `greet(name)` after hundreds of rollouts. Generalization untested and untestable.

### 5.7 Suitability Verdict by Trainer

| Trainer | Feasibility | Blocker |
|---|---|---|
| **REINFORCE** (bandit baseline) | Feasible today for single-task overfit demo | V1 (REST session), reward sparsity, 3-task overfit |
| **GRPO** (DeepSeek/Qwen style) | Feasible after V1 fix + P1-4 sandbox parallel | Group variance dominated by Brier/sandbox noise, not policy |
| **RLOO** | Same as GRPO | Same |
| **Full PPO with value fn** | Requires value head | Framework adapter missing (P1-5) |
| **DPO / KTO** (preference-based) | Incompatible | No preference pairs; reward is scalar not ordinal |

---

## 6. Phase 4 — Strategic Update Plan

### 6.1 Priority Tiers

| Priority | Count | Description |
|---|---|---|
| **P0 — Critical** | 5 | Correctness / security bugs blocking production |
| **P1 — High Impact** | 6 | Architectural changes unlocking training or SOTA positioning |
| **P2 — Improvement** | 10 | Polish, ops hardening, perf |
| **P3 — Research** | 6 | Novel directions, not required |

### 6.2 P0 — Critical

#### P0-1. Fix REST session persistence

- **What:** Replace the factory pattern in `app.py:88` with session-aware routes that reuse the existing `_sessions` pool.
- **Why:** V1 / H2 — REST path is broken today. OpenEnv judge cannot maintain episode state.
- **How:** Custom FastAPI router replaces `create_app(factory, ...)` contract. Accept `session_id` as header or query param on `/step`, `/state`. `/reset` returns a new `session_id`. Wire to existing `create_session` / `get_session`. If OpenEnv strict mode requires implicit single-session, document the extension.
- **Expected impact:** Unblocks real REST agents. Baseline runs become possible on deployed HF Space.
- **Dependencies:** None. Pure `app.py` rewrite. Tests in `test_app.py` need session-aware assertions.

#### P0-2. Sandbox isolation

- **What:** Replace `subprocess.run` with hardened runner supporting `bubblewrap` (Linux), `nsjail`, or `firejail`. Scrub env to an allowlist.
- **Why:** V2+V3 / H3 — agent code executes with parent env and full FS/network.
- **How:** New `SandboxRunner` abstraction with backends `{"bubblewrap","nsjail","firejail","subprocess"}`. Detect at startup, default to highest available. Env allowlist: `PATH`, `LANG`, `LC_*`, `HOME=/tmp/cf-home`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`. Drop everything else, especially `*_API_KEY`, `*_TOKEN`, `*_SECRET*`, `AWS_*`, `GROUNDLOOP_*`, `ANTHROPIC_*`, `OPENAI_*`. Network: `--unshare-net` / `--net none`.
- **Expected impact:** Eliminates secret exfil, network C2, FS escape. Required for multi-tenant training.
- **Dependencies:** Dockerfile adds `bwrap` (already in apt). Minor `runner.py` rewrite. `test_sandbox.py` regression tests for env scrubbing.

#### P0-3. Fix Brier calibration inversion

- **What:** Either (a) apply the uncertain floor regardless of `confidence is None` (treat None as 0.5 for floor check too), **or** (b) penalize None harder by treating as `effective_confidence=1.0` (worst-case over-claim).
- **Why:** V4 / H1 — current shape trains agents to omit confidence on good code.
- **How:** Recommend option (b) — forces declaration. Change `grader.py:27`: `effective_confidence = 1.0 if confidence is None else confidence`. Update docstring. Update MCP tool description. Update `test_grader.py`.
- **Expected impact:** Calibration signal aligned with claim. Training rewards honest calibration.
- **Dependencies:** P0-4 (reward unification) first — change must land in one place, not two.

#### P0-4. Unify reward computation

- **What:** Delete inline Brier math in `environment._handle_submit:393-397`. Make `compute_reward` return a `RewardBreakdown` record with `{reward, quality, brier, floor_applied, effective_confidence}`. Environment reads fields for audit.
- **Why:** V7 — split-brain drift risk.
- **How:** Add `@dataclass(frozen=True) class RewardBreakdown` in `grader.py`. `compute_reward` returns it. Bump `_codeforge_version` minor. Update `environment._handle_submit`, `test_grader.py`, `test_environment.py`.
- **Expected impact:** Single source of truth. Reviews simpler. P0-3 ships cleanly.
- **Dependencies:** None. Ship with P0-3.

#### P0-5. Missing-tool fail-loud

- **What:** Unavailable required tools must fail the submit, not score zero penalty.
- **Why:** V5 / H8 — container drift silently inflates rewards.
- **How:** `ParsedResult` gains `available: bool`. Default True. `sandbox.py:69-78` sets False on unavailable. `composite_score` returns `None` (or raises) if any required tool is unavailable; `_handle_submit` surfaces as error or sets reward=0 with explicit error message.
- **Expected impact:** Container health regressions fail loudly.
- **Dependencies:** `ParsedResult` schema bump. Audit entries reflect `available_tools: tuple[str,...]`.

### 6.3 P1 — High Impact

#### P1-1. Task-corpus expansion (curriculum)

- **What:** Port R2E-style extraction pipeline. Scan PyPI top-500 + GitHub trending Python, extract single-function or small-class units with existing pytest tests, synthesize briefs via LLM at **curriculum build time only**.
- **Why:** 3 tasks cannot train a generalist. SOTA peers have 1k–10k.
- **How:** New `codeforge/curriculum/extractor.py`. Offline pipeline emits `tasks_auto.jsonl`. Validation: every auto-task must reach target ≥ 0.85 with a reference implementation (LLM-generated, grader-verified). Reject the rest. Grading stays deterministic — LLM only used offline.
- **Expected impact:** Training becomes meaningful. Benchmarking gains credibility. Likely moves from "toy RL env" to "publishable".
- **Dependencies:** Offline LLM budget (one-time). No runtime changes.

#### P1-2. Incremental edit actions

- **What:** Add `edit_replace(file, old, new)`, `open_file(file)`, `list_files()` as new action types. Submit still required for grading.
- **Why:** Whole-file submit is anti-agentic. SWE-agent ACI primitives are proven more sample-efficient.
- **How:** 3 new values in `CodeForgeActionType`. Handlers mutate `self._current_files`. Cost=1 each. No direct reward. Budget rebalanced (easy 4→6, medium 6→10, hard 10→16).
- **Expected impact:** ~2x sample efficiency (per SWE-agent paper ablation on WCBase). Unlocks real-world dev loop.
- **Dependencies:** Budget rebalancing + curriculum update (P1-1 tasks generated with edit-first in mind).

#### P1-3. Semantic retrieval

- **What:** Hybrid BM25 + sentence-transformer retrieval.
- **Why:** Paraphrased queries miss BM25. Semantic gap is dominant retrieval failure.
- **How:** `kb/embedder.py` uses `sentence-transformers/all-MiniLM-L6-v2` (23MB, CPU-fast) or `fastembed` (ONNX, faster). Pre-embed corpus once, store in `skills_embeddings.npz`. Hybrid score `α*bm25 + (1-α)*cos`, α=0.5. RRF also valid.
- **Expected impact:** Recall@5 +10-20pp over BM25 alone on technical docs (MTEB 2024).
- **Dependencies:** +1 dep (`sentence-transformers` or `fastembed`). Embedding baked into Docker image. Corpus rebuild trigger.

#### P1-4. Parallel sandbox execution

- **What:** Run ruff + mypy + imports concurrently; pytest gated on imports passing.
- **Why:** H6 — sandbox wallclock dominates. Quick 3x win.
- **How:** `ThreadPoolExecutor(max_workers=3)` in `sandbox.run_sandbox`. Results collected. Pytest runs only if imports resolve.
- **Expected impact:** 4-8s/submit → 1.5-3s/submit. At 100k rollouts = 100+ hours saved.
- **Dependencies:** None.

#### P1-5. Trainer adapter

- **What:** `codeforge/adapters/verl.py`, `codeforge/adapters/trl_gym.py`. Expose env as gym.Env-compatible iterator for verl RLHF and TRL GRPO.
- **Why:** No DGX training path today. OpenEnv REST is inference-grade.
- **How:** Async wrapper around `CodeForgeEnvironment`, batched rollouts, reward returned as scalar per step. Advantage normalization trainer-side.
- **Expected impact:** Unlocks DGX GRPO/PPO training.
- **Dependencies:** Target trainer pinned. Needs user constraint input (verl vs TRL vs openrlhf).

#### P1-6. Reward shaping for RL trainability

- **What:** Replace forgeable `# cited:`-comment shaping with potential-based shaping + action-completion micro-bonuses.
- **Why:** V6 / H5 / H7 — current shaping is gameable and doesn't help credit assignment.
- **How:**
  - Remove `_CITATION_RE` substring match.
  - Add potential-based shaping: `φ(s) = 0.1 * (best_bm25_score_this_episode / 20.0)`. Reward shaping `F(s,s') = γ*φ(s') - φ(s)`. Preserves optimal policy (Ng, Harada, Russell 1999).
  - Add micro-bonus per informative action: `+0.005` on `query_kb` returning ≥1 result, cap at `0.02` per episode.
- **Expected impact:** Dense, non-exploitable signal. Empirical RL convergence 2-5x faster with potential-based shaping (literature).
- **Dependencies:** P0-4 reward unification first.

### 6.4 P2 — Improvement

| ID | What | Why | How | Impact |
|---|---|---|---|---|
| P2-1 | Per-file AST parsing | V8 — concat corrupts line numbers | `ground(files: dict)` aggregates per-file reports | Accurate audit, no false negatives |
| P2-2 | mypy daemon mode (`dmypy`) | H6 — cold mypy is 80% of sandbox time | `dmypy start` once per container, `dmypy check` per submit | 5-10x mypy speedup |
| P2-3 | Shadow-filename blocklist | V15 / H10 — `numpy.py` bypasses grounding | Extend `_FORBIDDEN_FILENAMES` with stdlib + top-1000 PyPI | Close grounding bypass |
| P2-4 | Ralph plateau stuck detection | V12 — infinite plateau loop | Count `plateau` toward stuck with weight 0.5 | Bounded episode length |
| P2-5 | Ralph refund on early termination | V9 — paid for unran iters | Refund `(max_iters - actual_iters) * 1` on `target_hit` | Fairer economics |
| P2-6 | Async httpx for LLMSynthesizer | V16 / H11 — sync blocks async loop | `httpx.AsyncClient` + `await` | Scales to 10+ concurrent sessions |
| P2-7 | Session auth on MCP SSE + REST | No auth documented | Bearer token middleware | Required for public deploy |
| P2-8 | Rate limiting | No throttle on `/reset`, `/step` | `slowapi` 60 req/min/IP | DoS mitigation |
| P2-9 | Corpus dedup + quality filter | ECC has duplicates across platform dirs | Hash-based dedup + min-length filter in scraper | Smaller index, faster BM25 |
| P2-10 | Evaluation harness | No baseline leaderboard | Fixed-seed eval set + pass@1/pass@5/avg-budget reports | Progress tracking |
| P2-11 | Update stale MCP tool description | V17 — "1006 skill nodes" vs actual 2,648 | Edit `mcp_server.py:63` | Doc sync |
| P2-12 | Per-file grounder line numbers | V8 | Track file-offset → line map | Debug UX |

### 6.5 P3 — Research

- **P3-1.** Procedural task generation via LLM+grader-validation loop (R2E pattern).
- **P3-2.** Multi-language support (TypeScript, Rust, Go) via pluggable tool registries. Each lang gets `sandbox/runners/<lang>.py`.
- **P3-3.** Learned reward model: train small reward head on audit traces to predict `quality` from `(current_files, action_history)`. Distills slow sandbox into fast differentiable signal for value function / critic bootstrap.
- **P3-4.** MCTS over action space: replace Ralph greedy hill-climb with UCT rollouts using sandbox as simulator. Budget-constrained depth-limited search.
- **P3-5.** Adversarial red-team corpus: tasks specifically designed to elicit hallucination. Briefs referencing `numpy.foo_that_doesnt_exist` to test if grounder catches the agent's implementation.
- **P3-6.** Difficulty ranking via item-response theory on agent performance logs. Dynamic curriculum scheduling.

---

## 7. Recommended Execution Order

1. **Week 1:** P0-1 (REST session) + P0-4 (unify reward) + P0-3 (calibration fix). One commit each. Unlocks deploy.
2. **Week 2:** P0-2 (sandbox isolation) + P0-5 (missing-tool fail-loud). Security-critical before any shared deployment.
3. **Week 3:** P1-4 (parallel sandbox) + P2-2 (dmypy). Wallclock win before training runs.
4. **Week 4-5:** P1-5 (trainer adapter) + P1-6 (potential-based shaping). Unlocks DGX training.
5. **Week 6+:** P1-1 (curriculum expansion — offline R2E pipeline) + P1-2 (edit actions) + P1-3 (semantic retrieval). Unlocks publishable RL training run.
6. **P2 items** interleaved as capacity allows.
7. **P3 items** post-training-run validation.

---

## 8. Open Questions for User

1. Target trainer: verl, TRL (GRPO), or openrlhf? Drives P1-5 shape.
2. Compute budget for training: single DGX node H100×8, multi-node, or smaller?
3. Success metric priority: OpenEnv judge pass, benchmark leaderboard (which?), internal eval, or novelty over SWE-agent/R2E?
4. Who is the end-user of the trained agent — Claude, GPT, Qwen-Coder, DeepSeek-Coder, or in-house?
5. Is the 3-task toy corpus a stepping-stone (curriculum expansion planned) or terminal (research demo only)?

---

**End of analysis.**

**Verdict in one line:** Novel grading architecture (ahead of SOTA) wrapped in a broken delivery surface (REST session bug) with a security-untenable sandbox and a calibration signal that trains the wrong behavior. Fix P0 before any deployment claim. P1 before any training claim.
