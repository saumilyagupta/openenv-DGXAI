# Cell 18 — Baseline Evaluation

`eval_baseline(...)` runs the **untrained Gemma 3n E2B** on the first 50 rows of
`val/briefs.jsonl` under frozen-greedy sampling and returns an `EvalReport`
with bootstrap CIs (`n_boot=10_000`, `rng_seed=20260426`).

**Contract:** evaluation.md §2.1, §3.1–§3.3, §3.8, §4, §5.

- 50 held-out val episodes, file-order (no shuffle).
- `env.reset(seed=hash((episode_id, "eval")) & 0xFFFFFFFF)`.
- Greedy: `temperature=0.0`, `num_generations=1`, `model.eval()` + `torch.no_grad()`.
- Wall-clock ceiling 20 min; raises `EvalBudgetExceededError` on overrun.
- No LLM-as-judge (forbidden imports listed in `_NO_LLM_JUDGE_FORBIDDEN_IMPORTS`).

The training-eval delegate is **injected** so unit tests stub model inference
on CPU-only CI (training_tests.md §5.3 `mock_cuda` pattern).
