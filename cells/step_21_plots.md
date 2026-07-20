# Cell 21 — Eval-Curve Renderer (4 PNG Panels)

`render_plots(baseline, final, wandb_run_id, out_dir)` produces the four plot
panels at DESIGN.md §15 pitch 1:00–2:00:

1. `per_reward_stack.png` — R1..R5 means vs training step (WandB history).
2. `drift_latency_vs_step.png` — drift-detection latency p50/p95 vs step.
3. `per_language_bars.png` — per-language R1..R5 cohort means.
4. `before_after_bars.png` — baseline vs final per-reward means + 95% CI.

**Contract:** evaluation.md §2.1, §3.4, §3.5, §3.8, §5.

- `matplotlib` only (no seaborn).
- Canonical figsize `(16, 9)` inches at `dpi=100` → 1600x900 px.
- `wandb_run_id=None` → skip the two history-driven plots; warn via
  `WandBHistoryUnavailableWarning`.
- Wall-clock budget 2 min; raises `EvalBudgetExceededError` on overrun.
