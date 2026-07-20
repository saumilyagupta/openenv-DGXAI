# Cell 22 — Markdown Summary Table (Baseline → Final)

`print_summary_table(baseline, final)` returns the multi-section markdown
summary that ships in the HF blog and DESIGN.md §15 pitch:

1. **Per-reward** (mean + 95% CI) — baseline → final → paired Δ with CI.
2. **Per-language** — baseline reward_mean → final → Δ.
3. **Drift-detection latency** — Stage 2/3 p50/p95 before vs after.
4. **Reward-hacking offenses** — per-class baseline → final counts.

**Contract:** evaluation.md §3.3, §3.4, §3.5; DESIGN.md §13 deliverables #6 / #7.
Numeric cells round to 3 decimals (latency to 2). Paired Δ pulled from
`final.breakdown['paired_ci']` (populated by `eval_final` in step_19).
