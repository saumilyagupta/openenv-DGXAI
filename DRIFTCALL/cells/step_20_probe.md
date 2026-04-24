# Cell 20 — Reward-Hacking Probe (200 episodes)

`probe_reward_hacking(checkpoint, ...)` scans `Rewards.breakdown.anti_hack`
across 200 held-out val episodes (`val/briefs.jsonl[50:250]`) for the 5
enumerated exploit classes plus any novel offense codes (threshold = 1).

**Contract:** evaluation.md §2.1, §2.3, §3.1, §3.6, §3.8, §4.4, §4.5, §5.

- Disjoint from the paired-comparison 50 episodes.
- All 5 known classes always emitted (count == 0 rows kept for the fixed table).
- Novel offense codes surfaced under `ProbeReport.novel_classes` and flagged
  with `UNKNOWN EXPLOIT CLASS` in the markdown writeup.
- `ProbeOnBaseModelError` raised if `model_path == "base"`.
- `ProbeInsufficientSamplesError` raised if `episodes < 50`.
- Wall-clock budget 60 min — `EvalBudgetExceededError` on overrun.
- No LLM-as-judge anywhere in the scoring path.
