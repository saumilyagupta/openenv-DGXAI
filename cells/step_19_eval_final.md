# Cell 19 — Final Evaluation (Post-Training LoRA)

`eval_final(checkpoint, ..., baseline=baseline_report)` runs the trained LoRA
on the **same** 50 paired episodes used by the baseline (evaluation.md §3.1)
and stores the paired-difference 95% CIs under
`EvalReport.breakdown['paired_ci']`.

**Contract:** evaluation.md §2.1, §3.1, §3.3, §3.8, §5 `EpisodeSetLeakError`.

- `EpisodeSetLeakError` raised at entry AND exit if `baseline.episode_ids ≠
  val/briefs.jsonl[0:50]` or the post-rollout report's IDs diverge.
- Paired bootstrap CI seed = `20260428` (evaluation.md §2.4).
- Wall-clock budget 20 min — same ceiling as baseline.
