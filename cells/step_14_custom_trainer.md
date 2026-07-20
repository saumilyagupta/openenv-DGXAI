# Step 14 — DriftCallGRPOTrainer + EpisodeDatasetAdapter

Custom TRL subclass `DriftCallGRPOTrainer` that replaces the single-prompt / single-completion rollout phase with the DriftCall multi-turn env loop (training.md §3.2.3). Its `_generate_and_score_completions` override runs G parallel multi-turn episodes via a caller-provided `RolloutGroupFn`, then hands terminal frozen `Episode` objects plus raw completion strings to `reward_fn` (step_13). Advantage + KL + optimizer steps are inherited unchanged from `GRPOTrainer`.

`EpisodeDatasetAdapter` is the stateless streaming iterator wired into `GRPOTrainer.train_dataset`. Each `__iter__` yield packages `{prompt, _meta}` where `_meta` carries `(goal, episode_seed, stage, language_weights)` — every scalar required by the rollout controller. Per-step record: one `task_generator.generate` call, one `apply_chat_template` render, monotonically increasing `episode_seed == stage_base_seed + step`.

Both types defer `trl` + `torch` imports until construction so the module loads on CPU-only CI.
