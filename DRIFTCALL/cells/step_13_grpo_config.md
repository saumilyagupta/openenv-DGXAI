# Step 13 — GRPO Config + Reward Wiring

Builds a TRL `GRPOConfig` matching `docs/modules/training.md §2.4` exactly — `use_bias_correction_kl=True`, FP16, gradient-checkpointing, `beta=0.04`, `per_device_train_batch_size=1`, `num_generations ∈ {4, 8}` with `grad_accum` flipped so effective rollouts/update stays at 32. Also provides the TRL-0.23-compatible `reward_fn(prompts, completions, *, _meta, episodes, **kwargs)` that delegates to `compute_rewards` pure function and returns list-of-floats in `[0, 1]` rounded to 3dp. No reward normalization pre-GRPO (training.md §3.2).
