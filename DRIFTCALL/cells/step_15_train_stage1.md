# Step 15 — Stage-1 GRPO training entry

Stage-1 is the curriculum origin (training.md §3.5, DESIGN.md §10.3): 150 GRPO steps, no drift, language mix 50% English / 30% Hinglish / 20% Hindi, `warmup_ratio=0.1`. `resume_from` is rejected — there is no prior stage. Saves checkpoints every 50 steps via `save_pretrained(safe_serialization=True)`; never the naive 4-bit -> 16-bit merge path (DESIGN.md §10.5).

`train(stage=1, num_steps=150, resume_from=None)` boots Gemma 4 E2B in 4-bit (FP16-pinned) via `boot_gemma`, re-asserts FP16 dtype (`BF16SlippageError` halts on V100 if any param leaks through as BF16; training.md §3.1), constructs the GRPOConfig + `EpisodeDatasetAdapter` + `DriftCallGRPOTrainer`, initialises wandb (offline-safe; `WandBStartupError` only when `WANDB_MODE != "offline"`), and runs `trainer.train()` for the requested step count. The `task_gen`, `env_factory`, and `rollout_group_fn` callables are passed by the notebook orchestrator so the cell stays decoupled from the env + data builders.

`build_run_plan` is the pure-function entry point — tests use it to verify the resolved arguments without exercising the GPU stack. `write_local_csv_row` mirrors every WandB log dict to `metrics.csv` with the stable 20-column schema from training.md §3.4 (NaN encoded as `"nan"`).
