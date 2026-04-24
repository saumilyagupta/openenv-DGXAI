# training_tests.md — Test Plan for `training/train_grpo.py`

**Target module:** `training/train_grpo.py` (primary), `training/eval_baseline.py`, `training/eval_final.py`
**Spec doc:** `DRIFTCALL/docs/modules/training.md` (final sealed)
**Cross-refs:** `DRIFTCALL/docs/modules/rewards.md` §3.1 (purity contract), `DRIFTCALL/CLAUDE.md` §3.1 (nine-section test-plan doc)
**Framework:** `pytest` + `hypothesis` + `unittest.mock` (CUDA mocked)
**Owner:** Person B (Rewards & Tests), co-signed by Person C (Training)
**CUDA policy:** **CUDA is mocked by default.** All unit and property tests run on CPU via the `mock_cuda` fixture (§5.3). Integration tests gate GPU paths behind a `@pytest.mark.cuda` marker that is skipped when `torch.cuda.is_available() is False` — the default CI/laptop environment.
**Numeric tolerance:** `math.isclose(a, b, abs_tol=1e-9, rel_tol=0.0)` for floats; exact equality for ints/enums/strings.

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `training/train_grpo.py` (CUDA-runtime-only lines excluded via `# pragma: no cover-nocuda` and validated by the `mock_cuda` fixture path). Every `GRPOConfig` invariant in training.md §2.4, every error mode in training.md §5, every edge case in training.md §7a–§7h, and every clause of training.md §3.1–§3.6 has at least one dedicated test. The reward-pipeline purity contract from rewards.md §3.1 is enforced as a first-class property test.

Fixtures listed in §5 split into two groups:

1. **Training-local fixtures** (defined in this plan, §5.1–§5.5): `grpo_config_stage1/2/3`, `toy_env`, `mock_cuda`, `episode_dataset_adapter`, `reward_fn_smoke`. These are authored here.
2. **Imported eval fixtures**: `eval_50_episodes_val_slice` is **defined in `evaluation_tests.md §5.1`** and imported unchanged for integration test §3.4 (paired baseline/final eval). This plan does NOT re-define it; it consumes the shared `tests/conftest.py` registration. See §5 footer cross-reference.

If any training-local fixture content changes here, `tests/conftest.py` MUST be updated in lockstep. Any modification to an imported eval fixture must happen in `evaluation_tests.md` first (the source of truth).

---

## 1. Unit tests

**Organisation:** one `pytest` module per behavior cluster. CUDA-free by default — every test either uses `mock_cuda` or asserts against pure-Python logic.

File layout under `tests/test_training/`:

```
tests/test_training/
  __init__.py
  test_grpo_config_invariants.py        # GRPOConfig assertions (§2.4)
  test_episode_dataset_adapter.py       # EpisodeDatasetAdapter iteration (§2.2)
  test_reward_fn_signature.py           # TRL 0.23 contract (§2.3)
  test_prompt_serialization.py          # apply_chat_template wiring (§3.2.1)
  test_rollout_padding.py               # Uneven termination padding/masking (§3.2.2)
  test_bf16_slippage_assertion.py       # V100 safety halt (§3.1)
  test_resume_rng_restore.py            # 5-RNG restore (§3.6)
  test_eval_sampling_policy.py          # eval() determinism (§2.1 `eval`)
  test_checkpoint_save.py               # safe_serialization=True (§3.6)
  test_wandb_offline_fallback.py        # LocalCSVCallback (§2.4.1)
  test_error_modes.py                   # All TrainingError subclasses (§5)
  test_grad_accum_fallback.py           # G=4 grad_accum=8 switch (§7b)
  test_reward_collapse_watchdog.py      # RewardCollapseError (§7d)
  test_kl_explosion_watchdog.py         # KLDivergenceExplosion (§7c)
  test_eval_budget_exceeded.py          # Time budgets (§7h / run-book §C5)
  test_language_cohort_collapse.py      # LanguageCohortCollapseError (§7f)
```

**Unit test case inventory — 34 cases total (exceeds the ≥ 25 requirement):**

### 1.1 `GRPOConfig` invariants — `test_grpo_config_invariants.py`

**Scope:** Every field asserted in training.md §2.4 must be exactly the documented value, and `build_grpo_config` must reject any deviation.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_config_stage1_uses_warmup_01` | `build_grpo_config(stage=1)`. | `math.isclose(cfg.warmup_ratio, 0.1, abs_tol=1e-9)`; `cfg.lr_scheduler_type == "cosine"`. |
| U2 | `test_config_stage2_uses_warmup_0` | `build_grpo_config(stage=2)`. | `cfg.warmup_ratio == 0.0` exactly (prevents double-warmup per §3.5). |
| U3 | `test_config_stage3_uses_warmup_0` | `build_grpo_config(stage=3)`. | `cfg.warmup_ratio == 0.0` exactly. |
| U4 | `test_config_bias_correction_kl_true` | `build_grpo_config(stage=1)`. | `cfg.use_bias_correction_kl is True` (TRL issue #4637, §3.3). |
| U5 | `test_config_fp16_true` | `build_grpo_config(stage=1)`. | `cfg.fp16 is True`; `getattr(cfg, "bf16", False) is False`. |
| U6 | `test_config_gradient_checkpointing_true` | `build_grpo_config(stage=1)`. | `cfg.gradient_checkpointing is True`. |
| U7 | `test_config_num_generations_default_8` | `build_grpo_config(stage=1)`. | `cfg.num_generations == 8`; `cfg.gradient_accumulation_steps == 4`; `cfg.num_generations * cfg.gradient_accumulation_steps == 32`. |
| U8 | `test_config_num_generations_4_flips_grad_accum_8` | `build_grpo_config(stage=1, num_generations=4)` (§7b fallback). | `cfg.num_generations == 4`; `cfg.gradient_accumulation_steps == 8`; product == 32. |
| U9 | `test_config_rejects_num_generations_not_in_set` | `build_grpo_config(stage=1, num_generations=16)`. | Raises `AssertionError` with substring `"num_generations in {4, 8}"`. |
| U10 | `test_config_beta_kl_is_0_04` | Any stage. | `math.isclose(cfg.beta, 0.04, abs_tol=1e-9)`. |
| U11 | `test_config_max_lengths_1024_2048` | Any stage. | `cfg.max_prompt_length == 1024`; `cfg.max_completion_length == 2048`. |
| U12 | `test_config_report_to_wandb` | Any stage. | `cfg.report_to == "wandb"`; `cfg.run_name == f"driftcall-stage{N}"` for N in {1,2,3}. |
| U13 | `test_config_per_device_batch_1` | Any stage. | `cfg.per_device_train_batch_size == 1`. |

### 1.2 `EpisodeDatasetAdapter` — `test_episode_dataset_adapter.py`

**Scope:** `__iter__` must yield exactly one record per GRPO step with `prompt` (string) + `_meta` dict carrying `(goal, episode_seed, stage, language_weights)`. Task generator called once per step with `seed=stage_base_seed+step`.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U14 | `test_adapter_yields_goalspec_prompt_pair` | Fixture `episode_dataset_adapter` (stage=1, stage_base_seed=1_000_000). `next(iter(adapter))`. | Yields `{"prompt": str, "_meta": {"goal": GoalSpec, "episode_seed": int, "stage": int, "language_weights": dict}}`. `prompt` starts with rendered system prompt. |
| U15 | `test_adapter_seeds_monotonically` | Iterate adapter 5 times. | `_meta["episode_seed"]` values are `[1_000_000, 1_000_001, 1_000_002, 1_000_003, 1_000_004]`. |
| U16 | `test_adapter_calls_task_generator_once_per_step` | Mock `task_generator.generate`. Iterate 3 times. | `generate.call_count == 3`; all calls have `stage=1` and `language_weights` matching fixture. |
| U17 | `test_adapter_prompt_uses_apply_chat_template` | Fixture with mocked tokenizer that records calls. Iterate once. | `tokenizer.apply_chat_template.call_args.kwargs == {"tokenize": False, "add_generation_prompt": True}` (training.md §3.2.1). |
| U18 | `test_adapter_system_prompt_is_pinned` | Inspect rendered prompt. | Contains the verbatim frozen string from `training/prompts.py::SYSTEM_PROMPT` (training.md §3.2.1 "pinned"). |

### 1.3 `reward_fn` TRL 0.23 signature — `test_reward_fn_signature.py`

**Scope:** Signature contract `reward_fn(prompts, completions, *, _meta, episodes, **kwargs) -> list[float]`; values in `[0, 1]` rounded to 3dp; per-rollout purity (no cross-rollout state leak).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U19 | `test_reward_fn_returns_list_of_g_floats` | G=8 terminal episodes (fixture `reward_fn_smoke`). | `len(result) == 8`; all `isinstance(x, float)`. |
| U20 | `test_reward_fn_values_in_unit_interval` | Same. | Every `0.0 <= x <= 1.0`. |
| U21 | `test_reward_fn_3_decimal_precision` | Same. | Every `x == round(x, 3)` (compare `x * 1000` to its int round). |
| U22 | `test_reward_fn_delegates_to_compute_rewards` | Monkey-patch `rewards.compute_rewards` to return a sentinel `Rewards` with `reward=0.777`. Call `reward_fn` with G=1 episode. | `result == [0.777]`. |
| U23 | `test_reward_fn_signature_trl_023_compatible` | `inspect.signature(reward_fn)`. | Signature has positional `prompts, completions`; keyword-only `_meta, episodes`; `**kwargs` tail. |

### 1.4 Uneven rollout termination — `test_rollout_padding.py`

**Scope:** Padding to `L_max` with `pad_token_id`; `completion_mask` has 1s for real tokens, 0s for padding; `gen_length_mean` is computed over **unpadded** lengths (§3.2.2).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U24 | `test_padding_to_l_max_uses_pad_token_id` | Mock rollouts with lengths `[3, 5, 7]`, `pad_token_id=0`. | Padded tensor shape `(3, 7)`; row 0 ends with `[0, 0, 0, 0]`; row 1 ends with `[0, 0]`; row 2 has no padding. |
| U25 | `test_completion_mask_marks_padding_zero` | Same. | `mask[0] == [1,1,1,0,0,0,0]`; `mask[1] == [1,1,1,1,1,0,0]`; `mask[2] == [1,1,1,1,1,1,1]`. |
| U26 | `test_gen_length_mean_excludes_padding` | Same. | `math.isclose(gen_length_mean, (3+5+7)/3, abs_tol=1e-9)` — NOT `(7+7+7)/3`. |
| U27 | `test_pad_token_fallback_to_eos_when_missing` | Tokenizer with `pad_token_id=None, eos_token_id=2`. | Padding uses `2`; `train()` entry assertion does NOT raise. |

### 1.5 Prompt serialization — `test_prompt_serialization.py`

**Scope:** `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`; `sort_keys=True` + `ensure_ascii=False` on every JSON dump; overflow policy drops oldest `tool` messages first (§3.2.1).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U28 | `test_tool_message_json_uses_sort_keys_ensure_ascii_false` | Serialize a tool result with Devanagari content. | JSON has sorted keys; Devanagari script preserved (no `र` escapes). |
| U29 | `test_overflow_drops_oldest_tool_message_first` | Construct a conversation whose rendered length exceeds 1024 tokens with 5 tool messages. | After overflow, oldest tool message dropped; system + seed_utterance + last 2 turns preserved; a `[truncated N older tool_results]` system banner prepended. |

### 1.6 BF16-slippage assertion — `test_bf16_slippage_assertion.py`

**Scope:** Immediately after `FastModel.from_pretrained`, `train()` asserts `next(model.parameters()).dtype == torch.float16` and halts otherwise (§3.1).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U30 | `test_bf16_slippage_raises_at_train_entry` | Mock `FastModel.from_pretrained` to return a model whose first param is `torch.bfloat16`. Call `train(stage=1, num_steps=1)`. | Raises `AssertionError` with substring `"BF16 slipped through"`; halts BEFORE `GRPOTrainer` construction (verify via spy). |
| U31 | `test_fp16_model_passes_assertion` | Mock returns a `torch.float16` model. | No raise; proceeds (rollout is further mocked out). |

### 1.7 Resume — 5-RNG restore — `test_resume_rng_restore.py`

**Scope:** On resume, `train()` restores all five RNG sources (`torch_cpu`, `torch_cuda`, `numpy`, `python_random`, TRL-internal via `trainer.train(resume_from_checkpoint=...)`) BEFORE any rollout or optimizer step (§3.6).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U32 | `test_resume_restores_torch_cpu_rng` | Save a known `torch.get_rng_state()` into `rng_states.pt`. Resume. | After resume, `torch.get_rng_state()` bytes match the saved bytes. |
| U33 | `test_resume_restores_torch_cuda_rng` | Save `torch.cuda.get_rng_state_all()` (mocked via `mock_cuda`). Resume. | `torch.cuda.set_rng_state_all` called with the saved per-device list. |
| U34 | `test_resume_restores_numpy_rng` | Save `np.random.get_state()`. Resume. | After resume, `np.random.get_state()` equals the saved state (tuple-by-tuple comparison). |
| U35 | `test_resume_restores_python_random_rng` | Save `random.getstate()`. Resume. | After resume, `random.getstate()` equals saved state. |
| U36 | `test_resume_restores_trl_rng_via_trainer_state` | Spy on `trainer.train`. Resume. | `trainer.train` called with `resume_from_checkpoint=Path(...)` pointing at the same dir that holds `rng_state.pth` (TRL's standard file). |
| U37 | `test_resume_missing_rng_states_is_soft_warning` | `rng_states.pt` absent; adapter present. | No raise; `train/rng_restore_fallback=1` logged; `random.seed(config_sha256_as_int)` invoked. |
| U38 | `test_resume_restores_before_any_rollout` | Spy on `rollout_group` and `set_rng_state`. | `set_rng_state` call occurs strictly before the first `rollout_group` call (verify via `mock_calls` order). |

### 1.8 Eval sampling policy — `test_eval_sampling_policy.py`

**Scope:** `eval()` uses `temperature=0.0`, `num_generations=1`, `model.eval()` on every submodule, `torch.no_grad()` wrapping; deterministic episode selection from `val/briefs.jsonl`; `ProbeOnBaseModelError` when `model_path=="base"` and caller passes probe-only flags (§2.1).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U39 | `test_eval_uses_temperature_zero` | Mock `model.generate`, call `eval(model_path=fixture_checkpoint, episodes=1)`. | `model.generate.call_args.kwargs["temperature"] == 0.0`. |
| U40 | `test_eval_uses_num_generations_1` | Same. | The effective `num_generations` at eval is `1` (no group structure). |
| U41 | `test_eval_dropout_off_everywhere` | Spy: set `model` with a `nn.Dropout` submodule. Call `eval`. | `model.eval()` called; every submodule's `.training` is `False` after entry; no `p.requires_grad is True` on any parameter during rollout. |
| U42 | `test_eval_seeds_from_hash_episode_id_eval_salt` | Capture `env.reset` seeds. `episode_id == "ep_xyz"`. | Seed `== hash(("ep_xyz", "eval")) & 0xFFFFFFFF`. Baseline and final eval invocations produce identical seed sequences. |
| U43 | `test_eval_reads_rows_in_file_order` | `val/briefs.jsonl` fixture with 3 rows. `eval(..., episodes=3)`. | `env.reset` calls' episode_ids match file order exactly. |
| U44 | `test_eval_never_writes_to_wandb` | Spy `wandb.log`. Call `eval`. | `wandb.log.call_count == 0`. |
| U45 | `test_probe_on_base_model_raises` | Call the reward-hacking probe entry-point with `model_path="base"`. | Raises `ProbeOnBaseModelError` with substring `"probe requires trained adapter"`. |

### 1.9 Checkpoint save — `test_checkpoint_save.py`

**Scope:** `save_pretrained(safe_serialization=True)`; `rng_states.pt` written alongside; `driftcall_meta.json` emitted; integrity re-load verify on save (§3.6 + §7e).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U46 | `test_save_uses_safe_serialization_true` | Mock model with `save_pretrained` spy. Trigger save. | `save_pretrained.call_args.kwargs["safe_serialization"] is True`. |
| U47 | `test_save_writes_rng_states_pt_alongside` | Trigger save. | `Path(out_dir, "rng_states.pt").exists()` is True; content is a dict with keys `{"torch_cpu","torch_cuda","numpy","python_random"}`. |
| U48 | `test_save_emits_driftcall_meta_json` | Trigger save. | `driftcall_meta.json` exists with keys from `CheckpointMeta` (stage, steps_completed, config_sha256, etc.). |
| U49 | `test_save_raises_checkpoint_io_error_on_integrity_mismatch` | Mock `save_pretrained` to write truncated bytes; on verify-reload, canonical-prompt output mismatches. | `CheckpointIOError` raised after 1 retry; error message mentions the output path. |
| U50 | `test_save_prohibited_merge_and_unload` | Ensure `train()` never calls `model.merge_and_unload()` nor any `dequantize→merge→requantize` path. | Spy on those methods; call_count == 0 after a full train+save cycle. |

### 1.10 WandB offline fallback — `test_wandb_offline_fallback.py`

**Scope:** `LocalCSVCallback` mirrors every `on_log` to `checkpoints/<run>/metrics.csv` with stable 20-column schema; `WandBStartupError` raised only when `WANDB_MODE != "offline"` AND `wandb.init()` fails (§2.4.1 + §3.4).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U51 | `test_wandb_offline_mode_skips_startup_error` | `WANDB_MODE=offline`; mock `wandb.init` to raise. | `train()` does NOT raise `WandBStartupError`; proceeds. |
| U52 | `test_wandb_online_mode_raises_startup_error` | `WANDB_MODE` unset; mock `wandb.init` to raise. | `WandBStartupError` raised at `train()` entry. |
| U53 | `test_local_csv_callback_writes_20_columns` | Trigger one `on_log` with a full logs dict. | CSV header row has exactly 20 columns in training.md §3.4 order starting with `step`; one data row appended. |
| U54 | `test_local_csv_nan_encoded_as_string` | Log dict contains `float("nan")`. | CSV row has `"nan"` (not `""` or empty). |
| U55 | `test_wandb_runtime_failure_non_fatal` | Training mid-run; `wandb.log` raises `ConnectionError`. | Training continues; warning logged; CSV row still appended. |

### 1.11 Error modes — `test_error_modes.py`

**Scope:** Every exception in training.md §5 has a dedicated raising test.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U56 | `test_out_of_memory_wrapped_from_cuda_oom` | Mock `model.generate` to raise `torch.cuda.OutOfMemoryError`. G=4 already. | `OutOfMemoryError` (training-local subclass) raised; `torch.cuda.empty_cache` called. |
| U57 | `test_non_finite_gradient_error_after_3_consecutive_skips` | Callback receives `grad_norm=float("inf")` 4 times in a row. | `NonFiniteGradientError` raised on the 4th; `train/skipped_updates == 3` before raise. |
| U58 | `test_kl_divergence_explosion_on_rolling_mean_over_10` | Feed 10-step window of `policy_kl` values averaging 10.5. | `KLDivergenceExplosion` raised; `debug/kl_explosion_dump.jsonl` written with last 20 rollouts. |
| U59 | `test_reward_collapse_error_on_r5_and_reward_drop` | `R5_mean` goes -0.05 → -0.35 and `reward_mean` drop > 0.15 within 10 steps. | `RewardCollapseError` raised. |
| U60 | `test_checkpoint_io_error_on_hf_hub_5xx_after_3_retries` | Mock `push_to_hub` to raise `HfHubHTTPError(status=503)` 3 times. | `CheckpointIOError` raised; backoffs `[2.0, 4.0, 8.0]` observed. |
| U61 | `test_tokenizer_mismatch_error_on_vocab_size_diff` | Resume from checkpoint whose tokenizer `vocab_size=32001` while base is `32000`. | `TokenizerMismatchError` raised at resume time. |
| U62 | `test_episode_parse_error_does_not_escape_trainer` | Model emits garbage JSON in rollout. | Episode continues; `train/episode_parse_failures` incremented; no exception propagates to `trainer.train()`. |
| U63 | `test_eval_model_load_error_no_fallback_to_base` | `eval(model_path=<bad path>)`; adapter file missing. | `EvalModelLoadError` raised; never silently evaluates on base. |
| U64 | `test_eval_budget_exceeded_20min` | `eval_baseline` wall-clock > 20 min budget (mock `time.monotonic`). | `EvalBudgetExceededError` raised with `phase="baseline"` and `limit_seconds=1200`. |
| U65 | `test_eval_budget_exceeded_60min_stage_eval` | Stage-level eval wall-clock > 60 min. | `EvalBudgetExceededError` raised with `limit_seconds=3600`. |
| U66 | `test_eval_budget_exceeded_2min_single_episode` | Single episode wall-clock > 2 min. | `EvalBudgetExceededError` raised with `limit_seconds=120`. |

### 1.12 G=4 fallback and grad_accum switch — `test_grad_accum_fallback.py`

**Scope:** On OOM at G=8, next group retries at G=4 with grad_accum=8; mid-accumulation-window G-flips are forbidden; window is abandoned (no optimizer step) per §7b.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U67 | `test_g8_oom_triggers_g4_fallback_on_next_group` | First group OOMs at G=8; second group succeeds at G=4. | `build_grpo_config` called again with `num_generations=4`; `gradient_accumulation_steps == 8`. |
| U68 | `test_mid_window_oom_abandons_window` | G=8 OOMs at group 3 of an 8-group accumulation window. | No optimizer step for that window (`optimizer.step.call_count` unchanged); fresh G=4/grad_accum=8 window starts. |
| U69 | `test_g4_still_oom_drops_max_completion_length_to_1536` | G=4 OOMs again. | `max_completion_length` flipped to 1536 for the remainder of the stage; WandB logs `train/max_completion_length=1536`. |
| U70 | `test_g4_oom_at_1536_raises_to_user` | G=4 at 1536 still OOMs. | `OutOfMemoryError` raised (training-local subclass); no further fallback. |

### 1.13 Language cohort collapse — `test_language_cohort_collapse.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U71 | `test_cohort_soft_warning_at_20_steps` | `reward_hi == NaN` for 20 consecutive logging ticks. | `train/cohort_collapse:hi` logged at warning level; no raise. |
| U72 | `test_cohort_hard_error_at_50_steps` | Same for 50 consecutive ticks. | `LanguageCohortCollapseError` raised. |
| U73 | `test_stage2_validates_non_english_weight_min_005` | `train(stage=2, ..., language_weights={"hi": 0.0, ...})`. | Raises `ValueError` at call time with substring `"weight >= 0.05 for non-English at stage >= 2"`. |

**Total unit cases: 73** (well in excess of the ≥ 25 floor; sectioning mirrors training.md §2–§7).

---

## 2. Property tests

Hypothesis strategies live in `tests/test_training/strategies.py`. All strategies produce **frozen** `Episode` / `Rewards` / `RolloutBatch` values to match the dataclass `frozen=True` invariant.

### 2.1 Reward-pipeline purity (rewards.md §3.1 enforced at the training boundary)

**P1. `compute_rewards` on `Episode` is pure — no state leak into training.**

- **Strategy:** `st.lists(episode_strategy(), min_size=2, max_size=8)` — a group of G episodes.
- **Invariant:** calling `reward_fn(prompts, completions, _meta=..., episodes=eps)` twice in succession on the same inputs yields identical outputs; running `reward_fn` on episode `i` alone vs. inside a batch of G produces the same scalar for episode `i`.
- **Mechanism:** wraps the rewards.md §3.1 "no RNG, no clock, no I/O" contract at the training-plumbing layer. If any global state (cache, counter, RNG) leaks, the two invocations will disagree.
- **Runs:** `settings(max_examples=200, deadline=2000)`.

**P2. No cross-rollout state leak within a group.**

- **Strategy:** `st.lists(episode_strategy(), min_size=4, max_size=8)`.
- **Invariant:** for any permutation `π` of the episode list, `reward_fn(...)[π(i)] == reward_fn(original)[i]` element-wise. Changing episode order must not change any scalar.
- **Purpose:** catches an implementation that accidentally shares a mutable buffer across rollouts in the G-group.

### 2.2 GRPO advantage is group-relative, not batch-relative

**P3. Rollout scalars feed advantage group-relative.**

- **Strategy:** `st.tuples(st.lists(st.floats(0, 1, allow_nan=False), min_size=8, max_size=8), st.lists(st.floats(0, 1, allow_nan=False), min_size=8, max_size=8))` — two groups of 8 rewards each.
- **Invariant:** compute `A = (r - mean(group)) / (std(group) + 1e-8)` per-group on each; result equals what TRL's internal advantage computes when `num_generations=8`. When the two groups are concatenated into a single batch, the per-group advantages must NOT change (i.e., `std(batch)` is never used as the denominator).
- **Runs:** 200 examples.

**P4. Group-relative advantage is shift-invariant.**

- **Strategy:** `st.lists(st.floats(0, 1), min_size=8, max_size=8)`, `st.floats(-100, 100)`.
- **Invariant:** `advantage(rewards) == advantage([r + c for r in rewards])` for any constant `c` (within 1e-6 after std re-normalization). Confirms the computation strips group mean.

### 2.3 Padding + masking preserves reward signal

**P5. Reward is invariant under completion-token padding.**

- **Strategy:** `st.lists(st.integers(min_value=1, max_value=100), min_size=4, max_size=8)` — G rollout lengths.
- **Invariant:** padding rollouts to `L_max` and applying `completion_mask` never changes the scalar returned by `reward_fn` (rewards see the frozen `Episode`, never the tokenized trace — training.md §3.2.2 "Reward invariance").
- **Runs:** 200 examples.

### 2.4 Prompt rendering determinism

**P6. `tokenizer.apply_chat_template` is byte-stable under equal inputs.**

- **Strategy:** `st.builds(goal_strategy(), st.text(min_size=1, max_size=100))`.
- **Invariant:** rendering the same `(system_prompt, messages)` twice yields byte-identical strings; `sort_keys=True` + `ensure_ascii=False` preserve Devanagari/Tamil/Kannada script.

### 2.5 Config immutability

**P7. `build_grpo_config` is a pure function — same args, same output.**

- **Strategy:** `st.sampled_from([1,2,3])` × `st.sampled_from([4, 8])`.
- **Invariant:** two invocations with identical `(stage, num_generations)` return `GRPOConfig` objects with identical field values (compare via `asdict` then dict equality).

**Total property-test families: 7** (exceeds the ≥ 5 floor).

---

## 3. Integration tests

Integration tests live under `tests/test_training/integration/`. All are marked `@pytest.mark.integration` and run in CI but are **skipped locally** unless `PYTEST_INTEGRATION=1`. CUDA-dependent assertions guarded by `@pytest.mark.cuda` and skipped when `torch.cuda.is_available() is False`.

### 3.1 10-step GRPO smoke test on V100 (mocked-GPU fallback on CPU)

**File:** `tests/test_training/integration/test_smoke_10step.py`

- **Setup:** `toy_env` fixture (§5) — a minimal `DriftCallEnv` wired with hand-crafted `GoalSpec`s that terminate in ≤ 3 turns. `grpo_config_stage1` with `num_steps=10`, `save_steps=5`, `logging_steps=1`. `mock_cuda` fixture installed to let the same test run on a laptop.
- **Run:** `train(stage=1, num_steps=10)`.
- **Assertions:**
  - Training completes without raising.
  - `trainer.state.global_step == 10`.
  - `checkpoints/stage1_final/adapter_model.safetensors` exists.
  - `checkpoints/stage1_final/rng_states.pt` exists.
  - `metrics.csv` has exactly 10 data rows (logging_steps=1) with the 20-column header.
  - `train/reward_mean` is finite on every row; no NaNs.
  - `@pytest.mark.cuda` guarded assertion (skipped on laptop): peak `torch.cuda.max_memory_allocated()` < 20 GB.

### 3.2 Stage transition — Stage 1 → Stage 2 with fresh GRPOConfig

**File:** `tests/test_training/integration/test_stage_transition.py`

- **Setup:** Run 3 steps at `stage=1` (saves `checkpoints/stage1_final`). Then call `train(stage=2, num_steps=3, resume_from=Path("checkpoints/stage1_final"))`.
- **Assertions:**
  - `build_grpo_config(stage=2)` produces `warmup_ratio == 0.0` (no double warmup).
  - TRL loads `trainer_state.json` — `trainer.state.global_step` continues from 3 (not reset to 0).
  - LR at Stage-2 step 0 matches the cosine-schedule value at cumulative step 3, NOT peak `5e-6`.
  - `language_weights` flipped to Stage-2 mix `{en:0.30, hinglish:0.30, hi:0.20, ta:0.10, kn:0.10}`; `env_factory` produces envs with `curriculum_stage=2`.
  - WandB run_name changes from `driftcall-stage1` to `driftcall-stage2` but `WANDB_RUN_GROUP == "curriculum-v1"` on both.

### 3.3 Resume from mid-step-crash

**File:** `tests/test_training/integration/test_resume_midcrash.py`

- **Setup:** Run 5 steps, save at step 5, then simulate a crash (raise `KeyboardInterrupt` from inside the rollout at step 7). Fresh process: `train(stage=1, num_steps=10, resume_from=Path("checkpoints/stage1_final"))`.
- **Assertions:**
  - Resume does NOT raise.
  - `trainer.state.global_step` starts from 5, not 0 or 7.
  - All 5 RNGs restored before any rollout (verify via spy ordering).
  - Final step count == 10 — i.e., exactly 5 additional steps run.
  - The 5 step-6..10 rollouts consume `task_generator.generate(seed=stage_base_seed + step)` with `step ∈ {5,6,7,8,9}` (reproducible derivation from step counter, not from saved state).

### 3.4 Eval baseline vs. final on deterministic 50-episode set

**File:** `tests/test_training/integration/test_eval_paired.py`

- **Setup:** `val/briefs.jsonl` fixture with exactly 50 deterministically-seeded `GoalSpec` rows.
  - Run `eval(model_path="base", episodes=50)` → `EvalReport` baseline.
  - Run `eval(model_path=fixture_stage3_checkpoint, episodes=50)` → `EvalReport` final.
- **Assertions:**
  - Both invocations call `env.reset` with **identical** seed sequences (paired comparison).
  - `baseline.reward_mean_ci[0] < final.reward_mean_ci[0]` — training improves reward.
  - `baseline.drift_detection_latency.stage2_mean > final.drift_detection_latency.stage2_mean` — latency drops.
  - `final.reward_hacking_offenses` counts are all bounded (≤ 5 across all codes) — no silent hacking.
  - `final.per_language` has 5 entries; `n_episodes` sums to 50 exactly.
  - Both reports' `curves` dicts are JSON-serializable (round-trip via `json.dumps(asdict(report))`).

### 3.5 WandB offline → online transition

**File:** `tests/test_training/integration/test_wandb_offline_then_online.py`

- **Setup:** Start with `WANDB_MODE=offline`; run 3 steps; clear env var; run 3 more steps.
- **Assertions:**
  - First 3 steps: no `wandb.init` network call.
  - `metrics.csv` captures all 6 steps (CSV is the authoritative record).
  - On second launch with `WANDB_MODE` unset, if `wandb.init` succeeds runs proceed; if it fails, `WandBStartupError` raised before rollout.

---

## 4. Coverage target

**Line coverage:** **100%** on `training/train_grpo.py`.
**Branch coverage:** **≥ 95%** on `training/train_grpo.py`.
**Rationale for < 100% branch:** genuinely-unreachable error paths (e.g., `torch.cuda.OutOfMemoryError` raised from inside `torch.cuda.empty_cache()` itself — a contradiction) may drop branch coverage by ≤ 1 point; documented with `# pragma: no branch` and justified inline.

**CUDA-runtime exclusions:** lines that require an actual CUDA device (e.g., `torch.cuda.max_memory_allocated()`) are tagged `# pragma: no cover-nocuda` and exercised only under the `mock_cuda` fixture path OR the `@pytest.mark.cuda` integration suite. The `mock_cuda` fixture (§5.3) replaces `torch.cuda.*` with CPU-compatible no-ops and stubs `torch.cuda.is_available() -> True`, `torch.cuda.OutOfMemoryError` as a raisable subclass, and `torch.cuda.get_rng_state_all() -> [torch.ByteTensor([0])]`. This collapses the CUDA-gated branch delta to zero for the in-test path while preserving the hardware-only lines for real-V100 CI.

**20-WandB-columns coverage assertion:** `test_local_csv_callback_writes_20_columns` (U53) is the authoritative registration test — it inspects the header row and asserts the exact 20-column ordering from training.md §3.4. Any column added / removed / reordered in the source without updating this test fails. The 20 columns are:

1. `step`
2. `train/reward_mean`
3. `train/reward_std`
4. `train/policy_kl`
5. `train/gen_length_mean`
6. `train/grad_norm`
7. `train/loss`
8. `train/learning_rate`
9. `train/R1_mean`
10. `train/R2_mean`
11. `train/R3_mean`
12. `train/R4_mean`
13. `train/R5_mean`
14. `train/drift_detected_rate`
15. `train/format_compliance_rate`
16. `train/hallucinated_field_count`
17. `train/reward_hi`
18. `train/reward_ta`
19. `train/reward_kn`
20. `train/reward_en`

(`train/reward_hinglish` is carried as column 21 in the source but the CSV schema pins 20; hinglish goes to WandB Table only. If the spec bumps the CSV schema, U53 must be updated in lockstep and `__CSV_COLUMNS__` bumped in `training/train_grpo.py`.)

**Coverage command (from `DRIFTCALL/` root):**

```bash
python3 -m pytest tests/test_training/ \
    --cov=training.train_grpo \
    --cov-report=term-missing \
    --cov-branch \
    --cov-fail-under=100
```

A `coverage.ini` `[run] branch = True` plus `[report] exclude_lines = ["# pragma: no cover-nocuda"]` completes the config.

---

## 5. Fixtures

All fixtures live in `tests/conftest.py`. Every fixture is frozen, deterministic, and **CUDA-free** by default. Fixtures §5.1–§5.5 are defined by this plan; §5.6 documents the imported cross-plan fixture whose definition lives in `evaluation_tests.md`.

### 5.1 `grpo_config_stage1`, `grpo_config_stage2`, `grpo_config_stage3`

```python
@pytest.fixture
def grpo_config_stage1() -> "GRPOConfig":
    from training.train_grpo import build_grpo_config
    return build_grpo_config(stage=1)

@pytest.fixture
def grpo_config_stage2() -> "GRPOConfig":
    from training.train_grpo import build_grpo_config
    return build_grpo_config(stage=2)

@pytest.fixture
def grpo_config_stage3() -> "GRPOConfig":
    from training.train_grpo import build_grpo_config
    return build_grpo_config(stage=3)
```

Each returns a `GRPOConfig` matching training.md §2.4 for the given stage. Training-local — not shared (evaluation uses its own sampling-policy assertions and does not instantiate `GRPOConfig`).

### 5.2 `toy_env`

```python
@pytest.fixture
def toy_env() -> "Callable[[], DriftCallEnv]":
    """
    Factory producing a minimal DriftCallEnv wired with:
      - 1 scripted domain (airline, single route HYD→BLR)
      - 2-turn max episode (force fast termination)
      - no drift at stage=1; single `price_rename` at stage=2/3
      - deterministic tool mock (fixed JSON responses)
    Returns a callable so each rollout gets a fresh env (training.md §3.2).
    """
    ...
```

Used in integration tests 3.1, 3.2, 3.3. Returns envs whose rollouts terminate in ≤ 3 turns to keep smoke-test runtime under 60 s.

### 5.3 `mock_cuda`

```python
@pytest.fixture
def mock_cuda(monkeypatch):
    """
    Installs CPU-compatible stubs for every torch.cuda.* call training uses:
      - is_available() -> True
      - device_count() -> 1
      - current_device() -> 0
      - empty_cache() -> None (no-op)
      - max_memory_allocated() -> 0
      - get_rng_state_all() -> [torch.ByteTensor([0])]
      - set_rng_state_all(states) -> None
      - OutOfMemoryError exists as a raisable subclass of RuntimeError
    Enabled by default on every unit test via autouse=False + explicit
    request; integration tests pick the real CUDA path when available.
    """
    ...
```

This is the **authoritative** CUDA mock. Unit tests request it explicitly; integration tests skip it when `@pytest.mark.cuda` is used and real CUDA is available.

### 5.4 `episode_dataset_adapter`

```python
@pytest.fixture
def episode_dataset_adapter(toy_env) -> "EpisodeDatasetAdapter":
    """
    Deterministic EpisodeDatasetAdapter:
      - stage=1
      - stage_base_seed=1_000_000
      - language_weights={"en": 1.0, ...rest zero}  (simplifies assertions)
      - task_gen = real task_generator (seeded, deterministic)
      - env_factory = toy_env
      - tokenizer = a stub with apply_chat_template spy (records kwargs)
    """
    ...
```

Used by unit tests §1.2. Training-local (wires a training-tokenizer spy). Evaluation iterates `eval_50_episodes_val_slice` directly rather than going through `EpisodeDatasetAdapter`, so this fixture is not shared.

### 5.5 `reward_fn_smoke`

```python
@pytest.fixture
def reward_fn_smoke() -> "Callable":
    """
    Returns a reward_fn pre-wired with:
      - monkey-patched rewards.compute_rewards that returns a canonical
        Rewards(reward=0.5, r1=..., r2=..., breakdown={...}) per episode
      - 8 pre-built terminal Episodes (happy + edge cases, language mix)
    Used for TRL-0.23 signature tests (§1.3) and property tests P1/P2.
    """
    ...
```

### 5.6 `eval_50_episodes_val_slice` (imported from `evaluation_tests.md`)

**Definition site:** `evaluation_tests.md §5.1` — the authoritative source of truth for the 50-row `val/briefs.jsonl` slice fixture. Consumed verbatim by integration test §3.4 (`test_eval_paired.py`) for the baseline-vs-final paired eval.

**Canonical name:** `eval_50_episodes_val_slice`. Any earlier draft name (`val_briefs_50`) is deprecated — use the canonical name only. Consumers in this plan:

- §3.4 `test_eval_paired` — calls `eval(model_path=..., episodes=50)` against the 50-row slice.
- §6 dependencies note (formerly referred to `val_briefs_50`) — updated to this canonical name.

**Import mechanism:** fixture is registered in `tests/conftest.py` by the evaluation suite; pytest auto-discovers it when this plan's integration tests request it by name. This plan MUST NOT redefine the fixture (doing so would fork the content and break the source-of-truth contract).

**Sharing attested in:** `evaluation_tests.md §5.6` cross-plan contract row 1.

---

## 6. Dependencies between test files (to keep CI green)

- `test_episode_dataset_adapter.py` depends on `toy_env` + deterministic `task_generator` (already covered in `task_generator_tests.md`).
- `test_rollout_padding.py` uses the stub tokenizer from `episode_dataset_adapter` fixture to avoid pulling Gemma tokenizer at unit-test time.
- `test_resume_rng_restore.py` uses `mock_cuda` — restoring CUDA RNG on CPU is a no-op stub but asserts the call flow.
- Integration tests share `val/briefs.jsonl` with `evaluation_tests.md` via the `eval_50_episodes_val_slice` fixture (defined in `evaluation_tests.md §5.1`, imported by this plan per §5.6 above). The two test plans must not double-write the file or redefine the fixture.

---

## 7. Open questions (for plan critic)

1. **Torchmetric for `policy_kl` rolling-mean window.** Spec says 10-step moving mean; use `torch.tensor` buffer or `collections.deque(maxlen=10)`? Proposal: deque (no GPU dep, trivially mockable). Confirmed in test U58.
2. **Integration test runtime ceiling.** `test_smoke_10step.py` budget is 60 s on laptop CPU with `mock_cuda`, 180 s on V100. If laptop exceeds 60 s, reduce toy_env max_turns to 2.
3. **`ProbeOnBaseModelError` source.** Training spec mentions `model_path=="base"` as a valid eval argument; the probe test U45 assumes the reward-hacking probe module raises it. If the probe lives in `tests/test_rewards/`, owning test module moves there — CLAUDE.md §2.2 (Person B). Left here for now since the error is the training-eval coupling's concern.

---

## 8. Running this suite

```bash
# Default (CPU, CUDA mocked, unit + property tests only)
python3 -m pytest tests/test_training/ -v \
    --cov=training.train_grpo --cov-branch --cov-report=term-missing

# Include integration tests (still CPU-safe via mock_cuda)
PYTEST_INTEGRATION=1 python3 -m pytest tests/test_training/ -v

# Real V100 smoke (onsite only)
PYTEST_INTEGRATION=1 python3 -m pytest tests/test_training/ -v -m cuda
```

---

**This test plan is a design doc. No test code is written until ≥ 2 fresh critics return `NOTHING_FURTHER` on it (CLAUDE.md §3.4).**
