# training.md — DriftCall Training Pipeline

**Module:** `training/train_grpo.py` (primary), `training/eval_baseline.py`, `training/eval_final.py`
**Owner:** Person C (Training & Data)
**Implements:** DESIGN.md §10 (Training Pipeline — §10.1 Stack, §10.2 GRPOConfig, §10.3 Curriculum, §10.4 Monitoring, §10.5 Checkpoint Saving), §3.2 (Training Topology), §3.5 (Hardware/Credit Budget), §14 (Risk Register — risks #1, #2, #4, #5)
**Consumes:** `driftcall.env.DriftCallEnv` (via in-process OpenEnv wrapper), `driftcall.rewards.compute_rewards` (pure fn on frozen `Episode`), `driftcall.task_generator.generate` (seeded per-reset `GoalSpec`)
**Produces:** LoRA adapter checkpoints on disk + HF Hub, WandB run with ≥ 13 monitoring columns, `EvalReport` JSON for baseline + final
**Status:** Design spec — implementation (`training/train_grpo.py`) does not start until ≥ 2 fresh critic agents return `NOTHING_FURTHER`.

---

## 1. Purpose

The training module is the **RL optimizer** for DriftCall. It fine-tunes `unsloth/gemma-4-E2B-it-bnb-4bit` via **GRPO with group-relative advantages** (TRL 0.23+ `GRPOTrainer`, Unsloth 2026.4.5+ `FastModel`) against the DriftCall environment so the model learns to (a) tool-use fluently, (b) detect mid-episode schema drift, (c) adapt tool calls to the post-drift schema, and (d) maintain calibrated confidence under uncertainty.

This module exists to:

1. **Wire the environment to GRPO correctly.** Rollouts are full episodes — multi-turn `reset()` → `step(action)` → … → terminal `SUBMIT`/`ABORT`/`TIMEOUT` sequences — and the **only** scalar GRPO consumes is `Rewards.reward` from `rewards.compute_rewards(episode)` (DESIGN.md §7.2, rewards.md §2.3). Per-reward components `R1..R5` and diagnostic `breakdown` are logged, never used as advantage.
2. **Run the 3-stage curriculum** (DESIGN.md §10.3): 150 warmup steps (no drift) → 200 single-drift steps → 150 compound-drift steps. Total **500 GRPO steps × G=8 rollouts × ~6 turns ≈ 24,000 individual agent trajectories** (DESIGN.md §10.3).
3. **Stay V100-safe.** Gemma 4 is BF16-native; V100 is FP16-only. Training runs in 4-bit NF4 + FP16 autocast with gradient checkpointing and `use_bias_correction_kl=True` to survive KL estimation numerics (DESIGN.md §14 risk #1, #2).
4. **Checkpoint without corrupting the 4-bit base.** Save LoRA adapters via `save_pretrained(safe_serialization=True)`; never naively upcast 4-bit→16-bit + merge for re-training (DESIGN.md §10.5, CLAUDE.md §9).
5. **Produce the before/after numbers** that drive the pitch: a `baseline` `EvalReport` (untrained Gemma 4 E2B on 50 held-out episodes) and a `final` `EvalReport` (trained LoRA on the same 50), with per-reward means, drift-detection-latency curve, and per-language breakdown (DESIGN.md §15 pitch 1:00–2:00 segment).

No audio in training. Text-in / text-out only — ASR/TTS are env-boundary concerns (DESIGN.md §9.4). No LLM-as-judge anywhere in the pipeline — the env is the judge (CLAUDE.md §0.5).

---

## 2. Interface

All snippets use `from __future__ import annotations`.

### 2.1 Top-level entry points

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal, Protocol

def train(
    stage: Literal[1, 2, 3],
    num_steps: int,
    resume_from: Path | None = None,
) -> "CheckpointPath":
    """
    Run GRPO for ``num_steps`` updates on curriculum ``stage``.

    - ``stage == 1``: warmup, no drift, Stage-1 language mix (50% en, 30% hinglish, 20% hi).
    - ``stage == 2``: single drift per episode, Stage-2 language mix.
    - ``stage == 3``: compound drift, same language mix as Stage 2.

    Behavior:
      1. Build the Unsloth FastModel + LoRA adapters (§3.1).
      2. Build ``GRPOConfig`` via ``build_grpo_config(stage)`` (§2.4).
      3. Build the episode-rollout dataset (seeded briefs from task_generator).
      4. Construct ``GRPOTrainer`` with the 5-reward pipeline wrapped as a single
         scalar reward function (§3.2).
      5. If ``resume_from`` is given, load adapters + optimizer/scheduler state
         (§3.6); else fresh init.
      6. Run trainer.train() for ``num_steps``.
      7. Save adapters to ``checkpoints/stage{N}_final`` with
         ``safe_serialization=True`` (§3.6, DESIGN.md §10.5).

    :raises OutOfMemoryError:        wrapped CUDA OOM; caller triggers G=4 fallback (§5, §7a/b).
    :raises NonFiniteGradientError:  NaN / inf detected in gradient norm (§5, §7c).
    :raises KLDivergenceExplosion:   ``policy_kl`` mean > 10.0 over a 10-step window (§5, §7c).
    :raises CheckpointIOError:       save / load / HF Hub push failure (§5, §7e).
    :returns: absolute path to the adapter checkpoint directory.
    """

def eval(
    model_path: Path,
    episodes: int,
) -> "EvalReport":
    """
    Evaluate ``model_path`` (base model + LoRA OR bare base model if
    ``model_path == "base"``) on ``episodes`` held-out seeds (default 50 from
    ``val/briefs.jsonl`` per DESIGN.md §8.6).

    Sampling policy (deterministic — baseline-vs-final paired comparison):
      - ``temperature = 0.0`` — greedy decoding, no stochasticity.
      - ``num_generations = 1`` — one rollout per episode (no group structure).
      - ``model.eval()`` + ``torch.no_grad()`` wrapped around the full rollout;
        dropout OFF on every module (``for m in model.modules(): m.eval()``).
      - No LoRA dropout, no attention dropout. LoRA adapters frozen
        (``p.requires_grad = False`` on every parameter).

    Episode selection (deterministic row iteration):
      - Read ``val/briefs.jsonl`` in file order (DESIGN.md §8.6 writes rows
        in stable seed-sorted order).
      - Take rows ``[0 : episodes]`` — baseline eval and final eval consume
        the IDENTICAL set of rows.
      - For each row ``i``: ``env.reset(seed=hash((episode_id_i, "eval")) & 0xFFFFFFFF)``
        — stable 32-bit seed, reproducible across reruns, distinct from
        training seeds (the "eval" salt prevents eval/train seed collision).
      - Baseline and final use IDENTICAL ``(episode_id_i, seed)`` pairs →
        paired-difference statistics are valid.

    Produces an EvalReport (§4.2) containing:
      - per-reward means (R1..R5), each with 95% bootstrap CI
      - ``reward_mean`` (scalar GRPO signal), ``brier_mean``
      - ``drift_detection_latency`` — turns between drift firing and first
        positive R2 branch hit (DESIGN.md §15 pitch 1:00–2:00)
      - per-language breakdown: {hi, ta, kn, en, hinglish} → per-reward means
      - ``floor_applied_rate`` (calibrated-surrender frequency)
      - ``hallucinated_field_rate`` (R5 branch (a) trigger rate)
      - ``reward_hacking_offenses`` (count by code, from breakdown)

    Never writes to WandB (eval is local-only); caller may serialize to JSON.

    :raises EvalModelLoadError: adapter load / merge failure (§5).
    :returns: EvalReport dataclass instance.
    """

def build_grpo_config(
    stage: Literal[1, 2, 3],
    *,
    num_generations: int = 8,
    resume_output_dir: Path | None = None,
) -> "GRPOConfig":
    """
    Build a TRL ``GRPOConfig`` matching DESIGN.md §10.2 exactly.

    ``num_generations`` defaults to 8 (G=8). Caller may pass 4 after an OOM
    fallback trigger (§7b).

    Invariants (enforced by __post_init__-style assertions after construction):
      - ``use_bias_correction_kl is True``  (DESIGN.md §10.2, risk #2)
      - ``fp16 is True``                    (V100 safety; DESIGN.md §14 risk #1)
      - ``gradient_checkpointing is True``
      - ``per_device_train_batch_size == 1``
      - ``gradient_accumulation_steps == (4 if num_generations == 8 else 8)``  (§7b)
      - ``num_generations in {4, 8}``
      - ``num_generations * gradient_accumulation_steps == 32``  (constant effective rollouts/update)
      - ``warmup_ratio == (0.1 if stage == 1 else 0.0)``  (§3.5)
      - ``beta == 0.04``                    (KL coefficient)
      - ``max_prompt_length == 1024``
      - ``max_completion_length == 2048``
      - ``report_to == "wandb"``
      - ``run_name == f"driftcall-stage{stage}"``
    """
```

### 2.2 Rollout wiring (in-process, no HTTP)

```python
class EpisodeSampler(Protocol):
    """
    Draws a task brief for a single prompt slot in the GRPO batch.
    One EpisodeSampler call → one GoalSpec → G parallel rollouts sharing
    that goal (the "group" in GRPO).
    """
    def __call__(self, step: int) -> "GoalSpec": ...


class EpisodeDatasetAdapter:
    """
    Streaming dataset source wired into TRL's ``GRPOTrainer.train_dataset``.

    This is NOT a traditional epoch-style dataset — it is an iterable that
    yields exactly one record per GRPO training step. Each record packages:
      - ``prompt``: the serialized initial observation (system prompt + goal
        brief + available_tools), already rendered via
        ``tokenizer.apply_chat_template(add_generation_prompt=True)`` as per §3.2.1.
      - ``_meta``: a dict carrying the ``GoalSpec``, the per-group
        ``episode_seed`` (derived from ``stage_base_seed + step``), the stage
        index, and the ``language_weights`` — every scalar required by the
        rollout controller + ``reward_fn`` to reconstruct the episode.

    Constructor:

        EpisodeDatasetAdapter(
            task_gen: TaskGenerator,                      # driftcall.task_generator module
            env_factory: Callable[[], "DriftCallEnv"],    # fresh env per rollout
            stage: Literal[1, 2, 3],
            stage_base_seed: int,                         # e.g. 1_000_000 * stage
            language_weights: dict[LanguageCode, float],
        )

    Iteration contract (``__iter__``):

        for step in itertools.count():
            goal = task_gen.generate(
                seed=stage_base_seed + step,
                stage=stage,
                language_weights=language_weights,
            )
            prompt_str = render_initial_prompt(goal, tokenizer)   # §3.2.1
            yield {
                "prompt": prompt_str,
                "_meta": {
                    "goal":             goal,                    # GoalSpec (frozen)
                    "episode_seed":     stage_base_seed + step,  # → env.reset() seed
                    "stage":            stage,
                    "language_weights": language_weights,
                },
            }

    The adapter is STATELESS beyond the monotonically-increasing ``step``
    counter; ``task_generator.generate`` is itself deterministic (blake2b
    sub-seeds off ``(seed, decision_tag)`` per task_generator.md §1), so
    resuming from a checkpoint simply resumes the step counter — no RNG
    state to save for the dataset itself (see §3.6 resume determinism).
    """


def rollout_group(
    model: "FastModel",
    tokenizer,
    goal: "GoalSpec",
    episode_seed: int,
    num_generations: int,
    env_factory: "Callable[[], DriftCallEnv]",
) -> "RolloutBatch":
    """
    Produce G=num_generations independent episode rollouts sharing the same goal.

    - For each of the G rollouts: instantiate a fresh env via ``env_factory()``,
      call ``env.reset(seed=derive_seed(episode_seed, g_index))`` so
      ``drift_schedule`` is fixed per-rollout (group variance comes from policy
      sampling, not env noise; DESIGN.md §6.2, §7.4).
    - Run the multi-turn loop: observation → serialize via §3.2.1 chat template
      → model.generate(...) → parse DriftCallAction → env.step → next observation.
    - On termination, the env calls ``compute_rewards`` and attaches the
      ``Rewards`` blob to the terminal observation's ``info`` dict.
    - Collect (trajectory_tokens, logprobs, reward, Rewards.breakdown) per rollout
      into a ``RolloutBatch`` (§4.1). Note: per-rollout ``RolloutStep`` sequences
      can have **different lengths** (some episodes SUBMIT at turn 3, others time
      out at ``max_turns``); only the scalar ``reward_scalar`` is group-normalized
      downstream. See §3.2.2 for uneven-termination padding and masking.

    :raises EpisodeParseError: model output does not yield a valid DriftCallAction
                               JSON (counted as format violation in R4, episode
                               continues up to ``max_turns`` or forced ABORT).
    """
```

### 2.3 Reward wire-up — plumbing via `_meta` (TRL 0.23 compatible)

TRL 0.23's `GRPOTrainer` calls `reward_funcs[i](prompts, completions, **kwargs)` with every non-standard dataset column forwarded as a kwarg. Because GRPO rollouts in DriftCall are **multi-turn** (not single-prompt / single-completion), the standard single-shot reward plumbing does not fit; we adopt a **custom `GRPOTrainer` subclass, `DriftCallGRPOTrainer`**, that owns the rollout loop and passes the terminal `Episode` directly into `reward_fn` (see §3.2.3 for the subclass).

```python
def reward_fn(
    prompts: list[str],
    completions: list[str],
    *,
    _meta: list[dict],                  # forwarded from EpisodeDatasetAdapter
    episodes: list["Episode"],          # injected by DriftCallGRPOTrainer after rollout
    **kwargs,
) -> list[float]:
    """
    TRL 0.23-compatible reward function, called once per GRPO step with the
    full group of G=num_generations rollouts batched along axis 0.

    Signature contract (TRL 0.23+):
      - ``prompts``      : list[str], len == G, the serialized initial prompts
      - ``completions``  : list[str], len == G, the raw text generated across
                           the multi-turn loop (assistant turns concatenated)
      - ``_meta``        : list[dict], len == G, carries goal + episode_seed
      - ``episodes``     : list[Episode], len == G, the frozen terminal episodes
                           (DriftCallGRPOTrainer injects this after rollout)
      - Returns          : list[float], len == G, each in [0, 1] rounded to 3dp

    Pure-functional on the episode trace (no state leak across rollouts):
      1. For each i in range(G): pull the terminal ``Episode`` from ``episodes[i]``.
      2. Call ``rewards.compute_rewards(episode)`` — pure, deterministic
         (rewards.md §3.1 "no RNG, no clock, no I/O").
      3. Return ``[r.reward for r in rewards_list]`` — 3-decimal floats in [0, 1].

    The per-reward components R1..R5 and ``Rewards.breakdown`` are logged to
    WandB via the training callback (§3.4). They are NOT returned to GRPO.
    GRPO sees one scalar per rollout; group-relative advantage normalization
    over G happens inside TRL and MUST NOT be pre-applied here (DESIGN.md §7.4).

    Alternative rejected: feeding ``episodes`` via ``_meta`` — would force
    JSON-serializing the frozen ``Episode`` dataclass through TRL's dataset
    collator, losing type safety. Custom trainer subclass is cleaner.
    """
```

### 2.4 `GRPOConfig` builder (verbatim DESIGN.md §10.2)

```python
def build_grpo_config(
    stage: Literal[1, 2, 3],
    *,
    num_generations: int = 8,
    resume_output_dir: Path | None = None,
) -> GRPOConfig:
    from trl import GRPOConfig

    # Warmup flows across stages (see §3.5): only Stage 1 warms the LR.
    warmup_ratio = 0.1 if stage == 1 else 0.0

    # Gradient accumulation compensates for G=4 fallback so the effective
    # rollouts-per-update stays at 32 (see §7b).
    grad_accum = 8 if num_generations == 4 else 4

    return GRPOConfig(
        # Optimizer
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,              # 0.1 on Stage 1, 0.0 on Stage 2/3 (§3.5)
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        # Batch topology (V100 32GB)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum, # 4 @ G=8, 8 @ G=4 → 32 rollouts/update (§7b)

        # GRPO group
        num_generations=num_generations,        # 8 default; 4 on OOM fallback
        max_prompt_length=1024,
        max_completion_length=2048,

        # KL
        beta=0.04,
        use_bias_correction_kl=True,            # MANDATORY — TRL issue #4637

        # Sampling
        temperature=0.9,
        top_p=0.95,

        # Precision — V100-safe (FP16; do NOT flip to bf16)
        fp16=True,
        gradient_checkpointing=True,

        # Logging + checkpoints
        logging_steps=5,
        save_steps=50,
        save_total_limit=10,
        output_dir=str(resume_output_dir or f"checkpoints/stage{stage}"),

        # Monitoring
        report_to="wandb",                      # offline-safe — see §2.4.1 + §6.1
        run_name=f"driftcall-stage{stage}",
    )
```

### 2.4.1 WandB offline-safe logging

`report_to="wandb"` is the primary monitoring surface, but the training run must survive a flaky network (Scaler School of Technology venue WiFi, DESIGN.md §3.5). Belt-and-braces:

- **Setup prereq:** `wandb login <api_key>` is a one-time setup step run before `train()` (documented in the run-book, Batch C5). Missing API key surfaces as `WandBStartupError` at `train()` entry **only when** `WANDB_MODE != "offline"` AND the initial `wandb.init()` call fails.
- **Offline mode:** when `os.environ.get("WANDB_MODE") == "offline"`, WandB runs write to `./wandb/offline-run-*/` and sync later (`wandb sync`). No network required. No `WandBStartupError`.
- **Runtime resilience:** wandb upload failures mid-run (network drop, 5xx) are **non-fatal** — TRL's WandB callback swallows them and retries, and we add a **custom `LocalCSVCallback(TrainerCallback)`** that mirrors every `logs` dict from `on_log` to `checkpoints/<run>/metrics.csv`. One row per `logging_steps=5` tick, all 20 monitoring columns (§3.4). This CSV is the authoritative record; WandB is best-effort.
- **CSV schema:** stable column order matching §3.4 columns 1–20; first row is the header; `step` as the first column; numeric values as floats (not strings); NaN encoded as `"nan"`. Append-mode; one file per run; never rotated.
- **Error-raising rule:** `WandBStartupError` raises ONLY if (a) `WANDB_MODE != "offline"` AND (b) initial `wandb.init()` raises. All post-init wandb errors are warnings. All missing-api-key errors are warnings if `WANDB_MODE == "offline"`.

---

## 3. Behavior Spec

### 3.1 Model + adapter construction (DESIGN.md §10.1)

```python
from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=torch.float16,             # explicit FP16 for V100 safety
)

model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

**Invariants:**
- `model.config.torch_dtype == torch.float16` on V100 (checked at load; raise if BF16 slipped through).
- Base weights are 4-bit NF4; only LoRA adapters (r=16, α=32, 7 target modules) are trainable.
- Gemma 4 E2B context is 128K (DESIGN.md §0 TL;DR); we use 4096 here — enough for a 6-turn episode with tool-call JSON payloads.

**BF16-slippage assertion (runs at `train()` entry, before any rollout):**

```python
# Immediately after FastModel.from_pretrained(..., dtype=torch.float16)
_first_param_dtype = next(model.parameters()).dtype
assert _first_param_dtype == torch.float16, (
    f"BF16 slipped through: V100 unsafe. "
    f"next(model.parameters()).dtype == {_first_param_dtype}, expected torch.float16. "
    f"Root cause: Unsloth auto-picked BF16 despite dtype=torch.float16 kwarg. "
    f"Halt training; do NOT proceed on V100."
)
```

The assertion fires on any trainable or non-trainable parameter returning `torch.bfloat16`. This is a hard halt — V100 (sm_70) has no BF16 tensor cores, and silently running BF16 via software emulation causes ~10× slowdown AND the numerical-instability patterns catalogued in §7a. The check runs **once**, at `train()` entry, before `GRPOTrainer` is constructed, before any optimizer state is built — so the assertion message fires cleanly on misconfiguration.

### 3.2 Episode rollout semantics (DESIGN.md §3.2, §7.4)

- **Group size G = 8 by default.** G=4 is the documented OOM-fallback value (§7b, DESIGN.md §3.2). No other G values are supported.
- **One `GoalSpec` per group, G rollouts share it.** This is GRPO's "group": the advantage is normalized relative to the mean reward *within* the group, so the group must share a prompt (DESIGN.md §7.4). `task_generator.generate(seed, stage, language_weights)` is called **once per group**; the returned `GoalSpec` is passed to all G `env.reset()` calls in that group.
- **Env determinism inside a group.** Each of the G rollouts instantiates its own `DriftCallEnv` and calls `env.reset(seed=derived_seed(goal, g_index))`. Because the drift schedule is itself seeded (DESIGN.md §6.2), the drift timing is fixed per-g-index but **variance across the group comes from policy sampling** at `temperature=0.9, top_p=0.95`. This is the correct GRPO signal: same problem, G different attempts.
- **Multi-turn rollout loop** (per rollout):
  1. `obs = env.reset(seed=…)`.
  2. Loop while `not obs.done`:
     - Serialize `obs` to the model prompt (tool-use chat template; system prompt declares available tools per `obs.available_tools`).
     - `tokens = model.generate(..., max_new_tokens=max_completion_length)`.
     - Parse assistant output as a `DriftCallAction` JSON (see §5 `EpisodeParseError`).
     - `obs = env.step(action)`.
     - If `obs.done`: collect the terminal `Rewards` blob from `obs.info["rewards"]` (env populates this via `compute_rewards(episode)` at termination — rewards.md §6.2).
  3. Emit `RolloutBatch` row.
- **No ASR/TTS in the loop.** `obs.last_transcript` is the pre-authored text from the task brief (`goal.seed_utterance`) for turn 0, and any subsequent `CLARIFY` replies come from a scripted simulated user (text-only). Audio pipeline is strictly env-boundary for deploy/demo (DESIGN.md §9.4, CLAUDE.md §9).
- **Reward is pure-functional on the episode trace.** `reward_fn` (§2.3) calls `rewards.compute_rewards(episode)` which is strictly deterministic, idempotent, no-state-leak (rewards.md §3.1). **No reward normalization pre-GRPO** — raw [0, 1] reward in; GRPO does group-relative internally (DESIGN.md §7.4).

### 3.2.1 Prompt serialization (Gemma 4 chat template, pinned)

Every observation-to-prompt rendering goes through exactly one code path. No ad-hoc string concatenation.

**Tokenizer call:**

```python
prompt_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

`messages` is a list of `{"role": "system" | "user" | "assistant" | "tool", "content": str}` dicts. `add_generation_prompt=True` appends the Gemma 4 assistant turn-start tokens so the model knows to emit the next assistant message.

**System prompt (verbatim, pinned):**

```
You are a concierge assistant. Use the provided tools. Respond in the caller's language. Submit with calibrated confidence.
```

This string is frozen in `training/prompts.py::SYSTEM_PROMPT` and asserted at `train()` entry. Do not paraphrase, do not extend, do not localize — it is part of the reproducibility contract (`CheckpointMeta.config_sha256` hashes this).

**Observation → message serialization (turn by turn):**

| Observation field | Message role | Content format |
|---|---|---|
| `obs.goal.seed_utterance` (turn 0 only) | `user` | Raw NFC-normalized string |
| `obs.available_tools` | embedded in `system` | Appended after the pinned system prompt as: `Available tools: <JSON array of tool schemas>` |
| `obs.tool_results` (all accumulated) | `tool` (one msg per entry) | `json.dumps({"tool": t.name, "args": t.args, "status": t.status, "response": t.response}, ensure_ascii=False, sort_keys=True)` |
| `obs.drift_log` (all accumulated) | appended to latest `user` | `json.dumps([{"turn": d.turn, "type": d.type, "domain": d.domain, "description": d.description} for d in obs.drift_log], ensure_ascii=False, sort_keys=True)` (empty list if no drifts) |
| `obs.last_transcript`, `obs.last_lang` (post-clarify) | `user` | `f"[lang={obs.last_lang}] {obs.last_transcript}"` |

`sort_keys=True` and `ensure_ascii=False` are mandatory — they make the serialization byte-stable across platforms and preserve Devanagari/Tamil/Kannada script without escaping.

**Conversation accumulation & truncation:**

- Maintain the full message history across all turns of a single rollout; do NOT re-render from scratch on every turn.
- Budget: total rendered prompt ≤ `max_prompt_length = 1024` tokens (GRPOConfig invariant).
- **Overflow policy** (applied greedily until under budget):
  1. Drop the **oldest `tool` message first** (FIFO over tool_results).
  2. If still over budget, drop the next-oldest `tool` message. Continue until only the **last 2 turns' tool_results** remain.
  3. The `system` prompt, `goal.seed_utterance` (turn 0 user), and the **last 2 turns** (user + tool results + assistant) are NEVER dropped.
  4. If after all the above the prompt still exceeds 1024 tokens, raise `EpisodeParseError` and terminate the episode — this is a pathological case (e.g., a single tool returning a 2KB JSON response) that should be logged for vendor-response-size tuning.

- On truncation, prepend a single `system` message `f"[truncated {n} older tool_results]"` so the policy knows context was elided.

### 3.2.2 Uneven rollout termination — padding and masking

Within a single GRPO group of G=8 rollouts sharing one `GoalSpec`, episodes do NOT all terminate at the same turn:
- Some rollouts hit `SUBMIT` at turn 3 (short completion_tokens sequence).
- Others burn through to `max_turns` (long completion_tokens sequence).
- Others hit `ABORT` at turn 5 mid-episode.

Group-relative advantage normalization is a **scalar** operation — it consumes one `reward_scalar` per rollout. It does NOT depend on trajectory length. So the fundamental GRPO math is unaffected by uneven termination. What DOES need handling is the **tokenization of the rollout for logprob + KL computation**.

**Padding rule:**
1. Within a group, let `L_max` = max completion_tokens length across the G rollouts.
2. For each rollout with length `L_i < L_max`, right-pad `completion_tokens` with `tokenizer.pad_token_id` up to `L_max`. (Gemma 4's tokenizer has a dedicated pad token; if absent, fall back to `tokenizer.eos_token_id` and assert existence at `train()` entry.)
3. Construct a `completion_mask` of shape `(G, L_max)`: `1` where the original token exists, `0` where it is padding.
4. Pass `completion_mask` into TRL's logprob + KL reduction — TRL 0.23's `GRPOTrainer` already supports this via its internal `completion_mask` handling; we rely on that, we do NOT roll our own KL reducer.

**Reward invariance:** `rewards.compute_rewards(episode)` receives the **frozen `Episode`** dataclass, not the tokenized trace. It does not see padding. Short rollouts and long rollouts feed their `reward_scalar` into group-relative normalization on equal footing — no length penalty, no length bonus.

**`RolloutStep` length variance:** the `Trajectory.steps` tuple in §4.1 has variable length per-rollout (the number of turns actually taken). `reward_scalar` is the only per-rollout quantity that feeds GRPO; the `steps` tuple is purely diagnostic (WandB gen-length logging + reward-hacking probe).

**`train/gen_length_mean` metric clarification:** §3.4 column 4 is computed over **unpadded** completion_tokens (average of `L_i` across the group), not `L_max`. Padding is purely a batching device; the metric tracks real policy behavior.

### 3.2.3 Custom trainer subclass — `DriftCallGRPOTrainer`

Standard TRL `GRPOTrainer.compute_reward` assumes single-prompt / single-completion rollouts, where `model.generate` produces exactly one completion from one prompt. DriftCall rollouts are **multi-turn** — the policy interleaves with `env.step` — so we subclass:

```python
from trl import GRPOTrainer

class DriftCallGRPOTrainer(GRPOTrainer):
    """
    Overrides the rollout phase to run multi-turn episodes via DriftCallEnv,
    then hands the terminal Episode blobs to reward_fn.

    Flow per training step:
      1. Pull one batch row from EpisodeDatasetAdapter (§2.2) — yields
         {"prompt": ..., "_meta": {"goal": ..., "episode_seed": ..., ...}}.
      2. Call self.rollout_group(model, tokenizer, goal, episode_seed, G) — §2.2.
         This runs G parallel multi-turn episodes, each with its own fresh env.
      3. Collect G terminal Episodes + G completion strings (concatenated
         assistant turns).
      4. Call reward_fn(prompts=[prompt]*G, completions=..., _meta=[meta]*G,
                        episodes=episodes) — §2.3.
      5. Feed the G reward scalars + (padded, masked) completion_tokens into
         the standard GRPO advantage + KL computation (inherited, unchanged).
    """

    def _generate_and_score_completions(self, inputs):
        # Override TRL's default generate path with our multi-turn rollout.
        ...
```

This is ~200 lines of glue; the advantage + KL + optimizer step are inherited unchanged from `GRPOTrainer`. **The inherited code path is what must stay untouched** — we only replace the "how completions are produced" phase, never the "how advantages are computed" phase.

### 3.3 GRPO advantage and KL (DESIGN.md §10.2, §14 risk #2)

- **Advantage is group-relative, NOT batch-relative.** GRPOTrainer computes `A_i = (r_i - mean(r_group)) / (std(r_group) + eps)` over the G rollouts that share one goal. The global batch across prompts is NEVER normalized (would destroy the group signal). This is TRL's default; we assert `config.num_generations > 1` and do not override the advantage estimator.
- **KL estimation uses `use_bias_correction_kl=True`.** Per TRL issue #4637, the naive KL estimator is biased for sparse-reward GRPO and can drive `policy_kl → ∞` within 50 steps; the bias-corrected form is mandatory. Config assertion in `build_grpo_config` rejects `False`.
- **KL coefficient `beta = 0.04`.** DESIGN.md §10.2. Monitored per §3.4.

### 3.3.1 Adaptive KL controller (AdaptiveKLCallback)

`beta = 0.04` is the **initial** coefficient — not a frozen invariant. A proportional controller adjusts it every `logging_steps=5` tick so the measured `train/policy_kl` tracks `target_kl` (default `BETA_KL = 0.04`). This keeps policy drift inside a narrow band across all three curriculum stages without operator intervention.

**Update rule** (log-space, symmetric):

```
err      = (kl - target_kl) / target_kl
new_beta = beta * exp(kp * err)
new_beta = clamp(new_beta, beta_min, beta_max)
```

- `kp = 2.0` — proportional gain (multiplicative step in log-space).
- `beta_min = 0.001`, `beta_max = 1.0` — hard clamps prevent collapse (β→0 lets the policy drift unboundedly) and over-anchoring (β→∞ freezes the policy).
- The controller is a **no-op** when `logs` is `None`, `"kl"` is missing, or the value is non-numeric / NaN / ±∞. No exceptions propagate from `on_log`.

**Wiring.** `DriftCallGRPOTrainer` adds an `AdaptiveKLCallback` by default. The callback attaches through `GRPOTrainer.add_callback` so `on_log(args, state, control, logs=...)` fires inside TRL's standard callback dispatch and mutates `args.beta` in-place — the next GRPO loss term picks up the new coefficient automatically.

**Escape hatches** (all at `DriftCallGRPOTrainer.__init__`):

| kwarg | default | purpose |
|---|---|---|
| `enable_adaptive_kl` | `True` | Opt out (reverts to constant `beta = BETA_KL`). |
| `adaptive_kl_target` | `BETA_KL` | Override target KL. |
| `adaptive_kl_kp` | `2.0` | Proportional gain. |
| `adaptive_kl_beta_min` | `0.001` | Lower clamp. |
| `adaptive_kl_beta_max` | `1.0` | Upper clamp. |

**Why proportional (not PI / PID).** Stage-3 runs only ~150 steps at `logging_steps=5` → ~30 controller ticks. An integral term overshoots in fewer than 5 ticks on a short horizon; pure P converges within ~10 ticks for ±100% KL error (verified in `test_monotonic_increase_toward_stable_point`, `test_integration_simulated_50_steps`). If longer runs ever ship, revisit.

**Why not touch `use_bias_correction_kl`.** §3.3 requires `use_bias_correction_kl=True` unconditionally (TRL issue #4637). The adaptive controller rides on top of that — it retargets the coefficient on an already-bias-corrected estimator.

### 3.3.2 Hardware mode (V100 default, H100 optional)

`build_grpo_config(stage, *, hardware="v100")` accepts two hardware modes; the V100 path is the bit-identical default that every existing test exercises, the H100 path is an opt-in for teams with sm_90 access.

| Knob | V100 (default) | H100 |
|---|---|---|
| `fp16` / `bf16` | `fp16=True`, `bf16=False` | `fp16=False`, `bf16=True` |
| `optim` | `paged_adamw_8bit` | `adamw_torch_fused` |
| `attn_implementation` | *(unset, uses model default)* | `flash_attention_3` |
| LoRA dtype assertion (`step_12`) | `torch.float16` via `assert_fp16_dtype` | **N/A**: `boot_gemma` runs the dtype check only under the V100 configuration. An H100 boot helper can skip it once the caller confirms BF16 is safe. |

**Invariants preserved across modes.** `beta`, `num_generations`, `gradient_accumulation_steps`, `use_bias_correction_kl`, `gradient_checkpointing`, `max_prompt_length`, `max_completion_length`, `warmup_ratio` are identical on both paths. See `test_hardware_h100_invariants_still_hold`.

**`assert_config_invariants` hardware inference.** When the `hardware=` kwarg is omitted (V100-era callers), the checker infers mode from `config.bf16`: truthy → H100 rules, else → V100 rules. This keeps legacy callers wire-compatible with no changes.

**`LORA_DROPOUT = 0.05`** (step_12). LoRA dropout was silently zero before; 0.05 matches `unsloth/gemma-4-E2B-it` reference notebooks and reduces small-run overfitting on the 500-step curriculum. Threaded through `BootConfig.lora_dropout` → `FastModel.get_peft_model(lora_dropout=...)`.

### 3.4 Monitoring — WandB columns (DESIGN.md §10.4)

The training callback logs the following **13 + 5 = 18 columns** per `logging_steps=5` tick (DESIGN.md §10.4 enumerates the first 13; per-language is a separate 5-column group):

**Core (7):**
1. `train/reward_mean`
2. `train/reward_std`
3. `train/policy_kl` — mean per-token KL vs reference (clipped to 10.0 in alerts; §7c)
4. `train/gen_length_mean` — mean completion token length
5. `train/grad_norm` — L2 over all trainable params (NaN/inf watchdog; §7c)
6. `train/loss`
7. `train/learning_rate`

**Per-reward (5):**
8. `train/R1_mean` — task completion
9. `train/R2_mean` — drift detection
10. `train/R3_mean` — constraint adherence
11. `train/R4_mean` — format compliance
12. `train/R5_mean` — anti-hack penalty (≤ 0)

**Reward-hacking probe (3):**
13. `train/drift_detected_rate` — fraction of episodes with R2 == 1.0 at stage ≥ 2
14. `train/format_compliance_rate` — fraction with R4 == 1.0
15. `train/hallucinated_field_count` — sum of R5 branch-(a) trips per logging window

**Per-language (5, DESIGN.md §10.4 bullet 8):**
16. `train/reward_hi`
17. `train/reward_ta`
18. `train/reward_kn`
19. `train/reward_en`
20. `train/reward_hinglish`

**Total monitoring columns = 20.** (The spec says "at least 18" — DESIGN.md §10.4 lists ~13 items plus the per-language breakdown group; we explicitly enumerate 20 so critics can verify coverage.)

**Completion sampling:** every 25 steps, log 3 random completions verbatim to WandB's `Table`. Human-readable by Person B for reward-hacking inspection (DESIGN.md §10.4 "Inspection" line).

### 3.5 Three-stage curriculum (DESIGN.md §10.3)

| Stage | Steps | Env config | Language mix | Goal |
|---|---|---|---|---|
| **1 Warmup** | 150 | `curriculum_stage=1` — no drift | `{en: 0.50, hinglish: 0.30, hi: 0.20, ta: 0.0, kn: 0.0}` | Learn tool use + R4 format |
| **2 Single-drift** | 200 | `curriculum_stage=2` — 1 drift per episode | `{en: 0.30, hinglish: 0.30, hi: 0.20, ta: 0.10, kn: 0.10}` | Learn drift detection (R2) |
| **3 Compound** | 150 | `curriculum_stage=3` — 2 drifts per episode | Same as Stage 2 | Learn cascading recovery |

Total: **500 GRPO steps × G=8 × ~6 turns ≈ 24,000 trajectories** (DESIGN.md §10.3). Fits in 30h V100 wall-clock per DESIGN.md §3.5 budget.

**Stage transitions:**
- Stage 1 exit criterion: `train/R1_mean ≥ 0.4` at step 100 (else escalate per CLAUDE.md §11).
- Stage 2 kicks off by `train(stage=2, num_steps=200, resume_from=Path("checkpoints/stage1_final"))`.
- Stage 3 kicks off by `train(stage=3, num_steps=150, resume_from=Path("checkpoints/stage2_final"))`.

**Stage-transition learning-rate behavior (resolved choice: CONTINUE cosine across all 500 steps):**

We pick **option (b) — one continuous cosine schedule flowing across Stage 1 → Stage 2 → Stage 3**, so the LR never re-warms mid-curriculum. The curve's total span is **500 steps** (150 + 200 + 150), with the 10% warmup applied ONLY in Stage 1 (the first 50 steps).

Mechanism:
- When `resume_from` is provided, `trainer.train()` loads `trainer_state.json` from the prior-stage checkpoint; this restores the global_step counter AND the LR scheduler state (HF Trainer convention).
- `build_grpo_config(stage=N)` is called fresh per stage ONLY to update `num_steps` (`max_steps` equivalent), `output_dir`, and `run_name`. The other fields are held constant.
- **Override: for Stage 2 and Stage 3 launches, `build_grpo_config` must pass `warmup_ratio=0.0`** — otherwise the fresh config would apply a 10%-of-stage-steps re-warmup on top of the restored scheduler, producing a "double warmup" artifact (LR drops to zero, climbs again mid-curriculum, destroys continuity).
- `build_grpo_config(stage, warmup_ratio=None)` default: `warmup_ratio=0.1 if stage == 1 else 0.0`. Assertion: `assert config.warmup_ratio == (0.1 if stage == 1 else 0.0)` inside `__post_init__`.
- Net effect: one cosine curve, `lr = 5e-6` at step 0 warming to peak `5e-6` at step 50, decaying monotonically to ~`0` at step 500. The dashboard shows three WandB runs (one per stage) but the LR trace concatenates smoothly.

Rejected alternative (a): "each stage is a fresh scheduler, accept the jagged LR." This produces three sawtooth LR curves and typically costs 5–10% in final reward because the policy sees its LR reset to peak mid-curriculum — avoidable. We only fall back to (a) if (b) proves buggy on the onsite V100 (e.g., trainer_state.json load fails); in that case flip `warmup_ratio` default back to `0.1` for all stages and log the regression to WandB as `train/scheduler_mode=jagged`.

### 3.6 Checkpoint saving — do NOT merge 4-bit naively (DESIGN.md §10.5, CLAUDE.md §9)

**Correct paths (three use-cases):**

```python
# (a) Periodic + final adapter-only save for re-training / resume.
model.save_pretrained(
    "checkpoints/stage{N}_final",
    safe_serialization=True,
)
tokenizer.save_pretrained("checkpoints/stage{N}_final")

# (b) HF Hub push (adapter-only; base stays 4-bit on Hub).
model.push_to_hub(
    "<team>/gemma-4-e2b-driftcall-lora",
    safe_serialization=True,
)

# (c) DEMO-ONLY merged 16-bit — NEVER load this for re-training.
#     Allowed because the demo Space uses ZeroGPU/A10G where 16-bit fits.
model.save_pretrained_merged(
    "checkpoints/merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)
```

**Prohibited:** `model.merge_and_unload()` after 4-bit loading, or any pattern that calls `dequantize → merge LoRA → requantize`. Per DESIGN.md §10.5 and CLAUDE.md §9 the requantization loss is severe and silent. If you must resume training, use (a) — adapters + 4-bit base are the resume artifact.

**`resume_from` behavior:**
- `resume_from` is a directory produced by path (a). `train()` calls `FastModel.from_pretrained(base)`, then `PeftModel.from_pretrained(model, resume_from)` to restore adapters, then `GRPOTrainer(...)` reloads optimizer + scheduler state from `resume_from/trainer_state.json` (HF convention).
- If `resume_from` exists but `adapter_model.safetensors` is missing or corrupt, raise `CheckpointIOError` (§5, §7e). Do NOT silently start fresh.

**Resume determinism — RNG state handling (mandatory):**

For bit-reproducible resume, every RNG consulted during training must be checkpointed alongside the adapter. The save path writes `rng_states.pt` into the same directory as the adapter:

```python
import torch, numpy as np, random

rng_states = {
    "torch_cpu":        torch.get_rng_state(),
    "torch_cuda":       torch.cuda.get_rng_state_all(),      # per-device
    "numpy":            np.random.get_state(),
    "python_random":    random.getstate(),
    # trl/transformers internal sampling RNG is covered by trainer.save_state()
    # which writes rng_state.pth alongside trainer_state.json — we keep that
    # standard path and add our own rng_states.pt for belt-and-braces.
}
torch.save(rng_states, Path(resume_from) / "rng_states.pt")
```

On resume, **before any rollout, before any optimizer step**, restore all four:

```python
rs = torch.load(Path(resume_from) / "rng_states.pt", weights_only=True)
torch.set_rng_state(rs["torch_cpu"])
torch.cuda.set_rng_state_all(rs["torch_cuda"])
np.random.set_state(rs["numpy"])
random.setstate(rs["python_random"])
# trainer.train(resume_from_checkpoint=...) restores TRL-internal RNG automatically.
```

**Sources of randomness, exhaustively enumerated:**

| Source | How saved | How restored |
|---|---|---|
| `torch` CPU RNG (sampling in generate) | `torch.get_rng_state()` | `torch.set_rng_state(...)` |
| `torch` CUDA RNG (GPU sampling) | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(...)` |
| `numpy.random` (incidental, e.g., bootstrap CIs at eval) | `np.random.get_state()` | `np.random.set_state(...)` |
| Python `random` (unsloth + peft internals, shuffling) | `random.getstate()` | `random.setstate(...)` |
| TRL / transformers sampling RNG | `trainer.save_state()` → `rng_state.pth` | `trainer.train(resume_from_checkpoint=...)` auto |
| `task_generator.generate` | **STATELESS** — blake2b derivation from `(stage_base_seed + step)` | No save needed; reconstructed from step counter |
| `drift_injector` schedule | STATELESS — seeded from `episode_seed` per task_generator.md §1 | No save needed |
| `DriftCallEnv.reset(seed=...)` | STATELESS per-call | No save needed |

The STATELESS sources (task_generator, drift_injector, env.reset) are reproducible **by construction** from the step counter — that's exactly why `EpisodeDatasetAdapter` stores no RNG of its own. The STATEFUL sources (torch, numpy, python-random, TRL-internal) all get checkpointed.

**Save-site integration:** `rng_states.pt` is written at the same moment as `adapter_model.safetensors` — either via a TRL `TrainerCallback.on_save(args, state, control, ...)` hook, or by wrapping `model.save_pretrained` in a helper that also calls `torch.save(rng_states, ...)`. Missing `rng_states.pt` on resume is a soft warning (not a halt) — `train()` falls back to seeding `random.seed(config_sha256_as_int)` and logs `train/rng_restore_fallback=1`; this preserves reproducibility at the `(stage_base_seed, step_counter)` level even if per-step determinism is lost.

---

## 4. Data Structures

All dataclasses frozen. `from __future__ import annotations` on every file.

### 4.1 `RolloutBatch` (internal, produced by `rollout_group`)

```python
@dataclass(frozen=True)
class RolloutStep:
    turn: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    completion_logprobs: tuple[float, ...]     # per-token logprobs under πθ
    ref_logprobs: tuple[float, ...]            # per-token logprobs under frozen base (for KL)
    parse_ok: bool                             # False if EpisodeParseError caught + ABORT

@dataclass(frozen=True)
class Trajectory:
    rollout_id: str                            # f"{goal.episode_id}:g{g_index}"
    goal: "GoalSpec"
    steps: tuple[RolloutStep, ...]
    terminal_episode: "Episode"                # frozen — fed to compute_rewards
    rewards: "Rewards"                         # result of compute_rewards(episode)

@dataclass(frozen=True)
class RolloutBatch:
    group_id: str                              # one per prompt slot per GRPO step
    goal: "GoalSpec"
    trajectories: tuple[Trajectory, ...]       # len == num_generations ∈ {4, 8}
    reward_scalars: tuple[float, ...]          # just Rewards.reward, in trajectory order
```

### 4.2 `EvalReport` (returned by `eval`)

```python
@dataclass(frozen=True)
class PerLanguageReport:
    language: Literal["hi", "ta", "kn", "en", "hinglish"]
    n_episodes: int
    reward_mean: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    r4_mean: float
    r5_mean: float

@dataclass(frozen=True)
class DriftDetectionLatency:
    """
    For each Stage-2/3 episode with R2 == 1.0, latency = (first turn where ANY
    R2 branch hit the drift) - (drift.turn). Reported as mean/median/p95 over
    all detected drifts, plus per-stage breakdown.
    """
    stage2_mean: float
    stage2_median: float
    stage2_p95: float
    stage3_mean: float
    stage3_median: float
    stage3_p95: float
    undetected_count: int                      # R2 == 0.0 drifts, excluded from latency stats

@dataclass(frozen=True)
class EvalReport:
    model_path: str                            # "base" or absolute checkpoint path
    n_episodes: int                            # default 50 (DESIGN.md §12.2 baseline gate)
    # Per-reward means with 95% bootstrap CI (tuple = (mean, lo, hi))
    reward_mean_ci: tuple[float, float, float]
    r1_mean_ci: tuple[float, float, float]
    r2_mean_ci: tuple[float, float, float]
    r3_mean_ci: tuple[float, float, float]
    r4_mean_ci: tuple[float, float, float]
    r5_mean_ci: tuple[float, float, float]
    brier_mean: float
    floor_applied_rate: float                  # fraction of episodes where uncertain floor fired
    hallucinated_field_rate: float             # R5 branch-(a) trigger rate
    reward_hacking_offenses: dict[str, int]    # {"hallucinated_field": 3, "probe_abuse": 1, ...}
    drift_detection_latency: DriftDetectionLatency
    per_language: tuple[PerLanguageReport, ...]
    curves: dict[str, tuple[tuple[int, float], ...]]
        # {"reward_vs_step": ((0, 0.18), (50, 0.31), ...), "R1_vs_step": (...), ...}
        # Used to render the 3-plot panel in DESIGN.md §15 pitch 1:00–2:00.
```

### 4.3 `CheckpointMeta` (sidecar JSON next to `adapter_model.safetensors`)

```python
@dataclass(frozen=True)
class CheckpointMeta:
    stage: Literal[1, 2, 3]
    steps_completed: int                       # absolute step count from stage start
    cumulative_steps: int                      # across stages (for WandB resume)
    wall_clock_seconds: float
    reward_mean_at_save: float
    base_model_id: str                         # "unsloth/gemma-4-E2B-it-bnb-4bit"
    unsloth_version: str                       # e.g. "2026.4.5"
    trl_version: str                           # e.g. "0.23.1"
    torch_version: str                         # e.g. "2.5.1"
    git_sha: str                               # training repo commit
    config_sha256: str                         # sha256 of GRPOConfig __repr__ — reproducibility key
```

Serialized to `checkpoints/stage{N}_final/driftcall_meta.json` alongside the HF standard files.

---

## 5. Error Modes

All training-specific exceptions subclass `TrainingError(Exception)`.

| Exception | Trigger | Handling |
|---|---|---|
| `OutOfMemoryError` (wrapped `torch.cuda.OutOfMemoryError`) | CUDA OOM during `model.generate` or backward pass | Catch at group boundary; call `torch.cuda.empty_cache()`; if `num_generations == 8`, retry the group at `num_generations=4` (G=4 fallback, §7b); if already 4 and OOM recurs → raise to user. |
| `NonFiniteGradientError` | `grad_norm` NaN or inf detected in training callback | Skip the update (zero grad, step scheduler); log to `train/skipped_updates`; if > 3 consecutive skips → raise and halt (§7c). |
| `KLDivergenceExplosion` | `train/policy_kl` 10-step moving mean > 10.0 | Halt training; dump WandB run URL + last-known-good checkpoint path; escalate to user per CLAUDE.md §11. Root cause is almost always `use_bias_correction_kl=False` slipping into config — `build_grpo_config` asserts this can't happen, so this firing is a critical regression. |
| `RewardCollapseError` | `train/reward_mean` delta > 0.15 downward within 10 steps AND `train/R5_mean` ≤ -0.3 simultaneously | Almost certainly reward hacking spike on R5 branch (a) (DESIGN.md §14 risk #5). Halt training; surface to Person B for probe inspection (§7d). |
| `CheckpointIOError` | `save_pretrained` raises, disk full, HF Hub 5xx after 3 retries with 2/4/8s backoff, or `adapter_model.safetensors` corrupt on resume | Raise — no fallback; checkpoint integrity is load-bearing for the before/after demo (§7e). |
| `TokenizerMismatchError` | `resume_from` tokenizer vocab size ≠ base tokenizer vocab size | Raise at resume time; refuse to start. Indicates the base model changed under us (e.g., someone upgraded Unsloth pinning a different Gemma 4 revision) — §7 edge case for spec completeness. |
| `EpisodeParseError` | Model output does not yield a valid `DriftCallAction` JSON | Caught **inside** the rollout loop; converted to a no-op action + R4 deduction (format violation); episode continues to next turn. Does NOT escape to the trainer. Logged to `train/episode_parse_failures`. |
| `EvalModelLoadError` | `eval()` cannot load base + adapter from `model_path` | Raise; `eval` never silently falls back to base. |
| `LanguageCohortCollapseError` | Per-language batch for a cohort (e.g. `hi`) is empty for ≥ 20 consecutive steps at `stage ∈ {2, 3}` | Soft warning by default (log `train/cohort_collapse:hi`); hard error if cohort collapses for ≥ 50 steps (§7f). Caused by upstream `language_weights` misconfiguration. |
| `WandBStartupError` | `wandb.init()` fails at `train()` entry AND `os.environ.get("WANDB_MODE") != "offline"` | Raise. If `WANDB_MODE=="offline"`, do not raise — proceed with local CSV only (§2.4.1). Mid-run wandb upload failures are NON-fatal (swallowed with warning) and the local CSV in `checkpoints/<run>/metrics.csv` is the authoritative record. |

**Policy:**
- **Strict on invariants** (GRPOConfig asserts, tokenizer-vocab-size, `use_bias_correction_kl`) — raise immediately.
- **Permissive on content** (a single EpisodeParseError is a format violation, not a crash).
- **Graceful on OOM** — G=8 → G=4 before giving up.
- **Hard stop on KL explosion and reward collapse** — these are training-ruining regressions and we do NOT paper over them.

---

## 6. Dependencies

### 6.1 External (pinned in `requirements.txt`)

- **Unsloth 2026.4.5+** — `FastModel` API for 4-bit Gemma 4 loading + LoRA + `save_pretrained_merged`. Pre-2026.4.5 versions have a KL-estimator bug (Unsloth discussion #4921, DESIGN.md §16.E).
- **TRL 0.23+** — `GRPOTrainer`, `GRPOConfig` with `use_bias_correction_kl` parameter (TRL issue #4637, DESIGN.md §14 risk #2).
- **PyTorch 2.5+** — Flash-Attention-2 via Unsloth, FP16 autocast stability on V100 (sm_70).
- **bitsandbytes** — NF4 4-bit weights (loaded via Unsloth; we do not call bnb directly).
- **peft** — LoRA adapter lifecycle (via Unsloth `FastModel.get_peft_model`).
- **accelerate** — device placement (via Unsloth).
- **wandb** — monitoring (DESIGN.md §10.4); `WANDB_PROJECT=driftcall`, `WANDB_RUN_GROUP=curriculum-v1`. Setup via `wandb login` before first `train()` invocation; runtime failures are non-fatal (§2.4.1). Local CSV at `checkpoints/<run>/metrics.csv` is the authoritative offline-safe mirror.
- **PyYAML** — already pulled by `task_generator` for `templates.yaml` loading.

### 6.2 Internal (DriftCall repo)

- **Reads `driftcall.env.DriftCallEnv`** — in-process, one fresh instance per rollout. Never over HTTP (DESIGN.md §3.2 "Env runs in-process with the trainer").
- **Reads `driftcall.rewards.compute_rewards`** — pure function; called inside `env.step` at termination and re-callable in `eval` to audit an offline trajectory.
- **Reads `driftcall.task_generator.generate`** — called once per GRPO group to produce the shared `GoalSpec`.
- **Reads `driftcall.models`** — `GoalSpec`, `DriftCallAction`, `ActionType`, `Episode`, `Rewards`, `LanguageCode`.
- **Writes to `checkpoints/stage{N}_final/`** — adapter + tokenizer + `driftcall_meta.json`.
- **Writes to HF Hub repo `<team>/gemma-4-e2b-driftcall-lora`** — adapter only, `safe_serialization=True`.

### 6.3 Hardware

- **V100 32GB** — primary training. FP16-only (no BF16). Flash-Attention-2 disabled by Unsloth on sm_70 — we accept the ~15% throughput cost in exchange for numerical stability.
- **No multi-GPU.** DESIGN.md §3.5 single-V100 budget.
- **Disk:** 50GB free required. Each checkpoint ≈ 150MB (adapters only) + optimizer state ≈ 300MB + `rng_states.pt` ≈ 1MB = ~500MB × `save_total_limit=10` = 5GB ceiling.

**GPU memory budget (peak-estimate line-item, V100 32GB):**

| Component | Estimate (GB) | Notes |
|---|---|---|
| Gemma 4 E2B base (4-bit NF4) | 2.0 | 2B params × 0.5 bytes/param + bnb quant metadata |
| LoRA adapters (r=16, α=32, 7 modules) + grads | 0.2 | ~20M trainable params × (FP16 weight + FP16 grad) ≈ 80 MB × 2 |
| Optimizer state (`paged_adamw_8bit`) | 0.3 | 8-bit moments for adapter params; paged to CPU on overflow |
| KV cache (G=8 parallel generations × 4096 seq) | 8.0 | Dominant for generate(); scales linearly in G and seq-len |
| Activations (`gradient_checkpointing=True`) | 6.0 | Checkpoint recompute drops this from ~18 GB unchecked to ~6 GB |
| FP16 autocast + CUDA workspace + fragmentation | 2.0 | Loss-scale buffer, comm buffers, allocator slack |
| **Peak @ G=8 Stage-3 worst case** | **~18.5** | Headroom: 32 − 18.5 = ~13.5 GB safety margin |
| **Peak @ G=4 fallback (§7b)** | **~14.0** | Headroom: 32 − 14 = ~18 GB — comfortable |

These are **upper-bound estimates**; real runs on Stage-1/2 typically peak at ~15–16 GB. The ~13.5 GB headroom on worst-case Stage-3 is what we rely on for stable training without needing gradient offload. Numbers are subject to ±2 GB variance based on Unsloth version and CUDA allocator behavior; treat this as a **planning table**, not a guarantee. If `nvidia-smi` shows sustained > 28 GB during a Stage-3 rollout, trigger the G=4 fallback (§7b) pre-emptively rather than waiting for OOM.

### 6.4 Data sources

- **Task briefs:** `task_generator.generate(seed, stage, language_weights)` — in-process, seeded. No external dataset read during training.
- **Eval briefs:** `val/briefs.jsonl` (DESIGN.md §8.6) — 500 held-out `GoalSpec`s. `eval()` reads the first `episodes` rows (default 50, DESIGN.md §12.2 baseline gate).

### 6.5 Non-dependencies (explicit)

- Does **not** read audio. DESIGN.md §9.4 bans TTS/ASR from the training loop.
- Does **not** call an LLM-as-judge. `compute_rewards` is pure-functional on the frozen episode (CLAUDE.md §0.5, rewards.md §3.1).
- Does **not** touch the MCP server, Gradio demo, or HF Space deployment infra — those are Person D's domain (CLAUDE.md §2.2).

---

## 7. Edge Cases

Minimum 6 explicitly required by the task; listed (a)–(f) to match the briefing, plus extras.

### 7a. V100 FP16 gradient instability on BF16-native model

**Symptom:** `train/grad_norm` spikes to inf within the first 20 steps; loss NaN; generated tokens go garbage.

**Root cause:** Gemma 4 weights were trained in BF16; mixed precision autocast with FP16 can underflow in softmax/GELU when attention scores are saturated. V100 lacks BF16 tensor cores, so BF16 is not an option.

**Mitigation (baked into this spec):**
1. `fp16=True` with Unsloth's autocast (narrowed FP16 regions; full FP32 for layernorms).
2. `max_grad_norm=1.0` (TRL default; explicit assertion in `build_grpo_config` that it stays ≤ 1.0).
3. Loss-scale monitored every 10 steps via the training callback; if loss-scale halves 3 times within 50 steps, log a WARN (precursor to catastrophic underflow).
4. Explicit `dtype=torch.float16` at `FastModel.from_pretrained` — Unsloth otherwise auto-picks BF16 on A100/H100 and silently runs FP16 on V100 via backend fallback; explicit beats implicit.
5. **Fallback (if instability persists):** reduce `learning_rate` 5e-6 → 2e-6, re-run.

### 7b. batch-1 G=8 OOM on long episodes

**Symptom:** `torch.cuda.OutOfMemoryError` in `model.generate` on a Stage-3 episode with `max_completion_length=2048` and a 6-turn trace.

**Root cause:** KV-cache for G=8 parallel generations + gradient checkpointing still peaks above 32GB on particularly long rollouts (Stage-3 compound drift often trips 8+ turns, blowing past the 6-turn avg assumption).

**Mitigation:**
1. **G=4 fallback at group granularity.** On OOM during `rollout_group`, `torch.cuda.empty_cache()`, set `config.num_generations=4` for the *next* group only (not permanently), and retry the failed group at G=4.
2. G=4 is still GRPO-valid (DESIGN.md §3.2 explicitly names G=4 as the OOM fallback). Group-relative advantage normalization works with G=4; variance just increases slightly.
3. **Preserve effective rollouts/update at 32** — when G flips 8 → 4, `gradient_accumulation_steps` flips 4 → 8 in the same call (`build_grpo_config(num_generations=4)` returns `grad_accum=8`). Effective rollouts/update stays `G × grad_accum = 32` so optimizer-step variance is unchanged. We do NOT accept the "16 rollouts/update" alternative — it measurably destabilizes paged-adamw-8bit on the 2B-param scale.
4. **Group boundary is the switch-point.** `num_generations` can only change between groups, never mid-group. All 4 (post-fallback) or 8 (pre-fallback) rollouts in a group share G. Within a single gradient-accumulation window (the 8 groups that feed one optimizer step under G=4), every group uses the same G — **mixed-G accumulation windows are FORBIDDEN** because their advantages would be on different variance scales. The fallback flag flips at an accumulation-window boundary too: if G=8 OOMs at group 3 of an 8-group window, that window is ABANDONED (no optimizer step), and a fresh G=4/grad_accum=8 window starts.
5. If G=4 OOMs too → permanently drop `max_completion_length` to 1536 for the remainder of the stage and log it to WandB; if that OOMs, raise to user (§5 `OutOfMemoryError`).
6. A monitoring counter `train/g4_fallback_rate` tracks how often G=4 fires; > 20% sustained is a signal to tune `max_completion_length` downward permanently.

### 7c. KL divergence spike (policy_kl > 10.0)

**Symptom:** `train/policy_kl` crosses 10.0 (mean over a 10-step window); generated tokens collapse into repetition or degenerate greedy patterns.

**Root cause (rank-ordered):**
1. `use_bias_correction_kl=False` slipped into config — asserted against in `build_grpo_config`, so this can only happen if someone monkey-patches the config post-construction. `KLDivergenceExplosion` raised.
2. `learning_rate` too high for the sparse-reward regime — drop to 2e-6 and resume from last checkpoint.
3. Reward signal collapsed (all rewards = 0 or all = 1), advantage becomes near-zero noise, KL drifts unbounded. Logged as `RewardCollapseError` instead (see 7d).

**Mitigation:**
1. **Detect early:** the training callback computes a 10-step rolling mean of `train/policy_kl` and raises `KLDivergenceExplosion` at threshold (§5). No recovery attempt — halt and escalate (CLAUDE.md §11).
2. **Prevention:** `beta=0.04` and `use_bias_correction_kl=True` are config invariants.
3. Forensic artifact: on halt, dump the last 20 rollout groups' (prompt, completions, rewards) to `debug/kl_explosion_dump.jsonl` for root-cause analysis.

### 7d. Reward hacking spike (R5 jumps)

**Symptom:** `train/R5_mean` drops from ~0.0 to ≤ -0.3 within a 10-step window AND `train/hallucinated_field_count` spikes concurrently AND `train/reward_mean` does NOT fall proportionally (agent is paying -0.05 on quality but gaining elsewhere → net positive to hack).

**Root cause:** GRPO found an exploit in one of the reward branches — most commonly R2 "bare drift assertion" under our anti-hack R5 = -0.3, or R5 branch (a) hallucinated fields that the group mean still rewards because the hallucination coincides with an otherwise-high R1 episode.

**Mitigation:**
1. `RewardCollapseError` raises at the R5-drop + reward-mean-drop threshold (§5).
2. Person B runs the reward-hacking probe (DESIGN.md §13 deliverable #9) on the last 200 episodes; if a new exploit pattern is found, update `anti_hack_penalty` logic in `rewards.py` per rewards.md §3.6, bump `config_sha256`, and resume from the **pre-regression** checkpoint (NOT the current one — its policy already learned the exploit).
3. Prevention: per rewards.md §3.6 the penalties stack additively to a -1.0 floor; any single exploit that yields > +0.05 net reward is the design failure, not the training failure.

### 7e. Resume from corrupted checkpoint

**Symptom:** `train(..., resume_from=Path(".../stage2_final"))` raises during `PeftModel.from_pretrained` or the `trainer_state.json` load.

**Possible corruption modes:**
1. `adapter_model.safetensors` truncated (disk full during save).
2. `adapter_config.json` missing.
3. `trainer_state.json` refers to an optimizer state file that doesn't exist.
4. Git LFS rehydration incomplete.
5. HF Hub download interrupted.

**Mitigation:**
1. `CheckpointIOError` raised — no silent fresh-start (explicit per §5).
2. **Integrity check on save:** after `save_pretrained`, immediately re-load the checkpoint into a scratch `FastModel` instance and verify the output of a canonical prompt matches the in-memory model's output. If mismatch → `CheckpointIOError`, retry save once, else raise.
3. **Integrity check on resume:** sha256 `adapter_model.safetensors` against `driftcall_meta.json`'s recorded hash (field to add to `CheckpointMeta` if the on-save verify approach proves too slow).
4. **Human-readable recovery instruction** in the error message: "resume_from={path} corrupt; try {path}.backup or {path-previous-stage}".

### 7f. Language-weight cohort collapse (no Hindi examples in stage batch)

**Symptom:** `train/reward_hi` is `NaN` for ≥ 20 consecutive steps at `stage ∈ {2, 3}`; the per-language bucket accumulated zero `hi` episodes.

**Root cause:** caller passed `language_weights={"hi": 0.0, ...}` (misconfiguration), OR the weighted sampler happened to skip `hi` for a very long run (improbable at `p=0.2` — requires `~(0.8)^20 ≈ 1.2%`, 1-in-80 unlucky streak).

**Mitigation:**
1. **Soft warning** at 20 consecutive steps missing a cohort — logged to `train/cohort_collapse:{lang}` and surfaces in WandB alerts.
2. **Hard error** (`LanguageCohortCollapseError`, §5) at 50 consecutive steps — this is certainly misconfig.
3. **Prevention:** `train()` validates `language_weights` at call time: for `stage in {2, 3}`, every non-English language must have weight ≥ 0.05. Stage-1 permits zero-weight `ta`/`kn` (per DESIGN.md §10.3 Stage-1 mix).

### 7g (extra). EpisodeParseError cascade

**Symptom:** Early-training policy emits malformed JSON for `tool_args` on every turn; every episode ABORTs on turn 1; all rewards collapse to R1=0, R4≈0.

**Mitigation:** This is the explicit case the curriculum stage-1 exists to solve — reward R4's `-0.2 per invalid JSON tool call` plus `-0.05 per missing rationale` provides gradient back to valid format. We do NOT intervene; the training signal is the fix. Monitor `train/episode_parse_failures` — expect it to drop below 0.1 by step 50 and below 0.02 by step 150. If it stalls above 0.3 by step 50, the system prompt is the bug, not the policy.

### 7h (extra). Resume across stages with different `num_generations`

**Symptom:** Stage 1 ran at G=8 without fallback; Stage 2 hits OOM on step 1 and falls back to G=4. WandB run grouping breaks because step counts differ.

**Mitigation:** `WANDB_RUN_GROUP="curriculum-v1"` groups all three stages under one dashboard; per-stage `run_name=f"driftcall-stage{N}"` keeps the curves distinct. The step axis is **per-stage**, not cumulative; `CheckpointMeta.cumulative_steps` preserves the global counter for pitch plots (DESIGN.md §15).

---

## 8. Examples

All three examples use `from __future__ import annotations` at file top.

### 8.1 Stage-1 training launch + expected first-50-step metrics

**Command (from `DRIFTCALL/` directory, DESIGN.md §12.3 onsite Day-1 hours 2–8):**

```bash
python3 training/train_grpo.py --stage 1 --steps 150
# internally calls: train(stage=1, num_steps=150, resume_from=None)
```

**Expected WandB curves by step 50** (empirical targets from smoke-test budget; critic may sanity-check against real first run):

| Column | Step 0 | Step 25 | Step 50 | Interpretation |
|---|---|---|---|---|
| `train/reward_mean` | ~0.18 | ~0.28 | ~0.42 | Policy discovers valid tool-call format |
| `train/R1_mean` | ~0.05 | ~0.18 | ~0.35 | Bookings start succeeding |
| `train/R4_mean` | ~0.30 | ~0.72 | ~0.91 | JSON format compliance climbs fast (dominant early signal) |
| `train/R5_mean` | ~-0.05 | ~-0.02 | ~-0.01 | Hack rate trivially low at Stage 1 |
| `train/policy_kl` | ~0.05 | ~0.12 | ~0.18 | Stable, well below the 10.0 halt threshold |
| `train/gen_length_mean` | ~850 | ~620 | ~510 | Policy learns concise tool calls |
| `train/episode_parse_failures` | ~0.45 | ~0.15 | ~0.06 | Format signal doing its job |
| `train/reward_hi` | ~0.12 | ~0.24 | ~0.38 | Hindi cohort keeps pace |

**Stage-1 exit gate:** `train/R1_mean ≥ 0.4` by step 100; if not met, halt and re-tune before Stage 2 (CLAUDE.md §11).

**Artifact produced:** `checkpoints/stage1_final/{adapter_model.safetensors, adapter_config.json, tokenizer.json, driftcall_meta.json, trainer_state.json, optimizer.pt, scheduler.pt}`.

### 8.2 Stage-2 resume with drift curriculum

**Command (DESIGN.md §12.3 onsite Day-1 hours 10–14):**

```bash
python3 training/train_grpo.py --stage 2 --steps 200 \
    --resume-from checkpoints/stage1_final
# internally: train(stage=2, num_steps=200, resume_from=Path("checkpoints/stage1_final"))
```

**What happens inside `train()`:**

```python
from __future__ import annotations

# 1. Load base 4-bit Gemma + attach Stage-1 adapters.
model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it-bnb-4bit",
    max_seq_length=4096, load_in_4bit=True, dtype=torch.float16,
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "checkpoints/stage1_final", is_trainable=True)

# 2. Build Stage-2 config.
config = build_grpo_config(stage=2, num_generations=8)

# 3. Build Stage-2 language weights (DESIGN.md §10.3 row 2).
lang_w = {"en": 0.30, "hinglish": 0.30, "hi": 0.20, "ta": 0.10, "kn": 0.10}

# 4. env factory produces a fresh env per rollout with curriculum_stage=2.
def env_factory():
    return DriftCallEnv(config={"curriculum_stage": 2})

# 5. GRPOTrainer wired up with reward_fn (§2.3) and episode sampler.
trainer = GRPOTrainer(
    model=model, args=config,
    train_dataset=EpisodeDatasetAdapter(
        sampler=lambda step: task_generator.generate(
            seed=stage_base_seed(stage=2) + step, stage=2, language_weights=lang_w,
        ),
        env_factory=env_factory,
    ),
    reward_funcs=[reward_fn],
    processing_class=tokenizer,
)
trainer.train()           # resumes from trainer_state.json in checkpoints/stage1_final
```

**Expected trajectory over Stage-2's 200 steps:**

- `train/R2_mean`: starts ~0.50 (Stage 1 neutral value carried forward), climbs to ~0.65 by step 100, ~0.80 by step 200.
- `train/drift_detected_rate`: ~0.08 at step 0 (pre-drift baseline) → ~0.55 by step 200 (matches DESIGN.md §15 "Drift detection goes from 8% to 71%" though the 71% is post-Stage-3).
- `train/reward_mean`: stable ~0.45 initially (drift is new friction), climbing to ~0.55 by step 200.

**Watch-out:** if `train/R2_mean` stalls below 0.55 by step 150, the R2 matching logic in rewards.md §3.3 (three branches) likely has a bug — escalate to Person B (rewards owner).

### 8.3 Final eval producing `EvalReport` with curves

**Command (DESIGN.md §12.4 Day-2 hours 4–6):**

```bash
python3 training/eval_final.py \
    --checkpoint checkpoints/stage3_final \
    --episodes 50 \
    --output eval/final_report.json
# internally: report = eval(model_path=Path("checkpoints/stage3_final"), episodes=50)
#             Path("eval/final_report.json").write_text(json.dumps(asdict(report), indent=2))
```

**Expected `EvalReport` shape (matches DESIGN.md §15 pitch numbers):**

```python
EvalReport(
    model_path="checkpoints/stage3_final",
    n_episodes=50,
    reward_mean_ci=(0.64, 0.58, 0.70),        # mean with 95% bootstrap CI
    r1_mean_ci=(0.64, 0.52, 0.74),            # "Task completion climbs from 18% to 64%"
    r2_mean_ci=(0.71, 0.60, 0.81),            # "Drift detection goes from 8% to 71%"
    r3_mean_ci=(0.82, 0.76, 0.88),
    r4_mean_ci=(0.94, 0.90, 0.97),
    r5_mean_ci=(-0.04, -0.09, -0.01),         # negligible post-training
    brier_mean=0.09,
    floor_applied_rate=0.12,                  # calibrated-surrender used on ~6/50 episodes
    hallucinated_field_rate=0.02,
    reward_hacking_offenses={
        "hallucinated_field": 1,
        "repeated_identical_calls": 0,
        "probe_abuse": 0,
        "bare_drift_assertion": 1,
        "protected_write": 0,
    },
    drift_detection_latency=DriftDetectionLatency(
        stage2_mean=1.4, stage2_median=1.0, stage2_p95=3.0,
        stage3_mean=1.8, stage3_median=2.0, stage3_p95=4.0,
        undetected_count=7,                   # ~14% undetected across stages 2+3
    ),
    # "Latency from drift-event to adaptation drops from 4.2 turns to 1.6" → mean ≈ 1.6 matches.
    per_language=(
        PerLanguageReport(language="en", n_episodes=15, reward_mean=0.68,
                          r1_mean=0.67, r2_mean=0.73, r3_mean=0.84,
                          r4_mean=0.95, r5_mean=-0.03),
        PerLanguageReport(language="hinglish", n_episodes=15, reward_mean=0.63,
                          r1_mean=0.60, r2_mean=0.73, r3_mean=0.81,
                          r4_mean=0.94, r5_mean=-0.04),
        PerLanguageReport(language="hi", n_episodes=10, reward_mean=0.61,
                          r1_mean=0.60, r2_mean=0.70, r3_mean=0.80,
                          r4_mean=0.93, r5_mean=-0.05),
        PerLanguageReport(language="ta", n_episodes=5, reward_mean=0.58,
                          r1_mean=0.60, r2_mean=0.60, r3_mean=0.80,
                          r4_mean=0.90, r5_mean=-0.06),
        PerLanguageReport(language="kn", n_episodes=5, reward_mean=0.56,
                          r1_mean=0.60, r2_mean=0.60, r3_mean=0.80,
                          r4_mean=0.88, r5_mean=-0.06),
    ),
    curves={
        "reward_vs_step": ((0, 0.18), (50, 0.42), (150, 0.48),
                           (350, 0.55), (500, 0.64)),
        "R1_vs_step":     ((0, 0.05), (50, 0.35), (150, 0.46),
                           (350, 0.58), (500, 0.64)),
        "R2_vs_step":     ((0, 0.50), (150, 0.50), (160, 0.55),
                           (350, 0.72), (500, 0.71)),
        "drift_latency_vs_step": ((150, 4.2), (250, 2.6), (350, 1.9), (500, 1.6)),
    },
)
```

The `curves` dict powers the **3-plot panel** in the 3-min pitch (DESIGN.md §15 segment 1:00–2:00).

**Paired with baseline eval** (`python3 training/eval_baseline.py --model base --episodes 50`):
- Baseline `reward_mean_ci ≈ (0.22, 0.16, 0.28)`, `r1_mean_ci ≈ (0.18, 0.08, 0.28)`, `r2_mean_ci ≈ (0.08, 0.02, 0.16)`.
- The before/after delta IS the pitch.

---

## 9. Open Questions

1. **Exact reference-model handling for KL.** TRL 0.23 uses the frozen base model as the KL reference by default. When resuming Stage 2 from Stage 1, should the KL reference be the *original* base or the *Stage-1-tuned* model? DESIGN.md §10 is silent. **Proposal:** keep the reference as the original 4-bit base across all three stages so KL has a stable anchor; curriculum drift is captured in reward, not in the KL reference. To confirm with Person C on onsite Day 1 hour 0–2.

2. **Unsloth version pinning — minimum patch level.** DESIGN.md §10.1 says "Unsloth 2026.4.5+"; the KL-fix patch mentioned in Unsloth discussion #4921 (DESIGN.md §16.E) landed in 2026.4.5 but subsequent patches may regress. **Proposal:** pin to exactly `unsloth==2026.4.5` for the hackathon window and validate via smoke test (DESIGN.md §16.A.1) before Stage-1 kickoff. Lock in `requirements.txt` before Batch C4.

3. **WandB alert thresholds.** The spec lists `policy_kl > 10.0` (halt) and `reward_mean` 10-step drop > 0.15 (halt-on-reward-collapse). These numbers are informed by CodeForge and public GRPO runs but not validated on DriftCall specifically. **Proposal:** start with these, tune after Stage-1 completes (data-driven).

4. **Eval bootstrap CI method.** §4.2 defines `_ci` as a 3-tuple `(mean, lo, hi)` at 95% via bootstrap. Percentile bootstrap or BCa? **Proposal:** plain percentile bootstrap with 1000 resamples — simpler, acceptable for n=50, and Person B's probe report uses the same method (reward-hacking probe in DESIGN.md §13 deliverable #9). Document as "percentile bootstrap, 1000 resamples, seed=0" in EvalReport JSON for reproducibility.

**Resolved in round-2 pass** (previously OQ #2): eval-set seeding policy. `eval()` consumes `val/briefs.jsonl` rows `[0:episodes]` in stable file order; each row seeded deterministically as `env.reset(seed=hash((episode_id, "eval")) & 0xFFFFFFFF)`; baseline and final evals use IDENTICAL `(row, seed)` pairs for paired-difference statistics. See §2.1 (`eval` docstring).

---

**End of spec. Implementation (`training/train_grpo.py`, `training/eval_baseline.py`, `training/eval_final.py`) does not start until ≥ 2 fresh critic agents return `NOTHING_FURTHER` on this doc.**
