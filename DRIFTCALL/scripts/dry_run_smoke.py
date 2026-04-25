"""Local CPU-friendly smoke that exercises the full DriftCall trainer path.

Bypasses Gemma boot (uses a tiny gpt2 model on CPU) but uses the real TRL
GRPOTrainer + DriftCall callbacks/dataset to flush out integration bugs in
trainer.__init__ and trainer.train() without needing a GPU.

Run from DRIFTCALL/:
    .venv/bin/python scripts/dry_run_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("[dry-run] importing real torch/transformers/trl/unsloth ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[dry-run] loading tiny model on CPU ...")
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Simple chat template so render_initial_prompt() works.
    tok.chat_template = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}"
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.to("cpu")

    print("[dry-run] building GRPOConfig ...")
    from cells.step_13_grpo_config import build_grpo_config
    config = build_grpo_config(stage=1, hardware="v100", resume_output_dir="/tmp/dry_run")
    config.max_steps = 1
    config.logging_steps = 1
    config.save_steps = 1_000_000  # don't save
    config.report_to = "none"  # don't report
    config.fp16 = False  # CPU
    config.bf16 = False

    print("[dry-run] building dataset ...")
    from cells.step_14_custom_trainer import EpisodeDatasetAdapter
    def fake_task_gen(*, seed, stage, language_weights):
        from cells.step_04_models import GoalSpec
        return GoalSpec(
            domain="travel",
            intent="book_ticket",
            slots={"city": "Mumbai"},
            constraints={},
            language="hi",
            seed_utterance="Book a ticket to Mumbai",
        )
    def fake_env_factory(**_):
        return None
    dataset = EpisodeDatasetAdapter(
        task_gen=fake_task_gen,
        env_factory=fake_env_factory,
        stage=1,
        stage_base_seed=0,
        language_weights={"hi-IN": 1.0},
        tokenizer=tok,
        num_steps=1,
    )
    print(f"[dry-run] len(dataset)={len(dataset)} sample={dataset[0]['prompt'][:60]!r}")

    print("[dry-run] building trainer ...")
    from cells.step_14_custom_trainer import make_driftcall_grpo_trainer_cls
    Trainer = make_driftcall_grpo_trainer_cls()

    def fake_rollout_group_fn(*args, **kwargs):
        return {"episodes": [], "completions": [], "rewards": [], "prompts": []}
    def fake_reward_fn(*args, **kwargs):
        return [0.0]

    trainer = Trainer(
        model=model,
        args=config,
        processing_class=tok,
        train_dataset=dataset,
        rollout_group_fn=fake_rollout_group_fn,
        env_factory=fake_env_factory,
        reward_fn_driftcall=fake_reward_fn,
    )
    print(f"[dry-run] trainer built: {type(trainer).__name__}")
    print(f"[dry-run] callbacks: {[type(c).__name__ for c in trainer.callback_handler.callbacks]}")

    print("[dry-run] checking lifecycle hooks fire ...")
    from transformers.trainer_callback import TrainerControl, TrainerState
    state = TrainerState()
    control = TrainerControl()
    # This is what triggered the prior on_train_begin crash:
    trainer.callback_handler.on_train_begin(config, state, control)
    print("[dry-run] on_train_begin OK")

    print("[dry-run] DONE — pipeline plumbing OK up to trainer init.")


if __name__ == "__main__":
    main()
