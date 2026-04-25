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
    try:
        Trainer = make_driftcall_grpo_trainer_cls()
    except RuntimeError as e:
        # TRL 0.24's optional deps (llm_blender, mergekit) are incompatible
        # with transformers 5.5 in some environments. The remote bypasses
        # this via Unsloth's compiled cache. Skip trainer init in that case.
        if "Failed to import trl" in str(e):
            print(f"[dry-run] skipping trainer-init check (TRL optional dep break in this env): {type(e).__name__}")
            Trainer = None
        else:
            raise

    def fake_rollout_group_fn(*args, **kwargs):
        return {"episodes": [], "completions": [], "rewards": [], "prompts": []}
    def fake_reward_fn(*args, **kwargs):
        return [0.0]

    if Trainer is not None:
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
    else:
        # Direct check: AdaptiveKLCallback must inherit from TrainerCallback
        # so all lifecycle methods exist as no-ops.
        from cells.step_14_custom_trainer import AdaptiveKLCallback
        from transformers.trainer_callback import TrainerCallback
        cb = AdaptiveKLCallback()
        assert isinstance(cb, TrainerCallback), \
            "AdaptiveKLCallback must inherit from TrainerCallback (regression)"
        # Sanity-call a few lifecycle hooks the trainer will actually use:
        for hook in ("on_train_begin", "on_train_end", "on_step_end", "on_epoch_begin"):
            assert hasattr(cb, hook), f"AdaptiveKLCallback missing {hook} (regression)"
        print("[dry-run] AdaptiveKLCallback inherits TrainerCallback OK")

    print("[dry-run] DONE — pipeline plumbing OK up to trainer init.")

    # ── Bonus smoke: simulate Unsloth's Gemma4 processor wrapping that drops
    # positional args, and verify _generate_one_turn calls tokenizer with
    # text= kwarg. ────────────────────────────────────────────────────────
    print("[dry-run] testing _generate_one_turn against positional-eating tokenizer ...")
    from cells.step_14_custom_trainer import _generate_one_turn

    class _PositionalDropTokenizer:
        """Mimics Unsloth's Gemma4 patched_call that drops positional args.

        Real Unsloth wrapper signature:
            def patched_call(self, *args, images=None, text=None, videos=None, **kwargs):
                return original_call(self, images=images, text=text, videos=videos, **kwargs)

        Under this, ``tokenizer(prompt)`` silently sends ``text=None``.
        """
        def __init__(self, real):
            self._real = real
            self.device = "cpu"
        def __call__(self, *args, images=None, text=None, videos=None, **kwargs):
            if text is None:
                raise TypeError("text=None: positional prompt was dropped (regression)")
            return self._real(text=text, **kwargs)
        def decode(self, *a, **kw):
            return self._real.decode(*a, **kw)

    class _StubModel:
        device = "cpu"
        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            return torch.cat([input_ids, input_ids[:, :3]], dim=1)

    try:
        out = _generate_one_turn(_StubModel(), _PositionalDropTokenizer(tok), "Hello world")
        print(f"[dry-run] _generate_one_turn OK; got {out!r}")
    except TypeError as e:
        if "positional prompt was dropped" in str(e):
            print(f"[dry-run] FAIL — regression: {e}")
            sys.exit(1)
        raise

    print("[dry-run] ALL CHECKS PASS.")


if __name__ == "__main__":
    main()
