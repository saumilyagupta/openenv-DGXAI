"""DriftCall training orchestrator — local GPU run. **DEPRECATED.**

This script wires the legacy TRL ``GRPOTrainer`` + Unsloth ``UnslothGRPOTrainer``
chain to ``cells/step_15/16/17_train_stage*.train()``. That stack hits Unsloth's
broken ``Linear4bit.forward`` patch on Gemma 3n's ``per_layer_model_projection``
inside TRL's training loop (our lazy fix-up patch fires too late) and breaks on
every TRL/Unsloth version bump.

**Use the new self-contained loop instead:**

    scripts/train_driftcall_grpo.py        # single stage
    scripts/train_full_gemma3n.sh          # all three stages

That loop bypasses TRL entirely and drives rollouts/rewards/updates directly,
with controlled patch ordering. See ``docs/modules/training.md`` §3.2.

This file is kept on disk because some environments still reference it.
Do not extend it. New training work goes in ``scripts/train_driftcall_grpo.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to PYTHONPATH so cells/ is importable without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# transformers 5.x removed the legacy ``TRANSFORMERS_CACHE`` symbol that
# ``llm_blender`` (a TRL 0.24 transitive dep) imports at module load time.
# Restore it BEFORE any ``from trl import ...`` so GRPOTrainer can boot.
def _patch_transformers_cache() -> None:
    try:
        import transformers.utils.hub as _hub
    except Exception:
        return
    if not hasattr(_hub, "TRANSFORMERS_CACHE"):
        _hub.TRANSFORMERS_CACHE = os.environ.get(
            "HF_HOME",
            os.path.expanduser("~/.cache/huggingface"),
        )


_patch_transformers_cache()


# Unsloth 2026.4.x ships a buggy ``Gemma3nRMSNorm_forward`` patch that
# unconditionally reads ``self.weight``; for Gemma3n's
# ``embedding_post_projection_norm`` (constructed with ``with_scale=False``)
# this raises AttributeError during model.generate(). Override the patched
# ``Gemma3nMultimodalEmbedder.forward`` with a with_scale-aware version.
def _patch_unsloth_gemma3n_rmsnorm() -> None:
    try:
        import torch
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nMultimodalEmbedder,
        )
    except Exception:
        return

    def _safe_rmsnorm(norm_module: Any, x: Any) -> Any:
        # Mirror the canonical Gemma3n RMSNorm forward but respect with_scale.
        normed = norm_module._norm(x.float())
        if getattr(norm_module, "with_scale", True):
            normed = normed * norm_module.weight.float()
        return normed.type_as(x)

    def _patched_forward(
        self: Any,
        input_ids: Any = None,
        inputs_embeds: Any = None,
    ) -> Any:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is not None:
            emb_norm = _safe_rmsnorm(self.soft_embedding_norm, inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = _safe_rmsnorm(self.hard_embedding_norm, hard_emb)
        old_dtype = emb_norm.dtype
        emb_norm = emb_norm.to(torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            emb_norm_proj = self.embedding_projection(emb_norm)
        emb_norm_proj = emb_norm_proj.to(old_dtype)
        return _safe_rmsnorm(self.embedding_post_projection_norm, emb_norm_proj)

    Gemma3nMultimodalEmbedder.forward = _patched_forward


def _patch_unsloth_bnb_linear4bit_quant_state() -> None:
    """Trigger ``fix_4bit_weight_quant_state_from_module`` on packed 4-bit
    weights that bnb stored with ``shape[0] == 1`` (transposed packing).

    Unsloth's stock patch only checks ``weight.shape[-1] == 1``; some Gemma3n
    layers (notably ``per_layer_model_projection``) ship with the packed
    dim on axis 0 instead, so the auto-fix never fires and we crash with
    ``mat1 and mat2 shapes cannot be multiplied (..., 1xPACKED)``.
    """
    try:
        import torch
        import bitsandbytes
        from unsloth_zoo.temporary_patches.bitsandbytes import (
            fix_4bit_weight_quant_state_from_module,
        )
    except Exception:
        return

    Linear4bit = bitsandbytes.nn.modules.Linear4bit
    _orig_forward = Linear4bit.forward

    def _safe_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        # Detect packed 4-bit tensors with no quant_state and a flat shape
        # in either orientation. Trigger the fix routine which restores
        # quant_state from the ``module`` attribute and reshapes the weight.
        try:
            if (
                getattr(weight, "quant_state", None) is None
                and weight.dim() == 2
                and (weight.shape[0] == 1 or weight.shape[-1] == 1)
            ):
                fix_4bit_weight_quant_state_from_module(self)
        except Exception:
            pass
        return _orig_forward(self, x)

    Linear4bit.forward = _safe_forward


# ---------------------------------------------------------------------------
# Action parser — extract DriftCallAction from model text output
# ---------------------------------------------------------------------------


def _parse_action(text: str) -> Any:
    """Parse a DriftCallAction from the model's assistant turn text.

    Tries JSON extraction first; falls back to an ABORT action on parse
    failure so the episode terminates cleanly rather than hanging.
    """
    from cells.step_04_models import ActionType, DriftCallAction

    text = text.strip()
    # Try to extract the first JSON object in the text
    try:
        start = text.index("{")
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            obj = json.loads(text[start:end])
            action_type_str = obj.get("action_type", "abort")
            try:
                atype = ActionType(action_type_str)
            except ValueError:
                atype = ActionType.ABORT
            return DriftCallAction(
                action_type=atype,
                tool_name=obj.get("tool_name"),
                tool_args=obj.get("tool_args"),
                message=obj.get("message"),
                confidence=obj.get("confidence"),
                rationale=obj.get("rationale"),
            )
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    # Fallback: submit if the model said something confidence-like, else abort
    lower = text.lower()
    if "submit" in lower:
        return DriftCallAction(
            action_type=ActionType.SUBMIT,
            message=text[:200],
            confidence=0.5,
        )
    return DriftCallAction(action_type=ActionType.ABORT, message=text[:200])


# ---------------------------------------------------------------------------
# Observation serializer — obs → messages list (training.md §3.2.1)
# ---------------------------------------------------------------------------


def _obs_to_messages(
    goal: Any,
    obs: Any,
    history: list[dict[str, str]],
    is_turn_zero: bool,
) -> list[dict[str, str]]:
    """Append the latest observation fields to the message history.

    training.md §3.2.1 — returns the updated history list (mutated in-place
    for efficiency; callers can deepcopy if needed).
    """
    from cells.step_14_custom_trainer import PINNED_SYSTEM_PROMPT

    if is_turn_zero:
        # Build the system prompt with available tools appended.
        tool_schemas = getattr(obs, "available_tools", [])
        system_content = PINNED_SYSTEM_PROMPT
        if tool_schemas:
            system_content += "\nAvailable tools: " + json.dumps(
                tool_schemas, ensure_ascii=False, sort_keys=True
            )
        history.clear()
        history.append({"role": "system", "content": system_content})
        history.append(
            {"role": "user", "content": getattr(goal, "seed_utterance", "")}
        )
    else:
        # Append any new tool results from the last step.
        tool_results = getattr(obs, "tool_results", [])
        if tool_results:
            for tr in tool_results[-1:]:  # only the latest tool result
                history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "tool": getattr(tr, "tool_name", ""),
                                "status": getattr(tr, "status", ""),
                                "response": getattr(tr, "response", {}),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
        # Append drift events if any.
        drift_log = getattr(obs, "drift_log", [])
        if drift_log:
            drift_json = json.dumps(
                [
                    {
                        "turn": getattr(d, "turn", 0),
                        "type": getattr(d, "drift_type", ""),
                        "domain": getattr(d, "domain", ""),
                        "description": getattr(d, "description", ""),
                    }
                    for d in drift_log[-3:]  # last 3 drifts to cap token budget
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            # Append as user message so the model sees the drift signal.
            history.append(
                {"role": "user", "content": f"[drift] {drift_json}"}
            )

    return history


def _derive_rollout_seed(goal: Any, g_index: int, episode_seed: int) -> int:
    """Deterministic seed per rollout within a group (training.md §3.2)."""
    payload = f"{episode_seed}:{getattr(goal, 'seed_utterance', '')}:{g_index}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# rollout_group_fn — the core multi-turn inference loop
# ---------------------------------------------------------------------------


def build_rollout_group_fn(
    *,
    max_turns: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
    top_p: float = 0.95,
    hardware: str = "v100",
) -> Any:
    """Build and return a RolloutGroupFn (training.md §3.2).

    Runs G independent multi-turn episodes with the live model. Each rollout:
      1. env.reset(seed=derived_seed)
      2. Serialise obs → messages, generate one action token-by-token
      3. Parse → DriftCallAction → env.step(action)
      4. Repeat until obs.done or max_turns reached
    Returns (tuple[Episode, ...], tuple[str, ...]) of length G.
    """

    def rollout_group_fn(
        *,
        model: Any,
        tokenizer: Any,
        goal: Any,
        episode_seed: int,
        num_generations: int,
        env_factory: Any,
    ) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        import torch
        from cells.step_04_models import ActionType, DriftCallAction

        # Apply Gemma3n + bnb patches on first rollout call, AFTER Unsloth has
        # already monkey-patched its (broken) versions.
        _patch_unsloth_gemma3n_rmsnorm()
        _patch_unsloth_bnb_linear4bit_quant_state()

        device = next(model.parameters()).device
        episodes_out: list[Any] = []
        completions_out: list[str] = []

        for g in range(num_generations):
            seed = _derive_rollout_seed(goal, g, episode_seed)
            env = env_factory()
            obs = env.reset(seed=seed)

            history: list[dict[str, str]] = []
            all_responses: list[str] = []
            is_turn_zero = True

            for _turn in range(max_turns):
                # Build messages for this turn.
                _obs_to_messages(goal, obs, history, is_turn_zero)
                is_turn_zero = False

                # Tokenize the conversation so far.
                try:
                    prompt_str = tokenizer.apply_chat_template(
                        history,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt_str = " ".join(
                        m.get("content", "") for m in history
                    )

                # Gemma 3n's processor is multimodal — pass `text=` explicitly
                # so the call dispatches to the text-only branch.
                inputs = tokenizer(
                    text=prompt_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(device)

                # Generate the assistant response.
                with torch.no_grad():
                    gen_kwargs: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    output_ids = model.generate(
                        **inputs, **gen_kwargs
                    )
                new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                response_text = tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                ).strip()

                all_responses.append(response_text)
                history.append({"role": "assistant", "content": response_text})

                # Parse and step the environment.
                action = _parse_action(response_text)
                obs = env.step(action)

                if obs.done:
                    break

            # Collect the completed episode from the env.
            episode = env.episode()
            episodes_out.append(episode)
            completions_out.append("\n".join(all_responses))

        return tuple(episodes_out), tuple(completions_out)

    return rollout_group_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_driftcall_train",
        description="DriftCall 3-stage GRPO training on a local GPU.",
    )
    p.add_argument("--stage1-steps", type=int, default=150)
    p.add_argument("--stage2-steps", type=int, default=200)
    p.add_argument("--stage3-steps", type=int, default=150)
    p.add_argument("--hardware", choices=["v100", "h100"], default="v100")
    p.add_argument("--output-dir", type=Path, default=Path("/workspace/checkpoints"))
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--probe-episodes", type=int, default=200)
    p.add_argument(
        "--skip-eval", action="store_true", help="Skip baseline/final eval + probe."
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained LoRA to HF Hub after stage 3.",
    )
    p.add_argument("--hf-repo", type=str, default=os.environ.get("DRIFTCALL_HF_REPO", ""))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.output_dir

    print(f"[train] hardware={args.hardware}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"[train] steps: stage1={args.stage1_steps}  stage2={args.stage2_steps}  stage3={args.stage3_steps}")
    print(f"[train] output_dir={out_dir}")

    from cells.step_07_task_generator import generate as task_gen_fn
    from cells.step_10_env import DriftCallEnv
    from cells.step_12_gemma_boot import BootConfig

    boot_config = BootConfig(hardware=args.hardware)
    rollout_group_fn = build_rollout_group_fn(hardware=args.hardware)

    # ---------------------------------------------------------------------------
    # Stage 1
    # ---------------------------------------------------------------------------
    print("\n[train] === Stage 1 ===")
    from cells.step_15_train_stage1 import train as train_stage1

    def env_factory_s1() -> DriftCallEnv:
        return DriftCallEnv(config={"curriculum_stage": 1, "audio_boundary_enabled": False})

    ckpt_s1 = train_stage1(
        num_steps=args.stage1_steps,
        output_dir=out_dir / "stage1" / "final",
        boot_config=boot_config,
        task_gen=task_gen_fn,
        env_factory=env_factory_s1,
        rollout_group_fn=rollout_group_fn,
    )
    print(f"[train] Stage 1 complete → {ckpt_s1}")

    # ---------------------------------------------------------------------------
    # Stage 2
    # ---------------------------------------------------------------------------
    print("\n[train] === Stage 2 ===")
    from cells.step_16_train_stage2 import train as train_stage2

    def env_factory_s2() -> DriftCallEnv:
        return DriftCallEnv(config={"curriculum_stage": 2, "audio_boundary_enabled": False})

    ckpt_s2 = train_stage2(
        num_steps=args.stage2_steps,
        resume_from=ckpt_s1,
        output_dir=out_dir / "stage2" / "final",
        boot_config=boot_config,
        task_gen=task_gen_fn,
        env_factory=env_factory_s2,
        rollout_group_fn=rollout_group_fn,
    )
    print(f"[train] Stage 2 complete → {ckpt_s2}")

    # ---------------------------------------------------------------------------
    # Stage 3
    # ---------------------------------------------------------------------------
    print("\n[train] === Stage 3 ===")
    from cells.step_17_train_stage3 import train as train_stage3

    def env_factory_s3() -> DriftCallEnv:
        return DriftCallEnv(config={"curriculum_stage": 3, "audio_boundary_enabled": False})

    ckpt_s3 = train_stage3(
        num_steps=args.stage3_steps,
        resume_from=ckpt_s2,
        output_dir=out_dir / "stage3" / "final",
        boot_config=boot_config,
        task_gen=task_gen_fn,
        env_factory=env_factory_s3,
        rollout_group_fn=rollout_group_fn,
    )
    print(f"[train] Stage 3 complete → {ckpt_s3}")

    if not args.skip_eval:
        # ---------------------------------------------------------------------------
        # Baseline + Final eval
        # ---------------------------------------------------------------------------
        print("\n[train] === Baseline eval ===")
        from cells.step_18_eval_baseline import eval_baseline
        eval_dir = Path(os.environ.get("DRIFTCALL_EVAL_DIR", "/workspace/eval_reports"))
        eval_dir.mkdir(parents=True, exist_ok=True)

        baseline_report = eval_baseline(
            n_episodes=args.eval_episodes,
            output_path=eval_dir / "baseline.json",
            env_factory=env_factory_s3,
            task_gen=task_gen_fn,
        )
        print(f"[train] Baseline eval: R1={getattr(baseline_report, 'r1_mean', '?'):.3f}")

        print("\n[train] === Final eval ===")
        from cells.step_19_eval_final import eval_final
        final_report = eval_final(
            checkpoint_path=ckpt_s3,
            n_episodes=args.eval_episodes,
            output_path=eval_dir / "final.json",
            env_factory=env_factory_s3,
            task_gen=task_gen_fn,
        )
        print(f"[train] Final eval: R1={getattr(final_report, 'r1_mean', '?'):.3f}")

    if args.push_to_hub and args.hf_repo:
        print(f"\n[train] === Pushing LoRA to HF Hub: {args.hf_repo} ===")
        from cells.step_24_deploy_hf import push_lora_to_hub
        result = push_lora_to_hub(
            ckpt_s3, repo_id=args.hf_repo, token=os.environ.get("HF_TOKEN")
        )
        if result.success:
            print(f"[train] Pushed to hub: {args.hf_repo}")
        else:
            print(f"[train] Push failed (rc={result.return_code}): {result.stderr[:200]}")

    print("\n[train] === COMPLETE ===")
    print(f"[train] Checkpoints at: {out_dir}/stage{{1,2,3}}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
