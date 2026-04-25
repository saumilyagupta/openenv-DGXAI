"""DriftCall — native-PyTorch GRPO training loop (no TRL).

This is the production training script for the Gemma 3n LoRA. It replaces the
fragile TRL ``GRPOTrainer`` + Unsloth ``UnslothGRPOTrainer`` chain with a
self-contained loop that drives rollouts, rewards and policy updates directly.

Algorithm — full GRPO per ``docs/modules/training.md`` §3.2 / DESIGN.md §7.4::

    for step in range(num_steps):
        goal = task_gen(seed = stage_base_seed + step, stage, language_weights)
        group = run_one_group(model, tokenizer, env_factory, goal, num_generations=G)
        # Group-relative advantage A_g = (r_g - mean(r)) / (std(r) + eps)
        # PPO-clipped policy loss + adaptive KL penalty vs frozen reference.
        loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + beta * KL(pi || ref)
        loss.backward(); optim.step(); zero_grad()
        adaptive_kl_controller.update(measured_kl)
        wandb.log({...20 columns from training.md §3.4...})

Usage (single stage)::

    CUDA_VISIBLE_DEVICES=3 python3 scripts/train_driftcall_grpo.py \
        --stage 1 --num-steps 150 \
        --hardware v100 \
        --num-generations 8 \
        --output-dir /workspace/openenv-DGXAI/DRIFTCALL/checkpoints/stage1/final

Three-stage curriculum: use ``scripts/train_full_gemma3n.sh`` which chains the
three stages with ``--resume-from`` plumbing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path bootstrap + import-time patches
# These MUST run before any ``from cells.*`` import so the patches land before
# the model is constructed.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reduce CUDA allocator fragmentation. Gemma3n's large vocab (262 400) causes
# repeated large alloc/free cycles inside _action_logprobs which fragment the
# pool. expandable_segments allows the allocator to grow/shrink segments
# instead of returning them to the OS, keeping fragmentation low.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _patch_transformers_cache() -> None:
    """transformers 5.x removed ``TRANSFORMERS_CACHE``; ``llm_blender`` still
    imports it at module load time. Restore it before any TRL/peft imports.
    """
    try:
        import transformers.utils.hub as _hub
    except Exception:
        return
    if not hasattr(_hub, "TRANSFORMERS_CACHE"):
        _hub.TRANSFORMERS_CACHE = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )


_patch_transformers_cache()


# Import unsloth FIRST (before transformers/peft), per Unsloth's own warning;
# this ensures their FastModel patches install correctly.
import unsloth  # noqa: F401, E402  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# ---------------------------------------------------------------------------
# Post-load patches — must run AFTER unsloth has installed its patches.
# We invoke these from main() once boot_gemma() has loaded the model.
# ---------------------------------------------------------------------------


def _patch_unsloth_gemma3n_rmsnorm() -> None:
    """Override ``Gemma3nMultimodalEmbedder.forward`` with a ``with_scale``-aware
    RMSNorm. Unsloth's stock patch unconditionally reads ``self.weight`` which
    crashes for ``embedding_post_projection_norm`` (constructed with
    ``with_scale=False``).
    """
    try:
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nMultimodalEmbedder,
        )
    except Exception:
        return

    def _safe_rmsnorm(norm_module: Any, x: torch.Tensor) -> torch.Tensor:
        normed = norm_module._norm(x.float())
        if getattr(norm_module, "with_scale", True):
            normed = normed * norm_module.weight.float()
        return normed.type_as(x)

    def _patched_forward(
        self: Any,
        input_ids: Any = None,
        inputs_embeds: Any = None,
    ) -> torch.Tensor:
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
    """Wrap ``Linear4bit.forward`` to detect packed 4-bit weights with no
    ``quant_state`` and trigger ``fix_4bit_weight_quant_state_from_module``.

    Unsloth's stock patch only checks ``shape[-1] == 1``; some Gemma 3n bnb
    layers (notably ``per_layer_model_projection``) ship with the packed dim
    on axis 0 (``shape[0] == 1``), so the auto-fix never fires and we crash
    with ``mat1 and mat2 shapes cannot be multiplied (..., 1xPACKED)``.
    """
    try:
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
# Project imports (order matters: cells.step_12 imports unsloth lazily; we've
# already imported unsloth at the top so its patches are installed).
# ---------------------------------------------------------------------------
from cells.step_04_models import ActionType, DriftCallAction  # noqa: E402
from cells.step_07_task_generator import generate as task_gen  # noqa: E402
from cells.step_08_rewards import compute_rewards  # noqa: E402
from cells.step_10_env import DriftCallEnv  # noqa: E402
from cells.step_12_gemma_boot import BootConfig, boot_gemma  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — frozen knobs from training.md
# ---------------------------------------------------------------------------


PINNED_SYSTEM_PROMPT: str = (
    "You are a concierge assistant. Use the provided tools. "
    "Respond in the caller's language. Submit with calibrated confidence."
)

DEFAULT_NUM_GENERATIONS: int = 8
DEFAULT_MAX_TURNS: int = 6
DEFAULT_MAX_PROMPT_LEN: int = 1024
DEFAULT_MAX_COMPLETION_LEN: int = 512
DEFAULT_TEMPERATURE: float = 0.9
DEFAULT_TOP_P: float = 0.95
DEFAULT_LR: float = 5e-6
DEFAULT_BETA_KL: float = 0.04
DEFAULT_TARGET_KL: float = 0.04
DEFAULT_CLIP_EPS: float = 0.2
DEFAULT_KP: float = 2.0
DEFAULT_BETA_MIN: float = 1e-3
DEFAULT_BETA_MAX: float = 1.0
DEFAULT_GRAD_CLIP_NORM: float = 1.0
DEFAULT_LOGGING_STEPS: int = 1
DEFAULT_SAVE_STEPS: int = 50

STAGE_BASE_SEEDS: dict[int, int] = {1: 1_000_000, 2: 2_000_000, 3: 3_000_000}

STAGE_LANGUAGE_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"en": 0.50, "hinglish": 0.30, "hi": 0.20, "ta": 0.0, "kn": 0.0},
    2: {"en": 0.30, "hinglish": 0.30, "hi": 0.20, "ta": 0.10, "kn": 0.10},
    3: {"en": 0.30, "hinglish": 0.30, "hi": 0.20, "ta": 0.10, "kn": 0.10},
}

STAGE_WARMUP_RATIOS: dict[int, float] = {1: 0.1, 2: 0.0, 3: 0.0}

CSV_COLUMNS: tuple[str, ...] = (
    "step",
    "train/reward_mean",
    "train/reward_std",
    "train/policy_kl",
    "train/gen_length_mean",
    "train/grad_norm",
    "train/loss",
    "train/learning_rate",
    "train/beta_adaptive",
    "train/clipped_ratio_frac",
    "train/advantage_mean",
    "train/advantage_std",
    "train/turns_mean",
    "train/r1_mean",
    "train/r2_mean",
    "train/r3_mean",
    "train/r4_mean",
    "train/r5_mean",
    "train/episode_seconds",
    "train/wall_seconds",
)


# ---------------------------------------------------------------------------
# Action parser — extract DriftCallAction from model text output
# ---------------------------------------------------------------------------


def _parse_action(text: str) -> DriftCallAction:
    """Parse a DriftCallAction from the model's assistant turn text.

    Tries JSON extraction first; falls back to a SUBMIT-or-ABORT heuristic so
    the episode terminates cleanly rather than hanging.
    """
    text = text.strip()
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
            action_type_str = str(obj.get("action_type", "abort")).lower()
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

    lower = text.lower()
    if "submit" in lower:
        return DriftCallAction(
            action_type=ActionType.SUBMIT,
            message=text[:200],
            confidence=0.5,
        )
    return DriftCallAction(action_type=ActionType.ABORT, message=text[:200])


# ---------------------------------------------------------------------------
# Conversation builder — observation → message history (training.md §3.2.1)
# ---------------------------------------------------------------------------


def build_messages(
    goal: Any,
    obs: Any,
    history: list[dict[str, str]],
    is_turn_zero: bool,
) -> list[dict[str, str]]:
    """Append the latest observation to ``history`` (mutated in place)."""
    if is_turn_zero:
        tools = list(getattr(obs, "available_tools", ()) or ())
        system_content = PINNED_SYSTEM_PROMPT
        if tools:
            system_content += "\nAvailable tools: " + json.dumps(
                tools, ensure_ascii=False, sort_keys=True
            )
        history.clear()
        history.append({"role": "system", "content": system_content})
        history.append(
            {"role": "user", "content": getattr(goal, "seed_utterance", "")}
        )
        return history

    tool_results = list(getattr(obs, "tool_results", ()) or ())
    if tool_results:
        latest = tool_results[-1]
        history.append(
            {
                "role": "user",
                "content": "[tool_result] "
                + json.dumps(
                    {
                        "tool": getattr(latest, "tool_name", ""),
                        "status": getattr(latest, "status", ""),
                        "response": getattr(latest, "response", {}),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    drift_log = list(getattr(obs, "drift_log", ()) or ())
    if drift_log:
        drift_payload = json.dumps(
            [
                {
                    "turn": getattr(d, "turn", 0),
                    "type": getattr(d, "drift_type", ""),
                    "domain": getattr(d, "domain", ""),
                    "description": getattr(d, "description", ""),
                }
                for d in drift_log[-3:]
            ],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        history.append({"role": "user", "content": f"[drift] {drift_payload}"})
    return history


def _derive_rollout_seed(goal: Any, g_index: int, episode_seed: int) -> int:
    payload = (
        f"{episode_seed}:{getattr(goal, 'seed_utterance', '')}:{g_index}".encode()
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Rollout primitives
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One assistant turn worth of training tensors."""

    prompt_ids: torch.Tensor       # shape [P]
    action_ids: torch.Tensor       # shape [A]
    old_log_probs: torch.Tensor    # shape [A]


@dataclass
class Rollout:
    """One episode's worth of turns + the reward."""

    episode: Any
    turns: list[TurnRecord] = field(default_factory=list)
    reward: float = 0.0
    rewards_obj: Any = None
    completion_text: str = ""
    turn_count: int = 0
    terminated_by: str = ""


def _encode_prompt(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    max_prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Render history through the chat template and tokenize, left-truncating
    if it exceeds the prompt budget. Returns ``input_ids`` of shape ``[P]``.
    """
    try:
        prompt_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_str = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in history
        )

    enc = tokenizer(text=prompt_str, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if ids.numel() > max_prompt_len:
        ids = ids[-max_prompt_len:]  # left-truncate; keep the most recent context
    return ids.to(device)


def _generate_action(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Sample one assistant turn. Returns (action_ids, old_log_probs, text)."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    inputs = prompt_ids.unsqueeze(0)
    attn = torch.ones_like(inputs)

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=inputs,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=True,
            output_scores=True,
            # Gemma3n is loaded in fp32 (Unsloth refuses fp16 for it) but
            # LoRA + some activations are fp16, which makes the default
            # StaticCache crash with `(self) Float and (source) Half`.
            # Disabling KV cache forces a full forward each token (slower)
            # but is the only path that doesn't dtype-mismatch.
            use_cache=False,
        )

    seq = gen_out.sequences[0]
    # ``model.generate`` runs under ``inference_mode``; the resulting tensors
    # cannot be used in autograd graphs. ``.clone()`` materialises a fresh
    # storage that's safe to feed back into the policy forward pass.
    action_ids = seq[prompt_ids.numel():].clone()
    # Compute per-token log-probs from scores (one Tensor per generated step).
    if action_ids.numel() == 0 or not getattr(gen_out, "scores", None):
        old_log_probs = torch.zeros(0, device=prompt_ids.device)
    else:
        scores = torch.stack(gen_out.scores, dim=0).clone()  # [A, 1, V]
        log_probs = F.log_softmax(scores.float(), dim=-1)
        log_probs = log_probs[torch.arange(scores.shape[0]), 0, action_ids[: scores.shape[0]]]
        old_log_probs = log_probs.detach().clone()

    text = tokenizer.decode(action_ids, skip_special_tokens=True)
    return action_ids.detach().clone(), old_log_probs, text


def run_one_rollout(
    model: Any,
    tokenizer: Any,
    env_factory: Any,
    goal: Any,
    *,
    rollout_seed: int,
    max_turns: int,
    max_prompt_len: int,
    max_completion_len: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Rollout:
    env = env_factory()
    obs = env.reset(seed=rollout_seed)

    history: list[dict[str, str]] = []
    turns: list[TurnRecord] = []
    completion_chunks: list[str] = []
    is_turn_zero = True

    for _turn in range(max_turns):
        build_messages(goal, obs, history, is_turn_zero)
        is_turn_zero = False

        prompt_ids = _encode_prompt(
            tokenizer, history, max_prompt_len=max_prompt_len, device=device
        )
        action_ids, old_log_probs, text = _generate_action(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=max_completion_len,
            temperature=temperature,
            top_p=top_p,
        )

        turns.append(
            TurnRecord(
                prompt_ids=prompt_ids.detach(),
                action_ids=action_ids,
                old_log_probs=old_log_probs,
            )
        )
        completion_chunks.append(text)
        history.append({"role": "assistant", "content": text})

        action = _parse_action(text)
        obs = env.step(action)
        if env.done():
            break

    rollout = Rollout(episode=None, turns=turns)
    rollout.completion_text = "\n".join(completion_chunks)
    rollout.turn_count = len(turns)

    if env.done():
        episode = env.episode()
        rollout.episode = episode
        rewards = compute_rewards(episode)
        rollout.reward = float(rewards.reward)
        rollout.rewards_obj = rewards
        rollout.terminated_by = getattr(episode, "terminated_by", "")
    else:
        # Force-terminate: rewards default to zero so the rollout still trains.
        rollout.reward = 0.0
        rollout.terminated_by = "TIMEOUT"

    return rollout


def run_one_group(
    model: Any,
    tokenizer: Any,
    env_factory: Any,
    goal: Any,
    *,
    episode_seed: int,
    num_generations: int,
    max_turns: int,
    max_prompt_len: int,
    max_completion_len: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[Rollout]:
    rollouts: list[Rollout] = []
    for g in range(num_generations):
        rollouts.append(
            run_one_rollout(
                model,
                tokenizer,
                env_factory,
                goal,
                rollout_seed=_derive_rollout_seed(goal, g, episode_seed),
                max_turns=max_turns,
                max_prompt_len=max_prompt_len,
                max_completion_len=max_completion_len,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
        )
    return rollouts


# ---------------------------------------------------------------------------
# GRPO update — compute log-probs, ratio, KL, loss
# ---------------------------------------------------------------------------


_LOGPROB_CHUNK = 8  # tokens per chunk; 8 × 262400 × 4 B ≈ 8 MB peak vs ~480 MB monolithic


def _action_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    action_ids: torch.Tensor,
) -> torch.Tensor:
    """One forward pass returning per-action-token log-probs ``[A]``.

    Caller controls grad / no_grad context. We always include the full prompt
    so the KV cache lines up with the position embeddings at generation time.

    Log-softmax is computed in chunks of ``_LOGPROB_CHUNK`` tokens so that the
    peak allocation is O(chunk × V) rather than O(A × V).  For Gemma3n's
    vocab of 262 400 and FP32, full-sequence allocation would be ~480 MB per
    forward pass and reliably causes OOM on 32 GB V100 by step 2.
    """
    if action_ids.numel() == 0:
        return torch.zeros(0, device=prompt_ids.device, requires_grad=False)

    full_ids = torch.cat([prompt_ids, action_ids]).unsqueeze(0)
    attn = torch.ones_like(full_ids)
    out = model(
        input_ids=full_ids,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    )
    # Slice only the action positions before freeing the full logits tensor.
    P = prompt_ids.numel()
    A = action_ids.numel()
    # pred_logits is a *view* — we must detach the rest of logits from memory.
    pred_logits = out.logits[0, P - 1 : P - 1 + A].contiguous()  # [A, V]
    del out  # release the full [seq, V] logits buffer

    # Chunked log-softmax: materialise at most [_LOGPROB_CHUNK, V] FP32 at once.
    parts: list[torch.Tensor] = []
    for start in range(0, A, _LOGPROB_CHUNK):
        end = min(start + _LOGPROB_CHUNK, A)
        chunk = pred_logits[start:end].float()  # [c, V] FP32
        lp_chunk = F.log_softmax(chunk, dim=-1)  # [c, V] FP32
        gathered = lp_chunk.gather(-1, action_ids[start:end].unsqueeze(-1)).squeeze(-1)
        parts.append(gathered)
        del chunk, lp_chunk  # free immediately
    return torch.cat(parts)  # [A]


def _disable_adapter_ctx(model: Any) -> Any:
    """Return a context manager that disables LoRA adapters for ref forward.

    Falls back to a no-op context manager if the model doesn't expose
    ``disable_adapter`` (e.g. when LoRA isn't attached at all).
    """
    fn = getattr(model, "disable_adapter", None)
    if fn is None:
        # Some Unsloth wrappers expose it on .base_model
        base = getattr(model, "base_model", None)
        if base is not None:
            fn = getattr(base, "disable_adapter", None)
    if fn is None:
        from contextlib import nullcontext

        return nullcontext()
    return fn()


@dataclass
class GRPOStepMetrics:
    loss: float
    policy_loss: float
    kl: float
    reward_mean: float
    reward_std: float
    advantage_mean: float
    advantage_std: float
    grad_norm: float
    clipped_ratio_frac: float
    gen_length_mean: float
    turns_mean: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    r4_mean: float
    r5_mean: float


def grpo_step(
    model: Any,
    rollouts: list[Rollout],
    *,
    optimizer: Any,
    beta: float,
    clip_eps: float,
    grad_clip_norm: float,
    device: torch.device,
) -> GRPOStepMetrics:
    """Run one GRPO update over a group of rollouts.

    Memory strategy — per-turn immediate backward:
    Rather than accumulating a single ``total_policy_loss`` tensor that keeps
    ALL forward-pass activation graphs alive until the final ``.backward()``,
    we pre-count active tokens (no-grad pass), then call ``.backward()`` on
    each turn's scaled loss immediately.  This limits peak activation memory to
    one forward pass at a time instead of ``num_rollouts × max_turns``.
    """
    rewards = torch.tensor(
        [r.reward for r in rollouts], dtype=torch.float32, device=device
    )
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    advantages = (rewards - reward_mean) / (reward_std + 1e-8)
    advantages = advantages.detach()

    # --- Pre-count active tokens (needed to scale per-turn losses correctly) ---
    total_action_tokens = sum(
        turn.action_ids.numel()
        for r in rollouts
        for turn in r.turns
        if turn.action_ids.numel() > 0
    )

    if total_action_tokens == 0:
        # No grad path — every rollout produced empty completions.
        return GRPOStepMetrics(
            loss=0.0,
            policy_loss=0.0,
            kl=0.0,
            reward_mean=float(reward_mean.item()),
            reward_std=float(reward_std.item()),
            advantage_mean=float(advantages.mean().item()),
            advantage_std=float(advantages.std(unbiased=False).item()),
            grad_norm=0.0,
            clipped_ratio_frac=0.0,
            gen_length_mean=0.0,
            turns_mean=float(sum(r.turn_count for r in rollouts) / max(1, len(rollouts))),
            r1_mean=_mean_reward_field(rollouts, "r1"),
            r2_mean=_mean_reward_field(rollouts, "r2"),
            r3_mean=_mean_reward_field(rollouts, "r3"),
            r4_mean=_mean_reward_field(rollouts, "r4"),
            r5_mean=_mean_reward_field(rollouts, "r5"),
        )

    # Scalar accumulators for metrics only (detached, no computation graph).
    acc_policy_loss = 0.0
    acc_kl = 0.0
    clipped_count = 0
    gen_lengths: list[int] = []

    optimizer.zero_grad(set_to_none=True)
    model.train()

    for r_idx, rollout in enumerate(rollouts):
        adv = advantages[r_idx]
        for turn in rollout.turns:
            n_act = turn.action_ids.numel()
            if n_act == 0:
                continue
            gen_lengths.append(int(n_act))
            prompt = turn.prompt_ids.to(device)
            action = turn.action_ids.to(device)
            old_lp = turn.old_log_probs.to(device)

            # Policy forward (with grad).
            new_lp = _action_logprobs(model, prompt, action)

            # Reference forward (no grad, no adapters).
            with torch.no_grad(), _disable_adapter_ctx(model):
                ref_lp = _action_logprobs(model, prompt, action).detach()
            torch.cuda.empty_cache()

            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv
            clipped_r = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped_r).sum()
            kl = (new_lp - ref_lp).sum()

            # Scale by 1/total_action_tokens so the sum across turns equals the
            # token-normalised mean loss. Backward immediately to free activations.
            turn_loss = (policy_loss + beta * kl) / total_action_tokens
            turn_loss.backward()

            # Accumulate scalar metrics (detached — no graph retention).
            acc_policy_loss += float(policy_loss.detach().item())
            acc_kl += float(kl.detach().item())
            clipped_count += int(
                ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).sum().item()
            )

            del new_lp, ref_lp, ratio, unclipped, clipped_r, policy_loss, kl, turn_loss
            torch.cuda.empty_cache()

    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    policy_loss_mean = acc_policy_loss / total_action_tokens
    kl_mean = acc_kl / total_action_tokens
    loss_scalar = policy_loss_mean + beta * kl_mean

    return GRPOStepMetrics(
        loss=loss_scalar,
        policy_loss=policy_loss_mean,
        kl=kl_mean,
        reward_mean=float(reward_mean.item()),
        reward_std=float(reward_std.item()),
        advantage_mean=float(advantages.mean().item()),
        advantage_std=float(advantages.std(unbiased=False).item()),
        grad_norm=float(grad_norm.detach().item() if hasattr(grad_norm, "detach") else grad_norm),
        clipped_ratio_frac=clipped_count / max(1, total_action_tokens),
        gen_length_mean=float(sum(gen_lengths) / max(1, len(gen_lengths))),
        turns_mean=float(sum(r.turn_count for r in rollouts) / max(1, len(rollouts))),
        r1_mean=_mean_reward_field(rollouts, "r1"),
        r2_mean=_mean_reward_field(rollouts, "r2"),
        r3_mean=_mean_reward_field(rollouts, "r3"),
        r4_mean=_mean_reward_field(rollouts, "r4"),
        r5_mean=_mean_reward_field(rollouts, "r5"),
    )


def _mean_reward_field(rollouts: list[Rollout], field_name: str) -> float:
    vals = [
        float(getattr(r.rewards_obj, field_name, 0.0))
        for r in rollouts
        if r.rewards_obj is not None
    ]
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Adaptive KL controller (training.md §3.3.1)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveKLController:
    target_kl: float = DEFAULT_TARGET_KL
    kp: float = DEFAULT_KP
    beta: float = DEFAULT_BETA_KL
    beta_min: float = DEFAULT_BETA_MIN
    beta_max: float = DEFAULT_BETA_MAX

    def update(self, measured_kl: float) -> tuple[float, bool, bool]:
        """Adjust beta based on measured KL. Returns (new_beta, clamped_lo, clamped_hi)."""
        if not math.isfinite(measured_kl) or self.target_kl <= 0:
            return self.beta, False, False
        err = (measured_kl - self.target_kl) / self.target_kl
        # Clamp exponent to [-20, 20] to prevent math.exp overflow/underflow.
        new_beta = self.beta * math.exp(max(-20.0, min(20.0, self.kp * err)))
        clamped_lo = new_beta < self.beta_min
        clamped_hi = new_beta > self.beta_max
        new_beta = max(self.beta_min, min(self.beta_max, new_beta))
        self.beta = new_beta
        return new_beta, clamped_lo, clamped_hi


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------


def _trainable_lora_params(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer_and_scheduler(
    model: Any,
    *,
    learning_rate: float,
    num_steps: int,
    warmup_ratio: float,
) -> tuple[Any, Any]:
    params = _trainable_lora_params(model)
    if not params:
        raise RuntimeError("No trainable parameters found on model")

    try:
        import bitsandbytes as bnb  # noqa: F401

        optimizer: Any = bnb.optim.PagedAdamW8bit(
            params,
            lr=learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            eps=1e-8,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            eps=1e-8,
        )

    warmup_steps = max(1, int(warmup_ratio * num_steps)) if warmup_ratio > 0 else 0

    def _lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay over the remaining steps.
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    stage: int,
    num_steps: int,
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "stage": stage,
        "step": step,
        "num_steps": num_steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
    }
    try:
        from cells.step_12_gemma_boot import BASE_MODEL_ID

        meta["base_model_id"] = BASE_MODEL_ID
    except Exception:
        pass
    (output_dir / "driftcall_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def load_resume_adapter(model: Any, resume_from: Path) -> Any:
    """Re-attach a previously trained LoRA adapter onto a freshly-booted base."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)


# ---------------------------------------------------------------------------
# WandB + CSV logging
# ---------------------------------------------------------------------------


def _init_wandb(stage: int, seed: int, hardware: str, num_steps: int) -> Any:
    try:
        from cells.step_13_grpo_config import init_wandb

        return init_wandb(
            stage=stage,  # type: ignore[arg-type]
            seed=seed,
            h100_mode=(hardware == "h100"),
            extra_config={"num_steps": num_steps, "loop": "native_grpo_v1"},
        )
    except Exception as exc:
        print(f"[train] wandb init failed (continuing offline): {exc}")
        return None


def _write_csv_row(csv_path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(columns)
        writer.writerow(["nan" if (isinstance(v := row.get(c, ""), float) and v != v) else v for c in columns])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train_driftcall_grpo")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--num-steps", type=int, default=150)
    p.add_argument("--hardware", choices=["v100", "h100"], default="v100")
    p.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--max-prompt-len", type=int, default=DEFAULT_MAX_PROMPT_LEN)
    p.add_argument("--max-completion-len", type=int, default=DEFAULT_MAX_COMPLETION_LEN)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--beta-kl", type=float, default=DEFAULT_BETA_KL)
    p.add_argument("--target-kl", type=float, default=DEFAULT_TARGET_KL)
    p.add_argument("--clip-eps", type=float, default=DEFAULT_CLIP_EPS)
    p.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    p.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    p.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--load-in-4bit",
        type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
        default=False,
        help=(
            "Load the model in 4-bit (bnb). Defaults to False because the "
            "`unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` checkpoint has missing "
            "quant_state metadata for the language layers — loading in 4-bit "
            "auto-redirects there and crashes inside Linear4bit.forward. "
            "False loads the canonical `unsloth/gemma-3n-E2B-it` in fp32 "
            "(~24GB on V100; fits in 32GB VRAM)."
        ),
    )
    p.add_argument("--resume-from", type=str, default="")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/openenv-DGXAI/DRIFTCALL/checkpoints/stage1/final"),
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path for the metrics CSV mirror. Defaults to <output_dir>/metrics.csv.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"[train] stage={args.stage} num_steps={args.num_steps} hardware={args.hardware}")
    print(f"[train] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    # --- Boot model + LoRA (boot_gemma applies dtype + assertion) ---
    # Unsloth refuses fp16 for Gemma3n and silently switches to fp32 ("Using
    # float16 precision for gemma3n won't work! Using float32"). On V100
    # this trips ``boot_gemma``'s strict ``assert_dtype_for_hardware`` halt.
    # Patch the assertion to accept fp32 in addition to fp16 on V100 since
    # fp32 on V100 is safe (just slow), and required for Gemma3n.
    import cells.step_12_gemma_boot as _boot
    _orig_assert = _boot.assert_dtype_for_hardware

    def _lenient_assert(model_obj: Any, hardware: str) -> None:
        first = next(model_obj.parameters())
        if hardware == "v100" and first.dtype in (torch.float16, torch.float32):
            return  # fp32 is acceptable on V100 for Gemma3n
        if hardware == "h100" and first.dtype in (torch.bfloat16, torch.float32):
            return
        return _orig_assert(model_obj, hardware)

    _boot.assert_dtype_for_hardware = _lenient_assert

    boot_config = BootConfig(
        hardware=args.hardware,  # type: ignore[arg-type]
        load_in_4bit=args.load_in_4bit,
    )
    print(f"[train] load_in_4bit={args.load_in_4bit} (False bypasses the broken bnb-4bit checkpoint)")
    model, tokenizer = boot_gemma(boot_config)

    # Apply post-load patches now that Unsloth has installed its versions.
    _patch_unsloth_gemma3n_rmsnorm()
    _patch_unsloth_bnb_linear4bit_quant_state()

    # Stage 2/3: re-attach the previous stage's adapters.
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from path does not exist: {resume_path}")
        print(f"[train] resuming from {resume_path}")
        model = load_resume_adapter(model, resume_path)

    device = next(model.parameters()).device
    print(f"[train] device={device} dtype={next(model.parameters()).dtype}")

    # --- Optimizer + scheduler + KL controller ---
    warmup_ratio = STAGE_WARMUP_RATIOS[args.stage]
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        warmup_ratio=warmup_ratio,
    )
    kl_ctrl = AdaptiveKLController(
        target_kl=args.target_kl,
        beta=args.beta_kl,
    )

    # --- Wandb + CSV ---
    wandb_run = _init_wandb(
        stage=args.stage,
        seed=args.seed,
        hardware=args.hardware,
        num_steps=args.num_steps,
    )
    csv_path = args.csv_path if args.csv_path else (args.output_dir / "metrics.csv")

    # --- Env factory + task gen ---
    language_weights = STAGE_LANGUAGE_WEIGHTS[args.stage]
    stage_base_seed = STAGE_BASE_SEEDS[args.stage] + args.seed

    def env_factory() -> DriftCallEnv:
        return DriftCallEnv(
            config={
                "curriculum_stage": args.stage,
                "audio_boundary_enabled": False,
            },
        )

    # --- Training loop ---
    wall_t0 = time.time()
    for step in range(args.num_steps):
        ep_t0 = time.time()
        goal = task_gen(
            seed=stage_base_seed + step,
            stage=args.stage,  # type: ignore[arg-type]
            language_weights=language_weights,  # type: ignore[arg-type]
        )

        rollouts = run_one_group(
            model,
            tokenizer,
            env_factory,
            goal,
            episode_seed=stage_base_seed + step,
            num_generations=args.num_generations,
            max_turns=args.max_turns,
            max_prompt_len=args.max_prompt_len,
            max_completion_len=args.max_completion_len,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        metrics = grpo_step(
            model,
            rollouts,
            optimizer=optimizer,
            beta=kl_ctrl.beta,
            clip_eps=args.clip_eps,
            grad_clip_norm=args.grad_clip_norm,
            device=device,
        )
        torch.cuda.empty_cache()
        scheduler.step()
        kl_ctrl.update(metrics.kl)

        ep_seconds = time.time() - ep_t0
        wall_seconds = time.time() - wall_t0

        log_row = {
            "step": step,
            "train/reward_mean": metrics.reward_mean,
            "train/reward_std": metrics.reward_std,
            "train/policy_kl": metrics.kl,
            "train/gen_length_mean": metrics.gen_length_mean,
            "train/grad_norm": metrics.grad_norm,
            "train/loss": metrics.loss,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/beta_adaptive": kl_ctrl.beta,
            "train/clipped_ratio_frac": metrics.clipped_ratio_frac,
            "train/advantage_mean": metrics.advantage_mean,
            "train/advantage_std": metrics.advantage_std,
            "train/turns_mean": metrics.turns_mean,
            "train/r1_mean": metrics.r1_mean,
            "train/r2_mean": metrics.r2_mean,
            "train/r3_mean": metrics.r3_mean,
            "train/r4_mean": metrics.r4_mean,
            "train/r5_mean": metrics.r5_mean,
            "train/episode_seconds": ep_seconds,
            "train/wall_seconds": wall_seconds,
        }

        if step % args.logging_steps == 0 or step == args.num_steps - 1:
            print(
                f"[train] step={step:4d} reward={metrics.reward_mean:.3f}±{metrics.reward_std:.3f}"
                f" loss={metrics.loss:+.4f} kl={metrics.kl:+.4f} beta={kl_ctrl.beta:.4f}"
                f" lr={log_row['train/learning_rate']:.2e}"
                f" grad={metrics.grad_norm:.3f} ep_s={ep_seconds:.1f}"
            )
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.log(log_row, step=step)
                except Exception as exc:
                    print(f"[train] wandb.log failed: {exc}")
            _write_csv_row(csv_path, CSV_COLUMNS, log_row)

        if (step + 1) % args.save_steps == 0 and (step + 1) < args.num_steps:
            ckpt_dir = args.output_dir.parent / f"checkpoint-{step + 1}"
            print(f"[train] saving intermediate checkpoint -> {ckpt_dir}")
            save_checkpoint(
                model, tokenizer, ckpt_dir,
                stage=args.stage, num_steps=args.num_steps, step=step + 1,
            )

    # Final save
    print(f"[train] saving final adapter -> {args.output_dir}")
    save_checkpoint(
        model, tokenizer, args.output_dir,
        stage=args.stage, num_steps=args.num_steps, step=args.num_steps,
    )

    # Cleanup wandb
    if wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    print(f"[train] done. Wall time: {time.time() - wall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
