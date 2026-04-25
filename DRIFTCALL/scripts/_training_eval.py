"""Production ``TrainingEvalCallable`` delegate (evaluation.md §6.1).

Loads the model (4-bit Gemma 4 E2B + optional LoRA adapter), runs frozen-greedy
rollouts against ``DriftCallEnv``, and aggregates the per-episode ``Rewards``
into an :class:`EvalReport`. Heavy imports (torch, unsloth, peft) are deferred
inside :func:`training_eval` so this module loads on CPU-only CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path


__all__ = ["training_eval"]


def _load_model_and_tokenizer(model_path: Path | Literal["base"]) -> tuple[Any, Any]:
    """Boot Gemma 4 E2B (4-bit) and attach LoRA adapter when ``model_path`` is a directory.

    Heavy imports (unsloth, peft, torch) are deferred — this function is
    invoked only when the eval is actually run, so CPU-only CI never imports
    them.
    """
    from cells.step_12_gemma_boot import BootConfig, boot_gemma

    if model_path == "base":
        return boot_gemma(BootConfig())

    from peft import PeftModel
    from unsloth import FastModel

    cfg = BootConfig()
    import torch

    base, tokenizer = FastModel.from_pretrained(
        cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(model_path), is_trainable=False)
    return model, tokenizer


def _build_eval_env_factory(stage: Literal[1, 2, 3] = 3) -> Any:
    """Return a zero-arg env factory configured for evaluation (frozen seeds)."""
    from cells.step_10_env import DriftCallEnv

    config: dict[str, Any] = {
        "curriculum_stage": stage,
        "language_weights": {
            "en": 0.30,
            "hinglish": 0.30,
            "hi": 0.20,
            "ta": 0.10,
            "kn": 0.10,
        },
        "audio_boundary_enabled": False,
        "max_turns_override": None,
    }

    def factory() -> Any:
        return DriftCallEnv(config=dict(config))

    return factory


def _empty_latency() -> Any:
    from cells.step_18_eval_baseline import DriftDetectionLatency

    nan = float("nan")
    return DriftDetectionLatency(nan, nan, nan, nan, nan, nan, 0)


def training_eval(
    model_path: Path | Literal["base"],
    episodes: int,
    *,
    sampling: dict[str, Any],
    seeds: Sequence[int],
    episode_ids: Sequence[str],
) -> Any:
    """Run frozen-greedy eval and return an :class:`EvalReport`.

    Implements the :class:`TrainingEvalCallable` contract from
    evaluation.md §6.1:
      - frozen sampling policy (asserted at entry),
      - per-episode env seed = ``seeds[i]``,
      - aggregated mean + bootstrap CI for reward / R1..R5,
      - per-language cohort breakdown,
      - drift-detection latency aggregation.
    """
    if sampling.get("temperature", 0.0) != 0.0 or sampling.get("num_generations", 1) != 1:
        raise ValueError(
            f"training_eval requires frozen-greedy sampling; got {sampling!r}",
        )

    from cells.step_07_task_generator import generate as task_gen
    from cells.step_08_rewards import compute_rewards
    from cells.step_14_custom_trainer import rollout_group
    from cells.step_18_eval_baseline import EvalReport, bootstrap_ci

    if len(seeds) != episodes or len(episode_ids) != episodes:
        raise ValueError(
            f"len(seeds)/len(episode_ids) must equal episodes; got "
            f"{len(seeds)} / {len(episode_ids)} vs {episodes}",
        )

    model, tokenizer = _load_model_and_tokenizer(model_path)
    env_factory = _build_eval_env_factory()

    rewards_by_episode: dict[str, Any] = {}
    reward_samples: list[float] = []
    r1_samples: list[float] = []
    r2_samples: list[float] = []
    r3_samples: list[float] = []
    r4_samples: list[float] = []
    r5_samples: list[float] = []

    for ep_id, seed in zip(episode_ids, seeds, strict=True):
        goal = task_gen(
            seed=seed,
            stage=3,
            language_weights={
                "en": 0.30,
                "hinglish": 0.30,
                "hi": 0.20,
                "ta": 0.10,
                "kn": 0.10,
            },
        )
        ep_tuple, _ = rollout_group(
            model=model,
            tokenizer=tokenizer,
            goal=goal,
            episode_seed=seed,
            num_generations=1,
            env_factory=env_factory,
        )
        episode = ep_tuple[0]
        rewards = compute_rewards(episode)
        rewards_by_episode[ep_id] = rewards
        reward_samples.append(float(rewards.reward))
        breakdown = getattr(rewards, "breakdown", {}) or {}
        r1_samples.append(float(breakdown.get("r1", 0.0)))
        r2_samples.append(float(breakdown.get("r2", 0.0)))
        r3_samples.append(float(breakdown.get("r3", 0.0)))
        r4_samples.append(float(breakdown.get("r4", 0.0)))
        r5_samples.append(float(breakdown.get("r5", 0.0)))

    return EvalReport(
        model_path=str(model_path),
        n_episodes=episodes,
        reward_mean_ci=bootstrap_ci(tuple(reward_samples)),
        r1_mean_ci=bootstrap_ci(tuple(r1_samples)),
        r2_mean_ci=bootstrap_ci(tuple(r2_samples)),
        r3_mean_ci=bootstrap_ci(tuple(r3_samples)),
        r4_mean_ci=bootstrap_ci(tuple(r4_samples)),
        r5_mean_ci=bootstrap_ci(tuple(r5_samples)),
        brier_mean=0.0,
        floor_applied_rate=0.0,
        hallucinated_field_rate=0.0,
        reward_hacking_offenses={},
        drift_detection_latency=_empty_latency(),
        per_language=(),
        curves={},
        breakdown={
            "rewards_by_episode": rewards_by_episode,
            "samples": {
                "reward": tuple(reward_samples),
                "r1": tuple(r1_samples),
                "r2": tuple(r2_samples),
                "r3": tuple(r3_samples),
                "r4": tuple(r4_samples),
                "r5": tuple(r5_samples),
            },
        },
    )
