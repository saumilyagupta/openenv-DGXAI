"""CLI runner: connect to the env Space, load the policy, run N episodes.

Usage::

    python -m deploy.inference.run \\
        --env-url https://dgxai-driftcall-env.hf.space \\
        --adapter-id DGXAI/gemma-3n-e2b-driftcall-lora \\
        --num-episodes 5 --seed 42 --curriculum-stage 2 \\
        --out-jsonl runs/inference.jsonl

Environment variables (lower priority than CLI flags):

    DRIFTCALL_ENV_URL    base URL of the env Space
    DRIFTCALL_ENV_TOKEN  bearer token
    HF_TOKEN             needed to download a private adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from deploy.inference.client import (
    DEFAULT_ENV_URL,
    DriftCallGymClient,
    GymCapacityError,
    GymClientError,
)
from deploy.inference.policy import (
    DEFAULT_ADAPTER_ID,
    DEFAULT_BASE_MODEL_ID,
    GemmaPolicy,
    PolicyConfig,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy.inference.run",
        description="Run inference episodes against the deployed DriftCall env Space.",
    )
    # Env Space connection
    p.add_argument("--env-url", type=str, default=None, help=f"default: {DEFAULT_ENV_URL}")
    p.add_argument("--env-token", type=str, default=None, help="DRIFTCALL_ENV_TOKEN")

    # Policy
    p.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL_ID)
    p.add_argument("--adapter-id", type=str, default=DEFAULT_ADAPTER_ID,
                   help='LoRA adapter to load. Pass "" to evaluate untrained baseline.')
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    # Episode / loop
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum-stage", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--max-turns", type=int, default=16,
                   help="Hard cap; the env enforces 16, but we stop early if needed.")

    # Output
    p.add_argument("--out-jsonl", type=Path, default=None,
                   help="If set, append one JSON line per episode (trace + reward).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def run_episode(
    client: DriftCallGymClient,
    policy: GemmaPolicy,
    *,
    seed: int,
    curriculum_stage: int,
    max_turns: int,
) -> dict[str, Any]:
    """Run a single episode and return a trace + summary dict."""
    t0 = time.time()
    obs, info = client.reset(seed=seed, curriculum_stage=curriculum_stage)
    trace: list[dict[str, Any]] = [{"actor": "env", "event": "reset", "info": info}]

    total_reward = 0.0
    terminated = truncated = False
    final_info: dict[str, Any] = {}
    for turn in range(max_turns):
        action = policy.act(obs)
        result = client.step(action)
        trace.append({
            "actor": "agent", "turn": turn, "action": action,
            "reward": result.reward, "terminated": result.terminated,
            "truncated": result.truncated,
        })
        total_reward += result.reward
        obs = result.observation
        terminated, truncated = result.terminated, result.truncated
        final_info = result.info
        if terminated or truncated:
            break

    return {
        "session_id": client.session_id,
        "seed": seed,
        "curriculum_stage": curriculum_stage,
        "turns": len([t for t in trace if t.get("actor") == "agent"]),
        "total_reward": round(total_reward, 6),
        "terminated": terminated,
        "truncated": truncated,
        "wall_seconds": round(time.time() - t0, 3),
        "final_info": final_info,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    policy = GemmaPolicy(PolicyConfig(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id or None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ))

    out_fh = None
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out_jsonl.open("a", encoding="utf-8")

    rewards: list[float] = []
    try:
        for ep in range(args.num_episodes):
            seed = args.seed + ep
            with DriftCallGymClient(env_url=args.env_url, auth_token=args.env_token) as client:
                # Health probe so we fail fast on a cold/down Space.
                try:
                    health = client.healthz()
                except GymClientError as exc:
                    print(f"[run] /healthz failed: {exc}", file=sys.stderr)
                    return 2
                if health != "ok":
                    print(f"[run] /healthz returned {health!r}", file=sys.stderr)
                    return 2

                try:
                    summary = run_episode(
                        client, policy,
                        seed=seed,
                        curriculum_stage=args.curriculum_stage,
                        max_turns=args.max_turns,
                    )
                except GymCapacityError as exc:
                    print(f"[run] env Space at capacity (M5): {exc}", file=sys.stderr)
                    return 3

            rewards.append(summary["total_reward"])
            print(
                f"[run] ep={ep} seed={seed} stage={args.curriculum_stage} "
                f"turns={summary['turns']} reward={summary['total_reward']:.4f} "
                f"term={summary['terminated']} trunc={summary['truncated']} "
                f"t={summary['wall_seconds']:.2f}s"
            )
            if out_fh is not None:
                out_fh.write(json.dumps(summary) + "\n")
                out_fh.flush()
    finally:
        if out_fh is not None:
            out_fh.close()

    if rewards:
        mean = sum(rewards) / len(rewards)
        print(f"[run] mean_reward={mean:.4f} over {len(rewards)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
