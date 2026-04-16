"""Baseline inference agent for CodeForge.

Demonstrates a full episode using all 6 action types via the REST API.
Usage: python3 CODEFORGE/inference.py
"""
from __future__ import annotations

import os
import sys

import httpx

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:7860")


def run_episode(task_level: str = "easy") -> dict[str, object]:
    """Run a complete easy episode demonstrating the CodeForge API."""
    with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
        # Reset
        resp = client.post("/reset", json={"task_level": task_level})
        resp.raise_for_status()
        obs = resp.json()
        print(
            f"Episode: {obs['episode_id']}, "
            f"Task: {obs['task_id']}, "
            f"Budget: {obs['budget_remaining']}"
        )

        # Step 1: Query KB
        resp = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "query_kb",
                    "claim": "python greeting function",
                },
            },
        )
        resp.raise_for_status()
        obs = resp.json()
        citations = obs.get("last_citations", [])
        print(
            f"Query KB: {len(citations)} citations, "
            f"Budget: {obs['budget_remaining']}"
        )

        # Step 2: Submit code
        code = (
            'from __future__ import annotations\n\n\n'
            'def greet(name: str) -> str:\n'
            '    return f"Hello, {name}!"\n'
        )
        resp = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "submit",
                    "files": {"main.py": code},
                    "confidence": 0.8,
                },
            },
        )
        resp.raise_for_status()
        obs = resp.json()
        print(
            f"Submit: reward={obs.get('last_reward', 0)}, "
            f"score={obs.get('previous_score', 0)}, "
            f"done={obs.get('is_done', False)}"
        )

        # Step 3: Get audit
        resp = client.post(
            "/step",
            json={"action": {"action_type": "get_audit"}},
        )
        resp.raise_for_status()
        obs = resp.json()
        print(f"Audit: {bool(obs.get('cumulative_audit_summary'))}")

        return obs  # type: ignore[return-value]


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_episode(level)
