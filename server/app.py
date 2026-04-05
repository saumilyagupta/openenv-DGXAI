from fastapi import FastAPI, Response
import logging
import os
import subprocess
import sys
import json

from openenv.core.env_server.http_server import create_app
from server.environment import EpistemicNavEnvironment
from server.grader import compute_reward
from models import EpistemicAction, EpistemicObservation, ActionType

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

_env_instance = EpistemicNavEnvironment()
app = create_app(lambda: _env_instance, EpistemicAction, EpistemicObservation)


@app.get("/", summary="Health check", description="Returns environment name and status. Used by automated validators to confirm the Space is live.")
def root() -> dict:
    return {
        "name": "epistemic-nav",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/tasks", summary="List tasks and action schema", description="Returns all available tasks (easy/medium/hard) with their IDs, descriptions, and the full action schema showing required fields for QUERY and COMMIT actions.")
def get_tasks() -> dict:
    return {
        "tasks": [
            {
                "id": "single_hop",
                "description": "Single-hop factual claim. One query sufficient.",
                "difficulty": "easy",
            },
            {
                "id": "multi_hop",
                "description": "Multi-hop claim requiring evidence synthesis from multiple sources.",
                "difficulty": "medium",
            },
            {
                "id": "contradictory",
                "description": "Contradictory evidence. Correct answer is uncertain.",
                "difficulty": "hard",
            },
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "enum": ["query", "commit"],
                "description": "QUERY to search evidence, COMMIT to submit verdict.",
            },
            "query_text": {
                "type": "string",
                "description": "Search query for evidence retrieval. Required when action_type=query.",
            },
            "verdict": {
                "type": "string",
                "enum": ["true", "false", "uncertain"],
                "description": "Verdict on the claim. Required when action_type=commit.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the verdict. Required when action_type=commit.",
            },
        },
    }


@app.get("/grader", summary="Get grader score", description="Returns the grader score for the current or most recently completed episode. Shows episode status, claim, reward, evidence count, and budget usage.")
def get_grader() -> dict:
    env = _env_instance
    if env.current_claim is None:
        return {"error": "No episode in progress or completed. Call /reset first."}

    obs = env.state
    if not obs.is_done:
        return {
            "status": "episode_in_progress",
            "claim": obs.claim,
            "budget_remaining": obs.budget_remaining,
            "evidence_count": len(obs.evidence_gathered),
            "last_reward": obs.last_reward,
        }

    return {
        "status": "episode_completed",
        "claim": obs.claim,
        "task_level": obs.task_level,
        "last_reward": obs.last_reward,
        "evidence_count": len(obs.evidence_gathered),
        "budget_used": env.max_budget - obs.budget_remaining,
    }


@app.post("/baseline", summary="Run baseline inference", description="Triggers the baseline inference script (inference.py) and returns scores for all 3 tasks. May take several minutes depending on LLM API latency. Requires API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables.")
def run_baseline() -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_path = os.path.join(project_root, "inference.py")

    try:
        result = subprocess.run(
            [sys.executable, inference_path],
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=project_root,
            env={**os.environ, "HF_SPACE": "http://localhost:7860"},
        )

        lines = result.stdout.strip().split("\n")
        scores: dict[str, list[float]] = {}
        for line in lines:
            if line.startswith("[END]"):
                parts = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                score = float(parts.get("score", "0"))
                task = "unknown"
                for prev in reversed(lines[:lines.index(line)]):
                    if prev.startswith("[START]"):
                        start_parts = dict(
                            p.split("=", 1) for p in prev.split()[1:] if "=" in p
                        )
                        task = start_parts.get("task", "unknown")
                        break
                scores.setdefault(task, []).append(score)

        summary = {
            task: round(sum(s) / len(s), 4) if s else 0.0
            for task, s in scores.items()
        }

        return {
            "status": "completed",
            "scores": summary,
            "raw_stdout": result.stdout,
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Baseline script exceeded 20 minute limit"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
