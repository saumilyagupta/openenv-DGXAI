"""
EpistemicNav baseline inference script.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import json
import os
import sys
import logging
from typing import Optional

from openai import OpenAI
from client import EpistemicEnv
from models import EpistemicAction, ActionType

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
HF_SPACE_URL = os.getenv("HF_SPACE", "http://localhost:7860")

BENCHMARK = "epistemic-nav"
TASK_MAP = {
    "easy": "single_hop",
    "medium": "multi_hop",
    "hard": "contradictory",
}
MAX_STEPS = 8
EPISODES_PER_TASK = 5
TEMPERATURE = 0.1

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference")

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")

SYSTEM_PROMPT = """You are a calibrated reasoning agent.
You will be given a claim to evaluate. You can query an evidence database or commit a verdict.
When committing, express a confidence between 0.0 and 1.0.
If evidence is contradictory or insufficient, verdict should be "uncertain" with confidence ~0.5.
Never guess with high confidence. Being right AND calibrated matters equally.

Respond with JSON only:
{"action": "query", "query_text": "..."}
OR
{"action": "commit", "verdict": "true|false|uncertain", "confidence": 0.0-1.0}"""


# ---------------------------------------------------------------------------
# Mandatory stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def agent_step(claim: str, evidence: list, budget: int) -> dict:
    evidence_text = "\n".join(f"- {e.text}" for e in evidence) or "None yet."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {claim}\n\nEvidence gathered:\n{evidence_text}\n\nBudget remaining: {budget} queries"},
    ]
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {"action": "commit", "verdict": "uncertain", "confidence": 0.5}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task_level: str) -> float:
    task_name = TASK_MAP.get(task_level, task_level)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_level=task_level)
    rewards: list[float] = []
    step_idx = 0
    last_error: Optional[str] = None

    try:
        while not obs.done and step_idx < MAX_STEPS:
            step_idx += 1
            decision = agent_step(
                obs.observation.claim,
                list(obs.observation.evidence_gathered),
                obs.observation.budget_remaining,
            )

            if decision.get("action") == "query" and obs.observation.budget_remaining > 0:
                action = EpistemicAction(
                    action_type=ActionType.QUERY,
                    query_text=decision.get("query_text", ""),
                )
                action_str = f"query('{decision.get('query_text', '')}')"
            else:
                verdict = decision.get("verdict", "uncertain")
                confidence = float(decision.get("confidence", 0.5))
                action = EpistemicAction(
                    action_type=ActionType.COMMIT,
                    verdict=verdict,
                    confidence=confidence,
                )
                action_str = f"commit('{verdict}',{confidence:.2f})"

            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)
            last_error = getattr(obs, "last_action_error", None)

            log_step(
                step=step_idx,
                action=action_str,
                reward=reward,
                done=obs.done,
                error=last_error,
            )

        # Force commit if we hit MAX_STEPS without finishing
        if not obs.done:
            step_idx += 1
            action = EpistemicAction(
                action_type=ActionType.COMMIT,
                verdict="uncertain",
                confidence=0.5,
            )
            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)
            log_step(
                step=step_idx,
                action="commit('uncertain',0.50)",
                reward=reward,
                done=obs.done,
                error=None,
            )

        score = rewards[-1] if rewards else 0.0
        success = score > 0.0
        log_end(success=success, steps=step_idx, score=score, rewards=rewards)
        return score

    except Exception as exc:
        log_end(success=False, steps=step_idx, score=0.0, rewards=rewards)
        logger.error("Episode failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from contextlib import contextmanager

    @contextmanager
    def open_env(base_url: str):
        client = EpistemicEnv(base_url=base_url)
        sync_method = getattr(client, "sync", None)
        if callable(sync_method):
            with sync_method() as env:
                yield env
        else:
            with client as env:
                yield env

    all_scores: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}

    with open_env(base_url=HF_SPACE_URL) as env:
        for task_level in ["easy", "medium", "hard"]:
            for _ in range(EPISODES_PER_TASK):
                score = run_episode(env, task_level)
                all_scores[task_level].append(score)

    summary = {
        level: round(sum(scores) / len(scores), 2) if scores else 0.0
        for level, scores in all_scores.items()
    }
    logger.info("Final scores: %s", summary)


if __name__ == "__main__":
    main()
