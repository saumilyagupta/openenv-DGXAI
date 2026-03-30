import os
import json
import asyncio
from openai import OpenAI
from models import EpistemicAction, ActionType
from client import EpistemicEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_SPACE_URL = os.environ.get('HF_SPACE', 'http://localhost:7860')

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")

SYSTEM_PROMPT = """You are a calibrated reasoning agent.
You will be given a claim to evaluate. You can query an evidence database or commit a verdict.
When committing, express a confidence between 0.0 and 1.0.
If evidence is contradictory or insufficient, verdict should be "uncertain" with confidence ~0.5.
Never guess with high confidence. Being right AND calibrated matters equally."""

def agent_step(claim: str, evidence: list, budget: int) -> dict:
    evidence_text = "\n".join(f"- {e.text}" for e in evidence) or "None yet."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Claim: {claim}

Evidence gathered:
{evidence_text}

Budget remaining: {budget} queries

Respond with JSON only:
{{"action": "query", "query_text": "..."}}
OR
{{"action": "commit", "verdict": "true|false|uncertain", "confidence": 0.0-1.0}}"""}
    ]
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    return json.loads(resp.choices[0].message.content)

def run_episode(env, task_level: str) -> float:
    obs = env.reset(task_level=task_level)
    total_reward = 0.0

    while not obs.done:
        decision = agent_step(obs.observation.claim, [e for e in obs.observation.evidence_gathered], obs.observation.budget_remaining)

        if decision.get("action") == "query" and obs.observation.budget_remaining > 0:
            action = EpistemicAction(
                action_type=ActionType.QUERY,
                query_text=decision.get("query_text", "")
            )
        else:
            action = EpistemicAction(
                action_type=ActionType.COMMIT,
                verdict=decision.get("verdict", "uncertain"),
                confidence=float(decision.get("confidence", 0.5))
            )

        obs = env.step(action)
        total_reward += obs.reward or 0.0

    return total_reward

def main():
    results = {"easy": [], "medium": [], "hard": []}
    with EpistemicEnv(base_url=HF_SPACE_URL) as env:
        for task_level in ["easy", "medium", "hard"]:
            for _ in range(2): # reduced to 2 for quick testing
                reward = run_episode(env, task_level)
                results[task_level].append(reward)

    print(json.dumps({
        "easy_mean":   round(sum(results["easy"])   / len(results["easy"]),   3) if results["easy"] else 0,
        "medium_mean": round(sum(results["medium"]) / len(results["medium"]), 3) if results["medium"] else 0,
        "hard_mean":   round(sum(results["hard"])   / len(results["hard"]),   3) if results["hard"] else 0,
    }, indent=2))

if __name__ == "__main__":
    main()
