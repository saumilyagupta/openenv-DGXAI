---
title: EpistemicNav
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# EpistemicNav — Adaptive Inquiry Agent Environment

An RL environment that trains LLM agents to reason accurately under uncertainty — rewarding calibrated confidence, not just correct answers.

## Concept

LLMs are systematically miscalibrated — overconfident when wrong, underconfident when right. No existing OpenEnv environment trains this. EpistemicNav fills that exact gap.

The agent is given a factual claim and a budget of queries. It can search a local evidence corpus (BM25 retrieval) or commit a verdict with a confidence score. The reward is the Brier score — a mathematical formula that penalises both wrong answers and miscalibrated confidence simultaneously.

## Action Space

The agent can perform two types of actions:

1. **`QUERY`**: Provide `query_text` to search the local evidence database using BM25. Deducts 1 from the budget. Reward is 0.0. Episode continues.
2. **`COMMIT`**: Provide a `verdict` ("true", "false", or "uncertain") and a `confidence` score (0.0 to 1.0). Calculates the Brier score reward. Ends the episode.

## Observation Space

At each step, the agent observes:
- `claim`: The factual statement to evaluate.
- `evidence_gathered`: A list of evidence snippets retrieved so far.
- `budget_remaining`: Number of queries remaining.
- `task_level`: The difficulty of the task ("easy", "medium", "hard").
- `episode_id`: Unique identifier for the episode.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```

2. **Run the FastAPI server locally:**
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 7860
   ```

3. **Run the inference script:**
   In another terminal, run:
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o-mini"
   export HF_TOKEN="your-token"
   python inference.py
   ```

## Task Descriptions

- **Task 1 — easy (`single_hop`)**: Single-hop factual claim. One query is sufficient to retrieve the answer.
- **Task 2 — medium (`multi_hop`)**: Multi-hop claim requiring synthesis of 3–4 evidence pieces.
- **Task 3 — hard (`contradictory`)**: Contradictory evidence. The ground truth is `"uncertain"`. The agent must recognise that evidence conflicts and express genuine uncertainty.
