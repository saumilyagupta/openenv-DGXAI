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

## Action Space

| Action | Fields | Effect |
|--------|--------|--------|
| `QUERY` | `query_text: str` | Searches evidence corpus via BM25. Costs 1 budget. Returns top-3 snippets. Reward: 0.0 |
| `COMMIT` | `verdict: "true"\|"false"\|"uncertain"`, `confidence: float [0,1]` | Ends episode. Computes Brier-score reward in [0.0, 1.0]. |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The factual statement to evaluate |
| `evidence_gathered` | `list[EvidenceSnippet]` | Evidence retrieved so far (id, text, relevance_score) |
| `budget_remaining` | `int` | Queries remaining (starts at 8) |
| `task_level` | `str` | "easy" / "medium" / "hard" |
| `episode_id` | `str` | Unique episode identifier |
| `is_done` | `bool` | Whether episode has ended |
| `last_reward` | `float \| null` | Reward from last action |

## Reward Function

- **Formula:** `reward = 0.9 * (1 - (confidence - correct)^2) + efficiency_bonus`
- Where `correct` = 1.0 if verdict matches ground_truth, else 0.0
- `efficiency_bonus` = 0.1 * (budget_remaining / max_budget) only when correct
- **Special case:** verdict="uncertain" AND ground_truth="uncertain" gives min reward 0.70, with bonus for confidence in [0.4, 0.7]
- **Range:** always [0.0, 1.0]

## Tasks

1. **easy (single_hop):** Single-hop factual claim. One BM25 query sufficient. Reward ceiling ~0.98.
2. **medium (multi_hop):** Multi-hop claim requiring synthesis of 3-4 evidence pieces. Reward ceiling ~0.88.
3. **hard (contradictory):** Contradictory evidence. Ground truth is "uncertain". Agent must recognize conflicting evidence. Reward floor 0.70.

## Data

400 claims (200 easy, 150 medium, 50 hard), 2000 evidence snippets, 15+ domains.

## Setup Instructions

**Docker:**
```bash
docker build -t epistemic-nav . && docker run -p 7860:7860 epistemic-nav
```

**Local:**
```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Inference:**
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token"
python inference.py
```

## API Endpoints

- `POST /reset` — Reset environment. Body: `{"task_level": "easy|medium|hard"}`
- `POST /step` — Take action. Body: EpistemicAction JSON
- `GET /state` — Get current observation
- `GET /` — Health check

## Validation

```bash
python scripts/validate.py
```
