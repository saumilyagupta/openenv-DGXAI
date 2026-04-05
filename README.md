---
title: EpistemicNav
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# EpistemicNav — Calibrated Reasoning Under Uncertainty

**The problem:** LLMs are dangerously miscalibrated. They express 95% confidence when wrong and hedge when right. Existing benchmarks measure *accuracy* — whether the model got the answer right. None measure *calibration* — whether the model's confidence matched its actual likelihood of being correct.

**EpistemicNav** is an OpenEnv-compliant RL environment that trains and evaluates LLM agents on **epistemic calibration**. Agents must gather evidence, assess its quality, and commit a verdict with a confidence score. The reward function uses the **Brier score** — a proper scoring rule from forecasting that penalizes both overconfidence on wrong answers *and* underconfidence on right ones.

The key insight: **"I don't know" is sometimes the right answer.** When evidence is contradictory, agents that correctly identify uncertainty are rewarded (min 0.70), while agents that guess confidently are punished.

## Why This Matters

| Problem | How EpistemicNav Addresses It |
|---------|------------------------------|
| LLMs hallucinate with high confidence | Brier score penalizes overconfidence on wrong answers |
| LLMs hedge on clear facts | Brier score penalizes underconfidence on right answers |
| No benchmark for calibration | First OpenEnv environment specifically targeting epistemic calibration |
| Binary correct/incorrect rewards | Continuous reward signal [0.0, 1.0] based on confidence-accuracy alignment |
| Agents can't say "I don't know" | `uncertain` is a first-class verdict with explicit reward floor |

## Tasks

| Task | ID | Difficulty | Description | Reward Ceiling |
|------|----|-----------|-------------|----------------|
| **Single-hop** | `single_hop` | Easy | Single factual claim. One BM25 query sufficient to find supporting/refuting evidence. | ~0.97 |
| **Multi-hop** | `multi_hop` | Medium | Claim requiring synthesis of 3-4 evidence pieces across related topics. | ~0.94 |
| **Contradictory** | `contradictory` | Hard | Deliberately conflicting evidence. Ground truth is `uncertain`. Agent must recognize conflict. | ~0.85 |

## Baseline Scores

### LLM Agent (Qwen2.5-72B-Instruct via HuggingFace Router)

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| **Mean reward** | **0.98** | **0.96** | **0.57** |
| Avg queries used | 1 | 1.5 | 2 |
| Typical verdict | true/false (high conf) | true/false (high conf) | mixed — uncertain or wrong |

The agent scores near-perfectly on easy/medium tasks with minimal queries. Hard tasks expose real calibration failures: Qwen correctly identified uncertainty on one claim (0.89) but overconfidently committed "false" on another contradictory claim (0.25) — demonstrating the environment genuinely challenges frontier models.

### Exploit-Resistance (Deterministic Strategy Baselines)

| Strategy | Easy | Medium | Hard | Analysis |
|----------|------|--------|------|----------|
| **Always "true"** (confidence 1.0) | 0.35 | 0.70 | 0.00 | Punished for overconfidence on false/uncertain claims |
| **Always "uncertain"** (confidence 0.5) | 0.68 | 0.68 | 0.90 | Good on hard, poor on easy — can't exploit by hedging everything |
| **Random verdict** (confidence 0.5) | ~0.55 | ~0.55 | ~0.68 | No strategy dominates without reasoning |

**Anti-exploit properties:**
- No single fixed strategy dominates all tasks
- Maximum reward requires reading evidence and calibrating confidence per-claim
- Gaming the grader (always same verdict) produces suboptimal scores
- Hard tasks specifically reward agents that can say "I don't know"

## Action Space

| Action | Fields | Effect |
|--------|--------|--------|
| `QUERY` | `query_text: str` | Searches evidence corpus via BM25. Costs 1 budget point. Returns top-3 snippets. Reward: 0.00-0.05 (based on evidence relevance and novelty) |
| `COMMIT` | `verdict: "true"\|"false"\|"uncertain"`, `confidence: float [0,1]` | Ends episode. Computes Brier-score reward in [0.0, 1.0] |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The factual statement to evaluate |
| `evidence_gathered` | `list[EvidenceSnippet]` | Evidence retrieved so far (id, text, relevance_score) |
| `budget_remaining` | `int` | Queries remaining (starts at 8) |
| `task_level` | `str` | `"easy"` / `"medium"` / `"hard"` |
| `episode_id` | `str` | Unique episode identifier |
| `is_done` | `bool` | Whether episode has ended |
| `last_reward` | `float \| null` | Reward from last action |

## Reward Function

The reward uses the **Brier score**, a proper scoring rule from probabilistic forecasting:

```
reward = 0.9 * (1 - (confidence - correct)^2) + efficiency_bonus
```

Where:
- `correct` = 1.0 if verdict matches ground truth, else 0.0
- `efficiency_bonus` = 0.1 * (budget_remaining / max_budget), only when correct
- **Special case:** `verdict="uncertain"` AND `ground_truth="uncertain"` gives minimum reward 0.70, with a bonus for confidence in [0.4, 0.7]
- **Query steps:** Small reward (0.00-0.05) based on evidence relevance and information gain
- **Range:** Always [0.0, 1.0]

**Properties:**
- Overconfident + wrong = very low reward (~0.09 for 0.95 confidence on wrong answer)
- Underconfident + right = moderate reward (~0.56 for 0.25 confidence on right answer)
- Calibrated + right = high reward (~0.97 for 0.85 confidence on right answer)
- Correctly uncertain = guaranteed reward (min 0.70)

## Data

- **400 claims** across 15 domains: astronomy, biology, economics, engineering, geography, history, law, linguistics, mathematics, medicine, nutrition, politics, science, sports, technology
- **2000 evidence snippets** with relevance tags for BM25 retrieval
- **Distribution:** 200 easy (single-hop), 150 medium (multi-hop), 50 hard (contradictory)

## Setup Instructions

### Docker (Recommended)

```bash
docker build -t epistemic-nav .
docker run -p 7860:7860 epistemic-nav
```

### Local

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export HF_SPACE="http://localhost:7860"
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment. Body: `{"task_level": "easy\|medium\|hard"}` |
| `/step` | POST | Take action. Body: EpistemicAction JSON |
| `/state` | GET | Get current observation |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | GET | Get grader score for current/completed episode |
| `/baseline` | POST | Trigger inference script and return baseline scores |
| `/` | GET | Health check |

## Validation

```bash
# Run local validation (starts server automatically)
python scripts/validate.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
├── server/
│   ├── app.py            # FastAPI app with all endpoints
│   ├── environment.py     # EpistemicNavEnvironment (step/reset/state)
│   ├── grader.py          # Brier score reward function
│   └── retriever.py       # BM25 evidence search
├── models.py              # Pydantic v2 Action/Observation models
├── client.py              # HTTP client for remote env
├── inference.py           # Baseline LLM agent
├── data/
│   ├── claims.json        # 400 claims (200 easy, 150 medium, 50 hard)
│   └── evidence.json      # 2000 evidence snippets
├── tests/                 # pytest suite (grader, retriever, environment)
├── scripts/
│   ├── validate.py        # Pre-submission validation
│   └── generate_data.py   # Data generation script
├── openenv.yaml           # OpenEnv metadata
└── Dockerfile             # HF Spaces deployment
```
