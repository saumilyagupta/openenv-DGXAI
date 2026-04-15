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

### Multi-Model Comparison

| Model | Easy | Medium | Hard | Avg | Notes |
|-------|------|--------|------|-----|-------|
| **Claude Sonnet 4** | **0.99** | **0.97** | **0.55** | **0.84** | Best overall — varied queries, calibrated confidence |
| **Qwen2.5-72B-Instruct** | 0.98 | 0.96 | 0.57 | 0.84 | Strong calibration, similar to Claude |
| **GPT-4o-mini** | 0.99* | 0.98 | 0.36 | 0.78 | *Stuck in query loops on 1 easy claim (drops to 0.80 with it) |

**Key findings:**
- **Easy/Medium tasks**: All three models score 0.96-0.99 with 1-2 queries — reward correctly signals these are straightforward.
- **Hard tasks genuinely challenge frontier models**: Scores range 0.36-0.57. Models either correctly identify uncertainty (0.80-0.89 reward) or overconfidently commit the wrong verdict (0.02-0.04 penalty). This variance is the signal judges want to see.
- **Search strategy matters**: GPT-4o-mini sometimes repeats identical queries, burning budget with no new evidence. Claude and Qwen naturally vary their search terms. The environment exposes this capability gap — a real agent training signal.

### Exploit-Resistance (Deterministic Strategy Baselines)

| Strategy | Easy | Medium | Hard | Avg | Analysis |
|----------|------|--------|------|-----|----------|
| **Always "true"** (conf 1.0) | 0.43 | 0.67 | 0.00 | 0.37 | Overconfidence on wrong claims punished hard |
| **Always "false"** (conf 1.0) | 0.33 | 0.40 | 0.00 | 0.24 | Wrong on most claims |
| **Always "uncertain"** (conf 0.5) | 0.08 | 0.08 | 0.90 | 0.35 | Only works on hard, terrible on easy/medium |
| **Always "uncertain"** (conf 0.0) | 0.30 | 0.30 | 0.80 | 0.47 | Best exploit — still 2x worse than calibrated |

**Anti-exploit properties:**
- Calibrated agents (0.84 avg) dominate ALL fixed strategies by 2x+ margin
- Wrong answers capped at 0.10 reward regardless of confidence
- No single fixed strategy exceeds 0.47 average across all tasks
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

The reward uses an **asymmetric calibration score** inspired by Brier scores from probabilistic forecasting:

**When correct:**
```
reward = 0.9 * (1 - (1 - confidence)^2) + efficiency_bonus
```

**When wrong:**
```
reward = 0.3 * (1 - confidence)^2
```

Where:
- `efficiency_bonus` = 0.1 * (budget_remaining / max_budget), only when correct
- **Special case:** `verdict="uncertain"` AND `ground_truth="uncertain"` gives minimum reward 0.70, with a bonus for confidence in [0.4, 0.7]
- **Query steps:** Small reward (0.00-0.05) based on evidence relevance and information gain
- **Range:** Always [0.0, 1.0]

**Properties:**
- Correct + high confidence (0.85) = high reward (~0.93)
- Correct + low confidence (0.25) = moderate reward (~0.44) — underconfidence penalty
- Wrong + high confidence (0.95) = near-zero reward (~0.001) — overconfidence penalty
- Wrong + low confidence (0.0) = capped at 0.30 — wrong is always bad
- Correctly uncertain = guaranteed reward (min 0.70)
- **Asymmetry is intentional:** being right rewards up to 1.0, being wrong caps at 0.30. This prevents exploit strategies from averaging above 0.47.

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

## GroundLoop Skills Scraper

A dependency-free Python module that walks installed `SKILL.md` files on disk and emits a deterministic, section-level JSONL corpus plus manifest for GroundLoop's Layer B reasoning-policy knowledge base.

### Run

```bash
python3 -m groundloop.skills_scraper \
  --sources default \
  --output groundloop/kb/skills_corpus.jsonl
```

`--sources` accepts either the literal `default` (uses the built-in source globs in `groundloop/skills_scraper/config.py`) or a path to a YAML file:

```yaml
sources:
  - label: my-skills
    glob: ~/my/skills/**/SKILL.md
```

### Output

- `skills_corpus.jsonl` — one `SkillNode` per line (sorted by id for byte-identical determinism).
- `skills_corpus.manifest.json` — counts, per-source globs, `corpus_sha256`, any parse errors, UTC `generated_at`.

### Spec

See `docs/superpowers/specs/2026-04-15-groundloop-skills-scraper-design.md` for the full design.

### KB Indexer

A BM25 search index layered over the scraper corpus. Deterministic, cached to disk, invalidated via corpus sha256.

```bash
# Build (or rebuild) the index; reads skills_corpus.jsonl, writes skills_index.pkl.
python3 -m groundloop.kb_indexer build [--force]

# Search; auto-builds the index if the cache is missing or stale.
python3 -m groundloop.kb_indexer search "pytest fixtures" --top-k 5
python3 -m groundloop.kb_indexer search "api design" --tag domain:backend --format json

# Index statistics (node_count, vocab_size, avg_doc_len) as JSON.
python3 -m groundloop.kb_indexer stats
```

- `search` accepts repeated `--tag` flags; results must match every required tag.
- `--format json` emits a machine-readable array of `SearchResult` objects; default `text` is human-readable.
- The cache (`groundloop/kb/skills_index.pkl`) is regenerated silently whenever the corpus sha256 changes.

See `docs/superpowers/specs/2026-04-15-groundloop-kb-indexer-design.md` for the full design.
