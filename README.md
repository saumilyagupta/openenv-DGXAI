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

## MCP Shell (Server)

`groundloop.mcp_shell` is a stdio MCP server that exposes the 5 GroundLoop tools to any MCP-compatible client (Claude Code, Cursor, Codex). Session state (graphs, runs, metrics) is held per-process in memory.

To attach GroundLoop as an MCP server in Claude Code, add the following to your MCP config:

```json
{
  "mcpServers": {
    "groundloop": {
      "command": "python3",
      "args": ["-m", "groundloop.mcp_shell"]
    }
  }
}
```

The five tools registered:

- `interrogate(brief)` — returns 3 Socratic clarifying questions about a project brief (stub until #5).
- `ingest_sources(source_globs)` — scrapes + indexes skill sources, returns a `graph_id` for subsequent `ground_check` calls. Pass `null` to load the default pre-built corpus at `groundloop/kb/skills_corpus.jsonl`.
- `ground_check(claim, graph_id, top_k, required_tags)` — BM25 search over a built graph; returns `verdict` (`grounded` / `uncertain` / `ungrounded`), citations, and softmax `confidence`.
- `autonomous_build(spec, graph_id, max_iters)` — registers a Ralph-loop run (stub until #7 ships); returns a `run_id`.
- `audit_report(run_id)` — returns structured run metadata and session metrics for the given `run_id`.

See `docs/superpowers/specs/2026-04-15-groundloop-mcp-shell-design.md` for the full design.

### Python Sandbox

`groundloop.python_sandbox` runs `ruff`, `mypy`, `pytest`, and an AST-based import probe against a candidate Python project, returning a structured `SandboxResult` plus a composite score in `[0.0, 1.0]` used by the Ralph loop.

```bash
# Score a project on disk (runs all default tools).
python3 -m groundloop.python_sandbox path/to/project

# JSON output, imports-only — useful for fast iteration.
python3 -m groundloop.python_sandbox path/to/project --format json --tool imports

# Select specific tools.
python3 -m groundloop.python_sandbox path/to/project --tool ruff --tool mypy --tool pytest
```

Programmatic use:

```python
from groundloop.python_sandbox import run_sandbox

# From a project directory:
result = run_sandbox(project_dir="./my_project")
print(result.composite_score, result.imports.unresolved)

# From an in-memory files dict:
result = run_sandbox(files={"main.py": "def f() -> int:\n    return 1\n"}, tools=("imports",))
```

The composite score penalises unresolved imports, ruff violations, mypy errors, and pytest failures. Missing tool binaries degrade gracefully (reported as `unavailable`).

See `docs/superpowers/specs/2026-04-15-groundloop-python-sandbox-design.md` for the full design.

### Ralph Orchestrator (autonomous loop)

The Ralph orchestrator runs a plan -> synthesize -> sandbox-score -> keep-or-revert loop over a skills KB, producing iteratively improved code files checkpointed to JSON.

Two synthesizer backends:

- **stub** (default): deterministic, no LLM; pulls fenced Python blocks from KB citations. Used in tests and offline runs.
- **openai**: uses `openai` SDK. Requires `OPENAI_API_KEY`. Optional: `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME` (defaults to `gpt-4o-mini`).

Run via CLI:

```bash
python3 -m groundloop.ralph_orchestrator run path/to/spec.txt \
  --corpus path/to/skills_corpus.jsonl \
  --initial-file main.py=path/to/main.py \
  --max-iters 5 --target-score 0.95 --synthesizer stub --format json
```

Programmatic use:

```python
from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator import LoopConfig, StubSynthesizer, run_loop

idx = SkillsIndex(corpus_path="corpus.jsonl")
idx.build()
result = run_loop(
    spec="Build a greet function",
    initial_files={"main.py": "def greet(n): return 'hi'\n"},
    index=idx,
    synthesizer=StubSynthesizer(),
    config=LoopConfig(max_iters=5),
)
print(result.terminated_by, result.final_score)
```

See `docs/superpowers/specs/2026-04-15-groundloop-ralph-orchestrator-design.md` for the full design.

## End-to-End GroundLoop

The five sub-projects (skills-scraper, kb-indexer, mcp-shell, python-sandbox,
ralph-orchestrator) plus lib-grounder, interrogator, and audit-reporter
compose into a single MCP-driven workflow.

1. **Install deps.**
   ```bash
   pip install -r requirements.txt
   ```
2. **Build the Layer B corpus** from your skill library.
   ```bash
   python3 -m groundloop.skills_scraper
   ```
   Writes `groundloop/kb/skills_corpus.jsonl`.
3. **Start the MCP server.**
   ```bash
   python3 -m groundloop.mcp_shell
   ```
   The server exposes five tools over stdio: `interrogate`, `ingest_sources`,
   `ground_check`, `autonomous_build`, `audit_report`.
4. **From an MCP client (Claude Code, Cursor, Codex, or the in-process
   `dispatch` function for testing)**, chain the tools:
   - `interrogate(brief)` — Socratic questions grounded in the skill KB.
   - `ingest_sources(source_globs=None)` — builds a graph from the default
     corpus; returns `graph_id`.
   - `ground_check(claim, graph_id)` — Layer-A (AST symbol grounding via
     `lib_grounder`) plus Layer-B (BM25 over skills). Returns verdict,
     citations, confidence, and when the claim contains code a `layer_a`
     sub-dict.
   - `autonomous_build(spec, graph_id)` — runs the ralph-orchestrator loop
     against the graph; returns a `run_id` and final files.
   - `audit_report(run_id)` — structured audit: iteration reasons, skill
     citations, score trajectory, termination reason.

### Quick in-process smoke test

```python
from groundloop.mcp_shell.server import dispatch
from groundloop.mcp_shell.session import SessionState

s = SessionState()
print(dispatch("interrogate", {"brief": "build a REST API"}, s))
r = dispatch("ingest_sources", {"source_globs": None}, s)
gid = r["graph_id"]
gc = dispatch("ground_check", {
    "claim": "```python\nimport os\nos.getcwd()\n```",
    "graph_id": gid,
}, s)
ab = dispatch("autonomous_build", {
    "spec": "build greet(name)", "graph_id": gid, "max_iters": 1,
}, s)
ar = dispatch("audit_report", {"run_id": ab["run_id"]}, s)
```

### Full verification

```bash
python3 -m pytest tests/groundloop/ --cov=groundloop --cov-report=term -q
ruff check groundloop/
mypy --strict groundloop/
```

Targets: 210+ tests pass, 85%+ overall coverage, zero lint or type errors.

---

## CodeForge OpenEnv (Round 2)

**CodeForge** is the Round-2 OpenEnv-compliant RL environment (`groundloop_env/`) built on top of the shipped GroundLoop primitives (`python_sandbox`, `kb_indexer`, `lib_grounder`, `ralph_orchestrator`). Agents receive a natural-language brief and iteratively synthesize a small Python codebase. Rewards combine a programmatic quality signal (ruff / mypy / pytest / import resolution) with AST-level symbol grounding.

> Round-1 EpistemicNav env still lives under `server/` and is unchanged.

### Run the server locally

```bash
uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860
```

Endpoints:
- `GET /` — health
- `GET /tasks` — list tasks + action schema
- `POST /reset` — `{ "task_level": "easy" | "medium" | "hard" }`
- `POST /step` — `CodeForgeAction` (QUERY_KB or SUBMIT)
- `GET /state` — current observation

### Run the baseline agent

```bash
python3 inference.py
```

Drives all three tasks (easy / medium / hard) end-to-end against `API_BASE_URL` (default `http://localhost:7860`) using stub solutions.

### Docker

```bash
docker build -t code-forge .
docker run -p 7860:7860 code-forge
```

### Tasks

| Level  | ID                   | Budget | Target score | Brief                                                 |
|--------|----------------------|--------|--------------|-------------------------------------------------------|
| easy   | greet_single_file    | 4      | 0.90         | Single-file `greet(name)` with type hints.            |
| medium | greet_with_tests     | 6      | 0.80         | `greet` + pytest + `ValueError` on `None`.            |
| hard   | multi_file_module    | 10     | 0.70         | Three-file module (entry + core + tests), mypy strict.|

### Reward formula

```
reward = 0.6 * sandbox_composite_score + 0.4 * grounding_score
```

- **sandbox_composite_score** ∈ [0, 1] comes from `groundloop.python_sandbox.run_sandbox` — runs `ruff`, `mypy`, `pytest`, and static import resolution on submitted files in a temp dir.
- **grounding_score** ∈ [0, 1] comes from `groundloop.lib_grounder.ground` — fraction of imported modules/attributes that resolve in the running interpreter.

Reward is deterministic and clamped to `[0.0, 1.0]`. Query steps always return `0.0`.

### Baseline Scores

Run: `python3 inference.py` (stub synthesizer, no API key required).

| Task | Difficulty | Baseline Reward |
|---|---|---|
| greet_single_file | easy | 1.000 |
| greet_with_tests | medium | 0.920 |
| multi_file_module | hard | 0.840 |

Measured against `groundloop/kb/skills_corpus.jsonl` with `uvicorn groundloop_env.app:app` on port 17863. Per-task tool sets: easy uses `ruff`/`imports`/`mypy` (no pytest — single file); medium & hard add `pytest`.

### Full verification

```bash
python3 -m pytest tests/ --cov=groundloop_env --cov-report=term -v
ruff check groundloop_env/
mypy --strict groundloop_env/
```
