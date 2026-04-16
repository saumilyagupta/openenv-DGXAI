# CodeForge

**An RL environment that forces LLM agents to write real, verified, grounded Python code.**

CodeForge is an [OpenEnv](https://github.com/openenv)-compliant reinforcement learning environment where the *environment is the judge, not the LLM*. An LLM agent receives a natural-language coding task ("implement `greet(name)`") and must produce working Python code through a sequence of actions. Every reward traces to real tool output, real AST grounding, and real skill corpus citations.

The LLM cannot grade itself. It cannot hallucinate APIs. It cannot skip verification.

```
LLM Agent ──> MCP Server ──> FastAPI ──> CodeForgeEnvironment
                                              |
                          +-------------------+-------------------+
                          v                   v                   v
                    Python Sandbox       AST Grounder        KB Indexer
                    (ruff,mypy,pytest)   (import resolve)    (BM25+clusters)
                          |                   |                   |
                          +--------+----------+                   |
                                   v                              |
                             Grader (reward)                      |
                             quality = 0.6*sandbox + 0.4*ground   |
                             + Brier calibration penalty          |
                                                                  |
                    Ralph Loop <----------------------------------+
                    (synthesize -> score -> keep if better)
                                   |
                             Audit Ledger
                             (every step recorded)
```

---

## How It Works

### The Grading Pipeline

When an LLM submits code, three things happen server-side (the LLM has zero control beyond submitting files):

1. **Python Sandbox** runs real CLI tools against the submitted code:
   - `ruff check` -- linting and style
   - `mypy --strict` -- type checking
   - `pytest` -- runs both the agent's tests AND hidden correctness tests
   - `import scan` -- verifies all imports resolve

2. **AST Grounder** parses the source code and checks every `import` and attribute access against the actual Python runtime via `importlib.util.find_spec()` and `hasattr()`. If the LLM invents `from magic_ai import solve`, the grounder catches it.

3. **Grader** combines the signals into a single reward:
   ```
   quality   = 0.6 * sandbox_score + 0.4 * groundedness
   brier     = min((confidence - quality)^2, 0.5)
   reward    = quality * (1 - brier)
   ```

### The 6 Actions

An agent interacts with CodeForge through 6 action types:

| Action | Cost | Reward | What It Does |
|--------|------|--------|-------------|
| `query_kb` | 1 | 0.0 | BM25 search over 2,648 skill corpus nodes |
| `query_cluster` | 1 | 0.0 | Browse related skills by topic cluster |
| `interrogate` | 1 | 0.0 | Get Socratic questions citing real skill nodes |
| `run_ralph` | N | calibrated | Autonomous synthesize-score-keep loop (N iterations) |
| `submit` | 1 | calibrated | Submit code for sandbox + grounding + Brier grading |
| `get_audit` | 0 | 0.0 | Read the full audit trail of the episode |

Budget is finite. Every action costs budget. When budget hits 0, the episode ends.

### The 3 Task Levels

| Level | Task | Budget | Target | What It Tests |
|-------|------|--------|--------|--------------|
| Easy | Implement `greet(name)` in one file | 4 | 0.90 | Basic code generation, type hints |
| Medium | Add error handling + write tests | 6 | 0.80 | Error handling, test writing, multi-file |
| Hard | Split into 3 files with full types | 10 | 0.70 | Architecture, imports, comprehensive testing |

Each task includes **hidden correctness tests** that the agent never sees. These are injected into the sandbox during grading to prevent "clean garbage" exploits (syntactically valid but semantically wrong code).

---

## Why the LLM Cannot Cheat

| Cheat Attempt | What Catches It |
|--------------|----------------|
| Submit code with `import nonexistent_lib` | AST grounder: `find_spec()` returns None, groundedness drops 40% |
| Call wrong method `os.path.joiiin()` | AST grounder: `hasattr(os.path, "joiiin")` is False |
| Declare high confidence on bad code | Brier penalty: `(0.95 - 0.30)^2 = 0.42`, reward drops 42% |
| Skip confidence to avoid penalty | `confidence=None` treated as 0.5 (mediocre), still penalized |
| Submit `conftest.py` to hijack pytest | Filename allowlist blocks it (regex: `^[a-z][a-z0-9_]*\.py$`) |
| Submit 1000 files as a DoS | Max 10 files, 50KB each, 200KB total |
| Write syntactically clean but wrong code | Hidden tests catch semantic incorrectness |
| Submit empty test `assert True` | Hidden tests run alongside, pytest fails on wrong behavior |
| Submit code with zero imports for free groundedness | Returns 0.5 (neutral), not 1.0 (perfect) |
| Submit unparseable code for free groundedness | Returns 0.0 (penalty), not 1.0 |

### The Brier Calibration System

The confidence field forces the LLM to put a number on how good its code is:

| Scenario | Confidence | Quality | Brier | Reward |
|----------|-----------|---------|-------|--------|
| Good code, well calibrated | 0.85 | 0.90 | 0.003 | 0.898 |
| Good code, overconfident | 0.99 | 0.70 | 0.084 | 0.641 |
| Bad code, honestly uncertain | 0.20 | 0.30 | 0.010 | 0.500 (floor) |
| Bad code, dishonestly confident | 0.90 | 0.30 | 0.360 | 0.192 |
| Skip confidence entirely | None->0.5 | 0.80 | 0.090 | 0.728 |

Honest, well-calibrated agents are rewarded. Overconfident agents are penalized. Skipping confidence is worse than being accurate.

---

## Project Structure

```
CODEFORGE/
├── codeforge/                     # The product (44 Python files)
│   ├── models.py                  # All 6 action types, observation, AuditEntry
│   ├── grader.py                  # Reward function (Brier + floor)
│   ├── grounder.py                # AST grounding (3 bug fixes baked in)
│   ├── shaping.py                 # Citation shaping bonus
│   ├── tasks.py                   # 3 task levels + hidden correctness tests
│   ├── observation.py             # Observation builder
│   ├── environment.py             # CodeForgeEnvironment (all 6 actions)
│   ├── app.py                     # FastAPI + session isolation
│   ├── mcp_server.py              # MCP server (10 tools, resources, prompts)
│   ├── sandbox/                   # Real tool execution
│   │   ├── sandbox.py             # run_sandbox() -- writes files, runs tools
│   │   ├── runner.py              # subprocess execution with timeouts
│   │   ├── tools.py               # Tool registry (ruff, mypy, pytest, imports)
│   │   ├── imports.py             # Import scanning
│   │   ├── metric.py              # composite_score (penalty-only, no double-counting)
│   │   └── models.py              # ToolResult, ParsedResult, SandboxResult
│   ├── kb/                        # Knowledge base
│   │   ├── indexer.py             # BM25 search over skill corpus
│   │   ├── cluster.py             # Jaccard clustering + connected components
│   │   ├── code_graph.py          # AST knowledge graph (ast + networkx)
│   │   ├── corpus_manager.py      # add/remove/refresh skills
│   │   ├── tokenizer.py           # Text tokenization
│   │   ├── models.py              # SearchResult, Cluster, ClusterManifest
│   │   └── skills_corpus.jsonl    # 2,648 skill nodes (baked in)
│   ├── ralph/                     # Autonomous improvement loop
│   │   ├── loop.py                # Score-gated retry loop
│   │   ├── synthesizer.py         # Protocol + Stub + LLM synthesizer
│   │   ├── planner.py             # Task decomposition into subtasks
│   │   ├── checkpoint.py          # Disk persistence
│   │   └── models.py              # LoopConfig, RunResult, Iteration
│   ├── interrogator/              # Socratic question generation
│   │   ├── interrogator.py        # Questions citing skill corpus nodes
│   │   └── models.py              # InterrogationResult
│   ├── audit/                     # Audit trail
│   │   ├── ledger.py              # Per-episode append-only log
│   │   ├── reporter.py            # Build reports from Ralph runs
│   │   └── models.py              # AuditReport
│   └── scraper/                   # Skill corpus generation
│       ├── pipeline.py            # discover -> parse -> chunk -> tag -> write
│       ├── discovery.py           # Glob-based file discovery
│       ├── parser.py              # YAML frontmatter + markdown parsing
│       ├── chunker.py             # Section-level chunking
│       ├── tagger.py              # Domain/topic tagging
│       └── writer.py              # JSONL serialization
├── tests/                         # 429 tests, 93% coverage
├── inference.py                   # Baseline agent (REST API client)
├── server/app.py                  # OpenEnv entry point
├── Dockerfile                     # python:3.11-slim + ruff/mypy/pytest
├── openenv.yaml                   # OpenEnv configuration
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Production dependencies
├── SYSTEM_DESIGN.md               # Full system design (1,942 lines)
└── everything-claude-code/        # Skill corpus source (306 unique skills)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- ruff, mypy, pytest (installed automatically in Docker)

### Local Development

```bash
# Install dependencies
cd CODEFORGE
pip install -r requirements.txt
pip install ruff mypy pytest

# Run tests
python3 -m pytest tests/ -v

# Start the server
uvicorn codeforge.app:app --host 0.0.0.0 --port 7860

# Run baseline agent
python3 inference.py
```

### Docker

```bash
cd CODEFORGE
docker build -t code-forge .
docker run -p 7860:7860 code-forge
```

### Using the REST API

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}'

# Submit code
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "submit",
      "files": {
        "main.py": "from __future__ import annotations\n\ndef greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n"
      },
      "confidence": 0.9
    }
  }'
```

### Using the MCP Server

Connect any MCP-compatible client (Claude, GPT, etc.) to CodeForge:

```python
from codeforge.mcp_server import CodeForgeMCPServer
from pathlib import Path

server = CodeForgeMCPServer(corpus_path=Path("codeforge/kb/skills_corpus.jsonl"))

# List available tools
tools = server.tool_definitions()  # 10 tools

# Start episode
result = server.handle_tool("codeforge_reset", {"task_level": "easy"})
session_id = result["session_id"]

# Query the knowledge base
result = server.handle_tool("codeforge_query_kb", {
    "session_id": session_id,
    "claim": "python greeting function type hints"
})

# Submit code
result = server.handle_tool("codeforge_submit", {
    "session_id": session_id,
    "files": {"main.py": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"},
    "confidence": 0.9
})
print(result["observation"]["last_reward"])  # 0.978
```

---

## The Reward System (3 Layers)

### Layer 1: Scoring Pipeline (`sandbox/metric.py`)

Runs real CLI tools and computes a composite score:

```
score = 1.0 - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
```

- `imports_penalty` = min(1.0, unresolved_count * 0.1)
- `ruff_penalty` = min(ruff_errors, 20) / 40
- `mypy_penalty` = min(mypy_errors, 20) / 40
- `pytest_penalty` = 0.5 if any test fails

No double-counting. Penalty-only. Supports per-tool filtering for subtask scoring.

### Layer 2: AST Grounding (`grounder.py`)

Parses source code, checks every import and attribute:

```
groundedness = grounded_symbols / total_symbols
```

Special cases:
- SyntaxError -> 0.0 (broken code is penalized)
- Zero symbols -> 0.5 (neutral, not a free pass)
- Full module path resolution (`os.path.join` checks against `os.path`, not `os`)

### Layer 3: Reward Function (`grader.py`)

Combines sandbox and grounding with Brier calibration:

```
quality = 0.6 * sandbox_score + 0.4 * groundedness
brier   = min((confidence - quality)^2, 0.5)
reward  = quality * (1 - brier)
```

The uncertain floor (0.50) protects honest agents who admit uncertainty, but cannot complete any task (targets: 0.70, 0.80, 0.90).

---

## The Skill Corpus

CodeForge ships with a frozen corpus of **2,648 skill nodes** scraped from 242 real SKILL.md files:

- 183 skills from [everything-claude-code](https://github.com/affaan-m/everything-claude-code) (157K stars)
- 59 locally installed Claude Code skills

The corpus is indexed with BM25 for full-text search and Jaccard-clustered into topic communities. The LLM queries the corpus via `query_kb` and `query_cluster` actions to find patterns, best practices, and guidance before writing code.

The corpus is **read-only and baked into the Docker image**. The LLM cannot modify it.

---

## Audit Trail

Every action in every episode is recorded in an append-only audit ledger:

```
step 0: action=query_kb, reward=0.0, cited_skills=["python-testing/Fixtures"]
step 1: action=interrogate, reward=0.0, cited_skills=["coding-standards/Types"]
step 2: action=submit, reward=0.978, quality=0.98, brier=0.002, confidence=0.9
```

After an episode, you can verify:
- Did the LLM consult the KB before submitting?
- Was the LLM calibrated? (confidence vs actual quality)
- What symbols were hallucinated?
- What was the total budget efficiency?

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROUNDLOOP_CORPUS_PATH` | `codeforge/kb/skills_corpus.jsonl` | Path to skill corpus |
| `CODEFORGE_MAX_SESSIONS` | `10` | Max concurrent MCP sessions |
| `CODEFORGE_SESSION_TTL` | `3600` | Session timeout (seconds) |
| `ANTHROPIC_API_KEY` | (none) | Required for LLM Synthesizer in Ralph |
| `API_BASE_URL` | `http://localhost:7860` | Used by inference.py |

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Source files | 44 |
| Test count | 429 |
| Test coverage | 93% |
| Ruff violations | 0 |
| Mypy --strict errors | 0 |
| Skill corpus nodes | 2,648 |
| Critic reviews | 10 module critics + 1 red team audit |
| Exploits found and closed | 3 |

---

## Design Documents

- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** -- Full system design (1,942 lines, 20 sections, 8 critics reviewed). The authoritative specification for every module, reward formula, MCP tool schema, and architectural decision.

---

## License

Part of the OpenEnv ecosystem. See the parent repository for license information.
