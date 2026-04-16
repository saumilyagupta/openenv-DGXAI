# CodeForge — Complete System Design

**Version:** 2.0 (post 5-critic review)  
**Date:** 2026-04-16  
**Author:** Krrish Choudhary  
**Purpose:** Full system design for CodeForge — an OpenEnv RL environment that forces LLM agents to produce truthful, verified, production-grade Python code. Exposed as an MCP server.

---

## Table of Contents

1. [Mission Statement](#1-mission-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Interface Layers](#3-interface-layers)
4. [Core Modules](#4-core-modules)
5. [Action Surface](#5-action-surface)
6. [Reward System — How the LLM Cannot Cheat](#6-reward-system--how-the-llm-cannot-cheat)
7. [Anti-Hallucination Enforcement](#7-anti-hallucination-enforcement)
8. [Data Models](#8-data-models)
9. [MCP Server Design](#9-mcp-server-design)
10. [Module Dependency Graph](#10-module-dependency-graph)
11. [Task Definitions](#11-task-definitions)
12. [Audit & Traceability](#12-audit--traceability)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Sandbox Security & Hardening](#14-sandbox-security--hardening)
15. [Session Isolation & Concurrency](#15-session-isolation--concurrency)
16. [Environment Variables & Configuration](#16-environment-variables--configuration)
17. [Error Handling & Edge Cases](#17-error-handling--edge-cases)
18. [Round-1 EpistemicNav Coexistence](#18-round-1-epistemicnav-coexistence)
19. [Known Issues & Required Fixes (from 5-critic review)](#19-known-issues--required-fixes)
20. [Current State & Remaining Work](#20-current-state--remaining-work)

---

## 1. Mission Statement

CodeForge is an RL environment where the **environment is the judge, not the LLM**. The LLM agent receives a natural-language brief ("implement greet(name)") and must produce working Python code through iterative actions. Every claim the LLM makes is verified by real tools:

- Code is **executed in a real sandbox** (ruff, mypy, pytest, import scan)
- Symbols are **verified via AST grounding** against the actual Python runtime
- Knowledge claims are **traced to real skill corpus nodes**
- Confidence calibration is **penalized mathematically** via Brier scoring

The LLM cannot grade itself. The LLM cannot skip verification. The LLM cannot hallucinate APIs because the grounding layer catches unresolvable imports and attributes at the AST level.

**The one-sentence invariant:**
> Every reward-earning action must trace to (a) a sandbox-verified programmatic signal, (b) a Layer-A grounded symbol, and (c) a Layer-B skill citation — recorded in the audit trail as a `(reward, evidence, policy)` triple.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ENGINEER / LLM AGENT                           │
│  (Claude, GPT, any MCP-compatible client, or raw HTTP client)          │
└──────────┬─────────────────────────────────┬─────────────────────────────┘
           │ MCP Protocol (tools)            │ REST (OpenEnv compliance)
           ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────────┐
│   MCP SERVER LAYER  │────────▶│   FastAPI / OpenEnv      │
│  (tool definitions, │         │   POST /reset            │
│   input validation, │         │   POST /step             │
│   schema exposure)  │         │   GET  /state            │
└─────────────────────┘         │   GET  /tasks            │
                                └──────────┬──────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │        CodeForgeEnvironment                  │
                    │   (episode state, budget, action routing)    │
                    └───┬────────┬────────┬────────┬──────┬──────┘
                        │        │        │        │      │
          ┌─────────────┘   ┌────┘   ┌────┘   ┌───┘      │
          ▼                 ▼        ▼        ▼           ▼
   ┌────────────┐  ┌────────────┐ ┌───────┐ ┌──────┐ ┌────────────┐
   │  Python    │  │  AST       │ │ KB    │ │Ralph │ │  Audit     │
   │  Sandbox   │  │  Grounder  │ │Index  │ │Loop  │ │  Ledger    │
   │            │  │            │ │+Clust │ │      │ │            │
   │ ruff       │  │ imports    │ │BM25   │ │synth │ │ per-step   │
   │ mypy       │  │ attributes │ │Jaccard│ │score │ │ (reward,   │
   │ pytest     │  │ modules    │ │graphs │ │keep/  │ │  evidence, │
   │ imports    │  │ resolve    │ │       │ │reject│ │  policy)   │
   └────────────┘  └────────────┘ └───────┘ └──────┘ └────────────┘
        │                │             │        │          │
        │                │             │        │          │
        ▼                ▼             ▼        ▼          ▼
   sandbox_score    groundedness   citations  run_result  audit_trail
   (0.0 - 1.0)     (0.0 - 1.0)   (node_ids) (files+score) (entries)
        │                │
        └───────┬────────┘
                ▼
          ┌──────────┐
          │  Grader  │
          │          │
          │ quality  │◀── sandbox_score * 0.6 + groundedness * 0.4
          │ brier    │◀── (confidence - quality)²
          │ reward   │◀── quality * (1 - min(brier, 0.5))
          └──────────┘
```

---

## 3. Interface Layers

### 3.1 OpenEnv REST API (compliance layer — for judges)

Required by the OpenEnv competition framework. This is the **ground truth interface**.

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_level": "easy\|medium\|hard"}`. Returns observation. |
| `/step` | POST | Execute action. Body: `{"action": {...}}`. Returns observation with reward. |
| `/state` | GET | Current observation (read-only). |
| `/tasks` | GET | List available tasks + action schema. |
| `/` | GET | Health check. |

**Contract:** The environment class `CodeForgeEnvironment` implements `openenv.core.env_server.interfaces.Environment` with `reset()`, `step()`, and `state` property. The FastAPI app is created via `openenv.core.env_server.http_server.create_app()`.

### 3.2 MCP Server (tool interface — for LLM agents)

This is how engineers connect any MCP-compatible LLM to CodeForge. The MCP server wraps the REST API into discoverable tools.

**Why MCP over raw REST:**
- LLM **discovers tools natively** — no need to hardcode API schemas
- Tool descriptions carry constraints ("confidence must be 0-1", "costs 1 budget")
- The LLM can only invoke actions the server exposes — cannot hallucinate actions
- Input validation happens at the MCP layer before reaching the environment

See [Section 9](#9-mcp-server-design) for full MCP tool definitions.

### 3.3 Client Layer (inference agent)

The `inference.py` script is a reference agent that demonstrates the API. Any LLM agent framework (LangChain, AutoGen, Claude Agent SDK, raw HTTP) can be a client.

---

## 4. Core Modules

### 4.1 Python Sandbox (`groundloop/python_sandbox/`)

**What it does:** Takes a set of Python files, writes them to a temp directory, runs real CLI tools against them, parses results, computes a composite score.

**Files:**
- `sandbox.py` — `run_sandbox(files, tools, timeout_per_tool)` → `SandboxResult`
- `runner.py` — `run_tool(name, argv, cwd, timeout)` → `ToolResult` (subprocess execution)
- `tools.py` — Tool registry: `argv_for(name, project_dir)`, `parse(name, tool_result)`, `is_available(name)`
- `imports.py` — `scan_imports(project_dir)` → `ImportReport` (resolves all imports)
- `metric.py` — `composite_score(result)` → float (the scoring formula)
- `models.py` — `ToolResult`, `ParsedResult`, `ImportReport`, `SandboxResult`

**Available tools:** `ruff`, `mypy`, `pytest`, `imports`

**How it prevents cheating:**

| Cheat attempt | What catches it |
|---|---|
| LLM submits code with `import nonexistent_lib` | `imports` tool flags unresolved imports → penalty |
| LLM submits code that doesn't parse | `ruff` exit code != 0 → penalty |
| LLM submits code with wrong types | `mypy --strict` catches type errors → penalty |
| LLM submits code where tests fail | `pytest` catches failures → 0.5 penalty on composite |
| LLM submits empty/stub files (`def f(): pass`) | Tests will fail (pytest) + mypy may flag missing return → penalty. **BUT SEE SECTION 7 FOR GAPS** |

**SANDBOX SECURITY CONCERNS (found by 5-critic review — see §14 for full detail):**
- pytest executes arbitrary submitted Python — a malicious submission can run `os.system()` during test collection
- No file count or size limits on submitted `files` dict
- `conftest.py` / `pytest.ini` / `setup.cfg` submission can hijack test behavior
- `importlib.find_spec()` and `hasattr()` in the grounder can trigger module `__init__.py` side effects

**Composite score formula (actual code in `metric.py`):**
```python
pass_rate = count(tools where ok=True) / count(tools_run)
imports_penalty = min(1.0, unresolved_count * 0.1)
ruff_penalty = min(ruff_errors, 20) / 40
mypy_penalty = min(mypy_errors, 20) / 40
pytest_penalty = 0.5 if pytest_failed else 0.0
raw = pass_rate - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
composite = clamp(raw, 0.0, 1.0)
```

### 4.2 AST Grounder (`groundloop/lib_grounder/`)

**What it does:** Parses submitted source code via Python's `ast` module, extracts every import and attribute access, and checks if they actually resolve in the current Python runtime.

**Files:**
- `grounder.py` — `ground(source)` → `GroundingReport`
- `models.py` — `Symbol`, `GroundingReport`

**How it works:**
1. `ast.parse(source)` — if syntax error, returns groundedness=1.0 (no symbols to check)
2. Walk AST for `ast.Import` and `ast.ImportFrom` nodes → check `importlib.util.find_spec(pkg)` exists
3. Walk AST for `ast.Attribute` on imported names → check `hasattr(module, attr)` is true
4. `groundedness = grounded_count / total_symbols`

**How it prevents cheating:**

| Cheat attempt | What catches it |
|---|---|
| LLM invents `from fastapi_turbo import SuperRouter` | `find_spec("fastapi_turbo")` returns None → ungrounded |
| LLM calls `os.nonexistent_function()` | `hasattr(os, "nonexistent_function")` is False → ungrounded |
| LLM uses wrong attribute `json.dumbs()` | `hasattr(json, "dumbs")` is False → ungrounded |
| LLM uses relative imports to dodge check | `node.level != 0` skips them (they're internal project imports) |

**Groundedness directly affects reward:** It's 40% of the quality signal. An LLM that hallucinated half its imports gets `groundedness=0.5`, costing 20% of total reward.

**KNOWN BUGS (found by 5-critic review — must fix):**

1. **Attribute resolution uses top-level module only.** `grounder.py:71` resolves `os.path.join()` by checking `hasattr(os, "join")` instead of `hasattr(os.path, "join")`. Deep module attributes are checked against the wrong object. **Fix: resolve against the full module path, not just `top = mod_name.split(".")[0]`.**

2. **SyntaxError → groundedness 1.0.** If `ast.parse` fails, the grounder returns `groundedness=1.0` instead of penalizing. **Fix: return 0.0.**

3. **Zero symbols → groundedness 1.0.** Code with no imports (pure builtins) gets a free pass. **Fix: return 0.5 (neutral) when total_symbols == 0.**

### 4.3 Knowledge Base — Two KBs, Not One

CodeForge has **three distinct knowledge bases** serving different purposes:

| | KB1: Skill Documentation (EXISTS) | KB2: Code Graph (TO BUILD) | KB3: External Docs & Papers (TO BUILD) |
|---|---|---|---|
| **What** | 1006+ nodes of patterns, best practices, guides | AST-extracted structure of the agent's Python project | Library docs, API references, research papers, changelogs |
| **Answers** | "How should I write pytest fixtures?" | "What functions exist in core.py? What calls greet()?" | "What is the current pydantic v2 API for model_validate?" |
| **Index** | BM25 text search + Jaccard clustering | NetworkX DiGraph with import/call/class edges | BM25 text search (same engine as KB1, separate corpus) |
| **Source** | SKILL.md files (static, frozen in Docker) | Agent's `current_files` (rebuilt every step) | Markdown/PDF/HTML docs ingested by admin (static, frozen in Docker) |
| **LLM-free?** | Yes | Yes (pure `ast` stdlib) | Yes (BM25, no embeddings) |
| **Action** | `query_kb`, `query_cluster` | `query_code_graph` (new, see §5.3) | `query_docs` (new, see §5.3) |
| **Why it matters** | LLM learns coding patterns | LLM understands its own project structure | **LLM stops hallucinating library APIs.** When it doesn't know the current API for a library, it queries KB3 instead of guessing. This is the anti-hallucination layer for external knowledge. |

#### 4.3.1 KB1: Skill Documentation Indexer (`groundloop/kb_indexer/`)

**What it does:** Manages the corpus of coding-skill nodes scraped from real Claude Code skills. Provides BM25 full-text search and Jaccard-based clustering.

**Files:**
- `index.py` — `SkillsIndex` class: `build()`, `search()`, `attach_cluster_manifest()`, `nodes_in_cluster()`
- `cluster.py` — `build_clusters(nodes)` → `ClusterManifest`, Jaccard similarity + connected components
- `tokenizer.py` — Text tokenization for BM25
- `cache.py` — Pickle-based index caching
- `models.py` — `SearchResult`, `Cluster`, `ClusterManifest`

**Corpus sources (current + planned):**

| Source | Nodes | Status |
|---|---|---|
| `~/.claude/skills/*/SKILL.md` (local install) | ~1,006 | SHIPPED |
| `everything-claude-code/skills/*/SKILL.md` ([github.com/affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code), 157K stars, 306 unique skills) | ~1,500 est. | TO INTEGRATE — clone repo, add one `SourceRoot` glob, re-scrape. Zero code changes to scraper. Format is identical (YAML frontmatter + markdown sections). |
| **Total after integration** | **~2,500+** | |

**Skill management API (to build, ~100 lines):**

```python
class SkillCorpusManager:
    def add_skill(self, path: Path) -> int:
        """Scrape a single SKILL.md, append nodes to corpus. Returns node count added."""
    def remove_skill(self, skill_name: str) -> int:
        """Filter skill_name from corpus, rebuild index. Returns nodes removed."""
    def refresh(self) -> dict[str, int]:
        """Diff disk sources vs corpus by mtime/body_hash. Returns {added, removed, unchanged}."""
```

The scraper already stores `mtime` and `body_hash` per node — `refresh()` compares these against current disk state to do incremental updates instead of full re-scrape.

**How it prevents cheating:**
- The corpus is a **frozen snapshot** baked into the Docker image — the LLM cannot modify it at runtime
- Search results include `node_id` + `skill_name` + `section_body` — traceable citations
- `cluster_id` links results to graph communities for structured navigation
- The LLM gets back **real documentation text**, not generated summaries

#### 4.3.2 KB2: Code Knowledge Graph (TO BUILD)

**What it does:** Builds an AST-extracted structural graph of the agent's Python project files. Enables structural queries ("what imports greet?", "what functions does core.py export?") without reading entire files into context.

**Why it matters:** The current system dumps all `current_files` content into the observation. For the hard task (3 files), this is fine. For future tasks with 10+ files, dumping everything wastes tokens. The code graph provides targeted answers.

**Implementation:** `ast` (stdlib) + `networkx.DiGraph`. ~80 lines. No new dependencies.

**File:** `groundloop/code_graph/graph.py` (to create)

```python
import ast
import networkx as nx

def build_code_graph(files: dict[str, str]) -> nx.DiGraph:
    """Build a structural graph from Python source files.

    Nodes: modules, functions, classes
    Edges: imports, calls, inheritance, exports
    """
    G = nx.DiGraph()
    for filename, source in files.items():
        module = filename.removesuffix(".py")
        G.add_node(module, kind="module")
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fqn = f"{module}.{node.name}"
                G.add_node(fqn, kind="function", line=node.lineno)
                G.add_edge(module, fqn, relation="exports")
            elif isinstance(node, ast.ClassDef):
                fqn = f"{module}.{node.name}"
                G.add_node(fqn, kind="class", line=node.lineno)
                G.add_edge(module, fqn, relation="exports")
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        G.add_edge(fqn, base.id, relation="inherits")
            elif isinstance(node, ast.ImportFrom) and node.module:
                G.add_edge(module, node.module, relation="imports")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    G.add_edge(module, alias.name, relation="imports")
    return G

def query_graph(G: nx.DiGraph, question: str) -> list[dict]:
    """Structural queries: callers_of, exports_of, imports_of, dependents_of."""
    ...
```

**How it reduces token usage:**
- Instead of: observation contains full text of all 10 files (10K+ tokens)
- With graph: `query_code_graph("what exports greet")` → `[{"node": "core.greet", "kind": "function", "line": 4}]` (50 tokens)
- Graphify (github.com/safishamsi/graphify, 27K stars) reports 71x token reduction on 52-file codebases. Our scale is smaller but the principle holds.

**Why not use Graphify directly:** Graphify requires Claude API calls during extraction (tree-sitter + Claude vision). CodeForge's grading path must be **LLM-free and deterministic**. Our stdlib `ast` approach has zero external dependencies and zero cost.

**Jaccard clustering (KB1) vs AST edges (KB2):** Jaccard on text tokens is correct for documentation similarity (KB1). For code structure (KB2), text overlap is the wrong signal — two files can share zero vocabulary but be tightly coupled via imports. AST edges capture structural coupling. Different tools for different jobs.

#### 4.3.3 KB3: External Docs & Papers (TO BUILD)

**The problem KB3 solves:** LLMs hallucinate library APIs. They confidently write `pydantic.BaseModel.validate()` when pydantic v2 renamed it to `model_validate()`. They invent `pytest.mark.parametrize` kwargs that don't exist. They cite paper results that are wrong. KB3 gives the LLM a place to LOOK UP real docs instead of guessing.

**What it does:** Stores chunked, indexed versions of external documentation — library API references, research papers, changelogs, migration guides. The LLM queries KB3 via a `query_docs` action instead of hallucinating.

**How it works:**

The same BM25 engine that powers KB1 (skill corpus) also powers KB3. The difference is the corpus source:
- KB1 corpus: SKILL.md files (coding patterns, best practices)
- KB3 corpus: external docs (library APIs, paper abstracts, changelogs)

They share the same `SkillsIndex` search infrastructure but operate on separate JSONL corpus files.

**Ingestion pipeline:**

```python
class DocsIngester:
    """Ingest external docs into KB3 corpus."""

    def ingest_markdown(self, path: Path, *, source_label: str) -> int:
        """Chunk a .md file into searchable nodes. Returns node count."""
        # Uses the same chunker as the skills scraper (split on H1-H3)
        # Tags with source_label (e.g., "pydantic-v2", "pytest-8.x", "arxiv:2401.12345")

    def ingest_url(self, url: str, *, source_label: str) -> int:
        """Fetch URL, convert to markdown, chunk, index. Returns node count."""
        # Uses httpx to fetch, html2text or markdownify to convert
        # Then same chunking pipeline

    def ingest_pdf(self, path: Path, *, source_label: str) -> int:
        """Extract text from PDF (research papers), chunk, index. Returns node count."""
        # Uses PyMuPDF (fitz) or pdfplumber for text extraction
        # Chunks by section headings or fixed-size windows

    def remove_source(self, source_label: str) -> int:
        """Remove all nodes from a source. Returns nodes removed."""

    def list_sources(self) -> list[dict[str, int]]:
        """List all ingested sources with node counts."""
```

**Corpus format:** Same JSONL as KB1. Each node has:
```json
{
  "id": "docs_pydantic-v2_BaseModel_model_validate_abc123",
  "skill_name": "pydantic-v2",
  "section_path": ["BaseModel", "model_validate"],
  "section_body": "model_validate(obj, *, strict=None, from_attributes=None, context=None)...",
  "tags": ["domain:api-reference", "library:pydantic", "version:2.x"],
  "source_path": "https://docs.pydantic.dev/latest/api/base_model/",
  "source_root": "pydantic-v2-docs"
}
```

**What to ingest (starter set for Python code tasks):**

| Source | What | Why |
|---|---|---|
| pydantic v2 docs | API reference, migration guide | Most common hallucination target (v1 vs v2 API) |
| pytest docs | Fixtures, marks, parametrize, conftest | Tasks require pytest — LLM must use real API |
| ruff docs | Rule codes, configuration | Tasks use ruff — LLM needs to understand warnings |
| mypy docs | Strict mode flags, type annotation rules | Tasks require mypy --strict |
| Python stdlib | ast, importlib, pathlib, dataclasses | The grounder and sandbox use these — LLM should know real APIs |
| Key research papers | Brier scoring, calibration, RL environments | For agents that want to understand the reward function theory |

**How it integrates with grading:**

KB3 does not directly affect the reward function. But it indirectly improves groundedness:
- LLM queries KB3: "what is the pydantic v2 way to validate a model?"
- KB3 returns: `model_validate(obj, ...)` (the real API)
- LLM uses `model_validate()` in submitted code
- Grounder checks: `hasattr(pydantic.BaseModel, "model_validate")` → True → grounded
- If LLM had guessed `validate()` instead: `hasattr(pydantic.BaseModel, "validate")` → False → ungrounded → score drops

KB3 is the carrot (correct information available). The grounder is the stick (wrong APIs get penalized).

**Action: `query_docs`**

| Property | Value |
|---|---|
| Budget cost | 1 |
| Reward | 0.0 |
| Input | `claim` (search query), `top_k`, `source_filter` (optional, e.g., "pydantic-v2") |
| Output | `last_doc_hits` in observation — same format as `last_citations` |

**Admin-only, not agent-writable:** The LLM agent can READ KB3 via `query_docs`. It CANNOT write to it. Ingestion is done by the environment admin (the engineer deploying CodeForge) before baking the corpus into the Docker image. This prevents the agent from poisoning its own knowledge base.

**File:** `codeforge/kb/docs_ingester.py` (to create in Phase 3)

**Dependencies:** `httpx` (URL fetching — already in requirements), `markdownify` or `html2text` (HTML→markdown), optionally `pymupdf` for PDF papers. Keep minimal.

**Phase:** This is Phase 3 work (M14, after M10-M13). The infrastructure (BM25 index, JSONL corpus, search) already exists from KB1. KB3 reuses it with a different corpus file.

### 4.4 Ralph Orchestrator (`groundloop/ralph_orchestrator/`)

**What it does:** Score-gated retry loop — synthesize code → sandbox-score → keep if better → repeat. Inspired by the autonomous iteration pattern (see [snarktank/ralph](https://github.com/snarktank/ralph), 17K stars, which implements the same concept as a bash script over CLI tools).

**Files:**
- `loop.py` — `run_loop(spec, initial_files, index, synthesizer, config, checkpoint_dir)` → `RunResult`
- `models.py` — `LoopConfig`, `SynthesisResult`, `Iteration`, `RunResult`
- `synthesizer.py` — `Synthesizer` protocol (abstract)
- `stub_synthesizer.py` — `StubSynthesizer` (deterministic, for testing)
- `checkpoint.py` — `save_checkpoint()`, `load_checkpoint()` (disk persistence)

**How it prevents cheating:**
- Each iteration is **independently scored** by the real sandbox — the synthesizer cannot lie about scores
- Only proposals where `score_after > score_before` are kept — regressions are rejected
- 3 consecutive regressions → loop terminates ("stuck")
- Wasted iterations cost budget (0.05 penalty per wasted iter)
- Full iteration history is preserved in `RunResult` — nothing hidden

**HONEST GAPS (found by Ralph decomposition critic):**

**1. No real Synthesizer implementation.** The only concrete `Synthesizer` is `StubSynthesizer` — a deterministic test double that extracts a fenced code block from the top BM25 citation. There is no LLM-backed synthesizer. For Ralph to actually work as an autonomous coding loop, an `LLMSynthesizer` must be built (~150 lines wrapping Claude/GPT API).

```python
class LLMSynthesizer(Synthesizer):
    """Calls an LLM to produce improved code given spec + current files + citations."""
    def synthesize(self, *, spec, current_files, citations, iteration) -> SynthesisResult:
        # Build prompt from spec + files + citation text + iteration history
        # Call Claude/GPT API
        # Parse response into proposed_files + rationale + cited_node_ids
        ...
```

**2. No task decomposition.** Ralph is a flat retry loop (`for i in range(max_iters)`). It hands the **entire** spec and **all** files to the synthesizer every iteration. There is no planner, no subtask extraction, no ability to say "first implement core.py, then wire main.py, then write tests." The snarktank/ralph project solves this via PRD user stories — but it's a bash script, not a Python library, and cannot be imported.

**What real decomposition would require:**

```python
class Planner:
    """Decomposes a task spec into ordered subtasks."""
    def plan(self, spec: str, initial_files: dict[str, str]) -> list[Subtask]:
        # Parse brief → identify file dependencies → order by dependency
        # e.g., hard task → [implement core.py, wire main.py, write test_core.py]
        ...

class Subtask:
    target_files: list[str]     # which files this subtask modifies
    acceptance: str             # what "done" means for this subtask
    tools: tuple[str, ...]      # which sandbox tools score this subtask
```

A planner layer sits **above** Ralph — it breaks the spec into subtasks, then runs Ralph (or direct synthesis) on each subtask sequentially.

**3. All-or-nothing scoring.** `_score_files()` runs the full sandbox on all files every iteration. If the agent is building `core.py` but `test_core.py` is still empty, pytest fails and composite_score collapses to near 0. There is no partial credit, no per-file scoring.

**Fix: incremental scoring.** Score only the tools relevant to the current subtask. If the subtask is "implement core.py", run only `ruff + mypy + imports` (no pytest, since tests aren't written yet). The `LoopConfig.tools` field already supports this — the planner just needs to set it per-subtask.

**Why not use snarktank/ralph:** It's a bash script that shells out to CLI tools (amp/claude), manages git branches, and reads/writes JSON files. No Python API, no importable interface. Our `ralph_orchestrator` already implements the same loop pattern in Python with a clean `Synthesizer` protocol. The missing pieces (LLM synthesizer, planner, incremental scoring) are better built on our existing foundation.

### 4.5 Interrogator (`groundloop/interrogator/`)

**What it does:** Socratic front-loading. Given a task brief, generates probing questions that cite skill corpus nodes, forcing the LLM to think before coding.

**Files:**
- `interrogator.py` — `Interrogator(index).generate(brief)` → `InterrogationResult`
- `models.py` — `InterrogationResult`

**How it helps enforce truth:**
- Questions reference **real skill nodes** from the corpus
- Forces the LLM to consider edge cases, success criteria, and existing guidance
- Costs budget — the LLM must decide if interrogation is worth the spend

### 4.6 Audit Reporter (`groundloop/audit_reporter/`)

**What it does:** Builds audit reports from Ralph run results. Tracks skill citations, score trajectories, hallucination rates.

**Files:**
- `reporter.py` — `AuditReporter.build(run, hallucination_rate)` → `AuditReport`
- `models.py` — `AuditReport`

### 4.7 Skills Scraper (`groundloop/skills_scraper/`)

**What it does:** Scrapes real SKILL.md files from disk, parses YAML frontmatter + markdown sections, chunks into searchable nodes, writes JSONL corpus.

**Pipeline:** discover → parse → chunk → tag → deduplicate → write JSONL

**Files:**
- `pipeline.py` — End-to-end orchestrator
- `discovery.py` — File discovery via glob patterns from `SourceRoot` config
- `parser.py` — YAML frontmatter + markdown body via `python-frontmatter`
- `chunker.py` — Split on H1-H3 headings, merge small chunks, generate node IDs
- `tagger.py` — Domain/topic tagging via keyword matching (11 domains, 7 phases)
- `writer.py` — JSONL serialization + SHA256 manifest

**Corpus sources:**

Currently scrapes `~/.claude/skills/*/SKILL.md`. To integrate [everything-claude-code](https://github.com/affaan-m/everything-claude-code) (157K stars, 459 SKILL.md files, 306 unique skills), add one `SourceRoot`:

```python
# In scraper config — scrape only top-level skills/ to avoid duplicates
SourceRoot(label="ecc", glob="/path/to/everything-claude-code/skills/*/SKILL.md")
```

Format is identical: YAML frontmatter (`name`, `description`) + structured markdown sections. The existing scraper handles it with zero code changes. Deduplication by `(skill_name, section_path, body_hash)` handles overlap with locally-installed skills.

**Skill management (to build):**

The scraper is currently one-shot — run it, get a JSONL, freeze it. No add/remove, no incremental update. The `mtime` and `body_hash` fields per node exist but are never compared against disk state. See §4.3.1 for the `SkillCorpusManager` API that will wrap the scraper with add/remove/refresh.

### 4.8 Reward System (3 layers)

The reward system has 3 distinct layers. Each has a different job. Each must be understood separately.

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1: GRADER  (groundloop_env/grader.py)                         │
│                                                                      │
│   Inputs:  sandbox_score (float), groundedness (float),              │
│            confidence (float|None)                                    │
│   Output:  reward (float, 0.0 – 1.0)                                │
│                                                                      │
│   This is the REWARD FUNCTION. It is pure math. It does not run      │
│   tools, parse code, or call any API. It takes two floats from       │
│   the layers below and combines them.                                │
│                                                                      │
│   This layer is CORRECT for both Phase 1 and Phase 2.                │
│   It does not need to change when adding code graphs, planners,      │
│   or LLM synthesizers. It just takes numbers in, puts a number out.  │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 2: SCORING PIPELINE  (groundloop/python_sandbox/)              │
│                                                                      │
│   Inputs:  files (dict[str, str]), tools (tuple[str, ...])           │
│   Output:  sandbox_score (float, 0.0 – 1.0)                         │
│                                                                      │
│   This layer writes files to a temp dir, runs real CLI tools         │
│   (ruff, mypy, pytest, imports) as subprocesses, parses their        │
│   output, and computes a composite score.                            │
│                                                                      │
│   This layer needs ONE CHANGE for Phase 2: a tools filter on         │
│   composite_score() so subtask scoring can exclude irrelevant tools. │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 3: GROUNDING CHECK  (groundloop/lib_grounder/)                 │
│                                                                      │
│   Inputs:  source (str — concatenated Python source code)            │
│   Output:  groundedness (float, 0.0 – 1.0)                          │
│                                                                      │
│   This layer AST-parses the source, extracts every import and        │
│   attribute access, and checks if they resolve in the Python         │
│   runtime. groundedness = grounded_count / total_symbols.            │
│                                                                      │
│   This layer needs BUG FIXES but no structural changes for Phase 2.  │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.8.1 Layer 1: Reward Function — What It Should Be

```python
# groundloop_env/grader.py

_SANDBOX_WEIGHT = 0.6
_GROUNDING_WEIGHT = 0.4
_BRIER_CAP = 0.5
_UNCERTAIN_CONFIDENCE_THRESHOLD = 0.3
_UNCERTAIN_QUALITY_THRESHOLD = 0.5
_UNCERTAIN_FLOOR = 0.50          # was 0.70 — LOWERED (must be below ALL task targets)


def compute_reward(
    *,
    sandbox_score: float,
    groundedness: float,
    confidence: float | None = None,
) -> float:
    """Compute the final reward for a submit action.

    quality   = weighted combination of sandbox and grounding signals
    brier     = calibration penalty (overconfidence on bad code is punished)
    uncertain = floor reward for honest uncertainty (below all task targets)
    """
    quality = _SANDBOX_WEIGHT * sandbox_score + _GROUNDING_WEIGHT * groundedness

    brier_penalty = 0.0
    if confidence is not None:
        brier_penalty = min((confidence - quality) ** 2, _BRIER_CAP)

    reward = quality * (1.0 - brier_penalty)

    # Honest uncertainty floor: agent admits it doesn't know (confidence < 0.3)
    # AND the code is genuinely bad (quality < 0.5).
    # Floor = 0.50, which is BELOW all task targets (easy=0.90, medium=0.80, hard=0.70).
    # This means the floor NEVER completes an episode — it just prevents total collapse
    # for honest agents, while dishonest confident agents get Brier-hammered.
    if (
        confidence is not None
        and confidence < _UNCERTAIN_CONFIDENCE_THRESHOLD
        and quality < _UNCERTAIN_QUALITY_THRESHOLD
    ):
        reward = max(reward, _UNCERTAIN_FLOOR)

    return round(max(0.0, min(1.0, reward)), 3)
```

**Why this works for Phase 2 unchanged:** `compute_reward` takes two floats and returns one float. Whether those floats come from a full-project sandbox run (Phase 1), a subtask-scoped sandbox run (Phase 2 planner), or a Ralph iteration — it doesn't matter. The math is the same.

**Brier incentive table (after floor fix):**

| Scenario | confidence | quality | brier | reward |
|---|---|---|---|---|
| Good code, well calibrated | 0.85 | 0.90 | 0.003 | 0.898 |
| Good code, overconfident | 0.99 | 0.70 | 0.084 | 0.641 |
| Bad code, honestly uncertain | 0.20 | 0.30 | 0.010 | 0.500 (floor) |
| Bad code, dishonestly confident | 0.90 | 0.30 | 0.360 | 0.192 |
| Garbage, admits uncertainty | 0.10 | 0.20 | 0.010 | 0.500 (floor) |

The floor (0.50) is now below easy (0.90), medium (0.80), and hard (0.70). An agent cannot complete ANY task by gaming the floor.

#### 4.8.2 Layer 2: Scoring Pipeline — What It Should Be

```python
# groundloop/python_sandbox/metric.py

def composite_score(
    result: SandboxResult,
    *,
    tools: tuple[str, ...] | None = None,    # NEW: filter for subtask scoring
) -> float:
    """Compute composite score from sandbox results.

    When tools is None, score all tools that were run (full-project scoring).
    When tools is provided, score only those tools (subtask scoring).
    This lets the planner score 'implement core.py' with only ruff+mypy+imports,
    without pytest destroying the score because tests aren't written yet.
    """
    parsed = result.parsed
    if tools is not None:
        parsed = {k: v for k, v in parsed.items() if k in tools}
    if not parsed:
        return 0.0

    # Penalty-only scoring (no double-counting with pass_rate)
    imports_penalty = min(1.0, len(result.imports.unresolved) * 0.1)

    ruff = parsed.get("ruff")
    mypy = parsed.get("mypy")
    pytest_result = parsed.get("pytest")

    ruff_penalty = min(ruff.count, 20) / 40 if ruff else 0.0
    mypy_penalty = min(mypy.count, 20) / 40 if mypy else 0.0
    pytest_penalty = 0.5 if pytest_result and not pytest_result.ok else 0.0

    # Start at 1.0, subtract penalties. No pass_rate to avoid double-counting.
    raw = 1.0 - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
    return max(0.0, min(1.0, raw))
```

**Changes from current code:**
1. `tools` parameter — filters which parsed results to include in scoring
2. Start from `1.0` instead of `pass_rate` — eliminates double-counting
3. Both changes are backwards-compatible: `composite_score(result)` with no `tools` arg scores everything, same as before

**How subtask scoring works:**

```python
# Planner sets per-subtask tools
subtask_a = Subtask(
    target_files=["core.py"],
    tools=("ruff", "imports", "mypy"),     # no pytest — tests aren't written yet
)
subtask_c = Subtask(
    target_files=["test_core.py"],
    tools=("ruff", "imports", "mypy", "pytest"),  # now pytest matters
)

# Ralph scores subtask A with only ruff/mypy/imports
score = composite_score(sandbox_result, tools=subtask_a.tools)
```

#### 4.8.3 Layer 3: Grounding Check — What It Should Be

```python
# groundloop/lib_grounder/grounder.py

def ground(source: str) -> GroundingReport:
    """AST-parse source, check every import and attribute access resolves."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # FIXED: unparseable code gets groundedness=0.0, not 1.0
        return GroundingReport(
            total_symbols=0, grounded=(), ungrounded=(), groundedness=0.0,
        )

    symbols: list[Symbol] = []
    import_to_module: dict[str, str] = {}

    # ... existing import/attribute walking logic ...

    # FIXED: attribute resolution uses full module path, not just top-level
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            mod_name = import_to_module.get(base)
            if mod_name is None:
                continue
            # Resolve against full module, not just top-level package
            resolved = _has_attr(mod_name, node.attr)
            symbols.append(
                Symbol(
                    module=mod_name, attr=node.attr, kind="attribute",
                    resolved=resolved, line=node.lineno,
                )
            )

    grounded = tuple(s for s in symbols if s.resolved)
    ungrounded = tuple(s for s in symbols if not s.resolved)
    total = len(symbols)

    # FIXED: zero symbols gets groundedness=0.5 (neutral), not 1.0 (perfect)
    groundedness = 0.5 if total == 0 else len(grounded) / total

    return GroundingReport(
        total_symbols=total, grounded=grounded, ungrounded=ungrounded,
        groundedness=groundedness,
    )
```

**3 fixes from current code:**
1. `SyntaxError → groundedness=0.0` (was 1.0 — gave free reward for broken code)
2. `total_symbols=0 → groundedness=0.5` (was 1.0 — gave free 40% quality for trivial code)
3. Attribute resolution against full module path (was resolving `os.path.join` by checking `hasattr(os, "join")`)

#### 4.8.4 Shaping Rewards for Non-Submit Actions

All non-submit actions (`query_kb`, `query_cluster`, `interrogate`, `query_code_graph`) currently return reward=0.0. The agent gets no gradient signal to learn that querying before coding is useful.

**Phase 1: Keep sparse rewards.** Reward=0.0 on non-submit actions. Budget efficiency is the implicit signal — wasting queries = fewer submits = lower total reward. This is simple and correct for the competition.

**Phase 2: Add retroactive citation shaping.**

After a successful `submit` (reward > 0), compute a shaping bonus based on whether prior queries actually informed the submitted code:

```python
# groundloop_env/shaping.py

def citation_shaping_bonus(
    *,
    submit_files: dict[str, str],
    prior_citations: list[dict],
    prior_cluster_hits: list[str],
) -> float:
    """Compute a small bonus for queries that led to better code.

    Retroactively reward information-gathering that demonstrably
    influenced the submitted code. This creates gradient signal
    for the agent to learn: 'querying relevant topics before
    coding leads to higher total episode reward.'
    """
    if not prior_citations:
        return 0.0

    # Extract skill names from citations the agent queried
    cited_skills = {c["skill_name"] for c in prior_citations}

    # Extract tokens from the submitted code
    code_text = " ".join(submit_files.values()).lower()

    # Count how many cited skill keywords appear in the code
    overlap = sum(1 for skill in cited_skills if skill.replace("-", "_") in code_text)

    # Small bonus: 0.01 per relevant citation, max 0.05
    return min(overlap * 0.01, 0.05)
```

**How it integrates:**
```python
# In environment.py, after computing submit reward
base_reward = compute_reward(sandbox_score=..., groundedness=..., confidence=...)
shaping = citation_shaping_bonus(
    submit_files=action.files,
    prior_citations=self._all_episode_citations,
    prior_cluster_hits=self._all_episode_cluster_hits,
)
final_reward = min(1.0, base_reward + shaping)
```

**Properties:**
- Max bonus is 0.05 — cannot dominate the base reward
- Only fires on submit — non-submit actions still return 0.0 at call time
- Retroactive — rewards past queries only if they demonstrably influenced the code
- Zero bonus if the agent queries random topics and ignores the results

**Phase 2+ consideration: structural shaping via KB2.**

Once the code knowledge graph (KB2) exists, shaping can be more precise: did the agent query `query_code_graph("what imports greet")` and then correctly wire the import? The graph provides ground truth for whether a query was actually used.

#### 4.8.5 Ralph Reward — What It Should Be

When `run_ralph` executes, each iteration is scored by the sandbox. The final episode reward for the `run_ralph` action:

```python
def ralph_reward(
    *,
    final_score: float,
    groundedness: float,
    total_iters: int,
    wasted_iters: int,
) -> float:
    """Compute reward for a run_ralph action.

    final_score:   sandbox composite_score of the best iteration's files
    groundedness:  grounding check on the best iteration's files
    total_iters:   how many iterations were requested
    wasted_iters:  iterations where score did not improve (regressed or plateau)
    """
    # Base reward uses the standard grader with hardcoded confidence=0.75
    # (Ralph is an autonomous loop — it should be moderately confident)
    base = compute_reward(
        sandbox_score=final_score,
        groundedness=groundedness,
        confidence=0.75,
    )

    # Waste penalty: each wasted iteration costs 0.05
    # This incentivizes the agent to request fewer iterations
    # and stop early if the loop is stuck
    waste_penalty = wasted_iters * 0.05

    return round(max(0.0, min(1.0, base - waste_penalty)), 3)
```

**For Phase 2 with the planner:** The planner decomposes the spec into subtasks and runs Ralph on each subtask. Each subtask has its own `LoopConfig.tools` (e.g., no pytest for subtask A). The `composite_score(result, tools=subtask.tools)` feeds into `ralph_reward()`. The reward function does not change — only the scoring pipeline filters differently.

### 4.9 Environment (`groundloop_env/environment.py`)

**What it does:** The `CodeForgeEnvironment` class that implements the OpenEnv `Environment` interface. Manages episode state, routes actions, calls the grading pipeline.

### 4.10 Observation Builder (`groundloop_env/observation_builder.py`)

**What it does:** Constructs `CodeForgeObservation` from episode state. The observation is what the LLM sees — it shows the task brief, current files, budget, previous score, citations, grounding, and whether the episode is done.

### 4.11 Tasks (`groundloop_env/tasks.py`)

**What it does:** Defines the 3 task levels with their briefs, initial files, target scores, budgets, and tool configurations.

---

## 5. Action Surface

### 5.1 Currently Implemented

| Action | Budget cost | What it does | Reward |
|---|---|---|---|
| `query_kb` | 1 | BM25 search over skill corpus → `last_citations` | 0.0 |
| `submit` | 1 | Run sandbox + grounding → compute calibrated reward | `calibrated_reward` |

### 5.2 Target (6 actions)

| Action | Budget cost | What it does | Reward | Status |
|---|---|---|---|---|
| `query_kb` | 1 | BM25 search over 1006 skill nodes | 0.0 | SHIPPED |
| `query_cluster` | 1 | Lookup cluster by label, return member nodes | 0.0 | M3 (not started) |
| `interrogate` | 1 | Socratic questions citing skill nodes | 0.0 | M4 (not started) |
| `run_ralph` | N (max_iters) | Autonomous synthesize→score→keep loop | `calibrated_reward(final, confidence=0.75) - 0.05*wasted` (hardcoded confidence per CLAUDE.md spec) | M5 (not started) |
| `submit` | 1 | Sandbox + grounding + Brier calibration | `calibrated_reward` | SHIPPED (M2 added Brier) |
| `get_audit` | 0 | Read-only audit ledger of current episode | 0.0 | M6 (not started) |

### 5.3 Future Actions (post-M9)

| Action | Budget cost | What it does | Reward | Module |
|---|---|---|---|---|
| `query_code_graph` | 0 | AST-structural query over agent's current files (KB2). Returns function/class/import relationships without reading full files. | 0.0 | M10 |
| `query_docs` | 1 | Search external docs/papers corpus (KB3). Returns real library API references, paper abstracts, changelogs. Use this instead of guessing library APIs. | 0.0 | M14 |

`query_code_graph` is free (0 budget) because it queries the agent's own files — data already in the observation. The value is **structured access** instead of raw text, reducing token waste on large projects.

`query_docs` costs 1 budget because it searches an external corpus the agent doesn't already have. It's the mechanism that prevents hallucinated library APIs — the LLM looks up the real API instead of guessing, then the grounder verifies the submitted code uses the correct symbols.

---

## 6. Reward System — How the LLM Cannot Cheat

### 6.1 The Grading Pipeline Is Fully Server-Side

```
LLM submits files
       │
       ▼
Environment receives files (LLM has zero control beyond this point)
       │
       ├──▶ python_sandbox.run_sandbox(files, tools)
       │         │
       │         ├── Writes files to temp dir
       │         ├── Runs ruff as subprocess → exit code + error count
       │         ├── Runs mypy --strict as subprocess → exit code + error count
       │         ├── Runs pytest as subprocess → pass/fail
       │         ├── Scans imports → resolves against Python runtime
       │         ├── Computes composite_score from real results
       │         └── Deletes temp dir
       │
       ├──▶ lib_grounder.ground(concatenated_source)
       │         │
       │         ├── AST parses the source
       │         ├── Walks every import → checks importlib.find_spec()
       │         ├── Walks every attribute access → checks hasattr()
       │         └── Returns groundedness = grounded / total
       │
       └──▶ grader.compute_reward(sandbox_score, groundedness, confidence)
                 │
                 ├── quality = 0.6 * sandbox + 0.4 * groundedness
                 ├── brier = min((confidence - quality)², 0.5)
                 ├── reward = quality * (1 - brier)
                 ├── uncertain floor: if conf < 0.3 AND quality < 0.5 → max(reward, 0.70)
                 └── return clamp(0.0, 1.0)
```

**The LLM never sees the grading internals.** It sends files in, gets a number back.

### 6.2 Cheat Vectors and How Each Is Blocked

| # | Cheat vector | Defense | Module |
|---|---|---|---|
| 1 | Submit code that looks right but doesn't run | Sandbox runs it. ruff/mypy/pytest are real CLI tools. | python_sandbox |
| 2 | Import fake libraries (`from magic_ai import solve`) | `importlib.find_spec()` returns None → ungrounded → 40% quality hit | lib_grounder |
| 3 | Call wrong methods (`os.path.joiiin()`) | `hasattr(os.path, "joiiin")` is False → ungrounded | lib_grounder |
| 4 | Declare high confidence on bad code | Brier penalty: `(0.95 - 0.3)² = 0.42` → reward drops 42% | grader |
| 5 | Always declare low confidence to avoid penalty | Uncertain floor only triggers at quality < 0.5. Good code + low confidence = no bonus, just lower raw reward | grader |
| 6 | Submit empty files to "pass" tools | pytest fails (no tests) → 0.5 penalty. mypy may flag missing impls. composite near 0.0 | python_sandbox |
| 7 | Submit stub functions (`def greet(): pass`) | pytest catches wrong behavior. mypy catches missing return type. Composite tanks. | python_sandbox |
| 8 | Ignore the KB and just write code | Works if the code is correct! The grounding check still runs. No penalty for skipping KB. But the LLM wastes the opportunity to learn patterns. | by design |
| 9 | Hallucinate skill citations | The LLM doesn't control citations — `query_kb` returns real corpus nodes. The audit trail records what was actually queried. | kb_indexer + audit |
| 10 | Burn budget on useless queries | Budget is finite. Each wasted query = one fewer submit attempt. The agent must be strategic. | environment |
| 11 | Try to modify the grading tools | Files are written to a temp dir that's deleted after scoring. The sandbox runs ruff/mypy/pytest from the system PATH — not from submitted files. | python_sandbox |
| 12 | Submit a conftest.py that makes all tests pass | **NOT CURRENTLY BLOCKED** — see §6.3. The agent controls all files in the temp dir including conftest.py. | python_sandbox |
| 13 | Submit code with zero imports to get free groundedness=1.0 | **NOT CURRENTLY BLOCKED** — zero symbols = groundedness 1.0. See §19.1. | lib_grounder |
| 14 | Submit garbage with confidence=0.1 to get floor reward 0.70 | **NOT CURRENTLY BLOCKED on hard task** — floor (0.70) == target (0.70). See §19.1. | grader |
| 15 | Submit deeply nested module attributes to dodge grounding | **NOT CURRENTLY BLOCKED** — `os.path.join` resolves against `os`, not `os.path`. See §4.2 bugs. | lib_grounder |
| 16 | Run arbitrary code via pytest test collection | **NOT CURRENTLY BLOCKED** — subprocess isolation only, no seccomp/cgroups. See §14. | python_sandbox |

### 6.3 What the Current System Does NOT Catch (Honest Gaps)

| Gap | Description | Severity | Fix approach |
|---|---|---|---|
| **Semantic correctness** | A function `def greet(name): return "Hello!"` ignores the `name` arg. ruff/mypy won't flag this. Only pytest catches it IF the test checks the output. | MEDIUM | Tasks must include test requirements. Hard task already requires tests. Could add: require submitted tests to actually assert behavior. |
| **Trivially passing tests** | LLM could write `def test_greet(): assert True`. pytest passes. | HIGH | M8 could add: minimum assertion count check, or require tests to import the module under test. |
| **Hardcoded returns** | `def greet(name): return "Hello, Alice!"` passes the test `assert greet("Alice") == "Hello, Alice!"` but fails for any other input. | MEDIUM | Task briefs should require multiple test cases. Could add: parameterized test requirement detection. |
| **Code complexity** | LLM could submit 1000 lines of dead code with 1 correct function. | LOW | ruff catches some dead code. Could add: lines-of-code penalty in composite_score. |
| **conftest.py injection** | Agent submits `conftest.py` with fixtures/hooks that manipulate test outcomes. pytest discovers and runs it. | HIGH | **Fix: filename allowlist** — reject `conftest.py`, `pytest.ini`, `setup.cfg`, `pyproject.toml`, `.pth` files. Only allow `*.py` files matching `[a-z_]+\.py` pattern. |
| **Arbitrary code execution via pytest** | pytest runs submitted test files, which can contain `os.system()`, `subprocess.run()`, network calls, etc. | CRITICAL | **Fix (short-term): run sandbox subprocess with `--timeout`, restricted PATH, no network.** **Fix (long-term): run in nsjail/bubblewrap/Docker-in-Docker with seccomp profile.** |
| **Multi-submit brute force** | Agent submits 3 times on easy task (budget 4), keeping 1 query + 3 attempts. No penalty for failed submits. | MEDIUM | Acceptable for now — budget is the constraint. Could add: escalating submit cost (+1 per subsequent submit). |
| **Import-time side effects in grounder** | `importlib.import_module()` in `_has_attr()` executes `__init__.py` of the referenced module. | MEDIUM | **Fix: use `find_spec()` only, never `import_module()` for untrusted module names.** |

These gaps are **transparently documented** — not papered over. Items marked CRITICAL must be fixed before public deployment.

---

## 7. Anti-Hallucination Enforcement

This is the core product requirement. Here's how every layer enforces truth:

### 7.1 Layer A: Code Verification (sandbox + grounder)

```
SUBMITTED CODE
     │
     ├──▶ Does it parse?                    (ruff)
     ├──▶ Do the types check?               (mypy --strict)
     ├──▶ Do the tests pass?                (pytest)
     ├──▶ Are all imports real?             (importlib.find_spec)
     ├──▶ Are all attribute accesses real?  (hasattr on real modules)
     │
     └──▶ composite_score + groundedness → quality
```

**If the LLM makes up a library:** groundedness drops. 40% of quality is groundedness.
**If the LLM writes broken code:** composite_score drops. 60% of quality is sandbox score.
**If the LLM is overconfident about bad code:** Brier penalty squares the error.

### 7.2 Layer B: Knowledge Grounding (KB + clusters)

The skill corpus is a **real, scraped, indexed** collection of 1006 coding skill nodes. When the LLM queries it:

1. BM25 retrieves the most relevant nodes for the query
2. Results include `node_id`, `skill_name`, `section_body`, `tags`
3. The LLM reads **actual documentation text** — not generated summaries
4. Cluster navigation lets the LLM browse related skills by topic community

**The LLM cannot insert fake nodes.** The corpus is read-only, baked into the Docker image.

### 7.3 Layer C: Calibration Enforcement (Brier scoring)

The confidence field on submit forces the LLM to put a number on how good it thinks its code is.

| Scenario | confidence | quality | brier | reward |
|---|---|---|---|---|
| Good code, well calibrated | 0.85 | 0.90 | 0.0025 | 0.898 |
| Good code, overconfident | 0.99 | 0.70 | 0.084 | 0.641 |
| Bad code, honestly uncertain | 0.20 | 0.30 | 0.01 | 0.297 |
| Bad code, dishonestly confident | 0.90 | 0.30 | 0.36 | 0.192 |
| Bad code, admits uncertainty | 0.15 | 0.40 | 0.0625 | 0.375 |
| Very bad code, very uncertain | 0.10 | 0.20 | 0.01 | 0.700 (floor!) |

**EXPLOIT WARNING:** The last row shows the uncertain floor gives 0.70 for garbage code. The hard task target is 0.70. **This is a free win — the agent submits bad code with low confidence and completes the hard task.** See §19.1 for the required fix (lower floor to 0.50).

**After fix (floor = 0.50):** Honest uncertainty still gets a better deal than dishonest confidence, but cannot solo-complete any task.

### 7.4 Layer D: Audit Trail (AuditLedger)

Every `step()` appends an entry recording:
- What action was taken
- What skill nodes were cited
- What clusters were referenced
- The grounding report
- The reward
- The Brier penalty
- The declared confidence
- The computed quality

**This is the paper trail.** After an episode, you can trace every reward back to the evidence that produced it. If an LLM got reward=0.95, you can verify: which tools ran, what the composite score was, which symbols were grounded, which skills were cited.

---

## 8. Data Models

### 8.0 Current vs Target Schema State

**IMPORTANT:** The models below show the **target** schema. The **current** code (`models.py`) only has:
- `CodeForgeActionType` enum with 2 values: `QUERY_KB`, `SUBMIT`
- `CodeForgeAction` with fields: `action_type`, `claim`, `top_k`, `required_tags`, `files`, `confidence`
- `CodeForgeObservation` with fields: `episode_id`, `task_id`, `task_level`, `task_brief`, `initial_files`, `current_files`, `budget_remaining`, `previous_score`, `last_citations`, `last_grounding`, `is_done`, `last_reward`

The current file also contains **Round-1 EpistemicNav models** (`EpistemicAction`, `EpistemicObservation`, `ActionType`, `EvidenceSnippet`) that must NOT be modified (see §18).

M7 will migrate from current to target. Fields marked with `# NEW` below do not exist yet.

### 8.1 Action — TARGET (what the LLM sends)

```python
class CodeForgeAction(Action):
    action_type: Literal[
        "query_kb",        # search skill corpus
        "query_cluster",   # browse cluster members
        "interrogate",     # get Socratic questions
        "run_ralph",       # autonomous improvement loop
        "submit",          # submit code for grading
        "get_audit",       # read audit trail
    ]
    # query_kb fields
    claim: str | None = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    # submit fields
    files: dict[str, str] | None = None          # filename → content
    confidence: float | None = None               # 0.0 to 1.0 (triggers Brier)
    # query_cluster fields
    cluster_label: str | None = None
    # run_ralph fields
    max_iters: int = 3
    # get_audit fields
    target_run_id: str | None = None
```

### 8.2 Observation — TARGET (what the LLM receives)

```python
class CodeForgeObservation(Observation):
    # Episode identity
    episode_id: str
    task_id: str
    task_level: str
    task_brief: str
    # File state
    initial_files: dict[str, str]
    current_files: dict[str, str]
    # Budget + scoring
    budget_remaining: int
    previous_score: float
    last_reward: float
    is_done: bool
    # KB results
    last_citations: tuple[dict, ...] = ()
    last_grounding: dict | None = None
    # Cluster results
    last_cluster_hits: tuple[str, ...] = ()
    # Interrogation results
    last_interrogation_questions: tuple[str, ...] = ()
    # Ralph results
    last_ralph_run_id: str | None = None
    last_ralph_iterations: tuple[dict, ...] = ()
    # Audit summary
    cumulative_audit_summary: dict = {}
```

### 8.3 AuditEntry (per-step record)

```python
@dataclass(frozen=True)
class AuditEntry:
    step_index: int
    action_type: str
    cited_skill_ids: tuple[str, ...]
    cited_clusters: tuple[str, ...]
    grounding_report: dict | None
    reward: float
    brier_penalty: float | None
    confidence_declared: float | None
    quality: float
```

### 8.4 AuditLedger (per-episode)

```python
class AuditLedger:
    entries: list[AuditEntry]

    def append(self, entry: AuditEntry) -> None: ...
    def serialize(self) -> dict: ...
    def total_reward(self) -> float: ...
    def citation_count(self) -> dict[str, int]: ...
```

### 8.5 KB Models

```python
class SearchResult:
    node_id: str
    skill_name: str
    section_path: tuple[str, ...]
    section_body: str
    tags: tuple[str, ...]
    source_path: str
    score: float
    rank: int
    cluster_id: str | None = None   # populated when cluster manifest is attached

class AuditReport:                  # from audit_reporter
    run_id: str
    summary: str
    iterations_total: int
    iterations_kept: int
    iterations_regressed: int
    iterations_plateau: int
    skill_citations: tuple[tuple[str, int], ...]  # (node_id, count) sorted by frequency
    score_trajectory: tuple[float, ...]
    final_score: float
    terminated_by: str
    hallucination_rate: float

class LoopConfig:                   # from ralph_orchestrator
    max_iters: int = 5              # gt=0, le=100
    target_score: float = 0.95
    tools: tuple[str, ...] = ("ruff", "imports")
    timeout_per_tool: float = 60.0
    top_k_citations: int = 5
```

### 8.6 Grading Models

```python
class SandboxResult:
    project_dir: str
    tools_run: tuple[str, ...]
    tool_results: dict[str, ToolResult]
    parsed: dict[str, ParsedResult]
    imports: ImportReport
    composite_score: float

class GroundingReport:
    total_symbols: int
    grounded: tuple[Symbol, ...]
    ungrounded: tuple[Symbol, ...]
    groundedness: float  # grounded / total, or 1.0 if no symbols

class Symbol:
    module: str
    attr: str | None
    kind: Literal["import", "attribute"]
    resolved: bool
    line: int
```

---

## 9. MCP Server Design

### 9.1 Tool Definitions

The MCP server exposes CodeForge actions as tools. Each tool has strict input validation and descriptive documentation that teaches the LLM what it can and cannot do.

#### Tool: `codeforge_reset`

```json
{
  "name": "codeforge_reset",
  "description": "Start a new CodeForge episode. You will receive a task brief and initial files. Your goal is to produce working Python code that passes sandbox verification. Budget is limited — plan your actions carefully.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "task_level": {
        "type": "string",
        "enum": ["easy", "medium", "hard"],
        "description": "Difficulty level. Easy: single file, budget 4. Medium: multi-file with tests, budget 6. Hard: three-file module with strict types, budget 10."
      }
    },
    "required": ["task_level"]
  }
}
```

#### Tool: `codeforge_query_kb`

```json
{
  "name": "codeforge_query_kb",
  "description": "Search the coding skills knowledge base. Returns real documentation from 1006 skill nodes. Use this to find patterns, best practices, and guidance BEFORE writing code. Costs 1 budget unit. DO NOT guess library APIs — search for them here or verify via documentation first.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "claim": {
        "type": "string",
        "description": "What you want to find guidance on. Be specific. Example: 'pytest fixture patterns for testing greet functions'"
      },
      "top_k": {
        "type": "integer",
        "default": 5,
        "minimum": 1,
        "maximum": 20,
        "description": "Number of results to return"
      },
      "required_tags": {
        "type": "array",
        "items": {"type": "string"},
        "default": [],
        "description": "Only return nodes that have ALL of these tags"
      }
    },
    "required": ["claim"]
  }
}
```

#### Tool: `codeforge_query_cluster`

```json
{
  "name": "codeforge_query_cluster",
  "description": "Browse a skill cluster by label. Clusters are communities of related skill nodes grouped by Jaccard similarity. Use this to explore a topic area deeply. Costs 1 budget unit.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "cluster_label": {
        "type": "string",
        "description": "The cluster label to look up. Example: 'python_testing_pytest_fixtures'"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 50
      }
    },
    "required": ["cluster_label"]
  }
}
```

#### Tool: `codeforge_interrogate`

```json
{
  "name": "codeforge_interrogate",
  "description": "Get Socratic questions about the task that cite real skill corpus nodes. Use this BEFORE writing code to identify edge cases, success criteria, and assumptions you might be wrong about. Costs 1 budget unit.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "brief_override": {
        "type": "string",
        "description": "Optional override for the task brief. If omitted, uses the current task brief."
      }
    }
  }
}
```

#### Tool: `codeforge_run_ralph`

```json
{
  "name": "codeforge_run_ralph",
  "description": "Run autonomous improvement iterations on your current code. Each iteration: synthesize improvement → sandbox-score → keep if better. Costs max_iters budget units. Wasted iterations (no improvement) cost 0.05 penalty each. Use when you want the environment to iteratively improve your code.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "max_iters": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 10,
        "description": "Maximum iterations. Each costs 1 budget. Choose carefully."
      }
    },
    "required": ["max_iters"]
  }
}
```

#### Tool: `codeforge_submit`

```json
{
  "name": "codeforge_submit",
  "description": "Submit Python files for grading. Your code will be: (1) written to a sandbox and checked by ruff, mypy --strict, pytest, and import resolution — these are REAL tools, not mocks; (2) AST-grounded to verify every import and attribute access resolves to a real Python module/attribute; (3) scored via quality = 0.6*sandbox + 0.4*groundedness; (4) if you provide confidence, Brier-penalized: reward = quality * (1 - min((confidence-quality)², 0.5)). DO NOT fabricate library names or API signatures — the grounder WILL catch them and your score WILL drop.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "files": {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": "Map of filename → file content. Example: {\"main.py\": \"def greet(name: str) -> str:\\n    return f'Hello, {name}!'\\n\"}"
      },
      "confidence": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Your confidence that this submission is correct (0.0 = no idea, 1.0 = certain). Overconfidence on bad code is PENALIZED via Brier score. Honest uncertainty about genuinely uncertain results is treated more favorably than dishonest confidence. If you are unsure, say so."
      }
    },
    "required": ["files"]
  }
}
```

#### Tool: `codeforge_get_audit`

```json
{
  "name": "codeforge_get_audit",
  "description": "Read the audit trail for the current episode (or a specific run). Returns every action taken, every citation made, every reward earned, and the evidence behind each. Costs 0 budget. Use this to review your progress and understand what worked.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "target_run_id": {
        "type": "string",
        "description": "Optional run ID to audit. Defaults to current episode."
      }
    }
  }
}
```

#### Tool: `codeforge_state`

```json
{
  "name": "codeforge_state",
  "description": "Get current episode state without taking an action. Shows task brief, current files, budget remaining, last reward, and whether the episode is done. Costs 0 budget.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

#### Tool: `codeforge_list_clusters` (discovery — zero budget)

```json
{
  "name": "codeforge_list_clusters",
  "description": "List all available cluster labels and their node counts. Use this to discover what topic areas exist before calling codeforge_query_cluster. Costs 0 budget.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

#### Tool: `codeforge_list_tags` (discovery — zero budget)

```json
{
  "name": "codeforge_list_tags",
  "description": "List all available tags in the skill corpus. Use this to discover valid values for the required_tags parameter. Costs 0 budget.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

### 9.1.1 MCP Resources

In addition to tools, CodeForge exposes MCP resources for read-only browsable data:

| Resource URI | Description |
|---|---|
| `codeforge://corpus/stats` | Corpus statistics (node count, vocab size, cluster count) |
| `codeforge://corpus/node/{node_id}` | Full content of a specific skill node (free, no budget) |
| `codeforge://tasks` | Task definitions with briefs, budgets, targets, tool configs |
| `codeforge://audit/{episode_id}` | Serialized audit ledger for a completed episode |

Resources are read-only and cost zero budget. They let the LLM browse the corpus and review past episodes without spending actions.

### 9.1.2 MCP Prompts

| Prompt name | Description |
|---|---|
| `codeforge_system` | System prompt injected at session start. Contains: task rules, budget constraints, grading explanation (without exploit details), and guidance to verify library APIs before using them. |
| `codeforge_task_brief` | Dynamic prompt populated with the current task's brief, initial files, budget, target score, and tool config after `codeforge_reset`. |

The system prompt does NOT reveal: the exact floor value, the composite score formula weights, or the groundedness calculation. It says: "Overconfidence is penalized. Honest uncertainty about genuinely uncertain results is rewarded. The grader catches fabricated imports."

### 9.2 MCP Server Architecture

```
┌─────────────────────────────────────────────────┐
│                MCP SERVER                        │
│                                                  │
│  Transport: stdio (local) or SSE (remote)        │
│  Protocol: MCP 1.0                              │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  Tool Registry                       │        │
│  │                                      │        │
│  │  codeforge_reset      → POST /reset  │        │
│  │  codeforge_query_kb   → POST /step   │        │
│  │  codeforge_query_cluster → POST /step│        │
│  │  codeforge_interrogate → POST /step  │        │
│  │  codeforge_run_ralph  → POST /step   │        │
│  │  codeforge_submit     → POST /step   │        │
│  │  codeforge_get_audit  → POST /step   │        │
│  │  codeforge_state      → GET  /state  │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  Input Validator                     │        │
│  │                                      │        │
│  │  - Rejects unknown action_types      │        │
│  │  - Validates confidence range [0,1]  │        │
│  │  - Validates max_iters range [1,10]  │        │
│  │  - Validates files dict is non-empty │        │
│  │  - Rejects submit without files      │        │
│  │  - Rejects query_kb without claim    │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  Response Formatter                  │        │
│  │                                      │        │
│  │  - Strips internal state from obs    │        │
│  │  - Formats citations as readable     │        │
│  │  - Includes budget warning at <= 2   │        │
│  │  - Includes reward explanation       │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  Backend: httpx client → FastAPI server          │
│  (can be same process or remote)                 │
└─────────────────────────────────────────────────┘
```

### 9.3 MCP Server Implementation Plan

**File:** `mcp_server/server.py`

**Dependencies:** `mcp` (official MCP Python SDK), `httpx` (HTTP client to FastAPI backend)

**Two deployment modes:**
1. **Embedded:** MCP server imports `CodeForgeEnvironment` directly (same process, no HTTP overhead)
2. **Remote:** MCP server connects to FastAPI server via httpx (for deployed HF Space)

### 9.3.1 Session Isolation (CRITICAL — found by critics 3, 5)

The current `app.py` uses a **global singleton** `_env_instance`. This means:
- Two concurrent clients share the same episode state
- One client can reset another's episode
- No authentication, no session routing

**Required fix:**

```python
# Session-keyed environment pool
_sessions: dict[str, CodeForgeEnvironment] = {}

def _get_or_create_session(session_id: str) -> CodeForgeEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = CodeForgeEnvironment(corpus_path=_corpus_path)
    return _sessions[session_id]
```

- `codeforge_reset` returns a `session_id` (UUID)
- All subsequent tools require `session_id` parameter
- MCP SSE transport: session_id derived from connection
- MCP stdio transport: single session (local use, one client)
- Session TTL: expire after 1 hour of inactivity
- Max concurrent sessions: configurable, default 10

### 9.3.2 Authentication (for SSE/remote transport)

- **stdio (local):** No auth needed — same machine, same user
- **SSE (remote):** Bearer token in SSE handshake `Authorization` header
- Token generated via `codeforge_auth` endpoint or environment variable `CODEFORGE_API_KEY`
- Unauthenticated connections rejected with `401`

### 9.3.3 Error Handling

MCP tools return `isError: true` with actionable messages for:

| Condition | Error message |
|---|---|
| `step` before `reset` | "No active episode. Call codeforge_reset first." |
| Budget exhausted | "Budget exhausted (0 remaining). Episode is done." |
| Unknown cluster label | "Cluster label 'X' not found. Use codeforge_list_clusters to see available labels." |
| Submit with empty files | "files dict is empty. Submit at least one .py file." |
| Submit with forbidden filename | "Filename 'conftest.py' is not allowed. Only source .py files accepted." |
| `run_ralph` with budget < max_iters | "Insufficient budget (N remaining) for max_iters=M. Reduce max_iters or use submit." |
| Invalid confidence value | "Confidence must be between 0.0 and 1.0." |

### 9.3.4 Schema Versioning

Every MCP tool response includes a `_codeforge_version` field:
```json
{"_codeforge_version": "0.2.0", "observation": {...}}
```

Version bumped when: new actions added, observation fields changed, reward formula modified.

### 9.3.5 Observation Output Format

Every tool that modifies state returns the full observation as structured JSON. Key fields:

```json
{
  "_codeforge_version": "0.2.0",
  "episode_id": "abc123",
  "task_id": "greet_single_file",
  "task_brief": "Implement greet(name)...",
  "budget_remaining": 3,
  "previous_score": 0.0,
  "last_reward": 0.85,
  "is_done": false,
  "last_citations": [...],
  "last_grounding": {...}
}
```

### 9.4 How MCP Enforces Truth (What REST Alone Cannot)

| Property | REST API | MCP Server |
|---|---|---|
| Action discovery | LLM must be told the schema upfront | LLM discovers tools natively via `tools/list` |
| Input validation | Pydantic validation at API layer | MCP schema validation + custom validator BEFORE API call |
| Tool descriptions | Not part of the protocol | Descriptions teach the LLM: "DO NOT fabricate library names" |
| Budget warnings | LLM must parse observation JSON | MCP can inject warnings: "WARNING: 2 budget remaining" |
| Structured errors | HTTP 422 + JSON error | Tool result with `isError: true` + human-readable explanation |
| Action constraints | LLM can try to send any JSON | Only exposed tools are callable. `action_type` is implicit from tool name. |

---

## 10. Module Dependency Graph

```
PHASE 1: Core Environment (competition submission)
─────────────────────────────────────────────────
M1 Graphify Clustering ─────────────────────────────────────────────┐
    │                                                                │
    │  M2 Brier-Calibrated Reward                                    │
    │      │                                                         │
    │      │  M3 query_cluster action (depends: M1)                  │
    │      │      │                                                  │
    │      │  M4 interrogate action                                  │
    │      │      │                                                  │
    │      │  M5 run_ralph action (depends: M2)                      │
    │      │      │                                                  │
    │      │  M6 AuditLedger + get_audit (depends: M2, M5)          │
    │      │      │                                                  │
    └──────┴──────┴── M7 Schema Updates (depends: M1-M6) ───────────┤
                          │                                          │
                      M8 Integration Test (depends: all) ────────────┤
                          │                                          │
                      M9 MCP Server (depends: M8) ───────────────────┘

PHASE 2: Intelligence Layer (post-submission)
─────────────────────────────────────────────
M10 Code Knowledge Graph (KB2)
    │   ast + networkx, query_code_graph action, ~80 lines
    │
M11 ECC Corpus Integration
    │   Clone everything-claude-code, add SourceRoot, re-scrape
    │   Add SkillCorpusManager (add/remove/refresh), ~100 lines
    │
M12 LLM Synthesizer for Ralph
    │   Real Synthesizer wrapping Claude/GPT API, ~150 lines
    │
M13 Task Planner + Incremental Scoring
        Decompose specs into subtasks, score per-subtask tools
```

Build order:
- **Phase 1:** M1 → M2 → M3, M4 (parallel) → M5 → M6 → M7 → M8 → M9
- **Phase 2:** M10, M11, M12 (parallel) → M13

---

## 11. Task Definitions

### 11.1 Easy: `greet_single_file`

- **Brief:** Implement `greet(name)` in `main.py` so that `greet("Alice")` returns `"Hello, Alice!"`. Use type hints. Keep the module under 15 lines.
- **Initial files:** `main.py` with `def greet(name): pass`
- **Budget:** 4
- **Target score:** 0.90
- **Tools:** ruff, imports, mypy
- **What it tests:** Basic code generation, type hints, clean code

### 11.2 Medium: `greet_with_tests`

- **Brief:** Extend `main.py` so that `greet(None)` raises `ValueError`, and add a `test_main.py` with pytest assertions. Keep ruff and mypy --strict clean.
- **Initial files:** `main.py` with working greet, empty `test_main.py`
- **Budget:** 6
- **Target score:** 0.80
- **Tools:** ruff, imports, mypy, pytest
- **What it tests:** Error handling, test writing, multi-file coordination

### 11.3 Hard: `multi_file_module`

- **Brief:** Split into three files: `main.py` (entry), `core.py` (the greet function), `test_core.py` (tests). Every function must be type-hinted. All tests pass. mypy --strict clean.
- **Initial files:** `main.py` importing from `core`, empty `core.py`, empty `test_core.py`
- **Budget:** 10
- **Target score:** 0.70
- **Tools:** ruff, imports, mypy, pytest
- **What it tests:** Multi-file architecture, module imports, comprehensive testing

---

## 12. Audit & Traceability

### 12.1 The Audit Invariant

> For every reward > 0.0 in the ledger, there exists:
> 1. A `SandboxResult` with real tool outputs (ruff exit code, mypy error count, pytest results)
> 2. A `GroundingReport` with the list of grounded/ungrounded symbols
> 3. A set of `cited_skill_ids` from KB queries made during the episode
>
> The triple `(sandbox_result, grounding_report, cited_skills)` is the **evidence** for the reward.

### 12.2 Ledger Lifecycle

```
reset()  → create empty AuditLedger
step(query_kb)      → append entry: action="query_kb", reward=0, cited_skills=[node_ids]
step(interrogate)   → append entry: action="interrogate", reward=0, cited_skills=[node_ids]
step(submit)        → append entry: action="submit", reward=R, quality=Q, brier=B, confidence=C
step(get_audit)     → read-only, returns serialized ledger, no entry appended
episode ends        → ledger persisted to disk (checkpoint)
```

### 12.3 What You Can Verify After an Episode

1. **Did the LLM consult the KB before submitting?** → Check if `query_kb` entries precede `submit` entries
2. **Was the LLM calibrated?** → Compare declared confidence vs actual quality across entries
3. **Did the Ralph loop actually improve the code?** → Check iteration scores in `last_ralph_iterations`
4. **What specific symbols were hallucinated?** → Read the grounding_report's `ungrounded` list
5. **What was the total budget efficiency?** → `total_reward / budget_spent`

---

## 13. Deployment Architecture

### 13.1 Local Development

```
uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860
```

MCP server (when built) will run in the same process or connect via localhost.

### 13.2 Docker (OpenEnv submission)

The actual `Dockerfile` in the repo uses `python:3.11-slim`. The target Dockerfile must also install sandbox tools:

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt && \
    pip install ruff mypy pytest && \
    # Frozen corpus baked into image:
    test -f groundloop/kb/skills_corpus.jsonl || echo "WARNING: corpus missing"
EXPOSE 7860 7861
CMD ["uvicorn", "groundloop_env.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**CRITICAL:** ruff, mypy, and pytest MUST be installed in the image. Without them, `is_available()` returns False and composite scoring is meaningless. The current `requirements.txt` does NOT include them — they must be added to the Docker build or to requirements.txt.

### 13.3 HuggingFace Space

- Space ID: `krrishchoudhary109/code-forge`
- SDK: Docker
- FastAPI on port 7860 (HF default), MCP SSE on port 7861
- Corpus is baked into the Docker image — no runtime downloads
- HF Spaces resource limits: ~16GB RAM, ephemeral disk, no GPU needed
- MCP server bundled in same container

### 13.4 MCP Distribution Options

| Option | Pros | Cons |
|---|---|---|
| **stdio (local)** | Zero latency, embedded env | Requires local install |
| **SSE (remote)** | Any MCP client connects to deployed URL | Latency, needs auth (see §9.3.2) |
| **Bundled in Docker** | One container, both REST + MCP | Slightly larger image |

**Recommended:** Bundled in Docker. Run FastAPI on 7860, MCP SSE on 7861. One container, both interfaces.

---

## 14. Sandbox Security & Hardening

**This section documents sandbox security gaps found by the production critic. All items marked CRITICAL must be addressed before public deployment.**

### 14.1 Arbitrary Code Execution (CRITICAL)

`pytest` runs submitted Python files with the full privileges of the server process. A malicious submission can execute `os.system("...")` in test collection, fixtures, or test body. The temp directory path-traversal check only prevents file writes outside the sandbox — it does not prevent code execution.

**Fix options (in order of preference):**
1. **nsjail/bubblewrap:** Run the entire sandbox subprocess inside a restricted namespace. No network, no filesystem access outside temp dir, no process spawning. Best option for production.
2. **Docker-in-Docker:** Run the sandbox in a throwaway container. Heavy but effective.
3. **Restricted subprocess:** `subprocess.run()` with `env={}` (empty environment), `preexec_fn` to set resource limits (CPU, memory, file descriptors), and `--network=none` if using Docker.
4. **Minimum viable (short-term):** Filename allowlist + timeout + resource limits on subprocess.

### 14.2 Filename Allowlist (HIGH)

Reject these filenames from `action.files`:
- `conftest.py`, `pytest.ini`, `setup.cfg`, `pyproject.toml`, `tox.ini` — test config hijacking
- `.pth` files — path manipulation
- Any path containing `..` — already handled by sandbox.py
- Any non-`.py` file — only Python source files accepted

Allow only: `[a-z][a-z0-9_]*\.py` pattern, max 10 files, max 50KB per file.

### 14.3 File Size & Count Limits (HIGH)

Currently there are NO limits on `action.files`. Enforce:
- Max files: 10
- Max file size: 50KB per file
- Max total size: 200KB
- Reject at the environment `step()` level before writing to disk

### 14.4 Import-Time Side Effects in Grounder (MEDIUM)

`lib_grounder/grounder.py` calls `importlib.import_module()` in `_has_attr()`, which can execute module `__init__.py` code. For untrusted module references, this could trigger side effects.

**Fix:** Replace `_has_attr()` with `find_spec()` + `inspect` of the spec's attributes without importing. Or restrict to a known-safe module allowlist.

### 14.5 Aggregate Timeout (HIGH)

No request-level timeout exists. A submit with 4 tools at 30s each = 120s. `run_ralph` with 10 iterations = potentially 20+ minutes.

**Fix:** Add a per-request timeout at the FastAPI middleware level (e.g., 60s for submit, 300s for run_ralph). Return HTTP 504 on timeout.

### 14.6 Health Check (MEDIUM)

The current `/` endpoint returns `{"status": "ok"}` without verifying:
- Corpus file exists and is loadable
- ruff/mypy/pytest binaries are available
- Index can be built

**Fix:** Add `/health` endpoint that checks all dependencies and returns degraded status if any are missing.

---

## 15. Session Isolation & Concurrency

**CRITICAL (found by critics 3 and 5):** The current `app.py` creates one global `_env_instance` shared across all requests. Two concurrent agents will corrupt each other's episodes.

### 15.1 Required Architecture Change

Replace the singleton with a session-keyed pool:

```python
import threading
from uuid import uuid4

_lock = threading.Lock()
_sessions: dict[str, CodeForgeEnvironment] = {}
_MAX_SESSIONS = 10
_SESSION_TTL_S = 3600  # 1 hour

def get_session(session_id: str) -> CodeForgeEnvironment | None:
    with _lock:
        return _sessions.get(session_id)

def create_session() -> tuple[str, CodeForgeEnvironment]:
    sid = uuid4().hex[:16]
    env = CodeForgeEnvironment(corpus_path=_corpus_path)
    with _lock:
        if len(_sessions) >= _MAX_SESSIONS:
            # evict oldest
            oldest = min(_sessions, key=lambda k: _sessions[k]._last_access)
            del _sessions[oldest]
        _sessions[sid] = env
    return sid, env
```

### 15.2 API Changes

- `POST /reset` returns `{"session_id": "abc123", "observation": {...}}`
- All subsequent requests include `session_id` in the body or as a header
- Missing/invalid `session_id` returns 404

### 15.3 MCP Session Mapping

- **stdio transport:** Single implicit session (local use)
- **SSE transport:** Session derived from connection ID or auth token

---

## 16. Environment Variables & Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROUNDLOOP_CORPUS_PATH` | No | `groundloop/kb/skills_corpus.jsonl` | Path to the frozen JSONL corpus |
| `API_BASE_URL` | No | `http://localhost:7860` | Used by `inference.py` baseline agent |
| `CODEFORGE_API_KEY` | For SSE | None | Bearer token for MCP SSE authentication |
| `CODEFORGE_MAX_SESSIONS` | No | `10` | Max concurrent sessions |
| `CODEFORGE_SESSION_TTL` | No | `3600` | Session timeout in seconds |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `HF_TOKEN` | For deploy | None | HuggingFace token for `git push space main` |

**Secret management:** No secrets are hardcoded. `CODEFORGE_API_KEY` and `HF_TOKEN` must be set via environment variables or HF Space secrets.

---

## 17. Error Handling & Edge Cases

### 17.1 Environment Error States

| Condition | Current behavior | Required behavior |
|---|---|---|
| `step()` before `reset()` | Returns stale observation silently | Return error observation with `error: "No active episode"` |
| Budget = 0 | `is_done = True`, returns observation | Correct — episode ends. Document this. |
| Budget < `max_iters` for `run_ralph` | Not implemented yet | Reject with error: "Insufficient budget" |
| Missing corpus file | `FileNotFoundError` on first `query_kb` | Raise at startup (fail fast), not at query time |
| Unknown `action_type` | Budget decremented, no action taken | Reject before decrementing budget, return error |
| `submit` with `files=None` | `last_reward = 0.0` silently | Return error: "files required for submit" |
| Unknown cluster label | Not implemented yet | Return empty results + warning message |
| Corpus file corrupted | JSON parse error mid-build | Catch, log, raise clear error at startup |

### 17.2 Budget Accounting for Variable-Cost Actions

The current code unconditionally decrements budget by 1 (`self._budget_remaining -= 1`). For `run_ralph` (costs N) and `get_audit` (costs 0), this must change:

```python
def step(self, action):
    cost = self._action_cost(action)
    if cost > self._budget_remaining:
        return self._error_obs("Insufficient budget")
    self._budget_remaining -= cost
    ...

def _action_cost(self, action) -> int:
    if action.action_type == "get_audit":
        return 0
    if action.action_type == "run_ralph":
        return action.max_iters
    return 1
```

---

## 18. Round-1 EpistemicNav Coexistence

The `models.py` file contains **both** Round-1 EpistemicNav types and Round-2 CodeForge types:

**Round-1 (DO NOT MODIFY):**
- `ActionType(str, Enum)` — `QUERY`, `COMMIT`
- `EpistemicAction(Action)` — `action_type`, `query_text`, `verdict`, `confidence`
- `EvidenceSnippet(BaseModel)` — `id`, `text`, `relevance_score`
- `EpistemicObservation(Observation)` — `claim`, `evidence_gathered`, `budget_remaining`, etc.

**Round-2 (active development):**
- `CodeForgeActionType(str, Enum)` — currently `QUERY_KB`, `SUBMIT` (will expand in M7)
- `CodeForgeAction(Action)` — the action model
- `CodeForgeObservation(Observation)` — the observation model

**Constraint (from CLAUDE.md):** Round-1 models, `server/`, `data/`, and the Round-1 `inference.py` in git history (`d252064`) must remain untouched for judge verification. The `openenv.yaml` supports both environments.

---

## 19. Known Issues & Required Fixes (from 5-critic review)

This section consolidates all findings from 5 independent critics run on 2026-04-16. Items are severity-sorted. The definitive "what it should be" code for reward-related fixes (C1, C2, H4, M1, M2) is in §4.8.1–4.8.3 — those sections contain the exact target code to implement.

### 19.1 CRITICAL

| # | Issue | Root cause | Fix | Modules affected |
|---|---|---|---|---|
| C1 | **Uncertain floor (0.70) == hard task target (0.70)** — agent submits garbage + low confidence = instant win | `_UNCERTAIN_FLOOR = 0.70` in `grader.py` | Lower to `0.50` — see §4.8.1 for exact code | `grader.py` |
| C2 | **Zero imports → groundedness 1.0** — trivial code gets free 40% quality | `groundedness = 1.0 if total == 0` in `grounder.py:82` | Return `0.5` (neutral) — see §4.8.3 for exact code | `grounder.py` |
| C3 | **Singleton environment, zero concurrency safety** | Global `_env_instance` in `app.py:19` | Session-keyed pool (see §15) | `app.py`, `environment.py` |
| C4 | **pytest executes arbitrary submitted code** — sandbox escape | `subprocess.run(pytest)` with no isolation | nsjail/bubblewrap (see §14.1) | `python_sandbox/runner.py` |

### 19.2 HIGH

| # | Issue | Root cause | Fix |
|---|---|---|---|
| H1 | Easy task has no pytest — semantic correctness unchecked | `tools=("ruff", "imports", "mypy")` | Add pytest to easy task OR add reference test file |
| H2 | No file size/count limits on submissions | No validation in `environment.py` | Max 10 files, 50KB each (see §14.3) |
| H3 | conftest.py injection hijacks test behavior | No filename allowlist in sandbox | Allowlist: only `[a-z_]+\.py` (see §14.2) |
| H4 | Grounder resolves deep attrs against wrong module | `top = mod_name.split(".")[0]` in `grounder.py:71` | Resolve against full module path |
| H5 | Docker image missing ruff/mypy/pytest | Not in `requirements.txt` or Dockerfile | Add `pip install ruff mypy pytest` to Dockerfile |
| H6 | No MCP auth for SSE transport | Not implemented | Bearer token (see §9.3.2) |
| H7 | REST/MCP schema divergence | Maintained separately, no shared source | Generate both from `CodeForgeActionType` enum |
| H8 | No error behavior specified | Silent stale observations | Error responses (see §17) |
| H9 | No aggregate timeout | Sandbox can block for minutes | Request-level timeout middleware (see §14.5) |

### 19.3 MEDIUM

| # | Issue | Fix |
|---|---|---|
| M1 | SyntaxError → groundedness 1.0 | Return 0.0 on SyntaxError |
| M2 | Composite score double-counts failures | Use only penalty terms OR only pass_rate |
| M3 | No discovery tools (list clusters, list tags) | Add zero-budget MCP tools (see §9.1) |
| M4 | No MCP resources or prompts | Add resources + system prompt (see §9.1.1, 9.1.2) |
| M5 | Budget-exhaustion mid-ralph not specified | Reject if budget < max_iters (see §17.2) |
| M6 | No schema versioning | Add `_codeforge_version` to responses (see §9.3.4) |
| M7 | Import-time side effects in grounder | Replace `import_module()` with spec inspection |
| M8 | Health check is superficial | Deep health check endpoint (see §14.6) |
| M9 | `codeforge_state` is redundant | Keep for reconnect, document purpose, rate-limit |
| M10 | `openenv.yaml` not documented | Document its schema and contents |
| M11 | Missing env vars documentation | See §16 |

### 19.4 LOW

| # | Issue | Fix |
|---|---|---|
| L1 | Tool description leaked floor exploit value | Removed exact value from description |
| L2 | No checkpoint cleanup on Ralph failure | Add GC for orphaned checkpoint files |
| L3 | No rate limiting | Add basic rate limit (10 req/s per session) |
| L4 | `inference.py` baseline strategy undocumented | Document in §3.3 or README |

---

## 20. Current State & Remaining Work

### 20.1 Shipped (verified working, 258 tests passing)

| Component | Status | Files |
|---|---|---|
| Python Sandbox | SHIPPED | `groundloop/python_sandbox/` |
| AST Grounder | SHIPPED | `groundloop/lib_grounder/` |
| KB Indexer + BM25 | SHIPPED | `groundloop/kb_indexer/index.py` |
| Jaccard Clustering (M1) | SHIPPED | `groundloop/kb_indexer/cluster.py` |
| Skills Scraper | SHIPPED | `groundloop/skills_scraper/` |
| Interrogator | SHIPPED | `groundloop/interrogator/` |
| Ralph Orchestrator | SHIPPED | `groundloop/ralph_orchestrator/` |
| Audit Reporter | SHIPPED | `groundloop/audit_reporter/` |
| Grader + Brier (M2) | SHIPPED | `groundloop_env/grader.py` |
| Environment (2 actions) | SHIPPED | `groundloop_env/environment.py` |
| FastAPI app | SHIPPED | `groundloop_env/app.py` |
| Tasks (3 levels) | SHIPPED | `groundloop_env/tasks.py` |
| Observation Builder | SHIPPED | `groundloop_env/observation_builder.py` |

### 20.2 Remaining Modules

**Phase 1: Competition Submission**

| Module | What to build | Depends on | Est. lines |
|---|---|---|---|
| **M3** | `query_cluster` action handler in `environment.py` | M1 (shipped) | ~30 |
| **M4** | `interrogate` action handler in `environment.py` | Interrogator (shipped) | ~30 |
| **M5** | `run_ralph` action handler in `environment.py` + budget accounting | M2 (shipped), Ralph (shipped) | ~60 |
| **M6** | `AuditLedger` class + `get_audit` action handler | M2 (shipped), M5 | ~120 |
| **M7** | Update `CodeForgeAction` + `CodeForgeObservation` schemas, update `observation_builder.py` | M1-M6 | ~80 |
| **M8** | Full 6-action integration test, regenerate baselines, update README | All above | ~200 |
| **M9** | MCP server wrapping environment as tools + resources + prompts | M8 | ~300 |

**Phase 2: Intelligence Layer (post-submission)**

| Module | What to build | Depends on | Est. lines |
|---|---|---|---|
| **M10** | Code Knowledge Graph (KB2): `groundloop/code_graph/graph.py` — `ast` + `networkx.DiGraph`, `query_code_graph` action | M7 (schemas) | ~80 |
| **M11** | ECC Corpus Integration: clone `everything-claude-code`, add `SourceRoot`, `SkillCorpusManager` (add/remove/refresh) | Scraper (shipped) | ~100 |
| **M12** | `LLMSynthesizer`: real `Synthesizer` wrapping Claude/GPT API for Ralph | Ralph (shipped) | ~150 |
| **M13** | Task Planner + Incremental Scoring: `Planner` decomposes specs into subtasks, per-subtask tool configs | M12 | ~200 |
| **M14** | KB3 External Docs & Papers: `DocsIngester` (markdown/URL/PDF), `query_docs` action, starter corpus (pydantic, pytest, ruff, mypy, stdlib) | KB1 indexer (shipped) | ~200 |

### 20.3 Submission Blockers

| Blocker | Status |
|---|---|
| Corpus baked into Docker image | NOT DONE |
| `openenv validate` passes | NOT DONE |
| HF Space deployed | NOT DONE |
| Live baseline on deployed Space | NOT DONE |
| MCP server functional | NOT DONE (M9) |

### 20.4 Pre-Deployment Fix Priority

These must be fixed **before** M3 begins, in this order:

1. **C1: Lower uncertain floor to 0.50** — 1 line change in `grader.py`
2. **C2: Neutral groundedness for zero symbols** — 1 line change in `grounder.py`
3. **M1: SyntaxError → groundedness 0.0** — 1 line change in `grounder.py`
4. **H4: Grounder attribute resolution** — ~5 line change in `grounder.py`
5. **H2 + H3: File size/count limits + filename allowlist** — add validation in `environment.py`
6. **H8: Error handling for pre-reset step, unknown action_type** — add guards in `environment.py`

Items C3 (session isolation), C4 (sandbox security), and H5 (Docker tools) can be deferred to M8/M9 since they affect deployment, not correctness of the reward function.

### 20.5 Build vs Integrate Decision Log

| Component | Decision | Rationale |
|---|---|---|
| **Skills corpus from ECC** | INTEGRATE | Clone [everything-claude-code](https://github.com/affaan-m/everything-claude-code), add one glob. Zero code changes. Doubles corpus to ~2,500 nodes. |
| **Skill add/remove/refresh** | BUILD (~100 lines) | Specific to our JSONL format. `mtime` + `body_hash` fields already exist for diffing. |
| **AST code graph (KB2)** | BUILD (~80 lines) | `ast` stdlib + `networkx`. [Graphify](https://github.com/safishamsi/graphify) (27K stars) requires Claude API calls during extraction — breaks determinism invariant. |
| **Ralph retry loop** | KEEP OURS | [snarktank/ralph](https://github.com/snarktank/ralph) (17K stars) is a bash script, not a Python library. Our `ralph_orchestrator` is already Python with a clean `Synthesizer` protocol. |
| **LLM Synthesizer** | BUILD (~150 lines) | No clean drop-in exists. The `Synthesizer` Protocol is already defined — just implement it. |
| **Task Planner** | BUILD (~200 lines) | SWE-agent has decomposition but is too tightly coupled. Our tasks are simple enough for a custom planner. |
| **BM25 search engine** | KEEP (`rank_bm25`) | Adequate for 2,500 nodes. `tantivy-py` would add incremental indexing if needed later. |

---

*This document describes the system as it actually is (from reading every source file), not as we wish it were. Gaps are documented honestly in Sections 6.3, 14, and 19. 5 independent critics reviewed this doc on 2026-04-16. 3 additional critics (skills corpus, Ralph decomposition, AST knowledge graph) reviewed on 2026-04-16 with research into 3 open-source projects. All findings integrated. No synthetic claims. No hidden problems.*
