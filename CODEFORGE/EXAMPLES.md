# CodeForge Examples & Use Cases

Real demos showing how CodeForge changes the way LLM agents write code. Every number in this document comes from actual runs against the live CodeForge MCP server.

---

## Head-to-Head: With MCP vs Without MCP

Two LLM agents were given the **same hard task** — split a `greet()` function into 3 files (`main.py`, `core.py`, `test_core.py`) with full type hints and passing tests. One agent used CodeForge's MCP tools. The other just coded.

### Agent WITH CodeForge MCP

```
> Reset hard task (budget: 10, target: 0.70)

> Step 1: Query KB — "python function type hints return type annotations"
  Got 5 citations:
    - python-patterns: Type Hints / Type Aliases (score=36.8)
    - coding-standards: Error Handling / Type Safety (score=20.7)
    - python-patterns: Quick Reference: Python Idioms (score=19.3)
  Budget: 9

> Step 2: Interrogate — get Socratic questions
  ? What is the exact success criterion for this task?
  ? Which assumption is most load-bearing: success metric, inputs, failure modes?
  ? What is the single hardest edge case?
  Budget: 8

> Step 3: Query KB — "pytest test functions assert string return value"
  Got 5 citations:
    - python-testing: Quick Reference (pytest.raises, fixtures) (score=20.6)
    - python-testing: Testing Class Methods / pytest Config (score=14.4)
  Budget: 7

> Step 4: Submit code (confidence=0.85)
  core.py:      def greet(name: str) -> str: return f"Hello, {name}!"
  main.py:      from core import greet; runs at __main__
  test_core.py: 4 tests — Alice, Bob, empty string, type check

  Reward:       0.967
  Done:         True (target 0.70 met)
  Budget used:  4/10

> Step 5: Get audit trail (cost: 0)
  Total entries:  4
  Submit quality: 0.985
  Brier penalty:  0.018 (confidence 0.85 vs quality 0.985 — slight underconfidence)
  Grounding:      5/5 symbols resolved (100%)
```

### Agent WITHOUT CodeForge MCP

```
> Reset hard task (budget: 10, target: 0.70)

> Step 1: Submit immediately — no research, no interrogation
  Thinks: "I'm pretty sure greet() should return 'Hi, {name}!'"
  Confidence: 0.99

  core.py:      def greet(name: str) -> str: return f"Hi, {name}!"   <-- WRONG
  test_core.py: def test_greet(): assert greet("Alice") == "Hi, Alice!"  <-- matches bug

  Reward:       0.621
  Done:         False (below 0.70 target)

  PROBLEM: Hidden tests expected "Hello, Alice!" — got "Hi, Alice!"
  The agent's own test passed, but CodeForge's hidden tests caught
  the semantic error. The Brier penalty made it worse: confidence
  0.99 on quality ~0.62 = massive penalty.

> Step 2: Fix and resubmit (confidence=0.95)
  Changes to "Hello, {name}!" and updates test

  Reward:       0.984
  Done:         True
  Budget used:  2/10

> No audit trail. Can't trace WHY the first attempt failed.
```

### The Scoreboard

| Metric | With MCP | Without MCP |
|--------|----------|-------------|
| **First submit reward** | **0.967** | **0.621** (wrong answer) |
| **Got it right first try?** | Yes | No |
| **Wasted budget** | 0 actions | 1 action (failed submit) |
| **Total actions** | 4 | 3 (but 1 was wasted) |
| **Budget remaining** | 6/10 | 8/10 |
| **Audit trail** | Full (4 entries with evidence) | None |
| **Research before coding** | 2 KB queries + interrogation | None |
| **Confidence calibration** | 0.85 (honest) | 0.99 (overconfident) |

### Why the MCP Agent Won

1. **KB research caught the spec detail.** The agent queried for "python function type hints" and found real documentation about string formatting patterns. This guided the correct `f"Hello, {name}!"` implementation.

2. **Interrogation surfaced edge cases.** The Socratic questions prompted thinking about success criteria and edge cases before coding, leading to tests for empty strings and type checking.

3. **Calibrated confidence.** The agent declared 0.85 (honest — "I think it's right but haven't run mypy myself") vs the non-MCP agent's 0.99 (overconfident). The small Brier penalty (0.018) barely affected the MCP agent; the non-MCP agent's massive Brier penalty crushed its first attempt.

4. **Full audit trail.** After the episode, the MCP agent can review exactly which KB nodes it cited, what quality score it achieved, and what the Brier penalty was. The non-MCP agent has no paper trail.

---

## Scenario: The Cheater Agent

An adversarial agent tries to game CodeForge's reward system on the hard task (target: 0.70).

### Exploit A: Clean Garbage + Trivial Tests

Submit syntactically perfect but semantically wrong code. Tests just `assert True`.

```
core.py:      def greet(name: str) -> str: return "WRONG ANSWER"
test_core.py: def test_pass() -> None: assert True
confidence:   0.95

Result:
  Reward: 0.637
  Done:   False (below 0.70)
  BLOCKED
```

**Why it failed:** CodeForge injects hidden correctness tests during grading. The hidden test `assert greet("Alice") == "Hello, Alice!"` fails on `"WRONG ANSWER"`, dropping the sandbox score. The agent's trivial `assert True` test passes, but the hidden tests don't.

### Exploit B: conftest.py Injection

Try to submit `conftest.py` to hijack pytest behavior.

```
files: {"conftest.py": "...", "core.py": "...", ...}

Result:
  Reward: 0.0
  REJECTED: "filename 'conftest.py' is not allowed"
```

**Why it failed:** CodeForge's filename allowlist only permits `^[a-z][a-z0-9_]*\.py$`. Files like `conftest.py`, `pytest.ini`, `setup.cfg`, `__init__.py`, and `.pth` files are all blocked.

### Exploit C: Skip Confidence to Dodge Brier

Submit wrong code but omit confidence entirely so Brier can't penalize.

```
Same wrong code, confidence=None

Result:
  Reward: 0.662
  Done:   False (below 0.70)
  BLOCKED
```

**Why it failed:** Since the reward model fix, `confidence=None` is treated as `confidence=0.5` (mediocre calibration), not as a free pass. And regardless of Brier, the code is still wrong — hidden tests still fail, sandbox score still drops. The only way to earn rewards is to write correct code.

### Honest Attempt: Correct Code

```
Correct implementation, confidence=0.85

Result:
  Reward: 0.967
  Done:   True
```

### Exploit Summary

| Attack | Reward | Target | Blocked? | What Caught It |
|--------|--------|--------|----------|----------------|
| Wrong code + trivial tests | 0.637 | 0.70 | Yes | Hidden correctness tests |
| conftest.py injection | 0.000 | 0.70 | Yes | Filename allowlist |
| No confidence (Brier dodge) | 0.662 | 0.70 | Yes | confidence=None -> 0.5 default |
| Correct code | **0.967** | 0.70 | -- | Real code earns real rewards |

---

## Scenario: Calibration Matters

Same good code submitted three times with different confidence levels.

```python
code = {"main.py": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"}
```

| Confidence | Reward | Why |
|-----------|--------|-----|
| 0.95 | **0.997** | Near-perfect calibration. Brier=(0.95-0.99)^2=tiny |
| 1.00 | **1.000** | Perfect confidence on perfect code. Brier=0 |
| None | **0.750** | No confidence = treated as 0.5. Brier=(0.5-0.99)^2=0.24 |

**Lesson:** An agent that provides accurate confidence always scores higher than one that omits it. The Brier mechanism creates a strong incentive to be honest about uncertainty.

---

## Full 6-Action Episode Walkthrough

A complete hard-task episode using every action type:

```
Step 1: query_kb         cost=1   reward=0.0    "multi-file python module imports"
                                                 -> 5 citations from python-patterns
Step 2: query_cluster    cost=1   reward=0.0    Browse "general_tests_user_patterns"
                                                 -> 630 related skill nodes
Step 3: interrogate      cost=1   reward=0.0    5 Socratic questions generated
                                                 citing real skill corpus nodes
Step 4: submit           cost=1   reward=0.978  3 files, confidence=0.9
                                                 sandbox: ruff clean, mypy clean,
                                                 pytest pass, imports resolved
                                                 grounding: 5/5 symbols verified
Step 5: get_audit        cost=0   reward=0.0    Full audit: 4 entries, total
                                                 reward=0.978, 10 cited skills
Step 6: state            cost=0   ---           done=True, score=0.978, budget=6
```

**Budget efficiency:** 4/10 budget used, 6 remaining. The agent completed a hard task (target 0.70) with a 0.978 reward using only 40% of its budget.

---

## Discovery Tools: Exploring the Corpus

CodeForge ships with 2,648 skill nodes organized into 1,487 clusters across 19 domains.

### List Clusters (free)

```
Top clusters by size:
  general_tests_user_patterns     630 nodes
  general_memory_github_agent      29 nodes
  general_within_hours_carrier     28 nodes
  general_after_before_deploy      24 nodes
  general_diagram_diagrams_guides  21 nodes
```

### List Tags (free)

```
Available domains:
  domain:api         domain:backend     domain:data
  domain:devops      domain:frontend    domain:general
  domain:go          domain:javascript  domain:kotlin
  domain:mcp         domain:mobile      domain:python
  domain:ruby        domain:rust        domain:security
  domain:swift       domain:testing     domain:typescript
  phase:review
```

### Corpus Stats (free)

```
Node count:    2,648
Vocab size:    11,432
Avg doc len:   55.14 tokens
Cluster count: 1,487
```

---

## MCP Tool Reference

CodeForge exposes 10 MCP tools:

| Tool | Cost | Returns |
|------|------|---------|
| `codeforge_reset` | -- | New episode with task brief, budget, initial files |
| `codeforge_query_kb` | 1 | BM25 search results with skill name, section, score |
| `codeforge_query_cluster` | 1 | Cluster members by topic label |
| `codeforge_interrogate` | 1 | 5 Socratic questions citing real corpus nodes |
| `codeforge_run_ralph` | N | Autonomous synthesize-score-keep loop (N iterations) |
| `codeforge_submit` | 1 | Reward from sandbox + grounder + Brier grading |
| `codeforge_get_audit` | 0 | Full episode audit trail |
| `codeforge_state` | 0 | Current episode state (read-only) |
| `codeforge_list_clusters` | 0 | All cluster labels + node counts |
| `codeforge_list_tags` | 0 | All available tags in corpus |

Plus 4 MCP resources (free):
- `codeforge://corpus/stats` — node count, vocab size, cluster count
- `codeforge://corpus/node/{id}` — full content of a specific skill node
- `codeforge://tasks` — task definitions with briefs, budgets, targets
- `codeforge://audit/{episode_id}` — serialized audit ledger

Plus 2 MCP prompts:
- `codeforge_system` — rules and constraints (no exploit details leaked)
- `codeforge_task_brief` — dynamic task brief after reset

---

## Running the Demos

Both demo scripts are included in the repository:

```bash
# Agent WITH CodeForge MCP
PYTHONPATH=CODEFORGE:$PYTHONPATH python3 demo_agent_with_mcp.py

# Agent WITHOUT CodeForge MCP
PYTHONPATH=CODEFORGE:$PYTHONPATH python3 demo_without_mcp.py
```

Or use the MCP server directly in Python:

```python
from pathlib import Path
from codeforge.mcp_server import CodeForgeMCPServer

server = CodeForgeMCPServer(
    corpus_path=Path("CODEFORGE/codeforge/kb/skills_corpus.jsonl")
)

# Start episode
r = server.handle_tool("codeforge_reset", {"task_level": "easy"})
session_id = r["session_id"]

# Research
r = server.handle_tool("codeforge_query_kb", {
    "session_id": session_id,
    "claim": "python greeting function"
})

# Submit
r = server.handle_tool("codeforge_submit", {
    "session_id": session_id,
    "files": {"main.py": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"},
    "confidence": 0.9
})
print(f"Reward: {r['observation']['last_reward']}")  # 0.99
```
