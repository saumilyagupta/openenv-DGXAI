# CodeForge OpenEnv — Design Spec

**Date:** 2026-04-15
**Sub-project:** the pivot — rebuild GroundLoop as an OpenEnv-compliant RL environment.
**Replaces:** `groundloop/mcp_shell/` (to be deleted).
**Reuses:** `groundloop/skills_scraper/`, `groundloop/kb_indexer/`, `groundloop/python_sandbox/`, `groundloop/lib_grounder/`, `groundloop/ralph_orchestrator/`, `groundloop/audit_reporter/`.
**Coexists with:** `server/` (round-1 EpistemicNav env, untouched for judge reference).

---

## 1. Purpose

An OpenEnv environment where the agent iteratively builds a Python codebase from a natural-language brief. Rewards measure: did the produced code compile, type-check, pass tests, and ground every symbol/import against the running Python environment? Same philosophical thread as Karpathy's Autoresearch (iterate toward a metric) and round-1 EpistemicNav (grounded, calibrated reasoning).

## 2. Scope

**In scope:**

- `groundloop_env/` package with OpenEnv-compliant `CodeForgeEnvironment(Environment)`.
- `models.py` top-level: typed Pydantic v2 `CodeForgeAction`, `CodeForgeObservation`.
- 3 tasks (easy / medium / hard) with programmatic graders.
- `openenv.yaml` describing the env.
- `Dockerfile` that builds a small python:3.11-slim image, binds `0.0.0.0:7860`.
- `inference.py` baseline agent (uses `ralph_orchestrator.run_loop` + OpenAI-compatible client).
- Updated `README.md` with env description, action/observation spaces, task descriptions, baseline scores.
- Deletion of `groundloop/mcp_shell/` + its tests.

**Out of scope:**

- HF Space deploy — out of scope for this sub-project (git push separately).
- Training agents (we're shipping the env, not a policy).

## 3. Architecture

```
groundloop_env/                 # NEW — the OpenEnv env package
  __init__.py
  environment.py                # CodeForgeEnvironment(Environment)
  app.py                        # FastAPI app via openenv.create_app
  tasks.py                      # 3 task definitions (brief, initial_files, target_score)
  grader.py                     # score-to-reward mapping (reuses python_sandbox)
  observation_builder.py        # builds CodeForgeObservation from state
  requirements.txt              # minimal server deps

models.py                       # UPDATED — add CodeForgeAction, CodeForgeObservation
                                # (keep EpistemicAction, EpistemicObservation for round 1)
openenv.yaml                    # UPDATED — new name, tasks, reward_range
Dockerfile                      # UPDATED — new entrypoint: groundloop_env.app
inference.py                    # REPLACED — new baseline using ralph_orchestrator

groundloop/mcp_shell/           # DELETE entirely
tests/groundloop/mcp_shell/     # DELETE entirely
groundloop/                     # kept utilities: skills_scraper, kb_indexer,
                                # python_sandbox, lib_grounder, ralph_orchestrator,
                                # interrogator, audit_reporter

server/                         # UNTOUCHED — round-1 EpistemicNav env
```

## 4. Data model

### 4.1 `CodeForgeAction`

```python
class CodeForgeActionType(str, Enum):
    QUERY_KB = "query_kb"
    SUBMIT = "submit"

class CodeForgeAction(BaseModel, frozen=True):
    action_type: CodeForgeActionType
    # QUERY_KB fields
    claim: str | None = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    # SUBMIT fields
    files: dict[str, str] | None = None  # path -> content
```

### 4.2 `CodeForgeObservation`

```python
class CodeForgeObservation(BaseModel, frozen=True):
    episode_id: str
    task_id: str
    task_level: str                          # "easy"|"medium"|"hard"
    task_brief: str                          # the natural-language goal
    initial_files: dict[str, str]            # the starter code
    current_files: dict[str, str]            # last submission (or initial)
    budget_remaining: int
    previous_score: float                    # last submit's composite_score
    last_citations: tuple[dict, ...]         # kb results from last QUERY_KB
    last_grounding: dict | None              # lib_grounder report on last submit
    is_done: bool
    last_reward: float
```

## 5. Tasks

Each task has `{id, level, brief, initial_files, target_score, max_budget}`.

1. **easy / `greet_single_file`** (budget 4)
   - brief: "Implement `greet(name)` in `main.py` so that `greet("Alice")` returns `"Hello, Alice!"`. Use type hints."
   - initial_files: `main.py = "def greet(name):\n    pass\n"`
   - target_score: 0.90

2. **medium / `greet_with_tests`** (budget 6)
   - brief: "Add a pytest in `test_main.py` asserting greet behavior, plus a guard so `greet(None)` raises `ValueError`. Keep mypy --strict clean."
   - initial_files: `main.py` (from easy solution), `test_main.py = ""`
   - target_score: 0.80

3. **hard / `multi_file_module`** (budget 10)
   - brief: "Split into `main.py` (entry), `core.py` (logic), `test_core.py` (tests). All modules type-hinted, tests pass, no `Any` leak."
   - initial_files: `main.py = "from core import greet\n\n\nif __name__ == \"__main__\":\n    print(greet(\"World\"))\n"`, `core.py = ""`, `test_core.py = ""`
   - target_score: 0.70

## 6. Semantics

### 6.1 `reset(task_level="easy")`

Pick the task by level. Initialize: `budget_remaining = task.max_budget`, `current_files = task.initial_files`, `previous_score = 0.0`, `is_done = False`. Return observation.

### 6.2 `step(action)`

- If `is_done`: return observation unchanged, reward 0.0.
- If `action.action_type == QUERY_KB`:
  - `budget_remaining -= 1`
  - Query `SkillsIndex.search(claim, top_k, required_tags)`.
  - Update `last_citations` on observation.
  - reward = 0.0 (information-gathering is free — same pattern as round 1).
- If `action.action_type == SUBMIT`:
  - `budget_remaining -= 1`
  - Update `current_files = action.files`.
  - Run `python_sandbox.run_sandbox(files=action.files, tools=("ruff", "imports", "mypy", "pytest"))`.
  - Compute `grounding = lib_grounder.ground(concatenated source)`.
  - reward = `grader.compute_reward(sandbox_score, grounding.groundedness, task)` — in `[0.0, 1.0]`.
  - If reward ≥ `task.target_score` OR `budget_remaining <= 0`: `is_done = True`.
- `state()` returns the current observation.

### 6.3 Reward shaping

```
reward = 0.6 * sandbox_composite_score + 0.4 * grounding_score
# round to 3 dp; clamp to [0.0, 1.0]
```

Deterministic: same inputs → same reward.

## 7. OpenEnv compliance

- `Environment` base class from `openenv.core.env_server.interfaces`.
- `FastAPI` app via `openenv.core.env_server.http_server.create_app(lambda: env, Action, Obs)`.
- Mounts at `/step`, `/reset`, `/state`.
- `/health` + `/tasks` custom endpoints (same pattern as round 1).
- `openenv.yaml` with new name (`code-forge`), 3 tasks, reward_range `[0.0, 1.0]`.

## 8. Error handling

| Condition | Action |
|---|---|
| SUBMIT with `files=None` | reward=0.0, info="missing_files" |
| Sandbox raises | reward=0.0, info captures error |
| QUERY_KB with graph not built | lazy-build on first QUERY, using `DEFAULT_CORPUS_PATH` |
| Default corpus missing | env startup fails fast with clear message |
| Budget exhausted | `is_done=True`, final reward kept |

## 9. Baseline `inference.py`

Runs all 3 tasks against the local HTTP server. Uses `EpistemicEnv`-style HTTP client from `client.py` (generic OpenEnv client), or the SDK client if present. Per task:

1. `reset(task_level=...)` → observation.
2. Agent loop: issue 2 QUERY_KB calls with relevant claims from brief → synthesize files via `ralph_orchestrator.StubSynthesizer` (no API key needed) → SUBMIT.
3. Collect final reward, log per-task.
4. Final: print baseline scores table.

Must run in < 20 minutes on 2 vCPU / 8 GB.

## 10. Testing

- `tests/groundloop_env/test_models.py` — CodeForgeAction/Observation validation.
- `tests/groundloop_env/test_tasks.py` — all 3 tasks load, have non-empty briefs.
- `tests/groundloop_env/test_grader.py` — reward monotonic in sandbox + grounding; clamped.
- `tests/groundloop_env/test_environment.py` — reset/step/state lifecycle; budget decrement; done detection; error paths.
- `tests/groundloop_env/test_app.py` — FastAPI TestClient hits `/reset`, `/step`, `/state`, `/tasks`.
- `tests/groundloop_env/test_e2e.py` — full episode (easy task) using StubSynthesizer; asserts reward in [0,1] and `done == True`.

Coverage target: **85%**.

## 11. Acceptance Criteria

1. `openenv.yaml` validates (name, tasks, reward_range).
2. `uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860` starts cleanly.
3. `POST /reset {"task_level": "easy"}` returns a valid `CodeForgeObservation`.
4. `POST /step {"action": {"action_type": "query_kb", "claim": "greet function"}}` returns citations in observation.
5. `POST /step {"action": {"action_type": "submit", "files": {"main.py": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"}}}` returns reward > 0.8 for the easy task.
6. `python inference.py` completes and prints a baseline score per task.
7. `docker build -t code-forge .` succeeds; `docker run -p 7860:7860 code-forge` starts.
8. `ruff check` + `mypy --strict` clean on `groundloop_env/`.
9. Full project test suite passes: `pytest tests/ -v`.
10. `groundloop/mcp_shell/` and `tests/groundloop/mcp_shell/` removed.

## 12. Dependencies

Already installed. Uses `openenv-core==0.2.1`, `fastapi`, `uvicorn`, `pydantic`, `rank_bm25`, `numpy`, `httpx`, `openai`, and all `groundloop/*` internal packages.

## 13. Deliverables

1. `groundloop_env/` package.
2. Updated `models.py`.
3. Updated `openenv.yaml`.
4. Updated `Dockerfile`.
5. Replaced `inference.py`.
6. Updated `README.md`.
7. Test suite.
8. Deleted `groundloop/mcp_shell/` + tests.
