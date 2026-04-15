# CodeForge OpenEnv Implementation Plan

> Use superpowers:subagent-driven-development.

**Goal:** Rebuild GroundLoop as an OpenEnv-compliant RL environment (`CodeForgeEnvironment`) that reuses our shipped Python primitives (`python_sandbox`, `kb_indexer`, `lib_grounder`, `ralph_orchestrator`) as the substrate.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-codeforge-env-design.md`.

---

## Task 1: Purge mcp-shell (dead code)

**Files to remove:**
- `groundloop/mcp_shell/` (entire dir)
- `tests/groundloop/mcp_shell/` (entire dir)

Why: spec §2 deletes this; it was the old framing. Keep every other `groundloop/*` package.

- [ ] `rm -rf groundloop/mcp_shell tests/groundloop/mcp_shell`
- [ ] Run `python3 -m pytest tests/groundloop/ -q` — expect the remaining test suites (scraper, kb_indexer, python_sandbox, ralph_orchestrator, lib_grounder, interrogator, audit_reporter) all still pass, and total test count drops by the mcp-shell count.
- [ ] Run `ruff check groundloop/` and `mypy --strict groundloop/` — expect clean.
- [ ] Commit: `chore: remove mcp_shell (superseded by OpenEnv env pivot)`

---

## Task 2: Add `CodeForgeAction` + `CodeForgeObservation` to `models.py`

**File:** `/home/krrish/Desktop/Project/openenv-DGXAI/models.py` (APPEND, do not delete round-1 types).

- [ ] Write failing test `tests/test_models_codeforge.py`:
```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation


def test_query_action_defaults():
    a = CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="hi")
    assert a.top_k == 5
    assert a.required_tags == ()


def test_submit_requires_files():
    # files field is optional in schema but semantically required at runtime;
    # the Pydantic shape itself allows None (runtime check happens in env).
    a = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files={"main.py": "x = 1"})
    assert a.files == {"main.py": "x = 1"}


def test_action_type_enum():
    with pytest.raises(ValidationError):
        CodeForgeAction(action_type="invalid")  # type: ignore[arg-type]


def test_observation_required_fields():
    obs = CodeForgeObservation(
        episode_id="e1", task_id="easy/greet_single_file", task_level="easy",
        task_brief="Build greet", initial_files={"main.py": ""},
        current_files={"main.py": ""}, budget_remaining=4, previous_score=0.0,
        last_citations=(), last_grounding=None, is_done=False, last_reward=0.0,
    )
    assert obs.episode_id == "e1"
```

- [ ] Append to `models.py`:
```python
from enum import Enum as _Enum


class CodeForgeActionType(str, _Enum):
    QUERY_KB = "query_kb"
    SUBMIT = "submit"


class CodeForgeAction(Action):
    action_type: CodeForgeActionType
    claim: Optional[str] = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    files: Optional[dict[str, str]] = None


class CodeForgeObservation(Observation):
    episode_id: str
    task_id: str
    task_level: str
    task_brief: str
    initial_files: dict[str, str]
    current_files: dict[str, str]
    budget_remaining: int
    previous_score: float
    last_citations: tuple[dict, ...] = ()
    last_grounding: Optional[dict] = None
    is_done: bool = False
    last_reward: float = 0.0
```

- [ ] Run: PASS (4 tests).
- [ ] Commit: `feat(codeforge): add CodeForgeAction + CodeForgeObservation models`

---

## Task 3: `groundloop_env/tasks.py` — the 3 tasks

**File:** `groundloop_env/__init__.py` (empty), `groundloop_env/tasks.py`, `tests/groundloop_env/__init__.py`, `tests/groundloop_env/conftest.py`, `tests/groundloop_env/test_tasks.py`.

- [ ] Create dirs + empty `__init__.py` files.
- [ ] `tests/groundloop_env/test_tasks.py`:
```python
from groundloop_env.tasks import TASKS, get_task


def test_three_tasks_present():
    levels = {t.task_level for t in TASKS}
    assert levels == {"easy", "medium", "hard"}


def test_get_task_by_level():
    t = get_task("easy")
    assert t.task_level == "easy"
    assert t.brief
    assert t.initial_files
    assert 0.0 < t.target_score <= 1.0


def test_get_task_invalid_level_raises():
    import pytest
    with pytest.raises(ValueError):
        get_task("trivial")
```

- [ ] `groundloop_env/tasks.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    task_id: str
    task_level: str              # "easy" | "medium" | "hard"
    brief: str
    initial_files: dict[str, str]
    target_score: float
    max_budget: int


TASKS: tuple[Task, ...] = (
    Task(
        task_id="greet_single_file",
        task_level="easy",
        brief=(
            "Implement `greet(name)` in `main.py` so that `greet(\"Alice\")` returns "
            "`\"Hello, Alice!\"`. Use type hints. Keep the module under 15 lines."
        ),
        initial_files={"main.py": "def greet(name):\n    pass\n"},
        target_score=0.90,
        max_budget=4,
    ),
    Task(
        task_id="greet_with_tests",
        task_level="medium",
        brief=(
            "Extend `main.py` so that `greet(None)` raises `ValueError`, "
            "and add a `test_main.py` with pytest assertions. Keep `ruff` and "
            "`mypy --strict` clean."
        ),
        initial_files={
            "main.py": (
                "from __future__ import annotations\n\n\n"
                "def greet(name: str) -> str:\n"
                "    return f\"Hello, {name}!\"\n"
            ),
            "test_main.py": "",
        },
        target_score=0.80,
        max_budget=6,
    ),
    Task(
        task_id="multi_file_module",
        task_level="hard",
        brief=(
            "Split into three files: `main.py` (entry), `core.py` (the greet "
            "function), `test_core.py` (tests). Every function must be type-hinted. "
            "All tests pass. `mypy --strict` clean."
        ),
        initial_files={
            "main.py": (
                "from __future__ import annotations\n\nfrom core import greet\n\n\n"
                "if __name__ == \"__main__\":\n"
                "    print(greet(\"World\"))\n"
            ),
            "core.py": "",
            "test_core.py": "",
        },
        target_score=0.70,
        max_budget=10,
    ),
)


def get_task(task_level: str) -> Task:
    for t in TASKS:
        if t.task_level == task_level:
            return t
    msg = f"unknown task_level: {task_level!r} (expected easy|medium|hard)"
    raise ValueError(msg)
```

- [ ] `tests/groundloop_env/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def tiny_corpus_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "groundloop" / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
```

- [ ] Run PASS.
- [ ] Commit: `feat(codeforge): 3 tasks (easy/medium/hard) with briefs + target scores`

---

## Task 4: `groundloop_env/grader.py` — score → reward

- [ ] `tests/groundloop_env/test_grader.py`:
```python
from groundloop_env.grader import compute_reward


def test_reward_in_range_monotonic_in_both_inputs():
    # Higher sandbox score, same grounding -> higher reward
    r_low = compute_reward(sandbox_score=0.2, groundedness=0.5)
    r_hi  = compute_reward(sandbox_score=0.9, groundedness=0.5)
    assert 0.0 <= r_low <= r_hi <= 1.0

    # Higher grounding, same sandbox -> higher reward
    r_ungrounded = compute_reward(sandbox_score=0.5, groundedness=0.0)
    r_grounded   = compute_reward(sandbox_score=0.5, groundedness=1.0)
    assert r_ungrounded < r_grounded


def test_reward_clamped_to_zero_one():
    assert compute_reward(sandbox_score=-1.0, groundedness=2.0) <= 1.0
    assert compute_reward(sandbox_score=-5.0, groundedness=-5.0) >= 0.0


def test_reward_deterministic():
    a = compute_reward(sandbox_score=0.7, groundedness=0.8)
    b = compute_reward(sandbox_score=0.7, groundedness=0.8)
    assert a == b
```

- [ ] `groundloop_env/grader.py`:
```python
from __future__ import annotations

_SANDBOX_WEIGHT = 0.6
_GROUNDING_WEIGHT = 0.4


def compute_reward(*, sandbox_score: float, groundedness: float) -> float:
    raw = _SANDBOX_WEIGHT * sandbox_score + _GROUNDING_WEIGHT * groundedness
    clamped = max(0.0, min(1.0, raw))
    return round(clamped, 3)
```

- [ ] Run PASS (3 tests).
- [ ] Commit: `feat(codeforge): deterministic reward = 0.6*sandbox + 0.4*grounding`

---

## Task 5: `groundloop_env/observation_builder.py`

- [ ] `tests/groundloop_env/test_observation_builder.py`:
```python
from groundloop_env.observation_builder import build_observation
from groundloop_env.tasks import get_task


def test_build_minimal_observation():
    t = get_task("easy")
    obs = build_observation(
        episode_id="e1", task=t, current_files=t.initial_files,
        budget_remaining=4, previous_score=0.0,
        last_citations=(), last_grounding=None, is_done=False, last_reward=0.0,
    )
    assert obs.episode_id == "e1"
    assert obs.task_level == "easy"
    assert obs.initial_files == t.initial_files
    assert obs.is_done is False
```

- [ ] `groundloop_env/observation_builder.py`:
```python
from __future__ import annotations

from models import CodeForgeObservation
from groundloop_env.tasks import Task


def build_observation(
    *,
    episode_id: str,
    task: Task,
    current_files: dict[str, str],
    budget_remaining: int,
    previous_score: float,
    last_citations: tuple[dict, ...] = (),
    last_grounding: dict | None = None,
    is_done: bool = False,
    last_reward: float = 0.0,
) -> CodeForgeObservation:
    return CodeForgeObservation(
        episode_id=episode_id,
        task_id=task.task_id,
        task_level=task.task_level,
        task_brief=task.brief,
        initial_files=dict(task.initial_files),
        current_files=dict(current_files),
        budget_remaining=budget_remaining,
        previous_score=previous_score,
        last_citations=last_citations,
        last_grounding=last_grounding,
        is_done=is_done,
        last_reward=last_reward,
    )
```

- [ ] Run PASS.
- [ ] Commit: `feat(codeforge): observation builder`

---

## Task 6: `groundloop_env/environment.py` — the env

- [ ] `tests/groundloop_env/test_environment.py`:
```python
from pathlib import Path

import pytest

from models import CodeForgeAction, CodeForgeActionType
from groundloop_env.environment import CodeForgeEnvironment


@pytest.fixture
def env(tiny_corpus_path: Path) -> CodeForgeEnvironment:
    return CodeForgeEnvironment(corpus_path=tiny_corpus_path)


def test_reset_returns_observation(env: CodeForgeEnvironment):
    obs = env.reset(task_level="easy")
    assert obs.task_level == "easy"
    assert obs.budget_remaining > 0
    assert obs.current_files == obs.initial_files
    assert obs.is_done is False


def test_query_kb_decrements_budget(env: CodeForgeEnvironment):
    obs = env.reset(task_level="easy")
    before = obs.budget_remaining
    action = CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="greet")
    obs2 = env.step(action)
    assert obs2.budget_remaining == before - 1
    # citations populated
    assert len(obs2.last_citations) > 0


def test_submit_returns_reward(env: CodeForgeEnvironment):
    env.reset(task_level="easy")
    good = {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    }
    action = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files=good)
    obs = env.step(action)
    # Must be done because target_score is met or budget exhausted after submit
    assert obs.last_reward >= 0.0
    assert obs.last_reward <= 1.0


def test_submit_missing_files(env: CodeForgeEnvironment):
    env.reset(task_level="easy")
    action = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files=None)
    obs = env.step(action)
    assert obs.last_reward == 0.0


def test_budget_exhaustion_marks_done(env: CodeForgeEnvironment):
    env.reset(task_level="easy")
    # Issue QUERY_KB enough times to exhaust budget
    for _ in range(10):
        obs = env.step(CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="x"))
        if obs.is_done:
            break
    assert obs.is_done is True
```

- [ ] `groundloop_env/environment.py`:
```python
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment

from models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation
from groundloop.kb_indexer.index import SkillsIndex
from groundloop.lib_grounder.grounder import ground
from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop_env.grader import compute_reward
from groundloop_env.observation_builder import build_observation
from groundloop_env.tasks import Task, get_task

_log = logging.getLogger(__name__)
_DEFAULT_CORPUS = Path("groundloop/kb/skills_corpus.jsonl")


class CodeForgeEnvironment(Environment):
    def __init__(self, *, corpus_path: Path | None = None) -> None:
        super().__init__()
        self._corpus_path = corpus_path or _DEFAULT_CORPUS
        self._index: SkillsIndex | None = None
        self._task: Task | None = None
        self._episode_id: str = ""
        self._budget_remaining: int = 0
        self._current_files: dict[str, str] = {}
        self._previous_score: float = 0.0
        self._last_citations: tuple[dict, ...] = ()
        self._last_grounding: dict | None = None
        self._is_done: bool = False
        self._last_reward: float = 0.0

    def _ensure_index(self) -> SkillsIndex:
        if self._index is None:
            if not self._corpus_path.is_file():
                msg = (
                    f"corpus not found: {self._corpus_path}. "
                    f"Run `python3 -m groundloop.skills_scraper` first."
                )
                raise FileNotFoundError(msg)
            idx = SkillsIndex(corpus_path=self._corpus_path)
            idx.build()
            self._index = idx
        return self._index

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> CodeForgeObservation:
        task_level = kwargs.get("task_level", "easy")
        task = get_task(task_level)
        self._task = task
        self._episode_id = episode_id or uuid.uuid4().hex[:12]
        self._budget_remaining = task.max_budget
        self._current_files = dict(task.initial_files)
        self._previous_score = 0.0
        self._last_citations = ()
        self._last_grounding = None
        self._is_done = False
        self._last_reward = 0.0
        _log.info("reset id=%s task=%s budget=%s", self._episode_id, task.task_id, task.max_budget)
        return self._build_obs()

    def step(self, action: CodeForgeAction, timeout_s: float | None = None, **kwargs: Any) -> CodeForgeObservation:
        if self._is_done or self._task is None:
            return self._build_obs()

        self._budget_remaining -= 1

        if action.action_type == CodeForgeActionType.QUERY_KB:
            self._handle_query(action)
            self._last_reward = 0.0
        elif action.action_type == CodeForgeActionType.SUBMIT:
            self._handle_submit(action)
        # No else — Pydantic guards

        if self._budget_remaining <= 0:
            self._is_done = True
        return self._build_obs()

    def state(self) -> CodeForgeObservation:
        return self._build_obs()

    def _handle_query(self, action: CodeForgeAction) -> None:
        try:
            idx = self._ensure_index()
        except FileNotFoundError as e:
            _log.warning("query: no corpus: %s", e)
            self._last_citations = ()
            return
        tags = set(action.required_tags) if action.required_tags else None
        results = idx.search(action.claim or "", top_k=action.top_k, required_tags=tags)
        self._last_citations = tuple(
            {
                "node_id": r.node_id,
                "skill_name": r.skill_name,
                "section_path": list(r.section_path),
                "section_body": r.section_body,
                "score": r.score,
                "rank": r.rank,
            }
            for r in results
        )

    def _handle_submit(self, action: CodeForgeAction) -> None:
        if action.files is None:
            self._last_reward = 0.0
            return
        self._current_files = dict(action.files)
        try:
            sandbox_result = run_sandbox(
                files=dict(action.files),
                tools=("ruff", "imports", "mypy", "pytest"),
                timeout_per_tool=30.0,
            )
            sandbox_score = sandbox_result.composite_score
        except Exception as e:  # noqa: BLE001 - env must never crash the server
            _log.exception("sandbox error: %s", e)
            sandbox_score = 0.0

        concatenated = "\n".join(action.files.values())
        grounding_report = ground(concatenated)
        self._last_grounding = grounding_report.model_dump()
        reward = compute_reward(
            sandbox_score=sandbox_score,
            groundedness=grounding_report.groundedness,
        )
        self._last_reward = reward
        self._previous_score = reward
        assert self._task is not None  # noqa: S101 - protected by outer guard
        if reward >= self._task.target_score:
            self._is_done = True

    def _build_obs(self) -> CodeForgeObservation:
        assert self._task is not None  # noqa: S101
        return build_observation(
            episode_id=self._episode_id,
            task=self._task,
            current_files=self._current_files,
            budget_remaining=self._budget_remaining,
            previous_score=self._previous_score,
            last_citations=self._last_citations,
            last_grounding=self._last_grounding,
            is_done=self._is_done,
            last_reward=self._last_reward,
        )
```

- [ ] Run test — expect PASS. Tests may be slow on first run due to index build (use `tiny_corpus_path` fixture to keep them fast).
- [ ] Commit: `feat(codeforge): CodeForgeEnvironment with reset/step/state`

---

## Task 7: `groundloop_env/app.py` — FastAPI binding

- [ ] `tests/groundloop_env/test_app.py`:
```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tiny_corpus_path: Path) -> TestClient:
    # Force env to use tiny corpus
    monkeypatch.setenv("GROUNDLOOP_CORPUS_PATH", str(tiny_corpus_path))
    # Reimport app module to pick up the env var
    import importlib
    import groundloop_env.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "code-forge"


def test_tasks_endpoint(client: TestClient):
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    levels = {t["difficulty"] for t in data["tasks"]}
    assert levels == {"easy", "medium", "hard"}


def test_reset_endpoint(client: TestClient):
    r = client.post("/reset", json={"task_level": "easy"})
    assert r.status_code == 200
    obs = r.json()
    assert obs["task_level"] == "easy"
```

- [ ] `groundloop_env/app.py`:
```python
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Response
from openenv.core.env_server.http_server import create_app

from models import CodeForgeAction, CodeForgeObservation
from groundloop_env.environment import CodeForgeEnvironment
from groundloop_env.tasks import TASKS

_log = logging.getLogger(__name__)

_corpus_path_str = os.environ.get("GROUNDLOOP_CORPUS_PATH")
_corpus_path = Path(_corpus_path_str) if _corpus_path_str else None

_env_instance = CodeForgeEnvironment(corpus_path=_corpus_path)
app: FastAPI = create_app(lambda: _env_instance, CodeForgeAction, CodeForgeObservation)


@app.get("/", summary="Health check")
def root() -> dict:
    return {"name": "code-forge", "status": "ok", "docs": "/docs"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/tasks", summary="List tasks + action schema")
def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "id": t.task_id,
                "difficulty": t.task_level,
                "brief": t.brief,
                "target_score": t.target_score,
                "max_budget": t.max_budget,
            }
            for t in TASKS
        ],
        "action_schema": {
            "action_type": {"type": "string", "enum": ["query_kb", "submit"]},
            "query_kb_fields": {"claim": "string", "top_k": "int", "required_tags": "list[str]"},
            "submit_fields": {"files": "dict[str, str] — path → content"},
        },
    }
```

- [ ] Run PASS (3 tests).
- [ ] Commit: `feat(codeforge): FastAPI app + /tasks endpoint`

---

## Task 8: Update `openenv.yaml` + `Dockerfile`

- [ ] Replace `openenv.yaml` contents with:
```yaml
name: code-forge
version: 0.1.0
description: >
  RL environment for training agents to iteratively synthesize Python
  codebases from natural-language briefs. Agents can query a knowledge
  base of coding-practice skills and submit candidate files; rewards
  combine a programmatic quality signal (ruff/mypy/pytest/imports) with
  AST-level symbol grounding.
author: krrishchoudhary109
tasks:
  - id: greet_single_file
    description: Single-file greet function with type hints.
    difficulty: easy
  - id: greet_with_tests
    description: Multi-file greet + pytest + error handling.
    difficulty: medium
  - id: multi_file_module
    description: Three-file module (entry + core + tests), mypy strict.
    difficulty: hard
max_budget: 10
reward_range: [0.0, 1.0]
hf_space: krrishchoudhary109/code-forge
python_requires: ">=3.11"
```

- [ ] Replace `Dockerfile` contents with:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the Layer B KB corpus at image build time so the server boots fast.
RUN python3 -m groundloop.skills_scraper || echo "scraper build skipped (no user skills in image)"

EXPOSE 7860
CMD ["uvicorn", "groundloop_env.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] Smoke test: `uvicorn groundloop_env.app:app --host 0.0.0.0 --port 17860 &` in a subshell, `curl -s http://localhost:17860/tasks`, then `kill %1`. Expect 3 tasks returned.
- [ ] Commit: `chore: update openenv.yaml + Dockerfile for code-forge`

---

## Task 9: Replace `inference.py` (baseline agent)

- [ ] `tests/test_inference.py`:
```python
def test_inference_module_imports():
    import inference
    assert hasattr(inference, "main")
```

- [ ] Replace `inference.py`:
```python
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx


_LOG = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MAX_ITERS_PER_TASK = 3
TIMEOUT_S = 120.0

_STUB_SOLUTIONS: dict[str, dict[str, str]] = {
    "easy": {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    },
    "medium": {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str | None) -> str:\n"
            "    if name is None:\n"
            "        msg = \"name must not be None\"\n"
            "        raise ValueError(msg)\n"
            "    return f\"Hello, {name}!\"\n"
        ),
        "test_main.py": (
            "from __future__ import annotations\n\n"
            "import pytest\n\n"
            "from main import greet\n\n\n"
            "def test_greet_hello() -> None:\n"
            "    assert greet(\"Alice\") == \"Hello, Alice!\"\n\n\n"
            "def test_greet_none_raises() -> None:\n"
            "    with pytest.raises(ValueError):\n"
            "        greet(None)\n"
        ),
    },
    "hard": {
        "main.py": (
            "from __future__ import annotations\n\nfrom core import greet\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    print(greet(\"World\"))\n"
        ),
        "core.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
        "test_core.py": (
            "from __future__ import annotations\n\n"
            "from core import greet\n\n\n"
            "def test_greet() -> None:\n"
            "    assert greet(\"World\") == \"Hello, World!\"\n"
        ),
    },
}


def _run_task(client: httpx.Client, level: str) -> dict[str, float | str]:
    reset_resp = client.post(f"{API_BASE_URL}/reset", json={"task_level": level})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    # One QUERY_KB for context
    query = {"action_type": "query_kb", "claim": obs.get("task_brief", level), "top_k": 3}
    q_resp = client.post(f"{API_BASE_URL}/step", json=query)
    q_resp.raise_for_status()

    # SUBMIT the stub solution
    submit = {"action_type": "submit", "files": _STUB_SOLUTIONS[level]}
    s_resp = client.post(f"{API_BASE_URL}/step", json=submit)
    s_resp.raise_for_status()
    final = s_resp.json()

    return {
        "task_level": level,
        "reward": final.get("last_reward", 0.0),
        "done": final.get("is_done", False),
        "citations_seen": len(final.get("last_citations", ())),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    results: list[dict] = []
    t0 = time.monotonic()
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for level in ("easy", "medium", "hard"):
            result = _run_task(client, level)
            results.append(result)
            _LOG.info("%s → reward=%.3f done=%s", level, result["reward"], result["done"])
    _LOG.info("total_time=%.2fs", time.monotonic() - t0)
    print(json.dumps({"baseline": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Run test PASS.
- [ ] Commit: `feat(codeforge): baseline inference.py (HTTP client, stub synthesizer)`

---

## Task 10: `groundloop_env/__init__.py` + e2e + README

- [ ] Populate `groundloop_env/__init__.py`:
```python
from __future__ import annotations

from groundloop_env.environment import CodeForgeEnvironment
from groundloop_env.grader import compute_reward
from groundloop_env.observation_builder import build_observation
from groundloop_env.tasks import TASKS, Task, get_task

__all__ = [
    "CodeForgeEnvironment",
    "TASKS",
    "Task",
    "build_observation",
    "compute_reward",
    "get_task",
]
```

- [ ] `tests/groundloop_env/test_e2e.py`:
```python
from pathlib import Path

from models import CodeForgeAction, CodeForgeActionType
from groundloop_env.environment import CodeForgeEnvironment


def test_e2e_easy_task_full_episode(tiny_corpus_path: Path):
    env = CodeForgeEnvironment(corpus_path=tiny_corpus_path)
    obs = env.reset(task_level="easy")
    assert obs.task_level == "easy"
    # 1 query
    env.step(CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="greet"))
    # 1 submit with a correct stub
    good = {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    }
    final = env.step(CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files=good))
    assert 0.0 <= final.last_reward <= 1.0
    # state() returns same observation
    assert env.state().episode_id == obs.episode_id
```

- [ ] Append README section `## CodeForge OpenEnv (Round 2)` with:
  - Env name + what it does.
  - How to run: `uvicorn groundloop_env.app:app --host 0.0.0.0 --port 7860`.
  - How to run baseline: `python3 inference.py`.
  - Docker: `docker build -t code-forge . && docker run -p 7860:7860 code-forge`.
  - Tasks table.
  - Reward formula: `0.6 * sandbox_composite_score + 0.4 * grounding_score`.
  - Note that round 1 env still lives under `server/`.

- [ ] Final verification:
```
python3 -m pytest tests/ -v --cov=groundloop_env --cov-report=term
ruff check groundloop_env/
mypy --strict groundloop_env/
```
Expect all tests pass, coverage ≥ 85% on `groundloop_env`.

- [ ] Live smoke with HTTP:
```
python3 -m groundloop.skills_scraper || true  # ensure corpus
uvicorn groundloop_env.app:app --host 127.0.0.1 --port 17861 &
sleep 2
curl -s http://localhost:17861/tasks | head -50
curl -s -X POST http://localhost:17861/reset -H 'content-type: application/json' -d '{"task_level": "easy"}' | head -20
kill %1
```

- [ ] Commit: `feat(codeforge): e2e test, public API, README, baseline smoke`

---

## Self-Review

- ✅ Every spec §3 file has a task.
- ✅ Every spec §11 acceptance criterion covered by a test or smoke step.
- ✅ `mcp_shell` deleted in Task 1; not referenced again.
- ✅ Round-1 `server/` untouched.
- ✅ Reuses shipped `groundloop/*` utilities; no new deps.
- ✅ No placeholders; all code blocks complete.
- ✅ Type consistency: `CodeForgeAction`/`CodeForgeObservation` shape matches across models.py → environment.py → app.py → inference.py → tests.
