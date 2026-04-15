from __future__ import annotations

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
    assert obs.last_reward >= 0.0
    assert obs.last_reward <= 1.0


def test_submit_missing_files(env: CodeForgeEnvironment):
    env.reset(task_level="easy")
    action = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files=None)
    obs = env.step(action)
    assert obs.last_reward == 0.0


def test_budget_exhaustion_marks_done(env: CodeForgeEnvironment):
    env.reset(task_level="easy")
    obs = None
    for _ in range(10):
        obs = env.step(CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="x"))
        if obs.is_done:
            break
    assert obs is not None
    assert obs.is_done is True
