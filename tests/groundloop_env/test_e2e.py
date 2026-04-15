from __future__ import annotations

from pathlib import Path

from models import CodeForgeAction, CodeForgeActionType
from groundloop_env.environment import CodeForgeEnvironment


def test_e2e_easy_task_full_episode(tiny_corpus_path: Path):
    env = CodeForgeEnvironment(corpus_path=tiny_corpus_path)
    obs = env.reset(task_level="easy")
    assert obs.task_level == "easy"
    env.step(CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="greet"))
    good = {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    }
    final = env.step(CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files=good))
    assert 0.0 <= final.last_reward <= 1.0
    assert env.state.episode_id == obs.episode_id
