"""End-to-end test: complete episode using all 6 action types."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeforge.environment import CodeForgeEnvironment
from codeforge.models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "test_corpus.jsonl"
    nodes = [
        {
            "id": f"test_{i:03d}",
            "skill_name": f"skill-{i}",
            "section_path": ["Testing", f"Section{i}"],
            "section_body": (
                "Use python functions for greeting. "
                "pytest fixtures help with testing. "
                "Type hints with str and return types."
            ),
            "tags": ["domain:testing"],
            "source_path": f"/test/{i}/SKILL.md",
        }
        for i in range(5)
    ]
    corpus.write_text(
        "\n".join(json.dumps(n) for n in nodes), encoding="utf-8"
    )
    return corpus


# ---------------------------------------------------------------------------
# Full episode tests
# ---------------------------------------------------------------------------


class TestFullEpisodeEasy:
    """Test a full easy episode: query_kb -> interrogate -> submit -> get_audit."""

    def test_full_episode(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        obs = env.reset(task_level="easy")
        assert obs.budget_remaining == 4
        assert not obs.is_done
        assert isinstance(obs, CodeForgeObservation)

        # Action 1: query_kb (cost 1)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB,
                claim="python greeting function",
            )
        )
        assert obs.budget_remaining == 3
        assert len(obs.last_citations) > 0
        assert obs.last_reward == 0.0

        # Action 2: interrogate (cost 1)
        obs = env.step(
            CodeForgeAction(action_type=CodeForgeActionType.INTERROGATE)
        )
        assert obs.budget_remaining == 2
        assert len(obs.last_interrogation_questions) > 0
        assert obs.last_reward == 0.0

        # Action 3: submit with real code (cost 1)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                files={
                    "main.py": (
                        'from __future__ import annotations\n\n\n'
                        'def greet(name: str) -> str:\n'
                        '    return f"Hello, {name}!"\n'
                    ),
                },
                confidence=0.85,
            )
        )
        assert obs.budget_remaining == 1
        assert obs.last_reward > 0

        # Action 4: get_audit (cost 0)
        obs = env.step(
            CodeForgeAction(action_type=CodeForgeActionType.GET_AUDIT)
        )
        assert obs.budget_remaining == 1  # get_audit costs 0
        assert obs.cumulative_audit_summary  # non-empty


class TestFullEpisodeWithQueryCluster:
    """Test query_cluster action in an episode flow."""

    def test_query_cluster_then_submit(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        obs = env.reset(task_level="easy")

        # query_cluster with an unknown label -> empty result
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_CLUSTER,
                cluster_label="unknown_cluster",
            )
        )
        assert obs.budget_remaining == 3
        assert obs.last_cluster_hits == ()

        # submit
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                files={
                    "main.py": (
                        'from __future__ import annotations\n\n\n'
                        'def greet(name: str) -> str:\n'
                        '    return f"Hello, {name}!"\n'
                    ),
                },
                confidence=0.8,
            )
        )
        assert obs.last_reward > 0


class TestFullEpisodeWithRalph:
    """Test run_ralph action with budget accounting."""

    def test_ralph_budget_accounting(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        obs = env.reset(task_level="hard")
        assert obs.budget_remaining == 10

        # run_ralph costs max_iters
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.RUN_RALPH,
                max_iters=3,
            )
        )
        assert obs.budget_remaining == 7  # 10 - 3
        assert obs.last_ralph_run_id is not None

    def test_ralph_then_submit(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        env.reset(task_level="hard")

        # ralph first
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.RUN_RALPH,
                max_iters=2,
            )
        )
        assert obs.budget_remaining == 8

        # then submit
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                files={
                    "main.py": (
                        'from __future__ import annotations\n\n'
                        'from core import greet\n\n\n'
                        'if __name__ == "__main__":\n'
                        '    print(greet("World"))\n'
                    ),
                    "core.py": (
                        'from __future__ import annotations\n\n\n'
                        'def greet(name: str) -> str:\n'
                        '    return f"Hello, {name}!"\n'
                    ),
                    "test_core.py": (
                        'from __future__ import annotations\n\n'
                        'from core import greet\n\n\n'
                        'def test_greet() -> None:\n'
                        '    assert greet("Alice") == "Hello, Alice!"\n'
                    ),
                },
                confidence=0.7,
            )
        )
        assert obs.budget_remaining == 7
        assert obs.last_reward > 0


class TestBudgetExhaustionE2E:
    """Test that budget exhaustion ends the episode."""

    def test_budget_runs_out(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        obs = env.reset(task_level="easy")  # budget=4

        for i in range(4):
            obs = env.step(
                CodeForgeAction(
                    action_type=CodeForgeActionType.QUERY_KB,
                    claim=f"query {i}",
                )
            )

        assert obs.is_done is True
        assert obs.budget_remaining == 0

    def test_step_after_done_returns_done(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        env.reset(task_level="easy")

        # Exhaust budget
        for _ in range(4):
            env.step(
                CodeForgeAction(
                    action_type=CodeForgeActionType.QUERY_KB, claim="x"
                )
            )

        # Step after done
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB, claim="y"
            )
        )
        assert obs.is_done is True


class TestAllSixActionsInOneEpisode:
    """Use all 6 action types in a single hard episode."""

    def test_all_six_actions(self, tmp_corpus: Path) -> None:
        env = CodeForgeEnvironment(corpus_path=tmp_corpus)
        obs = env.reset(task_level="hard")  # budget=10

        # 1. query_kb (cost 1, remaining 9)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB,
                claim="python function typing",
            )
        )
        assert obs.budget_remaining == 9
        assert len(obs.last_citations) > 0

        # 2. query_cluster (cost 1, remaining 8)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_CLUSTER,
                cluster_label="some_cluster",
            )
        )
        assert obs.budget_remaining == 8

        # 3. interrogate (cost 1, remaining 7)
        obs = env.step(
            CodeForgeAction(action_type=CodeForgeActionType.INTERROGATE)
        )
        assert obs.budget_remaining == 7
        assert len(obs.last_interrogation_questions) > 0

        # 4. run_ralph (cost 2, remaining 5)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.RUN_RALPH,
                max_iters=2,
            )
        )
        assert obs.budget_remaining == 5
        assert obs.last_ralph_run_id is not None

        # 5. submit (cost 1, remaining 4)
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                files={
                    "main.py": (
                        'from __future__ import annotations\n\n'
                        'from core import greet\n\n\n'
                        'if __name__ == "__main__":\n'
                        '    print(greet("World"))\n'
                    ),
                    "core.py": (
                        'from __future__ import annotations\n\n\n'
                        'def greet(name: str) -> str:\n'
                        '    return f"Hello, {name}!"\n'
                    ),
                    "test_core.py": (
                        'from __future__ import annotations\n\n'
                        'from core import greet\n\n\n'
                        'def test_greet() -> None:\n'
                        '    assert greet("Alice") == "Hello, Alice!"\n'
                    ),
                },
                confidence=0.75,
            )
        )
        assert obs.budget_remaining == 4
        assert obs.last_reward > 0

        # 6. get_audit (cost 0, remaining 4)
        obs = env.step(
            CodeForgeAction(action_type=CodeForgeActionType.GET_AUDIT)
        )
        assert obs.budget_remaining == 4
        assert obs.cumulative_audit_summary
        # Episode may be done if submit hit the target score (0.70 for hard)
        # The important thing: all 6 actions executed successfully
