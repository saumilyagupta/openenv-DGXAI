from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from codeforge.models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation
from codeforge.observation import build_observation
from codeforge.shaping import citation_shaping_bonus
from codeforge.tasks import Task, get_task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EASY_TASK = get_task("easy")


def _make_corpus(tmp_path: Path) -> Path:
    """Create a minimal 3-node JSONL corpus for testing."""
    corpus = tmp_path / "test_corpus.jsonl"
    nodes = [
        {
            "id": f"node_{i}",
            "skill_name": f"skill-{i}",
            "section_path": ["root", f"section-{i}"],
            "section_body": f"python typing hints example {i} def greet name str return hello",
            "tags": ["domain:python", "testing"],
            "source_path": f"/skills/skill-{i}/SKILL.md",
        }
        for i in range(3)
    ]
    corpus.write_text(
        "\n".join(json.dumps(n) for n in nodes) + "\n", encoding="utf-8"
    )
    return corpus


# ---------------------------------------------------------------------------
# Observation builder tests
# ---------------------------------------------------------------------------


class TestBuildObservation:
    def test_returns_correct_type(self) -> None:
        obs = build_observation(
            episode_id="ep123",
            task=_EASY_TASK,
            current_files={"main.py": "pass"},
            budget_remaining=3,
            previous_score=0.0,
        )
        assert isinstance(obs, CodeForgeObservation)

    def test_fields_match_input(self) -> None:
        obs = build_observation(
            episode_id="ep123",
            task=_EASY_TASK,
            current_files={"main.py": "pass"},
            budget_remaining=3,
            previous_score=0.5,
            is_done=True,
            last_reward=0.8,
        )
        assert obs.episode_id == "ep123"
        assert obs.task_id == _EASY_TASK.task_id
        assert obs.task_level == _EASY_TASK.task_level
        assert obs.task_brief == _EASY_TASK.brief
        assert obs.current_files == {"main.py": "pass"}
        assert obs.budget_remaining == 3
        assert obs.previous_score == 0.5
        assert obs.is_done is True
        assert obs.last_reward == 0.8
        assert obs.reward == 0.8
        assert obs.done is True

    def test_default_optional_fields(self) -> None:
        obs = build_observation(
            episode_id="ep1",
            task=_EASY_TASK,
            current_files={},
            budget_remaining=4,
            previous_score=0.0,
        )
        assert obs.last_citations == ()
        assert obs.last_grounding is None
        assert obs.is_done is False
        assert obs.last_reward == 0.0
        assert obs.last_cluster_hits == ()
        assert obs.last_interrogation_questions == ()
        assert obs.last_ralph_run_id is None
        assert obs.last_ralph_iterations == ()
        assert obs.cumulative_audit_summary == {}
        assert obs.error is None

    def test_error_field_set(self) -> None:
        obs = build_observation(
            episode_id="ep1",
            task=_EASY_TASK,
            current_files={},
            budget_remaining=0,
            previous_score=0.0,
            error="something broke",
        )
        assert obs.error == "something broke"


# ---------------------------------------------------------------------------
# Citation shaping tests
# ---------------------------------------------------------------------------


class TestCitationShapingBonus:
    def test_no_citations_returns_zero(self) -> None:
        bonus = citation_shaping_bonus(
            submit_files={"main.py": "x = 1"},
            prior_citations=[],
            prior_cluster_hits=[],
        )
        assert bonus == 0.0

    def test_bonus_computed_when_skill_cited_in_comment(self) -> None:
        bonus = citation_shaping_bonus(
            submit_files={"main.py": "# cited: skill-1\nimport os"},
            prior_citations=[{"skill_name": "skill-1"}],
            prior_cluster_hits=[],
        )
        assert bonus == pytest.approx(0.01)

    def test_bonus_capped_at_005(self) -> None:
        citations_comments = "\n".join(f"# cited: skill-{i}" for i in range(10))
        files = {"main.py": citations_comments}
        citations = [{"skill_name": f"skill-{i}"} for i in range(10)]
        bonus = citation_shaping_bonus(
            submit_files=files,
            prior_citations=citations,
            prior_cluster_hits=[],
        )
        assert bonus == pytest.approx(0.05)

    def test_no_match_returns_zero(self) -> None:
        bonus = citation_shaping_bonus(
            submit_files={"main.py": "unrelated code"},
            prior_citations=[{"skill_name": "other-thing"}],
            prior_cluster_hits=[],
        )
        assert bonus == 0.0

    def test_non_string_skill_name_ignored(self) -> None:
        bonus = citation_shaping_bonus(
            submit_files={"main.py": "code"},
            prior_citations=[{"skill_name": 123}],
            prior_cluster_hits=[],
        )
        assert bonus == 0.0


# ---------------------------------------------------------------------------
# Filename validation tests
# ---------------------------------------------------------------------------

from codeforge.environment import _validate_files  # noqa: E402


class TestValidateFiles:
    def test_valid_filenames_pass(self) -> None:
        assert _validate_files({"main.py": "x = 1", "core.py": "y = 2"}) is None

    def test_conftest_rejected(self) -> None:
        err = _validate_files({"conftest.py": ""})
        assert err is not None
        assert "conftest.py" in err

    def test_non_py_rejected(self) -> None:
        err = _validate_files({"main.txt": ""})
        assert err is not None
        assert "main.txt" in err

    def test_uppercase_rejected(self) -> None:
        err = _validate_files({"Main.py": ""})
        assert err is not None
        assert "Main.py" in err

    def test_empty_files_rejected(self) -> None:
        err = _validate_files({})
        assert err is not None
        assert "empty" in err

    def test_file_count_limit(self) -> None:
        files = {f"f{i}.py": "" for i in range(11)}
        err = _validate_files(files)
        assert err is not None
        assert "too many" in err

    def test_file_size_limit(self) -> None:
        files = {"main.py": "x" * (50 * 1024 + 1)}
        err = _validate_files(files)
        assert err is not None
        assert "exceeds" in err

    def test_total_size_limit(self) -> None:
        # 5 files, each just under 50KB, but total > 200KB
        files = {f"f{i}.py": "x" * 45000 for i in range(5)}
        err = _validate_files(files)
        assert err is not None
        assert "total" in err.lower()

    def test_forbidden_filenames(self) -> None:
        for name in ("pytest.ini", "setup.cfg", "pyproject.toml", "tox.ini"):
            # These are not .py so they fail the regex check
            err = _validate_files({name: ""})
            assert err is not None, f"{name} should be rejected"

    def test_leading_digit_rejected(self) -> None:
        err = _validate_files({"1main.py": ""})
        assert err is not None

    def test_underscore_start_rejected(self) -> None:
        err = _validate_files({"_main.py": ""})
        assert err is not None


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

from codeforge.environment import CodeForgeEnvironment  # noqa: E402


class TestEnvironment:
    @pytest.fixture()
    def corpus_path(self, tmp_path: Path) -> Path:
        return _make_corpus(tmp_path)

    @pytest.fixture()
    def env(self, corpus_path: Path) -> CodeForgeEnvironment:
        return CodeForgeEnvironment(corpus_path=corpus_path)

    def test_reset_returns_valid_observation(self, env: CodeForgeEnvironment) -> None:
        obs = env.reset()
        assert isinstance(obs, CodeForgeObservation)
        assert obs.episode_id != ""
        assert obs.budget_remaining == _EASY_TASK.max_budget
        assert obs.is_done is False

    def test_step_before_reset_returns_error(
        self, env: CodeForgeEnvironment
    ) -> None:
        action = CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="hi")
        obs = env.step(action)
        assert obs.error is not None
        assert "no active episode" in obs.error.lower()

    def test_query_kb_returns_citations(self, env: CodeForgeEnvironment) -> None:
        env.reset()
        action = CodeForgeAction(
            action_type=CodeForgeActionType.QUERY_KB,
            claim="python typing hints",
        )
        obs = env.step(action)
        assert obs.last_citations != ()
        assert obs.last_reward == 0.0
        assert obs.budget_remaining == _EASY_TASK.max_budget - 1

    def test_query_cluster_unknown_label_returns_empty(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        action = CodeForgeAction(
            action_type=CodeForgeActionType.QUERY_CLUSTER,
            cluster_label="nonexistent_cluster",
        )
        obs = env.step(action)
        assert obs.last_cluster_hits == ()
        assert obs.last_reward == 0.0

    def test_interrogate_returns_questions(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        action = CodeForgeAction(action_type=CodeForgeActionType.INTERROGATE)
        obs = env.step(action)
        assert obs.last_interrogation_questions != ()
        assert obs.last_reward == 0.0

    def test_submit_with_valid_files_returns_reward(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            files={"main.py": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'},
            confidence=0.8,
        )
        obs = env.step(action)
        assert obs.last_reward > 0.0
        assert obs.previous_score > 0.0

    def test_submit_with_none_files_returns_error(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            files=None,
            confidence=0.5,
        )
        obs = env.step(action)
        assert obs.error is not None
        assert "files" in obs.error.lower()

    def test_submit_with_forbidden_filename_returns_error(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            files={"conftest.py": ""},
            confidence=0.5,
        )
        obs = env.step(action)
        assert obs.error is not None
        assert "conftest.py" in obs.error

    def test_get_audit_cost_zero(self, env: CodeForgeEnvironment) -> None:
        env.reset()
        initial_budget = _EASY_TASK.max_budget
        action = CodeForgeAction(action_type=CodeForgeActionType.GET_AUDIT)
        obs = env.step(action)
        assert obs.budget_remaining == initial_budget  # cost 0
        assert obs.last_reward == 0.0

    def test_run_ralph_budget_accounting(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset(task_level="hard")  # max_budget=10
        action = CodeForgeAction(
            action_type=CodeForgeActionType.RUN_RALPH,
            max_iters=3,
        )
        obs = env.step(action)
        # run_ralph costs max_iters = 3
        assert obs.budget_remaining == 10 - 3

    def test_run_ralph_insufficient_budget_returns_error(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()  # easy task, budget=4
        action = CodeForgeAction(
            action_type=CodeForgeActionType.RUN_RALPH,
            max_iters=5,
        )
        obs = env.step(action)
        assert obs.error is not None
        assert "budget" in obs.error.lower()
        # Budget should NOT be decremented
        assert obs.budget_remaining == _EASY_TASK.max_budget

    def test_budget_exhaustion_sets_is_done(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()  # budget=4
        for _ in range(4):
            action = CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB, claim="x"
            )
            obs = env.step(action)
        assert obs.is_done is True
        assert obs.budget_remaining == 0

    def test_target_score_hit_sets_is_done(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        # Submit perfect code — should hit high reward
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            files={"main.py": (
                'from __future__ import annotations\n\n\n'
                'def greet(name: str) -> str:\n'
                '    return f"Hello, {name}!"\n'
            )},
            confidence=0.95,
        )
        obs = env.step(action)
        # If reward >= target_score, done
        if obs.last_reward >= _EASY_TASK.target_score:
            assert obs.is_done is True

    def test_unknown_action_type_returns_error_no_budget_decrement(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        # We can't create a CodeForgeAction with an invalid type via the enum,
        # so we test by constructing an action and patching the type
        action = CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="x")
        # Manually override the action_type to something invalid
        action.__dict__["action_type"] = "invalid_action"
        obs = env.step(action)
        assert obs.error is not None
        assert obs.budget_remaining == _EASY_TASK.max_budget

    def test_step_after_done_returns_done_observation(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        # Exhaust budget
        for _ in range(4):
            env.step(
                CodeForgeAction(
                    action_type=CodeForgeActionType.QUERY_KB, claim="x"
                )
            )
        # One more step after done
        obs = env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB, claim="x"
            )
        )
        assert obs.is_done is True

    def test_state_property(self, env: CodeForgeEnvironment) -> None:
        env.reset()
        st = env.state
        assert isinstance(st, CodeForgeObservation)

    def test_get_audit_returns_audit_data(
        self, env: CodeForgeEnvironment
    ) -> None:
        env.reset()
        # Do a query first to have audit data
        env.step(
            CodeForgeAction(
                action_type=CodeForgeActionType.QUERY_KB, claim="x"
            )
        )
        obs = env.step(
            CodeForgeAction(action_type=CodeForgeActionType.GET_AUDIT)
        )
        assert obs.cumulative_audit_summary is not None
        assert isinstance(obs.cumulative_audit_summary, dict)
