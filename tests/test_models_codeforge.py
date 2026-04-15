from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation


def test_query_action_defaults():
    a = CodeForgeAction(action_type=CodeForgeActionType.QUERY_KB, claim="hi")
    assert a.top_k == 5
    assert a.required_tags == ()


def test_submit_requires_files():
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
