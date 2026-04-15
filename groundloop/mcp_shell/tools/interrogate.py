from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import InterrogateInput

_QUESTION_TEMPLATES = (
    "What are the exact success criteria for '{brief_head}'?",
    "Which external systems or libraries are in or out of scope for '{brief_head}'?",
    "What is the single most failure-prone assumption baked into '{brief_head}'?",
)


def handle_interrogate(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = InterrogateInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    brief_head = inp.brief[:80]
    questions = [t.format(brief_head=brief_head) for t in _QUESTION_TEMPLATES]
    return {"status": "ok", "questions": questions}
