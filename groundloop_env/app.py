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
