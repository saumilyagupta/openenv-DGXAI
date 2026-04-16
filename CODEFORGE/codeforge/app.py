from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Response
from openenv.core.env_server.http_server import create_app

from codeforge.environment import CodeForgeEnvironment
from codeforge.models import CodeForgeAction, CodeForgeObservation
from codeforge.tasks import TASKS

_log = logging.getLogger(__name__)

_corpus_path_str = os.environ.get("GROUNDLOOP_CORPUS_PATH")
_corpus_path = Path(_corpus_path_str) if _corpus_path_str else None

# ---------------------------------------------------------------------------
# Session-keyed environment pool (SYSTEM_DESIGN §15)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_sessions: dict[str, CodeForgeEnvironment] = {}
_MAX_SESSIONS = int(os.environ.get("CODEFORGE_MAX_SESSIONS", "10"))
_SESSION_TTL_S = int(os.environ.get("CODEFORGE_SESSION_TTL", "3600"))


def _get_or_create_env() -> CodeForgeEnvironment:
    """For OpenEnv compliance — creates a single-session env.

    The session pool is used by the MCP server layer (Phase 2).
    """
    return CodeForgeEnvironment(corpus_path=_corpus_path)


def get_session(session_id: str) -> CodeForgeEnvironment | None:
    """Retrieve an existing session by ID."""
    with _lock:
        return _sessions.get(session_id)


def create_session() -> tuple[str, CodeForgeEnvironment]:
    """Create a new session with a unique ID, evicting oldest if at capacity."""
    sid = uuid4().hex[:16]
    env = CodeForgeEnvironment(corpus_path=_corpus_path)
    with _lock:
        if len(_sessions) >= _MAX_SESSIONS:
            oldest_key = next(iter(_sessions))
            del _sessions[oldest_key]
        _sessions[sid] = env
    return sid, env


# ---------------------------------------------------------------------------
# OpenEnv compliance app
# ---------------------------------------------------------------------------
app: FastAPI = create_app(_get_or_create_env, CodeForgeAction, CodeForgeObservation)


@app.get("/", summary="Health check")
def root() -> dict[str, str]:
    return {"name": "code-forge", "version": "0.2.0", "status": "ok", "docs": "/docs"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/tasks", summary="List tasks + action schema")
def list_tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "id": t.task_id,
                "difficulty": t.task_level,
                "brief": t.brief,
                "target_score": t.target_score,
                "max_budget": t.max_budget,
                "tools": list(t.tools),
            }
            for t in TASKS
        ],
        "action_schema": {
            "action_types": [
                "query_kb",
                "query_cluster",
                "interrogate",
                "run_ralph",
                "submit",
                "get_audit",
            ],
        },
    }


@app.get("/health/deep", summary="Deep health check")
def health_check() -> dict[str, Any]:
    """Check all dependencies: corpus file, tools."""
    checks: dict[str, bool] = {
        "ruff": shutil.which("ruff") is not None,
        "mypy": shutil.which("mypy") is not None,
        "pytest": shutil.which("pytest") is not None,
    }
    corpus_ok = _corpus_path is not None and Path(_corpus_path).is_file()
    checks["corpus"] = corpus_ok
    all_ok = all(checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
