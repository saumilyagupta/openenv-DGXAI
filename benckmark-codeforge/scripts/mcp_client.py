"""
Thin wrapper around CodeForgeMCPServer for benchmark use.

Provides an async-friendly interface that, per MBPP task:
  1. resets a CodeForge session (task_level="hard" gives us budget=10)
  2. calls codeforge_query_kb to fetch citations
  3. calls codeforge_interrogate to fetch Socratic questions

These two signals are then injected into the LLM prompt. We do NOT call
codeforge_submit for MBPP samples because MBPP has its own hidden tests
and our local sandbox is the authoritative grader.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Put CODEFORGE on sys.path so `from codeforge...` imports resolve
_CF_ROOT = (ROOT.parent / "CODEFORGE").resolve()
if str(_CF_ROOT) not in sys.path:
    sys.path.insert(0, str(_CF_ROOT))


@dataclass(frozen=True)
class MCPContext:
    session_id: str
    citations: list[dict[str, Any]]
    questions: list[str]
    kb_budget_remaining: int
    error: str | None = None


class MCPClient:
    """Embedded CodeForge MCP client — no HTTP, direct Python call."""

    def __init__(self, corpus_path: Path, *, task_level: str = "hard", top_k: int = 3) -> None:
        self._corpus_path = corpus_path
        self._task_level = task_level
        self._top_k = top_k
        self._server = None  # lazy
        # CodeForgeMCPServer is NOT thread-safe — session dict mutations race.
        # Serialize all handle_tool calls with this lock.
        self._lock = threading.Lock()

    def _ensure_server(self) -> Any:
        if self._server is None:
            from codeforge.mcp_server import CodeForgeMCPServer  # type: ignore

            self._server = CodeForgeMCPServer(corpus_path=self._corpus_path)
        return self._server

    def gather_context(self, claim: str) -> MCPContext:
        """Run reset → query_kb → interrogate for a single MBPP problem.

        Thread-safe: all MCP calls are serialized under a single lock because
        CodeForgeMCPServer mutates a shared session dict without its own
        synchronization.
        """
        server = self._ensure_server()
        try:
            with self._lock:
                reset = server.handle_tool("codeforge_reset", {"task_level": self._task_level})
                sid = reset["session_id"]

                kb = server.handle_tool(
                    "codeforge_query_kb",
                    {"session_id": sid, "claim": claim or "python function", "top_k": self._top_k},
                )
                kb_obs = kb.get("observation", {}) or {}
                citations = list(kb_obs.get("last_citations", []) or [])
                budget = int(kb_obs.get("budget_remaining", -1))

                interr = server.handle_tool(
                    "codeforge_interrogate",
                    {"session_id": sid},
                )
                interr_obs = interr.get("observation", {}) or {}
                questions = list(interr_obs.get("last_interrogation_questions", []) or [])

            return MCPContext(
                session_id=sid,
                citations=citations,
                questions=questions,
                kb_budget_remaining=budget,
            )
        except Exception as e:  # defensive — benchmark must not crash on one bad task
            return MCPContext(
                session_id="",
                citations=[],
                questions=[],
                kb_budget_remaining=0,
                error=f"{type(e).__name__}: {e}",
            )
