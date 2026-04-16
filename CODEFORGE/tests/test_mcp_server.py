from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from codeforge.mcp_server import CodeForgeMCPServer

# ---------------------------------------------------------------------------
# Fixture: tiny JSONL corpus for tests
# ---------------------------------------------------------------------------

_CORPUS_NODES = [
    {
        "id": "node_001",
        "skill_name": "python-testing",
        "section_path": ["Testing", "pytest basics"],
        "section_body": "Use pytest for testing Python code. Fixtures, parametrize, conftest.",
        "tags": ["python", "testing", "pytest"],
        "source_path": "skills/python-testing/SKILL.md",
        "mtime": 1700000000,
        "body_hash": "abc123",
    },
    {
        "id": "node_002",
        "skill_name": "python-patterns",
        "section_path": ["Patterns", "type hints"],
        "section_body": "Use type annotations on all function signatures. PEP 484.",
        "tags": ["python", "typing"],
        "source_path": "skills/python-patterns/SKILL.md",
        "mtime": 1700000001,
        "body_hash": "def456",
    },
    {
        "id": "node_003",
        "skill_name": "python-patterns",
        "section_path": ["Patterns", "error handling"],
        "section_body": "Always handle errors comprehensively. Provide user-friendly messages.",
        "tags": ["python", "errors"],
        "source_path": "skills/python-patterns/SKILL.md",
        "mtime": 1700000002,
        "body_hash": "ghi789",
    },
]


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    p = tmp_path / "skills_corpus.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for node in _CORPUS_NODES:
            f.write(json.dumps(node) + "\n")
    return p


@pytest.fixture()
def server(corpus_path: Path) -> CodeForgeMCPServer:
    return CodeForgeMCPServer(corpus_path=corpus_path)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_returns_all_10_tools(self, server: CodeForgeMCPServer) -> None:
        defs = server.tool_definitions()
        assert len(defs) == 10
        names = {d["name"] for d in defs}
        expected = {
            "codeforge_reset",
            "codeforge_query_kb",
            "codeforge_query_cluster",
            "codeforge_interrogate",
            "codeforge_run_ralph",
            "codeforge_submit",
            "codeforge_get_audit",
            "codeforge_state",
            "codeforge_list_clusters",
            "codeforge_list_tags",
        }
        assert names == expected

    def test_each_tool_has_description_and_schema(self, server: CodeForgeMCPServer) -> None:
        for d in server.tool_definitions():
            assert "description" in d and d["description"]
            assert "inputSchema" in d


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_creates_session(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        assert "_codeforge_version" in result
        assert "session_id" in result
        assert result["observation"]["task_id"] == "greet_single_file"
        assert result["observation"]["budget_remaining"] == 4
        assert result["observation"]["is_done"] is False

    def test_reset_defaults_to_easy(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_reset", {})
        assert result["observation"]["task_id"] == "greet_single_file"

    def test_reset_medium(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_reset", {"task_level": "medium"})
        assert result["observation"]["task_id"] == "greet_with_tests"
        assert result["observation"]["budget_remaining"] == 6

    def test_reset_hard(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_reset", {"task_level": "hard"})
        assert result["observation"]["task_id"] == "multi_file_module"
        assert result["observation"]["budget_remaining"] == 10


# ---------------------------------------------------------------------------
# Version field
# ---------------------------------------------------------------------------

class TestVersionField:
    def test_version_in_reset_response(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        assert result["_codeforge_version"] == "0.2.0"

    def test_version_in_state_response(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_state", {"session_id": sid})
        assert result["_codeforge_version"] == "0.2.0"


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_two_sessions_independent(self, server: CodeForgeMCPServer) -> None:
        r1 = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        r2 = server.handle_tool("codeforge_reset", {"task_level": "hard"})
        sid1, sid2 = r1["session_id"], r2["session_id"]
        assert sid1 != sid2

        s1 = server.handle_tool("codeforge_state", {"session_id": sid1})
        s2 = server.handle_tool("codeforge_state", {"session_id": sid2})
        assert s1["observation"]["task_id"] == "greet_single_file"
        assert s2["observation"]["task_id"] == "multi_file_module"

    def test_invalid_session_returns_error(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_state", {"session_id": "nonexistent"})
        assert result.get("isError") is True
        assert "session" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Query KB
# ---------------------------------------------------------------------------

class TestQueryKB:
    def test_query_kb_returns_citations(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_query_kb", {
            "session_id": sid,
            "claim": "pytest testing patterns",
        })
        obs = result["observation"]
        assert len(obs["last_citations"]) > 0
        assert obs["budget_remaining"] == 3  # started at 4, cost 1

    def test_query_kb_with_top_k(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_query_kb", {
            "session_id": sid,
            "claim": "python",
            "top_k": 2,
        })
        obs = result["observation"]
        assert len(obs["last_citations"]) <= 2


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_submit_returns_reward(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_submit", {
            "session_id": sid,
            "files": {"main.py": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"},
            "confidence": 0.8,
        })
        obs = result["observation"]
        assert isinstance(obs["last_reward"], (int, float))
        assert obs["budget_remaining"] < 4

    def test_submit_without_files_is_error(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_submit", {"session_id": sid})
        obs = result["observation"]
        assert obs.get("error") is not None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_current_observation(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_state", {"session_id": sid})
        obs = result["observation"]
        assert obs["episode_id"] == reset["observation"]["episode_id"]
        assert obs["task_brief"]


# ---------------------------------------------------------------------------
# Get Audit
# ---------------------------------------------------------------------------

class TestGetAudit:
    def test_get_audit_returns_audit_summary(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        # Do a query first to populate audit
        server.handle_tool("codeforge_query_kb", {
            "session_id": sid,
            "claim": "test",
        })
        result = server.handle_tool("codeforge_get_audit", {"session_id": sid})
        obs = result["observation"]
        assert "cumulative_audit_summary" in obs

    def test_get_audit_zero_cost(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        budget_before = reset["observation"]["budget_remaining"]
        result = server.handle_tool("codeforge_get_audit", {"session_id": sid})
        budget_after = result["observation"]["budget_remaining"]
        assert budget_after == budget_before


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

class TestDiscoveryTools:
    def test_list_clusters(self, server: CodeForgeMCPServer) -> None:
        # Need to trigger index building via a reset + query first
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_list_clusters", {"session_id": sid})
        assert "clusters" in result
        assert isinstance(result["clusters"], list)
        assert result["_codeforge_version"] == "0.2.0"

    def test_list_tags(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_list_tags", {"session_id": sid})
        assert "tags" in result
        assert isinstance(result["tags"], list)
        assert result["_codeforge_version"] == "0.2.0"


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

class TestUnknownTool:
    def test_unknown_tool_returns_error(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_unknown", {})
        assert result.get("isError") is True


# ---------------------------------------------------------------------------
# Session eviction
# ---------------------------------------------------------------------------

class TestSessionEviction:
    def test_max_sessions_evicts_oldest(self) -> None:
        """When max_sessions is reached, the oldest session is evicted."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for node in _CORPUS_NODES:
                f.write(json.dumps(node) + "\n")
            corpus = Path(f.name)
        try:
            srv = CodeForgeMCPServer(corpus_path=corpus, max_sessions=2)
            r1 = srv.handle_tool("codeforge_reset", {"task_level": "easy"})
            r2 = srv.handle_tool("codeforge_reset", {"task_level": "medium"})
            r3 = srv.handle_tool("codeforge_reset", {"task_level": "hard"})
            # Session 1 should have been evicted
            sid1 = r1["session_id"]
            result = srv.handle_tool("codeforge_state", {"session_id": sid1})
            assert result.get("isError") is True
            # Session 2 and 3 should still be alive
            sid2, sid3 = r2["session_id"], r3["session_id"]
            assert srv.handle_tool("codeforge_state", {"session_id": sid2}).get("isError") is not True
            assert srv.handle_tool("codeforge_state", {"session_id": sid3}).get("isError") is not True
        finally:
            corpus.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Resource definitions
# ---------------------------------------------------------------------------

class TestResources:
    def test_resource_definitions(self, server: CodeForgeMCPServer) -> None:
        resources = server.resource_definitions()
        uris = {r["uri"] for r in resources}
        assert "codeforge://corpus/stats" in uris
        assert "codeforge://tasks" in uris

    def test_read_corpus_stats(self, server: CodeForgeMCPServer) -> None:
        # Need an active session to have an index built
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        data = server.read_resource("codeforge://corpus/stats", session_id=sid)
        assert "node_count" in data

    def test_read_tasks(self, server: CodeForgeMCPServer) -> None:
        data = server.read_resource("codeforge://tasks")
        assert "tasks" in data
        assert len(data["tasks"]) == 3


# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_prompt_definitions(self, server: CodeForgeMCPServer) -> None:
        prompts = server.prompt_definitions()
        names = {p["name"] for p in prompts}
        assert "codeforge_system" in names
        assert "codeforge_task_brief" in names

    def test_get_system_prompt(self, server: CodeForgeMCPServer) -> None:
        messages = server.get_prompt("codeforge_system")
        assert len(messages) >= 1
        assert "grader" in messages[0]["content"].lower() or "sandbox" in messages[0]["content"].lower()

    def test_get_task_brief_prompt(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        messages = server.get_prompt("codeforge_task_brief", session_id=sid)
        assert len(messages) >= 1
        # Should contain the task brief text
        assert "greet" in messages[0]["content"].lower()

    def test_get_task_brief_no_session_id(self, server: CodeForgeMCPServer) -> None:
        messages = server.get_prompt("codeforge_task_brief")
        assert "error" in messages[0]["content"].lower()

    def test_get_task_brief_invalid_session(self, server: CodeForgeMCPServer) -> None:
        messages = server.get_prompt("codeforge_task_brief", session_id="bad")
        assert "invalid" in messages[0]["content"].lower()

    def test_get_unknown_prompt(self, server: CodeForgeMCPServer) -> None:
        messages = server.get_prompt("nonexistent")
        assert "unknown" in messages[0]["content"].lower()


# ---------------------------------------------------------------------------
# Additional handler coverage
# ---------------------------------------------------------------------------

class TestQueryClusterHandler:
    def test_query_cluster_works(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_query_cluster", {
            "session_id": sid,
            "cluster_label": "nonexistent_cluster",
        })
        obs = result["observation"]
        assert obs["budget_remaining"] == 3

    def test_query_cluster_invalid_session(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_query_cluster", {
            "session_id": "bad",
            "cluster_label": "x",
        })
        assert result.get("isError") is True


class TestInterrogateHandler:
    def test_interrogate_works(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_interrogate", {
            "session_id": sid,
        })
        obs = result["observation"]
        assert obs["budget_remaining"] == 3

    def test_interrogate_invalid_session(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_interrogate", {
            "session_id": "bad",
        })
        assert result.get("isError") is True


class TestRunRalphHandler:
    def test_run_ralph_works(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        result = server.handle_tool("codeforge_run_ralph", {
            "session_id": sid,
            "max_iters": 1,
        })
        obs = result["observation"]
        assert obs["budget_remaining"] == 3  # cost=1

    def test_run_ralph_invalid_session(self, server: CodeForgeMCPServer) -> None:
        result = server.handle_tool("codeforge_run_ralph", {
            "session_id": "bad",
            "max_iters": 1,
        })
        assert result.get("isError") is True


class TestBudgetWarning:
    def test_budget_warning_at_two(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        # Budget 4. Use 2 queries to get to budget=2
        server.handle_tool("codeforge_query_kb", {
            "session_id": sid, "claim": "a",
        })
        result = server.handle_tool("codeforge_query_kb", {
            "session_id": sid, "claim": "b",
        })
        assert "budget_warning" in result

    def test_budget_warning_at_one(self, server: CodeForgeMCPServer) -> None:
        reset = server.handle_tool("codeforge_reset", {"task_level": "easy"})
        sid = reset["session_id"]
        # Use 3 queries to reach budget=1
        server.handle_tool("codeforge_query_kb", {
            "session_id": sid, "claim": "a",
        })
        server.handle_tool("codeforge_query_kb", {
            "session_id": sid, "claim": "b",
        })
        result = server.handle_tool("codeforge_query_kb", {
            "session_id": sid, "claim": "c",
        })
        assert "budget_warning" in result


class TestResourceEdgeCases:
    def test_read_corpus_stats_no_session(self, server: CodeForgeMCPServer) -> None:
        data = server.read_resource("codeforge://corpus/stats")
        assert "error" in data

    def test_read_corpus_stats_invalid_session(self, server: CodeForgeMCPServer) -> None:
        data = server.read_resource("codeforge://corpus/stats", session_id="bad")
        assert "error" in data

    def test_read_unknown_resource(self, server: CodeForgeMCPServer) -> None:
        data = server.read_resource("codeforge://unknown")
        assert "error" in data
