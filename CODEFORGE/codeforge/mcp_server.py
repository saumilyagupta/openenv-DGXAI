from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from codeforge.environment import CodeForgeEnvironment
from codeforge.models import CodeForgeAction, CodeForgeActionType
from codeforge.ralph.synthesizer import Synthesizer
from codeforge.tasks import TASKS

_log = logging.getLogger(__name__)
_VERSION = "0.2.0"

_Handler = Callable[
    ["CodeForgeMCPServer", "dict[str, Any]"],
    "dict[str, Any]",
]

_SID_DESC = "Session ID from codeforge_reset."
_SID_PROP: dict[str, str] = {
    "type": "string",
    "description": _SID_DESC,
}

# -------------------------------------------------------------------
# Tool schema definitions (SYSTEM_DESIGN §9.1)
# -------------------------------------------------------------------

_TOOL_DEFS: tuple[dict[str, Any], ...] = (
    {
        "name": "codeforge_reset",
        "description": (
            "Start a new CodeForge episode. You will receive a task "
            "brief and initial files. Your goal is to produce working "
            "Python code that passes sandbox verification. Budget is "
            "limited — plan your actions carefully."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_level": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": (
                        "Difficulty level. Easy: single file, budget "
                        "4. Medium: multi-file with tests, budget 6. "
                        "Hard: three-file module, budget 10."
                    ),
                },
            },
            "required": ["task_level"],
        },
    },
    {
        "name": "codeforge_query_kb",
        "description": (
            "Search the coding skills knowledge base. Returns real "
            "documentation from 1006 skill nodes. Use this to find "
            "patterns, best practices, and guidance BEFORE writing "
            "code. Costs 1 budget unit. DO NOT guess library APIs — "
            "search for them here or verify via documentation first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "claim": {
                    "type": "string",
                    "description": (
                        "What you want to find guidance on. Be "
                        "specific. Example: 'pytest fixture patterns "
                        "for testing greet functions'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of results to return",
                },
                "required_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "Only return nodes that have ALL of these "
                        "tags"
                    ),
                },
            },
            "required": ["session_id", "claim"],
        },
    },
    {
        "name": "codeforge_query_cluster",
        "description": (
            "Browse a skill cluster by label. Clusters are communities "
            "of related skill nodes grouped by Jaccard similarity. Use "
            "this to explore a topic area deeply. Costs 1 budget unit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "cluster_label": {
                    "type": "string",
                    "description": (
                        "The cluster label to look up. Example: "
                        "'python_testing_pytest_fixtures'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["session_id", "cluster_label"],
        },
    },
    {
        "name": "codeforge_interrogate",
        "description": (
            "Get Socratic questions about the task that cite real skill "
            "corpus nodes. Use this BEFORE writing code to identify "
            "edge cases, success criteria, and assumptions you might be "
            "wrong about. Costs 1 budget unit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "brief_override": {
                    "type": "string",
                    "description": (
                        "Optional override for the task brief. "
                        "If omitted, uses the current task brief."
                    ),
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "codeforge_run_ralph",
        "description": (
            "Run autonomous improvement iterations on your current "
            "code. Each iteration: synthesize improvement → "
            "sandbox-score → keep if better. Costs max_iters budget "
            "units. Wasted iterations (no improvement) cost 0.05 "
            "penalty each. Use when you want the environment to "
            "iteratively improve your code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "max_iters": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "description": (
                        "Maximum iterations. Each costs 1 budget. "
                        "Choose carefully."
                    ),
                },
            },
            "required": ["session_id", "max_iters"],
        },
    },
    {
        "name": "codeforge_submit",
        "description": (
            "Submit Python files for grading. Your code will be: "
            "(1) written to a sandbox and checked by ruff, mypy "
            "--strict, pytest, and import resolution — these are REAL "
            "tools, not mocks; (2) AST-grounded to verify every "
            "import and attribute access resolves to a real Python "
            "module/attribute; (3) scored via quality = 0.6*sandbox + "
            "0.4*groundedness; (4) if you provide confidence, "
            "Brier-penalized: reward = quality * (1 - "
            "min((confidence-quality)^2, 0.5)). DO NOT fabricate "
            "library names or API signatures — the grounder WILL "
            "catch them and your score WILL drop."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "Map of filename to file content. Example: "
                        '{"main.py": "def greet(name: str) -> str:'
                        "\\n    return f'Hello, {name}!'\\n\"}"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Your confidence that this submission is "
                        "correct (0.0 = no idea, 1.0 = certain). "
                        "Overconfidence on bad code is PENALIZED. "
                        "Honest uncertainty is treated more "
                        "favorably. If you are unsure, say so."
                    ),
                },
            },
            "required": ["session_id", "files"],
        },
    },
    {
        "name": "codeforge_get_audit",
        "description": (
            "Read the audit trail for the current episode (or a "
            "specific run). Returns every action taken, every citation "
            "made, every reward earned, and the evidence behind each. "
            "Costs 0 budget. Use this to review your progress and "
            "understand what worked."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": _SID_PROP,
                "target_run_id": {
                    "type": "string",
                    "description": (
                        "Optional run ID to audit. "
                        "Defaults to current episode."
                    ),
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "codeforge_state",
        "description": (
            "Get current episode state without taking an action. Shows "
            "task brief, current files, budget remaining, last reward, "
            "and whether the episode is done. Costs 0 budget."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": _SID_PROP},
            "required": ["session_id"],
        },
    },
    {
        "name": "codeforge_list_clusters",
        "description": (
            "List all available cluster labels and their node counts. "
            "Use this to discover what topic areas exist before "
            "calling codeforge_query_cluster. Costs 0 budget."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": _SID_PROP},
        },
    },
    {
        "name": "codeforge_list_tags",
        "description": (
            "List all available tags in the skill corpus. Use this to "
            "discover valid values for the required_tags parameter. "
            "Costs 0 budget."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": _SID_PROP},
        },
    },
)

# -------------------------------------------------------------------
# Resource definitions
# -------------------------------------------------------------------

_RESOURCE_DEFS: tuple[dict[str, str], ...] = (
    {
        "uri": "codeforge://corpus/stats",
        "name": "Corpus Statistics",
        "description": (
            "Corpus statistics (node count, vocab size, cluster count)"
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "codeforge://corpus/node/{node_id}",
        "name": "Skill Node",
        "description": (
            "Full content of a specific skill node (free, no budget)"
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "codeforge://tasks",
        "name": "Task Definitions",
        "description": (
            "Task definitions with briefs, budgets, targets, tools"
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "codeforge://audit/{episode_id}",
        "name": "Audit Ledger",
        "description": (
            "Serialized audit ledger for a completed episode"
        ),
        "mimeType": "application/json",
    },
)

# -------------------------------------------------------------------
# Prompt text
# -------------------------------------------------------------------

_SYSTEM_PROMPT_TEXT = (
    "You are solving a CodeForge episode. Your code is graded by "
    "REAL tools (ruff, mypy --strict, pytest, import resolution) in "
    "a sandbox. Every import and attribute access is AST-grounded "
    "against the real Python runtime. Overconfidence is penalized "
    "via Brier scoring. Honest uncertainty about genuinely uncertain "
    "results is rewarded.\n\n"
    "Rules:\n"
    "- DO NOT fabricate library names or API signatures — "
    "the grounder catches them.\n"
    "- DO NOT submit stubs (pass, ..., NotImplementedError) — "
    "they score zero.\n"
    "- Use codeforge_query_kb to find patterns BEFORE writing code.\n"
    "- Use codeforge_interrogate to identify edge cases.\n"
    "- Budget is limited. Plan actions carefully.\n"
    "- If unsure of your confidence, set it low — "
    "the grader rewards honesty.\n"
)

_SESSION_ERR = "Invalid session_id: {sid!r}. Call codeforge_reset first."


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _obs_to_dict(obs: Any) -> dict[str, Any]:
    """Convert a CodeForgeObservation to a serializable dict."""
    result: dict[str, Any] = json.loads(obs.model_dump_json())
    return result


def _make_response(
    obs: Any,
    *,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned response dict from an observation."""
    result: dict[str, Any] = {"_codeforge_version": _VERSION}
    if session_id is not None:
        result["session_id"] = session_id
    result["observation"] = _obs_to_dict(obs)
    if extra:
        result.update(extra)
    budget = result["observation"].get("budget_remaining", 0)
    if isinstance(budget, int) and 0 < budget <= 2:
        result["budget_warning"] = (
            f"WARNING: {budget} budget remaining — plan carefully."
        )
    return result


def _session_error(sid: str) -> dict[str, Any]:
    """Return an isError response for a missing session."""
    return {
        "isError": True,
        "error": _SESSION_ERR.format(sid=sid),
        "_codeforge_version": _VERSION,
    }


def _require_session(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> tuple[str, CodeForgeEnvironment | None]:
    """Extract session_id and look up the environment."""
    sid: str = arguments.get("session_id", "")
    return sid, server._get_session(sid)


# -------------------------------------------------------------------
# CodeForgeMCPServer — embedded mode
# -------------------------------------------------------------------


class CodeForgeMCPServer:
    """MCP server wrapping CodeForgeEnvironment (SYSTEM_DESIGN §9).

    Embedded mode: imports CodeForgeEnvironment directly.
    Each tool call is routed to a session-keyed environment.

    **Universal LLM support:** The Ralph loop's synthesizer is configurable.
    Whichever LLM connects to this MCP server can provide its own config::

        # Ollama (local, free)
        server = CodeForgeMCPServer(llm_provider="ollama", llm_model="llama3")

        # OpenAI
        server = CodeForgeMCPServer(llm_provider="openai", llm_model="gpt-4o")

        # Anthropic
        server = CodeForgeMCPServer(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")

        # Any OpenAI-compatible (vLLM, LM Studio, Together, Groq)
        server = CodeForgeMCPServer(
            llm_provider="openai",
            llm_base_url="http://localhost:8000/v1",
            llm_model="my-model",
        )

    When no LLM config is provided, Ralph uses the deterministic
    StubSynthesizer (no API calls needed).
    """

    def __init__(
        self,
        *,
        corpus_path: Path | None = None,
        max_sessions: int = 10,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self._corpus_path = corpus_path
        self._sessions: dict[str, CodeForgeEnvironment] = {}
        self._max_sessions = max_sessions
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model

    # -- Session management ------------------------------------------

    def _get_session(
        self,
        session_id: str,
    ) -> CodeForgeEnvironment | None:
        return self._sessions.get(session_id)

    def _create_session(
        self,
    ) -> tuple[str, CodeForgeEnvironment]:
        sid = uuid4().hex[:16]
        # Build synthesizer from LLM config (if provided)
        synth: Synthesizer | None = None
        if self._llm_provider:
            from codeforge.ralph.synthesizer import LLMSynthesizer

            synth = LLMSynthesizer(
                provider=self._llm_provider,
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
                model=self._llm_model,
            )

        env = CodeForgeEnvironment(
            corpus_path=self._corpus_path,
            synthesizer=synth,
        )
        if len(self._sessions) >= self._max_sessions:
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]
        self._sessions[sid] = env
        return sid, env

    # -- Public: definitions -----------------------------------------

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool schemas matching SYSTEM_DESIGN §9.1."""
        return [dict(d) for d in _TOOL_DEFS]

    def resource_definitions(self) -> list[dict[str, str]]:
        """Return MCP resource definitions."""
        return [dict(r) for r in _RESOURCE_DEFS]

    def prompt_definitions(self) -> list[dict[str, Any]]:
        """Return MCP prompt definitions."""
        return [
            {
                "name": "codeforge_system",
                "description": (
                    "System prompt injected at session start. "
                    "Contains task rules, budget constraints, "
                    "grading explanation."
                ),
                "arguments": [],
            },
            {
                "name": "codeforge_task_brief",
                "description": (
                    "Dynamic prompt populated with the current "
                    "task's brief, initial files, budget, target "
                    "score, and tool config."
                ),
                "arguments": [
                    {
                        "name": "session_id",
                        "description": _SID_DESC,
                        "required": True,
                    },
                ],
            },
        ]

    # -- Public: handle_tool -----------------------------------------

    def handle_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Route tool call to handler, return result dict."""
        handler = _HANDLERS.get(tool_name)
        if handler is None:
            return {
                "isError": True,
                "error": f"Unknown tool: {tool_name!r}",
                "_codeforge_version": _VERSION,
            }
        return handler(self, arguments)

    # -- Public: resources -------------------------------------------

    def read_resource(
        self,
        uri: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Read an MCP resource by URI."""
        if uri == "codeforge://corpus/stats":
            if session_id is None:
                return {
                    "_codeforge_version": _VERSION,
                    "error": "session_id required for corpus stats",
                }
            env = self._get_session(session_id)
            if env is None:
                return {"_codeforge_version": _VERSION, "error": "Invalid session_id"}
            idx = env._ensure_index()
            stats = idx.stats()
            cluster_count = len(idx.all_cluster_labels())
            return {
                "node_count": stats["node_count"],
                "vocab_size": stats["vocab_size"],
                "avg_doc_len": stats["avg_doc_len"],
                "cluster_count": cluster_count,
            }
        if uri.startswith("codeforge://corpus/node/"):
            node_id = uri.removeprefix("codeforge://corpus/node/")
            if session_id is None:
                return {
                    "_codeforge_version": _VERSION,
                    "error": "session_id required for node lookup",
                }
            env = self._get_session(session_id)
            if env is None:
                return {"_codeforge_version": _VERSION, "error": "Invalid session_id"}
            idx = env._ensure_index()
            for node in idx._nodes:
                if node.get("id") == node_id:
                    return {"_codeforge_version": _VERSION, "node": node}
            return {"_codeforge_version": _VERSION, "error": f"Node {node_id!r} not found"}
        if uri == "codeforge://tasks":
            return {
                "_codeforge_version": _VERSION,
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
            }
        if uri.startswith("codeforge://audit/"):
            episode_id = uri.removeprefix("codeforge://audit/")
            if session_id is None:
                return {
                    "_codeforge_version": _VERSION,
                    "error": "session_id required for audit lookup",
                }
            env = self._get_session(session_id)
            if env is None:
                return {"_codeforge_version": _VERSION, "error": "Invalid session_id"}
            if env._ledger is not None:
                return {
                    "_codeforge_version": _VERSION,
                    "episode_id": episode_id,
                    "audit": env._ledger.serialize(),
                }
            return {
                "_codeforge_version": _VERSION,
                "error": "No audit data for this session",
            }
        return {"_codeforge_version": _VERSION, "error": f"Unknown resource URI: {uri!r}"}

    # -- Public: prompts ---------------------------------------------

    def get_prompt(
        self,
        name: str,
        *,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Return prompt messages for the given prompt name."""
        if name == "codeforge_system":
            return [
                {"role": "system", "content": _SYSTEM_PROMPT_TEXT},
            ]
        if name == "codeforge_task_brief":
            if session_id is None:
                return [
                    {
                        "role": "system",
                        "content": (
                            "Error: session_id required for "
                            "task_brief prompt."
                        ),
                    },
                ]
            env = self._get_session(session_id)
            if env is None:
                return [
                    {
                        "role": "system",
                        "content": "Error: invalid session_id.",
                    },
                ]
            obs = env.state
            task = env._task
            target = task.target_score if task is not None else 0.0
            content = (
                f"## Task: {obs.task_id}\n"
                f"**Level:** {obs.task_level}\n"
                f"**Brief:** {obs.task_brief}\n"
                f"**Budget:** {obs.budget_remaining}\n"
                f"**Target score:** {target}\n\n"
                "### Initial files\n"
            )
            for fname, body in obs.initial_files.items():
                content += (
                    f"\n**{fname}:**\n```python\n{body}\n```\n"
                )
            return [{"role": "system", "content": content}]
        return [
            {
                "role": "system",
                "content": f"Unknown prompt: {name!r}",
            },
        ]


# -------------------------------------------------------------------
# Tool handlers (private, keyed by tool name)
# -------------------------------------------------------------------


def _handle_reset(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    task_level = arguments.get("task_level", "easy")
    sid, env = server._create_session()
    obs = env.reset(task_level=task_level)
    return _make_response(obs, session_id=sid)


def _handle_query_kb(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.QUERY_KB,
        claim=arguments.get("claim"),
        top_k=arguments.get("top_k", 5),
        required_tags=tuple(arguments.get("required_tags", ())),
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_query_cluster(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.QUERY_CLUSTER,
        cluster_label=arguments.get("cluster_label"),
        top_k=arguments.get("top_k", 10),
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_interrogate(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.INTERROGATE,
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_run_ralph(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.RUN_RALPH,
        max_iters=arguments.get("max_iters", 3),
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_submit(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.SUBMIT,
        files=arguments.get("files"),
        confidence=arguments.get("confidence"),
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_get_audit(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    action = CodeForgeAction(
        action_type=CodeForgeActionType.GET_AUDIT,
        target_run_id=arguments.get("target_run_id"),
    )
    obs = env.step(action)
    return _make_response(obs, session_id=sid)


def _handle_state(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    sid, env = _require_session(server, arguments)
    if env is None:
        return _session_error(sid)
    obs = env.state
    return _make_response(obs, session_id=sid)


def _handle_list_clusters(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _sid, env = _require_session(server, arguments)
    if env is None:
        return {"_codeforge_version": _VERSION, "clusters": []}
    try:
        idx = env._ensure_index()
    except FileNotFoundError:
        return {"_codeforge_version": _VERSION, "clusters": []}
    labels = idx.all_cluster_labels()
    cluster_info: list[dict[str, Any]] = []
    for label in labels:
        cluster = idx.cluster_by_label(label)
        if cluster is not None:
            cluster_info.append({
                "label": cluster.label,
                "node_count": cluster.node_count,
            })
    return {"_codeforge_version": _VERSION, "clusters": cluster_info}


def _handle_list_tags(
    server: CodeForgeMCPServer,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _sid, env = _require_session(server, arguments)
    if env is None:
        return {"_codeforge_version": _VERSION, "tags": []}
    try:
        idx = env._ensure_index()
    except FileNotFoundError:
        return {"_codeforge_version": _VERSION, "tags": []}
    return {
        "_codeforge_version": _VERSION,
        "tags": sorted(idx.all_tags()),
    }


# Handler dispatch table
_HANDLERS: dict[str, _Handler] = {
    "codeforge_reset": _handle_reset,
    "codeforge_query_kb": _handle_query_kb,
    "codeforge_query_cluster": _handle_query_cluster,
    "codeforge_interrogate": _handle_interrogate,
    "codeforge_run_ralph": _handle_run_ralph,
    "codeforge_submit": _handle_submit,
    "codeforge_get_audit": _handle_get_audit,
    "codeforge_state": _handle_state,
    "codeforge_list_clusters": _handle_list_clusters,
    "codeforge_list_tags": _handle_list_tags,
}
