"""
CodeForge MCP Agent Demo — An LLM agent using CodeForge MCP to build a multi-file Python module.

This script simulates an LLM agent that uses the CodeForge MCP server tools
to research, plan, and submit code for the hard task (multi_file_module).
"""
from __future__ import annotations

import json
import sys
import textwrap

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.path.insert(0, "CODEFORGE")

from pathlib import Path
from codeforge.mcp_server import CodeForgeMCPServer

SEP = "=" * 70


def pprint_json(data: object, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, default=str)


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Instantiate server
# ---------------------------------------------------------------------------

print(SEP)
print("  AGENT WITH CODEFORGE MCP")
print("  Demonstrating a full episode using all MCP tools")
print(SEP)

corpus = Path("CODEFORGE/codeforge/kb/skills_corpus.jsonl")
print(f"\nInitializing CodeForgeMCPServer with corpus: {corpus}")
server = CodeForgeMCPServer(corpus_path=corpus)
print("Server ready.\n")

# ===========================================================================
# STEP 1: Reset a hard task session
# ===========================================================================

section("STEP 1 — Reset hard task")

print('Thinking: The hard task is multi_file_module — three files: main.py,')
print('core.py, test_core.py. Budget is 10. Target score is 0.70.')
print('I need to split a greet() function across core.py and main.py, then')
print('write tests in test_core.py. Let me start by resetting.\n')

print('> Calling: codeforge_reset(task_level="hard")')
reset_result = server.handle_tool("codeforge_reset", {"task_level": "hard"})

session_id = reset_result["session_id"]
obs = reset_result["observation"]

print(f'  Session ID: {session_id}')
print(f'  Task ID:    {obs["task_id"]}')
print(f'  Task level: {obs["task_level"]}')
print(f'  Brief:      {obs["task_brief"]}')
print(f'  Budget:     {obs["budget_remaining"]}')
print(f'  Initial files: {list(obs["initial_files"].keys())}')

for fname, content in obs["initial_files"].items():
    print(f'\n  --- {fname} ---')
    for line in content.splitlines():
        print(f'  {line}')

print(f'\nThinking: I have 10 budget. The task wants me to split greet()')
print(f'into core.py, import it in main.py, and write tests in test_core.py.')
print(f'core.py and test_core.py start empty. main.py already imports from core.')
print(f'Let me research best practices first.')

# ===========================================================================
# STEP 2: Query KB — research best practices
# ===========================================================================

section("STEP 2 — Research: Python type hints and error handling")

claim1 = "python function type hints return type annotations best practices"
print(f'> Calling: codeforge_query_kb(session_id=..., claim="{claim1}")')

kb_result1 = server.handle_tool("codeforge_query_kb", {
    "session_id": session_id,
    "claim": claim1,
    "top_k": 5,
})
obs1 = kb_result1["observation"]
citations1 = obs1.get("last_citations", [])

print(f'  Budget remaining: {obs1["budget_remaining"]}')
print(f'  Got {len(citations1)} citations:')
for i, c in enumerate(citations1, 1):
    skill = c.get("skill_name", "?")
    score = c.get("score", 0)
    body = str(c.get("section_body", ""))[:120]
    print(f'    {i}. {skill} (score={score:.1f})')
    print(f'       "{body}..."')

print(f'\nThinking: The KB tells me to use proper type hints on all function')
print(f'signatures with return types. I should annotate greet(name: str) -> str.')
print(f'Let me also check testing patterns.')

# ===========================================================================
# STEP 3: Interrogate — get Socratic questions about the task
# ===========================================================================

section("STEP 3 — Interrogate: Socratic questions about the task")

print('> Calling: codeforge_interrogate(session_id=...)')
interr_result = server.handle_tool("codeforge_interrogate", {
    "session_id": session_id,
})
obs_interr = interr_result["observation"]
questions = obs_interr.get("last_interrogation_questions", [])

print(f'  Budget remaining: {obs_interr["budget_remaining"]}')
print(f'  Got {len(questions)} Socratic questions:')
for i, q in enumerate(questions, 1):
    print(f'    ? {q}')

print(f'\nThinking: The interrogator reminds me to consider edge cases.')
print(f'I need to make sure: (1) all functions have type hints, (2) mypy --strict')
print(f'passes, (3) test_core.py actually tests the greet function, (4) imports')
print(f'resolve correctly across files.')

# ===========================================================================
# STEP 4: Query KB again — pytest testing patterns
# ===========================================================================

section("STEP 4 — Research: pytest testing patterns")

claim2 = "pytest test functions assert string return value testing conventions"
print(f'> Calling: codeforge_query_kb(session_id=..., claim="{claim2}")')

kb_result2 = server.handle_tool("codeforge_query_kb", {
    "session_id": session_id,
    "claim": claim2,
    "top_k": 5,
})
obs2 = kb_result2["observation"]
citations2 = obs2.get("last_citations", [])

print(f'  Budget remaining: {obs2["budget_remaining"]}')
print(f'  Got {len(citations2)} citations:')
for i, c in enumerate(citations2, 1):
    skill = c.get("skill_name", "?")
    score = c.get("score", 0)
    body = str(c.get("section_body", ""))[:120]
    print(f'    {i}. {skill} (score={score:.1f})')
    print(f'       "{body}..."')

print(f'\nThinking: Good. I have enough context from the KB. Time to write code.')
print(f'The KB citations about testing conventions confirm I should use simple')
print(f'assert statements and test both normal cases and edge cases (empty string).')

# ===========================================================================
# STEP 5: Submit code — all 3 files
# ===========================================================================

section("STEP 5 — Submit code with confidence")

# Build the three files based on the task brief and KB guidance.
# The task says: greet("Alice") returns "Hello, Alice!"
# Must be: core.py has greet(), main.py imports it, test_core.py tests it.
# All must have type hints. mypy --strict clean.

core_py = textwrap.dedent("""\
    from __future__ import annotations


    def greet(name: str) -> str:
        \"\"\"Return a greeting for the given name.\"\"\"
        return f"Hello, {name}!"
""")

main_py = textwrap.dedent("""\
    from __future__ import annotations

    from core import greet


    if __name__ == "__main__":
        print(greet("World"))
""")

test_core_py = textwrap.dedent("""\
    from __future__ import annotations

    from core import greet


    def test_greet_alice() -> None:
        assert greet("Alice") == "Hello, Alice!"


    def test_greet_bob() -> None:
        assert greet("Bob") == "Hello, Bob!"


    def test_greet_empty_string() -> None:
        assert greet("") == "Hello, !"


    def test_greet_returns_str() -> None:
        result = greet("Test")
        assert isinstance(result, str)
""")

files = {
    "core.py": core_py,
    "main.py": main_py,
    "test_core.py": test_core_py,
}

print('Thinking: I am fairly confident this code is correct.')
print('- core.py: greet() with full type hints, returns f-string')
print('- main.py: imports from core, runs if __name__ == "__main__"')
print('- test_core.py: 4 tests covering Alice, Bob, empty string, type check')
print('- All files have from __future__ import annotations')
print('- No external dependencies, only stdlib + local imports')
print('- I will set confidence = 0.85 — high but not perfect, because I cannot')
print('  be 100% sure about mypy --strict without running it myself.\n')

print('> Calling: codeforge_submit(session_id=..., files={...}, confidence=0.85)')
print('  Files submitted:')
for fname, content in files.items():
    lines = content.strip().splitlines()
    print(f'\n  --- {fname} ({len(lines)} lines) ---')
    for line in lines:
        print(f'  {line}')

submit_result = server.handle_tool("codeforge_submit", {
    "session_id": session_id,
    "files": files,
    "confidence": 0.85,
})

obs_submit = submit_result["observation"]
reward = obs_submit["last_reward"]
score = obs_submit["previous_score"]
done = obs_submit["is_done"]
budget = obs_submit["budget_remaining"]
grounding = obs_submit.get("last_grounding")
error = obs_submit.get("error")

print(f'\n  RESULT:')
print(f'  Reward:           {reward}')
print(f'  Previous score:   {score}')
print(f'  Budget remaining: {budget}')
print(f'  Episode done:     {done}')
if error:
    print(f'  Error:            {error}')

if grounding:
    print(f'\n  Grounding report:')
    print(f'    Total symbols:  {grounding.get("total_symbols", "?")}')
    print(f'    Groundedness:   {grounding.get("groundedness", "?")}')
    grounded_syms = grounding.get("grounded", [])
    ungrounded_syms = grounding.get("ungrounded", [])
    print(f'    Grounded:       {len(grounded_syms)} symbols')
    print(f'    Ungrounded:     {len(ungrounded_syms)} symbols')
    if ungrounded_syms:
        for u in ungrounded_syms:
            print(f'      UNGROUNDED: {u.get("module", "?")}.{u.get("attr", "?")} (line {u.get("line", "?")})')

if "budget_warning" in submit_result:
    print(f'\n  {submit_result["budget_warning"]}')

print(f'\nThinking: Reward is {reward}. Target is 0.70.')
if reward >= 0.70:
    print('The submission passed the target score. The sandbox verified my code')
    print('with real ruff, mypy, and pytest. The grounder confirmed all imports resolve.')
else:
    print('The submission did not reach the target. Let me check the audit to understand why.')

# ===========================================================================
# STEP 6: Get audit — review what happened
# ===========================================================================

section("STEP 6 — Get audit trail")

print('> Calling: codeforge_get_audit(session_id=...)')
audit_result = server.handle_tool("codeforge_get_audit", {
    "session_id": session_id,
})
obs_audit = audit_result["observation"]
audit_summary = obs_audit.get("cumulative_audit_summary", {})

print(f'  Budget remaining: {obs_audit["budget_remaining"]} (get_audit costs 0)')

entries = audit_summary.get("entries", [])
print(f'  Total audit entries: {len(entries)}')
for i, entry in enumerate(entries):
    action_type = entry.get("action_type", "?")
    rwd = entry.get("reward", 0)
    quality = entry.get("quality", 0)
    brier = entry.get("brier_penalty")
    conf = entry.get("confidence_declared")
    cited_ids = entry.get("cited_skill_ids", [])
    print(f'\n  Entry {i}: action={action_type}')
    print(f'    Reward: {rwd}')
    print(f'    Quality: {quality}')
    if brier is not None:
        print(f'    Brier penalty: {brier}')
    if conf is not None:
        print(f'    Confidence declared: {conf}')
    if cited_ids:
        print(f'    Cited skills: {len(cited_ids)} node(s)')

# ===========================================================================
# STEP 7: Check final state
# ===========================================================================

section("STEP 7 — Final state check")

print('> Calling: codeforge_state(session_id=...)')
state_result = server.handle_tool("codeforge_state", {"session_id": session_id})
obs_state = state_result["observation"]

print(f'  Episode ID:       {obs_state["episode_id"]}')
print(f'  Task:             {obs_state["task_id"]}')
print(f'  Budget remaining: {obs_state["budget_remaining"]}')
print(f'  Last reward:      {obs_state["last_reward"]}')
print(f'  Is done:          {obs_state["is_done"]}')
print(f'  Current files:    {list(obs_state["current_files"].keys())}')

# ===========================================================================
# SUMMARY
# ===========================================================================

section("SUMMARY (WITH MCP)")

total_actions = len(entries)
budget_used = 10 - obs_state["budget_remaining"]

print(f'Actions taken:    {total_actions}')
print(f'Budget used:      {budget_used}/10')
print(f'Final reward:     {obs_state["last_reward"]}')
print(f'Target score:     0.70')
print(f'Task completed:   {"Yes" if obs_state["is_done"] else "No"}')

# Summarize what each action did
print(f'\nAction timeline:')
action_names = []
for entry in entries:
    at = entry.get("action_type", "?")
    rwd = entry.get("reward", 0)
    action_names.append(at)
    print(f'  {at:20s} reward={rwd}')

print(f'\nKey benefit: KB citations from steps 2 and 4 guided the agent to use')
print(f'proper type hints (from __future__ import annotations, return type')
print(f'annotations) and pytest patterns (simple assert, edge case coverage).')
print(f'The interrogator in step 3 prompted consideration of edge cases.')
print(f'The grounding check verified all imports resolve to real modules.')
print(f'The Brier calibration rewarded honest confidence estimation.')

print(f'\n{SEP}')
print(f'  END OF DEMO')
print(f'{SEP}')
