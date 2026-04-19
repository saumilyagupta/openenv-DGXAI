"""Demo: LLM Agent WITHOUT CodeForge MCP guardrails.

Simulates what happens when an LLM just writes code without consulting
the knowledge base, interrogating the task, or checking the audit trail.

Run from repo root:
    PYTHONPATH=CODEFORGE:$PYTHONPATH python3 demo_without_mcp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure CODEFORGE is on the path
sys.path.insert(0, "CODEFORGE")

from codeforge.mcp_server import CodeForgeMCPServer

DIVIDER = "=" * 60


def main() -> None:
    print(f"\n{DIVIDER}")
    print("  AGENT WITHOUT CODEFORGE MCP")
    print(f"{DIVIDER}\n")

    server = CodeForgeMCPServer(
        corpus_path=Path("CODEFORGE/codeforge/kb/skills_corpus.jsonl"),
    )

    # ----------------------------------------------------------------
    # STEP 1: Reset hard task
    # ----------------------------------------------------------------
    print("[STEP 1] Reset hard task")
    print("> No KB available. No interrogation. Just going to code it.\n")

    result = server.handle_tool("codeforge_reset", {"task_level": "hard"})
    obs = result["observation"]
    session_id = result["session_id"]

    print(f"  Task:    {obs['task_id']}")
    print(f"  Level:   {obs['task_level']}")
    print(f"  Brief:   {obs['task_brief']}")
    print(f"  Budget:  {obs['budget_remaining']}")
    print(f"  Files:   {list(obs['initial_files'].keys())}")
    print()

    # ----------------------------------------------------------------
    # STEP 2: Submit immediately with a subtle bug (wrong greeting)
    # ----------------------------------------------------------------
    print("-" * 60)
    print("[STEP 2] Submit immediately (no research)")
    print('> Thinking: I\'m pretty sure greet() should return "Hi, {name}!"')
    print("  Confidence: 0.99 (I'm sure this is right)")
    print()

    # The LLM's buggy code: says "Hi" instead of "Hello"
    buggy_core = (
        'from __future__ import annotations\n\n\n'
        'def greet(name: str) -> str:\n'
        '    """Return a greeting for *name*."""\n'
        '    return f"Hi, {name}!"\n'
    )
    buggy_test = (
        'from __future__ import annotations\n\n'
        'from core import greet\n\n\n'
        'def test_greet_alice() -> None:\n'
        '    assert greet("Alice") == "Hi, Alice!"\n\n\n'
        'def test_greet_returns_str() -> None:\n'
        '    assert isinstance(greet("Bob"), str)\n'
    )
    main_py = (
        'from __future__ import annotations\n\nfrom core import greet\n\n\n'
        'if __name__ == "__main__":\n'
        '    print(greet("World"))\n'
    )

    result2 = server.handle_tool("codeforge_submit", {
        "session_id": session_id,
        "files": {
            "main.py": main_py,
            "core.py": buggy_core,
            "test_core.py": buggy_test,
        },
        "confidence": 0.99,
    })
    obs2 = result2["observation"]

    print(f"> Calling: codeforge_submit(files={{...}}, confidence=0.99)")
    print(f"  Reward:          {obs2['last_reward']}")
    print(f"  Previous score:  {obs2['previous_score']}")
    print(f"  Budget left:     {obs2['budget_remaining']}")
    print(f"  Done:            {obs2['is_done']}")
    print()
    print('  PROBLEM: Hidden tests expected "Hello, Alice!" but I wrote "Hi, Alice!"')
    print("  The sandbox caught my semantic error via hidden test suite.")
    print("  Without CodeForge's KB, I had no guidance to check the exact spec.")
    if obs2.get("last_grounding"):
        print(f"  Grounding: {json.dumps(obs2['last_grounding'], indent=4)}")
    print()

    # ----------------------------------------------------------------
    # STEP 3: Fix and resubmit with correct code
    # ----------------------------------------------------------------
    print("-" * 60)
    print("[STEP 3] Fix and resubmit")
    print('> Changing to "Hello, {name}!" and updating test')
    print("  Confidence: 0.95 (still overconfident -- no KB research done)")
    print()

    correct_core = (
        'from __future__ import annotations\n\n\n'
        'def greet(name: str) -> str:\n'
        '    """Return a greeting for *name*."""\n'
        '    return f"Hello, {name}!"\n'
    )
    correct_test = (
        'from __future__ import annotations\n\n'
        'from core import greet\n\n\n'
        'def test_greet_alice() -> None:\n'
        '    assert greet("Alice") == "Hello, Alice!"\n\n\n'
        'def test_greet_bob() -> None:\n'
        '    assert greet("Bob") == "Hello, Bob!"\n\n\n'
        'def test_greet_returns_str() -> None:\n'
        '    assert isinstance(greet("X"), str)\n\n\n'
        'def test_greet_empty() -> None:\n'
        '    assert greet("") == "Hello, !"\n'
    )

    result3 = server.handle_tool("codeforge_submit", {
        "session_id": session_id,
        "files": {
            "main.py": main_py,
            "core.py": correct_core,
            "test_core.py": correct_test,
        },
        "confidence": 0.95,
    })
    obs3 = result3["observation"]

    print(f"> Calling: codeforge_submit(files={{...}}, confidence=0.95)")
    print(f"  Reward:          {obs3['last_reward']}")
    print(f"  Previous score:  {obs3['previous_score']}")
    print(f"  Budget left:     {obs3['budget_remaining']}")
    print(f"  Done:            {obs3['is_done']}")
    if obs3.get("last_grounding"):
        print(f"  Grounding: {json.dumps(obs3['last_grounding'], indent=4)}")
    print()

    # ----------------------------------------------------------------
    # STEP 4: No audit trail
    # ----------------------------------------------------------------
    print("-" * 60)
    print("[STEP 4] No audit trail")
    print("> Without the MCP workflow, I never called get_audit.")
    print("  I can't trace WHY my first submission failed,")
    print("  what the Brier penalty was, or what citations I missed.")
    print("  I just guessed and retried.")
    print()

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    print(DIVIDER)
    print("  SUMMARY (WITHOUT MCP)")
    print(DIVIDER)
    print()
    print(f"  Actions taken:     3 (reset + 2 submits)")
    print(f"  Budget used:       {10 - obs3['budget_remaining']}/10")
    print(f"  First reward:      {obs2['last_reward']} (buggy code, overconfident)")
    print(f"  Final reward:      {obs3['last_reward']}")
    print(f"  Task completed:    {obs3['is_done']}")
    print(f"  Wasted actions:    1 (the failed submit)")
    print()
    print("  Key problems:")
    print("    1. No KB research -> wrong implementation on first try")
    print("    2. Overconfidence (0.99) on buggy code -> Brier penalty")
    print("    3. No interrogation -> missed edge cases")
    print("    4. No audit trail -> can't trace decisions or learn from mistakes")
    print("    5. No cluster browsing -> no awareness of related patterns")
    print()
    print("  Contrast with MCP agent:")
    print("    - MCP agent queries KB BEFORE writing code")
    print("    - MCP agent interrogates to discover edge cases")
    print("    - MCP agent uses honest confidence (calibrated)")
    print("    - MCP agent has full audit trail for every decision")
    print("    - MCP agent gets it right on the first submit")
    print()
    print(DIVIDER)


if __name__ == "__main__":
    main()
