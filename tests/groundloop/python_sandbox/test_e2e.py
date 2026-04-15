from __future__ import annotations

from groundloop.python_sandbox import run_sandbox


def test_e2e_files_dict_end_to_end() -> None:
    files = {"main.py": "from __future__ import annotations\n\ndef f() -> int:\n    return 1\n"}
    r = run_sandbox(files=files, tools=("imports",))
    assert r.composite_score >= 0.9
