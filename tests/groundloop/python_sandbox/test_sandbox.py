from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.python_sandbox.sandbox import run_sandbox


def test_sandbox_clean_project_scores_high(clean_project: Path) -> None:
    r = run_sandbox(project_dir=clean_project, tools=("ruff", "imports"))
    assert r.composite_score >= 0.9
    assert r.imports.unresolved == ()


def test_sandbox_broken_project_scores_low(broken_project: Path) -> None:
    r = run_sandbox(project_dir=broken_project)
    assert r.composite_score <= 0.5
    assert "nonexistent_zzz" in r.imports.unresolved


def test_sandbox_files_dict_creates_tmp(tmp_path: Path) -> None:
    files = {"main.py": "x = 1\n"}
    r = run_sandbox(files=files, tools=("imports",))
    assert r.imports.total == 0


def test_sandbox_requires_one_of_inputs() -> None:
    with pytest.raises(ValueError):
        run_sandbox()
