from __future__ import annotations

from pathlib import Path

from groundloop.python_sandbox.imports import scan_imports


def test_clean_project_has_no_unresolved(clean_project: Path) -> None:
    r = scan_imports(clean_project)
    assert r.unresolved == ()
    assert r.total >= 1


def test_broken_project_flags_nonexistent(broken_project: Path) -> None:
    r = scan_imports(broken_project)
    assert "nonexistent_zzz" in r.unresolved


def test_scan_empty_dir(tmp_path: Path) -> None:
    r = scan_imports(tmp_path)
    assert r.total == 0
    assert r.unresolved == ()


def test_scan_ignores_relative_imports(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("from . import b\n")
    r = scan_imports(tmp_path)
    assert r.unresolved == ()
