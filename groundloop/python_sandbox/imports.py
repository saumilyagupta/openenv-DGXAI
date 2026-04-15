from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path

from groundloop.python_sandbox.models import ImportReport

_log = logging.getLogger(__name__)


def _top_level(name: str) -> str:
    return name.split(".")[0]


def _extract_imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(_top_level(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            out.add(_top_level(node.module))
    return out


def _local_modules(project_dir: Path) -> set[str]:
    local: set[str] = set()
    for py in project_dir.rglob("*.py"):
        if py.name == "__init__.py":
            local.add(py.parent.name)
        else:
            local.add(py.stem)
    return local


def scan_imports(project_dir: Path) -> ImportReport:
    by_file: dict[str, tuple[str, ...]] = {}
    all_pkgs: set[str] = set()
    total = 0

    for py in sorted(project_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as e:
            _log.warning("imports: parse error %s: %s", py, e)
            by_file[str(py.relative_to(project_dir))] = ("__parse_error__",)
            continue
        pkgs = _extract_imports(tree)
        total += len(pkgs)
        by_file[str(py.relative_to(project_dir))] = tuple(sorted(pkgs))
        all_pkgs.update(pkgs)

    local = _local_modules(project_dir)
    unresolved = tuple(
        sorted(
            p for p in all_pkgs
            if p not in local and importlib.util.find_spec(p) is None
        )
    )
    return ImportReport(total=total, unresolved=unresolved, by_file=by_file)
