from __future__ import annotations

import ast
import importlib
import importlib.util
import logging

from groundloop.lib_grounder.models import GroundingReport, Symbol

_log = logging.getLogger(__name__)


def _module_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_attr(module: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - defensive; any import-time failure = unresolved
        return False
    return hasattr(mod, attr)


def ground(source: str) -> GroundingReport:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return GroundingReport(
            total_symbols=0, grounded=(), ungrounded=(), groundedness=1.0,
        )

    symbols: list[Symbol] = []
    import_to_module: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                resolved = _module_spec(pkg)
                symbols.append(
                    Symbol(
                        module=alias.name, attr=None, kind="import",
                        resolved=resolved, line=node.lineno,
                    )
                )
                import_to_module[alias.asname or pkg] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            resolved_mod = _module_spec(node.module)
            for alias in node.names:
                attr_resolved = resolved_mod and _has_attr(node.module, alias.name)
                symbols.append(
                    Symbol(
                        module=node.module, attr=alias.name, kind="import",
                        resolved=attr_resolved, line=node.lineno,
                    )
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            mod_name = import_to_module.get(base)
            if mod_name is None:
                continue
            top = mod_name.split(".")[0]
            resolved = _has_attr(top, node.attr)
            symbols.append(
                Symbol(
                    module=mod_name, attr=node.attr, kind="attribute",
                    resolved=resolved, line=node.lineno,
                )
            )

    grounded = tuple(s for s in symbols if s.resolved)
    ungrounded = tuple(s for s in symbols if not s.resolved)
    total = len(symbols)
    groundedness = 1.0 if total == 0 else len(grounded) / total
    return GroundingReport(
        total_symbols=total, grounded=grounded, ungrounded=ungrounded,
        groundedness=groundedness,
    )
