from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Symbol(BaseModel):
    """A single symbol extracted from source code by AST walking."""

    model_config = ConfigDict(frozen=True)
    module: str
    attr: str | None
    kind: Literal["import", "attribute"]
    resolved: bool
    line: int


class GroundingReport(BaseModel):
    """Result of grounding analysis on source code."""

    model_config = ConfigDict(frozen=True)
    total_symbols: int
    grounded: tuple[Symbol, ...]
    ungrounded: tuple[Symbol, ...]
    groundedness: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_spec(name: str) -> bool:
    """Return True if the module can be found by the import system."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _has_attr(module_name: str, attr: str) -> bool:
    """Check if *module_name* exposes *attr*.

    Uses the FULL module path (e.g. ``os.path``) — not just
    the top-level package.  This is the fix for SYSTEM_DESIGN §4.8.3
    bug #3.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(mod, attr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ground(source: str) -> GroundingReport:
    """AST-parse *source*, check every import and attribute access resolves.

    Three fixes baked in from day one (SYSTEM_DESIGN §4.8.3):
    1. SyntaxError → groundedness=0.0  (was 1.0)
    2. Zero symbols → groundedness=0.5 (was 1.0)
    3. Attribute resolution against full module path (was top-level only)
    """
    # ----- parse --------------------------------------------------------
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # FIX 1: unparseable code → 0.0, not 1.0
        return GroundingReport(
            total_symbols=0,
            grounded=(),
            ungrounded=(),
            groundedness=0.0,
        )

    symbols: list[Symbol] = []
    import_to_module: dict[str, str] = {}

    # ----- walk imports -------------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                resolved = _module_spec(alias.name)
                symbols.append(
                    Symbol(
                        module=alias.name,
                        attr=None,
                        kind="import",
                        resolved=resolved,
                        line=node.lineno,
                    )
                )
                import_to_module[alias.asname or pkg] = alias.name

        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            resolved_mod = _module_spec(node.module)
            for alias in (node.names or []):
                attr_resolved = resolved_mod and _has_attr(node.module, alias.name)
                symbols.append(
                    Symbol(
                        module=node.module,
                        attr=alias.name,
                        kind="import",
                        resolved=attr_resolved,
                        line=node.lineno,
                    )
                )

    # ----- walk attribute accesses --------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue

        # Resolve the chain: e.g. os.path.join → base="os", chain=["path"], attr="join"
        chain: list[str] = []
        cursor: ast.expr = node.value
        while isinstance(cursor, ast.Attribute):
            chain.append(cursor.attr)
            cursor = cursor.value
        if not isinstance(cursor, ast.Name):
            continue

        base = cursor.id
        mod_name = import_to_module.get(base)
        if mod_name is None:
            continue

        # Build the full module path for chained access:
        # import os.path → import_to_module["os"] = "os.path"
        # os.path.join → chain=["path"], we need to resolve "join" against "os.path"
        # The chain intermediates are sub-module parts already covered by mod_name.
        # We check the final attr against the deepest resolvable module.
        if chain:
            # chain was built bottom-up, reverse to get top-down order
            chain.reverse()
            # Build candidate module: mod_name + chain parts
            full_mod = mod_name + "." + ".".join(chain)
            # Try the full module first; fall back to mod_name if it doesn't exist
            check_mod = full_mod if _module_spec(full_mod) else mod_name
        else:
            check_mod = mod_name

        # FIX 3: resolve against full module path, not just top-level
        resolved = _has_attr(check_mod, node.attr)
        symbols.append(
            Symbol(
                module=check_mod,
                attr=node.attr,
                kind="attribute",
                resolved=resolved,
                line=node.lineno,
            )
        )

    # ----- compute groundedness -----------------------------------------
    grounded = tuple(s for s in symbols if s.resolved)
    ungrounded = tuple(s for s in symbols if not s.resolved)
    total = len(symbols)

    # FIX 2: zero symbols → 0.5 (neutral), not 1.0
    groundedness = 0.5 if total == 0 else len(grounded) / total

    return GroundingReport(
        total_symbols=total,
        grounded=grounded,
        ungrounded=ungrounded,
        groundedness=groundedness,
    )
