from __future__ import annotations

import ast
from typing import Any

import networkx as nx  # type: ignore[import-untyped]


def build_code_graph(files: dict[str, str]) -> nx.DiGraph:
    """Build a structural graph from Python source files.

    Nodes: modules, functions, classes
    Edges: imports, calls, inheritance, exports
    """
    g: nx.DiGraph = nx.DiGraph()
    for filename, source in files.items():
        module = filename.removesuffix(".py")
        g.add_node(module, kind="module")
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fqn = f"{module}.{node.name}"
                g.add_node(fqn, kind="function", line=node.lineno)
                g.add_edge(module, fqn, relation="exports")
            elif isinstance(node, ast.ClassDef):
                fqn = f"{module}.{node.name}"
                g.add_node(fqn, kind="class", line=node.lineno)
                g.add_edge(module, fqn, relation="exports")
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        g.add_edge(fqn, base.id, relation="inherits")
            elif isinstance(node, ast.ImportFrom) and node.module:
                g.add_edge(module, node.module, relation="imports")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    g.add_edge(module, alias.name, relation="imports")
    return g


def query_graph(g: nx.DiGraph, question: str) -> list[dict[str, Any]]:
    """Structural queries on the code graph.

    Supported question prefixes:
    - "exports_of <module>" -- functions/classes exported by module
    - "imports_of <module>" -- modules imported by module
    - "dependents_of <module>" -- modules that import this module
    - "all_modules" -- list all module nodes
    - "all_functions" -- list all function nodes
    - "all_classes" -- list all class nodes
    """
    parts = question.strip().split(maxsplit=1)
    if len(parts) < 1 or not parts[0]:
        return []

    cmd = parts[0].lower()
    target = parts[1] if len(parts) > 1 else ""

    if cmd == "exports_of":
        if target not in g:
            return []
        return [
            {"node": n, **g.nodes[n]}
            for n in g.successors(target)
            if g.edges[target, n].get("relation") == "exports"
        ]

    if cmd == "imports_of":
        if target not in g:
            return []
        return [
            {"node": n}
            for n in g.successors(target)
            if g.edges[target, n].get("relation") == "imports"
        ]

    if cmd == "dependents_of":
        if target not in g:
            return []
        return [
            {"node": n}
            for n in g.predecessors(target)
            if g.edges[n, target].get("relation") == "imports"
        ]

    if cmd == "all_modules":
        return [
            {"node": n, **d}
            for n, d in g.nodes(data=True)
            if d.get("kind") == "module"
        ]

    if cmd == "all_functions":
        return [
            {"node": n, **d}
            for n, d in g.nodes(data=True)
            if d.get("kind") == "function"
        ]

    if cmd == "all_classes":
        return [
            {"node": n, **d}
            for n, d in g.nodes(data=True)
            if d.get("kind") == "class"
        ]

    return []
