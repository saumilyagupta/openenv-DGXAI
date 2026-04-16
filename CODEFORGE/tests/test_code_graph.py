from __future__ import annotations

import pytest
import networkx as nx  # type: ignore[import-untyped]

from codeforge.kb.code_graph import build_code_graph, query_graph


class TestBuildCodeGraph:
    """Tests for build_code_graph()."""

    def test_empty_files_returns_empty_graph(self) -> None:
        g = build_code_graph({})
        assert isinstance(g, nx.DiGraph)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    def test_single_function_creates_node_and_exports_edge(self) -> None:
        files = {"math_utils.py": "def add(a, b):\n    return a + b\n"}
        g = build_code_graph(files)
        assert "math_utils" in g.nodes
        assert g.nodes["math_utils"]["kind"] == "module"
        assert "math_utils.add" in g.nodes
        assert g.nodes["math_utils.add"]["kind"] == "function"
        assert g.nodes["math_utils.add"]["line"] == 1
        assert g.has_edge("math_utils", "math_utils.add")
        assert g.edges["math_utils", "math_utils.add"]["relation"] == "exports"

    def test_class_creates_node_and_exports_edge(self) -> None:
        files = {"shapes.py": "class Circle:\n    pass\n"}
        g = build_code_graph(files)
        assert "shapes.Circle" in g.nodes
        assert g.nodes["shapes.Circle"]["kind"] == "class"
        assert g.nodes["shapes.Circle"]["line"] == 1
        assert g.has_edge("shapes", "shapes.Circle")
        assert g.edges["shapes", "shapes.Circle"]["relation"] == "exports"

    def test_class_inheritance_creates_inherits_edge(self) -> None:
        source = "class Animal:\n    pass\n\nclass Dog(Animal):\n    pass\n"
        files = {"animals.py": source}
        g = build_code_graph(files)
        assert g.has_edge("animals.Dog", "Animal")
        assert g.edges["animals.Dog", "Animal"]["relation"] == "inherits"

    def test_import_from_creates_imports_edge(self) -> None:
        files = {"app.py": "from os.path import join\n"}
        g = build_code_graph(files)
        assert g.has_edge("app", "os.path")
        assert g.edges["app", "os.path"]["relation"] == "imports"

    def test_plain_import_creates_imports_edge(self) -> None:
        files = {"app.py": "import json\nimport os\n"}
        g = build_code_graph(files)
        assert g.has_edge("app", "json")
        assert g.edges["app", "json"]["relation"] == "imports"
        assert g.has_edge("app", "os")
        assert g.edges["app", "os"]["relation"] == "imports"

    def test_syntax_error_file_skipped_but_module_node_added(self) -> None:
        files = {"broken.py": "def foo(:\n"}
        g = build_code_graph(files)
        # Module node is still added before parsing
        assert "broken" in g.nodes
        assert g.nodes["broken"]["kind"] == "module"
        # But no function nodes are extracted
        assert "broken.foo" not in g.nodes

    def test_multiple_files_cross_module_edges(self) -> None:
        files = {
            "core.py": "def greet(name):\n    return f'Hello {name}'\n",
            "app.py": "from core import greet\n\ndef main():\n    greet('world')\n",
        }
        g = build_code_graph(files)
        # core module and its function
        assert "core" in g.nodes
        assert "core.greet" in g.nodes
        # app imports core
        assert g.has_edge("app", "core")
        assert g.edges["app", "core"]["relation"] == "imports"
        # app has main function
        assert "app.main" in g.nodes

    def test_multiple_functions_in_one_file(self) -> None:
        source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        files = {"utils.py": source}
        g = build_code_graph(files)
        assert "utils.foo" in g.nodes
        assert "utils.bar" in g.nodes
        assert g.nodes["utils.foo"]["line"] == 1
        assert g.nodes["utils.bar"]["line"] == 4

    def test_multiple_inheritance_bases(self) -> None:
        source = "class A:\n    pass\nclass B:\n    pass\nclass C(A, B):\n    pass\n"
        files = {"multi.py": source}
        g = build_code_graph(files)
        assert g.has_edge("multi.C", "A")
        assert g.has_edge("multi.C", "B")
        assert g.edges["multi.C", "A"]["relation"] == "inherits"
        assert g.edges["multi.C", "B"]["relation"] == "inherits"


class TestQueryGraph:
    """Tests for query_graph()."""

    @pytest.fixture()
    def sample_graph(self) -> nx.DiGraph:
        files = {
            "core.py": (
                "def greet(name):\n"
                "    return f'Hello {name}'\n\n"
                "class Greeter:\n"
                "    pass\n"
            ),
            "app.py": (
                "from core import greet\n"
                "import json\n\n"
                "def main():\n"
                "    greet('world')\n"
            ),
            "helpers.py": "from core import greet\n\ndef helper():\n    pass\n",
        }
        return build_code_graph(files)

    def test_exports_of_returns_functions_and_classes(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "exports_of core")
        node_names = [r["node"] for r in results]
        assert "core.greet" in node_names
        assert "core.Greeter" in node_names

    def test_exports_of_unknown_module_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "exports_of nonexistent")
        assert results == []

    def test_imports_of_returns_imported_modules(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "imports_of app")
        node_names = [r["node"] for r in results]
        assert "core" in node_names
        assert "json" in node_names

    def test_imports_of_module_with_no_imports(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "imports_of core")
        assert results == []

    def test_dependents_of_returns_importing_modules(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "dependents_of core")
        node_names = [r["node"] for r in results]
        assert "app" in node_names
        assert "helpers" in node_names

    def test_dependents_of_unimported_module(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "dependents_of helpers")
        assert results == []

    def test_all_modules(self, sample_graph: nx.DiGraph) -> None:
        results = query_graph(sample_graph, "all_modules")
        node_names = [r["node"] for r in results]
        assert "core" in node_names
        assert "app" in node_names
        assert "helpers" in node_names
        assert len(node_names) == 3

    def test_all_functions(self, sample_graph: nx.DiGraph) -> None:
        results = query_graph(sample_graph, "all_functions")
        node_names = [r["node"] for r in results]
        assert "core.greet" in node_names
        assert "app.main" in node_names
        assert "helpers.helper" in node_names

    def test_all_classes(self, sample_graph: nx.DiGraph) -> None:
        results = query_graph(sample_graph, "all_classes")
        node_names = [r["node"] for r in results]
        assert "core.Greeter" in node_names
        assert len(node_names) == 1

    def test_unknown_command_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "foobar core")
        assert results == []

    def test_empty_question_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "")
        assert results == []

    def test_whitespace_only_question_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "   ")
        assert results == []

    def test_imports_of_unknown_module_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "imports_of nonexistent")
        assert results == []

    def test_dependents_of_unknown_module_returns_empty(
        self, sample_graph: nx.DiGraph
    ) -> None:
        results = query_graph(sample_graph, "dependents_of nonexistent")
        assert results == []
