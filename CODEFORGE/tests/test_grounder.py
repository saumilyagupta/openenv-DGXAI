from __future__ import annotations

import pytest

from codeforge.grounder import ground


class TestBasicImportResolution:
    """Test basic import/from-import resolution."""

    def test_import_os_is_grounded(self) -> None:
        report = ground("import os\n")
        assert report.groundedness == 1.0
        assert report.total_symbols == 1
        assert len(report.grounded) == 1
        assert len(report.ungrounded) == 0
        assert report.grounded[0].module == "os"
        assert report.grounded[0].resolved is True

    def test_unresolved_import(self) -> None:
        report = ground("import nonexistent_lib_xyz\n")
        assert report.total_symbols == 1
        assert len(report.ungrounded) == 1
        assert report.ungrounded[0].resolved is False
        assert report.groundedness == 0.0

    def test_from_import_grounded(self) -> None:
        report = ground("from os.path import join\n")
        assert report.total_symbols == 1
        assert len(report.grounded) == 1
        assert report.grounded[0].module == "os.path"
        assert report.grounded[0].attr == "join"
        assert report.grounded[0].resolved is True
        assert report.groundedness == 1.0

    def test_from_import_bad_attr(self) -> None:
        report = ground("from os import nonexistent_fn\n")
        assert report.total_symbols == 1
        assert len(report.ungrounded) == 1
        assert report.ungrounded[0].resolved is False
        assert report.groundedness == 0.0


class TestSyntaxErrorFix:
    """Bug Fix 1: SyntaxError -> groundedness=0.0 (was 1.0)."""

    def test_syntax_error_returns_zero_groundedness(self) -> None:
        report = ground("def foo(:\n")  # invalid syntax
        assert report.groundedness == 0.0
        assert report.total_symbols == 0
        assert len(report.grounded) == 0
        assert len(report.ungrounded) == 0

    def test_syntax_error_multiline(self) -> None:
        report = ground("if True\n    pass\n")  # missing colon
        assert report.groundedness == 0.0


class TestZeroSymbolsFix:
    """Bug Fix 2: Zero symbols -> groundedness=0.5 (was 1.0)."""

    def test_no_imports_returns_half(self) -> None:
        report = ground("x = 1 + 2\nprint(x)\n")
        assert report.groundedness == 0.5
        assert report.total_symbols == 0

    def test_empty_source_returns_half(self) -> None:
        report = ground("")
        assert report.groundedness == 0.5
        assert report.total_symbols == 0

    def test_only_builtins_returns_half(self) -> None:
        report = ground("result = len([1, 2, 3])\nprint(result)\n")
        assert report.groundedness == 0.5
        assert report.total_symbols == 0


class TestDeepAttributeResolutionFix:
    """Bug Fix 3: Attribute resolution uses FULL module path."""

    def test_os_path_join_resolves_correctly(self) -> None:
        source = "import os.path\nos.path.join('a', 'b')\n"
        report = ground(source)
        # Should find the import symbol + the attribute usage
        attrs = [s for s in report.grounded if s.kind == "attribute"]
        # os.path.join should resolve against os.path (which has join)
        assert len(attrs) >= 1
        join_sym = attrs[0]
        assert join_sym.attr == "join"
        assert join_sym.resolved is True

    def test_os_path_nonexistent_ungrounded(self) -> None:
        source = "import os.path\nos.path.nonexistent_attr_xyz()\n"
        report = ground(source)
        attrs = [s for s in report.ungrounded if s.kind == "attribute"]
        assert len(attrs) >= 1
        assert attrs[0].attr == "nonexistent_attr_xyz"
        assert attrs[0].resolved is False


class TestRelativeImportsSkipped:
    """Relative imports (level != 0) are skipped."""

    def test_relative_import_skipped(self) -> None:
        # Relative imports can't be resolved without package context
        source = "from . import something\n"
        report = ground(source)
        # Relative imports should be skipped, so no symbols
        assert report.total_symbols == 0
        assert report.groundedness == 0.5


class TestMixedSymbols:
    """Mixed grounded/ungrounded gives correct ratio."""

    def test_mixed_ratio(self) -> None:
        source = "import os\nimport nonexistent_lib_abc\n"
        report = ground(source)
        assert report.total_symbols == 2
        assert len(report.grounded) == 1
        assert len(report.ungrounded) == 1
        assert report.groundedness == pytest.approx(0.5)

    def test_two_grounded_one_ungrounded(self) -> None:
        source = "import os\nimport sys\nimport nonexistent_lib_abc\n"
        report = ground(source)
        assert report.total_symbols == 3
        assert len(report.grounded) == 2
        assert len(report.ungrounded) == 1
        assert report.groundedness == pytest.approx(2.0 / 3.0)


class TestSymbolKinds:
    """Verify Symbol fields are set correctly."""

    def test_import_symbol_kind(self) -> None:
        report = ground("import os\n")
        sym = report.grounded[0]
        assert sym.kind == "import"
        assert sym.module == "os"
        assert sym.attr is None
        assert sym.line == 1

    def test_attribute_symbol_kind(self) -> None:
        source = "import os\nos.getcwd()\n"
        report = ground(source)
        attrs = [s for s in report.grounded if s.kind == "attribute"]
        assert len(attrs) == 1
        assert attrs[0].module == "os"
        assert attrs[0].attr == "getcwd"
        assert attrs[0].line == 2

    def test_from_import_symbol_kind(self) -> None:
        report = ground("from os.path import join\n")
        sym = report.grounded[0]
        assert sym.kind == "import"
        assert sym.module == "os.path"
        assert sym.attr == "join"


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_aliased_import(self) -> None:
        source = "import os as operating_system\n"
        report = ground(source)
        assert report.total_symbols == 1
        assert len(report.grounded) == 1

    def test_multiple_from_imports(self) -> None:
        source = "from os.path import join, exists, isfile\n"
        report = ground(source)
        assert report.total_symbols == 3
        assert len(report.grounded) == 3
        assert report.groundedness == 1.0

    def test_import_from_none_module_skipped(self) -> None:
        # from-import with module=None (e.g., `from import x` — not valid,
        # but if somehow AST produces it, skip gracefully)
        # This is tested via relative import which has level != 0
        source = "from . import foo\n"
        report = ground(source)
        assert report.total_symbols == 0

    def test_groundedness_report_is_frozen(self) -> None:
        report = ground("import os\n")
        with pytest.raises(Exception):
            report.groundedness = 0.0  # type: ignore[misc]
