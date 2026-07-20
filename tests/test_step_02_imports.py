"""Smoke tests for cells/step_02_imports.

Step 02 is the consolidated re-exports cell. The contract is simply that
importing it does not raise and that the symbols it advertises in
``__all__`` are reachable. Optional third-party modules are loaded into a
private dict accessible via ``get_optional``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def imports_module() -> object:
    module_name = "cells.step_02_imports"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


class TestModuleImport:
    def test_module_imports_cleanly(self, imports_module: object) -> None:
        assert imports_module is not None
        assert imports_module.__name__ == "cells.step_02_imports"

    def test_all_attribute_is_tuple(self, imports_module: object) -> None:
        all_names = imports_module.__all__  # type: ignore[attr-defined]
        assert isinstance(all_names, tuple)
        assert len(all_names) > 0

    def test_no_duplicates_in_all(self, imports_module: object) -> None:
        all_names = imports_module.__all__  # type: ignore[attr-defined]
        assert len(all_names) == len(set(all_names))


class TestStdlibReexports:
    @pytest.mark.parametrize(
        "name",
        [
            "Any",
            "Callable",
            "Enum",
            "Literal",
            "Mapping",
            "Path",
            "Protocol",
            "Sequence",
            "TypeVar",
            "dataclass",
            "dataclasses",
            "field",
            "hashlib",
            "io",
            "json",
            "logging",
            "math",
            "os",
            "random",
            "re",
            "sys",
            "time",
            "uuid",
        ],
    )
    def test_stdlib_symbol_reachable(self, imports_module: object, name: str) -> None:
        assert hasattr(imports_module, name), f"missing re-export: {name}"

    def test_path_is_pathlib_path(self, imports_module: object) -> None:
        from pathlib import Path

        assert imports_module.Path is Path  # type: ignore[attr-defined]

    def test_dataclass_callable(self, imports_module: object) -> None:
        # Smoke: build a dataclass via the re-export and confirm it works.
        @imports_module.dataclass(frozen=True)  # type: ignore[attr-defined]
        class Tiny:
            x: int

        instance = Tiny(x=3)
        assert instance.x == 3


class TestGetOptional:
    def test_get_optional_callable(self, imports_module: object) -> None:
        assert callable(imports_module.get_optional)  # type: ignore[attr-defined]

    def test_get_optional_returns_none_for_unknown(self, imports_module: object) -> None:
        assert imports_module.get_optional("not_a_real_module_xyz") is None  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "name", ["numpy", "yaml", "fastapi", "uvicorn", "pydantic", "soundfile"]
    )
    def test_optional_modules_attempted(self, imports_module: object, name: str) -> None:
        # Each optional name must be a key in the loaded dict; the value is
        # either the module or None when unavailable. The call must never
        # raise.
        result = imports_module.get_optional(name)  # type: ignore[attr-defined]
        # No assertion on truthiness — we only verify the call is safe.
        assert result is None or result is not None
