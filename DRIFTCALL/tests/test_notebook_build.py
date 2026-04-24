"""Tests for notebooks/build_notebook.py (CLAUDE.md §5)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from notebooks import build_notebook

REPO_ROOT = Path(__file__).resolve().parent.parent
CELLS_DIR = REPO_ROOT / "cells"
DEFAULT_OUT = REPO_ROOT / "notebooks" / "train_driftcall.ipynb"

_STEP_PY = re.compile(r"^step_(\d+)_.+\.py$")


def _py_cells() -> list[Path]:
    return sorted(
        (p for p in CELLS_DIR.iterdir() if _STEP_PY.match(p.name)),
        key=lambda p: p.name,
    )


def _step_number(path: Path) -> int:
    match = _STEP_PY.match(path.name)
    assert match is not None, f"unexpected cell filename {path.name}"
    return int(match.group(1))


@pytest.fixture(autouse=True)
def _clean_ipynb() -> None:
    if DEFAULT_OUT.exists():
        DEFAULT_OUT.unlink()


def test_build_creates_ipynb() -> None:
    out = build_notebook.build()
    assert out == DEFAULT_OUT
    assert DEFAULT_OUT.exists()
    data = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert "cells" in data
    assert isinstance(data["cells"], list)


def test_cell_count() -> None:
    build_notebook.build()
    data = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    py_sources = _py_cells()
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == len(py_sources)


def test_markdown_pairing() -> None:
    build_notebook.build()
    data = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    cells = data["cells"]
    py_sources = _py_cells()

    md_pairs: dict[int, Path] = {}
    for py in py_sources:
        md = py.with_suffix(".md")
        if md.exists():
            md_pairs[_step_number(py)] = md

    for py in py_sources:
        step = _step_number(py)
        idx = next(
            (
                i
                for i, c in enumerate(cells)
                if c["cell_type"] == "code" and py.read_text(encoding="utf-8") in "".join(c["source"])
            ),
            None,
        )
        assert idx is not None, f"code cell for step {step} not found"
        if step in md_pairs:
            assert idx > 0, f"markdown expected before code cell for step {step}"
            prev = cells[idx - 1]
            assert prev["cell_type"] == "markdown"
            md_text = md_pairs[step].read_text(encoding="utf-8")
            assert md_text in "".join(prev["source"])


def test_numeric_ordering() -> None:
    build_notebook.build()
    data = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    py_sources = _py_cells()
    assert len(code_cells) == len(py_sources)
    for cell, py in zip(code_cells, py_sources, strict=True):
        source = "".join(cell["source"])
        assert py.read_text(encoding="utf-8") in source


def test_idempotent() -> None:
    build_notebook.build()
    first = DEFAULT_OUT.read_bytes()
    DEFAULT_OUT.unlink()
    build_notebook.build()
    second = DEFAULT_OUT.read_bytes()
    assert first == second


def test_missing_md_ok(tmp_path: Path) -> None:
    cells_src = tmp_path / "cells"
    cells_src.mkdir()
    (cells_src / "step_01_only_code.py").write_text(
        '"""Doc."""\n\nprint("hi")\n', encoding="utf-8"
    )
    (cells_src / "step_02_with_md.py").write_text(
        '"""Doc2."""\n\nprint("bye")\n', encoding="utf-8"
    )
    (cells_src / "step_02_with_md.md").write_text("# Paired cell\n", encoding="utf-8")

    out_dir = tmp_path / "notebooks"
    out_path = out_dir / "out.ipynb"
    result = build_notebook.build(cells_dir=cells_src, output_path=out_path)
    assert result == out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    cells = data["cells"]
    # Expect: [code(step01), markdown(step02), code(step02)]
    assert [c["cell_type"] for c in cells] == ["code", "markdown", "code"]


def test_ignores_non_step_files(tmp_path: Path) -> None:
    cells_src = tmp_path / "cells"
    cells_src.mkdir()
    (cells_src / "step_01_main.py").write_text("x = 1\n", encoding="utf-8")
    (cells_src / "__init__.py").write_text("", encoding="utf-8")
    (cells_src / "helpers.py").write_text("y = 2\n", encoding="utf-8")
    (cells_src / "notes.md").write_text("stray\n", encoding="utf-8")

    out_path = tmp_path / "notebooks" / "out.ipynb"
    build_notebook.build(cells_dir=cells_src, output_path=out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 1
    assert "x = 1" in "".join(code_cells[0]["source"])


def test_numeric_sort_not_lexicographic(tmp_path: Path) -> None:
    cells_src = tmp_path / "cells"
    cells_src.mkdir()
    (cells_src / "step_02_b.py").write_text("b = 1\n", encoding="utf-8")
    (cells_src / "step_10_j.py").write_text("j = 1\n", encoding="utf-8")
    (cells_src / "step_01_a.py").write_text("a = 1\n", encoding="utf-8")

    out_path = tmp_path / "notebooks" / "out.ipynb"
    build_notebook.build(cells_dir=cells_src, output_path=out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    code_sources = ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]
    assert "a = 1" in code_sources[0]
    assert "b = 1" in code_sources[1]
    assert "j = 1" in code_sources[2]


def test_main_entrypoint_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_build() -> Path:
        called["hit"] = True
        return DEFAULT_OUT

    monkeypatch.setattr(build_notebook, "build", fake_build)
    build_notebook.main()
    assert called.get("hit") is True
