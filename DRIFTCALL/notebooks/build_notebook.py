"""Notebook builder for DriftCall — compiles cells/ into train_driftcall.ipynb.

Contract (CLAUDE.md §5):
    * Read ``cells/step_NN_*.py`` in numeric ascending order of ``NN``.
    * For each ``.py``, if a sibling ``cells/step_NN_<name>.md`` exists,
      emit it as a **markdown** cell *before* the code cell.
    * Emit each ``.py`` verbatim as a **code** cell (no transformation).
    * Write the result to ``notebooks/train_driftcall.ipynb``.
    * Re-runs are byte-identical given the same inputs (deterministic order,
      no volatile metadata such as timestamps or kernel specs).

The builder is deliberately simple: it does not evaluate cells, strip comments,
reformat source, or add boilerplate. It is a pure compilation from the hand-
authored ``cells/`` tree to a single ``.ipynb`` artifact.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Final

import jupytext
import nbformat

_nb_v4: Any = nbformat.v4
_jupytext_write: Any = jupytext.write

_STEP_FILENAME: Final[re.Pattern[str]] = re.compile(
    r"^step_(?P<num>\d+)_(?P<name>[A-Za-z0-9_]+)\.py$"
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_CELLS_DIR: Final[Path] = _REPO_ROOT / "cells"
DEFAULT_OUTPUT_PATH: Final[Path] = _REPO_ROOT / "notebooks" / "train_driftcall.ipynb"

_NOTEBOOK_METADATA: Final[dict[str, object]] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
    },
}


def _discover_cells(cells_dir: Path) -> list[tuple[int, Path, Path | None]]:
    """Return ``(step_num, py_path, md_path_or_none)`` tuples, sorted by step."""
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"cells directory not found: {cells_dir}")

    entries: list[tuple[int, Path, Path | None]] = []
    for py_path in cells_dir.iterdir():
        if not py_path.is_file():
            continue
        match = _STEP_FILENAME.match(py_path.name)
        if match is None:
            continue
        step_num = int(match["num"])
        md_candidate = py_path.with_suffix(".md")
        md_path = md_candidate if md_candidate.is_file() else None
        entries.append((step_num, py_path, md_path))

    entries.sort(key=lambda item: (item[0], item[1].name))
    return entries


def _build_notebook(cells_dir: Path) -> Any:
    notebook = _nb_v4.new_notebook()
    notebook["metadata"] = dict(_NOTEBOOK_METADATA)

    cells: list[Any] = []
    for step, py_path, md_path in _discover_cells(cells_dir):
        if md_path is not None:
            md_source = md_path.read_text(encoding="utf-8")
            md_cell = _nb_v4.new_markdown_cell(md_source)
            md_cell["id"] = _stable_cell_id(step, "md", md_source)
            cells.append(md_cell)
        py_source = py_path.read_text(encoding="utf-8")
        code_cell = _nb_v4.new_code_cell(py_source)
        code_cell["id"] = _stable_cell_id(step, "py", py_source)
        cells.append(code_cell)

    # Strip volatile per-cell metadata so the output is deterministic.
    for cell in cells:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["cells"] = cells
    return notebook


def _stable_cell_id(step: int, kind: str, source: str) -> str:
    """Deterministic 8-char cell id — ensures byte-identical rebuilds."""
    digest = hashlib.sha256(f"{step}:{kind}:{source}".encode()).hexdigest()
    return digest[:8]


def build(
    cells_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Compile ``cells/`` into a Colab-ready notebook.

    Parameters
    ----------
    cells_dir:
        Directory containing ``step_NN_*.py`` (and optional ``.md``) files.
        Defaults to ``<repo>/cells``.
    output_path:
        Where to write the ``.ipynb``. Defaults to
        ``<repo>/notebooks/train_driftcall.ipynb``.

    Returns
    -------
    Path
        The absolute path of the written notebook.
    """
    cells_dir = (cells_dir or DEFAULT_CELLS_DIR).resolve()
    output_path = (output_path or DEFAULT_OUTPUT_PATH).resolve()

    notebook = _build_notebook(cells_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use jupytext for the write so the notebook is round-trippable
    # with the ``.py`` percent format if needed downstream.
    _jupytext_write(notebook, str(output_path), fmt="ipynb")
    return output_path


def main() -> None:
    """CLI entrypoint. Prints the built notebook path on success."""
    path = build()
    print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``.
    main()
