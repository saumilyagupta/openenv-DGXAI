from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def clean_project(tmp_path: Path) -> Path:
    src = Path(__file__).parent / "fixtures" / "clean_project"
    dst = tmp_path / "clean_project"
    import shutil
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def broken_project(tmp_path: Path) -> Path:
    src = Path(__file__).parent / "fixtures" / "broken_project"
    dst = tmp_path / "broken_project"
    import shutil
    shutil.copytree(src, dst)
    return dst
