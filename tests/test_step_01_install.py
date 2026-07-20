"""Smoke tests for cells/step_01_install.

Step 01 is the dependency-install bootstrap. In CI we only verify that the
module imports cleanly without invoking pip/network calls and that its
declared public API behaves correctly. The cell itself short-circuits when
``pytest`` is in ``sys.modules`` (see ``_skip_marker`` in the source), so
importing is safe.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def install_module() -> object:
    """Import a fresh copy of the install cell under pytest skip-marker."""

    module_name = "cells.step_01_install"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


class TestModuleImport:
    def test_module_imports_cleanly(self, install_module: object) -> None:
        assert install_module is not None
        assert install_module.__name__ == "cells.step_01_install"

    def test_skip_marker_active_under_pytest(self, install_module: object) -> None:
        # The cell guards its top-level side effects with pytest detection.
        rc = install_module._rc
        assert rc == 0

    def test_public_api_exposed(self, install_module: object) -> None:
        for name in (
            "is_installed",
            "is_colab",
            "pip_install",
            "hf_login_if_token_present",
            "install",
            "REQUIREMENTS_FILENAME",
        ):
            assert hasattr(install_module, name)


class TestIsInstalled:
    def test_returns_true_for_stdlib_alias(self, install_module: object) -> None:
        assert install_module.is_installed("os") is True  # type: ignore[attr-defined]

    def test_returns_false_for_unknown_package(self, install_module: object) -> None:
        assert install_module.is_installed("definitely_not_a_real_pkg_xyz") is False  # type: ignore[attr-defined]

    def test_strips_version_specifiers(self, install_module: object) -> None:
        # A pinned spec should resolve to the bare module name.
        assert install_module.is_installed("os==1.0.0") is True  # type: ignore[attr-defined]

    def test_strips_extras_brackets(self, install_module: object) -> None:
        # ``uvicorn[standard]`` is aliased to ``uvicorn``; if uvicorn is not
        # present this returns False — either way the call must not raise.
        result = install_module.is_installed("uvicorn[standard]")  # type: ignore[attr-defined]
        assert isinstance(result, bool)

    def test_handles_dashes_in_distribution_name(self, install_module: object) -> None:
        # ``faster-whisper`` -> ``faster_whisper``; the bool call must not raise.
        result = install_module.is_installed("faster-whisper")  # type: ignore[attr-defined]
        assert isinstance(result, bool)


class TestIsColab:
    def test_returns_false_outside_colab(self, install_module: object) -> None:
        # Local CI is not Colab; google.colab is not importable here.
        assert install_module.is_colab() is False  # type: ignore[attr-defined]


class TestPipInstall:
    def test_invokes_subprocess_with_pip_install(
        self, install_module: object, tmp_path: Path
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.32.0\n", encoding="utf-8")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            rc = install_module.pip_install(req)  # type: ignore[attr-defined]
        assert rc == 0
        assert mock_run.called
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == sys.executable
        assert "pip" in cmd
        assert "install" in cmd
        assert str(req) in cmd

    def test_propagates_nonzero_return_code(
        self, install_module: object, tmp_path: Path
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("nonsense\n", encoding="utf-8")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            rc = install_module.pip_install(req)  # type: ignore[attr-defined]
        assert rc == 1


class TestHfLoginIfTokenPresent:
    def test_returns_false_when_token_absent(
        self, install_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert install_module.hf_login_if_token_present() is False  # type: ignore[attr-defined]

    def test_returns_true_when_token_present_and_login_succeeds(
        self, install_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_dummytoken")
        fake_login = MagicMock()
        fake_module = MagicMock()
        fake_module.login = fake_login
        with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            assert install_module.hf_login_if_token_present() is True  # type: ignore[attr-defined]
        fake_login.assert_called_once()
        kwargs = fake_login.call_args.kwargs
        assert kwargs["token"] == "hf_dummytoken"
        assert kwargs["add_to_git_credential"] is False


class TestInstall:
    def test_returns_zero_when_requirements_missing(
        self,
        install_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force ``_find_requirements`` to find nothing by chdir'ing to an
        # empty dir AND patching the alt candidate.
        monkeypatch.chdir(tmp_path)
        with patch.object(install_module, "_find_requirements", return_value=None):
            rc = install_module.install()  # type: ignore[attr-defined]
        assert rc == 0

    def test_skips_pip_when_all_pkgs_already_importable(
        self,
        install_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("os\nsys\n# comment line\n\n", encoding="utf-8")
        with (
            patch.object(install_module, "_find_requirements", return_value=req),
            patch.object(install_module, "is_colab", return_value=False),
            patch.object(install_module, "pip_install") as mock_pip,
            patch.object(install_module, "hf_login_if_token_present", return_value=False),
        ):
            rc = install_module.install(force=False)  # type: ignore[attr-defined]
        assert rc == 0
        mock_pip.assert_not_called()

    def test_invokes_pip_when_force_true(
        self,
        install_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("os\n", encoding="utf-8")
        with (
            patch.object(install_module, "_find_requirements", return_value=req),
            patch.object(install_module, "pip_install", return_value=0) as mock_pip,
            patch.object(install_module, "hf_login_if_token_present", return_value=False),
        ):
            rc = install_module.install(force=True)  # type: ignore[attr-defined]
        assert rc == 0
        mock_pip.assert_called_once_with(req)

    def test_invokes_pip_under_colab_runtime(
        self,
        install_module: object,
        tmp_path: Path,
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("os\n", encoding="utf-8")
        with (
            patch.object(install_module, "_find_requirements", return_value=req),
            patch.object(install_module, "is_colab", return_value=True),
            patch.object(install_module, "pip_install", return_value=0) as mock_pip,
            patch.object(install_module, "hf_login_if_token_present", return_value=False),
        ):
            rc = install_module.install(force=False)  # type: ignore[attr-defined]
        assert rc == 0
        mock_pip.assert_called_once_with(req)

    def test_propagates_pip_failure(
        self,
        install_module: object,
        tmp_path: Path,
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("totally_unknown_pkg_xyz\n", encoding="utf-8")
        with (
            patch.object(install_module, "_find_requirements", return_value=req),
            patch.object(install_module, "is_colab", return_value=False),
            patch.object(install_module, "pip_install", return_value=2) as mock_pip,
            patch.object(install_module, "hf_login_if_token_present") as mock_hf,
        ):
            rc = install_module.install(force=True)  # type: ignore[attr-defined]
        assert rc == 2
        mock_pip.assert_called_once()
        mock_hf.assert_not_called()


class TestFindRequirements:
    def test_returns_path_when_file_exists_in_cwd(
        self,
        install_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("os\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = install_module._find_requirements()  # type: ignore[attr-defined]
        assert result == req

    def test_returns_none_when_no_requirements_anywhere(
        self,
        install_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.object(install_module, "__file__", str(tmp_path / "nope.py")):
            result = install_module._find_requirements()  # type: ignore[attr-defined]
        assert result is None


class TestRequirementsConstant:
    def test_filename_is_requirements_txt(self, install_module: object) -> None:
        assert install_module.REQUIREMENTS_FILENAME == "requirements.txt"  # type: ignore[attr-defined]
