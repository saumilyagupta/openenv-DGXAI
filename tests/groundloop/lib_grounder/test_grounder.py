from __future__ import annotations

from groundloop.lib_grounder.grounder import ground


def test_clean_import_resolves() -> None:
    r = ground("import os\n")
    assert r.groundedness == 1.0
    assert len(r.grounded) == 1


def test_bogus_package_unresolved() -> None:
    r = ground("import nonexistent_zzz_pkg_987\n")
    assert r.groundedness < 1.0
    assert any(s.module == "nonexistent_zzz_pkg_987" for s in r.ungrounded)


def test_from_import_attribute() -> None:
    r = ground("from os import getcwd\n")
    assert r.groundedness == 1.0


def test_from_import_bogus_attr() -> None:
    r = ground("from os import not_a_real_attr_zzz\n")
    assert r.groundedness < 1.0


def test_attribute_access_resolved() -> None:
    r = ground("import os\nos.getcwd\n")
    assert r.groundedness == 1.0


def test_attribute_access_hallucinated() -> None:
    r = ground("import os\nos.totally_fake_method_xyz\n")
    assert r.groundedness < 1.0


def test_empty_source() -> None:
    r = ground("")
    assert r.groundedness == 1.0
    assert r.total_symbols == 0


def test_syntax_error_returns_empty() -> None:
    r = ground("def (")
    assert r.total_symbols == 0
    assert r.groundedness == 1.0


def test_relative_import_skipped() -> None:
    r = ground("from . import foo\n")
    assert r.total_symbols == 0
