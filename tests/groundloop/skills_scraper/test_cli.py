from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.skills_scraper.cli import _load_sources_yaml, main


def test_cli_with_yaml_sources(fixtures_dir: Path, tmp_path: Path) -> None:
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "sources:\n"
        f"  - label: fake\n"
        f"    glob: {fixtures_dir}/**/SKILL.md\n"
    )
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(sources_yaml),
        "--output", str(out),
    ])
    assert exit_code == 0
    assert out.exists()


@pytest.mark.skipif(
    not Path("~/.claude/skills").expanduser().exists(),
    reason="requires installed Claude Code skills",
)
def test_cli_sources_default(tmp_path: Path) -> None:
    # Smoke test: --sources default runs against the real system; exit 0 iff
    # the user has any installed SKILL.md (expected on this dev machine).
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", "default",
        "--output", str(out),
    ])
    assert exit_code == 0
    assert out.exists()


def test_load_sources_yaml_bad_schema_raises_typeerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_sources_dict: true\n")
    with pytest.raises(TypeError):
        _load_sources_yaml(bad)


def test_load_sources_yaml_non_list_raises_typeerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("sources: not-a-list\n")
    with pytest.raises(TypeError):
        _load_sources_yaml(bad)


def test_cli_sources_missing_file_exits_1(tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(tmp_path / "does_not_exist.yaml"),
        "--output", str(out),
    ])
    assert exit_code == 1


def test_cli_zero_nodes_nonzero_exit(tmp_path: Path) -> None:
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text(
        "sources:\n"
        f"  - label: nada\n"
        f"    glob: {tmp_path}/nonexistent/*.md\n"
    )
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(empty_yaml),
        "--output", str(out),
    ])
    assert exit_code == 1
