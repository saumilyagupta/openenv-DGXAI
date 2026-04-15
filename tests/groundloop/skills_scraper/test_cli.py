from __future__ import annotations

from pathlib import Path

from groundloop.skills_scraper.cli import main


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
