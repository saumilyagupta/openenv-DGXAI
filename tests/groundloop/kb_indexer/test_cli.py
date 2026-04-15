from __future__ import annotations

import json
from pathlib import Path

from groundloop.kb_indexer.cli import main


def test_cli_build(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    rc = main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    assert rc == 0
    assert cache.exists()


def test_cli_build_missing_corpus_exits_1(tmp_path: Path) -> None:
    rc = main(["build", "--corpus", str(tmp_path / "none.jsonl"), "--cache", str(tmp_path / "c.pkl")])
    assert rc == 1


def test_cli_search_json(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main([
        "search", "pytest fixtures",
        "--corpus", str(tiny_corpus_path),
        "--cache", str(cache),
        "--format", "json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) >= 1
    assert data[0]["skill_name"] == "python-testing"


def test_cli_stats(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main(["stats", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "node_count" in out


def test_cli_search_with_tag_filter(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main([
        "search", "testing",
        "--corpus", str(tiny_corpus_path),
        "--cache", str(cache),
        "--tag", "domain:security",
        "--format", "json",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    for item in data:
        assert "domain:security" in item["tags"]
