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


def test_cli_build_cache_hit(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    assert rc == 0
    assert "cache hit" in capsys.readouterr().out


def test_cli_search_text_format(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main([
        "search", "pytest fixtures",
        "--corpus", str(tiny_corpus_path),
        "--cache", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "python-testing" in out
    assert "score=" in out


def test_cli_search_missing_corpus_exits_1(tmp_path: Path) -> None:
    rc = main([
        "search", "q",
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--cache", str(tmp_path / "c.pkl"),
    ])
    assert rc == 1


def test_cli_stats_missing_corpus_exits_1(tmp_path: Path) -> None:
    rc = main([
        "stats",
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--cache", str(tmp_path / "c.pkl"),
    ])
    assert rc == 1


def test_cli_build_force_rebuilds(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    cache = tmp_path / "idx.pkl"
    main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache)])
    capsys.readouterr()
    rc = main(["build", "--corpus", str(tiny_corpus_path), "--cache", str(cache), "--force"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cache hit" not in out
    assert "built:" in out


def test_cli_cluster(tiny_corpus_path: Path, tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "cluster_manifest.json"
    rc = main([
        "cluster",
        "--corpus", str(tiny_corpus_path),
        "--manifest", str(manifest_path),
        "--threshold", "0.1",
    ])
    assert rc == 0
    assert manifest_path.exists()
    m = json.loads(manifest_path.read_text())
    assert m["total_nodes_clustered"] >= 1
    assert m["corpus_sha256"]


def test_cli_cluster_missing_corpus(tmp_path: Path) -> None:
    rc = main([
        "cluster",
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--manifest", str(tmp_path / "m.json"),
    ])
    assert rc == 1


def test_cli_cluster_malformed_corpus(tmp_path: Path, capsys) -> None:
    bad_corpus = tmp_path / "bad.jsonl"
    bad_corpus.write_text('{"id": "n1"}\nnot-json\n', encoding="utf-8")
    rc = main([
        "cluster",
        "--corpus", str(bad_corpus),
        "--manifest", str(tmp_path / "m.json"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed JSON" in err
    assert ":2" in err
