from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from codeforge.kb.cluster import build_clusters
from codeforge.kb.indexer import SkillsIndex
from codeforge.kb.models import ClusterManifest, SearchResult


# ---------------------------------------------------------------------------
# Fixture: tiny corpus JSONL
# ---------------------------------------------------------------------------

_TINY_NODES: list[dict[str, object]] = [
    {
        "id": "a001",
        "skill_name": "python-testing",
        "skill_description": "pytest",
        "skill_type": "flexible",
        "section_path": ["Fixtures"],
        "section_title": "Fixtures",
        "section_body": (
            "Use pytest fixtures for setup and teardown. "
            "Scope can be function module or session."
        ),
        "source_path": "/fake/python-testing/SKILL.md",
        "source_root": "user-skills",
        "tags": ["domain:python", "phase:test"],
        "trigger_hints": "pytest",
        "mtime": 0.0,
        "body_hash": "h1",
        "alias_sources": [],
    },
    {
        "id": "b002",
        "skill_name": "api-design",
        "skill_description": "apis",
        "skill_type": None,
        "section_path": ["Endpoints"],
        "section_title": "Endpoints",
        "section_body": (
            "Design REST endpoints with clear resource naming "
            "and proper HTTP verbs."
        ),
        "source_path": "/fake/api-design/SKILL.md",
        "source_root": "user-skills",
        "tags": ["domain:api", "phase:plan"],
        "trigger_hints": "api",
        "mtime": 0.0,
        "body_hash": "h2",
        "alias_sources": [],
    },
    {
        "id": "c003",
        "skill_name": "security-review",
        "skill_description": "security",
        "skill_type": "rigid",
        "section_path": ["Checklist"],
        "section_title": "Checklist",
        "section_body": (
            "Check authentication authorization input validation "
            "secrets and injection risks."
        ),
        "source_path": "/fake/security-review/SKILL.md",
        "source_root": "user-skills",
        "tags": ["domain:security", "phase:review"],
        "trigger_hints": "security",
        "mtime": 0.0,
        "body_hash": "h3",
        "alias_sources": [],
    },
    {
        "id": "d004",
        "skill_name": "python-testing",
        "skill_description": "pytest",
        "skill_type": "flexible",
        "section_path": ["Parametrization"],
        "section_title": "Parametrization",
        "section_body": (
            "Use pytest mark parametrize to run a test with multiple inputs."
        ),
        "source_path": "/fake/python-testing/SKILL.md",
        "source_root": "user-skills",
        "tags": ["domain:python", "phase:test"],
        "trigger_hints": "pytest",
        "mtime": 0.0,
        "body_hash": "h4",
        "alias_sources": [],
    },
    {
        "id": "e005",
        "skill_name": "backend-patterns",
        "skill_description": "backend",
        "skill_type": "flexible",
        "section_path": ["Layers"],
        "section_title": "Layers",
        "section_body": (
            "Separate routes from services from repositories "
            "to enable testing and reuse."
        ),
        "source_path": "/fake/backend-patterns/SKILL.md",
        "source_root": "user-skills",
        "tags": ["domain:backend", "phase:plan"],
        "trigger_hints": "backend",
        "mtime": 0.0,
        "body_hash": "h5",
        "alias_sources": [],
    },
]


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    lines = [json.dumps(n) for n in _TINY_NODES]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture()
def built_index(corpus_path: Path) -> SkillsIndex:
    idx = SkillsIndex(corpus_path=corpus_path)
    idx.build()
    return idx


# ---------------------------------------------------------------------------
# build() tests
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_loads_nodes(self, built_index: SkillsIndex) -> None:
        stats = built_index.stats()
        assert stats["node_count"] == 5

    def test_build_missing_corpus_raises(self, tmp_path: Path) -> None:
        idx = SkillsIndex(corpus_path=tmp_path / "missing.jsonl")
        with pytest.raises(FileNotFoundError):
            idx.build()

    def test_build_empty_corpus(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        idx = SkillsIndex(corpus_path=p)
        idx.build()
        assert idx.stats()["node_count"] == 0


# ---------------------------------------------------------------------------
# search() tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_results(self, built_index: SkillsIndex) -> None:
        results = built_index.search("pytest fixtures")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranks_relevant_first(
        self, built_index: SkillsIndex,
    ) -> None:
        results = built_index.search("pytest fixtures", top_k=3)
        # The python-testing Fixtures node should score highest
        assert results[0].skill_name == "python-testing"
        assert "Fixtures" in results[0].section_path

    def test_search_respects_top_k(self, built_index: SkillsIndex) -> None:
        results = built_index.search("test", top_k=2)
        assert len(results) <= 2

    def test_search_with_required_tags(
        self, built_index: SkillsIndex,
    ) -> None:
        results = built_index.search(
            "test",
            required_tags={"domain:python"},
        )
        for r in results:
            assert "domain:python" in r.tags

    def test_search_required_tags_no_match(
        self, built_index: SkillsIndex,
    ) -> None:
        results = built_index.search(
            "test",
            required_tags={"domain:nonexistent"},
        )
        assert results == []

    def test_search_empty_query(self, built_index: SkillsIndex) -> None:
        results = built_index.search("")
        assert results == []

    def test_search_assigns_ranks(self, built_index: SkillsIndex) -> None:
        results = built_index.search("pytest", top_k=3)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_search_scores_are_floats(
        self, built_index: SkillsIndex,
    ) -> None:
        results = built_index.search("REST endpoints")
        for r in results:
            assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# attach_cluster_manifest + cluster methods
# ---------------------------------------------------------------------------


class TestCluster:
    def test_attach_cluster_manifest(
        self, built_index: SkillsIndex,
    ) -> None:
        manifest = build_clusters(
            _TINY_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="abc123",
        )
        built_index.attach_cluster_manifest(manifest)
        # After attach, search results should have cluster_id set
        results = built_index.search("pytest fixtures")
        assert any(r.cluster_id is not None for r in results)

    def test_cluster_by_label_found(
        self, built_index: SkillsIndex,
    ) -> None:
        manifest = build_clusters(
            _TINY_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="abc123",
        )
        built_index.attach_cluster_manifest(manifest)
        labels = built_index.all_cluster_labels()
        assert len(labels) > 0
        cluster = built_index.cluster_by_label(labels[0])
        assert cluster is not None

    def test_cluster_by_label_missing(
        self, built_index: SkillsIndex,
    ) -> None:
        result = built_index.cluster_by_label("nonexistent_label")
        assert result is None

    def test_nodes_in_cluster(self, built_index: SkillsIndex) -> None:
        manifest = build_clusters(
            _TINY_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="abc123",
        )
        built_index.attach_cluster_manifest(manifest)
        labels = built_index.all_cluster_labels()
        assert len(labels) > 0
        nodes = built_index.nodes_in_cluster(labels[0])
        assert len(nodes) > 0
        for n in nodes:
            assert isinstance(n, SearchResult)
            assert n.cluster_id is not None

    def test_nodes_in_cluster_missing_label(
        self, built_index: SkillsIndex,
    ) -> None:
        results = built_index.nodes_in_cluster("missing_cluster")
        assert results == []


# ---------------------------------------------------------------------------
# stats(), all_cluster_labels(), all_tags()
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_stats_returns_metrics(self, built_index: SkillsIndex) -> None:
        stats = built_index.stats()
        assert stats["node_count"] == 5
        assert stats["vocab_size"] > 0
        assert stats["avg_doc_len"] > 0.0

    def test_all_cluster_labels(self, built_index: SkillsIndex) -> None:
        manifest = build_clusters(
            _TINY_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="abc123",
        )
        built_index.attach_cluster_manifest(manifest)
        labels = built_index.all_cluster_labels()
        assert isinstance(labels, list)
        assert all(isinstance(lbl, str) for lbl in labels)

    def test_all_cluster_labels_no_manifest(
        self, built_index: SkillsIndex,
    ) -> None:
        labels = built_index.all_cluster_labels()
        assert labels == []

    def test_all_tags(self, built_index: SkillsIndex) -> None:
        tags = built_index.all_tags()
        assert isinstance(tags, set)
        assert "domain:python" in tags
        assert "domain:api" in tags
        assert "phase:test" in tags
