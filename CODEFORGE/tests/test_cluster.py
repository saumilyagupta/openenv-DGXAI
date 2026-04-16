from __future__ import annotations

import pytest

from codeforge.kb.cluster import build_clusters
from codeforge.kb.models import Cluster, ClusterManifest


# ---------------------------------------------------------------------------
# Fixture: test nodes (same format as JSONL corpus)
# ---------------------------------------------------------------------------

_SIMILAR_NODES: list[dict[str, object]] = [
    {
        "id": "n1",
        "skill_name": "python-testing",
        "section_path": ["Fixtures"],
        "section_body": (
            "Use pytest fixtures for setup and teardown of test state."
        ),
        "tags": ["domain:python", "phase:test"],
        "source_path": "/fake/python-testing/SKILL.md",
    },
    {
        "id": "n2",
        "skill_name": "python-testing",
        "section_path": ["Parametrize"],
        "section_body": (
            "Use pytest parametrize to run tests with multiple inputs."
        ),
        "tags": ["domain:python", "phase:test"],
        "source_path": "/fake/python-testing/SKILL.md",
    },
    {
        "id": "n3",
        "skill_name": "python-testing",
        "section_path": ["Coverage"],
        "section_body": (
            "Use pytest coverage to measure test coverage and quality."
        ),
        "tags": ["domain:python", "phase:test"],
        "source_path": "/fake/python-testing/SKILL.md",
    },
    {
        "id": "n4",
        "skill_name": "api-design",
        "section_path": ["REST"],
        "section_body": (
            "Design REST endpoints with HTTP verbs and resource naming."
        ),
        "tags": ["domain:api", "phase:plan"],
        "source_path": "/fake/api-design/SKILL.md",
    },
    {
        "id": "n5",
        "skill_name": "security-review",
        "section_path": ["Auth"],
        "section_body": (
            "Review authentication and authorization mechanisms for security."
        ),
        "tags": ["domain:security", "phase:review"],
        "source_path": "/fake/security-review/SKILL.md",
    },
]


class TestBuildClusters:
    def test_returns_manifest(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="test_sha",
        )
        assert isinstance(manifest, ClusterManifest)

    def test_manifest_fields(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="test_sha",
            generated_at="2026-01-01T00:00:00+00:00",
        )
        assert manifest.corpus_sha256 == "test_sha"
        assert manifest.jaccard_threshold == 0.15
        assert manifest.generated_at == "2026-01-01T00:00:00+00:00"

    def test_total_nodes_clustered(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="test_sha",
        )
        assert manifest.total_nodes_clustered == len(_SIMILAR_NODES)

    def test_clusters_contain_valid_data(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="test_sha",
        )
        for cluster in manifest.clusters:
            assert isinstance(cluster, Cluster)
            assert cluster.node_count == len(cluster.member_node_ids)
            assert cluster.cluster_id  # not empty
            assert cluster.label  # not empty
            assert cluster.dominant_domain  # not empty

    def test_similar_nodes_cluster_together(self) -> None:
        # The 3 python-testing nodes share lots of vocabulary
        # and should cluster together with low threshold
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.10,
            corpus_sha256="test_sha",
        )
        # Find the cluster containing n1
        n1_cluster = None
        for c in manifest.clusters:
            if "n1" in c.member_node_ids:
                n1_cluster = c
                break
        assert n1_cluster is not None
        # n2 should be in the same cluster (similar pytest vocabulary)
        assert "n2" in n1_cluster.member_node_ids

    def test_high_threshold_produces_more_singletons(self) -> None:
        low = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.05,
            corpus_sha256="s",
        )
        high = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.80,
            corpus_sha256="s",
        )
        # Higher threshold = fewer edges = more clusters/singletons
        assert high.singletons >= low.singletons

    def test_singleton_detection(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.99,
            corpus_sha256="s",
        )
        # At 0.99 threshold almost nothing clusters
        # Every node should be its own cluster
        assert manifest.singletons == len(_SIMILAR_NODES)

    def test_cluster_labels_include_domain(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.10,
            corpus_sha256="s",
        )
        for c in manifest.clusters:
            # Label should contain dominant_domain
            assert c.dominant_domain in c.label

    def test_clusters_sorted_by_size_desc(self) -> None:
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.15,
            corpus_sha256="s",
        )
        sizes = [c.node_count for c in manifest.clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_empty_nodes(self) -> None:
        manifest = build_clusters(
            [],
            jaccard_threshold=0.15,
            corpus_sha256="s",
        )
        assert manifest.total_clusters == 0
        assert manifest.total_nodes_clustered == 0
        assert manifest.singletons == 0

    def test_connected_components_grouping(self) -> None:
        # If A-B similar and B-C similar but A-C not directly,
        # all three should still be in the same component
        manifest = build_clusters(
            _SIMILAR_NODES,
            jaccard_threshold=0.10,
            corpus_sha256="s",
        )
        all_node_ids: set[str] = set()
        for c in manifest.clusters:
            for nid in c.member_node_ids:
                assert nid not in all_node_ids, "Node in multiple clusters"
                all_node_ids.add(nid)
        # All input node ids accounted for
        expected = {str(n["id"]) for n in _SIMILAR_NODES}
        assert all_node_ids == expected
