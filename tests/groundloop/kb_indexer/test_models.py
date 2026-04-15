import pytest
from pydantic import ValidationError

from groundloop.kb_indexer.models import Cluster, ClusterManifest, SearchResult


def test_search_result_frozen():
    r = SearchResult(
        node_id="abc",
        skill_name="python-testing",
        section_path=("Fixtures",),
        section_body="body",
        tags=("domain:python",),
        source_path="/p",
        score=1.23,
        rank=1,
    )
    with pytest.raises(ValidationError):
        r.rank = 2  # type: ignore[misc]


def test_search_result_requires_fields():
    with pytest.raises(ValidationError):
        SearchResult()  # type: ignore[call-arg]


def test_cluster_frozen():
    c = Cluster(
        cluster_id="abc", label="python_testing",
        dominant_domain="python", top_tokens=("pytest",),
        node_count=3, member_node_ids=("n1", "n2", "n3"),
    )
    with pytest.raises(ValidationError):
        c.label = "other"  # type: ignore[misc]


def test_cluster_manifest_shape():
    m = ClusterManifest(
        generated_at="t", corpus_sha256="sha",
        jaccard_threshold=0.15, total_clusters=1,
        total_nodes_clustered=3, singletons=0,
        clusters=(Cluster(
            cluster_id="c1", label="l", dominant_domain="python",
            top_tokens=(), node_count=3, member_node_ids=("n1", "n2", "n3"),
        ),),
    )
    assert m.total_clusters == 1


def test_search_result_cluster_id_defaults_none():
    r = SearchResult(
        node_id="n1", skill_name="s", section_path=("x",),
        section_body="b", tags=("domain:python",), source_path="/p",
        score=1.0, rank=1,
    )
    assert r.cluster_id is None
