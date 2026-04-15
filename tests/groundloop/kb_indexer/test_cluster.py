from __future__ import annotations

from groundloop.kb_indexer.cluster import (
    _connected_components,
    _jaccard,
    _label_cluster,
    build_clusters,
)


def test_jaccard_identity():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_jaccard_empty_both():
    assert _jaccard(set(), set()) == 0.0


def test_connected_components_simple():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}, "d": set()}
    comps = _connected_components(["a", "b", "c", "d"], adj)
    assert len(comps) == 2
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 3]


def test_label_cluster_uses_top_tokens_and_domain():
    nodes_data = [
        {"tokens": ["pytest", "fixture", "test"], "domain": "python",
         "tags": ("domain:python",)},
        {"tokens": ["pytest", "coverage", "test"], "domain": "python",
         "tags": ("domain:python",)},
    ]
    label, top_tokens, dominant = _label_cluster(nodes_data)
    assert dominant == "python"
    assert "pytest" in top_tokens
    assert label.startswith("python_")


def test_build_clusters_on_tiny_fixture(tiny_corpus_path):
    import json
    nodes = [
        json.loads(ln)
        for ln in tiny_corpus_path.read_text().splitlines()
        if ln.strip()
    ]
    manifest = build_clusters(nodes, jaccard_threshold=0.1)
    assert manifest.total_nodes_clustered == len(nodes)
    assert manifest.total_clusters >= 1
    seen: set[str] = set()
    for c in manifest.clusters:
        for nid in c.member_node_ids:
            assert nid not in seen, f"node {nid} in multiple clusters"
            seen.add(nid)
    assert len(seen) == len(nodes)


def test_build_clusters_determinism(tiny_corpus_path):
    import json
    nodes = [
        json.loads(ln)
        for ln in tiny_corpus_path.read_text().splitlines()
        if ln.strip()
    ]
    m1 = build_clusters(nodes, jaccard_threshold=0.1, corpus_sha256="sha")
    m2 = build_clusters(nodes, jaccard_threshold=0.1, corpus_sha256="sha")
    assert m1.total_clusters == m2.total_clusters
    assert tuple(c.cluster_id for c in m1.clusters) == tuple(
        c.cluster_id for c in m2.clusters
    )


def test_save_and_load_manifest_roundtrip(tiny_corpus_path, tmp_path):
    import json
    from groundloop.kb_indexer.cluster import load_manifest, save_manifest

    nodes = [
        json.loads(ln)
        for ln in tiny_corpus_path.read_text().splitlines()
        if ln.strip()
    ]
    manifest = build_clusters(nodes, jaccard_threshold=0.1)
    out = tmp_path / "m.json"
    save_manifest(manifest, out)
    loaded = load_manifest(out)
    assert loaded.total_clusters == manifest.total_clusters
    assert loaded.total_nodes_clustered == manifest.total_nodes_clustered


def test_build_clusters_empty_nodes():
    m = build_clusters(
        [], jaccard_threshold=0.15, generated_at="t", corpus_sha256="",
    )
    assert m.total_clusters == 0
    assert m.total_nodes_clustered == 0
    assert m.singletons == 0
    assert m.clusters == ()


def test_build_clusters_single_node():
    nodes = [{
        "id": "n1", "skill_name": "solo", "section_path": ["x"],
        "section_body": "alone body here with tokens for counting",
        "tags": ["domain:python"], "source_path": "/s",
    }]
    m = build_clusters(
        nodes, jaccard_threshold=0.15, generated_at="t", corpus_sha256="",
    )
    assert m.total_clusters == 1
    assert m.singletons == 1


def test_build_clusters_all_disjoint_at_threshold_one():
    nodes = [
        {
            "id": f"n{i}", "skill_name": "s", "section_path": ["x"],
            "section_body": f"alpha beta unique_word_{i}",
            "tags": ["domain:python"], "source_path": "/s",
        }
        for i in range(5)
    ]
    m = build_clusters(
        nodes, jaccard_threshold=1.0, generated_at="t", corpus_sha256="",
    )
    assert m.singletons == 5
    assert m.total_clusters == 5


def test_build_clusters_all_connected_at_zero_threshold():
    nodes = [
        {
            "id": f"n{i}", "skill_name": "s", "section_path": ["x"],
            "section_body": "alpha beta gamma delta",
            "tags": ["domain:python"], "source_path": "/s",
        }
        for i in range(4)
    ]
    m = build_clusters(
        nodes, jaccard_threshold=0.0, generated_at="t", corpus_sha256="",
    )
    assert m.total_clusters == 1
    assert m.clusters[0].node_count == 4
