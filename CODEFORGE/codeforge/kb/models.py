from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    node_id: str
    skill_name: str
    section_path: tuple[str, ...]
    section_body: str
    tags: tuple[str, ...]
    source_path: str
    score: float
    rank: int
    cluster_id: str | None = None


class Cluster(BaseModel):
    model_config = ConfigDict(frozen=True)
    cluster_id: str
    label: str
    dominant_domain: str
    top_tokens: tuple[str, ...]
    node_count: int
    member_node_ids: tuple[str, ...]


class ClusterManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: str
    corpus_sha256: str
    jaccard_threshold: float
    total_clusters: int
    total_nodes_clustered: int
    singletons: int
    clusters: tuple[Cluster, ...]
