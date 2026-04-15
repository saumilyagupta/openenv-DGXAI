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
