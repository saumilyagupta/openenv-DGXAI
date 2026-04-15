from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SourceRoot(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    glob: str


class ParsedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)
    frontmatter: dict
    body: str
    mtime: float


class SectionChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    section_path: tuple[str, ...]
    section_body: str


class ScrapeError(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    stage: str
    reason: str


class SkillNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    skill_name: str
    skill_description: str
    skill_type: Literal["flexible", "rigid"] | None
    section_path: tuple[str, ...]
    section_title: str
    section_body: str
    source_path: str
    source_root: str
    tags: tuple[str, ...]
    trigger_hints: str
    mtime: float
    body_hash: str
    alias_sources: tuple[str, ...]


class ScrapeManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: str
    sources: list[SourceRoot]
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[ScrapeError]
    corpus_sha256: str
