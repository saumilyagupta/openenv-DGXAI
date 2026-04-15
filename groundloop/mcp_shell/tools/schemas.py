from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InterrogateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brief: str = Field(min_length=1)
    graph_id: str | None = None


class IngestSourcesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_globs: list[str] | None = None


class GroundCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)
    required_tags: list[str] = Field(default_factory=list)


class AutonomousBuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    max_iters: int = Field(default=3, gt=0, le=20)


class AuditReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
