from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool


class ParsedResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ok: bool
    count: int
    details: dict[str, Any]


class ImportReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    total: int
    unresolved: tuple[str, ...]
    by_file: dict[str, tuple[str, ...]]


class SandboxResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    project_dir: str
    tools_run: tuple[str, ...]
    tool_results: dict[str, ToolResult]
    parsed: dict[str, ParsedResult]
    imports: ImportReport
    composite_score: float
    generated_at: str
