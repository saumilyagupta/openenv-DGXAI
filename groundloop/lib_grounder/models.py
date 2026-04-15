from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Symbol(BaseModel):
    model_config = ConfigDict(frozen=True)
    module: str
    attr: str | None
    kind: Literal["import", "attribute"]
    resolved: bool
    line: int


class GroundingReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_symbols: int
    grounded: tuple[Symbol, ...]
    ungrounded: tuple[Symbol, ...]
    groundedness: float
