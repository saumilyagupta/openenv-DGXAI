from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class InterrogationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    questions: tuple[str, ...]
    cited_node_ids: tuple[str, ...]
