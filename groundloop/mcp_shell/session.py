from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    spec: str
    graph_id: str
    status: str = "pending_orchestrator"
    iterations: int = 0
    notes: list[str] = field(default_factory=list)


class SessionState:
    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}
        self._runs: dict[str, RunRecord] = {}
        self._metrics: dict[str, int] = {
            "tool_calls": 0,
            "graphs_built": 0,
            "ground_checks": 0,
        }

    def inc(self, key: str, by: int = 1) -> None:
        if key not in self._metrics:
            self._metrics[key] = 0
        self._metrics[key] += by

    def metrics_snapshot(self) -> dict[str, int]:
        return dict(self._metrics)

    def register_graph(self, graph_id: str, index: Any) -> None:
        self._graphs[graph_id] = index

    def get_graph(self, graph_id: str) -> Any | None:
        return self._graphs.get(graph_id)

    def create_run(self, *, spec: str, graph_id: str) -> RunRecord:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        rec = RunRecord(run_id=run_id, spec=spec, graph_id=graph_id)
        self._runs[run_id] = rec
        return rec

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)
