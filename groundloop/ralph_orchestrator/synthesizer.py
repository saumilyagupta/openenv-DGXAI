from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.models import SynthesisResult


class Synthesizer(Protocol):
    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult: ...
