from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from codeforge.models import AuditEntry


class AuditLedger:
    """Per-episode, per-step append-only audit log."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def total_reward(self) -> float:
        return sum(e.reward for e in self._entries)

    def citation_count(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for entry in self._entries:
            counts.update(entry.cited_skill_ids)
        return dict(counts)

    def step_count(self) -> int:
        return len(self._entries)

    def serialize(self) -> dict[str, object]:
        return {
            "entries": [asdict(e) for e in self._entries],
            "total_reward": self.total_reward(),
            "step_count": self.step_count(),
            "citation_count": self.citation_count(),
        }
