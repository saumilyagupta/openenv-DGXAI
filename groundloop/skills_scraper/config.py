from __future__ import annotations

from groundloop.skills_scraper.models import SourceRoot


DEFAULT_SOURCES: list[SourceRoot] = [
    SourceRoot(label="user-skills", glob="~/.claude/skills/*/SKILL.md"),
    SourceRoot(label="agent-skills", glob="~/.claude/.agents/skills/*/SKILL.md"),
    SourceRoot(label="cursor-skills", glob="~/.claude/.cursor/skills/*/SKILL.md"),
    SourceRoot(label="plugin-skills", glob="~/.claude/plugins/marketplaces/*/*/SKILL.md"),
]

MIN_CHUNK_CHARS = 80
DEFAULT_OUTPUT = "groundloop/kb/skills_corpus.jsonl"
