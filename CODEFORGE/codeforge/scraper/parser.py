from __future__ import annotations

from pathlib import Path

import frontmatter  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict


class ParsedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)
    frontmatter: dict[str, object]
    body: str
    mtime: float


class ParseError(Exception):
    """Raised when a SKILL.md cannot be parsed."""


def parse_skill(path: Path) -> ParsedSkill:
    """Parse a SKILL.md file: extract YAML frontmatter and markdown body."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ParseError(f"unreadable: {path}: {e}") from e
    try:
        post = frontmatter.loads(raw)
    except yaml.YAMLError as e:
        raise ParseError(f"malformed yaml in {path}: {e}") from e
    mtime = path.stat().st_mtime
    return ParsedSkill(frontmatter=dict(post.metadata), body=post.content, mtime=mtime)
