from __future__ import annotations

from pathlib import Path

import frontmatter  # type: ignore[import-untyped]
import yaml

from groundloop.skills_scraper.models import ParsedSkill


class ParseError(Exception):
    """Raised when a SKILL.md cannot be parsed (malformed YAML, I/O error)."""


def parse_skill(path: Path) -> ParsedSkill:
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
