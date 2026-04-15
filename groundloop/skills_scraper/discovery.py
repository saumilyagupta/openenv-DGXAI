from __future__ import annotations

import glob as glob_mod
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from groundloop.skills_scraper.models import SourceRoot

_log = logging.getLogger(__name__)


def walk_sources(sources: list[SourceRoot]) -> Iterator[tuple[Path, SourceRoot]]:
    """Yield (path, source_root) for every readable SKILL.md matched by a source glob."""
    for root in sources:
        pattern = os.path.expanduser(root.glob)
        for match in glob_mod.glob(pattern, recursive=True):
            path = Path(match)
            if not path.is_file():
                _log.warning("discovery: skipping non-file %s", path)
                continue
            try:
                path.stat()
            except OSError as e:
                _log.warning("discovery: unreadable %s: %s", path, e)
                continue
            yield path, root
