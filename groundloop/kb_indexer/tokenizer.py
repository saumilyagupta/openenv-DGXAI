from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if t]
