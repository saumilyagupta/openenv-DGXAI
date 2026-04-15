from __future__ import annotations

import logging
import pickle
from pathlib import Path

_log = logging.getLogger(__name__)


def save_cache(state: dict, cache_path: Path, corpus_sha256: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sha256": corpus_sha256, **state}
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(cache_path: Path, expected_sha256: str) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        _log.warning("cache: corrupt or unreadable %s: %s", cache_path, e)
        return None
    if payload.get("sha256") != expected_sha256:
        return None
    return payload
