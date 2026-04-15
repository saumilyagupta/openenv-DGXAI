from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from pydantic import ValidationError

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell import config as cfg
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import IngestSourcesInput
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper

_log = logging.getLogger(__name__)


def _derive_graph_id(source_globs: list[str] | None) -> str:
    # Hash of input sources only (not derived paths) so graph_id is stable
    # across invocations and can itself name the per-graph cache dir.
    payload = "|".join(source_globs) if source_globs else "<default-corpus>"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{cfg.DEFAULT_GRAPH_ID_PREFIX}{digest}"


def handle_ingest_sources(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = IngestSourcesInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    t0 = time.monotonic()

    graph_id = _derive_graph_id(inp.source_globs)

    # Spec §6: scraper/indexer failures must be wrapped into a structured
    # response, never propagate past the handler. Broad Exception is
    # appropriate at the MCP handler boundary.
    try:
        if inp.source_globs is None:
            corpus_path = cfg.DEFAULT_CORPUS_PATH
            if not corpus_path.exists():
                return {
                    "status": "error",
                    "reason": "missing_default_corpus",
                    "detail": (
                        f"{corpus_path} not built; "
                        "run `python -m groundloop.skills_scraper`."
                    ),
                }
        else:
            # Stable per-graph cache dir under the user-facing kb/ tree so
            # re-ingesting the same globs reuses on-disk state instead of
            # leaking a fresh /tmp/groundloop_mcp_* dir per call.
            graph_dir = cfg.DEFAULT_CACHE_PATH.parent / graph_id
            graph_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = graph_dir / "corpus.jsonl"
            sources = [
                SourceRoot(label=f"mcp-{i}", glob=g)
                for i, g in enumerate(inp.source_globs)
            ]
            result = run_scraper(sources=sources, output=corpus_path)
            if result.total_nodes == 0:
                return {"status": "error", "reason": "no_nodes_scraped"}

        index = SkillsIndex(corpus_path=corpus_path)
        index.build()
    except Exception as e:  # noqa: BLE001 — MCP handler boundary
        _log.exception("ingest_sources failed")
        return {"status": "error", "reason": "ingest_failed", "detail": str(e)}

    session.register_graph(graph_id, index)
    session.inc("graphs_built")

    return {
        "status": "ok",
        "graph_id": graph_id,
        "nodes": int(index.stats()["node_count"]),
        "build_ms": round((time.monotonic() - t0) * 1000, 2),
    }
