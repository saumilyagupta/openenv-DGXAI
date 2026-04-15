from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell import config as cfg
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import IngestSourcesInput
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def _derive_graph_id(corpus_path: Path, source_globs: list[str] | None) -> str:
    payload = str(corpus_path) + "|" + "|".join(source_globs or [])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{cfg.DEFAULT_GRAPH_ID_PREFIX}{digest}"


def handle_ingest_sources(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = IngestSourcesInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    t0 = time.monotonic()

    if inp.source_globs is None:
        corpus_path = cfg.DEFAULT_CORPUS_PATH
        if not corpus_path.exists():
            return {
                "status": "error",
                "reason": "missing_default_corpus",
                "detail": f"{corpus_path} not built; run `python -m groundloop.skills_scraper`.",
            }
    else:
        sources = [SourceRoot(label=f"mcp-{i}", glob=g) for i, g in enumerate(inp.source_globs)]
        tmp = Path(tempfile.mkdtemp(prefix="groundloop_mcp_"))
        corpus_path = tmp / "corpus.jsonl"
        result = run_scraper(sources=sources, output=corpus_path)
        if result.total_nodes == 0:
            return {"status": "error", "reason": "no_nodes_scraped"}

    index = SkillsIndex(corpus_path=corpus_path)
    index.build()
    graph_id = _derive_graph_id(corpus_path, inp.source_globs)
    session.register_graph(graph_id, index)
    session.inc("graphs_built")

    return {
        "status": "ok",
        "graph_id": graph_id,
        "nodes": int(index.stats()["node_count"]),
        "build_ms": round((time.monotonic() - t0) * 1000, 2),
    }
