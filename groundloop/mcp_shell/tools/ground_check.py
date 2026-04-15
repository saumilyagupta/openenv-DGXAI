from __future__ import annotations

import math
import re
from typing import Any

from pydantic import ValidationError

from groundloop.lib_grounder.grounder import ground as lib_ground
from groundloop.mcp_shell import config as cfg
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import GroundCheckInput

_FENCED_PY_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_CODE_HINT_RE = re.compile(r"\b(import |def |class )")


def _softmax_first(scores: list[float]) -> float:
    if not scores:
        return 0.0
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return exps[0] / total if total > 0 else 0.0


def _extract_code(claim: str) -> str | None:
    blocks = _FENCED_PY_RE.findall(claim)
    if blocks:
        return "\n".join(blocks)
    if _CODE_HINT_RE.search(claim):
        return claim
    return None


def handle_ground_check(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = GroundCheckInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    session.inc("ground_checks")

    index = session.get_graph(inp.graph_id)
    if index is None:
        return {"status": "error", "reason": "unknown_graph_id", "detail": inp.graph_id}

    tags = set(inp.required_tags) if inp.required_tags else None
    results = index.search(inp.claim, top_k=inp.top_k, required_tags=tags)

    layer_a: dict[str, Any] | None = None
    code = _extract_code(inp.claim)
    if code is not None:
        report = lib_ground(code)
        layer_a = {
            "total_symbols": report.total_symbols,
            "groundedness": report.groundedness,
            "ungrounded": [
                {
                    "module": s.module,
                    "attr": s.attr,
                    "kind": s.kind,
                    "line": s.line,
                }
                for s in report.ungrounded
            ],
        }

    if not results:
        response: dict[str, Any] = {
            "status": "ok",
            "verdict": "ungrounded",
            "citations": [],
            "confidence": 0.0,
        }
        if layer_a is not None:
            response["layer_a"] = layer_a
        return response

    scores = [r.score for r in results]
    confidence = _softmax_first(scores)
    if layer_a is not None:
        confidence = confidence * layer_a["groundedness"]
    verdict = "grounded" if confidence >= cfg.GROUND_CHECK_GROUNDED_THRESHOLD else "uncertain"

    citations = [
        {
            "node_id": r.node_id,
            "skill_name": r.skill_name,
            "section_path": list(r.section_path),
            "section_body": r.section_body,
            "tags": list(r.tags),
            "source_path": r.source_path,
            "score": r.score,
            "rank": r.rank,
        }
        for r in results
    ]

    response = {
        "status": "ok",
        "verdict": verdict,
        "citations": citations,
        "confidence": confidence,
    }
    if layer_a is not None:
        response["layer_a"] = layer_a
    return response
