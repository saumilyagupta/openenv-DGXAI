from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence

from openai import OpenAI

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.models import SynthesisResult

_log = logging.getLogger(__name__)

_SYSTEM = (
    "You generate production Python code. Use only the provided skill citations. "
    "Return a single JSON object with keys: proposed_files (mapping filename->content), "
    "rationale (string), cited_node_ids (array of node_id strings). No prose."
)


class OpenAISynthesizer:
    def __init__(self, *, model: str | None = None, timeout: float = 60.0) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            msg = "OPENAI_API_KEY env var not set"
            raise RuntimeError(msg)
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._model = model or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self._timeout = timeout
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        cites_str = "\n\n".join(
            f"[{c.node_id}] {c.skill_name}/{'/'.join(c.section_path)}\n{c.section_body}"
            for c in citations
        )
        files_str = "\n\n".join(f"### {name}\n{content}" for name, content in current_files.items())
        user = (
            f"Spec: {spec}\n\nIteration: {iteration}\n\n"
            f"Current files:\n{files_str}\n\nCitations:\n{cites_str}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                timeout=self._timeout,
            )
            content = resp.choices[0].message.content or ""
            payload = json.loads(content)
            return SynthesisResult(
                proposed_files=dict(payload["proposed_files"]),
                rationale=str(payload.get("rationale", "")),
                cited_node_ids=tuple(payload.get("cited_node_ids", ())),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            _log.warning("openai_synthesizer: parse error: %s", e)
            return SynthesisResult(
                proposed_files=dict(current_files),
                rationale="parse_error",
                cited_node_ids=(),
            )
