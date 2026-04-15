# GroundLoop Ralph-Orchestrator Implementation Plan

> Use superpowers:subagent-driven-development.

**Goal:** The core Ralph loop: plan→synthesize→sandbox-score→keep-or-revert→repeat, with a pluggable Synthesizer and JSON checkpoint.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-ralph-orchestrator-design.md`.

---

## File Structure

```
groundloop/ralph_orchestrator/
  __init__.py
  __main__.py
  models.py
  synthesizer.py         # Protocol + SynthesisResult
  stub_synthesizer.py
  openai_synthesizer.py
  checkpoint.py
  loop.py
  cli.py
tests/groundloop/ralph_orchestrator/
  fixtures/
    spec_simple.txt
    initial_files/main.py
  conftest.py
  test_models.py
  test_stub_synthesizer.py
  test_openai_synthesizer.py
  test_checkpoint.py
  test_loop.py
  test_cli.py
  test_e2e.py
```

---

## Task 1: Scaffold + fixtures

- [ ] Create package + test dirs, empty `__init__.py` files.
- [ ] `tests/groundloop/ralph_orchestrator/fixtures/spec_simple.txt`:
```
Build a Python module that exposes a greet(name) -> str function using pytest idioms from the skills KB.
```
- [ ] `tests/groundloop/ralph_orchestrator/fixtures/initial_files/main.py`:
```python
from __future__ import annotations


def greet(name: str) -> str:
    return "hi"
```
- [ ] `conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def spec_path() -> Path:
    return Path(__file__).parent / "fixtures" / "spec_simple.txt"


@pytest.fixture
def initial_files(tmp_path: Path) -> dict[str, str]:
    src = Path(__file__).parent / "fixtures" / "initial_files" / "main.py"
    return {"main.py": src.read_text(encoding="utf-8")}


@pytest.fixture
def tiny_corpus_path() -> Path:
    return Path(__file__).parent.parent / "kb_indexer" / "fixtures" / "tiny_corpus.jsonl"
```
- [ ] Commit: `chore: scaffold groundloop/ralph_orchestrator package + fixtures`

---

## Task 2: Models

- [ ] `tests/groundloop/ralph_orchestrator/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from groundloop.ralph_orchestrator.models import Iteration, LoopConfig, RunResult, SynthesisResult


def test_loop_config_defaults():
    c = LoopConfig()
    assert c.max_iters == 5
    assert 0.0 < c.target_score <= 1.0


def test_iteration_frozen():
    it = Iteration(
        index=0, cited_node_ids=(), rationale="r",
        proposed_files={"a.py": ""}, sandbox_score_before=0.0,
        sandbox_score_after=1.0, kept=True, reason="score_improved",
    )
    with pytest.raises(ValidationError):
        it.kept = False  # type: ignore[misc]


def test_synthesis_result_requires_fields():
    with pytest.raises(ValidationError):
        SynthesisResult()  # type: ignore[call-arg]


def test_run_result_iterations_immutable():
    r = RunResult(
        run_id="r", spec="s", started_at="t", ended_at="t",
        final_score=1.0, final_files={"main.py": ""}, iterations=(),
        terminated_by="target_hit",
    )
    assert r.iterations == ()
```
- [ ] Implement `groundloop/ralph_orchestrator/models.py`:
```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TerminationReason = Literal["target_hit", "max_iters", "stuck"]
IterationReason = Literal[
    "score_improved", "score_regressed", "target_hit", "sandbox_error", "synthesizer_error"
]


class LoopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_iters: int = Field(default=5, gt=0, le=100)
    target_score: float = Field(default=0.95, gt=0.0, le=1.0)
    tools: tuple[str, ...] = ("ruff", "imports")
    timeout_per_tool: float = 60.0
    top_k_citations: int = Field(default=5, gt=0, le=50)


class SynthesisResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    proposed_files: dict[str, str]
    rationale: str
    cited_node_ids: tuple[str, ...]


class Iteration(BaseModel):
    model_config = ConfigDict(frozen=True)
    index: int
    cited_node_ids: tuple[str, ...]
    rationale: str
    proposed_files: dict[str, str]
    sandbox_score_before: float
    sandbox_score_after: float
    kept: bool
    reason: IterationReason


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    spec: str
    started_at: str
    ended_at: str
    final_score: float
    final_files: dict[str, str]
    iterations: tuple[Iteration, ...]
    terminated_by: TerminationReason
```
- [ ] Run: PASS.
- [ ] Commit: `feat(ralph-orchestrator): Pydantic models`

---

## Task 3: Synthesizer protocol

- [ ] Implement `groundloop/ralph_orchestrator/synthesizer.py`:
```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.models import SynthesisResult


class Synthesizer(Protocol):
    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult: ...
```
- [ ] No tests needed (pure protocol). Commit: `feat(ralph-orchestrator): Synthesizer protocol`

---

## Task 4: StubSynthesizer

- [ ] `tests/groundloop/ralph_orchestrator/test_stub_synthesizer.py`:
```python
from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def _cit(node_id: str = "n1", body: str = "some body", rank: int = 1, score: float = 1.0) -> SearchResult:
    return SearchResult(
        node_id=node_id, skill_name="python-testing", section_path=("Fixtures",),
        section_body=body, tags=("domain:python",), source_path="/x", score=score, rank=rank,
    )


def test_stub_no_citations_is_noop():
    s = StubSynthesizer()
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1"}, citations=[], iteration=0)
    assert out.proposed_files == {"main.py": "x = 1"}
    assert out.cited_node_ids == ()


def test_stub_appends_comment_when_no_code_block():
    s = StubSynthesizer()
    cit = _cit(body="Discussion about fixtures, no code blocks here.")
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1"}, citations=[cit], iteration=0)
    assert "consulted: python-testing" in out.proposed_files["main.py"]
    assert "n1" in out.cited_node_ids


def test_stub_extracts_fenced_python():
    body = "Example:\n```python\ndef demo() -> int:\n    return 42\n```\n"
    s = StubSynthesizer()
    cit = _cit(body=body)
    out = s.synthesize(spec="x", current_files={"main.py": "x = 1\n"}, citations=[cit], iteration=0)
    assert "demo" in out.proposed_files["main.py"]
    assert "_from_python_testing_1" in out.proposed_files["main.py"]


def test_stub_deterministic():
    body = "```python\ndef demo(): pass\n```"
    s = StubSynthesizer()
    cit = _cit(body=body)
    a = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    b = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    assert a.proposed_files == b.proposed_files
    assert a.rationale == b.rationale


def test_stub_idempotent_on_already_applied_code():
    body = "```python\ndef demo(): pass\n```"
    s = StubSynthesizer()
    cit = _cit(body=body)
    once = s.synthesize(spec="x", current_files={"main.py": ""}, citations=[cit], iteration=0)
    twice = s.synthesize(spec="x", current_files=once.proposed_files, citations=[cit], iteration=1)
    assert twice.proposed_files == once.proposed_files
```
- [ ] Implement `groundloop/ralph_orchestrator/stub_synthesizer.py`:
```python
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.models import SynthesisResult

_FENCED_PY_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


class StubSynthesizer:
    """Deterministic, KB-grounded stub. No LLM. Used in tests and as default."""

    def synthesize(
        self,
        *,
        spec: str,
        current_files: Mapping[str, str],
        citations: Sequence[SearchResult],
        iteration: int,
    ) -> SynthesisResult:
        del spec, iteration  # inputs retained for protocol; stub ignores

        if not citations:
            return SynthesisResult(
                proposed_files=dict(current_files),
                rationale="no_citations",
                cited_node_ids=(),
            )

        top = citations[0]
        main = current_files.get("main.py", "")
        blocks = _FENCED_PY_RE.findall(top.section_body)

        applied_code: list[str] = []
        wrapper_name = f"_from_{top.skill_name.replace('-', '_')}_{top.rank}"
        if blocks and wrapper_name not in main:
            body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in blocks[0].splitlines())
            snippet = f"\n\ndef {wrapper_name}() -> None:\n{body}\n"
            applied_code.append(snippet)
            new_main = main + snippet
            rationale = f"Applied suggestion from {top.skill_name}/{'/'.join(top.section_path)}"
        else:
            comment = f"# consulted: {top.skill_name}/{'/'.join(top.section_path)}\n"
            new_main = main + (comment if comment not in main else "")
            rationale = f"Consulted {top.skill_name}/{'/'.join(top.section_path)} (no code block)"

        new_files = {**current_files, "main.py": new_main}
        return SynthesisResult(
            proposed_files=new_files,
            rationale=rationale,
            cited_node_ids=(top.node_id,),
        )
```
- [ ] Run: PASS.
- [ ] Commit: `feat(ralph-orchestrator): deterministic stub synthesizer`

---

## Task 5: OpenAISynthesizer (optional path)

- [ ] `tests/groundloop/ralph_orchestrator/test_openai_synthesizer.py`:
```python
from __future__ import annotations

import os

import pytest

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator import openai_synthesizer as mod


def _cit() -> SearchResult:
    return SearchResult(
        node_id="n1", skill_name="python-testing", section_path=("Fixtures",),
        section_body="Use pytest fixtures.", tags=("domain:python",),
        source_path="/x", score=1.0, rank=1,
    )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = self._Chat(content)

    class _Chat:
        def __init__(self, content: str) -> None:
            self.completions = _FakeClient._Completions(content)

    class _Completions:
        def __init__(self, content: str) -> None:
            self._content = content

        def create(self, **_: object) -> object:
            class _Msg:
                def __init__(self, c: str) -> None:
                    self.content = c
            class _Choice:
                def __init__(self, c: str) -> None:
                    self.message = _Msg(c)
            class _Resp:
                def __init__(self, c: str) -> None:
                    self.choices = [_Choice(c)]
            return _Resp(self._content)


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mod.OpenAISynthesizer()


def test_openai_parses_valid_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    content = '{"proposed_files": {"main.py": "x = 2"}, "rationale": "ok", "cited_node_ids": ["n1"]}'
    monkeypatch.setattr(mod, "OpenAI", lambda **kw: _FakeClient(content))
    s = mod.OpenAISynthesizer()
    out = s.synthesize(spec="s", current_files={"main.py": "x = 1"}, citations=[_cit()], iteration=0)
    assert out.proposed_files == {"main.py": "x = 2"}
    assert out.cited_node_ids == ("n1",)


def test_openai_parse_error_is_noop(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(mod, "OpenAI", lambda **kw: _FakeClient("not json"))
    s = mod.OpenAISynthesizer()
    current = {"main.py": "x = 1"}
    out = s.synthesize(spec="s", current_files=current, citations=[_cit()], iteration=0)
    assert out.proposed_files == current
    assert out.rationale == "parse_error"
    assert out.cited_node_ids == ()
```
- [ ] Implement `groundloop/ralph_orchestrator/openai_synthesizer.py`:
```python
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence

from openai import OpenAI  # type: ignore[import-not-found]

from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.models import SynthesisResult

_log = logging.getLogger(__name__)

_SYSTEM = (
    "You generate production Python code. Use only the provided skill citations. "
    "Return a single JSON object with keys: proposed_files (mapping filename->content), "
    "rationale (string), cited_node_ids (array of node_id strings). No prose."
)


class OpenAISynthesizer:
    def __init__(self, *, model: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            msg = "OPENAI_API_KEY env var not set"
            raise RuntimeError(msg)
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._model = model or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
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
```
- [ ] Run: PASS.
- [ ] Commit: `feat(ralph-orchestrator): optional OpenAI synthesizer`

---

## Task 6: Checkpoint save/load

- [ ] `tests/groundloop/ralph_orchestrator/test_checkpoint.py`:
```python
from pathlib import Path

from groundloop.ralph_orchestrator.checkpoint import load_checkpoint, save_checkpoint
from groundloop.ralph_orchestrator.models import RunResult


def _result() -> RunResult:
    return RunResult(
        run_id="r1", spec="s", started_at="t0", ended_at="t1",
        final_score=1.0, final_files={"main.py": "x = 1\n"},
        iterations=(), terminated_by="target_hit",
    )


def test_checkpoint_roundtrip(tmp_path: Path):
    out = save_checkpoint(_result(), tmp_path)
    loaded = load_checkpoint(out)
    assert loaded.run_id == "r1"
    assert loaded.final_files == {"main.py": "x = 1\n"}


def test_checkpoint_atomic_write(tmp_path: Path):
    # Sanity: no stray .tmp files left behind
    save_checkpoint(_result(), tmp_path)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
```
- [ ] Implement `groundloop/ralph_orchestrator/checkpoint.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

from groundloop.ralph_orchestrator.models import RunResult


def save_checkpoint(run: RunResult, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_dir / f"run_{run.run_id}.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final


def load_checkpoint(checkpoint_path: Path) -> RunResult:
    return RunResult.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
```
- [ ] Run: PASS.
- [ ] Commit: `feat(ralph-orchestrator): atomic JSON checkpoint save/load`

---

## Task 7: Core loop

- [ ] `tests/groundloop/ralph_orchestrator/test_loop.py`:
```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.kb_indexer.models import SearchResult
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig, SynthesisResult


class _NoopSynth:
    def synthesize(self, *, spec, current_files, citations, iteration):
        return SynthesisResult(
            proposed_files=dict(current_files), rationale="noop", cited_node_ids=(),
        )


class _ImproverSynth:
    """Synthesizer that produces increasingly clean code each call."""
    def __init__(self):
        self.calls = 0

    def synthesize(self, *, spec, current_files, citations, iteration):
        self.calls += 1
        if self.calls == 1:
            # add a valid function
            new = {**current_files, "main.py": "def ok() -> int:\n    return 1\n"}
        else:
            new = dict(current_files)
        return SynthesisResult(proposed_files=new, rationale="improve", cited_node_ids=())


class _RegressorSynth:
    def synthesize(self, *, spec, current_files, citations, iteration):
        # inject a broken import on every call to regress score
        bad = "import nonexistent_zzz_{i}\n".format(i=iteration)
        return SynthesisResult(
            proposed_files={**current_files, "main.py": bad},
            rationale="regress",
            cited_node_ids=(),
        )


def test_loop_terminates_on_max_iters(tiny_corpus_path: Path, initial_files: Mapping[str, str]):
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_NoopSynth(), config=LoopConfig(max_iters=3, target_score=1.1),
    )
    assert result.terminated_by == "max_iters"
    assert len(result.iterations) == 3


def test_loop_target_hit_early(tiny_corpus_path: Path):
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    # target_score above 0 and initial is perfectly clean -> score=1.0 -> target hit on score_before
    perfect = {"main.py": "from __future__ import annotations\n\n\ndef ok() -> int:\n    return 1\n"}
    result = run_loop(
        spec="x", initial_files=perfect, index=idx,
        synthesizer=_NoopSynth(), config=LoopConfig(max_iters=3, target_score=0.95),
    )
    assert result.terminated_by == "target_hit"
    assert result.iterations == ()


def test_loop_stuck_after_three_regressions(tiny_corpus_path: Path, initial_files: Mapping[str, str]):
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_RegressorSynth(), config=LoopConfig(max_iters=10, target_score=1.1),
    )
    assert result.terminated_by == "stuck"
    # Exactly 3 iterations before stuck fires
    assert len(result.iterations) == 3
    assert all(it.kept is False for it in result.iterations)


def test_loop_writes_checkpoint(tmp_path: Path, tiny_corpus_path: Path, initial_files: Mapping[str, str]):
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    result = run_loop(
        spec="x", initial_files=initial_files, index=idx,
        synthesizer=_NoopSynth(),
        config=LoopConfig(max_iters=1, target_score=1.1),
        checkpoint_dir=tmp_path,
    )
    assert (tmp_path / f"run_{result.run_id}.json").exists()
```
- [ ] Implement `groundloop/ralph_orchestrator/loop.py`:
```python
from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop.ralph_orchestrator.checkpoint import save_checkpoint
from groundloop.ralph_orchestrator.models import (
    Iteration,
    IterationReason,
    LoopConfig,
    RunResult,
    TerminationReason,
)
from groundloop.ralph_orchestrator.synthesizer import Synthesizer

_log = logging.getLogger(__name__)
_STUCK_THRESHOLD = 3


def _score_files(files: Mapping[str, str], config: LoopConfig) -> float:
    try:
        result = run_sandbox(
            files=dict(files), tools=config.tools, timeout_per_tool=config.timeout_per_tool,
        )
    except Exception as e:  # noqa: BLE001 - sandbox errors must not kill the loop
        _log.exception("sandbox error: %s", e)
        return 0.0
    return result.composite_score


def run_loop(
    *,
    spec: str,
    initial_files: Mapping[str, str],
    index: SkillsIndex,
    synthesizer: Synthesizer,
    config: LoopConfig | None = None,
    checkpoint_dir: Path | None = None,
) -> RunResult:
    cfg = config or LoopConfig()
    run_id = f"ralph_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    current: dict[str, str] = dict(initial_files)
    iterations: list[Iteration] = []
    consecutive_regressions = 0
    terminated_by: TerminationReason = "max_iters"

    for i in range(cfg.max_iters):
        score_before = _score_files(current, cfg)
        if score_before >= cfg.target_score:
            terminated_by = "target_hit"
            break

        citations = index.search(spec, top_k=cfg.top_k_citations)

        try:
            synth = synthesizer.synthesize(
                spec=spec, current_files=current, citations=citations, iteration=i,
            )
            synth_reason: IterationReason | None = None
        except Exception as e:  # noqa: BLE001 - synthesizer errors must not kill the loop
            _log.exception("synthesizer error: %s", e)
            synth = None
            synth_reason = "synthesizer_error"

        if synth is None:
            iterations.append(
                Iteration(
                    index=i, cited_node_ids=(), rationale=f"synth_error",
                    proposed_files=current, sandbox_score_before=score_before,
                    sandbox_score_after=score_before, kept=False,
                    reason=synth_reason or "synthesizer_error",
                )
            )
            consecutive_regressions += 1
        else:
            score_after = _score_files(synth.proposed_files, cfg)
            kept = score_after > score_before
            reason: IterationReason = "score_improved" if kept else "score_regressed"
            if kept:
                current = dict(synth.proposed_files)
                consecutive_regressions = 0
            else:
                consecutive_regressions += 1
            iterations.append(
                Iteration(
                    index=i,
                    cited_node_ids=synth.cited_node_ids,
                    rationale=synth.rationale,
                    proposed_files=synth.proposed_files,
                    sandbox_score_before=score_before,
                    sandbox_score_after=score_after,
                    kept=kept,
                    reason=reason,
                )
            )

        if checkpoint_dir is not None:
            try:
                _ = save_checkpoint(
                    RunResult(
                        run_id=run_id, spec=spec, started_at=started_at,
                        ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        final_score=iterations[-1].sandbox_score_after,
                        final_files=current, iterations=tuple(iterations),
                        terminated_by="max_iters",
                    ),
                    checkpoint_dir,
                )
            except OSError as e:
                _log.warning("checkpoint write failed: %s", e)

        if consecutive_regressions >= _STUCK_THRESHOLD:
            terminated_by = "stuck"
            break
    else:
        terminated_by = "max_iters"

    final_score = (
        iterations[-1].sandbox_score_after if iterations else _score_files(current, cfg)
    )
    result = RunResult(
        run_id=run_id, spec=spec, started_at=started_at,
        ended_at=datetime.now(UTC).isoformat(timespec="seconds"),
        final_score=final_score, final_files=current,
        iterations=tuple(iterations), terminated_by=terminated_by,
    )
    if checkpoint_dir is not None:
        try:
            save_checkpoint(result, checkpoint_dir)
        except OSError as e:
            _log.warning("final checkpoint write failed: %s", e)
    return result
```
- [ ] Run: PASS (4 tests).
- [ ] Commit: `feat(ralph-orchestrator): core run_loop with keep/revert and stuck detection`

---

## Task 8: CLI

- [ ] `tests/groundloop/ralph_orchestrator/test_cli.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from groundloop.ralph_orchestrator.cli import main


def test_cli_runs_with_stub(
    spec_path: Path, tiny_corpus_path: Path, tmp_path: Path, capsys,
):
    initial = tmp_path / "main.py"
    initial.write_text("from __future__ import annotations\n\n\ndef greet(n: str) -> str:\n    return 'hi'\n")
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tiny_corpus_path),
        "--initial-file", f"main.py={initial}",
        "--max-iters", "1",
        "--target-score", "1.1",
        "--synthesizer", "stub",
        "--format", "json",
    ])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "run_id" in data
    assert "terminated_by" in data


def test_cli_missing_corpus(tmp_path: Path, spec_path: Path):
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tmp_path / "nope.jsonl"),
        "--initial-file", f"main.py={tmp_path / 'whatever.py'}",
    ])
    assert rc == 1


def test_cli_missing_initial_file(tmp_path: Path, spec_path: Path, tiny_corpus_path: Path):
    rc = main([
        "run", str(spec_path),
        "--corpus", str(tiny_corpus_path),
        "--initial-file", f"main.py={tmp_path / 'missing.py'}",
    ])
    assert rc == 1
```
- [ ] Implement `groundloop/ralph_orchestrator/cli.py`:
```python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.ralph_orchestrator")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the Ralph loop")
    run.add_argument("spec_file", type=Path)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--initial-file", action="append", required=True,
                     help="Repeatable: relname=path/on/disk")
    run.add_argument("--max-iters", type=int, default=5)
    run.add_argument("--target-score", type=float, default=0.95)
    run.add_argument("--checkpoint-dir", type=Path, default=None)
    run.add_argument("--synthesizer", choices=("stub", "openai"), default="stub")
    run.add_argument("--format", choices=("text", "json"), default="text")
    return p.parse_args(argv)


def _build_synth(kind: str):
    if kind == "openai":
        from groundloop.ralph_orchestrator.openai_synthesizer import OpenAISynthesizer
        return OpenAISynthesizer()
    return StubSynthesizer()


def _load_initial(pairs: list[str]) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            return None
        name, path = pair.split("=", 1)
        p = Path(path)
        if not p.is_file():
            return None
        out[name] = p.read_text(encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    initial = _load_initial(args.initial_file)
    if initial is None:
        print("ERROR: one or more --initial-file paths missing or malformed", file=sys.stderr)
        return 1
    spec = args.spec_file.read_text(encoding="utf-8")

    idx = SkillsIndex(corpus_path=args.corpus)
    idx.build()

    try:
        synth = _build_synth(args.synthesizer)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    cfg = LoopConfig(max_iters=args.max_iters, target_score=args.target_score)
    result = run_loop(
        spec=spec, initial_files=initial, index=idx, synthesizer=synth,
        config=cfg, checkpoint_dir=args.checkpoint_dir,
    )
    if args.format == "json":
        print(result.model_dump_json())
    else:
        print(f"run_id={result.run_id} terminated_by={result.terminated_by} "
              f"final_score={result.final_score:.3f} iters={len(result.iterations)}")
    return 0
```
- [ ] `__main__.py`:
```python
from groundloop.ralph_orchestrator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] Run: PASS.
- [ ] Commit: `feat(ralph-orchestrator): CLI (stub + openai synthesizer backends)`

---

## Task 9: Public API + E2E + acceptance smoke

- [ ] `tests/groundloop/ralph_orchestrator/test_e2e.py`:
```python
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.ralph_orchestrator import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer


def test_e2e_stub_tiny_corpus(tiny_corpus_path: Path, initial_files: Mapping[str, str]):
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    synth = StubSynthesizer()
    result = run_loop(
        spec="Build a greet function using pytest patterns",
        initial_files=initial_files, index=idx, synthesizer=synth,
        config=LoopConfig(max_iters=2, target_score=1.1),
    )
    assert len(result.iterations) >= 1
    # Confirm at least one iteration cites a real tiny-corpus node_id
    seen = {nid for it in result.iterations for nid in it.cited_node_ids}
    assert seen  # non-empty
    assert result.final_score >= 0.0
```
- [ ] Populate `groundloop/ralph_orchestrator/__init__.py`:
```python
from __future__ import annotations

from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import (
    Iteration,
    LoopConfig,
    RunResult,
    SynthesisResult,
)
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer
from groundloop.ralph_orchestrator.synthesizer import Synthesizer

__all__ = [
    "Iteration",
    "LoopConfig",
    "RunResult",
    "StubSynthesizer",
    "SynthesisResult",
    "Synthesizer",
    "run_loop",
]
```
- [ ] Run full suite:
```
python3 -m pytest tests/groundloop/ralph_orchestrator/ -v --cov=groundloop.ralph_orchestrator --cov-report=term
ruff check groundloop/ralph_orchestrator/
mypy --strict groundloop/ralph_orchestrator/
```
Expect all pass, coverage ≥ 85%.
- [ ] CLI smoke test (capture output):
```
python3 -m groundloop.ralph_orchestrator run \
  tests/groundloop/ralph_orchestrator/fixtures/spec_simple.txt \
  --corpus tests/groundloop/kb_indexer/fixtures/tiny_corpus.jsonl \
  --initial-file main.py=tests/groundloop/ralph_orchestrator/fixtures/initial_files/main.py \
  --max-iters 2 --target-score 1.1 --synthesizer stub --format json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("iters=", len(d["iterations"]), "term=", d["terminated_by"])'
```
Expect: non-zero iterations, `terminated_by` in `{"max_iters","stuck","target_hit"}`.
- [ ] Append README subsection `### Ralph Orchestrator (autonomous loop)` describing stub vs openai synthesizers and env vars (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`).
- [ ] Commit: `feat(ralph-orchestrator): public API + e2e + README`

---

## Self-Review

- ✅ Every spec §4 component has an implementing task.
- ✅ All 9 §7 acceptance criteria: #1 (via test_loop timing, <5s by construction), #2 (stub determinism test), #3 (target-hit-early test), #4 (max-iters test), #5 (stuck test), #6 (checkpoint roundtrip test), #7 (ruff+mypy), #8 (coverage), #9 (e2e asserts cited node_ids).
- ✅ No placeholders.
- ✅ Type consistency: `SynthesisResult` fields same across stub + openai + loop consumers.
