from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment

from models import CodeForgeAction, CodeForgeActionType, CodeForgeObservation
from groundloop.kb_indexer.index import SkillsIndex
from groundloop.lib_grounder.grounder import ground
from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop_env.grader import compute_reward
from groundloop_env.observation_builder import build_observation
from groundloop_env.tasks import Task, get_task

_log = logging.getLogger(__name__)
_DEFAULT_CORPUS = Path("groundloop/kb/skills_corpus.jsonl")


class CodeForgeEnvironment(Environment):
    def __init__(self, *, corpus_path: Path | None = None) -> None:
        super().__init__()
        self._corpus_path = corpus_path or _DEFAULT_CORPUS
        self._index: SkillsIndex | None = None
        self._task: Task | None = None
        self._episode_id: str = ""
        self._budget_remaining: int = 0
        self._current_files: dict[str, str] = {}
        self._previous_score: float = 0.0
        self._last_citations: tuple[dict[str, object], ...] = ()
        self._last_grounding: dict[str, object] | None = None
        self._is_done: bool = False
        self._last_reward: float = 0.0

    def _ensure_index(self) -> SkillsIndex:
        if self._index is None:
            if not self._corpus_path.is_file():
                msg = (
                    f"corpus not found: {self._corpus_path}. "
                    f"Run `python3 -m groundloop.skills_scraper` first."
                )
                raise FileNotFoundError(msg)
            idx = SkillsIndex(corpus_path=self._corpus_path)
            idx.build()
            self._index = idx
        return self._index

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> CodeForgeObservation:
        task_level = kwargs.get("task_level", "easy")
        task = get_task(task_level)
        self._task = task
        self._episode_id = episode_id or uuid.uuid4().hex[:12]
        self._budget_remaining = task.max_budget
        self._current_files = dict(task.initial_files)
        self._previous_score = 0.0
        self._last_citations = ()
        self._last_grounding = None
        self._is_done = False
        self._last_reward = 0.0
        _log.info("reset id=%s task=%s budget=%s", self._episode_id, task.task_id, task.max_budget)
        return self._build_obs()

    def step(
        self,
        action: CodeForgeAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CodeForgeObservation:
        if self._is_done or self._task is None:
            return self._build_obs()

        self._budget_remaining -= 1

        if action.action_type == CodeForgeActionType.QUERY_KB:
            self._handle_query(action)
            self._last_reward = 0.0
        elif action.action_type == CodeForgeActionType.SUBMIT:
            self._handle_submit(action)

        if self._budget_remaining <= 0:
            self._is_done = True
        return self._build_obs()

    @property
    def state(self) -> CodeForgeObservation:
        return self._build_obs()

    def _handle_query(self, action: CodeForgeAction) -> None:
        try:
            idx = self._ensure_index()
        except FileNotFoundError as e:
            _log.warning("query: no corpus: %s", e)
            self._last_citations = ()
            return
        tags = set(action.required_tags) if action.required_tags else None
        results = idx.search(action.claim or "", top_k=action.top_k, required_tags=tags)
        self._last_citations = tuple(
            {
                "node_id": r.node_id,
                "skill_name": r.skill_name,
                "section_path": list(r.section_path),
                "section_body": r.section_body,
                "score": r.score,
                "rank": r.rank,
            }
            for r in results
        )

    def _handle_submit(self, action: CodeForgeAction) -> None:
        if action.files is None:
            self._last_reward = 0.0
            return
        self._current_files = dict(action.files)
        assert self._task is not None  # noqa: S101
        try:
            sandbox_result = run_sandbox(
                files=dict(action.files),
                tools=self._task.tools,
                timeout_per_tool=30.0,
            )
            sandbox_score = sandbox_result.composite_score
        except Exception as e:  # noqa: BLE001 - env must never crash the server
            _log.exception("sandbox error: %s", e)
            sandbox_score = 0.0

        concatenated = "\n".join(action.files.values())
        grounding_report = ground(concatenated)
        self._last_grounding = grounding_report.model_dump()
        reward = compute_reward(
            sandbox_score=sandbox_score,
            groundedness=grounding_report.groundedness,
            confidence=action.confidence,
        )
        self._last_reward = reward
        self._previous_score = reward
        assert self._task is not None  # noqa: S101 - protected by outer guard
        if reward >= self._task.target_score:
            self._is_done = True

    def _build_obs(self) -> CodeForgeObservation:
        assert self._task is not None  # noqa: S101
        return build_observation(
            episode_id=self._episode_id,
            task=self._task,
            current_files=self._current_files,
            budget_remaining=self._budget_remaining,
            previous_score=self._previous_score,
            last_citations=self._last_citations,
            last_grounding=self._last_grounding,
            is_done=self._is_done,
            last_reward=self._last_reward,
        )
