from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment

from codeforge.audit.ledger import AuditLedger
from codeforge.grader import compute_reward
from codeforge.grounder import ground
from codeforge.interrogator.interrogator import Interrogator
from codeforge.kb.cluster import build_clusters
from codeforge.kb.indexer import SkillsIndex
from codeforge.models import AuditEntry, CodeForgeAction, CodeForgeActionType, CodeForgeObservation
from codeforge.observation import build_observation
from codeforge.ralph.loop import run_loop
from codeforge.ralph.models import LoopConfig
from codeforge.ralph.synthesizer import StubSynthesizer
from codeforge.sandbox.sandbox import run_sandbox
from codeforge.shaping import citation_shaping_bonus
from codeforge.tasks import Task, get_task

_log = logging.getLogger(__name__)
_DEFAULT_CORPUS = Path(__file__).resolve().parent / "kb" / "skills_corpus.jsonl"

# ---------------------------------------------------------------------------
# Filename validation (SYSTEM_DESIGN §14.2, §14.3)
# ---------------------------------------------------------------------------
_FILENAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.py$")
_FORBIDDEN_FILENAMES = frozenset({
    "conftest.py", "pytest.ini", "setup.cfg", "pyproject.toml", "tox.ini",
})
_MAX_FILES = 10
_MAX_FILE_SIZE = 50 * 1024  # 50 KB
_MAX_TOTAL_SIZE = 200 * 1024  # 200 KB


def _validate_files(files: dict[str, str]) -> str | None:
    """Return an error message if *files* violates submission constraints, else None."""
    if not files:
        return "files dict is empty"
    if len(files) > _MAX_FILES:
        return f"too many files ({len(files)} > {_MAX_FILES})"
    total_size = 0
    for name, content in files.items():
        if name in _FORBIDDEN_FILENAMES:
            return f"filename '{name}' is not allowed"
        if not _FILENAME_RE.match(name):
            return f"filename '{name}' must match [a-z][a-z0-9_]*.py"
        size = len(content.encode("utf-8"))
        if size > _MAX_FILE_SIZE:
            return f"file '{name}' exceeds {_MAX_FILE_SIZE} bytes"
        total_size += size
    if total_size > _MAX_TOTAL_SIZE:
        return f"total size ({total_size}) exceeds {_MAX_TOTAL_SIZE} bytes"
    return None


# ---------------------------------------------------------------------------
# Valid action types (for fast membership check)
# ---------------------------------------------------------------------------
_VALID_ACTION_TYPES = frozenset(member.value for member in CodeForgeActionType)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CodeForgeEnvironment(Environment):  # type: ignore[type-arg]
    """OpenEnv-compliant RL environment with all 6 CodeForge actions.

    Implements SYSTEM_DESIGN §4.9, §5.2, §17.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, corpus_path: Path | None = None) -> None:
        super().__init__()
        self._corpus_path = corpus_path or _DEFAULT_CORPUS
        self._index: SkillsIndex | None = None
        self._task: Task | None = None
        self._episode_id: str = ""
        self._budget_remaining: int = 0
        self._current_files: dict[str, str] = {}
        self._previous_score: float = 0.0
        self._is_done: bool = False

        # Per-step state
        self._last_citations: tuple[dict[str, object], ...] = ()
        self._last_grounding: dict[str, object] | None = None
        self._last_reward: float = 0.0
        self._last_cluster_hits: tuple[str, ...] = ()
        self._last_interrogation_questions: tuple[str, ...] = ()
        self._last_ralph_run_id: str | None = None
        self._last_ralph_iterations: tuple[dict[str, object], ...] = ()

        # Brier/quality tracking for audit entries
        self._last_brier_penalty: float | None = None
        self._last_quality: float = 0.0

        # Episode-level accumulators
        self._all_episode_citations: list[dict[str, object]] = []
        self._all_episode_cluster_hits: list[str] = []
        self._ledger: AuditLedger | None = None
        self._step_index: int = 0

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> SkillsIndex:
        if self._index is None:
            if not self._corpus_path.is_file():
                msg = (
                    f"corpus not found: {self._corpus_path}. "
                    f"Run the skills scraper first."
                )
                raise FileNotFoundError(msg)
            idx = SkillsIndex(corpus_path=self._corpus_path)
            idx.build()
            # Build and attach clusters
            import json
            nodes: list[dict[str, Any]] = []
            with self._corpus_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        nodes.append(json.loads(line))
            manifest = build_clusters(nodes)
            idx.attach_cluster_manifest(manifest)
            self._index = idx
        return self._index

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> CodeForgeObservation:
        task_level: str = kwargs.get("task_level", "easy")
        task = get_task(task_level)
        self._task = task
        self._episode_id = episode_id or uuid.uuid4().hex[:12]
        self._budget_remaining = task.max_budget
        self._current_files = dict(task.initial_files)
        self._previous_score = 0.0
        self._is_done = False

        # Reset per-step
        self._last_citations = ()
        self._last_grounding = None
        self._last_reward = 0.0
        self._last_cluster_hits = ()
        self._last_interrogation_questions = ()
        self._last_ralph_run_id = None
        self._last_ralph_iterations = ()

        # Reset episode accumulators
        self._all_episode_citations = []
        self._all_episode_cluster_hits = []
        self._ledger = AuditLedger()
        self._step_index = 0

        _log.info(
            "reset id=%s task=%s budget=%s",
            self._episode_id, task.task_id, task.max_budget,
        )
        return self._build_obs()

    def step(
        self,
        action: CodeForgeAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CodeForgeObservation:
        # --- Pre-check: no active episode --------------------------------
        if self._task is None:
            return self._error_obs("No active episode — call reset() first")

        # --- Pre-check: episode already done -----------------------------
        if self._is_done:
            return self._build_obs()

        # --- Pre-check: valid action_type --------------------------------
        action_type_str = str(action.action_type)
        if action_type_str not in _VALID_ACTION_TYPES:
            return self._error_obs(f"Unknown action_type: {action_type_str!r}")

        # --- Budget check (variable cost) --------------------------------
        cost = self._action_cost(action)
        if cost > self._budget_remaining:
            return self._error_obs(
                f"Insufficient budget: need {cost}, have {self._budget_remaining}"
            )
        self._budget_remaining -= cost

        # --- Clear per-step state ----------------------------------------
        self._last_reward = 0.0
        self._last_citations = ()
        self._last_grounding = None
        self._last_cluster_hits = ()
        self._last_interrogation_questions = ()
        self._last_ralph_run_id = None
        self._last_ralph_iterations = ()
        error: str | None = None

        # --- Route to handler --------------------------------------------
        try:
            if action_type_str == CodeForgeActionType.QUERY_KB:
                error = self._handle_query_kb(action)
            elif action_type_str == CodeForgeActionType.QUERY_CLUSTER:
                error = self._handle_query_cluster(action)
            elif action_type_str == CodeForgeActionType.INTERROGATE:
                error = self._handle_interrogate(action)
            elif action_type_str == CodeForgeActionType.SUBMIT:
                error = self._handle_submit(action)
            elif action_type_str == CodeForgeActionType.RUN_RALPH:
                error = self._handle_run_ralph(action)
            elif action_type_str == CodeForgeActionType.GET_AUDIT:
                error = self._handle_get_audit(action)
        except Exception as exc:
            _log.exception("handler error: %s", exc)
            error = f"Internal error: {exc}"

        # --- Append audit entry ------------------------------------------
        assert self._ledger is not None
        _cited: list[str] = []
        _cite: dict[str, object]
        for _cite in self._last_citations:
            _cited.append(str(_cite.get("node_id", "")))
        cited_ids: tuple[str, ...] = tuple(_cited)
        self._ledger.append(
            AuditEntry(
                step_index=self._step_index,
                action_type=action_type_str,
                cited_skill_ids=cited_ids,
                cited_clusters=self._last_cluster_hits,
                grounding_report=(
                    self._last_grounding if self._last_grounding else None
                ),
                reward=self._last_reward,
                brier_penalty=(
                    self._last_brier_penalty
                    if action_type_str == CodeForgeActionType.SUBMIT
                    else None
                ),
                confidence_declared=(
                    action.confidence
                    if action_type_str == CodeForgeActionType.SUBMIT
                    else None
                ),
                quality=(
                    self._last_quality
                    if action_type_str == CodeForgeActionType.SUBMIT
                    else self._previous_score
                ),
            ),
        )
        self._step_index += 1

        # --- Check budget exhaustion -------------------------------------
        if self._budget_remaining <= 0:
            self._is_done = True

        return self._build_obs(error=error)

    @property
    def state(self) -> CodeForgeObservation:
        if self._task is None:
            return self._error_obs("No active episode — call reset() first")
        return self._build_obs()

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    @staticmethod
    def _action_cost(action: CodeForgeAction) -> int:
        """Variable-cost budget accounting (SYSTEM_DESIGN §17.2)."""
        if str(action.action_type) == CodeForgeActionType.GET_AUDIT:
            return 0
        if str(action.action_type) == CodeForgeActionType.RUN_RALPH:
            return action.max_iters
        return 1

    # ------------------------------------------------------------------
    # Action handlers (each returns an error string or None)
    # ------------------------------------------------------------------

    def _handle_query_kb(self, action: CodeForgeAction) -> str | None:
        try:
            idx = self._ensure_index()
        except FileNotFoundError as e:
            _log.warning("query_kb: no corpus: %s", e)
            self._last_citations = ()
            return None
        tags = set(action.required_tags) if action.required_tags else None
        results = idx.search(
            action.claim or "", top_k=action.top_k, required_tags=tags,
        )
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
        self._all_episode_citations.extend(self._last_citations)
        return None

    def _handle_query_cluster(self, action: CodeForgeAction) -> str | None:
        try:
            idx = self._ensure_index()
        except FileNotFoundError as e:
            _log.warning("query_cluster: no corpus: %s", e)
            self._last_cluster_hits = ()
            return None
        label = action.cluster_label or ""
        results = idx.nodes_in_cluster(label)
        if not results:
            self._last_cluster_hits = ()
            return None
        self._last_cluster_hits = tuple(r.node_id for r in results)
        self._all_episode_cluster_hits.extend(self._last_cluster_hits)
        return None

    def _handle_interrogate(self, action: CodeForgeAction) -> str | None:
        idx: SkillsIndex | None
        try:
            idx = self._ensure_index()
        except FileNotFoundError:
            idx = None
        interrogator = Interrogator(idx)
        assert self._task is not None
        result = interrogator.generate(self._task.brief)
        self._last_interrogation_questions = result.questions
        return None

    def _handle_submit(self, action: CodeForgeAction) -> str | None:
        if action.files is None:
            return "files required for submit"
        file_err = _validate_files(action.files)
        if file_err is not None:
            return file_err

        self._current_files = dict(action.files)
        assert self._task is not None
        # Run sandbox
        try:
            sandbox_result = run_sandbox(
                files=dict(action.files),
                tools=self._task.tools,
                timeout_per_tool=30.0,
            )
            sandbox_score = sandbox_result.composite_score
        except Exception as e:
            _log.exception("sandbox error: %s", e)
            sandbox_score = 0.0

        # Run grounder
        concatenated = "\n".join(action.files.values())
        grounding_report = ground(concatenated)
        self._last_grounding = grounding_report.model_dump()

        # Compute reward with Brier calibration
        quality = 0.6 * sandbox_score + 0.4 * grounding_report.groundedness
        brier_penalty: float | None = None
        if action.confidence is not None:
            brier_penalty = min((action.confidence - quality) ** 2, 0.5)
        self._last_brier_penalty = brier_penalty
        self._last_quality = quality

        reward = compute_reward(
            sandbox_score=sandbox_score,
            groundedness=grounding_report.groundedness,
            confidence=action.confidence,
        )

        # Apply citation shaping bonus only on successful submits (§4.8.4)
        if reward > 0:
            shaping = citation_shaping_bonus(
                submit_files=action.files,
                prior_citations=self._all_episode_citations,
                prior_cluster_hits=self._all_episode_cluster_hits,
            )
            reward = round(min(1.0, reward + shaping), 3)

        self._last_reward = reward
        self._previous_score = reward

        # Check target score
        if reward >= self._task.target_score:
            self._is_done = True

        return None

    def _handle_run_ralph(self, action: CodeForgeAction) -> str | None:
        assert self._task is not None
        try:
            idx = self._ensure_index()
        except FileNotFoundError as e:
            return f"corpus not available: {e}"

        config = LoopConfig(
            max_iters=action.max_iters,
            target_score=self._task.target_score,
            tools=self._task.tools,
        )
        synthesizer = StubSynthesizer()
        result = run_loop(
            spec=self._task.brief,
            initial_files=self._current_files,
            index=idx,
            synthesizer=synthesizer,
            config=config,
        )

        self._last_ralph_run_id = result.run_id
        self._last_ralph_iterations = tuple(
            it.model_dump() for it in result.iterations
        )
        self._current_files = dict(result.final_files)

        # Compute ralph reward (SYSTEM_DESIGN §4.8.5)
        concatenated = "\n".join(result.final_files.values())
        grounding_report = ground(concatenated)
        self._last_grounding = grounding_report.model_dump()

        wasted = sum(
            1 for it in result.iterations if it.reason in ("score_regressed", "score_plateau")
        )
        base = compute_reward(
            sandbox_score=result.final_score,
            groundedness=grounding_report.groundedness,
            confidence=0.75,
        )
        waste_penalty = wasted * 0.05
        ralph_reward = round(max(0.0, min(1.0, base - waste_penalty)), 3)

        self._last_reward = ralph_reward
        self._previous_score = ralph_reward
        return None

    def _handle_get_audit(self, action: CodeForgeAction) -> str | None:
        # Audit data is populated in _build_obs via cumulative_audit_summary
        return None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _build_obs(self, *, error: str | None = None) -> CodeForgeObservation:
        assert self._task is not None
        audit_summary: dict[str, object] | None = None
        if self._ledger is not None:
            audit_summary = self._ledger.serialize()
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
            last_cluster_hits=self._last_cluster_hits,
            last_interrogation_questions=self._last_interrogation_questions,
            last_ralph_run_id=self._last_ralph_run_id,
            last_ralph_iterations=self._last_ralph_iterations,
            cumulative_audit_summary=audit_summary,
            error=error,
        )

    def _error_obs(self, msg: str) -> CodeForgeObservation:
        """Return an error observation without modifying episode state."""
        if self._task is None:
            # No task set — use a dummy task for the observation structure
            dummy = get_task("easy")
            return build_observation(
                episode_id=self._episode_id or "none",
                task=dummy,
                current_files=self._current_files,
                budget_remaining=self._budget_remaining,
                previous_score=self._previous_score,
                is_done=self._is_done,
                error=msg,
            )
        return self._build_obs(error=msg)
