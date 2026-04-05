import uuid
import random
import logging
from typing import Dict, Any, Optional

from openenv.core.env_server.interfaces import Environment
from models import EpistemicAction, EpistemicObservation, ActionType, EvidenceSnippet
from server.retriever import BM25Retriever
from server.grader import compute_reward
import json

logger = logging.getLogger(__name__)


class EpistemicNavEnvironment(Environment):
    def __init__(self, max_budget: int = 8, data_dir: str = "data"):
        super().__init__()
        self.max_budget = max_budget
        self.retriever = BM25Retriever(f"{data_dir}/evidence.json")
        self.claims = self._load_claims(f"{data_dir}/claims.json")

        # State
        self.current_claim: Optional[Dict[str, Any]] = None
        self.evidence_gathered: list[EvidenceSnippet] = []
        self.budget_remaining: int = max_budget
        self.episode_id: str = ""
        self._is_done: bool = False
        self._last_reward: Optional[float] = None

    def _load_claims(self, path: str) -> list[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> EpistemicObservation:
        task_level = kwargs.get("task_level", "easy")

        # Filter claims by task level
        level_claims = [c for c in self.claims if c.get("task_level") == task_level]
        if not level_claims:
            # Fallback if specific level is not found
            level_claims = self.claims if self.claims else [
                {"id": "claim_fallback", "text": "The sky is blue.", "ground_truth": "true", "task_level": task_level}
            ]

        self.current_claim = random.choice(level_claims)
        self.evidence_gathered = []
        self.budget_remaining = self.max_budget
        self.episode_id = str(uuid.uuid4())
        self._is_done = False
        self._last_reward = None
        logger.info(
            "episode_reset id=%s task_level=%s claim_id=%s budget=%s",
            self.episode_id,
            task_level,
            self.current_claim.get("id", "unknown"),
            self.budget_remaining,
        )

        return self._get_observation()

    def step(
        self,
        action: EpistemicAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EpistemicObservation:
        # Guard: no actions after episode is done
        if self._is_done:
            logger.warning("step_after_done id=%s — call /reset first", self.episode_id)
            return self._get_observation(is_done=True, last_reward=self._last_reward)

        # Validate commit fields
        if action.action_type == ActionType.COMMIT:
            if action.verdict and action.verdict not in ("true", "false", "uncertain"):
                logger.warning("invalid_verdict id=%s verdict=%r — defaulting to uncertain", self.episode_id, action.verdict)
                action = EpistemicAction(
                    action_type=ActionType.COMMIT,
                    verdict="uncertain",
                    confidence=action.confidence if action.confidence is not None else 0.5,
                )

        if self.budget_remaining <= 0 and action.action_type != ActionType.COMMIT:
            # Force commit if budget is 0
            logger.info(
                "budget_exhausted id=%s forcing_commit verdict=uncertain confidence=0.5",
                self.episode_id,
            )
            action = EpistemicAction(
                action_type=ActionType.COMMIT,
                verdict="uncertain",
                confidence=0.5
            )

        reward = 0.0
        done = False

        if action.action_type == ActionType.QUERY:
            self.budget_remaining -= 1
            added_snippets = 0
            total_relevance = 0.0
            if action.query_text:
                results = self.retriever.search(action.query_text)
                for res in results:
                    snippet = EvidenceSnippet(
                        id=res["id"],
                        text=res["text"],
                        relevance_score=res["relevance_score"]
                    )
                    if not any(e.id == snippet.id for e in self.evidence_gathered):
                        self.evidence_gathered.append(snippet)
                        added_snippets += 1
                        total_relevance += res["relevance_score"]

            # Intermediate reward: small signal based on evidence quality
            # - Novel evidence with high relevance scores higher
            # - Duplicate/low-relevance queries score near zero
            # - Capped at 0.01 per query, max ~0.08 over 8 queries
            if added_snippets > 0 and total_relevance > 0:
                relevance_signal = min(total_relevance / (added_snippets * 20.0), 1.0)
                reward = round(0.01 * relevance_signal * min(added_snippets, 3) / 3.0, 4)
                reward = min(reward, 0.01)
            else:
                reward = 0.0
            done = False
            logger.info(
                "episode_step id=%s action=query query=%r added_evidence=%s total_evidence=%s budget_remaining=%s reward=%.3f done=%s",
                self.episode_id,
                action.query_text or "",
                added_snippets,
                len(self.evidence_gathered),
                self.budget_remaining,
                reward,
                done,
            )

        elif action.action_type == ActionType.COMMIT:
            ground_truth = self.current_claim.get("ground_truth", "uncertain")
            reward = compute_reward(
                verdict=action.verdict or "uncertain",
                confidence=action.confidence if action.confidence is not None else 0.5,
                ground_truth=ground_truth,
                budget_remaining=self.budget_remaining,
                max_budget=self.max_budget
            )
            done = True
            logger.info(
                "episode_step id=%s action=commit verdict=%s confidence=%.3f ground_truth=%s budget_remaining=%s reward=%.3f done=%s",
                self.episode_id,
                action.verdict or "uncertain",
                action.confidence if action.confidence is not None else 0.5,
                ground_truth,
                self.budget_remaining,
                reward,
                done,
            )

        self._is_done = done
        self._last_reward = reward
        obs = self._get_observation(is_done=done, last_reward=reward)
        obs.done = done
        obs.reward = reward
        return obs

    def _get_observation(self, is_done: bool = False, last_reward: float = None) -> EpistemicObservation:
        return EpistemicObservation(
            claim=self.current_claim["text"] if self.current_claim else "",
            evidence_gathered=self.evidence_gathered,
            budget_remaining=self.budget_remaining,
            task_level=self.current_claim.get("task_level", "easy") if self.current_claim else "easy",
            episode_id=self.episode_id,
            is_done=is_done,
            last_reward=last_reward
        )

    @property
    def state(self) -> EpistemicObservation:
        return self._get_observation(is_done=self._is_done, last_reward=self._last_reward)
