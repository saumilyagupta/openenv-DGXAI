"""Custom trainer + dataset adapter (docs/modules/training.md §2.2, §3.2.3).

Two public types:

- :class:`EpisodeDatasetAdapter` — stateless iterable feeding
  ``GRPOTrainer.train_dataset``. Each ``__iter__`` tick yields
  ``{"prompt": str, "_meta": {...}}`` where ``_meta`` carries the
  ``GoalSpec``, the monotonically-derived ``episode_seed``, the curriculum
  ``stage``, and the ``language_weights``. One call to
  ``task_generator.generate`` per step; one call to
  ``tokenizer.apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True)`` to render the prompt.

- :class:`DriftCallGRPOTrainer` — ``GRPOTrainer`` subclass whose
  ``_generate_and_score_completions`` override runs G multi-turn episodes
  via a caller-provided ``RolloutGroupFn`` and plumbs the resulting
  frozen ``Episode`` tuple into ``reward_fn`` (step_13) before handing the
  G reward scalars + padded completions back to the inherited GRPO
  advantage / KL / optimizer step path. **The inherited code path is
  untouched** (training.md §3.2.3).

``trl`` and ``torch`` are imported lazily. Pure-Python fallbacks for
``_generate_and_score_completions`` are provided so the class shape
can be verified on CPU-only CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator


PINNED_SYSTEM_PROMPT: str = (
    "You are a concierge assistant. Use the provided tools. "
    "Respond in the caller's language. Submit with calibrated confidence."
)

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]


class EpisodeSampler(Protocol):
    """Draws a ``GoalSpec`` for one prompt slot (training.md §2.2)."""

    def __call__(self, step: int) -> Any: ...


class EnvFactory(Protocol):
    """Returns a fresh ``DriftCallEnv`` per rollout (training.md §3.2)."""

    def __call__(self) -> Any: ...


class RolloutGroupFn(Protocol):
    """Runs G multi-turn rollouts sharing one goal.

    Returns a tuple ``(episodes, completions)`` of length G each.
    """

    def __call__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        goal: Any,
        episode_seed: int,
        num_generations: int,
        env_factory: EnvFactory,
    ) -> tuple[tuple[Any, ...], tuple[str, ...]]: ...


@dataclass(frozen=True)
class AdapterRecord:
    """Frozen view of one :class:`EpisodeDatasetAdapter` yield.

    Tests consume this view rather than dict-typing ``_meta`` inline.
    """

    prompt: str
    goal: Any
    episode_seed: int
    stage: Literal[1, 2, 3]
    language_weights: dict[LanguageCode, float]


def render_initial_prompt(tokenizer: Any, goal: Any) -> str:
    """Render the turn-0 chat template (training.md §3.2.1).

    Messages: pinned system prompt + ``goal.seed_utterance`` as the user
    turn. ``add_generation_prompt=True`` tells Gemma to emit an assistant
    turn. Tool schemas live in later turns so only these two messages
    appear at ``step == 0``.
    """
    seed_utterance = getattr(goal, "seed_utterance", "")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": PINNED_SYSTEM_PROMPT},
        {"role": "user", "content": seed_utterance},
    ]
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return str(result)


class EpisodeDatasetAdapter:
    """Stateless streaming dataset (training.md §2.2).

    Constructor signature matches training.md §2.2: a ``task_gen`` callable
    accepting ``(seed, stage, language_weights)``, an ``env_factory``
    producing fresh envs, the curriculum ``stage``, a ``stage_base_seed``
    used to derive per-step ``episode_seed``, the per-language sampling
    ``language_weights``, and the ``tokenizer`` used to render prompts.

    Iteration is infinite — exactly one record per GRPO training step.
    Step counter is local to ``__iter__`` so resume simply restarts from
    whatever step TRL's ``resume_from_checkpoint`` restores.
    """

    def __init__(
        self,
        *,
        task_gen: Callable[..., Any],
        env_factory: EnvFactory,
        stage: Literal[1, 2, 3],
        stage_base_seed: int,
        language_weights: dict[LanguageCode, float],
        tokenizer: Any,
    ) -> None:
        self.task_gen = task_gen
        self.env_factory = env_factory
        self.stage: Literal[1, 2, 3] = stage
        self.stage_base_seed = stage_base_seed
        self.language_weights = dict(language_weights)
        self.tokenizer = tokenizer

    def _build_record(self, step: int) -> dict[str, Any]:
        episode_seed = self.stage_base_seed + step
        goal = self.task_gen(
            seed=episode_seed,
            stage=self.stage,
            language_weights=self.language_weights,
        )
        prompt = render_initial_prompt(self.tokenizer, goal)
        return {
            "prompt": prompt,
            "_meta": {
                "goal": goal,
                "episode_seed": episode_seed,
                "stage": self.stage,
                "language_weights": dict(self.language_weights),
            },
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        step = 0
        while True:
            yield self._build_record(step)
            step += 1

    def peek(self, step: int) -> AdapterRecord:
        """Materialize the record at ``step`` without advancing iteration.

        Used by tests (§1.2 U14–U18) to assert record shape at arbitrary
        steps without consuming a generator.
        """
        rec = self._build_record(step)
        meta = rec["_meta"]
        return AdapterRecord(
            prompt=rec["prompt"],
            goal=meta["goal"],
            episode_seed=meta["episode_seed"],
            stage=meta["stage"],
            language_weights=meta["language_weights"],
        )


def _import_grpo_trainer() -> type[Any]:
    """Lazy import of ``trl.GRPOTrainer``; isolated for mocking in tests."""
    from trl import GRPOTrainer

    return cast("type[Any]", GRPOTrainer)


def _make_driftcall_init(
    base_cls: type[Any],
) -> Callable[..., None]:
    """Build an ``__init__`` bound to ``base_cls``; avoids super() recursion
    when the returned class is itself further subclassed."""

    def _init(
        self: Any,
        *args: Any,
        rollout_group_fn: RolloutGroupFn,
        env_factory: EnvFactory,
        reward_fn_driftcall: Callable[..., list[float]],
        **kwargs: Any,
    ) -> None:
        base_cls.__init__(self, *args, **kwargs)
        self.rollout_group_fn = rollout_group_fn
        self.env_factory = env_factory
        self.reward_fn_driftcall = reward_fn_driftcall

    return _init


def _driftcall_generate_and_score_completions(
    self: Any, inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run the multi-turn rollout, then call ``reward_fn``.

    Expects ``inputs`` to carry one row per prompt slot with the
    ``_meta`` dict produced by :class:`EpisodeDatasetAdapter`.
    Returns a dict with keys ``episodes``, ``completions``, ``rewards``,
    ``prompts`` — each length G (num_generations).
    """
    if not inputs:
        raise ValueError("inputs must be a non-empty list")

    row = inputs[0]
    meta = row["_meta"]
    prompt = row["prompt"]
    goal = meta["goal"]
    episode_seed = meta["episode_seed"]

    num_generations = int(getattr(self.args, "num_generations", 8))
    episodes, completions = self.rollout_group_fn(
        model=self.model,
        tokenizer=self.processing_class,
        goal=goal,
        episode_seed=episode_seed,
        num_generations=num_generations,
        env_factory=self.env_factory,
    )

    if len(episodes) != num_generations or len(completions) != num_generations:
        raise ValueError(
            f"rollout_group_fn produced {len(episodes)} episodes and "
            f"{len(completions)} completions; expected {num_generations} each"
        )

    prompts = [prompt] * num_generations
    metas = [dict(meta) for _ in range(num_generations)]
    rewards = self.reward_fn_driftcall(
        prompts=prompts,
        completions=list(completions),
        _meta=metas,
        episodes=list(episodes),
    )

    return {
        "episodes": episodes,
        "completions": completions,
        "rewards": rewards,
        "prompts": prompts,
    }


def make_driftcall_grpo_trainer_cls(base_cls: type[Any] | None = None) -> type[Any]:
    """Build the :class:`DriftCallGRPOTrainer` class bound to ``base_cls``.

    Default ``base_cls`` is ``trl.GRPOTrainer`` (imported lazily). Tests
    pass a stub base class so they can exercise the override path without
    TRL installed.

    GRPOTrainer subclass with multi-turn rollout override
    (training.md §3.2.3). Construction adds three DriftCall-specific
    kwargs over the standard ``GRPOTrainer.__init__``:

    - ``rollout_group_fn``: :class:`RolloutGroupFn` running G multi-turn
      episodes and returning ``(episodes, completions)``.
    - ``env_factory``: :class:`EnvFactory` producing a fresh
      ``DriftCallEnv`` per rollout.
    - ``reward_fn_driftcall``: the step_13 ``reward_fn`` — called
      directly with the frozen ``Episode`` tuple after rollout.

    ``_generate_and_score_completions`` replaces the TRL default.
    Advantage + KL + optimizer step paths are inherited unchanged.
    """
    resolved_base: type[Any] = (
        base_cls if base_cls is not None else _import_grpo_trainer()
    )
    return type(
        "DriftCallGRPOTrainer",
        (resolved_base,),
        {
            "__init__": _make_driftcall_init(resolved_base),
            "_generate_and_score_completions": _driftcall_generate_and_score_completions,
            "__doc__": "GRPOTrainer subclass with multi-turn rollout override.",
        },
    )


def driftcall_grpo_trainer_methods() -> tuple[str, ...]:
    """Return the method names the subclass overrides (introspection helper).

    Used by the shape test (U in §1.x) to verify the override surface.
    """
    return ("__init__", "_generate_and_score_completions")


__all__ = [
    "AdapterRecord",
    "EnvFactory",
    "EpisodeDatasetAdapter",
    "EpisodeSampler",
    "LanguageCode",
    "PINNED_SYSTEM_PROMPT",
    "RolloutGroupFn",
    "driftcall_grpo_trainer_methods",
    "make_driftcall_grpo_trainer_cls",
    "render_initial_prompt",
]
