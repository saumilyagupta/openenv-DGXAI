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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

from cells.step_13_grpo_config import BETA_KL

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
    when the returned class is itself further subclassed.

    DriftCall-specific kwargs added on top of ``GRPOTrainer.__init__``:

    - ``rollout_group_fn``, ``env_factory``, ``reward_fn_driftcall`` — the
      multi-turn rollout override surface (see class docstring).
    - ``enable_adaptive_kl`` (default ``True``) — auto-attach an
      :class:`AdaptiveKLCallback` so β retargets to the measured KL each
      logging tick (training.md §3.3.1). Set ``False`` to disable.
    - ``adaptive_kl_target`` — override the default ``target_kl=BETA_KL``.
    - ``adaptive_kl_kp`` — override the proportional gain.
    - ``adaptive_kl_beta_min`` / ``adaptive_kl_beta_max`` — override clamp
      bounds.
    """

    def _init(
        self: Any,
        *args: Any,
        rollout_group_fn: RolloutGroupFn,
        env_factory: EnvFactory,
        reward_fn_driftcall: Callable[..., list[float]],
        enable_adaptive_kl: bool = True,
        adaptive_kl_target: float | None = None,
        adaptive_kl_kp: float = DEFAULT_KP,
        adaptive_kl_beta_min: float = DEFAULT_BETA_MIN,
        adaptive_kl_beta_max: float = DEFAULT_BETA_MAX,
        **kwargs: Any,
    ) -> None:
        base_cls.__init__(self, *args, **kwargs)
        self.rollout_group_fn = rollout_group_fn
        self.env_factory = env_factory
        self.reward_fn_driftcall = reward_fn_driftcall

        if enable_adaptive_kl:
            target = (
                adaptive_kl_target if adaptive_kl_target is not None else BETA_KL
            )
            callback = AdaptiveKLCallback(
                target_kl=target,
                kp=adaptive_kl_kp,
                beta_min=adaptive_kl_beta_min,
                beta_max=adaptive_kl_beta_max,
            )
            self.adaptive_kl_callback = callback
            add_callback = getattr(base_cls, "add_callback", None)
            if callable(add_callback):
                # Production path (TRL ≥ 0.23): register through the TRL
                # callback handler so ``on_log`` fires alongside default
                # loggers with the correct ``args``/``state``/``control``.
                self.add_callback(callback)
            else:
                # Fallback: minimal bases in tests lack ``add_callback``.
                # Keep a private list so callers can still invoke the hook.
                if not hasattr(self, "_driftcall_callbacks"):
                    self._driftcall_callbacks = []
                self._driftcall_callbacks.append(callback)
        else:
            self.adaptive_kl_callback = None

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


def _derive_rollout_seed(episode_seed: int, g_index: int) -> int:
    """Stable per-rollout seed (training.md §3.2: ``derive_seed(goal, g_index)``).

    Mixed deterministically so the drift schedule for rollout ``g`` is fixed
    across runs but each of the G rollouts in a group sees different drift
    timing.
    """
    return (episode_seed * 1_000_003 + g_index) & 0xFFFFFFFF


def _parse_action_from_completion(text: str) -> Any:
    """Parse the model's assistant turn into a :class:`DriftCallAction`.

    Pulls the first JSON object from ``text`` (the chat template wraps the
    assistant turn in code-fence-or-bare JSON; we accept either). Falls back
    to an ``ABORT`` action when the completion is unparseable so the env
    surfaces a format violation through R4 instead of crashing the rollout
    (training.md §3.2 ``EpisodeParseError`` is logged, episode continues).
    """
    import json as _json

    from cells.step_04_models import ActionType, DriftCallAction

    snippet = text.strip()
    start = snippet.find("{")
    end = snippet.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return DriftCallAction(action_type=ActionType.ABORT)
    try:
        payload = _json.loads(snippet[start : end + 1])
    except _json.JSONDecodeError:
        return DriftCallAction(action_type=ActionType.ABORT)
    if not isinstance(payload, dict):
        return DriftCallAction(action_type=ActionType.ABORT)

    action_type_raw = payload.get("action_type")
    try:
        action_type = ActionType(action_type_raw) if action_type_raw is not None else ActionType.ABORT
    except ValueError:
        action_type = ActionType.ABORT
    return DriftCallAction(
        action_type=action_type,
        tool_name=payload.get("tool_name"),
        tool_args=payload.get("tool_args"),
        message=payload.get("message"),
        confidence=payload.get("confidence"),
        rationale=payload.get("rationale"),
    )


def rollout_group(
    *,
    model: Any,
    tokenizer: Any,
    goal: Any,
    episode_seed: int,
    num_generations: int,
    env_factory: EnvFactory,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Run G=``num_generations`` independent multi-turn rollouts sharing one goal.

    Implements the contract in training.md §2.2 ``rollout_group``:

    1. For each ``g_index`` in ``[0, G)``: instantiate a fresh env via
       ``env_factory()`` and reset with ``derive_seed(episode_seed, g_index)``
       so the drift schedule is fixed per-rollout but variance comes from
       policy sampling (DESIGN.md §6.2, §7.4).
    2. Walk the multi-turn loop: render observation -> ``model.generate`` ->
       parse :class:`DriftCallAction` JSON -> ``env.step`` until ``done``.
    3. Return the tuple ``(episodes, completions)``: terminal :class:`Episode`
       per rollout and the concatenated assistant text per rollout.

    The function is intentionally minimal — model invocation happens through
    the standard transformers ``model.generate`` interface so it works with
    any HF-compatible policy. Heavy imports (``torch``) are deferred.
    """
    from cells.step_10_env import DriftCallEnvError

    episodes: list[Any] = []
    completions: list[str] = []

    for g in range(num_generations):
        env = env_factory()
        seed = _derive_rollout_seed(episode_seed, g)
        obs = env.reset(seed=seed)
        prompt = render_initial_prompt(tokenizer, goal)
        completion_chunks: list[str] = []

        while True:
            generated_text = _generate_one_turn(model, tokenizer, prompt)
            completion_chunks.append(generated_text)
            action = _parse_action_from_completion(generated_text)
            try:
                obs = env.step(action)
            except DriftCallEnvError:
                break
            if obs is None or getattr(obs, "_done", False) or getattr(obs, "done", False):
                break
            # Append assistant turn + the next observation's user-facing fields
            # to the running prompt so subsequent generations see history.
            prompt = (
                f"{prompt}\n{generated_text}\n"
                f"[turn={obs.turn}] {obs.last_transcript}"
            )

        episode = getattr(env, "_episode", None)
        if episode is None:
            episode = env.state()
        episodes.append(episode)
        completions.append("\n".join(completion_chunks))
        env.close()

    return tuple(episodes), tuple(completions)


def _generate_one_turn(model: Any, tokenizer: Any, prompt: str) -> str:
    """Tokenize ``prompt``, invoke ``model.generate``, decode the new tokens.

    Heavy imports (``torch``) deferred so this module loads on CPU-only CI.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return cast("str", tokenizer.decode(new_tokens, skip_special_tokens=True))


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


# ---------------------------------------------------------------------------
# Adaptive KL controller (training.md §3.3 — retarget β from measured KL)
# ---------------------------------------------------------------------------


DEFAULT_BETA_MIN: float = 0.001
DEFAULT_BETA_MAX: float = 1.0
DEFAULT_KP: float = 2.0


class AdaptiveKLCallback:
    """Retarget β each step based on the ratio of measured KL to ``target_kl``.

    Proportional controller with symmetric log-space update:

        err     = (kl - target_kl) / target_kl
        new_beta = beta * exp(kp * err)
        new_beta = clamp(new_beta, beta_min, beta_max)

    When ``kl`` matches ``target_kl``, ``err == 0`` and β is left unchanged.
    Safe on missing / NaN / non-numeric KL signals (no-op, no exception).

    Inherits from :class:`transformers.trainer_callback.TrainerCallback` when
    available (production path); falls back to ``object`` on CPU-only CI so
    the class can be exercised without installing transformers.
    """

    def __init__(
        self,
        target_kl: float = BETA_KL,
        *,
        kp: float = DEFAULT_KP,
        beta_min: float = DEFAULT_BETA_MIN,
        beta_max: float = DEFAULT_BETA_MAX,
    ) -> None:
        if target_kl <= 0.0:
            raise ValueError(f"target_kl must be > 0; got {target_kl}")
        if beta_min <= 0.0 or beta_max <= 0.0:
            raise ValueError(
                f"beta bounds must be > 0; got min={beta_min}, max={beta_max}"
            )
        if beta_min > beta_max:
            raise ValueError(
                f"beta_min ({beta_min}) must be <= beta_max ({beta_max})"
            )
        self.target_kl = float(target_kl)
        self.kp = float(kp)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def _coerce_kl(self, raw: Any) -> float | None:
        """Return a finite float or ``None`` — propagates no-op on bad input."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def _next_beta(self, beta: float, kl: float) -> tuple[float, bool, bool]:
        """Return ``(new_beta, clamped_to_min, clamped_to_max)``."""
        err = (kl - self.target_kl) / self.target_kl
        # Clamp the exponent so extreme KL spikes don't overflow math.exp;
        # the result is clamped anyway and exp(±50) easily saturates either bound.
        exponent = max(-50.0, min(50.0, self.kp * err))
        scaled = beta * math.exp(exponent)
        if scaled <= self.beta_min:
            return self.beta_min, True, False
        if scaled >= self.beta_max:
            return self.beta_max, False, True
        return scaled, False, False

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        *,
        logs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        """TRL hook — called with every ``trainer.log(...)`` dict.

        On a well-formed KL signal: mutates ``args.beta`` with the new
        coefficient and writes five diagnostic fields back into ``logs``
        so TRL's default reporter forwards them to wandb / CSV / etc.:

        - ``train/beta_adaptive``       current KL coefficient
        - ``train/kl_measured``         sanitised KL input
        - ``train/kl_target``           constant — aids chart-by-reference
        - ``train/beta_clamped_to_min`` 0/1 — fires on collapse
        - ``train/beta_clamped_to_max`` 0/1 — fires on runaway divergence
        """
        if logs is None:
            return control
        if "kl" not in logs:
            return control
        kl = self._coerce_kl(logs["kl"])
        if kl is None:
            return control
        beta = float(getattr(args, "beta", BETA_KL))
        new_beta, clamped_lo, clamped_hi = self._next_beta(beta, kl)
        args.beta = new_beta
        logs["train/beta_adaptive"] = new_beta
        logs["train/kl_measured"] = kl
        logs["train/kl_target"] = self.target_kl
        logs["train/beta_clamped_to_min"] = 1 if clamped_lo else 0
        logs["train/beta_clamped_to_max"] = 1 if clamped_hi else 0
        return control


__all__ = [
    "AdapterRecord",
    "AdaptiveKLCallback",
    "DEFAULT_BETA_MAX",
    "DEFAULT_BETA_MIN",
    "DEFAULT_KP",
    "EnvFactory",
    "EpisodeDatasetAdapter",
    "EpisodeSampler",
    "LanguageCode",
    "PINNED_SYSTEM_PROMPT",
    "RolloutGroupFn",
    "driftcall_grpo_trainer_methods",
    "make_driftcall_grpo_trainer_cls",
    "render_initial_prompt",
    "rollout_group",
]
