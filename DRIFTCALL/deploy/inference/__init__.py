"""DriftCall OpenEnv gym client + Gemma-3n+LoRA inference policy.

This package provides an OpenEnv-compliant client that connects to the
deployed env Space (``DGXAI/driftcall-env``) and a policy wrapper that
loads the trained LoRA adapter (``DGXAI/gemma-3n-e2b-driftcall-lora``)
on top of the base model (``unsloth/gemma-3n-E2B-it``).

Public entrypoints
------------------

- :class:`DriftCallGymClient` — thin gym-style wrapper around the REST env
  Space. Exposes ``reset(seed=, curriculum_stage=, ...)``, ``step(action)``,
  ``state()``, ``close()``.
- :class:`GemmaPolicy` — turn-level policy loaded from base + LoRA.
- ``run`` — CLI entry. Usage::

    python -m deploy.inference.run \\
        --env-url https://dgxai-driftcall-env.hf.space \\
        --adapter-id DGXAI/gemma-3n-e2b-driftcall-lora \\
        --seed 42 --curriculum-stage 2 --num-episodes 5
"""

from __future__ import annotations

from deploy.inference.client import DriftCallGymClient
from deploy.inference.policy import GemmaPolicy

__all__ = ["DriftCallGymClient", "GemmaPolicy"]
