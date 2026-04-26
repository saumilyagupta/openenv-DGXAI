"""Gemma-3n + DriftCall LoRA policy for OpenEnv inference.

Loads ``unsloth/gemma-3n-E2B-it`` plus the trained adapter
(``DGXAI/gemma-3n-e2b-driftcall-lora`` by default) and exposes an
:meth:`act(observation) -> action` method that the gym runner calls.

Heavy deps (``torch``, ``transformers``, ``peft``, ``unsloth``) are imported
lazily so this module can be imported on machines without a GPU (e.g., CI
runners that only need to type-check the gym client).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID: str = "unsloth/gemma-3n-E2B-it"
DEFAULT_ADAPTER_ID: str = "DGXAI/gemma-3n-e2b-driftcall-lora"
DEFAULT_MAX_NEW_TOKENS: int = 256
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9


@dataclass
class PolicyConfig:
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    adapter_id: str | None = DEFAULT_ADAPTER_ID
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))


class GemmaPolicy:
    """LoRA-augmented Gemma-3n policy.

    Lazy-loads the model on first :meth:`act` call so construction is cheap.
    Pass ``adapter_id=None`` to evaluate the **untrained baseline**.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """Materialize the model + tokenizer on the local GPU."""
        if self._model is not None:
            return
        # Lazy imports — keep top-level import cheap.
        import torch  # noqa: F401
        from unsloth import FastModel  # type: ignore[import-untyped]

        cfg = self.config
        logger.info("loading base=%s adapter=%s", cfg.base_model_id, cfg.adapter_id)
        model, tokenizer = FastModel.from_pretrained(
            cfg.base_model_id,
            max_seq_length=4096,
            load_in_4bit=False,        # 16-bit LoRA path; matches training.
            full_finetuning=False,
            token=cfg.hf_token,
        )
        if cfg.adapter_id:
            from peft import PeftModel  # type: ignore[import-untyped]
            model = PeftModel.from_pretrained(model, cfg.adapter_id, token=cfg.hf_token)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Generate one ``DriftCallAction`` from an observation.

        Strategy: render the observation as a chat-template prompt, generate
        up to ``max_new_tokens``, then parse the first JSON object out of the
        response. If parsing fails, fall back to ``end_episode`` so the env
        always advances.
        """
        if self._model is None:
            self.load()
        assert self._model is not None and self._tokenizer is not None

        prompt = self._render_prompt(observation)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        import torch
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        completion = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return self._parse_action(completion)

    # ── Prompt + parsing helpers ──────────────────────────────────────

    @staticmethod
    def _render_prompt(observation: dict[str, Any]) -> str:
        """Compact prompt: brief + optional history + JSON-action instruction."""
        brief = observation.get("brief") or observation.get("instruction") or ""
        history = observation.get("history") or observation.get("turns") or []
        history_str = ""
        if history:
            lines = []
            for t in history[-6:]:  # last 6 turns
                actor = t.get("actor", "?")
                content = t.get("content") or t.get("text") or t.get("response") or ""
                lines.append(f"[{actor}] {content}")
            history_str = "\n".join(lines)
        instruction = (
            "Reply with EXACTLY one JSON object matching the DriftCallAction schema. "
            'Examples: {"type":"book","vendor":"airline","payload":{...}} or '
            '{"type":"end_episode"}.'
        )
        return f"BRIEF:\n{brief}\n\nHISTORY:\n{history_str}\n\n{instruction}\nACTION:"

    @staticmethod
    def _parse_action(completion: str) -> dict[str, Any]:
        """Best-effort extraction of the first JSON object in the completion."""
        match = re.search(r"\{.*\}", completion, re.DOTALL)
        if not match:
            return {"type": "end_episode"}
        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"type": "end_episode"}
        if not isinstance(action, dict) or "type" not in action:
            return {"type": "end_episode"}
        return action
