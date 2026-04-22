from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OllamaResponse:
    text: str
    eval_count: int
    total_duration_ns: int
    done_reason: str
    raw: dict[str, Any]


class OllamaClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

    async def generate(self, client: httpx.AsyncClient, prompt: str) -> OllamaResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = await client.post(
                    self.endpoint,
                    json=self._payload(prompt),
                    timeout=self.timeout_seconds,
                )
                r.raise_for_status()
                data = r.json()
                return OllamaResponse(
                    text=data.get("response", ""),
                    eval_count=int(data.get("eval_count", 0)),
                    total_duration_ns=int(data.get("total_duration", 0)),
                    done_reason=str(data.get("done_reason", "")),
                    raw=data,
                )
            except (httpx.HTTPError, ValueError) as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError(f"Ollama generate failed after {self.max_retries} retries: {last_exc}")
