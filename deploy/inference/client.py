"""OpenEnv gym client for the DriftCall env Space.

Talks to the deployed FastAPI / OpenEnv Space (``app.py`` at the repo root)
over the documented REST surface:

    POST /reset           POST /step
    GET  /state           POST /close
    GET  /healthz         (unauthenticated)

Auth: ``Authorization: Bearer <DRIFTCALL_ENV_TOKEN>`` plus ``X-Session-Id``
on every mutating request. Errors come back as the documented envelope::

    {"error": {"code": "<slug>", "message": "<str>", "request_id": "<id>"}}

The client is **synchronous** (uses ``requests``) and intentionally minimal:
it is not a full gymnasium ``Env`` subclass; it exposes the four verbs that
match :mod:`gymnasium`'s API plus :meth:`state` for read-only inspection.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default endpoint — overridden by ``DRIFTCALL_ENV_URL`` or the constructor.
DEFAULT_ENV_URL: str = "https://dgxai-driftcall-env.hf.space"

# Conservative HTTP timeouts; the env Space target is < 8s warm.
_CONNECT_TIMEOUT_S: float = 5.0
_READ_TIMEOUT_S: float = 30.0
_HTTP_TIMEOUT: tuple[float, float] = (_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S)

# Bounded retry on M5 (max_sessions) and 5xx; everything else fails fast.
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_S: float = 1.5


class GymClientError(RuntimeError):
    """Base class for client-side failures (HTTP, auth, schema)."""


class GymAuthError(GymClientError):
    """M1 unauthorized — token missing or invalid."""


class GymSessionError(GymClientError):
    """M2/M3/M4/M12 — session-id missing, not found, expired, reset_in_progress."""


class GymCapacityError(GymClientError):
    """M5 max_sessions — too many concurrent sessions on the Space."""


@dataclass
class StepResult:
    """Tuple-style step return preserving the OpenEnv response payload."""

    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Gymnasium-compatible tuple: ``(obs, reward, terminated, truncated, info)``."""
        return (self.observation, self.reward, self.terminated, self.truncated, self.info)


class DriftCallGymClient:
    """Thin OpenEnv REST client matching the gymnasium API verbs.

    Parameters
    ----------
    env_url:
        Base URL of the deployed Space (no trailing slash). Defaults to the
        ``DRIFTCALL_ENV_URL`` env var, then ``DEFAULT_ENV_URL``.
    auth_token:
        Bearer token. Defaults to ``DRIFTCALL_ENV_TOKEN``.
    session_id:
        Stable per-episode id. Defaults to a fresh ``secrets.token_urlsafe(24)``.
    timeout:
        ``(connect, read)`` HTTP timeouts in seconds.
    """

    def __init__(
        self,
        env_url: str | None = None,
        auth_token: str | None = None,
        session_id: str | None = None,
        timeout: tuple[float, float] = _HTTP_TIMEOUT,
    ) -> None:
        self._url = (env_url or os.environ.get("DRIFTCALL_ENV_URL") or DEFAULT_ENV_URL).rstrip("/")
        self._token = auth_token or os.environ.get("DRIFTCALL_ENV_TOKEN")
        self._session_id = session_id or self._fresh_session_id()
        self._timeout = timeout
        self._http = requests.Session()
        self._closed = False

    @staticmethod
    def _fresh_session_id() -> str:
        # Constraint from app.py: ``[A-Za-z0-9_-]{1,64}``.
        return secrets.token_urlsafe(24)

    @property
    def session_id(self) -> str:
        """Stable per-instance id sent as ``X-Session-Id`` on every call."""
        return self._session_id

    def _auth_headers(self, request_id: str | None = None) -> dict[str, str]:
        if not self._token:
            raise GymAuthError(
                "DRIFTCALL_ENV_TOKEN not set — pass auth_token=... or export it."
            )
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Session-Id": self._session_id,
            "X-Request-Id": request_id or uuid.uuid4().hex,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return headers

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._url}{path}"
        headers = self._auth_headers()
        body = payload or {}
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.post(url, json=body, headers=headers, timeout=self._timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp.json()
            self._raise_for_response(resp)
            return {}  # unreachable — _raise_for_response always raises
        raise GymClientError(f"POST {path} failed after {_MAX_RETRIES} retries: {last_exc!r}")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._url}{path}"
        # /healthz is unauthenticated; everything else uses bearer + session.
        headers = {} if path == "/healthz" else self._auth_headers()
        try:
            resp = self._http.get(url, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise GymClientError(f"GET {path} network error: {exc!r}") from exc
        if resp.status_code == 200:
            return resp.json() if path != "/healthz" else {"status": resp.text.strip()}
        self._raise_for_response(resp)
        return {}

    @staticmethod
    def _raise_for_response(resp: requests.Response) -> None:
        try:
            payload = resp.json()
        except ValueError:
            payload = {"error": {"code": "non_json_response", "message": resp.text[:200]}}
        err = payload.get("error", payload)
        code = err.get("code", "unknown")
        msg = err.get("message", "")
        rid = err.get("request_id", "")
        full = f"{resp.status_code} {code}: {msg} (request_id={rid})"
        if resp.status_code == 401:
            raise GymAuthError(full)
        if resp.status_code in (400, 404, 409) and code in {
            "missing_session_id",
            "session_not_found",
            "session_expired",
            "reset_in_progress",
        }:
            raise GymSessionError(full)
        if resp.status_code == 429 or code == "max_sessions":
            raise GymCapacityError(full)
        raise GymClientError(full)

    # ── Gymnasium-compatible verbs ────────────────────────────────────

    def healthz(self) -> str:
        """Probe ``/healthz``; returns ``"ok"`` when the Space is up."""
        return self._get("/healthz").get("status", "")

    def reset(
        self,
        *,
        seed: int | None = None,
        curriculum_stage: int | None = None,
        language_weights: dict[str, float] | None = None,
        audio_boundary_enabled: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the env, returning ``(observation, info)``.

        All four reset params are optional (DESIGN.md §3.3 / openenv.yaml).
        """
        if self._closed:
            raise GymClientError("Client is closed; create a new instance.")
        body: dict[str, Any] = {}
        if seed is not None:
            body["seed"] = int(seed)
        if curriculum_stage is not None:
            if curriculum_stage not in (1, 2, 3):
                raise ValueError("curriculum_stage must be 1, 2, or 3")
            body["curriculum_stage"] = int(curriculum_stage)
        if language_weights is not None:
            body["language_weights"] = dict(language_weights)
        if audio_boundary_enabled is not None:
            body["audio_boundary_enabled"] = bool(audio_boundary_enabled)
        payload = self._post("/reset", body)
        observation = payload.get("observation", {})
        info = payload.get("info", {})
        return observation, info

    def step(self, action: dict[str, Any]) -> StepResult:
        """Advance one turn. ``action`` shape matches ``DriftCallAction``.

        Returns a :class:`StepResult` whose ``.as_tuple()`` is the standard
        gymnasium ``(obs, reward, terminated, truncated, info)`` 5-tuple.
        """
        if self._closed:
            raise GymClientError("Client is closed; create a new instance.")
        if not isinstance(action, dict):
            raise TypeError("action must be a dict matching DriftCallAction schema")
        payload = self._post("/step", {"action": action})
        return StepResult(
            observation=payload.get("observation", {}),
            reward=float(payload.get("reward", 0.0)),
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            info=payload.get("info", {}),
        )

    def state(self) -> dict[str, Any]:
        """Read-only ``DriftCallState`` snapshot (no side effects)."""
        if self._closed:
            raise GymClientError("Client is closed; create a new instance.")
        return self._get("/state")

    def close(self) -> None:
        """Evict the server-side session. Safe to call multiple times."""
        if self._closed:
            return
        try:
            self._post("/close", {})
        except GymSessionError:
            # Session already gone — treat as success.
            logger.debug("close() — session already evicted server-side")
        finally:
            self._closed = True
            self._http.close()

    # ── Context manager support ───────────────────────────────────────

    def __enter__(self) -> DriftCallGymClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
