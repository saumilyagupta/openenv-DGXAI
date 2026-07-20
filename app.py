"""DriftCall env Space — FastAPI + OpenEnv-compliant REST surface.

Implements ``docs/modules/deploy_env_space.md`` and DESIGN.md §3.3 / §11.1.

Endpoints:
    GET  /healthz   → 200 text/plain "ok" (unauthenticated)
    POST /reset     → 200 application/json (create / recycle session)
    POST /step      → 200 application/json (advance one turn)
    GET  /state     → 200 application/json (read DriftCallState)
    POST /close     → 200 application/json (evict session)

Headers (mutating endpoints): ``Authorization: Bearer <DRIFTCALL_ENV_TOKEN>``
and ``X-Session-Id: <[A-Za-z0-9_-]{1,64}>``.

Error modes (deploy_env_space.md §5):
    M1 401 unauthorized          M7  400 bad_json
    M2 400 missing_session_id    M8  400 invalid_action
    M3 404 session_not_found     M9  500 internal_error
    M4 404 session_expired       M10 500 io_error
    M5 429 max_sessions          M11 413 payload_too_large
    M6 503 model_not_ready       M12 409 reset_in_progress

All error bodies: ``{"error": {"code": <slug>, "message": <str>,
"request_id": <asgi-id>}}``; ``Cache-Control: no-store``; only M5 carries
``Retry-After: 30``. No stack traces ever leak across the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from cells.step_04_models import ActionType, DriftCallAction
from cells.step_10_env import (
    DriftCallEnv,
    EnvClosedError,
    EnvNotReadyError,
    EpisodeAlreadyTerminalError,
    InvalidActionError,
    InvalidConfigError,
    UnknownDomainError,
    UnknownToolError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SESSIONS: int = 10
_TTL_S: float = 3600.0
_SWEEP_INTERVAL_S: float = 60.0
_MAX_SESSION_ID_LEN: int = 64
_SESSION_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BODY_BYTES: int = 1 * 1024 * 1024  # 1 MiB
_RETRY_AFTER_S: str = "30"
_TOKEN_ENV_VAR: str = "DRIFTCALL_ENV_TOKEN"


# ---------------------------------------------------------------------------
# Time source (test-overridable)
# ---------------------------------------------------------------------------


def _monotonic() -> float:
    """Indirection for tests to monkeypatch."""

    return time.monotonic()


# ---------------------------------------------------------------------------
# Errors / envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ApiError(Exception):
    """Internal exception → uniform error envelope (deploy_env_space.md §5)."""

    code: str
    message: str
    http_status: int
    retry_after: bool = False


_NO_STORE: dict[str, str] = {"Cache-Control": "no-store"}


def _error_response(err: _ApiError, request_id: str) -> JSONResponse:
    body = {
        "error": {
            "code": err.code,
            "message": err.message,
            "request_id": request_id,
        }
    }
    headers = dict(_NO_STORE)
    if err.retry_after:
        headers["Retry-After"] = _RETRY_AFTER_S
    return JSONResponse(status_code=err.http_status, content=body, headers=headers)


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionEntry:
    """Frozen per project rule — every touch produces a new entry."""

    env: DriftCallEnv
    created_at: float
    last_touched: float
    reset_count: int
    lock: asyncio.Lock


class SessionCache:
    """In-memory session registry with LRU + TTL eviction."""

    def __init__(self, *, max_sessions: int = _MAX_SESSIONS, ttl_s: float = _TTL_S) -> None:
        self._max = max_sessions
        self._ttl = ttl_s
        self._store: dict[str, SessionEntry] = {}
        self._guard = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._store)

    def get(self, sid: str) -> SessionEntry | None:
        return self._store.get(sid)

    async def acquire_lock(self, sid: str) -> asyncio.Lock:
        """Return (or lazily create) the per-session lock."""
        async with self._guard:
            entry = self._store.get(sid)
            if entry is not None:
                return entry.lock
            return asyncio.Lock()

    async def insert_or_replace(self, sid: str, env_factory: Callable[[], DriftCallEnv]) -> SessionEntry:
        """Insert a new env or replace an existing one (in-place reset)."""
        async with self._guard:
            now = _monotonic()
            existing = self._store.get(sid)
            if existing is not None:
                # In-place reset (§7.1 case after winner completed).
                try:
                    existing.env.close()
                except Exception:
                    logger.exception("env.close() raised on in-place reset for sid=%s", sid)
                env = env_factory()
                entry = SessionEntry(
                    env=env,
                    created_at=now,
                    last_touched=now,
                    reset_count=existing.reset_count + 1,
                    lock=existing.lock,
                )
                self._store[sid] = entry
                return entry
            # New session — enforce cap.
            if len(self._store) >= self._max:
                # Try LRU evict only if any entry is older than the others by TTL/2.
                victim_sid = min(self._store, key=lambda k: self._store[k].last_touched)
                victim = self._store[victim_sid]
                age = now - victim.last_touched
                if age <= 0.0:
                    raise _ApiError(
                        code="max_sessions",
                        message=f"max concurrent sessions reached ({self._max})",
                        http_status=429,
                        retry_after=True,
                    )
                try:
                    victim.env.close()
                except Exception:
                    logger.exception("env.close() raised on LRU eviction for sid=%s", victim_sid)
                self._store.pop(victim_sid, None)
            env = env_factory()
            entry = SessionEntry(
                env=env,
                created_at=now,
                last_touched=now,
                reset_count=0,
                lock=asyncio.Lock(),
            )
            self._store[sid] = entry
            return entry

    def touch(self, sid: str) -> tuple[SessionEntry | None, bool]:
        """Update last_touched. Returns ``(entry, was_expired)``.

        - ``(entry, False)`` on hit
        - ``(None, True)`` if the entry was present but evicted by this call
          due to TTL expiry
        - ``(None, False)`` if there was never an entry under this sid
        """
        entry = self._store.get(sid)
        if entry is None:
            return None, False
        now = _monotonic()
        if now - entry.last_touched > self._ttl:
            try:
                entry.env.close()
            except Exception:
                logger.exception("env.close() raised on expired touch for sid=%s", sid)
            self._store.pop(sid, None)
            return None, True
        new = replace(entry, last_touched=now)
        self._store[sid] = new
        return new, False

    def evict(self, sid: str) -> SessionEntry | None:
        """Pop a session out of the cache. Returns the removed entry or None."""
        return self._store.pop(sid, None)

    def sweep(self) -> int:
        """Synchronous TTL sweep — evict every entry past TTL."""
        now = _monotonic()
        expired = [sid for sid, e in self._store.items() if now - e.last_touched > self._ttl]
        for sid in expired:
            entry = self._store.pop(sid)
            try:
                entry.env.close()
            except Exception:
                logger.exception("env.close() raised on sweep for sid=%s", sid)
        if expired:
            logger.info(
                json.dumps(
                    {
                        "event": "session_sweep",
                        "expired_count": len(expired),
                        "cache_size": len(self._store),
                    }
                )
            )
        return len(expired)


# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------


@dataclass
class _AppState:
    """Mutable (intentional) — owned by lifespan; readers go through getters."""

    cache: SessionCache
    models_ready: bool = False
    sweep_task: asyncio.Task[None] | None = None
    bearer_token: str = ""


def _get_state(app: FastAPI) -> _AppState:
    state: _AppState = app.state.driftcall
    return state


# ---------------------------------------------------------------------------
# Lifespan — eager-load Kokoro + Whisper before serving (M6 guard)
# ---------------------------------------------------------------------------


def _eager_load_models() -> None:
    """Force-load TTS + ASR singletons. Test patches this to avoid network."""
    from cells.step_09_audio import get_asr_engine, get_tts_engine

    get_tts_engine()
    get_asr_engine()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cache = SessionCache()
    token = os.environ.get(_TOKEN_ENV_VAR, "")
    if not token:
        # Fail-fast per deploy_env_space.md §3.5.
        raise RuntimeError(
            f"{_TOKEN_ENV_VAR} environment variable not set; refusing to start"
        )
    state = _AppState(cache=cache, bearer_token=token)
    app.state.driftcall = state

    # Eager model load (M6 guard — must complete before serving).
    try:
        await asyncio.to_thread(_eager_load_models)
    except Exception:
        logger.exception("eager model load failed")
        raise
    state.models_ready = True

    # Background TTL sweep.
    async def _sweep_loop() -> None:
        try:
            while True:
                await asyncio.sleep(_SWEEP_INTERVAL_S)
                cache.sweep()
        except asyncio.CancelledError:
            raise

    state.sweep_task = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        if state.sweep_task is not None:
            state.sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.sweep_task


# ---------------------------------------------------------------------------
# Body-size middleware (M11)
# ---------------------------------------------------------------------------


class _BodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, max_bytes: int = _MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                cl_int = int(cl)
            except ValueError:
                cl_int = -1
            if cl_int > self._max_bytes:
                err = _ApiError(
                    code="payload_too_large",
                    message="request body exceeds 1 MiB",
                    http_status=413,
                )
                return _error_response(err, _request_id(request))
        return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers — auth, headers, body parsing
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    return str(id(request))


def _check_bearer(request: Request, state: _AppState) -> None:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise _ApiError(
            code="unauthorized",
            message="missing or non-Bearer Authorization header",
            http_status=401,
        )
    token = auth[len("Bearer ") :].strip()
    if token != state.bearer_token or not token:
        raise _ApiError(
            code="unauthorized",
            message="invalid bearer token",
            http_status=401,
        )


def _check_session_header(request: Request) -> str:
    sid = request.headers.get("x-session-id", "")
    if not sid or not _SESSION_ID_RE.match(sid):
        raise _ApiError(
            code="missing_session_id",
            message="X-Session-Id header missing or malformed",
            http_status=400,
        )
    return sid


def _check_models_ready(state: _AppState) -> None:
    if not state.models_ready:
        raise _ApiError(
            code="model_not_ready",
            message="audio models still loading; retry shortly",
            http_status=503,
        )


async def _parse_json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        raise _ApiError(
            code="payload_too_large",
            message="request body exceeds 1 MiB",
            http_status=413,
        )
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _ApiError(
            code="bad_json",
            message=f"malformed JSON: {exc.__class__.__name__}",
            http_status=400,
        ) from exc
    if not isinstance(parsed, dict):
        raise _ApiError(
            code="bad_json",
            message="request body must be a JSON object",
            http_status=400,
        )
    return parsed


# ---------------------------------------------------------------------------
# Action / config validation (envelope-level — env owns deep validation)
# ---------------------------------------------------------------------------


def _build_action(raw: Any) -> DriftCallAction:
    if not isinstance(raw, dict):
        raise _ApiError(
            code="invalid_action",
            message="action must be a JSON object",
            http_status=400,
        )
    atype_raw = raw.get("action_type")
    if not isinstance(atype_raw, str):
        raise _ApiError(
            code="invalid_action",
            message="action.action_type must be a string",
            http_status=400,
        )
    try:
        atype = ActionType(atype_raw)
    except ValueError as exc:
        raise _ApiError(
            code="invalid_action",
            message=f"unknown action_type {atype_raw!r}",
            http_status=400,
        ) from exc

    tool_name = raw.get("tool_name")
    tool_args = raw.get("tool_args")
    message = raw.get("message")
    confidence = raw.get("confidence")
    rationale = raw.get("rationale")

    # Action-type contract checks (deep checks happen inside env._validate_action).
    if atype == ActionType.TOOL_CALL and (
        tool_name is None or not isinstance(tool_name, str) or tool_args is None
    ):
        raise _ApiError(
            code="invalid_action",
            message="TOOL_CALL requires tool_name (str) and tool_args (object)",
            http_status=400,
        )
    return DriftCallAction(
        action_type=atype,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_args=tool_args if isinstance(tool_args, dict) else None,
        message=message if isinstance(message, str) else None,
        confidence=float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        rationale=rationale if isinstance(rationale, str) else None,
    )


def _build_env_config(reset_body: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = reset_body.get("config")
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise _ApiError(
            code="invalid_action",
            message="config must be a JSON object",
            http_status=400,
        )
    return raw_cfg


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert frozen dataclasses / tuples / enums to JSON-safe form."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, ActionType):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Endpoint handlers (one function per route)
# ---------------------------------------------------------------------------


async def _handle_reset(request: Request, state: _AppState) -> Response:
    _check_bearer(request, state)
    _check_models_ready(state)
    sid = _check_session_header(request)
    body = await _parse_json_body(request)
    cfg = _build_env_config(body)
    seed_raw = body.get("seed")
    if seed_raw is not None and (not isinstance(seed_raw, int) or isinstance(seed_raw, bool)):
        raise _ApiError(
            code="invalid_action",
            message="seed must be an int or null",
            http_status=400,
        )
    seed: int | None = seed_raw if isinstance(seed_raw, int) and not isinstance(seed_raw, bool) else None

    cache = state.cache
    # Per-session reset lock (§7.1).
    existing = cache.get(sid)
    if existing is not None and existing.lock.locked():
        raise _ApiError(
            code="reset_in_progress",
            message="concurrent /reset on same session id",
            http_status=409,
        )

    # Acquire lock (creates one if not present).
    lock = await cache.acquire_lock(sid)
    if lock.locked():
        raise _ApiError(
            code="reset_in_progress",
            message="concurrent /reset on same session id",
            http_status=409,
        )

    async with lock:
        def _factory() -> DriftCallEnv:
            try:
                return DriftCallEnv(cfg)
            except InvalidConfigError as exc:
                raise _ApiError(
                    code="invalid_action",
                    message=f"invalid config: {exc}",
                    http_status=400,
                ) from exc

        try:
            entry = await cache.insert_or_replace(sid, _factory)
        except _ApiError:
            raise
        except Exception as exc:
            logger.exception("env construction failed for sid=%s", sid)
            raise _ApiError(
                code="internal_error",
                message="env construction failed",
                http_status=500,
            ) from exc

        try:
            obs = await asyncio.to_thread(entry.env.reset, seed)
        except InvalidConfigError as exc:
            cache.evict(sid)
            raise _ApiError(
                code="invalid_action",
                message=f"invalid config at reset: {exc}",
                http_status=400,
            ) from exc
        except OSError as exc:
            cache.evict(sid)
            raise _ApiError(
                code="io_error",
                message=f"I/O error during reset: {exc.__class__.__name__}",
                http_status=500,
            ) from exc
        except Exception as exc:
            cache.evict(sid)
            logger.exception("env.reset raised for sid=%s", sid)
            raise _ApiError(
                code="internal_error",
                message="env.reset raised",
                http_status=500,
            ) from exc

    body_out = {
        "observation": _to_jsonable(obs),
        "episode_id": entry.env.state().episode_id,
        "max_turns": entry.env.state().max_turns,
    }
    return JSONResponse(status_code=200, content=body_out)


async def _handle_step(request: Request, state: _AppState) -> Response:
    _check_bearer(request, state)
    _check_models_ready(state)
    sid = _check_session_header(request)
    body = await _parse_json_body(request)
    raw_action = body.get("action")
    action = _build_action(raw_action)

    entry, was_expired = state.cache.touch(sid)
    if entry is None:
        if was_expired:
            raise _ApiError(
                code="session_expired",
                message="session TTL expired; call /reset",
                http_status=404,
            )
        raise _ApiError(
            code="session_not_found",
            message="X-Session-Id has no live session; call /reset",
            http_status=404,
        )

    try:
        obs = await asyncio.to_thread(entry.env.step, action)
    except (InvalidActionError, UnknownToolError, UnknownDomainError) as exc:
        raise _ApiError(
            code="invalid_action",
            message=str(exc),
            http_status=400,
        ) from exc
    except (EnvNotReadyError, EnvClosedError, EpisodeAlreadyTerminalError) as exc:
        raise _ApiError(
            code="invalid_action",
            message=str(exc),
            http_status=400,
        ) from exc
    except OSError as exc:
        raise _ApiError(
            code="io_error",
            message=f"I/O error during step: {exc.__class__.__name__}",
            http_status=500,
        ) from exc
    except Exception as exc:
        logger.exception("env.step raised for sid=%s", sid)
        raise _ApiError(
            code="internal_error",
            message="env.step raised",
            http_status=500,
        ) from exc

    reward: float | None = None
    info: dict[str, Any] = {}
    if entry.env.done():
        try:
            rewards = entry.env.rewards()
            reward = float(getattr(rewards, "reward", 0.0))
            info["terminated_by"] = entry.env.episode().terminated_by
        except Exception:
            reward = None

    body_out = {
        "observation": _to_jsonable(obs),
        "reward": reward,
        "done": bool(entry.env.done()),
        "info": info,
    }
    return JSONResponse(status_code=200, content=body_out)


async def _handle_state(request: Request, state: _AppState) -> Response:
    _check_bearer(request, state)
    _check_models_ready(state)
    sid = _check_session_header(request)
    entry, was_expired = state.cache.touch(sid)
    if entry is None:
        if was_expired:
            raise _ApiError(
                code="session_expired",
                message="session TTL expired; call /reset",
                http_status=404,
            )
        raise _ApiError(
            code="session_not_found",
            message="X-Session-Id has no live session; call /reset",
            http_status=404,
        )
    try:
        st = entry.env.state()
    except EnvNotReadyError as exc:
        raise _ApiError(
            code="invalid_action",
            message=str(exc),
            http_status=400,
        ) from exc
    body_out = {"state": _to_jsonable(st), "turn": st.turn}
    return JSONResponse(status_code=200, content=body_out)


async def _handle_close(request: Request, state: _AppState) -> Response:
    _check_bearer(request, state)
    _check_models_ready(state)
    sid = _check_session_header(request)
    entry = state.cache.evict(sid)
    if entry is None:
        return JSONResponse(status_code=200, content={"closed": True, "final_state": None})
    final_state: Any = None
    try:
        final_state = _to_jsonable(entry.env.state())
    except EnvNotReadyError:
        final_state = None
    try:
        entry.env.close()
    except Exception:
        logger.exception("env.close raised on /close for sid=%s", sid)
    return JSONResponse(status_code=200, content={"closed": True, "final_state": final_state})


# ---------------------------------------------------------------------------
# App factory + route wiring
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct a fresh FastAPI app. Used by tests to get an isolated instance."""
    app = FastAPI(lifespan=lifespan, title="DriftCall Env", version="0.1.0")
    app.add_middleware(_BodySizeMiddleware, max_bytes=_MAX_BODY_BYTES)

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse(content="ok", status_code=200)

    @app.post("/reset")
    async def reset_route(request: Request) -> Response:
        try:
            return await _handle_reset(request, _get_state(app))
        except _ApiError as err:
            return _error_response(err, _request_id(request))

    @app.post("/step")
    async def step_route(request: Request) -> Response:
        try:
            return await _handle_step(request, _get_state(app))
        except _ApiError as err:
            return _error_response(err, _request_id(request))

    @app.get("/state")
    async def state_route(request: Request) -> Response:
        try:
            return await _handle_state(request, _get_state(app))
        except _ApiError as err:
            return _error_response(err, _request_id(request))

    @app.post("/close")
    async def close_route(request: Request) -> Response:
        try:
            return await _handle_close(request, _get_state(app))
        except _ApiError as err:
            return _error_response(err, _request_id(request))

    return app


app = create_app()


__all__ = [
    "SessionCache",
    "SessionEntry",
    "app",
    "create_app",
    "lifespan",
]
