"""Tests for ``app.py`` — DriftCall env Space FastAPI surface.

Implements ``docs/tests/deploy_env_space_tests.md`` §1 / §2 / §5.
Heavy deps are stubbed (no real Kokoro/Whisper/HF), and ``DriftCallEnv`` is
exercised in-process via the cells modules.

All tests use ``fastapi.testclient.TestClient`` — no real socket, no real
audio, no real network.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import secrets
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Rewards stub — installed before importing app.py, so DriftCallEnv submit
# paths don't pull in cells.step_08_rewards' real maths during tests.
# ---------------------------------------------------------------------------


def _install_rewards_stub() -> types.ModuleType:
    name = "cells.step_08_rewards"
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)

    @dataclasses.dataclass(frozen=True)
    class Rewards:
        r1: float = 1.0
        r2: float = 1.0
        r3: float = 1.0
        r4: float = 1.0
        r5: float = 0.0
        reward: float = 0.85

    def compute_rewards(_episode: Any) -> Rewards:
        return Rewards()

    mod.Rewards = Rewards
    mod.compute_rewards = compute_rewards
    mod.RewardComputationError = type("RewardComputationError", (Exception,), {})
    sys.modules[name] = mod
    return mod


_install_rewards_stub()


# ---------------------------------------------------------------------------
# Audio stub — patched into cells.step_09_audio so app.lifespan succeeds
# without instantiating Kokoro / Whisper.
# ---------------------------------------------------------------------------


class _StubTTS:
    def synthesize(self, *_args: Any, **_kw: Any) -> bytes:
        return b"\x00\x00"


class _StubASR:
    def transcribe(self, *_args: Any, **_kw: Any) -> Any:
        return types.SimpleNamespace(text="hi", language_detected="en", confidence=0.5, duration_s=0.1)


@pytest.fixture(autouse=True)
def _stub_audio_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``app._eager_load_models`` to return synthetic engines.

    We need the patch in place BEFORE ``TestClient`` triggers the lifespan.
    Importing ``app`` here ensures the module exists; if the test re-imports
    ``app`` later (fresh-app tests), it must re-apply the patch itself.
    """

    if "app" not in sys.modules:
        # Cannot import yet because DRIFTCALL_ENV_TOKEN may not be set; the
        # client fixture re-imports and re-patches.
        return
    monkeypatch.setattr("app._eager_load_models", lambda: None)


# ---------------------------------------------------------------------------
# Token + client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def valid_bearer_token() -> str:
    return secrets.token_urlsafe(32)


@pytest.fixture
def session_id_alpha() -> str:
    return "session-alpha-0001"


@pytest.fixture
def session_id_beta() -> str:
    return "session-beta-0002"


@pytest.fixture
def fastapi_test_client(
    monkeypatch: pytest.MonkeyPatch, valid_bearer_token: str
) -> Iterator[TestClient]:
    monkeypatch.setenv("DRIFTCALL_ENV_TOKEN", valid_bearer_token)

    # Re-import app fresh so it reads the new env var.
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "_eager_load_models", lambda: None)
    with TestClient(app_module.app) as client:
        client.app_module = app_module
        yield client


def _auth_headers(token: str, sid: str | None = None) -> dict[str, str]:
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if sid is not None:
        h["X-Session-Id"] = sid
    return h


def assert_error_envelope(resp: Any, code: str, http_status: int) -> None:
    assert resp.status_code == http_status, (resp.status_code, resp.text)
    assert resp.headers.get("Cache-Control") == "no-store", resp.headers
    body = resp.json()
    assert set(body.keys()) == {"error"}, body
    assert body["error"]["code"] == code, body
    assert isinstance(body["error"]["message"], str)
    assert "request_id" in body["error"]
    if code == "max_sessions":
        assert resp.headers.get("Retry-After") == "30"
    else:
        assert resp.headers.get("Retry-After") is None


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_returns_200_plaintext_ok(fastapi_test_client: TestClient) -> None:
    resp = fastapi_test_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"
    assert resp.headers["Content-Type"].startswith("text/plain")


def test_healthz_unauthenticated(fastapi_test_client: TestClient) -> None:
    """No bearer header — still returns 200 (probe endpoint)."""

    resp = fastapi_test_client.get("/healthz")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Bearer auth (M1)
# ---------------------------------------------------------------------------


def test_reset_missing_authorization_returns_401_M1(
    fastapi_test_client: TestClient, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers={"X-Session-Id": session_id_alpha, "Content-Type": "application/json"},
        json={},
    )
    assert_error_envelope(resp, code="unauthorized", http_status=401)


def test_step_bad_bearer_returns_401_M1(
    fastapi_test_client: TestClient, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers("not-the-token", session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="unauthorized", http_status=401)


def test_state_missing_bearer_returns_401_M1(
    fastapi_test_client: TestClient, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.get(
        "/state",
        headers={"X-Session-Id": session_id_alpha},
    )
    assert_error_envelope(resp, code="unauthorized", http_status=401)


def test_close_wrong_scheme_returns_401_M1(
    fastapi_test_client: TestClient, session_id_alpha: str, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/close",
        headers={
            "Authorization": f"Basic {valid_bearer_token}",
            "X-Session-Id": session_id_alpha,
        },
    )
    assert_error_envelope(resp, code="unauthorized", http_status=401)


# ---------------------------------------------------------------------------
# X-Session-Id (M2)
# ---------------------------------------------------------------------------


def test_reset_missing_x_session_id_returns_400_M2(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, sid=None),
        json={},
    )
    assert_error_envelope(resp, code="missing_session_id", http_status=400)


def test_step_malformed_x_session_id_returns_400_M2(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, "bad session!"),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="missing_session_id", http_status=400)


def test_step_x_session_id_over_64_chars_returns_400_M2(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, "a" * 65),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="missing_session_id", http_status=400)


# ---------------------------------------------------------------------------
# /reset (happy + errors)
# ---------------------------------------------------------------------------


def test_reset_happy_path_returns_200(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 42, "config": {"curriculum_stage": 1}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"observation", "episode_id", "max_turns"}
    assert isinstance(body["episode_id"], str) and body["episode_id"]
    assert isinstance(body["max_turns"], int)
    assert 1 <= body["max_turns"] <= 16
    assert isinstance(body["observation"], dict)
    assert resp.headers["Content-Type"].startswith("application/json")


def test_reset_with_language_weights_returns_200(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 7, "config": {"language_weights": {"hi": 0.5, "en": 0.5}}},
    )
    assert resp.status_code == 200, resp.text


def test_reset_bad_json_returns_400_M7(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        content=b"{not json",
    )
    assert_error_envelope(resp, code="bad_json", http_status=400)


def test_reset_invalid_curriculum_stage_returns_400_M8(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"config": {"curriculum_stage": 99}},
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


def test_reset_payload_over_1mib_returns_413_M11(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    big_payload = {"config": {"language_weights": {"en": 1.0}, "_pad": "x" * (1_100_000)}}
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        content=json.dumps(big_payload).encode("utf-8"),
    )
    assert_error_envelope(resp, code="payload_too_large", http_status=413)


# ---------------------------------------------------------------------------
# /step (happy + errors)
# ---------------------------------------------------------------------------


def test_step_happy_path_returns_200(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1, "config": {"curriculum_stage": 1}},
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hello"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"observation", "reward", "done", "info"}
    assert body["reward"] is None or isinstance(body["reward"], float)
    assert isinstance(body["done"], bool)


def test_step_unknown_session_returns_404_M3(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, "never-existed-0001"),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="session_not_found", http_status=404)


def test_step_invalid_action_shape_returns_400_M8(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    # tool_call without tool_name
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "tool_call"}},
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


def test_step_unknown_action_type_returns_400_M8(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "warp_drive", "message": "boom"}},
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


def test_step_internal_exception_returns_500_M9_no_stacktrace(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    app_module = sys.modules["app"]

    cache: Any = app_module.app.state.driftcall.cache
    entry = cache.get(session_id_alpha)
    assert entry is not None

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("boom-detail")

    monkeypatch.setattr(entry.env, "step", _boom)

    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="internal_error", http_status=500)
    assert "boom-detail" not in resp.text


# ---------------------------------------------------------------------------
# /state (happy + expired)
# ---------------------------------------------------------------------------


def test_state_happy_path_returns_200(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "ping"}},
    )
    resp = fastapi_test_client.get(
        "/state", headers=_auth_headers(valid_bearer_token, session_id_alpha)
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"state", "turn"}
    assert body["turn"] == 1


def test_state_expired_session_returns_404_M4(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    app_module = sys.modules["app"]
    real_mono = app_module._monotonic
    advance = {"v": 0.0}
    monkeypatch.setattr(app_module, "_monotonic", lambda: real_mono() + advance["v"])
    advance["v"] = 3601.0  # past TTL

    resp = fastapi_test_client.get(
        "/state",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
    )
    assert_error_envelope(resp, code="session_expired", http_status=404)


# ---------------------------------------------------------------------------
# /close (happy + already-evicted)
# ---------------------------------------------------------------------------


def test_close_happy_path_returns_200_and_final_state(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    resp = fastapi_test_client.post(
        "/close",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"closed", "final_state"}
    assert body["closed"] is True
    assert isinstance(body["final_state"], dict)


def test_close_on_already_evicted_session_returns_200_with_null_final_state(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/close",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
    )
    assert resp.status_code == 200
    assert resp.json() == {"closed": True, "final_state": None}


# ---------------------------------------------------------------------------
# Session cache direct unit tests
# ---------------------------------------------------------------------------


def test_cache_lru_eviction_on_11th_session(
    monkeypatch: pytest.MonkeyPatch, valid_bearer_token: str
) -> None:
    import asyncio

    monkeypatch.setenv("DRIFTCALL_ENV_TOKEN", valid_bearer_token)
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    cache = app_module.SessionCache(max_sessions=10, ttl_s=3600.0)

    real_mono = app_module._monotonic
    counter = {"v": 0.0}

    def fake_mono() -> float:
        counter["v"] += 1.0
        return real_mono() + counter["v"]

    monkeypatch.setattr(app_module, "_monotonic", fake_mono)

    closes: list[str] = []

    class _SpyEnv:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closes.append(self.name)

    async def _run() -> None:
        for i in range(10):
            await cache.insert_or_replace(f"s{i}", lambda i=i: _SpyEnv(f"s{i}"))
        assert cache.size == 10
        await cache.insert_or_replace("s10", lambda: _SpyEnv("s10"))

    asyncio.run(_run())
    assert cache.size == 10
    assert "s10" in cache._store
    assert "s0" not in cache._store
    assert "s0" in closes


def test_cache_max_sessions_returns_429_M5_with_retry_after(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    # Fill cache with 10 freshly-touched sessions.
    for i in range(10):
        resp = fastapi_test_client.post(
            "/reset",
            headers=_auth_headers(valid_bearer_token, f"sid-{i:02d}"),
            json={"seed": i},
        )
        assert resp.status_code == 200, resp.text

    # Freeze time so LRU ages all entries to 0 — cache.insert_or_replace
    # raises max_sessions because no entry is older than any other.
    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    monotonic_at_now = app_module._monotonic()
    for sid, entry in list(cache._store.items()):
        cache._store[sid] = dataclasses.replace(entry, last_touched=monotonic_at_now + 1)

    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, "sid-overflow"),
        json={"seed": 99},
    )
    assert_error_envelope(resp, code="max_sessions", http_status=429)


# ---------------------------------------------------------------------------
# Lifespan / model-not-ready (M6)
# ---------------------------------------------------------------------------


def test_step_before_lifespan_complete_returns_503_M6(
    valid_bearer_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRIFTCALL_ENV_TOKEN", valid_bearer_token)
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "_eager_load_models", lambda: None)
    # Build a fresh app and seed state manually with models_ready=False so
    # we can hit the M6 guard without driving the lifespan.
    fresh_app = app_module.create_app()
    fresh_app.state.driftcall = app_module._AppState(
        cache=app_module.SessionCache(),
        models_ready=False,
        bearer_token=valid_bearer_token,
    )
    # TestClient without entering the context manager skips lifespan.
    client = TestClient(fresh_app)
    resp = client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, "sid-1"),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="model_not_ready", http_status=503)


# ---------------------------------------------------------------------------
# Reset_in_progress (M12)
# ---------------------------------------------------------------------------


def test_concurrent_reset_returns_409_M12(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First reset acquires the lock; we manually flip lock state to simulate
    an in-flight reset and assert the second reset is rejected."""

    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )

    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    entry = cache.get(session_id_alpha)
    assert entry is not None

    # Manually acquire the per-session lock so the next /reset sees it locked.
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(entry.lock.acquire())
        try:
            resp = fastapi_test_client.post(
                "/reset",
                headers=_auth_headers(valid_bearer_token, session_id_alpha),
                json={"seed": 2},
            )
            assert_error_envelope(resp, code="reset_in_progress", http_status=409)
        finally:
            entry.lock.release()
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Body shape conformance (§2.1.1)
# ---------------------------------------------------------------------------


def test_reset_response_envelope_keys(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 0},
    )
    body = resp.json()
    assert set(body.keys()) == {"observation", "episode_id", "max_turns"}


def test_step_response_envelope_keys(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 0},
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    body = resp.json()
    assert set(body.keys()) == {"observation", "reward", "done", "info"}


def test_state_response_envelope_keys(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 0},
    )
    resp = fastapi_test_client.get(
        "/state", headers=_auth_headers(valid_bearer_token, session_id_alpha)
    )
    body = resp.json()
    assert set(body.keys()) == {"state", "turn"}


def test_close_response_envelope_keys(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 0},
    )
    resp = fastapi_test_client.post(
        "/close", headers=_auth_headers(valid_bearer_token, session_id_alpha)
    )
    body = resp.json()
    assert set(body.keys()) == {"closed", "final_state"}


# ---------------------------------------------------------------------------
# Cross-session isolation
# ---------------------------------------------------------------------------


def test_cross_session_isolation(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    session_id_beta: str,
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_beta),
        json={"seed": 2},
    )
    fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "alpha-1"}},
    )
    fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "alpha-2"}},
    )
    state_a = fastapi_test_client.get(
        "/state", headers=_auth_headers(valid_bearer_token, session_id_alpha)
    ).json()
    state_b = fastapi_test_client.get(
        "/state", headers=_auth_headers(valid_bearer_token, session_id_beta)
    ).json()
    assert state_a["turn"] == 2
    assert state_b["turn"] == 0


# ---------------------------------------------------------------------------
# Build action helper (unit)
# ---------------------------------------------------------------------------


def test_build_action_rejects_non_dict() -> None:
    if "app" in sys.modules:
        del sys.modules["app"]
    import os as _os

    _os.environ["DRIFTCALL_ENV_TOKEN"] = "token"
    app_module = importlib.import_module("app")
    with pytest.raises(app_module._ApiError) as exc:
        app_module._build_action("not-a-dict")
    assert exc.value.code == "invalid_action"


def test_build_action_rejects_unknown_type() -> None:
    app_module = sys.modules["app"]
    with pytest.raises(app_module._ApiError) as exc:
        app_module._build_action({"action_type": "warp_drive"})
    assert exc.value.code == "invalid_action"


def test_build_action_accepts_speak() -> None:
    app_module = sys.modules["app"]
    action = app_module._build_action({"action_type": "speak", "message": "hi"})
    assert action.action_type.value == "speak"
    assert action.message == "hi"


# ---------------------------------------------------------------------------
# Sweep eviction (direct cache test)
# ---------------------------------------------------------------------------


def test_cache_sweep_evicts_stale(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    real_mono = app_module._monotonic
    monkeypatch.setattr(app_module, "_monotonic", lambda: real_mono() + 4000.0)
    evicted = cache.sweep()
    assert evicted >= 1
    assert session_id_alpha not in cache._store


# ---------------------------------------------------------------------------
# /step on unknown action ensures 400 (not 500)
# ---------------------------------------------------------------------------


def test_step_with_missing_action_field_returns_400(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {}},  # action_type missing
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


# ---------------------------------------------------------------------------
# Invalid seed type → 400
# ---------------------------------------------------------------------------


def test_reset_seed_must_be_int_or_null(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": "forty-two"},
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


# ---------------------------------------------------------------------------
# Body must be a JSON object (M7)
# ---------------------------------------------------------------------------


def test_reset_with_array_body_returns_400_M7(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        content=b"[1, 2, 3]",
    )
    assert_error_envelope(resp, code="bad_json", http_status=400)


# ---------------------------------------------------------------------------
# Async support: ensure pytest-asyncio is reachable, otherwise mark as skip
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    try:
        import pytest_asyncio
        _ = pytest_asyncio
    except ImportError:
        skip_async = pytest.mark.skip(reason="pytest-asyncio not installed")
        for item in items:
            if "asyncio" in item.keywords:
                item.add_marker(skip_async)


# ---------------------------------------------------------------------------
# In-place reset (resets the same session id while it exists)
# ---------------------------------------------------------------------------


def test_reset_in_place_replaces_env(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    resp2 = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 2},
    )
    assert resp2.status_code == 200
    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    entry = cache.get(session_id_alpha)
    assert entry is not None
    assert entry.reset_count == 1


# ---------------------------------------------------------------------------
# Config must be a dict
# ---------------------------------------------------------------------------


def test_reset_with_non_dict_config_returns_400(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"config": "not-a-dict"},
    )
    assert_error_envelope(resp, code="invalid_action", http_status=400)


# ---------------------------------------------------------------------------
# Async lock acquire: cover SessionCache.acquire_lock branch (existing entry)
# ---------------------------------------------------------------------------


def test_acquire_lock_returns_existing(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    import asyncio

    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    entry = cache.get(session_id_alpha)
    assert entry is not None
    loop = asyncio.new_event_loop()
    try:
        lock = loop.run_until_complete(cache.acquire_lock(session_id_alpha))
        assert lock is entry.lock
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# _to_jsonable: cover ActionType + nested dict + tuple branches
# ---------------------------------------------------------------------------


def test_to_jsonable_handles_action_type_and_dataclass() -> None:
    if "app" not in sys.modules:
        importlib.import_module("app")
    app_module = sys.modules["app"]
    from cells.step_04_models import ActionType

    out = app_module._to_jsonable(ActionType.SPEAK)
    assert out == "speak"
    out2 = app_module._to_jsonable({"a": ActionType.TOOL_CALL, "b": (1, 2)})
    assert out2 == {"a": "tool_call", "b": [1, 2]}


# ---------------------------------------------------------------------------
# Body-size middleware via Content-Length header
# ---------------------------------------------------------------------------


def test_body_size_middleware_rejects_oversize_content_length(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    headers = _auth_headers(valid_bearer_token, session_id_alpha)
    headers["content-length"] = str(_MIB := 2 * 1024 * 1024)
    # We don't actually need to send 2 MiB — the middleware reads CL first.
    resp = fastapi_test_client.post(
        "/reset",
        headers=headers,
        content=b"{}",  # CL header lies; that's fine for the test.
    )
    assert_error_envelope(resp, code="payload_too_large", http_status=413)


# ---------------------------------------------------------------------------
# Build env-config rejects non-dict
# ---------------------------------------------------------------------------


def test_build_env_config_non_dict_raises() -> None:
    app_module = sys.modules["app"]
    with pytest.raises(app_module._ApiError) as exc:
        app_module._build_env_config({"config": [1, 2]})
    assert exc.value.code == "invalid_action"


def test_build_env_config_none_returns_empty() -> None:
    app_module = sys.modules["app"]
    out = app_module._build_env_config({})
    assert out == {}


# ---------------------------------------------------------------------------
# Cache evict on missing returns None
# ---------------------------------------------------------------------------


def test_cache_evict_unknown_returns_none() -> None:
    app_module = sys.modules["app"]
    cache = app_module.SessionCache()
    assert cache.evict("not-there") is None


# ---------------------------------------------------------------------------
# Touch on unknown sid → (None, False)
# ---------------------------------------------------------------------------


def test_cache_touch_unknown_returns_none_not_expired() -> None:
    app_module = sys.modules["app"]
    cache = app_module.SessionCache()
    entry, expired = cache.touch("unknown")
    assert entry is None
    assert expired is False


# ---------------------------------------------------------------------------
# /step on closed env returns 400
# ---------------------------------------------------------------------------


def test_step_after_close_returns_404(
    fastapi_test_client: TestClient, valid_bearer_token: str, session_id_alpha: str
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    fastapi_test_client.post(
        "/close", headers=_auth_headers(valid_bearer_token, session_id_alpha)
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="session_not_found", http_status=404)


# ---------------------------------------------------------------------------
# /reset env.reset OSError path → 500 io_error
# ---------------------------------------------------------------------------


def test_reset_env_reset_oserror_returns_500_io(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force env.reset to raise OSError → io_error 500."""

    app_module = sys.modules["app"]
    real_factory = app_module.DriftCallEnv

    class _BadEnv:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def reset(self, seed: int | None = None) -> Any:
            raise OSError("disk full")

        def close(self) -> None:
            return None

    monkeypatch.setattr(app_module, "DriftCallEnv", _BadEnv)
    try:
        resp = fastapi_test_client.post(
            "/reset",
            headers=_auth_headers(valid_bearer_token, session_id_alpha),
            json={"seed": 1},
        )
        assert_error_envelope(resp, code="io_error", http_status=500)
    finally:
        monkeypatch.setattr(app_module, "DriftCallEnv", real_factory)


def test_reset_env_reset_arbitrary_raise_returns_500(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = sys.modules["app"]

    class _BadEnv:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def reset(self, seed: int | None = None) -> Any:
            raise RuntimeError("bork")

        def close(self) -> None:
            return None

    monkeypatch.setattr(app_module, "DriftCallEnv", _BadEnv)
    resp = fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    assert_error_envelope(resp, code="internal_error", http_status=500)


# ---------------------------------------------------------------------------
# Step OSError → 500 io_error
# ---------------------------------------------------------------------------


def test_step_oserror_returns_500_io(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_test_client.post(
        "/reset",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"seed": 1},
    )
    app_module = sys.modules["app"]
    cache = app_module.app.state.driftcall.cache
    entry = cache.get(session_id_alpha)
    assert entry is not None
    monkeypatch.setattr(
        entry.env, "step", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk"))
    )
    resp = fastapi_test_client.post(
        "/step",
        headers=_auth_headers(valid_bearer_token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert_error_envelope(resp, code="io_error", http_status=500)


# ---------------------------------------------------------------------------
# State on never-reset session via stale cache hack
# ---------------------------------------------------------------------------


def test_state_unknown_session_returns_404_not_found(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.get(
        "/state", headers=_auth_headers(valid_bearer_token, "no-such-sid")
    )
    assert_error_envelope(resp, code="session_not_found", http_status=404)


def test_close_unknown_session_returns_200(
    fastapi_test_client: TestClient, valid_bearer_token: str
) -> None:
    resp = fastapi_test_client.post(
        "/close", headers=_auth_headers(valid_bearer_token, "no-such-sid")
    )
    assert resp.status_code == 200
    assert resp.json() == {"closed": True, "final_state": None}
