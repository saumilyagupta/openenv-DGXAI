"""End-to-end FastAPI flow tests for the DriftCall env Space.

Implements docs/tests/deploy_env_space_tests.md §3 (integration scenarios)
adapted to the in-process ``fastapi.testclient.TestClient`` so the suite
stays fast and CI-friendly. Real subprocess / Docker / HF Hub variants
(I2, I3) are out-of-scope for the standard pytest run and are noted in
docs/tests/deploy_env_space_tests.md §3.2 / §3.3 as opt-in CI jobs.

Scope:
    - I1 happy path: /reset → 5× /step → /state → /close (full episode trace).
    - Trace property: action history mirrored in /state across the run.
    - Cross-session isolation (slim variant of I4).
    - 11th-session 429 cap under contention (slim variant of I4).
    - Cold-start guard: /step before lifespan completes returns 503 (I5).
"""

from __future__ import annotations

import dataclasses
import importlib
import secrets
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Light stubs identical to test_app.py — keep this file independent so it
# can run in isolation without depending on test_app's import-side-effects.
# ---------------------------------------------------------------------------


def _install_rewards_stub() -> types.ModuleType:
    name = "cells.step_08_rewards"
    if name in sys.modules:
        return sys.modules[name]
    import pathlib
    real_path = pathlib.Path(__file__).resolve().parent.parent / "cells" / "step_08_rewards.py"
    if real_path.is_file():
        return importlib.import_module(name)
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

    rce = type("RewardComputationError", (Exception,), {})
    mod.__dict__["Rewards"] = Rewards
    mod.__dict__["compute_rewards"] = compute_rewards
    mod.__dict__["RewardComputationError"] = rce
    sys.modules[name] = mod
    return mod


_install_rewards_stub()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def valid_bearer_token() -> str:
    return secrets.token_urlsafe(32)


@pytest.fixture
def session_id_alpha() -> str:
    return "session-alpha-e2e"


@pytest.fixture
def session_id_beta() -> str:
    return "session-beta-e2e"


@pytest.fixture
def fastapi_test_client(
    monkeypatch: pytest.MonkeyPatch, valid_bearer_token: str,
) -> Iterator[TestClient]:
    """Boot a fresh app with bearer token + stubbed audio loaders."""
    monkeypatch.setenv("DRIFTCALL_ENV_TOKEN", valid_bearer_token)
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "_eager_load_models", lambda: None)
    with TestClient(app_module.app) as client:
        client.app_module = app_module  # type: ignore[attr-defined]
        yield client


def _auth_headers(token: str, sid: str | None = None) -> dict[str, str]:
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if sid is not None:
        h["X-Session-Id"] = sid
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TRACE_ACTIONS: tuple[dict[str, Any], ...] = (
    {"action_type": "speak", "message": "hello user"},
    {"action_type": "tool_call", "tool_name": "airline.search",
     "tool_args": {"from": "HYD", "to": "BLR", "date": "2026-04-30"}},
    {"action_type": "speak", "message": "found a flight"},
    {"action_type": "clarify", "message": "evening or morning?"},
    {"action_type": "speak", "message": "wrapping up"},
)


def _post_reset(
    client: TestClient, token: str, sid: str, *,
    seed: int = 42, stage: int = 1,
) -> dict[str, Any]:
    resp = client.post(
        "/reset",
        headers=_auth_headers(token, sid),
        json={"seed": seed, "config": {"curriculum_stage": stage}},
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def _post_step(
    client: TestClient, token: str, sid: str, action: dict[str, Any],
) -> dict[str, Any]:
    resp = client.post(
        "/step",
        headers=_auth_headers(token, sid),
        json={"action": action},
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


# ---------------------------------------------------------------------------
# E1 — Full episode trace: /reset → 5× /step → /state → /close
# ---------------------------------------------------------------------------


def test_e2e_full_episode_trace(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
) -> None:
    """Full happy-path episode flow (I1 adapted to in-process TestClient).

    Asserts every endpoint envelope, that turn advances strictly monotonically,
    and that /state mirrors the action history at the end of the run.
    """
    client = fastapi_test_client
    token = valid_bearer_token
    sid = session_id_alpha

    # /reset
    reset_body = _post_reset(client, token, sid, seed=42, stage=1)
    assert set(reset_body.keys()) == {"observation", "episode_id", "max_turns"}
    episode_id = reset_body["episode_id"]
    assert isinstance(episode_id, str) and episode_id
    assert isinstance(reset_body["max_turns"], int)
    assert reset_body["max_turns"] >= 5  # need budget for 5 steps
    obs = reset_body["observation"]
    assert isinstance(obs, dict)
    assert obs.get("turn") == 0

    # 5 /step calls
    last_turn = -1
    dones: list[bool] = []
    for action in _TRACE_ACTIONS:
        step_body = _post_step(client, token, sid, action)
        assert set(step_body.keys()) == {"observation", "reward", "done", "info"}
        assert isinstance(step_body["done"], bool)
        reward = step_body["reward"]
        assert reward is None or isinstance(reward, float)
        if reward is not None:
            assert -1.0 <= reward <= 1.0
        obs = step_body["observation"]
        assert isinstance(obs, dict)
        turn = obs.get("turn")
        assert isinstance(turn, int)
        assert turn > last_turn  # strictly monotone
        last_turn = turn
        dones.append(step_body["done"])
        if step_body["done"]:
            break

    assert last_turn >= 1, "expected at least one step to advance turn"

    # /state mirrors history
    resp = client.get("/state", headers=_auth_headers(token, sid))
    assert resp.status_code == 200, resp.text
    state_body = resp.json()
    assert set(state_body.keys()) == {"state", "turn"}
    assert state_body["turn"] == last_turn
    state = state_body["state"]
    assert isinstance(state, dict)
    assert state.get("episode_id") == episode_id
    actions_in_state = state.get("actions") or []
    assert len(actions_in_state) == last_turn
    # First recorded action should match what we sent.
    first = actions_in_state[0]
    assert first["action_type"] == _TRACE_ACTIONS[0]["action_type"]

    # /close
    resp = client.post("/close", headers=_auth_headers(token, sid))
    assert resp.status_code == 200, resp.text
    close_body = resp.json()
    assert close_body["closed"] is True
    assert isinstance(close_body.get("final_state"), (dict, type(None)))

    # /close on already-evicted session — still 200 with final_state == None.
    resp2 = client.post("/close", headers=_auth_headers(token, sid))
    assert resp2.status_code == 200
    assert resp2.json() == {"closed": True, "final_state": None}


# ---------------------------------------------------------------------------
# E2 — Cross-session isolation (slim I4)
# ---------------------------------------------------------------------------


def test_e2e_two_sessions_progress_independently(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
    session_id_beta: str,
) -> None:
    client = fastapi_test_client
    token = valid_bearer_token

    _post_reset(client, token, session_id_alpha, seed=1)
    _post_reset(client, token, session_id_beta, seed=2)

    # Drive alpha 3 turns; beta 1 turn.
    for _ in range(3):
        _post_step(client, token, session_id_alpha,
                   {"action_type": "speak", "message": "a"})
    _post_step(client, token, session_id_beta,
               {"action_type": "speak", "message": "b"})

    a = client.get("/state", headers=_auth_headers(token, session_id_alpha)).json()
    b = client.get("/state", headers=_auth_headers(token, session_id_beta)).json()
    assert a["turn"] == 3
    assert b["turn"] == 1
    assert a["state"]["episode_id"] != b["state"]["episode_id"]


# ---------------------------------------------------------------------------
# E3 — 11th session under contention returns 429 max_sessions
# ---------------------------------------------------------------------------


def test_e2e_11th_session_returns_429_max_sessions(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the cache cap (§3.2 max_sessions = 10) by freezing the clock so
    LRU eviction cannot fire (no entry is older than itself).
    """
    client = fastapi_test_client
    token = valid_bearer_token
    app_module = sys.modules["app"]
    # Freeze the monotonic clock — every entry has identical last_touched.
    monkeypatch.setattr(app_module, "_monotonic", lambda: 1000.0)

    # Fill cache with 10 sessions.
    for i in range(10):
        resp = client.post(
            "/reset",
            headers=_auth_headers(token, f"e2e-cap-{i:02d}"),
            json={"seed": i, "config": {"curriculum_stage": 1}},
        )
        assert resp.status_code == 200, (i, resp.text)

    # 11th session — clock frozen → no eviction candidate older than itself.
    resp11 = client.post(
        "/reset",
        headers=_auth_headers(token, "e2e-cap-11"),
        json={"seed": 11, "config": {"curriculum_stage": 1}},
    )
    assert resp11.status_code == 429, resp11.text
    body = resp11.json()
    assert body["error"]["code"] == "max_sessions"
    assert resp11.headers.get("Retry-After") == "30"
    assert resp11.headers.get("Cache-Control") == "no-store"


# ---------------------------------------------------------------------------
# E4 — Cold-start /step returns 503 model_not_ready (I5 slim variant)
# ---------------------------------------------------------------------------


def test_e2e_step_before_models_ready_returns_503_M6(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
) -> None:
    """Flip the models_ready flag to False and confirm the M6 guard fires."""
    client = fastapi_test_client
    token = valid_bearer_token
    app_module = sys.modules["app"]

    # First reset normally (so we have a session to /step against later).
    _post_reset(client, token, session_id_alpha)

    # Now simulate cold-start by forcing models_ready=False on the live state.
    app_module.app.state.driftcall.models_ready = False

    resp = client.post(
        "/step",
        headers=_auth_headers(token, session_id_alpha),
        json={"action": {"action_type": "speak", "message": "hi"}},
    )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error"]["code"] == "model_not_ready"
    assert resp.headers.get("Cache-Control") == "no-store"

    # Restore for any subsequent fixture teardown.
    app_module.app.state.driftcall.models_ready = True


# ---------------------------------------------------------------------------
# E5 — Reward envelope is float-or-null on every successful /step
# ---------------------------------------------------------------------------


def test_e2e_reward_is_float_or_null_on_each_step(
    fastapi_test_client: TestClient,
    valid_bearer_token: str,
    session_id_alpha: str,
) -> None:
    """P6 (envelope reward shape) materialised across the trace."""
    client = fastapi_test_client
    token = valid_bearer_token
    sid = session_id_alpha

    _post_reset(client, token, sid)
    for action in _TRACE_ACTIONS:
        body = _post_step(client, token, sid, action)
        reward = body["reward"]
        assert reward is None or (
            isinstance(reward, float) and -1.0 <= reward <= 1.0
        )
        if body["done"]:
            break
