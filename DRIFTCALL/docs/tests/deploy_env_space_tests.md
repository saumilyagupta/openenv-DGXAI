# deploy_env_space_tests.md — Test Plan for `docs/modules/deploy_env_space.md`

**Target artifact:** `app.py` (FastAPI entrypoint) + `driftcall/routes/*.py` (per-endpoint handlers: `reset.py`, `step.py`, `state.py`, `close.py`, `health.py`) + `driftcall/session_cache.py` (in-process session cache + eviction sweep) + `Dockerfile` + `openenv.yaml`
**Spec doc:** `DRIFTCALL/docs/modules/deploy_env_space.md` (final, sealed 2026-04-24)
**Framework:** `pytest` + `httpx` (via `fastapi.testclient.TestClient`) + `hypothesis` (properties) + `docker` CLI (integration only)
**Owner:** Person B (Rewards & Tests) — domain-reviewed by Person D (Deploy & Story)
**Implements:** deploy_env_space.md §2 (interface), §3 (behavior), §4 (data structures), §5 (error modes M1–M12), §7 (edge cases); `DRIFTCALL/CLAUDE.md §3.1` (nine-section test-plan doc — this plan supplies the five required sections: Unit, Property, Integration, Coverage, Fixtures).
**Coverage targets:** **100% line** + **≥ 95% branch** on `app.py` + `driftcall/routes/*.py` + `driftcall/session_cache.py`. All 12 error modes **M1–M12** must be raised by at least one test.
**Numeric invariants:** HTTP status codes are exact integers (200, 400, 401, 404, 409, 413, 429, 500, 503). TTL values in tests use `time.monotonic()` monkey-patched via `freezegun`-style fixture — wall-clock is never read directly. Bearer tokens are `secrets.token_urlsafe(32)` strings; never hardcoded magic values outside the `valid_bearer_token` fixture.
**Mandatory assertion on every error response:** `json.loads(resp.text) == {"error": {"code": <slug>, "message": <str>}}` and `resp.headers["Cache-Control"] == "no-store"` — enforced by helper `assert_error_envelope(resp, code, http_status)` that all error-path tests call.
**Mandatory assertion on every success response:** `resp.headers["Content-Type"].startswith("application/json")` (except `/healthz` which is `text/plain`).

Fixtures defined in §5 are **shared** with `deploy_demo_space_tests.md` (same names, same canonicalised content). If any fixture changes here, the shared copy in `tests/conftest.py` MUST be updated in lockstep, and `deploy_demo_space_tests.md §5` cross-checked.

---

## 1. Unit Tests

**Organisation:** one `pytest` sub-package mirroring the route layout under `tests/test_deploy_env/`:

```
tests/test_deploy_env/
  __init__.py
  conftest.py                        # fixtures from §5, plus assert_error_envelope helper
  test_healthz.py                    # /healthz — unauthenticated, cheap
  test_auth.py                       # bearer enforcement across all mutating endpoints
  test_session_header.py             # X-Session-Id header validation
  test_reset.py                      # POST /reset happy + error paths
  test_step.py                       # POST /step happy + error paths
  test_state.py                      # GET /state happy + error paths
  test_close.py                      # POST /close happy + error paths
  test_body_schemas.py               # §2.1.1 shape conformance (envelope, not inner dataclass)
  test_session_cache_unit.py         # LRU, TTL, eviction sweep — direct cache tests
  test_error_modes_mapping.py        # M1..M12 matrix — every error mode hit at least once
  test_status_code_map.py            # every row of §2.2 table asserted
  test_lifespan_eager_load.py        # app.py lifespan loads Kokoro+Whisper BEFORE serving
```

**Unit test case inventory — 28 cases total (exceeds the ≥ 20 requirement).**

### 1.1 `/healthz` — `test_healthz.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_healthz_returns_200_plaintext_ok` | No auth header. | `resp.status_code == 200`; `resp.text == "ok"`; `resp.headers["Content-Type"].startswith("text/plain")`; endpoint does **not** require bearer (§3.5 "unauthenticated"). |
| U2 | `test_healthz_works_when_models_loaded` | Lifespan fixture loads stub Kokoro+Whisper. | `resp.status_code == 200`; no 503 raised even under no-auth request. Confirms `/healthz` bypass is independent of model readiness gate for probe liveness. |

### 1.2 Bearer auth — `test_auth.py`

Applies to every mutating endpoint (`/reset`, `/step`, `/state`, `/close`).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U3 | `test_reset_missing_authorization_returns_401_M1` | POST `/reset` with `X-Session-Id` but **no** `Authorization` header. | `assert_error_envelope(resp, code="unauthorized", http_status=401)`; matches **M1**. |
| U4 | `test_step_bad_bearer_returns_401_M1` | POST `/step` with `Authorization: Bearer not-the-token`. | `assert_error_envelope(resp, code="unauthorized", http_status=401)`; matches **M1**. Body must **not** leak the expected token. |
| U5 | `test_state_missing_bearer_returns_401_M1` | GET `/state` with no `Authorization`. | `assert_error_envelope(resp, code="unauthorized", http_status=401)`. |
| U6 | `test_close_wrong_scheme_returns_401_M1` | POST `/close` with `Authorization: Basic <token>` (wrong scheme). | `assert_error_envelope(resp, code="unauthorized", http_status=401)`. Only `Bearer` scheme accepted (§3.5). |

### 1.3 `X-Session-Id` header — `test_session_header.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U7 | `test_reset_missing_x_session_id_returns_400_M2` | POST `/reset` with valid bearer, **no** `X-Session-Id`. | `assert_error_envelope(resp, code="missing_session_id", http_status=400)`; matches **M2**. |
| U8 | `test_step_malformed_x_session_id_returns_400_M2` | POST `/step` with `X-Session-Id: "bad session!"` (space + `!`, violates `[A-Za-z0-9_-]` charset). | `assert_error_envelope(resp, code="missing_session_id", http_status=400)`; matches **M2** (treated as "not a valid session id"). |
| U9 | `test_step_x_session_id_over_64_chars_returns_400_M2` | POST `/step` with `X-Session-Id` of length 65. | `assert_error_envelope(resp, code="missing_session_id", http_status=400)`; matches **M2**. |

### 1.4 `POST /reset` — `test_reset.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U10 | `test_reset_happy_path_returns_200_and_observation_envelope` | Valid bearer, `X-Session-Id: session_id_alpha`, body `{"seed": 42, "config": {"curriculum_stage": 1}}`. | `resp.status_code == 200`; body top-level keys `== {"observation", "episode_id", "max_turns"}`; `episode_id` is a uuid4 string; `max_turns` is `int`, 1 ≤ value ≤ 16; `observation` is a dict. Envelope conformance per §2.1.1. |
| U11 | `test_reset_with_language_weights_returns_200` | Valid bearer, valid session id, body `{"config": {"language_weights": {"hi": 0.5, "ta": 0.5}}}`. | `resp.status_code == 200`; observation includes the requested language distribution's imprint (via `info.config_echo` if exposed — else just assert envelope). |
| U12 | `test_reset_bad_json_returns_400_M7` | POST `/reset` with body `b"{not json"` and `Content-Type: application/json`. | `assert_error_envelope(resp, code="bad_json", http_status=400)`; matches **M7**. |
| U13 | `test_reset_invalid_curriculum_stage_returns_400_M8` | Body `{"config": {"curriculum_stage": 99}}`. | `assert_error_envelope(resp, code="invalid_action", http_status=400)`; matches **M8** (dataclass validation failure on reset config). |
| U14 | `test_reset_payload_over_1mib_returns_413_M11` | Body size = 1 MiB + 1 byte (padded `config` dict). | `assert_error_envelope(resp, code="payload_too_large", http_status=413)`; matches **M11**. |

### 1.5 `POST /step` — `test_step.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U15 | `test_step_happy_path_returns_200` | Session pre-created via `/reset`; body `{"action": {"action_type": "tool_call", "tool_name": "airline.search", "tool_args": {}}}`. | `resp.status_code == 200`; body keys `== {"observation", "reward", "done", "info"}`; `reward` is `float` **or** `None`; `done` is `bool`. Envelope per §2.1.1. |
| U16 | `test_step_unknown_session_returns_404_M3` | No prior `/reset`; POST `/step` with `X-Session-Id: never-existed-0001`. | `assert_error_envelope(resp, code="session_not_found", http_status=404)`; matches **M3**. |
| U17 | `test_step_invalid_action_shape_returns_400_M8` | Session pre-created; body `{"action": {"action_type": "tool_call"}}` (missing `tool_name`). | `assert_error_envelope(resp, code="invalid_action", http_status=400)`; matches **M8**. |
| U18 | `test_step_internal_exception_returns_500_M9_no_stacktrace` | Monkey-patch `env.step` to raise `RuntimeError("boom")`. | `assert_error_envelope(resp, code="internal_error", http_status=500)`; matches **M9**. `"boom"` does **not** appear in body (stack-trace suppression §5 rule 1). `resp.json()["error"]["request_id"]` is present (ASGI scope id). |

### 1.6 `GET /state` — `test_state.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U19 | `test_state_happy_path_returns_200` | Session pre-created via `/reset` then two `/step`s. | `resp.status_code == 200`; body keys `== {"state", "turn"}`; `turn == 2` (int). Envelope per §2.1.1. |
| U20 | `test_state_expired_session_returns_404_M4` | Session exists at `t0`; monotonic clock advanced by 3601 s via fixture; sweep runs; GET `/state`. | `assert_error_envelope(resp, code="session_expired", http_status=404)`; matches **M4**. |

### 1.7 `POST /close` — `test_close.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U21 | `test_close_happy_path_returns_200_and_final_state` | Session pre-created. | `resp.status_code == 200`; body keys `== {"closed", "final_state"}`; `closed is True`; `final_state` is a dict. |
| U22 | `test_close_on_already_evicted_session_returns_200_with_null_final_state` | Session was evicted by sweep before `/close` arrives. | `resp.status_code == 200`; `resp.json() == {"closed": True, "final_state": None}` (§2.1.1 "null if session was already evicted"). |

### 1.8 Session cache direct unit tests — `test_session_cache_unit.py`

These bypass HTTP and call the cache API directly, to pin the policy invariants from §3.2.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U23 | `test_cache_lru_eviction_on_11th_session` | Fill cache with sessions `s0..s9` (max=10); insert `s10`. | Cache size remains `== 10`; `s0` (oldest `last_touched`) is evicted; `s10` is present; `env.close()` was called on the evicted entry (spy assertion). §3.2 invariant. |
| U24 | `test_cache_ttl_sweep_evicts_stale_entries` | Insert `s_old` at `t0`; advance monotonic clock by 3601 s; call `cache.sweep()`. | `s_old` no longer in cache; spy confirms `env.close()` called; cache remains internally consistent (len == 0). §3.3. |
| U25 | `test_cache_max_sessions_returns_429_M5_with_retry_after` | Cache full of 10 fresh sessions (all touched < 1 s ago); POST `/reset` with a new `X-Session-Id`. | `resp.status_code == 429`; `assert_error_envelope(resp, code="max_sessions", http_status=429)`; `resp.headers["Retry-After"] == "30"` (only M5 carries Retry-After — §5 rules). Matches **M5**. |

### 1.9 Error-mode matrix — `test_error_modes_mapping.py`

One parametrized test asserting **M1..M12** are each reachable and return the expected HTTP code + slug. Parameters:

```
[
  ("M1",  "unauthorized",       401, <bad_bearer_request>),
  ("M2",  "missing_session_id", 400, <no_session_header_request>),
  ("M3",  "session_not_found",  404, <step_on_unknown_sid>),
  ("M4",  "session_expired",    404, <step_after_ttl_expiry>),
  ("M5",  "max_sessions",       429, <reset_when_cache_full>),
  ("M6",  "model_not_ready",    503, <step_before_lifespan_load>),
  ("M7",  "bad_json",           400, <malformed_body>),
  ("M8",  "invalid_action",     400, <wrong_action_shape>),
  ("M9",  "internal_error",     500, <env_step_raises>),
  ("M10", "io_error",           500, <tmpfs_full_monkeypatch>),
  ("M11", "payload_too_large",  413, <oversize_body>),
  ("M12", "reset_in_progress",  409, <concurrent_reset_same_sid>),
]
```

| # | Name | Setup | Assertion |
|---|---|---|---|
| U26 | `test_error_modes_M1_through_M12_full_matrix` | Parametrized over the 12 tuples above. | For every row: `resp.status_code == expected_http`; `resp.json()["error"]["code"] == expected_slug`; `resp.headers["Cache-Control"] == "no-store"`; `resp.headers.get("Retry-After")` is `"30"` iff row is M5 else absent. |

### 1.10 Lifespan eager load — `test_lifespan_eager_load.py`

| # | Name | Setup | Assertion |
|---|---|---|---|
| U27 | `test_lifespan_loads_models_before_serving_requests` | Instrument `audio.tts_kokoro.load` and `audio.asr_whisper.load` with call-counter. Start app via `LifespanManager`; issue `/reset` immediately after startup event fires. | Call-counters `== 1` each **before** any request handler runs (assertion inside lifespan startup). Request returns 200, never 503. §7.3. |
| U28 | `test_step_before_lifespan_complete_returns_503_M6` | Monkey-patch lifespan to defer model load; issue `/step` during the deferred window. | `assert_error_envelope(resp, code="model_not_ready", http_status=503)`; matches **M6**. Confirms the guard exists before models are ready. |

---

## 2. Property Tests

Hypothesis-driven invariants on the deployment surface. Minimum **5 properties**; this plan specifies **7** (two extra for margin).

### 2.1 `P1` — `/step` is idempotent on invalid action (env state unchanged)

**Strategy:** `invalid_action_strategy = hypothesis.strategies.dictionaries(...)` producing action bodies that fail pydantic validation (missing fields, wrong types, unknown `action_type`).

**Invariant:**
```
pre_state  = GET /state (turn = T)
resp       = POST /step with invalid action     # → 400 M8
post_state = GET /state
assert pre_state == post_state                  # turn unchanged, drift_schedule unchanged
assert resp.status_code == 400
```

Confirms §7.5 transactional step semantics: state only mutates after all work succeeds; a rejected action is a no-op.

### 2.2 `P2` — Session expiration is monotonic and consistent

**Strategy:** `st.integers(min_value=0, max_value=7200)` for synthetic elapsed seconds.

**Invariant:**
```
For any elapsed ∈ [0, 7200]:
  if elapsed < 3600: /step returns 200 (session alive)
  if elapsed >= 3600: /step returns 404 M4 (session expired)
Once expired, the session NEVER becomes alive again without a new /reset.
```

Tests monotone one-way transition: `alive → expired` is terminal. §3.2 TTL = 3600 s.

### 2.3 `P3` — Error envelope shape is universal

**Strategy:** parametrized across all 12 error-triggering inputs (from U26 matrix).

**Invariant:** every error response satisfies:
```
body = resp.json()
set(body.keys()) == {"error"}
set(body["error"].keys()) >= {"code", "message"}
isinstance(body["error"]["code"], str) and body["error"]["code"] != ""
isinstance(body["error"]["message"], str)
"traceback" not in json.dumps(body).lower()
"bearer" not in body["error"]["message"].lower()   # no token leakage
```

### 2.4 `P4` — `X-Session-Id` charset and length round-trip

**Strategy:** `st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=64)` generates valid session ids; a second strategy generates invalid ones (containing `!@# `, length 0, length 65+).

**Invariant:**
```
valid_sid   → /reset returns 200
invalid_sid → /reset returns 400 M2
After /reset with valid_sid:
  GET /state with the same sid returns 200
  GET /state with ANY other sid returns 404 M3
```

### 2.5 `P5` — LRU eviction preserves cache size cap

**Strategy:** `st.lists(st.text(alphabet=string.ascii_letters, min_size=8, max_size=16), min_size=11, max_size=50, unique=True)` — sequences of distinct session ids.

**Invariant:** after POSTing `/reset` for every sid in the list (one at a time):
```
len(cache) == min(len(sids), 10)
The 10 present sids are exactly the 10 most-recently-inserted (by last_touched).
No env instance is leaked (every evicted env had .close() called exactly once).
```

### 2.6 `P6` — Reward field is float-or-null

**Strategy:** parametrized over valid actions per `DriftCallAction` shape.

**Invariant:** every `/step` 200-response body satisfies:
```
reward = body["reward"]
assert reward is None or (isinstance(reward, float) and -1.0 <= reward <= 1.0)
assert isinstance(body["done"], bool)
```

Pins §2.1.1 envelope: `reward: float | null`, range aligned with `openenv.yaml` `reward.range: [-1.0, 1.0]` (§4.3).

### 2.7 `P7` — Concurrent `/reset` on same sid never produces two envs

**Strategy:** `hypothesis.stateful.RuleBasedStateMachine` driving concurrent `/reset` calls on the same `X-Session-Id` via `anyio.create_task_group`.

**Invariant:**
```
Across N concurrent /reset calls on the same sid:
  exactly one succeeds with 200 (winner)
  the remaining N-1 return 409 M12 (reset_in_progress)
  cache ends with exactly one env under that sid
  no env instance is leaked
```

§7.1 per-session asyncio lock invariant.

---

## 3. Integration Tests

Cross-cutting scenarios that exercise real subsystems. Marked `@pytest.mark.integration`; run in CI only, not in the fast `pytest tests/` loop.

### 3.1 `I1` — End-to-end curl flow: `/reset` → 6× `/step` → `/state` → `/close`

**Mechanism:** `subprocess.run(["curl", ...])` against a locally-booted FastAPI app (via `uvicorn` subprocess, port 7860). Uses the **real** `curl` binary to exercise headers + HTTP/1.1 semantics exactly as judges will.

**Flow:**
1. Start uvicorn in a subprocess, wait for `/healthz` to return `ok` (max 45 s, matches `HEALTHCHECK --start-period=45s` in §4.2).
2. `curl -X POST /reset` with bearer + `X-Session-Id: e2e-001`, body `{"seed": 42, "config": {"curriculum_stage": 1}}`. Assert 200.
3. Loop 6 times: `curl -X POST /step` with a `tool_call` action. Assert 200 each time; accumulate `done` values.
4. `curl /state`. Assert 200; `turn >= 6`.
5. `curl -X POST /close`. Assert 200; `closed is True`.
6. Kill uvicorn subprocess; assert no zombie process.

**Budget:** single test must complete under 60 s including subprocess boot.

### 3.2 `I2` — Docker build locally + `openenv validate`

**Mechanism:** `docker build -t driftcall-env:test -f DRIFTCALL/Dockerfile DRIFTCALL/` then `docker run -d -p 7860:7860 -e DRIFTCALL_ENV_TOKEN=test-token driftcall-env:test`, then `openenv validate http://localhost:7860 --auth-bearer test-token`.

**Assertions:**
1. `docker build` exits 0.
2. Image size < 2 GB (`docker image inspect driftcall-env:test --format '{{.Size}}'` < `2 * 1024**3`).
3. Container healthz returns `ok` within 60 s of `docker run`.
4. `openenv validate` exits 0 and its stdout contains each of:
   - `openenv.yaml parses, schema v1.0`
   - `POST /reset` success line
   - `POST /step` success line
   - `GET /state` success line
   - `POST /close` success line
   - `6 endpoints validated, 0 errors`
5. Container cleanup: `docker rm -f` in `finally` block.

**Gating:** marked `@pytest.mark.skipif(not shutil.which("docker"))` — locally opt-in, mandatory in CI.

### 3.3 `I3` — HF Space deploy dry-run (no actual push)

**Mechanism:** `hf upload --dry-run <team>/driftcall-env . --repo-type=space`. Captures the file manifest that **would** be pushed.

**Assertions:**
1. Exit code 0.
2. Manifest includes: `app.py`, `openenv.yaml`, `Dockerfile`, `requirements.txt`, `README.md`, `driftcall/` subtree.
3. Manifest **excludes**: `tests/`, `training/`, `data/raw/`, `.env*`, `*.ipynb`, `.git/`.
4. `README.md` YAML frontmatter contains required keys: `title`, `sdk: docker`, `app_port: 7860`, `emoji`, `colorFrom`, `colorTo` (§4.4).
5. **No actual network call** to `huggingface.co` — enforced via `monkeypatch` on `huggingface_hub` outbound session to raise if reached.

### 3.4 `I4` — Concurrent 10-session load test

**Mechanism:** `anyio.create_task_group` spawning 10 coroutines, each driving a unique `X-Session-Id` through `/reset` → 3× `/step` → `/close` against `TestClient(app)`.

**Assertions:**
1. All 10 `/reset` calls return 200 (cache is exactly at cap).
2. An 11th concurrent `/reset` (while the first 10 are still `last_touched < TTL`) returns **429 M5** with `Retry-After: 30` (proves cap enforcement under contention).
3. All 30 `/step` calls (3 × 10 sessions) return 200; no cross-session state bleed — each session's `observation.turn` progresses independently (`1, 2, 3`).
4. All 10 `/close` calls return 200.
5. Wall-clock budget: total test completes in < 30 s on CI 2-vCPU runner.

### 3.5 `I5` — Cold-start lifespan blocks request serving until models loaded

**Mechanism:** Instrument `audio.tts_kokoro.load` with an artificial 2 s `anyio.sleep`. Boot the app via `LifespanManager` and concurrently fire a `/reset` request at `t=0` (before startup completes).

**Assertions:**
1. The `/reset` request **blocks** until lifespan startup is complete — it does **not** return 503 during the loading window if `app.py` correctly awaits lifespan before accepting requests (this is the FastAPI default).
2. If instead we disable the lifespan gate (test variant), the request returns **503 M6** with `code="model_not_ready"` — proves M6 is reachable and the guard is load-bearing.
3. `/healthz` responds 200 throughout (probe endpoint is cheap and does not require models — §3.5 "unauthenticated").

### 3.6 `I6` — TTL sweep liveness under sustained traffic

**Mechanism:** Run the `TestClient` against the app for 70 s of simulated traffic (monotonic clock advanced via fixture), issuing one `/reset` per synthetic minute with a fresh `X-Session-Id`. Sweep runs every 60 s per §3.3.

**Assertions:**
1. After the 61st synthetic second, the first session's entry has been evicted by the sweep task.
2. A `/step` on that first session returns 404 M4.
3. The sweep task itself does not raise; logs contain exactly one "swept 1 expired session" structured log line per sweep cycle (§3.7 logging fields).

---

## 4. Coverage Target

**Targets (enforced in CI via `pytest --cov-fail-under`):**

| Artifact | Line coverage | Branch coverage |
|---|---|---|
| `app.py` | **100%** | **≥ 95%** |
| `driftcall/routes/reset.py` | **100%** | **≥ 95%** |
| `driftcall/routes/step.py` | **100%** | **≥ 95%** |
| `driftcall/routes/state.py` | **100%** | **≥ 95%** |
| `driftcall/routes/close.py` | **100%** | **≥ 95%** |
| `driftcall/routes/health.py` | **100%** | **100%** (trivial file) |
| `driftcall/session_cache.py` | **100%** | **≥ 95%** |

**Command:**
```
pytest tests/test_deploy_env/ \
  --cov=app \
  --cov=driftcall.routes \
  --cov=driftcall.session_cache \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=100
```

**Branch-coverage carve-outs (documented pragmas, not silent):** the `except asyncio.CancelledError: raise` guard at the bottom of the sweep task's loop is excluded via `# pragma: no cover` — re-raising a cancellation is standard-library contract and triggering it requires injecting a cancellation into the `lifespan` shutdown, which is covered by the lifespan test (I5) at the event-loop level.

**Error-mode coverage ledger — every one of M1..M12 is raised by at least one test:**

| Mode | Raised by | HTTP |
|---|---|---|
| M1 `unauthorized` | U3, U4, U5, U6, U26 | 401 |
| M2 `missing_session_id` | U7, U8, U9, U26, P4 | 400 |
| M3 `session_not_found` | U16, U26, P4 | 404 |
| M4 `session_expired` | U20, U26, P2, I6 | 404 |
| M5 `max_sessions` | U25, U26, I4 | 429 |
| M6 `model_not_ready` | U28, U26, I5 | 503 |
| M7 `bad_json` | U12, U26 | 400 |
| M8 `invalid_action` | U13, U17, U26, P1 | 400 |
| M9 `internal_error` | U18, U26 | 500 |
| M10 `io_error` | U26 (monkeypatched tmpfs full) | 500 |
| M11 `payload_too_large` | U14, U26 | 413 |
| M12 `reset_in_progress` | U26, P7 | 409 |

**HTTP status codes asserted at least once:** `200, 400, 401, 404, 409, 413, 429, 500, 503` — all nine from §2.2.

---

## 5. Fixtures

Defined in `tests/conftest.py` (project-wide) and imported by `tests/test_deploy_env/conftest.py`. **Shared** with `deploy_demo_space_tests.md` — any change here propagates there and vice versa.

### 5.1 `fastapi_test_client`

```
@pytest.fixture
def fastapi_test_client(monkeypatch, valid_bearer_token, stub_audio_models):
    """
    Boots the FastAPI app with lifespan, stub Kokoro+Whisper loaded,
    and bearer token injected into app config.

    Yields a `fastapi.testclient.TestClient` that supports all HTTP verbs
    against the live app (in-process, no socket).

    Lifecycle: uses LifespanManager to fire startup/shutdown events;
    cache is flushed between tests via autouse cache-reset fixture.
    """
    monkeypatch.setenv("DRIFTCALL_ENV_TOKEN", valid_bearer_token)
    from app import app
    with TestClient(app) as client:
        yield client
```

Used by: every unit test in §1, properties P1–P7, integration tests I1, I4, I5, I6.

### 5.2 `valid_bearer_token`

```
@pytest.fixture(scope="session")
def valid_bearer_token() -> str:
    """A freshly-generated URL-safe token, session-scoped so it is stable
    across tests in one pytest run but distinct between runs."""
    return secrets.token_urlsafe(32)
```

Used by: every test that asserts 200 on a mutating endpoint, plus the "bad bearer" tests (which receive `valid_bearer_token + "x"` as the wrong token).

### 5.3 `session_id_alpha`

```
@pytest.fixture
def session_id_alpha() -> str:
    """Deterministic session id for tests that only need one sid."""
    return "session-alpha-0001"
```

Charset and length both pass the header validator (§2.1 headers table).

### 5.4 `session_id_beta`

```
@pytest.fixture
def session_id_beta() -> str:
    """Second deterministic session id for cross-session tests
    (e.g., asserting no state bleed between alpha and beta)."""
    return "session-beta-0002"
```

### 5.5 Helper fixtures (non-shared, internal to this test package)

- `stub_audio_models` — monkeypatches `audio.tts_kokoro.load` and `audio.asr_whisper.load` to return lightweight stubs so lifespan completes in < 50 ms. Used everywhere except I5 (which tests real-ish load behavior).
- `monotonic_clock` — monkeypatches `time.monotonic()` to advance deterministically; used by U20, U24, P2, I6.
- `cache_reset` (autouse) — clears `session_cache._store` between tests; prevents cross-test bleed.
- `assert_error_envelope(resp, code, http_status)` — imported helper, asserts envelope shape + `Cache-Control: no-store` header + optional `Retry-After` when `code == "max_sessions"`.
- `one_mib_plus_one_body` — precomputed `bytes` payload for U14 (M11 oversize test).

**Fixture ownership note:** `fastapi_test_client`, `valid_bearer_token`, `session_id_alpha`, `session_id_beta` live in `tests/conftest.py` at the project root and are the shared set with `deploy_demo_space_tests.md`. Helper fixtures (§5.5) are local to `tests/test_deploy_env/conftest.py` and are **not** shared.

---

## 6. Non-goals (out of scope for this plan)

- Deep per-field validation of `DriftCallObservation` / `DriftCallAction` / `DriftCallState` — owned by `env_tests.md` + `models_tests.md`.
- Reward math correctness — owned by `rewards_tests.md`.
- Kokoro / Whisper model quality — owned by `audio_tests.md`.
- Actual HF Hub pushes — forbidden in tests (§3.3 dry-run only); real push happens in Batch C3 manual verification.
- GPU behavior — deployment is CPU-only (deploy_env_space.md §1, §6.5 explicit non-dependency).
- Cross-worker cache coherence — documented as acceptable 404 path in §3.2 of the spec; not a test target for this hackathon (future hardening).
