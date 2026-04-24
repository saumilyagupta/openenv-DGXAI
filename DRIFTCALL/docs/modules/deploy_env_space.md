# deploy_env_space.md — DriftCall Env HF Space Deployment

**Owner:** Person D (Deploy & Story)
**Implements:** DESIGN.md §3.3 (Deployed Env Topology), §11.1 (Env Space files), §13 (Deliverables)
**Depends on:** `docs/modules/env.md` (FastAPI surface contract), `docs/modules/models.md` (dataclass wire format), `docs/modules/audio.md` (Kokoro + Whisper runtime)
**Status:** DRAFT — pending ≥ 2 fresh critic rounds

---

## 1. Purpose

`driftcall-env` is the production hosting target for the DriftCall OpenEnv RL environment. It runs on a **free-tier Hugging Face Space (Docker SDK, CPU basic, 2 vCPU / 16 GB RAM)** and is the artifact the hackathon judges exercise via `openenv validate`. The Space exposes a FastAPI application implementing the OpenEnv REST contract (`/reset`, `/step`, `/state`, `/close`) plus a lightweight session cache so concurrent training / evaluation runs can share one deployment without state bleed.

The Space is **intentionally CPU-only**. Kokoro TTS (82 M params) and `faster-whisper-small` int8 (~244 M params) both run at roughly real-time on a single modern CPU core; the training topology (DESIGN.md §3.2, §9.4) never loads TTS/ASR because GRPO operates text-in / text-out. This module owns:

1. The Dockerfile (multi-stage build, <2 GB final image, pre-pulled audio weights).
2. `openenv.yaml` metadata (required for `openenv validate`).
3. `requirements.txt` pin set (fastapi, uvicorn, openenv, kokoro, faster-whisper, plus transitive deps).
4. The Space README (Space card) — must satisfy HF Space schema + hackathon submission rules.
5. The session cache implementation sketch delegated to `app.py` (full code in `docs/modules/env.md`; this doc specifies the cache's **deployment constraints** only).
6. The deployment command set (build, push, validate).

This doc is a design spec, not an executable. It must contain every decision needed so a single operator can ship the env Space in one 30-minute sitting on Apr 25 morning (DESIGN.md §12.2 pre-onsite hour 16 gate).

---

## 2. Interface

### 2.1 External HTTP surface (served by the Space)

The Space exposes the OpenEnv REST surface on **port 7860** (HF Spaces Docker SDK convention — any other port is unreachable). All endpoints accept and return `application/json`. Session identity is carried as a request header so the cache can dispatch to the right env instance.

```
POST   /reset           → 200 application/json   # create or recycle a session, return initial observation
POST   /step            → 200 application/json   # advance one turn; returns observation + reward + done
GET    /state           → 200 application/json   # read the current DriftCallState (debug / judge inspection)
POST   /close           → 200 application/json   # explicitly evict a session
GET    /healthz         → 200 text/plain "ok"    # Space healthcheck (HF pings this to mark the Space "running")
GET    /                → 200 text/html          # minimal landing page (see §4.4); NOT the agent surface
```

**Headers (all mutating endpoints):**

| Header | Required | Notes |
|---|---|---|
| `Authorization: Bearer <DRIFTCALL_ENV_TOKEN>` | yes (see §3.5) | Space secret; judge receives this via submission form |
| `X-Session-Id: <uuid4-or-caller-chosen>` | yes | Opaque string, max 64 chars, `[A-Za-z0-9_-]` only |
| `Content-Type: application/json` | yes | UTF-8 |

The endpoint contracts (request / response shapes) are owned by `docs/modules/env.md` and serialize the `DriftCallObservation` / `DriftCallState` / `DriftCallAction` dataclasses defined in `docs/modules/models.md`. This doc only pins the **deployment-visible** aspects: port, headers, auth, status codes.

> **Cross-doc sync note (2026-04-24):** DESIGN.md §3.3 was updated to match this doc's choice of carrying session identity via the `X-Session-Id` HTTP header (previously documented there as a `session_id` query param). Both docs now agree. No behavior change in this spec — the note is recorded so reviewers don't perceive divergence.

### 2.1.1 Success body shapes (top-level only)

Top-level JSON shapes for each success response. Inner dataclass fields (`DriftCallObservation`, `DriftCallAction`, `DriftCallState`) are owned by `docs/modules/env.md` and `docs/modules/models.md` — this section pins only the envelope each endpoint returns.

**`POST /reset`**

Request:
```json
{
  "config": {
    "curriculum_stage": 1,
    "language_weights": { "hi": 0.4, "ta": 0.2, "kn": 0.2, "hinglish": 0.2 },
    "audio_boundary_enabled": true
  },
  "seed": 42
}
```
- `config.curriculum_stage`: `1 | 2 | 3`
- `config.language_weights`: object, keys are language codes, values sum to 1.0
- `config.audio_boundary_enabled`: bool
- `seed`: `int | null`

Response:
```json
{
  "observation": { "...DriftCallObservation..." },
  "episode_id": "uuid4-string",
  "max_turns": 12
}
```

**`POST /step`**

Request:
```json
{ "action": { "...DriftCallAction..." } }
```

Response:
```json
{
  "observation": { "...DriftCallObservation..." },
  "reward": 0.0,
  "done": false,
  "info": { "...opaque..." }
}
```
- `reward`: `float | null` (null when reward is deferred to episode end)

**`GET /state`**

Response:
```json
{
  "state": { "...DriftCallState..." },
  "turn": 3
}
```

**`POST /close`**

Response:
```json
{
  "closed": true,
  "final_state": { "...DriftCallState... | null" }
}
```
- `final_state`: `object | null` (null if session was already evicted)

Deeper field-level detail for `DriftCallObservation`, `DriftCallAction`, and `DriftCallState` lives in `docs/modules/env.md` and `docs/modules/models.md` — do not duplicate it here.

### 2.2 Status code map

| Code | Meaning | Triggered by |
|---|---|---|
| 200 | Success | Normal return |
| 400 | Malformed JSON / missing header / invalid action shape | Parsing or dataclass validation failure |
| 401 | Missing or bad bearer | §3.5 auth check |
| 404 | `X-Session-Id` not in cache (for `/step` / `/state` / `/close`) | Session expired, evicted, or never created |
| 409 | Concurrent `/reset` on same session id (see §7, case 1) | Cache key collision during init |
| 429 | Max concurrent sessions reached | §3.2 cap hit |
| 500 | Unhandled exception inside env step | Bug; logged, stack trace NOT returned in body |
| 503 | Model weights not yet loaded on cold-start | §7, case 3 |

All error bodies are `{"error": {"code": "<slug>", "message": "<user-safe string>"}}`. Internal stack traces never cross the wire.

### 2.3 Outbound network

The Space makes **zero outbound HTTP calls at runtime**. Kokoro and Whisper weights are baked into the image (§4.2); no HF Hub fetches, no telemetry, no phone-home. This is load-bearing because HF Spaces free CPU tier often has slow / rate-limited egress, and because reproducibility demands an offline image.

### 2.4 Container entrypoint

```dockerfile
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
```

Two uvicorn workers (not four) — CPU basic tier has 2 vCPUs, and Kokoro/Whisper hold the GIL on synthesis/transcription; more workers just contend for the same cores.

---

## 3. Behavior Spec

### 3.1 Session lifecycle

A session is an instance of `DriftCallEnvironment` (the class whose full behavior lives in `docs/modules/env.md`). The deployment layer treats each session as an opaque object with `reset()`, `step()`, `state()`, `close()` methods and does not introspect it.

```
client                              Space (app.py)                cache
   │  POST /reset {seed, config}      │                              │
   │  X-Session-Id: S1                │                              │
   ├─────────────────────────────────▶│  look up S1                  │
   │                                  ├─────────────────────────────▶│
   │                                  │◀───── miss ──────────────────┤
   │                                  │  construct env, bind seed    │
   │                                  │  store (env, last_touched)   │
   │                                  ├─────────────────────────────▶│
   │                                  │  env.reset(...) → obs        │
   │◀────────────  200 obs ───────────┤                              │
   │                                  │                              │
   │  POST /step                      │                              │
   ├─────────────────────────────────▶│  lookup S1 → hit             │
   │                                  │  touch last_touched = now    │
   │                                  │  env.step(...) → obs,r,done  │
   │◀────────── 200 obs,r ────────────┤                              │
```

### 3.2 Cache policy (deployment-level invariants)

The cache is an in-process dict, keyed by `X-Session-Id`. The implementation lives in `app.py` (`docs/modules/env.md` §3 "session cache"), but this doc locks the policy:

| Invariant | Value | Source |
|---|---|---|
| Max concurrent sessions | **10** | DESIGN.md §3.3 |
| TTL (time since `last_touched`) | **3600 s = 1 hr** | DESIGN.md §3.3 |
| Storage | In-memory only (no Redis, no disk) | Free tier has no persistent disk writable at runtime; container state resets on Space rebuild |
| Eviction policy | LRU when cap reached; stale-TTL sweep every 60 s | §3.3 |
| Cross-process sharing | None — each uvicorn worker has its own cache | Acceptable because cache is advisory; clients that get routed to a different worker on re-connect re-issue `/reset` |

**Consequence of the "per-worker cache" choice:** a client's session id may land on worker W1 for `/reset` and W2 for `/step` (uvicorn uses round-robin-ish scheduling on the OS socket). In that case `/step` returns 404 and the client must re-`/reset`. This is acceptable for the hackathon because:

1. Training / eval runs keep a persistent HTTP connection via `requests.Session`, which typically pins to one worker for the life of the socket.
2. Judges use one session end-to-end; they hit `/reset` and then replay steps over the same connection.
3. Two-worker degradation is documented in the Space README so judges don't get silently surprised.

A future hardening path (not in-scope for this hackathon) is to run `--workers 1` with thread pool, or share the cache via `multiprocessing.Manager`. Both are listed in §9.

### 3.3 Eviction sweep

A background asyncio task (started in `app.py` `lifespan`) runs every 60 s:

```
for sid, entry in list(cache.items()):
    if now() - entry.last_touched > TTL:
        env = cache.pop(sid).env
        env.close()  # frees whatever audio buffers the env holds
```

LRU eviction on `/reset` when `len(cache) >= 10` drops the oldest `last_touched` entry first; the new session replaces it.

### 3.4 Streaming / keep-alive

All endpoint responses are single JSON bodies — **no SSE, no websockets, no chunked streaming**. OpenEnv's client library (`openenv.HTTPEnvClient`) uses blocking `POST` + `json()` and a shared `requests.Session`; anything exotic risks failing `openenv validate`. A `/step` call may take up to ~5 s when an audio pass is involved (Kokoro synth + Whisper transcribe on CPU), so we set `--timeout-keep-alive 30` to keep the socket alive comfortably below the 60 s HF Spaces proxy timeout.

### 3.5 Authentication

A single shared-secret bearer guards all mutating endpoints. The token is injected as a HF Space **Secret** named `DRIFTCALL_ENV_TOKEN` and read by `app.py` at import time. `/healthz` is **unauthenticated** (HF Space probes have no bearer).

- Token format: 32+ byte URL-safe random (`secrets.token_urlsafe(32)`).
- Token rotation: delete the Space secret and push a new one; all in-flight sessions 401 on the next request.
- Missing secret at Space boot → container exits 1 (fail-fast).
- The token is bundled with the hackathon submission package so judges can exercise `openenv validate` against the live Space.

### 3.6 Determinism

The deployment does not itself introduce nondeterminism. `env.py` owns seed handling; the cache is a pass-through. However, **two CPU-bound sources of wall-clock variance** can change observable latency (`tool_results[i].latency_ms` is wall-clock, not simulated):

1. Kokoro synth time on the first call after cold start can be 2–3× steady-state due to JIT / lazy graph compile.
2. Whisper VAD + decode time varies with input length.

Neither perturbs reward math — `latency_ms` is informational, never scored.

### 3.7 Logging

Structured JSON logs to stdout (HF Spaces captures stdout into the Logs tab). One log line per request, fields: `ts`, `level`, `session_id`, `endpoint`, `status`, `latency_ms`, `turn`, `err_code` (nullable). No PII, no audio bytes, no bearer token. The full `DriftCallAction` body is logged at DEBUG only, disabled by default.

---

## 4. Data structures

### 4.1 `SessionEntry`

```python
@dataclass(frozen=True)
class SessionEntry:
    env: DriftCallEnvironment        # opaque; see docs/modules/env.md
    created_at: float                # time.monotonic() at /reset
    last_touched: float               # time.monotonic() at every /step|/state
    reset_count: int                 # incremented on in-place /reset (§7, case 1)
```

Frozen per project rule (CLAUDE.md §7). `last_touched` updates produce a new `SessionEntry`; the cache dict replaces the old entry.

### 4.2 Dockerfile layout

Multi-stage build. Stage 1 installs wheels into a throwaway image; stage 2 copies only the site-packages dir and the app code. Target final image < 2 GB (DESIGN.md Risk 10).

```
# -------- Stage 1: builder --------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# Pre-pull model weights so first /reset is fast
RUN pip install --prefix=/install huggingface_hub
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('hexgrad/Kokoro-82M', cache_dir='/weights'); \
               snapshot_download('Systran/faster-whisper-small', cache_dir='/weights')"

# -------- Stage 2: runtime --------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY --from=builder /weights /root/.cache/huggingface
WORKDIR /app
COPY app.py openenv.yaml ./
COPY driftcall/ ./driftcall/
COPY data/ ./data/
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).read()" || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2", "--timeout-keep-alive", "30", "--log-level", "info"]
```

Key decisions:

- `python:3.11-slim` base: smallest stable Python base with glibc (alpine would force musl-incompatible wheels for `faster-whisper` / `ctranslate2`).
- `ffmpeg` installed because Whisper's audio loader shells out to it for anything non-WAV.
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` are hard guarantees — if a download is attempted at runtime it raises, never silently fetches and hangs (§5, mode M6).
- Weights land under `/root/.cache/huggingface`; that's where both Kokoro and faster-whisper look by default.

### 4.3 `openenv.yaml`

```yaml
# openenv.yaml — consumed by `openenv validate`
# Schema source: https://github.com/meta-pytorch/OpenEnv
schema_version: "1.0"
env:
  id: driftcall
  version: "0.1.0"
  display_name: "DriftCall — Indic Voice Concierge under Schema Drift"
  description: >
    OpenEnv-compliant RL environment where a voice-first agent must complete
    Indic consumer concierge tasks while the vendor APIs undergo mid-episode
    schema, policy, T&C, pricing, and auth drift. Five independent reward
    components; deterministic seeded drift; Hindi/Tamil/Kannada/Hinglish
    briefs via Kokoro TTS + faster-whisper ASR.
  license: apache-2.0
  tags:
    - openenv
    - rl
    - voice
    - indic
    - schema-drift
  entrypoint:
    type: http
    base_url: "https://<team>-driftcall-env.hf.space"
    endpoints:
      reset: "/reset"
      step: "/step"
      state: "/state"
      close: "/close"
      health: "/healthz"
    auth:
      type: bearer
      secret_env: DRIFTCALL_ENV_TOKEN
  action_space:
    ref: "docs/modules/models.md#DriftCallAction"
  observation_space:
    ref: "docs/modules/models.md#DriftCallObservation"
  episode:
    max_turns: 16        # worst case, stage-3 curriculum (DESIGN.md §4.5)
    reset_config:
      seed: { type: int, required: false }
      curriculum_stage: { type: int, range: [1, 3], required: false }
      language_weights: { type: object, required: false }
  reward:
    shape: scalar
    range: [-1.0, 1.0]
    components:
      ref: "docs/modules/rewards.md"
```

Field names match the OpenEnv v1.0 schema (`entrypoint.type`, `action_space.ref`, etc.). The `ref` pointers resolve to paths inside the repo; `openenv validate` reads them to assert the env is self-describing.

### 4.4 `README.md` (Space card)

```
---
title: DriftCall Env
emoji: 🧭
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv — Indic voice concierge under schema drift.
---
```

Below the YAML header: one-paragraph description, `openenv validate` command, auth note, link to GitHub, link to the demo Space, link to the HF Hub model + dataset. The README is also rendered as the root `/` route's fallback (Docker Spaces serve nothing at `/` otherwise).

### 4.5 `requirements.txt`

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
pydantic==2.*
openenv==0.2.*            # or whatever is current at build time; version-pin in PR
kokoro==0.9.*
faster-whisper==1.1.*
ctranslate2==4.5.*        # pinned to match faster-whisper's wheel
soundfile==0.12.*
numpy<2.0
huggingface_hub==0.26.*   # only used at build time (snapshot_download)
```

The version set matches `docs/modules/audio.md` §6.1 (upstream consumer) exactly. Pinning is deliberate: the env Space is a reproducibility artifact; judges may rebuild it months from now.

---

## 5. Error modes

Every failure path that can cross the HTTP boundary:

| ID | When | HTTP | Body `error.code` | Recovery |
|---|---|---|---|---|
| M1 | No `Authorization` header, or bad bearer | 401 | `unauthorized` | Client fixes token |
| M2 | No `X-Session-Id` on `/reset`/`/step`/`/state`/`/close` | 400 | `missing_session_id` | Client adds header |
| M3 | `/step`/`/state`/`/close` with unknown session id | 404 | `session_not_found` | Client re-issues `/reset` |
| M4 | Session was in cache but TTL expired between request and handler | 404 | `session_expired` | Client re-issues `/reset` |
| M5 | `/reset` when cache is full and LRU victim cannot be evicted (all 10 slots freshly `last_touched`) | 429 | `max_sessions` | Client backs off and retries; `Retry-After: 30` header set |
| M6 | Kokoro or Whisper model weights missing at startup (image build was broken) | 503 | `model_not_ready` | **Operator** fixes image; client cannot recover |
| M7 | Malformed JSON in request body | 400 | `bad_json` | Client fixes payload |
| M8 | Action fails pydantic / dataclass validation (wrong `ActionType`, missing `tool_name` for `TOOL_CALL`) | 400 | `invalid_action` | Client fixes action |
| M9 | Unhandled exception in `env.step` | 500 | `internal_error` | Logged with request id; client SHOULD NOT retry same action |
| M10 | Disk full writing tmp WAV in audio pipeline | 500 | `io_error` | Very rare on HF Spaces (no writable persistent disk, but /tmp is tmpfs and can fill); operator action |
| M11 | Request body exceeds 1 MiB | 413 | `payload_too_large` | Client trims (should never happen; actions are small) |
| M12 | Concurrent `/reset` on same session id (two requests race) | 409 | `reset_in_progress` | Client serializes resets on its side |

Rules:

- No stack traces in response bodies. `request_id` (uvicorn's ASGI scope id) is included so operators can grep logs.
- All error responses include `Cache-Control: no-store`.
- M5 (`429`) is the **only** code that includes `Retry-After`. Others are terminal for the request.

---

## 6. Dependencies

### 6.1 Upstream (consumed by the deployment artifact)

- **`docs/modules/env.md`** — defines `DriftCallEnvironment.__init__/reset/step/state/close` and the FastAPI route handlers. This doc references but does not duplicate env behavior.
- **`docs/modules/models.md`** — every dataclass crossing the HTTP boundary.
- **`docs/modules/audio.md`** — Kokoro + Whisper integration; tells this doc which weights to pre-pull and what CPU footprint to budget.
- **`docs/modules/rewards.md`** — cited from `openenv.yaml` `reward.components.ref`.
- **DESIGN.md §3.3, §9.1, §9.2, §11.1, §13, Risk 10** — authoritative.

### 6.2 External runtime dependencies (pinned in §4.5)

`fastapi`, `uvicorn[standard]`, `openenv`, `kokoro`, `faster-whisper`, `ctranslate2`, `soundfile`, `pydantic`, `numpy<2.0`, `huggingface_hub` (build-time only).

### 6.3 Hugging Face platform dependencies

- **Space SDK:** `docker` (NOT `gradio`/`static`). The Docker SDK is the only path that lets us bake weights into the image and pin `uvicorn` workers.
- **Space hardware:** `cpu-basic` (free). 2 vCPU, 16 GB RAM, 50 GB ephemeral disk, **no persistent storage**, no GPU.
- **Space secrets:** `DRIFTCALL_ENV_TOKEN` (required).
- **Space env vars:** none (all config is baked in or via `X-Session-Id`).
- **Space region:** default (us-east-1); we do not need region pinning for CPU-basic.

### 6.4 Downstream consumers (who pings this Space)

- `training/eval_baseline.py` and `training/eval_final.py` (DESIGN.md §12) — the training-side `HTTPEnvClient`.
- `demo/app_gradio.py` — the demo Space (documented in `docs/modules/deploy_demo_space.md`) uses this env over HTTP for live runs.
- `openenv validate .` — run against the Space URL as part of the hackathon submission gate.
- Hackathon judges — direct HTTP exercise via curl / the `openenv` CLI.

### 6.5 Explicit non-dependencies

- **No GPU** at runtime (load-bearing; DESIGN.md §3.3).
- **No LLM weights** on the env Space (Gemma 4 lives on the demo Space or on the trainer's local V100).
- **No training code** (`training/` is NOT copied into the image; see §4.2 `COPY` list).
- **No HF Hub network** at runtime (§2.3, §4.2 offline envs).

---

## 7. Edge cases

Six cases the deployment plan must handle correctly. Each is load-bearing for either the 30-min deploy window or the judge's `openenv validate` run.

### 7.1 Concurrent `/reset` on the same session id

Client A and client B both POST `/reset` with `X-Session-Id: S1` within the same ~100 ms window. The cache uses a per-session asyncio lock; the second request observes the session mid-construction.

**Handling:**
- If the first request is still inside `env.__init__`, the second request gets `409 reset_in_progress`. Client is expected to serialize on its side.
- If the first request has completed, the second request performs an in-place reset: the old env is `.close()`'d, a new env replaces it, `reset_count += 1`. This matches `gym`'s idempotent reset semantics.
- `seed` is honored on the winning reset; the losing (409'd) request's seed is discarded.

### 7.2 `/step` on an evicted session

A client idles for 65 minutes between `/step` calls. The sweep task evicts the session at minute 60. The client's next `/step` returns `404 session_expired`.

**Handling:**
- The client MUST re-issue `/reset` with the same or new seed; it cannot resume mid-episode. This is explicit in the Space README.
- No attempt is made to persist episode state across evictions. The free tier has no writable persistent disk, and replaying a seeded episode is cheap (< 1 s on the CPU basic tier).
- `env.close()` is called on eviction to release the Kokoro audio buffer (saves ~80 MB resident per lingering session).

### 7.3 Cold-start model-weight load race

The Space boots. Uvicorn workers start and each lazily triggers a Kokoro + Whisper load on the first audio-involving `/step`. Whisper's CTranslate2 model load takes ~3–5 s; Kokoro takes ~2 s. A `/step` arriving before load completes can block up to ~8 s.

**Handling:**
- `app.py`'s `lifespan` startup hook performs an **eager** load of both models during container boot. This turns cold-start latency into Space "Starting…" time (which HF surfaces via the spinner) instead of a hung client request.
- If eager load fails (bad weights, disk corruption), the container exits 1 and HF's Space restart loop catches it — operator sees the Space status as "Error" instead of silently hanging.
- The first `/healthz` probe is expected at +30 s (`--start-period=45s` on the HEALTHCHECK gives us a comfortable margin).

### 7.4 Kokoro voice pack missing for a language

Kokoro is loaded at startup but an individual voice pack for `language="kn"` (Kannada) is missing from the snapshot cache due to a partial download.

**Handling:**
- `audio/tts_kokoro.py` (per `docs/modules/audio.md` §5) raises `VoicePackMissingError`. The env treats this as a SPEAK-action failure and returns a `tool_results` entry with `status="schema_error"` and `response={"error": "voice_unavailable"}`. The episode continues; reward R4 (format compliance) may drop but R1/R2 are unaffected.
- The image build in §4.2 pre-pulls the **full** Kokoro snapshot (`snapshot_download('hexgrad/Kokoro-82M')`), which includes all voice packs. If a voice pack is missing at runtime, the image is broken — operator fixes the Dockerfile and rebuilds.

### 7.5 HTTP timeout mid-`/step`

A `/step` takes 35 s because Whisper is processing a long utterance and the Space is also handling three concurrent episodes. The HF Space edge proxy has a 60 s idle timeout — we stay under it but only barely.

**Handling:**
- `--timeout-keep-alive 30` means uvicorn holds the connection; the HTTP client's TCP timeout should be ≥ 60 s (default `requests.Session` timeout is infinite — safe).
- Inside `env.step`, audio ops have **hard caps** owned by `audio/*.py`: Whisper `max_duration_s=30`, Kokoro synth implicitly bounded by text length. The env cannot produce a `/step` longer than ~40 s at p99.
- If a `/step` does exceed 60 s (e.g., 10 concurrent sessions all doing audio at once on 2 vCPU), the proxy closes the socket and the client sees `ConnectionError`. Client re-issues; the session is still in the cache and the step was effectively a no-op on the server side because responses are atomic-on-return (state is only mutated after all work succeeds — see `docs/modules/env.md` §3 transactional step semantics).

### 7.6 Out-of-memory during concurrent audio

Five sessions simultaneously run audio-heavy `/step`s. Each Whisper int8 model takes ~250 MB RAM; Kokoro takes ~350 MB. Naive loading would hit `5 × 600 MB = 3 GB` plus Python overhead — well within the 16 GB tier budget, but the Space can still OOM if the image unexpectedly loads fp32 weights.

**Handling:**
- Whisper is forced to `compute_type="int8"` and Kokoro to fp32 (its default is already smallest viable). `audio/*.py` asserts these at load time.
- The models are **singletons** shared across sessions (they are stateless w.r.t. concurrent calls; CTranslate2 releases the GIL during decode). Memory budget is therefore `~600 MB total`, not per-session.
- If an OOM happens, the container is killed by the HF Space OOM-killer and auto-restarts. We lose all in-flight sessions; clients re-`/reset`. The eviction sweep and TTL ensure no permanently-dead sessions pile up.

---

## 8. Examples

### 8.1 End-to-end `/reset` → `/step` flow via curl

```bash
# Assume DRIFTCALL_ENV_TOKEN is set locally for scripting convenience.
TOKEN="${DRIFTCALL_ENV_TOKEN:?export DRIFTCALL_ENV_TOKEN first}"
BASE="https://<team>-driftcall-env.hf.space"

# 1. Reset with seed 42, stage 2 curriculum.
curl -sS -X POST "$BASE/reset" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Session-Id: demo-001" \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "config": {"curriculum_stage": 2}}'
# → 200 {"observation": {"turn": 0, "goal": {...}, "last_transcript": "Bhai Friday ko...", ...}}

# 2. Step: call airline.search.
curl -sS -X POST "$BASE/step" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Session-Id: demo-001" \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "tool_call",
      "tool_name": "airline.search",
      "tool_args": {"origin": "DEL", "destination": "BLR", "date": "2026-04-26"}
    }
  }'
# → 200 {"observation": {...}, "reward": 0.0, "done": false, "info": {"drift_fired": []}}

# 3. Inspect state (judge-only, optional).
curl -sS "$BASE/state" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Session-Id: demo-001"
# → 200 {"episode_id": "...", "turn": 1, "max_turns": 12, "drift_schedule": [...], ...}

# 4. Close.
curl -sS -X POST "$BASE/close" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Session-Id: demo-001"
# → 200 {"closed": true}
```

### 8.2 Container build + smoke + push

```bash
# Local build (from DRIFTCALL/ repo root)
docker build -t driftcall-env:local .

# Local smoke (bind a dummy secret)
docker run --rm -p 7860:7860 \
  -e DRIFTCALL_ENV_TOKEN=dev-local-token \
  driftcall-env:local

# In another shell:
curl -sS http://localhost:7860/healthz            # → "ok"
curl -sS -X POST http://localhost:7860/reset \
  -H "Authorization: Bearer dev-local-token" \
  -H "X-Session-Id: smoke" \
  -H "Content-Type: application/json" -d '{}'
# → 200 with initial observation

# Push to HF Space via the new `hf` CLI.
# The team-lead brief flags that `huggingface-cli` is deprecated; we migrate
# DriftCall/CLAUDE.md §6 row "HF push env" to `hf upload` in a follow-up PR.
hf upload <team>/driftcall-env . --repo-type=space
# (Requires `pip install huggingface_hub>=0.25` and `hf auth login` completed.)
```

### 8.3 `openenv validate` against the live Space

```bash
# Against local container:
openenv validate http://localhost:7860 \
  --auth-bearer dev-local-token

# Against deployed Space:
openenv validate https://<team>-driftcall-env.hf.space \
  --auth-bearer "$DRIFTCALL_ENV_TOKEN"

# Expected output:
#   ✓ openenv.yaml parses, schema v1.0
#   ✓ GET  /healthz → 200 ok
#   ✓ POST /reset   → 200, observation matches observation_space.ref
#   ✓ POST /step    → 200, observation + reward + done
#   ✓ GET  /state   → 200, DriftCallState matches schema
#   ✓ POST /close   → 200
#   ✓ 6 endpoints validated, 0 errors
```

Running this before submission is the DESIGN.md §12.2 hour-16 gate. If it fails, we fix before moving to training.

---

## 9. Open questions

1. **OpenEnv schema version pin:** `openenv==0.2.*` in §4.5 is a placeholder. Confirm the exact current release on the hackathon kickoff morning (Apr 25) and tighten the pin; `openenv validate` schema fields may have shifted between 0.1 and 0.2.
2. **Per-worker cache divergence:** documented in §3.2 as acceptable. Re-evaluate after local load-testing — if even training hits the cross-worker 404 path > 1% of the time, switch to `--workers 1` with a bigger thread pool.
3. **HF Space CPU cold-start time:** the free CPU basic tier can sleep on idle and take 60–120 s to wake. This doc assumes Space is "always-on" because we exercise it during development; if the judge hits a cold Space, the first `/reset` may appear hung. Risk-register coverage owned by `docs/modules/risk_book.md`.
4. **`DRIFTCALL_ENV_TOKEN` rotation during the hackathon:** if the token leaks mid-judging, rotating it 401s the judge mid-run. Do we need a two-token grace period? Likely no (hackathon is 48 h and we trust submission channels), but flag for Person D's risk book.
5. **CLAUDE.md §6 `hf upload` migration:** the hackathon briefing flags `huggingface-cli` as deprecated. Update `DRIFTCALL/CLAUDE.md` §6 rows ("HF push env", "HF push dataset") to `hf upload ... --repo-type=...` in a separate small PR so this design doc doesn't diverge from the command catalogue. Own: Person D.
6. **Image-size margin vs §1.1 Whisper upgrade path:** if `docs/modules/audio.md` §1.1's WER bail-out triggers and we swap to `faster-whisper-medium`, final image grows from ~1.2 GB to ~1.8 GB. Still under the 2 GB Risk-10 bound but with less slack. Re-check image size after any audio-weights change.
7. **`/state` access control:** should `/state` require the same bearer as mutating endpoints, or should we expose a narrower "episode summary" for judges without the full vendor-states dump? Current design keeps full state behind the bearer; revisit if leaderboard ops ask for a public read-only pane.
