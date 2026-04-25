# deploy_demo_space_tests.md — Test Plan for `demo/app_gradio.py` + demo/ modules

**Owner:** Person B (Rewards & Tests), secondary: Person D (Deploy & Story — authored module)
**Target module:** `DRIFTCALL/docs/modules/deploy_demo_space.md` (sealed)
**Implements coverage for:** DESIGN.md §3.4 (Demo topology), §11.2 (Pitch surface), §15 (3-min pitch flow), §10.5 (LoRA merge/swap)
**Frameworks:** `pytest`, `hypothesis`, `gradio` (testing hooks via `blocks.launch(prevent_thread_lock=True)`), `httpx` (HTTP probes), `unittest.mock`
**Status:** DRAFT — pending ≥ 1 fresh critic round (test-plan gate is lighter per CLAUDE.md §3.2 Batch D4)

---

## 0. Scope & Non-goals

The demo Space is the **storytelling surface** (not the OpenEnv grading surface). Its correctness is measured by:

1. **UI component wiring** — every declared `gr.Audio`, `gr.Radio`, `gr.Dropdown`, `gr.DataFrame`, `gr.Textbox`, and reset button mounts, renders, and is reachable from `build_ui()` (`deploy_demo_space.md §2.2`).
2. **Model hot-swap correctness** — `ModelLoader.generate(checkpoint="base" | "trained")` calls `peft`'s `disable_adapter()` / `set_adapter("driftcall")` exactly where the spec mandates (`§2.3`, `§3.2`).
3. **Session isolation** — the process-wide registry keys by UUID, caps at 10 concurrent sessions, enforces a 900 s idle TTL, and never leaks state across tabs (`§3.3`).
4. **LoRA-404 degradation path** — when `<team>/gemma-3n-e2b-driftcall-lora` is missing, `is_trained_available()` returns `False`, the `trained` radio option is greyed out, and the Space still launches (`§3.2`, `§7.4`, error 5.2).
5. **Latency budget** — one full `infer_turn` (mic → ASR → env → model → TTS → return) stays under 8 s on a stubbed ZeroGPU path and under 12 s on the A10G path (`§3.6`).
6. **9 error modes (5.1–5.9)** — every error in `deploy_demo_space.md §5` has a test that triggers it and asserts the user-facing `status_msg` plus safe-default positional returns.
7. **Manual drift injection** — `DriftToggleBridge.queue()` / `consume()` is idempotent, coalesces double-press, and passes `force_drift_pattern=...` to `env.step()` at the correct turn (`§3.8`, `§7.3`).
8. **Fallback video trigger** — when both ZeroGPU and A10G are unavailable, the launch workflow aborts with a deterministic status and the pre-recorded video CTA surfaces (`§3.1`, `risk_book.md` cross-ref).

**Out of scope:** real ZeroGPU runtime (mocked via `spaces.GPU` pass-through patch); real Gemma 3n E2B inference (stubbed via `mock_gemma4_e2b_model` fixture — see §5); real Kokoro TTS synthesis (reuses `audio_tests.md` fixtures via shared conftest); browser-level mic permission (asserted only via `audio_tuple=None` branch); HF Hub push (`openenv validate` runs on the **env** Space, not demo per `§1` of the module doc).

Every test below cites the clause in `docs/modules/deploy_demo_space.md` it covers via `deploy_demo_space.md §X.Y` in the docstring.

---

## 1. Unit tests

All unit tests live in `DRIFTCALL/tests/test_deploy_demo_space.py`. Fixtures (§5) are imported from `tests/conftest.py`. Import lines under test:

```python
from demo.app_gradio import build_ui, infer_turn, warmup_on_boot
from demo.session import (
    DemoSessionState,
    TraceRow,
    get_session,
    reset_session,
    gc_sessions,
)
from demo.model_loader import (
    ModelLoader,
    CheckpointId,
    get_model_loader,
    TrainedAdapterMissingError,
    CheckpointMismatchError,
)
from demo.drift_toggle import DriftToggleBridge
from demo.trace_panel import render_trace
```

The Gradio app is exercised via the `gradio_test_client` fixture which wraps `blocks.launch(prevent_thread_lock=True, server_port=0, prevent_thread_lock=True)` and yields an `httpx.Client` bound to the returned `local_url`. All UI assertions go through this client; we never drive a real browser.

### 1.1 Gradio UI mounts all declared components

**Test 1.1.1 — `test_build_ui_mounts_microphone_component`**
Asserts `build_ui()` returns a `gr.Blocks` whose component tree contains exactly one `gr.Audio(sources=["microphone"])`. Locates the component by walking `blocks.blocks.values()` and filtering `isinstance(c, gr.Audio) and "microphone" in c.sources`. Cites `deploy_demo_space.md §2.2`.

**Test 1.1.2 — `test_build_ui_mounts_checkpoint_radio`**
Asserts there is a `gr.Radio` with `choices == ["base", "trained"]` and `value == "base"` when `is_trained_available()` returns True. Cites `§3.2`.

**Test 1.1.3 — `test_build_ui_greys_trained_when_lora_missing`**
Patches `ModelLoader.is_trained_available` → `False`. Rebuilds UI. Asserts the radio `choices == ["base"]` and its label contains the warning string `"Trained adapter unavailable at boot"`. Cites `§7.4`, error 5.2.

**Test 1.1.4 — `test_build_ui_mounts_drift_toggle_dropdown`**
Asserts a `gr.Dropdown` with `choices == list(DRIFT_PATTERN_IDS) + [None]` and `value is None` by default. Confirms the 20 patterns from `DESIGN.md §6.3` are enumerated exactly. Cites `§2.2`, `§3.8`.

**Test 1.1.5 — `test_build_ui_mounts_trace_dataframe`**
Asserts a `gr.DataFrame` with `wrap=True, max_height=400, interactive=False` and column headers `["turn_idx", "actor", "action_or_event", "tool_response_preview", "reward_delta"]` exist. Cites `§2.6`, `§3.4`.

**Test 1.1.6 — `test_build_ui_mounts_tts_audio_output`**
Asserts a `gr.Audio(type="numpy")` output component exists, label contains `"Speaker"`, and is wired downstream of the `infer_turn` event. Cites `§2.2`, `§3.6`, `§9.5`.

**Test 1.1.7 — `test_build_ui_mounts_reset_button_and_textbox_fallback`**
Asserts presence of a `gr.Button` labelled `"New episode"` and a `gr.Textbox` fallback input (placeholder contains `"type a brief"`). Cites `§2.2`, `§3.5`, open question 2.

### 1.2 `ModelLoader` peft adapter hot-swap

**Test 1.2.1 — `test_generate_base_calls_disable_adapter`**
Patches `PeftModel` with `MagicMock`. Calls `loader.generate(messages, checkpoint="base", seed=0)`. Asserts `model.disable_adapter.call_count == 1` and `model.set_adapter.call_count == 0`. Cites `§3.2` step 2.

**Test 1.2.2 — `test_generate_trained_calls_set_adapter_driftcall`**
Calls `loader.generate(messages, checkpoint="trained", seed=0)`. Asserts `model.set_adapter.call_args == mock.call("driftcall")`, then `model.enable_adapter_layers.call_count == 1`. Cites `§3.2` step 3.

**Test 1.2.3 — `test_generate_trained_raises_when_adapter_missing`**
Constructs `ModelLoader` with `_trained_available=False`. Calls `generate(..., checkpoint="trained")`. Asserts `TrainedAdapterMissingError` is raised with message matching `"Trained adapter unavailable"`. Cites error 5.2.

**Test 1.2.4 — `test_generate_deterministic_given_fixed_seed_and_greedy`**
With `temperature=0.0` and `seed=42`, calls `generate(messages, checkpoint="base")` twice. Asserts byte-identical output strings. Cites `§2.3` "Deterministic given (messages, checkpoint, seed, temperature>0)".

**Test 1.2.5 — `test_is_trained_available_returns_false_on_boot_404`**
Patches `PeftModel.from_pretrained` to raise `huggingface_hub.utils.EntryNotFoundError`. Instantiates `ModelLoader`. Asserts `loader.is_trained_available() is False` and no exception escapes `__init__`. Cites `§7.4`.

### 1.3 Session UUID registry

**Test 1.3.1 — `test_get_session_creates_fresh_env_on_first_call`**
Calls `get_session("uuid-A")`. Asserts returned state has `.env` attribute of type `DriftCallEnv`, `.episode_trace == []`, `.turn_idx == 0`, `.current_checkpoint == "base"`. Cites `§4.1`, `§2.4`.

**Test 1.3.2 — `test_get_session_is_idempotent_for_same_uuid`**
Calls `get_session("uuid-A")` twice. Asserts `state_a1 is state_a2` (identity). Cites `§2.4` "Idempotent".

**Test 1.3.3 — `test_get_session_isolates_different_uuids`**
Calls `get_session("uuid-A")` and `get_session("uuid-B")`. Asserts `state_a is not state_b` and `state_a.env is not state_b.env`. Cites `§3.3` "Cross-tab isolation".

**Test 1.3.4 — `test_get_session_enforces_max_concurrent_10`**
Seeds registry with 10 sessions. Calls `get_session("uuid-11")`. Asserts `SessionCapacityError` is raised (error 5.7). Cites `§3.3` "Max concurrent sessions: 10".

**Test 1.3.5 — `test_gc_sessions_evicts_idle_past_900s`**
Seeds 3 sessions with `last_activity_ms = now - 1000_000` (past TTL) and 2 with `last_activity_ms = now`. Calls `gc_sessions(max_idle_s=900)`. Asserts return value `== 3` and registry size `== 2`. Cites `§3.3` "Idle TTL: 900 s".

**Test 1.3.6 — `test_reset_session_closes_env_and_clears_trace`**
Creates session, populates `.episode_trace` with 5 rows, calls `reset_session("uuid-A")`. Asserts old env's `.close()` was called, new state has empty trace, `turn_idx == 0`, checkpoint preserved. Cites `§3.5` hard reset, `§2.4`.

### 1.4 Latency budget enforcement (< 8 s warm)

**Test 1.4.1 — `test_infer_turn_completes_within_8s_on_stubbed_zerogpu`**
Uses `mock_gemma4_e2b_model` fixture (returns a fixed string after a 2.0 s sleep to simulate warm generate). Stubs ASR (0.3 s), TTS (0.3 s), env.step (0.1 s). Wraps call in `time.perf_counter()`. Asserts elapsed `< 8.0 s`. Cites `§3.6` table "Total ≈ 7.0 s".

**Test 1.4.2 — `test_infer_turn_timeout_at_60s_raises_and_returns_safe_defaults`**
Patches `ModelLoader.generate` to `time.sleep(61)`. Calls `infer_turn(...)` inside a `concurrent.futures.ThreadPoolExecutor` with 60 s timeout. Asserts the returned tuple carries `status_msg` containing `"Turn timed out after 60 s"`, empty transcript, 1 s of silence at 16 kHz, empty DataFrame, empty dict. Cites error 5.9.

### 1.5 All 9 error modes (5.1–5.9)

**Test 1.5.1 — `test_error_5_1_zerogpu_unavailable_auto_retries_once`**
Patches `@spaces.GPU` wrapper to raise `ZeroGPUUnavailableError` once then succeed. Calls `infer_turn`. Asserts one retry occurred (log contains `"retrying in 5 s"`) and final return is normal. Cites error 5.1.

**Test 1.5.2 — `test_error_5_1_second_rejection_falls_to_cpu`**
Patches to raise twice. Asserts the third call uses `device_map="cpu"` and `status_msg` contains `"GPU unavailable"`. Cites error 5.1.

**Test 1.5.3 — `test_error_5_2_trained_adapter_missing_silent_fallback_to_base`**
With `is_trained_available()=False`, user submits with `checkpoint="trained"` (race condition). Asserts `generate` runs with `checkpoint="base"` effective and `status_msg` mentions `"Trained adapter unavailable"`. Cites error 5.2.

**Test 1.5.4 — `test_error_5_3_mic_denied_returns_safe_defaults_when_textbox_empty`**
Calls `infer_turn(audio_tuple=None, text_input="")`. Asserts `status_msg == "No audio received; press mic or type a brief."` and all other outputs are safe defaults. Cites error 5.3.

**Test 1.5.5 — `test_error_5_4_cuda_oom_retries_with_shrunk_context`**
Patches `generate` first call → `torch.cuda.OutOfMemoryError`, second call → returns string. Asserts `torch.cuda.empty_cache` was called, second call was invoked with `max_new_tokens=128`, oldest message dropped. Cites error 5.4.

**Test 1.5.6 — `test_error_5_4_second_oom_fails_turn`**
Patches both calls to raise OOM. Asserts `status_msg == "GPU out of memory this turn; reducing context and retrying."` and safe defaults returned (no third retry). Cites error 5.4.

**Test 1.5.7 — `test_error_5_5_checkpoint_mismatch_on_boot_treated_as_5_2`**
Patches `PeftModel.from_pretrained` to raise `CheckpointMismatchError`. Asserts `ModelLoader.__init__` swallows and sets `_trained_available=False`. Cites error 5.5.

**Test 1.5.8 — `test_error_5_6_audio_decode_error_safe_defaults_no_state_mutation`**
Patches ASR to raise `AudioDecodeError`. Calls `infer_turn`. Asserts `status_msg == "Could not decode mic audio; please try again."` AND `session.turn_idx` is unchanged AND `session.episode_trace` length unchanged. Cites error 5.6.

**Test 1.5.9 — `test_error_5_7_session_capacity_error`**
Saturate with 10 sessions; 11th call. Asserts the returned tuple's `status_msg == "Demo at capacity — try again in a minute."` and no 11th session created. Cites error 5.7.

**Test 1.5.10 — `test_error_5_8_env_step_error_trace_reflects_rejection`**
Patches `DriftCallEnv.step` to raise `EnvStepError("invalid_action")`. Asserts trace DataFrame contains a row `actor="env"`, `reward_delta=0.0`, `action_or_event` contains `"invalid_action"`. Cites error 5.8.

**Test 1.5.11 — `test_error_5_9_timeout_no_state_mutation`**
Forces `@spaces.GPU(duration=60)` exhaustion. Asserts `session.episode_trace` length unchanged (session atomicity). Cites error 5.9, `§3.9`.

### 1.6 Miscellaneous invariants

**Test 1.6.1 — `test_infer_turn_never_writes_to_disk`**
Wraps call with `unittest.mock.patch("builtins.open")` counting writes. Asserts `write_count == 0`. Cites `§2.2` contract "Never writes to disk".

**Test 1.6.2 — `test_infer_turn_never_calls_push_to_hub`**
Patches `huggingface_hub.HfApi.upload_file` → `raise AssertionError`. Runs a full turn. Asserts no exception raised. Cites `§2.2` contract.

> **Unit test total: 26** (meets `≥ 20` requirement).

---

## 2. Property tests (hypothesis)

All property tests live in `DRIFTCALL/tests/test_deploy_demo_space_properties.py`.

### 2.1 Checkpoint toggle purity (base → trained → base ≈ initial memory)

```python
@given(st.lists(st.sampled_from(["base", "trained"]), min_size=2, max_size=20))
@settings(max_examples=50, deadline=None)
def test_property_checkpoint_toggle_is_memory_pure(toggle_sequence, mock_gemma4_e2b_model):
    """For any sequence of toggles that begins and ends at 'base', GPU memory
    after == GPU memory before (within a 2 MiB slack for CUDA allocator noise).

    Asserts: `peft` adapter state is a pointer-flip, never a re-mount.
    Cites deploy_demo_space.md §3.2 step 4."""
```

**Property 2.1** — Toggle purity: `mem_before == mem_after ± 2 MiB` when the sequence is a palindrome-class that returns to `base`. Uses a fake memory counter incremented only on `from_pretrained` (which must never be called post-boot).

### 2.2 Generate output is adapter-scoped

**Property 2.2** — For any `messages` drawn from `st.lists(message_strategy)`, `generate(messages, "base")` and `generate(messages, "trained")` produce **different** token sequences with probability ≥ 0.99 when the stubbed "trained" model is parameterised to add a deterministic suffix the base lacks. Guards against accidentally-no-op adapter swap.

### 2.3 Session isolation under concurrency (10 concurrent)

```python
@given(st.lists(
    st.tuples(st.uuids(), st.sampled_from(["base", "trained"])),
    min_size=2, max_size=10, unique_by=lambda t: t[0]))
@settings(max_examples=30, deadline=None)
def test_property_sessions_never_leak_state(session_batch):
    """For any mix of up to 10 concurrent sessions, each session's
    episode_trace after a parallel `infer_turn` round contains ONLY rows
    whose (session_id) matches that session.
    Cites §3.3 'Cross-tab isolation'."""
```

**Property 2.3** — Cross-session purity: every `TraceRow` observed in session X's trace was produced by a call invoked with that UUID. Uses `threading.Thread` fan-out.

### 2.4 Session UUID collision safety

**Property 2.4** — For any byte-string `stale_uuid` not currently in the registry, `get_session(stale_uuid)` returns a fresh session whose `episode_trace == []` and whose `env` is a fresh instance. No inherited state. Cites `§7.7`.

### 2.5 Drift queue coalescence

**Property 2.5** — For any sequence `patterns: list[str]` of length ≥ 2 enqueued back-to-back, exactly one call to `consume(session_id)` returns `patterns[-1]` (last-write-wins) and a second immediate `consume` returns `None`. Cites `§2.5` invariant, `§3.8`, `§7.3.2`.

### 2.6 `render_trace` never mutates state

**Property 2.6** — For any non-empty `DemoSessionState`, two consecutive `render_trace(state)` calls produce DataFrames that are `DataFrame.equals(...)` each other AND the state's `.episode_trace` object identity and contents are unchanged (assert via `id()` + list snapshot equality). Cites `§2.6` "Never mutates state".

> **Property test total: 6** (meets `≥ 5` requirement).

---

## 3. Integration tests

All integration tests live in `DRIFTCALL/tests/test_deploy_demo_space_integration.py`. They exercise the full wiring with Gemma 3n E2B stubbed, real Whisper ASR (CPU), real Kokoro TTS (CPU), and the real `DriftCallEnv` (audio-enabled, mock vendors).

### 3.1 Full demo flow — mic → Whisper → env → Gemma stub → Kokoro → speaker

**Test 3.1.1 — `test_integration_full_turn_hindi_brief`**
Loads `synth_audio_hindi_brief` (6 s synthetic Hindi WAV from §5). Calls `infer_turn(audio_tuple=(16000, wav), checkpoint="trained", manual_drift=None, session_id="int-1")`. Asserts:
- Transcript is non-empty Hindi string
- `env.step` was called exactly once
- Generate was called with `checkpoint="trained"`
- Returned audio tuple `(sr, np.ndarray)` has `sr == 24000` (Kokoro default) and `len(wav) / sr` between 0.5 s and 10 s
- Trace DataFrame has ≥ 2 rows including `actor="user"` and `actor="agent"`
- `status_msg == ""` (success)

Cites `deploy_demo_space.md §8.2`, `§3.6`.

**Test 3.1.2 — `test_integration_before_after_same_brief`**
Runs 3.1.1 twice — once with `checkpoint="base"`, once with `checkpoint="trained"`. Asserts both runs succeed, trace DataFrames differ in at least one `action_or_event` cell, and the "trained" run produces a non-zero `R2` (drift-detection) reward row while "base" run does not. Cites `§8.2` "side-by-side is the pitch".

### 3.2 Manual drift-injection fires at correct turn

**Test 3.2.1 — `test_integration_manual_drift_fires_on_next_step`**
Session has completed 2 turns (no drift yet). Calls `bridge.queue("int-1", "tc_baggage_halved")`. Calls `infer_turn(...)` for turn 3. Asserts:
- `env.step` was called with `force_drift_pattern="tc_baggage_halved"`
- Trace DataFrame contains a row `actor="drift"`, `action_or_event="manual:tc_baggage_halved"` pinned to `turn_idx=3`
- `bridge.consume("int-1")` now returns `None` (queue drained)

Cites `§8.3`, `§3.8`.

**Test 3.2.2 — `test_integration_manual_drift_overrides_probabilistic`**
Patches the injector RNG so it would fire `schema_rename_price` this turn. Queue `"tc_baggage_halved"` anyway. Assert trace shows exactly ONE drift row (the manual one) and the probabilistic fire is suppressed. Cites `§7.3.1` "judge intent > RNG".

**Test 3.2.3 — `test_integration_double_press_coalesces`**
Queue `"pattern_A"`, then immediately queue `"pattern_B"`. Run `infer_turn`. Assert `env.step` received `force_drift_pattern="pattern_B"`. Assert trace has one drift row with `"manual:pattern_B"`. Cites `§7.3.2`.

### 3.3 A10G fallback via env var

**Test 3.3.1 — `test_integration_a10g_fallback_via_env_var`**
Sets `DRIFTCALL_HARDWARE_FALLBACK=a10g` in env. Patches the YAML writer. Invokes the `deploy_check` workflow helper that reads the fallback var and produces the README front-matter. Asserts the output YAML contains `hardware: a10g-small` (not `zero-gpu`). Cites `§3.1` step 2, `§3.7`.

**Test 3.3.2 — `test_integration_spaces_decorator_is_noop_on_a10g`**
With `DRIFTCALL_HARDWARE=a10g`, patch `spaces.GPU` to `lambda f: f` (identity). Run `infer_turn`. Assert function ran synchronously (no queue acquisition) and latency budget of `< 12 s` holds. Cites `§3.1` "A10G is stateful … decorator is a pass-through".

### 3.4 Pre-recorded video fallback

**Test 3.4.1 — `test_integration_aborts_deploy_when_both_gpus_unavailable`**
Patches ZeroGPU probe → `unavailable`, A10G probe → `unavailable`. Runs the `deploy_check` workflow. Asserts it raises `DeploymentAbortedError("both-gpus-unavailable")` and emits a log line `"Fall back to pre-recorded video — see risk_book.md"`. Cites `§3.1` step 3, `risk_book.md` cross-reference.

### 3.5 Warmup keepalive smoke

**Test 3.5.1 — `test_integration_warmup_on_boot_pages_in_kernels`**
Calls `warmup_on_boot()` with all heavy deps stubbed. Asserts: ASR `warmup_audio()` invoked once, TTS synth of a silent 200 ms clip invoked once, `ModelLoader.generate` invoked once with a dummy prompt. Total elapsed bounded `< 3 s` under stubs. Cites `§2.2`, `§3.6` "~15 s on ZeroGPU".

> **Integration test total: 9** scenarios spanning mic→speaker, drift toggle, hardware fallback, and warmup.

---

## 4. Coverage target

| Target | Metric | Threshold |
|---|---|---|
| `demo/app_gradio.py` | line coverage | **100%** |
| `demo/app_gradio.py` | branch coverage | **≥ 95%** |
| `demo/session.py` | line coverage | ≥ 95% |
| `demo/model_loader.py` | line coverage | ≥ 90% |
| `demo/drift_toggle.py` | line coverage | 100% |
| `demo/trace_panel.py` | line coverage | 100% |
| `demo/app_gradio.py` — all 9 error modes | at least one test raising AND asserting `status_msg` | **9/9 exercised** |

Enforced via `pytest --cov=demo --cov-branch --cov-fail-under=95`. The 100%-line / 95%-branch gate on `app_gradio.py` is motivated by the fact that this file is the sole entry point the judge interacts with; uncovered lines are by definition pitch-fatal.

**Branch coverage notes:** the following branches are explicitly asserted and must not drop below 95%:
- `is_trained_available()` True vs False at `build_ui()` time → 1.1.2, 1.1.3
- `audio_tuple is None` vs present at `infer_turn` → 1.5.4
- `manual_drift is None` vs present at `infer_turn` → 3.2.1
- Error 5.1 first-retry vs second-rejection vs third-rejection → 1.5.1, 1.5.2
- Error 5.4 first-OOM vs second-OOM → 1.5.5, 1.5.6
- `_trained_available` True / False inside `generate()` → 1.2.3, 1.5.3

---

## 5. Fixtures

All fixtures live in `DRIFTCALL/tests/conftest.py` and are **shared with `env_tests.md`** so the env Space and demo Space test suites stub the same model weights, audio clips, and LoRA paths. Divergence here would double fixture maintenance.

### 5.1 `gradio_test_client`

```python
@pytest.fixture(scope="function")
def gradio_test_client():
    """Launch build_ui() on a free port, yield an httpx.Client bound to it,
    tear down on exit.

    Usage:
        def test_ui(gradio_test_client):
            resp = gradio_test_client.get("/")
            assert resp.status_code == 200
    """
    blocks = build_ui()
    _, local_url, _ = blocks.launch(
        server_port=0, prevent_thread_lock=True, quiet=True, show_error=False,
    )
    try:
        with httpx.Client(base_url=local_url, timeout=30.0) as client:
            yield client
    finally:
        blocks.close()
```

Port `0` → OS-assigned free port; `prevent_thread_lock=True` → `launch()` returns immediately so tests can drive it.

### 5.2 `mock_gemma4_e2b_model`

```python
@pytest.fixture(scope="session")
def mock_gemma4_e2b_model(monkeypatch_session):
    """Patch ModelLoader._load_base and PeftModel.from_pretrained to return
    a lightweight stub that:
      - Tokenizes via a tiny vocab of 256 BPE tokens
      - Generate returns a deterministic string based on (checkpoint, seed,
        messages-hash) — NO actual forward pass
      - Simulates a 2.0 s sleep on first call (cold-start) and 0.1 s subsequently
      - Tracks call count on disable_adapter / set_adapter / enable_adapter_layers
    """
```

Shared with `env_tests.md` §5 so Gemma stubs are byte-identical across suites.

### 5.3 `trained_lora_path`

```python
@pytest.fixture(scope="session")
def trained_lora_path(tmp_path_factory):
    """Produce a synthetic peft adapter directory that peft can load without
    touching the Hub. Contains:
      - adapter_config.json with base_model_name_or_path matching mock_gemma4_e2b_model
      - adapter_model.safetensors with 16 × 16 × 2 LoRA weights (random, seeded=0)
    Returns Path to the directory. Used to exercise mount/unmount paths."""
```

Shared with `training_tests.md` §5 and `env_tests.md` §5.

### 5.4 `synth_audio_hindi_brief`

```python
@pytest.fixture(scope="session")
def synth_audio_hindi_brief():
    """Return a (16000, np.float32[1D]) tuple representing a 6 s synthetic
    Hindi-coded audio clip matching DESIGN.md §15 brief. Generated by
    Kokoro-TTS from the fixed string:
      "Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad"
    with voice_pack='hi_male' and seed=0. Cached on disk at
    tests/fixtures/synth_hindi_brief.wav to avoid regeneration."""
```

Shared with `audio_tests.md` §5 and `env_tests.md` §5 (same caller produces same audio bytes). Cache key: `sha256(text + voice + seed)`.

### 5.5 Additional shared fixtures (from conftest)

| Fixture | Source file | Purpose |
|---|---|---|
| `mock_driftcall_env` | env_tests conftest | In-process `DriftCallEnv(audio_enabled=True)` with mock vendors |
| `fresh_session_registry` | this file | Clears `demo.session._REGISTRY` before every test |
| `mock_zerogpu_available` | this file | Patches `spaces.GPU` to a pass-through identity |
| `frozen_time_ms` | this file | Freezes `time.time_ns()` for TTL sweep tests |
| `mock_drift_injector_rng` | drift_injector_tests conftest | Patches `random.Random` inside injector for deterministic fires |

---

## 6. Test execution

Run from `DRIFTCALL/` directory:

```bash
# Full suite
pytest tests/test_deploy_demo_space*.py -v --cov=demo --cov-branch --cov-fail-under=95

# Only unit
pytest tests/test_deploy_demo_space.py -v

# Only properties (hypothesis)
pytest tests/test_deploy_demo_space_properties.py -v --hypothesis-show-statistics

# Only integration
pytest tests/test_deploy_demo_space_integration.py -v

# Single error mode
pytest tests/test_deploy_demo_space.py -v -k "error_5_4"
```

CI gate (`.github/workflows/test.yml`): all three files run on every PR that touches `demo/` or `DRIFTCALL/docs/modules/deploy_demo_space.md`. Coverage drop below 95% branch / 100% line on `app_gradio.py` → red check.

---

## 7. Critic gate

Per CLAUDE.md §3.2 Batch D4, this test plan is gated by ≥ 1 fresh critic round. The critic verifies:

1. All 9 error modes from `deploy_demo_space.md §5` are exercised (cross-reference §1.5).
2. All 7 edge cases from `deploy_demo_space.md §7` are exercised somewhere (cross-reference §3, §2.4, §2.5).
3. Fixture names match exactly those declared in §5 here and in `env_tests.md` §5 (shared contract).
4. No unit test requires real GPU, real HF Hub, or real ZeroGPU runtime (all mocked per scope).
5. Unit count `≥ 20` (actual: 26), property count `≥ 5` (actual: 6), integration count `≥ 5` (actual: 9).
6. Coverage targets match the 100%-line / 95%-branch gate on `app_gradio.py`.

Critic response format per CLAUDE.md §3.4: `NOTHING_FURTHER` or `NEEDS_CHANGES: <file:section citations>`.
