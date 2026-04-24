# Step 23 — Inline Gradio demo

Builds the Colab + HF Spaces demo UI for DriftCall. Implements `docs/modules/deploy_demo_space.md` §2.2-§2.6, §3.2, §3.3, §3.6, §4.1, §5 and DESIGN.md §11.2, §15.

`build_demo()` (alias `build_ui()`) returns a Gradio 5.x `gr.Blocks` graph with mic input, base/trained checkpoint radio, drift-injection dropdown enumerating the 20 patterns plus `None`, transcript textbox, trace `gr.DataFrame` (5 columns), TTS `gr.Audio(type="numpy")` output, reset button, and a `gr.State` UUID. Heavy deps (`gradio`, `spaces`, `peft`, `transformers`, `torch`, `huggingface_hub`) are loaded lazily so the cell imports cleanly on CPU-only CI.

`infer_turn(audio_tuple, checkpoint, manual_drift, session_id, text_input=None)` is the single mic-to-speaker entrypoint. Catches every error 5.1-5.9; on any failure path returns safe defaults (empty transcript, 1 s silence at 16 kHz, empty DataFrame, empty dict) plus a user-facing `status_msg`. Never writes to disk; never calls `push_to_hub`.

`ModelLoader` is the process-wide base-model singleton. `boot()` loads the 4-bit base then attempts `PeftModel.from_pretrained(..., adapter_name="driftcall")`. On 404 / CheckpointMismatch it sets `_trained_available=False` and stays in baseline-only mode. `generate(messages, checkpoint=...)` hot-swaps via `disable_adapter()` for `"base"` and `set_adapter("driftcall") + enable_adapter_layers()` for `"trained"` — never a double load.

Sessions live in a process-wide registry keyed by UUID. `get_session` is idempotent, caps at 10 concurrent sessions (raises `SessionCapacityError`), and refreshes `last_activity_ms`. `gc_sessions(900)` evicts idle entries. `reset_session` closes the env and clears the trace; the checkpoint is preserved per §3.5.

`DriftToggleBridge` queues one pattern id per session with last-write-wins coalescence (§3.8, §7.3). `consume()` drains and returns `None` afterwards.

`render_trace` is a pure function returning a 5-column DataFrame (`turn_idx, actor, action_or_event, tool_response_preview, reward_delta`) — never mutates state.
