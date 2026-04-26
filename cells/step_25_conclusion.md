# Step 25 — Conclusion

Final notebook cell. Prints the eval metrics table, HF Hub links (model + dataset + env Space + demo Space), pitch summary, and the closing line `"Built in 48h, Apache 2.0, see DESIGN.md"`.

`render_conclusion()` is a pure function returning the rendered text — useful for tests. `main(stream)` writes the rendered text to `sys.stdout` (or any caller-supplied stream).

Locked metrics from DESIGN.md §15 + `pitch_demo.md` §3.4 Section 3:
- Task completion (R1): 18% → 64% (+46pp)
- Drift detection (R2): 8% → 71% (+63pp)
- Adaptation latency: 4.2 turns → 1.6 turns
- Anti-hack penalty (R5): ≈ 0
- Format compliance (R4): 0.41 → 0.92

Locked HF Hub links (`pitch_demo.md` §2.3):
- `huggingface.co/driftcall/gemma-3n-e2b-driftcall-lora`
- `huggingface.co/datasets/driftcall/driftcall-indic-briefs`
- `huggingface.co/spaces/driftcall/driftcall-env`
- `huggingface.co/spaces/driftcall/driftcall-demo`

Frozen dataclasses (`FinalMetric`, `HubLink`) keep the locked content tamper-evident.
