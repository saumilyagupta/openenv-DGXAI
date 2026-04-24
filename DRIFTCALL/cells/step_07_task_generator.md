# Generate task briefs

Pure, seeded, deterministic procedural generator that expands the YAML template library into concrete `GoalSpec` briefs for `DriftCallEnv.reset()`. Identical `(seed, stage, language_weights)` triples always produce byte-identical seed utterances after NFC normalization — no global RNG, no `time.time()`, no `hash()`.
