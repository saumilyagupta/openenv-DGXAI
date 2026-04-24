# Cell 11 — DriftCallEnv smoke test

Boots `DriftCallEnv` with a Stage-1 English airline configuration, runs one
episode (search → book → submit, confidence=0.8), computes rewards via
`compute_rewards`, and prints a compact summary table to stdout. Per
`docs/modules/env.md` §8.1 (happy-path trace) and `DESIGN.md` §16.A.2 — this
is the first end-to-end sanity check that every cell from 04 → 10 composes
into a working episode.
