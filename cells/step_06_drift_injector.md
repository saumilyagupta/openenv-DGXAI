## Step 06 — Drift Injector

Schedules, applies, and catalogues the 20 canonical drift patterns (5 schema + 5 policy + 5 T&C + 3 pricing + 2 transversal payment-auth) per DESIGN.md §6 and docs/modules/drift_injector.md. Deterministic scheduler (blake2b-seeded RNG) produces `()`, `(e,)`, or `(e1, e2)` for stage 1/2/3; `apply_drift` returns a new frozen `DriftCallState` with mutated vendor schema, bumped schema version, and the fired event appended.
