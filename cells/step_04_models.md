# Step 04 — Core Dataclasses

Declares the seven immutable types that cross module boundaries in DriftCall: `ActionType`, `DriftCallAction`, `ToolResult`, `DriftEvent`, `GoalSpec`, `DriftCallObservation`, and `DriftCallState`. All dataclasses are `frozen=True`; the module is pure shape with zero runtime behavior, imported by every other cell, the FastAPI server, and the reward suite.
