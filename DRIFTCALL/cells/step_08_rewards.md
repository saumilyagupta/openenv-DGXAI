## step_08_rewards

Pure-functional reward pipeline for DriftCall (DESIGN.md §7, docs/modules/rewards.md).
Converts a frozen `Episode` into a frozen `Rewards` record through five independent
signals (R1..R5), Brier calibration, an uncertain floor, and a 3-decimal final reward.
No LLM judge, no I/O, no clock — every computation is reproducible from the transcript
alone.
