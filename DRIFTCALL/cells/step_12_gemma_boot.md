# Step 12 — Gemma 3n E2B Boot

Loads `unsloth/gemma-3n-E2B-it` via `unsloth.FastModel` in 4-bit Dynamic NF4 with hardware-aware precision (FP16 on V100, BF16 on H100), attaches LoRA adapters (r=16, α=32, vision towers frozen, language + attention + MLP trainable), and asserts the first parameter's dtype matches the target hardware — the mandatory dtype-slippage halt from `docs/modules/training.md §3.1`. Unsloth/torch imports are lazy so this cell loads on CPU-only machines; heavy work happens only when `boot_gemma()` is called with a real GPU.
