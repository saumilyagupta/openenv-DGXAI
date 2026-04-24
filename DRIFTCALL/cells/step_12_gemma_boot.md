# Step 12 — Gemma 4 E2B Boot

Loads `unsloth/gemma-4-E2B-it-bnb-4bit` via `unsloth.FastModel` in 4-bit NF4 with explicit FP16 (V100-safe), attaches LoRA adapters (r=16, α=32, 7 target modules), and asserts the first parameter's dtype is `torch.float16` — the mandatory BF16-slippage halt from `docs/modules/training.md §3.1`. Unsloth/torch imports are lazy so this cell loads on CPU-only machines; heavy work happens only when `boot_gemma()` is called with a real GPU.
