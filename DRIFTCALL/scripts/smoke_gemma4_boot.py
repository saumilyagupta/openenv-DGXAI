"""Gemma 4 E2B boot smoke test on V100.

DESIGN.md §16.A.1 — the mandatory kickoff check before dispatching Batch C1.

Loads `unsloth/gemma-4-E2B-it-bnb-4bit` in 4-bit on the local V100, asserts the
compute dtype is FP16 (NOT BF16 — the V100 has no native BF16 tensor cores and
trying to run BF16 mixed precision triggers grad instability, DESIGN.md §14
Risk 01), generates one short Hindi completion, and prints `SMOKE PASS` or
`SMOKE FAIL` with diagnostic info.

Usage
-----
Full GPU run (what the team lead runs on the V100 box before dispatching C1)::

    python3 scripts/smoke_gemma4_boot.py

CPU sanity (import-only — validates the smoke script itself is syntactically
sound and its direct imports resolve on machines without CUDA / Unsloth)::

    python3 scripts/smoke_gemma4_boot.py --no-gpu

The `--no-gpu` mode deliberately does NOT import `unsloth` or `torch`; those
pull CUDA runtime libs and fail on pure-CPU dev boxes. It validates only the
script's own importability and the argparse surface, then exits 0.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Final

_MODEL_ID: Final[str] = "unsloth/gemma-4-E2B-it-bnb-4bit"
_MAX_SEQ_LEN: Final[int] = 4096
_SEED_UTTERANCE: Final[str] = "नमस्ते, आप कैसे हैं?"
_MAX_NEW_TOKENS: Final[int] = 40


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="smoke_gemma4_boot",
        description=(
            "DriftCall Phase C kickoff smoke test — confirm Gemma 4 E2B boots "
            "on the local V100 with FP16 precision (DESIGN.md §16.A.1)."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help=(
            "Skip weight load + CUDA init. Validates only that this script "
            "imports cleanly and its argparse surface works. Exits 0 on "
            "success. Use on CI / dev boxes without a GPU."
        ),
    )
    return parser.parse_args(argv)


def _print_pass(detail: str) -> None:
    print(f"SMOKE PASS — {detail}")


def _print_fail(detail: str) -> None:
    print(f"SMOKE FAIL — {detail}", file=sys.stderr)


def _run_cpu_sanity() -> int:
    """Import-only sanity path. Exits 0 if the script itself is healthy."""
    _print_pass(
        "CPU sanity — script imports, argparse OK. "
        "Run without --no-gpu on the V100 to exercise the real weight load."
    )
    return 0


def _run_gpu_smoke() -> int:
    """Full V100 boot test. Lazy-imports heavyweight deps so --no-gpu stays light."""
    try:
        import torch
    except ImportError as exc:
        _print_fail(f"torch import failed: {exc}. Install with `pip install torch`.")
        return 1

    if not torch.cuda.is_available():
        _print_fail(
            "CUDA unavailable. The Gemma 4 E2B 4-bit checkpoint requires a "
            "CUDA GPU (V100 target). Use --no-gpu for a CPU sanity check."
        )
        return 1

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[smoke] CUDA device: {device_name} (compute capability {capability})")

    try:
        from unsloth import FastModel
    except ImportError as exc:
        _print_fail(
            f"unsloth import failed: {exc}. "
            "Install pinned deps with `pip install -e '.[dev]'`."
        )
        return 1

    try:
        model, tokenizer = FastModel.from_pretrained(
            _MODEL_ID,
            max_seq_length=_MAX_SEQ_LEN,
            load_in_4bit=True,
            dtype=torch.float16,
        )
    except Exception as exc:
        _print_fail(f"FastModel.from_pretrained raised: {exc!r}")
        traceback.print_exc()
        return 1

    dtype = getattr(model, "dtype", None)
    if dtype is not torch.float16:
        _print_fail(
            f"model.dtype={dtype!r} — expected torch.float16. "
            "V100 lacks native BF16 tensor cores; BF16 training is "
            "unstable (DESIGN.md §14 Risk 01)."
        )
        return 1

    try:
        inputs = tokenizer(_SEED_UTTERANCE, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        _print_fail(f"generate() round-trip failed: {exc!r}")
        traceback.print_exc()
        return 1

    preview = decoded.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:117] + "..."

    _print_pass(
        f"Gemma 4 E2B loaded in FP16 4-bit on {device_name}; "
        f"generated {outputs.shape[-1] - inputs['input_ids'].shape[-1]} new tokens. "
        f'Completion preview: "{preview}"'
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.no_gpu:
        return _run_cpu_sanity()
    return _run_gpu_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
