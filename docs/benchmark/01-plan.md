# 01 — Plan

## Goal

Measure code-generation skill of hosted **Gemma4** on ~1000 standard Python coding problems. Quantify the effect of including a sample unit test in the prompt.

## Benchmark

**MBPP** (Mostly Basic Python Problems). 974 samples in `full` config, `test` split.

Why MBPP:
- Standard, widely cited (Austin et al. 2021)
- Each problem has 3 hidden assert tests → reliable pass@1
- Scale matches the user's "~1000 sample" target
- Purely Python, deterministic evaluation possible

## Experimental conditions

Each sample runs under **two prompt modes** against the same model + same hidden tests.

| Mode | Prompt content | Purpose |
|---|---|---|
| `no_test` | problem text only | measure pure code-gen ability |
| `with_test` | problem text + 1 sample assert | measure spec-following with a worked example |

## Deliverables

1. `benckmark-codeforge/results/metrics_no_test.json`
2. `benckmark-codeforge/results/metrics_with_test.json`
3. `benckmark-codeforge/results/comparison.csv` — per-task side-by-side
4. `benckmark-codeforge/results/report.md`
5. `docs/benchmark/09-results.md` — written analysis

## Folder layout

See repo-level `benckmark-codeforge/README.md`.

## Non-goals

- Not evaluating multi-turn dialog, tool use, or agent loops.
- Not measuring cost per solve in dollars (on-prem Ollama).
- Not comparing against other models in this pass (can extend later).

## Timeline

| Phase | Time |
|---|---|
| Scaffold + docs | done |
| Harness scripts | ~15 min |
| Dataset fetch | ~1 min |
| Smoke (10 samples, both modes) | ~2 min |
| Full run (974 × 2 = 1948 gens, concurrency=8) | ~20 min |
| Report | ~5 min |
