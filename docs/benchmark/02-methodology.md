# 02 — Methodology

## Evaluation metric

**pass@1** with greedy decoding (temperature=0). A sample passes iff:

1. Generated text contains a parseable Python code block
2. `ast.parse(code)` succeeds
3. Sandbox exec returns exit code 0 on `code + "\n" + "\n".join(test_list)` within timeout
4. All 3 hidden asserts evaluate True

One generation per sample per mode. No sampling, no self-consistency, no best-of-n in the primary run.

## Paired comparison protocol

Same model, same hidden tests, same seed policy (T=0). Only the prompt differs between `no_test` and `with_test`. This isolates the effect of exposing a sample test.

### Flip matrix (per task)

|  | `with_test` pass | `with_test` fail |
|---|---|---|
| `no_test` pass | both-pass | regression (rare, noise indicator) |
| `no_test` fail | **lift** (spec-following wins) | both-fail (hard problems) |

### Significance

McNemar's test on the flip matrix:

```
b = no_test pass ∧ with_test fail
c = no_test fail ∧ with_test pass
χ² = (|b - c| - 1)² / (b + c)    # with continuity correction
```

p < 0.05 → the condition genuinely changes model performance.

## Failure taxonomy

Record for every failed sample:
- `syntax_error` — `ast.parse` fails on extracted code
- `no_code_block` — generation has no fenced block
- `import_error` — sandbox raises ImportError
- `name_error` — undefined name / wrong function signature
- `assertion_error` — function runs but produces wrong output
- `timeout` — CPU or wall clock limit hit
- `runtime_error` — any other exception

## Reproducibility

- Config file committed (`config.yaml`)
- Dataset hash recorded in `results/metrics_*.json`
- Ollama response `done_reason`, `eval_count`, `total_duration` captured per sample
- Fixed temperature = 0.0 for primary run
- `num_predict` capped at 512

## What is NOT measured

- pass@k for k>1 (would require n sampled generations; out of scope for this pass)
- Latency distribution (only aggregate)
- Token-level log-probs (Ollama /api/generate does not expose them reliably)
