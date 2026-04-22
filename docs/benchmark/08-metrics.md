# 08 — Metrics

## Primary

| Metric | Definition |
|---|---|
| **pass@1 (no_test)** | share of tasks where greedy gen passes all 3 hidden asserts, `no_test` mode |
| **pass@1 (with_test)** | same, `with_test` mode |
| **lift** | `pass@1(with_test) − pass@1(no_test)` in percentage points |

## Secondary

| Metric | Definition |
|---|---|
| syntax_valid_rate | `ast.parse` succeeds on extracted code |
| no_code_block_rate | model did not return a fenced block AND response did not parse as Python |
| timeout_rate | sandbox hit wall/cpu limit |
| mean_latency_s | mean wall latency per Ollama call |
| mean_eval_count | mean tokens generated per sample |

## Failure breakdown

Distribution over:
`pass / no_code_block / syntax_error / assertion_error / name_error / import_error / timeout / runtime_error`

Reported for each mode.

## Flip matrix

2×2 table of `no_test × with_test` pass/fail per task. Derived:

- both_pass
- both_fail
- only_no_test (regression)
- only_with_test (lift cases)

## Significance test

McNemar with continuity correction:

```
b = only_no_test
c = only_with_test
chi2 = (abs(b - c) - 1)^2 / (b + c)     # if b + c > 0
p ≈ via chi-square survival, df=1
```

## Difficulty buckets

Group tasks by reference-solution length (`len(code.split("\n"))`):

- easy: ≤ 5 lines
- medium: 6–10 lines
- hard: 11+ lines

Report pass@1 per bucket per mode.
