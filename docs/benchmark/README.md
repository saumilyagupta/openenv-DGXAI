# Benchmark Documentation — MBPP on Gemma4

Comparison benchmark of hosted Gemma4 (Ollama) on MBPP, evaluated in two prompting conditions: **no-test** vs **with-test**.

## Index

| # | Doc | Purpose |
|---|---|---|
| 01 | [plan.md](01-plan.md) | Scope, deliverables, folder layout |
| 02 | [methodology.md](02-methodology.md) | Eval protocol, pass@k, paired comparison |
| 03 | [dataset.md](03-dataset.md) | MBPP schema, licensing, splits |
| 04 | [setup.md](04-setup.md) | Install, env, connectivity |
| 05 | [prompt-template.md](05-prompt-template.md) | Exact prompts for both modes |
| 06 | [sandbox.md](06-sandbox.md) | Execution isolation, timeouts, failure modes |
| 07 | [running.md](07-running.md) | CLI, resume, concurrency |
| 08 | [metrics.md](08-metrics.md) | Metric definitions |
| 09 | [results.md](09-results.md) | Filled after run |

## Artifacts

All code, datasets, raw outputs live in `benckmark-codeforge/` at repo root.
