# 07 — Running

## Full run

```bash
cd benckmark-codeforge
python scripts/run_benchmark.py --config config.yaml
python scripts/report_gen.py
```

## Debug / smoke run

```bash
python scripts/run_benchmark.py --config config.yaml --limit 10
```

## Single mode

```bash
python scripts/run_benchmark.py --config config.yaml --modes no_test
python scripts/run_benchmark.py --config config.yaml --modes with_test
```

## Resume

`runner.resume: true` (default) skips any task already written under `results/raw/{mode}/{task_id}.json`. To redo, delete raw files.

## Concurrency

`runner.concurrency: 8` — parallel in-flight requests to Ollama. Reduce if the host saturates or returns 5xx.

## Output files

```
results/
├── raw/no_test/<task_id>.json
├── raw/with_test/<task_id>.json
├── logs/run.log
├── metrics_no_test.json
├── metrics_with_test.json
├── comparison.csv
└── report.md
```

Each raw file:
```json
{
  "task_id": 11,
  "mode": "no_test",
  "prompt": "...",
  "response": "...",
  "extracted_code": "...",
  "passed": true,
  "reason": "pass",
  "sandbox_stdout": "",
  "sandbox_stderr": "",
  "latency_seconds": 1.82,
  "eval_count": 134,
  "done_reason": "stop"
}
```

## Report regeneration

`scripts/report_gen.py` reads `results/raw/**` and rewrites `metrics_*.json`, `comparison.csv`, `report.md`. Safe to rerun.
