# benckmark-codeforge

MBPP benchmark harness for hosted Gemma4 (Ollama). Comparison mode: **no-test** vs **with-test** prompting.

## Layout

```
benckmark-codeforge/
├── scripts/          # harness code
├── dataset/          # mbpp.jsonl (downloaded on first run)
├── results/
│   ├── raw/
│   │   ├── no_test/      # per-task generations, mode = no test hint
│   │   └── with_test/    # per-task generations, mode = test hint in prompt
│   ├── logs/
│   ├── metrics_no_test.json
│   ├── metrics_with_test.json
│   ├── comparison.csv
│   └── report.md
├── config.yaml
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --config config.yaml
python scripts/report_gen.py
```

## Docs

See `../docs/benchmark/` — plan, methodology, dataset schema, prompt templates, sandbox details, metrics, results.

## Model

Hosted Gemma4 via Ollama — `http://172.22.2.151:7021/api/generate`.

## Dataset

MBPP (Mostly Basic Python Problems) — 974 samples. Fetched from HuggingFace `google-research-datasets/mbpp`.
