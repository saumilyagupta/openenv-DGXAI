# 04 — Setup

## Prereqs

- Python 3.10+
- Network access to `http://172.22.2.151:7021` (the hosted Ollama instance)
- `gemma4` model loaded on that host

## Install

```bash
cd benckmark-codeforge
pip install -r requirements.txt
```

## Connectivity check

```bash
curl -s -m 15 -X POST http://172.22.2.151:7021/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","prompt":"print hello","stream":false,"options":{"num_predict":20}}'
```

Expected: JSON with `"done_reason":"stop"` and non-empty `"response"`.

## Dataset fetch

Auto on first run, or manual:

```bash
python scripts/dataset_loader.py --out dataset/mbpp.jsonl
```

## Smoke test

```bash
python scripts/run_benchmark.py --config config.yaml --limit 10
```

Should produce `results/raw/no_test/*.json` and `results/raw/with_test/*.json` for 10 task_ids each, and two metrics files.

## Environment variables

None required. All knobs live in `config.yaml`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Connection refused | Verify Ollama reachable: `curl http://172.22.2.151:7021/api/tags` |
| Model not found | `ollama pull gemma4` on the host |
| Sandbox timeouts on every task | Raise `sandbox.cpu_timeout_seconds` in config |
| No code extracted | Model not emitting fenced blocks — lower temperature, check prompt template |
