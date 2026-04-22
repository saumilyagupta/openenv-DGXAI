# 03 — Dataset

## MBPP

**Mostly Basic Python Problems** (Austin et al. 2021). Hand-curated short Python tasks, each with a natural-language description and 3 assert statements.

Source: HuggingFace `google-research-datasets/mbpp`
Config: `full`. Splits: `train=374`, `test=500`, `validation=90`, `prompt=10`. Total = **974**.
This benchmark concatenates all splits (`split: all`) to hit the ~1000-sample target the user requested. The model never sees the reference solutions, so train-vs-test leakage is not a concern within this eval — contamination from pretraining is the real caveat (see below).
License: CC-BY-4.0.

## Schema

Each record:

```json
{
  "task_id": 2,
  "text": "Write a function to find the similar elements from the given two tuple lists.",
  "code": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)",
  "test_list": [
    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

Fields used by the harness:
- `task_id` — primary key, filename for raw results
- `text` — problem description (the ask)
- `test_list` — 3 asserts run in the sandbox (all must pass)
- `test_setup_code` — rarely non-empty, prepended to sandbox script when present

Fields IGNORED:
- `code` — reference solution. Must NEVER be shown to the model.
- `challenge_test_list` — optional stretch tests, not used in headline metric.

## Local storage

After fetch: `benckmark-codeforge/dataset/mbpp.jsonl` (one JSON record per line). Dataset SHA256 recorded in metrics output.

## Split policy

Use the canonical `test` split of the `full` config → 974 samples. No train/val data touched.

## Contamination notice

MBPP is public and likely in Gemma4's pretraining. Results should be read as an *upper bound* of capability — models may have memorized some samples. This affects absolute pass@1 but does not invalidate the paired `no_test` vs `with_test` comparison, since contamination would lift both conditions equally.
