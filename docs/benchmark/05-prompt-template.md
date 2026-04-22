# 05 — Prompt Template

Both modes share a system preamble and a strict output contract (Python fenced block). Only the body differs.

## Mode `no_test`

```
You are an expert Python programmer. Solve the following problem.

Problem:
{text}

Write the complete function implementation. Wrap your code in a single fenced Python code block starting with ```python and ending with ```. No prose outside the block.
```

## Mode `with_test`

```
You are an expert Python programmer. Solve the following problem.

Problem:
{text}

Your solution must satisfy this test:
{test_list[0]}

Write the complete function implementation. Wrap your code in a single fenced Python code block starting with ```python and ending with ```. No prose outside the block.
```

## Variable substitution

- `{text}` — MBPP `text` field, stripped
- `{test_list[0]}` — MBPP `test_list[0]`, used only in `with_test` mode

## Code extraction rule

1. Regex `` ```(?:python)?\s*\n(.*?)``` `` on the full response (DOTALL).
2. If no fence: fall back to the entire response if it parses as Python.
3. If neither: mark `no_code_block`, fail the task.

## Notes

- Zero-shot. No few-shot examples.
- No system role separation — Ollama `/api/generate` takes a single prompt string.
- Exactly one hidden test is the maximum shown in `with_test` mode. The remaining 2 asserts are reserved for sandbox judgment.
- Temperature = 0.0 and `num_predict = 512` keep output deterministic and bounded.
