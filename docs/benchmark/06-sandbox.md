# 06 — Sandbox

## Isolation strategy

Each candidate runs in a **fresh `subprocess.run`** with:
- CWD = temp directory (wiped after run)
- `stdin` empty
- `stdout` + `stderr` captured
- `timeout` = `sandbox.wall_timeout_seconds`
- On POSIX, CPU rlimit via `resource.setrlimit(RLIMIT_CPU, …)`
- On Windows, CPU rlimit not available → rely on wall timeout

The parent harness never `exec`s generated code. All execution is in an out-of-process interpreter.

## Script composition

```python
# test_setup_code (if present)
{generated_code}

# asserts
{test_list[0]}
{test_list[1]}
{test_list[2]}
```

Written to a temp `.py` file. Executed as `python <file>`.

## Pass/fail rule

- exit 0, stdout+stderr contain no uncaught exception → **pass**
- non-zero exit, timeout, or any exception → **fail** (with reason)

## Timeout configuration

- `cpu_timeout_seconds: 10` — POSIX CPU-bound bound
- `wall_timeout_seconds: 15` — hard kill for runaway IO/threads

## Security posture

MBPP is a trusted dataset and the model is an on-prem Gemma. The generated code may still be hostile in edge cases, so:

- Sandbox CWD is a throwaway temp dir
- No network calls blocked by default — the harness relies on the host OS. Users running untrusted models should wrap the sandbox in Docker / firejail / nsjail for real isolation.
- Limit: `cpu_timeout`, `wall_timeout`. No memory rlimit in v1 (can be added via `RLIMIT_AS`).

## Recorded failure reasons

| Reason | Detection |
|---|---|
| `no_code_block` | extraction step returned empty |
| `syntax_error` | `ast.parse` raises before sandbox |
| `timeout` | `subprocess.TimeoutExpired` |
| `assertion_error` | stderr contains `AssertionError` |
| `name_error` | stderr contains `NameError` |
| `import_error` | stderr contains `ImportError` |
| `runtime_error` | other non-zero exit |
| `pass` | exit 0 |
