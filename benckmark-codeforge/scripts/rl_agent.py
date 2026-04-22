"""
RL agent harness — iterative solve with CodeForge MCP + sandbox reward signal.

Flow per MBPP task (mirrors demo_agent_with_mcp.py pattern):
    iter 0:
        1. reset CodeForge session (task_level=hard → budget 10)
        2. query_kb(claim=problem_text)
        3. interrogate() → Socratic questions
        4. build prompt with problem + sample_test + citations + questions
        5. qwen generates code
        6. sandbox runs hidden tests → quality, reason
        7. reward = quality * (1 - brier(conf, quality))

    iter 1..N if not passed:
        1. Optional: query_kb with new claim derived from failure type
        2. build refinement prompt with previous code + sandbox stderr
        3. qwen regenerates
        4. sandbox judges
        5. reward computed

Stops on: pass (quality=1.0) or max_iters reached.

Reward model (per SYSTEM_DESIGN §4.8.1):
    quality   = fraction of hidden tests passed
    confidence = provided by config or fixed 0.7
    brier     = min((confidence - quality)^2, 0.5)
    reward    = quality * (1 - brier)
    uncertain floor = 0.50 if confidence<0.3 AND quality<0.5 (not used here)
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_loader import ensure_dataset  # noqa: E402
from mcp_client import MCPClient  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402
from sandbox_runner import extract_code, run_sandbox  # noqa: E402


# ------------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------------

_INSTR = (
    "You are an expert Python programmer. Solve the problem with MINIMAL code — "
    "no docstrings, no comments, no explanation."
)

_FMT = (
    "Output format: a single fenced Python code block. Start with ```python and end "
    "with ```. Nothing else outside the block."
)


def _format_citations(citations: list[dict[str, Any]], chars: int = 350) -> str:
    if not citations:
        return ""
    blocks: list[str] = []
    for i, c in enumerate(citations, 1):
        name = c.get("skill_name", "?")
        body = str(c.get("section_body", "")).strip()
        if len(body) > chars:
            body = body[:chars].rstrip() + "..."
        blocks.append(f"[{i}] {name}\n{body}")
    return "Reference material:\n\n" + "\n\n".join(blocks)


def _format_questions(questions: list[str], n: int = 3) -> str:
    if not questions:
        return ""
    lines = [f"- {q}" for q in questions[:n]]
    return "Things to consider:\n" + "\n".join(lines)


def build_initial_prompt(
    problem: str,
    sample_test: str,
    citations: list[dict[str, Any]],
    questions: list[str],
) -> str:
    parts: list[str] = [_INSTR, ""]
    cb = _format_citations(citations)
    if cb:
        parts.extend([cb, ""])
    qb = _format_questions(questions)
    if qb:
        parts.extend([qb, ""])
    parts.extend(["Problem:", problem.strip(), ""])
    if sample_test:
        parts.extend(["Your solution must satisfy this test:", sample_test, ""])
    parts.append(_FMT)
    return "\n".join(parts)


def build_refine_prompt(
    problem: str,
    sample_test: str,
    prev_code: str,
    sandbox_stderr: str,
    sandbox_stdout: str,
    citations: list[dict[str, Any]],
    iter_idx: int,
) -> str:
    # Trim traceback for prompt budget
    err_trim = (sandbox_stderr or "").strip()
    if len(err_trim) > 800:
        err_trim = err_trim[-800:]
    out_trim = (sandbox_stdout or "").strip()
    if len(out_trim) > 300:
        out_trim = out_trim[-300:]

    parts: list[str] = [_INSTR, ""]
    parts.append(
        f"Your previous attempt (iteration {iter_idx}) FAILED the hidden tests. "
        "Fix the bug. Do not repeat the same mistake."
    )
    parts.append("")
    cb = _format_citations(citations, chars=250)
    if cb:
        parts.extend([cb, ""])
    parts.extend(["Problem:", problem.strip(), ""])
    if sample_test:
        parts.extend(["Expected to satisfy:", sample_test, ""])
    parts.extend(["Previous code:", "```python", prev_code.strip(), "```", ""])
    if err_trim:
        parts.extend(["Sandbox error:", err_trim, ""])
    if out_trim:
        parts.extend(["Sandbox stdout:", out_trim, ""])
    parts.append("Analyze the error, identify the root cause, then rewrite.")
    parts.append(_FMT)
    return "\n".join(parts)


# ------------------------------------------------------------------------
# Quality + reward
# ------------------------------------------------------------------------

_ASSERT = re.compile(r"^\s*assert\s", re.MULTILINE)


def count_passed_asserts(
    code: str,
    test_list: list[str],
    setup_code: str = "",
    *,
    python_exec: str = sys.executable,
    wall_timeout_seconds: float = 15.0,
) -> tuple[int, int]:
    """Run each assert independently, count how many pass.

    Returns (passed_count, total_count).
    """
    if not code.strip():
        return 0, len(test_list)
    try:
        ast.parse(code)
    except SyntaxError:
        return 0, len(test_list)

    passed = 0
    total = len(test_list)
    for t in test_list:
        script = ""
        if setup_code.strip():
            script += setup_code + "\n\n"
        script += code + "\n\n" + t + "\n"
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "cand.py"
                p.write_text(script, encoding="utf-8")
                cp = subprocess.run(
                    [python_exec, str(p)],
                    capture_output=True,
                    text=True,
                    timeout=wall_timeout_seconds,
                    cwd=td,
                )
                if cp.returncode == 0:
                    passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return passed, total


def compute_reward(quality: float, confidence: float = 0.7) -> dict[str, float]:
    """Per SYSTEM_DESIGN §4.8.1: reward = quality * (1 - brier)."""
    brier = min((confidence - quality) ** 2, 0.5)
    reward = quality * (1.0 - brier)
    return {
        "quality": round(quality, 4),
        "confidence": round(confidence, 4),
        "brier": round(brier, 4),
        "reward": round(max(0.0, min(1.0, reward)), 4),
    }


# ------------------------------------------------------------------------
# Per-iteration state
# ------------------------------------------------------------------------


@dataclass
class IterRecord:
    iter_idx: int
    prompt: str
    response: str
    extracted_code: str
    extract_reason: str
    passed: bool
    sandbox_reason: str
    sandbox_stdout: str
    sandbox_stderr: str
    asserts_passed: int
    asserts_total: int
    quality: float
    brier: float
    reward: float
    latency_seconds: float
    eval_count: int
    done_reason: str
    kb_skills: list[str] = field(default_factory=list)
    questions_count: int = 0


# ------------------------------------------------------------------------
# Per-task RL loop
# ------------------------------------------------------------------------


async def rl_solve(
    task: dict[str, Any],
    ollama: OllamaClient,
    http_client: httpx.AsyncClient,
    mcp: MCPClient,
    sandbox_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    sem: asyncio.Semaphore,
    logger: logging.Logger,
    results_dir: Path,
) -> dict[str, Any]:
    tid = task["task_id"]
    out_path = results_dir / "raw" / "rl" / f"{tid}.json"
    if out_path.exists() and agent_cfg.get("resume", True):
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    max_iters = int(agent_cfg.get("max_iters", 5))
    confidence = float(agent_cfg.get("confidence", 0.7))
    requery_every_iter = bool(agent_cfg.get("requery_kb_on_refine", True))

    iterations: list[IterRecord] = []
    final_passed = False
    first_pass_iter: int | None = None
    mcp_total_latency = 0.0
    initial_citations: list[dict[str, Any]] = []
    initial_questions: list[str] = []

    async with sem:
        # ---- initial MCP gather
        t_mcp = time.perf_counter()
        ctx = await asyncio.get_running_loop().run_in_executor(
            None, lambda: mcp.gather_context(task["text"])
        )
        mcp_total_latency += time.perf_counter() - t_mcp
        initial_citations = ctx.citations
        initial_questions = ctx.questions

        prev_code = ""
        prev_stderr = ""
        prev_stdout = ""

        for it in range(max_iters):
            # build prompt
            if it == 0:
                citations = initial_citations
                questions = initial_questions
                prompt = build_initial_prompt(
                    task["text"], task["test_list"][0] if task["test_list"] else "",
                    citations, questions,
                )
            else:
                # optional re-query KB with error hint
                citations = initial_citations
                if requery_every_iter:
                    t_re = time.perf_counter()
                    fail_claim = f"{task['text']} | failure: {prev_stderr[:200]}"
                    ctx2 = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: mcp.gather_context(fail_claim)
                    )
                    mcp_total_latency += time.perf_counter() - t_re
                    citations = ctx2.citations or initial_citations
                questions = []
                prompt = build_refine_prompt(
                    task["text"],
                    task["test_list"][0] if task["test_list"] else "",
                    prev_code,
                    prev_stderr,
                    prev_stdout,
                    citations,
                    iter_idx=it,
                )

            # LLM
            t0 = time.perf_counter()
            try:
                resp = await ollama.generate(http_client, prompt)
                text = resp.text
                eval_count = resp.eval_count
                done_reason = resp.done_reason
            except Exception as e:
                text = ""
                eval_count = 0
                done_reason = f"error:{type(e).__name__}"
            latency = time.perf_counter() - t0

            # extract + sandbox
            ext = extract_code(text)
            if ext.reason != "ok" or not ext.code.strip():
                passed = False
                reason = ext.reason or "no_code_block"
                sb_stdout = ""
                sb_stderr = ""
                asserts_ok = 0
                asserts_total = len(task["test_list"])
            else:
                sb = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: run_sandbox(
                        ext.code,
                        task["test_list"],
                        task.get("test_setup_code", ""),
                        python_exec=sandbox_cfg.get("python_exec", sys.executable),
                        wall_timeout_seconds=sandbox_cfg.get("wall_timeout_seconds", 15.0),
                    ),
                )
                passed = sb.passed
                reason = sb.reason
                sb_stdout = sb.stdout
                sb_stderr = sb.stderr
                # per-assert count (for partial quality reward)
                asserts_ok, asserts_total = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: count_passed_asserts(
                        ext.code,
                        task["test_list"],
                        task.get("test_setup_code", ""),
                        python_exec=sandbox_cfg.get("python_exec", sys.executable),
                        wall_timeout_seconds=sandbox_cfg.get("wall_timeout_seconds", 15.0),
                    ),
                )

            quality = asserts_ok / max(asserts_total, 1)
            rw = compute_reward(quality, confidence=confidence)

            iter_rec = IterRecord(
                iter_idx=it,
                prompt=prompt,
                response=text,
                extracted_code=ext.code,
                extract_reason=ext.reason,
                passed=passed,
                sandbox_reason=reason,
                sandbox_stdout=sb_stdout,
                sandbox_stderr=sb_stderr,
                asserts_passed=asserts_ok,
                asserts_total=asserts_total,
                quality=rw["quality"],
                brier=rw["brier"],
                reward=rw["reward"],
                latency_seconds=round(latency, 3),
                eval_count=eval_count,
                done_reason=done_reason,
                kb_skills=[c.get("skill_name", "?") for c in (citations or [])[:3]],
                questions_count=len(questions),
            )
            iterations.append(iter_rec)

            logger.info(
                "task=%s iter=%d pass=%s quality=%.2f reward=%.2f reason=%s lat=%.1fs",
                tid, it, passed, rw["quality"], rw["reward"], reason, latency,
            )

            if passed:
                final_passed = True
                first_pass_iter = it
                break

            prev_code = ext.code
            prev_stderr = sb_stderr
            prev_stdout = sb_stdout

    record = {
        "task_id": tid,
        "final_passed": final_passed,
        "first_pass_iter": first_pass_iter,
        "iters_used": len(iterations),
        "max_iters": max_iters,
        "best_quality": max((i.quality for i in iterations), default=0.0),
        "best_reward": max((i.reward for i in iterations), default=0.0),
        "mcp_total_latency_s": round(mcp_total_latency, 3),
        "iterations": [asdict(i) for i in iterations],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


# ------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------


async def run_all(
    tasks: list[dict[str, Any]],
    ollama: OllamaClient,
    mcp: MCPClient,
    sandbox_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    concurrency: int,
    logger: logging.Logger,
    results_dir: Path,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(ollama.timeout_seconds + 10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        coros = [
            rl_solve(t, ollama, client, mcp, sandbox_cfg, agent_cfg, sem, logger, results_dir)
            for t in tasks
        ]
        results: list[dict[str, Any]] = []
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            results.append(r)
            done += 1
            if done % 10 == 0 or done == len(coros):
                logger.info("RL progress %d/%d", done, len(coros))
        return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    first_pass = sum(1 for r in records if r.get("first_pass_iter") == 0)
    any_pass = sum(1 for r in records if r.get("final_passed"))
    iters_to_pass = [
        (r["first_pass_iter"] + 1) for r in records if r.get("first_pass_iter") is not None
    ]
    avg_iter_pass = round(sum(iters_to_pass) / len(iters_to_pass), 2) if iters_to_pass else 0.0
    mean_reward = round(
        sum(r.get("best_reward", 0.0) for r in records) / max(n, 1), 4
    )
    mean_quality = round(
        sum(r.get("best_quality", 0.0) for r in records) / max(n, 1), 4
    )
    # distribution of iters used
    dist: dict[str, int] = {}
    for r in records:
        k = str(r.get("iters_used", 0))
        dist[k] = dist.get(k, 0) + 1

    return {
        "n": n,
        "first_try_pass_at_1": round(first_pass / n, 4) if n else 0.0,
        "final_pass_at_k": round(any_pass / n, 4) if n else 0.0,
        "mean_iters_to_pass": avg_iter_pass,
        "mean_best_reward": mean_reward,
        "mean_best_quality": mean_quality,
        "iter_distribution": dist,
    }


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rl_agent")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(logs_dir / "rl_run.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = cfg["paths"]
    results_dir = ROOT / paths["results_dir"]
    dataset_path = ROOT / paths["dataset_file"]
    logs_dir = ROOT / paths["logs_dir"]
    logger = setup_logger(logs_dir)

    tasks, digest = ensure_dataset(
        dataset_path,
        hf_repo=cfg["dataset"]["hf_repo"],
        hf_config=cfg["dataset"]["hf_config"],
        split=cfg["dataset"]["split"],
    )
    logger.info("dataset: %d tasks, sha256=%s", len(tasks), digest)

    limit = args.limit or cfg["dataset"].get("limit")
    if limit:
        tasks = tasks[:limit]
        logger.info("limited to %d tasks", len(tasks))

    model_cfg = cfg["model"]
    ollama = OllamaClient(
        endpoint=model_cfg["endpoint"],
        model=model_cfg["name"],
        temperature=model_cfg.get("temperature", 0.0),
        num_predict=model_cfg.get("num_predict", 1024),
        timeout_seconds=model_cfg.get("timeout_seconds", 120.0),
        max_retries=model_cfg.get("max_retries", 3),
        retry_backoff_seconds=model_cfg.get("retry_backoff_seconds", 2.0),
    )

    mcp_cfg = cfg.get("mcp", {})
    corpus_path = (ROOT / mcp_cfg.get("corpus_path", "../CODEFORGE/codeforge/kb/skills_corpus.jsonl")).resolve()
    mcp = MCPClient(
        corpus_path,
        task_level=mcp_cfg.get("task_level", "hard"),
        top_k=int(mcp_cfg.get("top_k", 3)),
    )
    mcp._ensure_server()  # noqa: SLF001

    agent_cfg = cfg.get("rl_agent", {}) or {}
    if args.max_iters is not None:
        agent_cfg["max_iters"] = args.max_iters
    agent_cfg.setdefault("max_iters", 5)
    agent_cfg.setdefault("confidence", 0.7)
    agent_cfg.setdefault("requery_kb_on_refine", True)
    agent_cfg.setdefault("resume", True)

    concurrency = args.concurrency or cfg["runner"].get("concurrency", 2)
    logger.info(
        "=== RL run: %d tasks, max_iters=%d, concurrency=%d, model=%s ===",
        len(tasks), agent_cfg["max_iters"], concurrency, model_cfg["name"],
    )

    records = asyncio.run(
        run_all(tasks, ollama, mcp, cfg["sandbox"], agent_cfg, concurrency, logger, results_dir)
    )

    summary = summarize(records)
    summary["model"] = model_cfg["name"]
    summary["max_iters"] = agent_cfg["max_iters"]
    summary["dataset_sha256"] = digest

    out = results_dir / "metrics_rl.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("=== DONE === first@1=%.4f final@k=%.4f mean_iters=%.2f",
                summary["first_try_pass_at_1"],
                summary["final_pass_at_k"],
                summary["mean_iters_to_pass"])


if __name__ == "__main__":
    main()
