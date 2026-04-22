from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_loader import ensure_dataset  # noqa: E402
from mcp_client import MCPClient  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402
from sandbox_runner import extract_code, run_sandbox  # noqa: E402


def load_config(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_result_path(results_dir: Path, mode: str, task_id: int) -> Path:
    return results_dir / "raw" / mode / f"{task_id}.json"


async def run_one(
    task: dict[str, Any],
    mode: str,
    mode_cfg: dict[str, Any],
    client: httpx.AsyncClient,
    ollama: OllamaClient,
    sandbox_cfg: dict[str, Any],
    results_dir: Path,
    logger: logging.Logger,
    sem: asyncio.Semaphore,
    mcp: MCPClient | None,
) -> dict[str, Any]:
    out_path = task_result_path(results_dir, mode, task["task_id"])
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    citations: list[dict[str, Any]] = []
    questions: list[str] = []
    mcp_meta: dict[str, Any] = {}

    async with sem:
        if mode_cfg.get("use_mcp") and mcp is not None:
            t_mcp = time.perf_counter()
            ctx = await asyncio.get_running_loop().run_in_executor(
                None, lambda: mcp.gather_context(task["text"])
            )
            logger.info("task=%s MCP done in %.1fs (cite=%d, q=%d, err=%s)",
                        task["task_id"], time.perf_counter() - t_mcp,
                        len(ctx.citations), len(ctx.questions), ctx.error)
            citations = ctx.citations
            questions = ctx.questions
            mcp_meta = {
                "session_id": ctx.session_id,
                "budget_remaining": ctx.kb_budget_remaining,
                "n_citations": len(citations),
                "n_questions": len(questions),
                "citation_skills": [c.get("skill_name") for c in citations[:5]],
                "error": ctx.error,
            }

        prompt = build_prompt(
            mode,
            task["text"],
            task["test_list"],
            citations=citations,
            questions=questions,
            snippet_chars=int(mode_cfg.get("snippet_chars", 400)),
        )
        t0 = time.perf_counter()
        try:
            resp = await ollama.generate(client, prompt)
            text = resp.text
            eval_count = resp.eval_count
            done_reason = resp.done_reason
            gen_err = None
        except Exception as e:
            text = ""
            eval_count = 0
            done_reason = "error"
            gen_err = str(e)
        latency = time.perf_counter() - t0

    extraction = extract_code(text)
    if extraction.reason != "ok":
        passed = False
        reason = extraction.reason
        sb_stdout = ""
        sb_stderr = ""
    else:
        sb = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_sandbox(
                extraction.code,
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

    record = {
        "task_id": task["task_id"],
        "mode": mode,
        "prompt": prompt,
        "response": text,
        "extracted_code": extraction.code,
        "passed": passed,
        "reason": reason,
        "sandbox_stdout": sb_stdout,
        "sandbox_stderr": sb_stderr,
        "latency_seconds": round(latency, 3),
        "eval_count": eval_count,
        "done_reason": done_reason,
        "gen_error": gen_err,
        "mcp": mcp_meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info("task=%s mode=%s pass=%s reason=%s latency=%.2fs", task["task_id"], mode, passed, reason, latency)
    return record


async def run_mode(
    tasks: list[dict[str, Any]],
    mode: str,
    mode_cfg: dict[str, Any],
    ollama: OllamaClient,
    sandbox_cfg: dict[str, Any],
    results_dir: Path,
    concurrency: int,
    logger: logging.Logger,
    mcp: MCPClient | None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(ollama.timeout_seconds + 10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        coros = [
            run_one(t, mode, mode_cfg, client, ollama, sandbox_cfg, results_dir, logger, sem, mcp)
            for t in tasks
        ]
        results: list[dict[str, Any]] = []
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(coros):
                logger.info("mode=%s progress %d/%d", mode, done, len(coros))
    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    passed = sum(1 for r in records if r["passed"])
    reasons: dict[str, int] = {}
    latencies: list[float] = []
    eval_counts: list[int] = []
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        latencies.append(float(r["latency_seconds"]))
        eval_counts.append(int(r["eval_count"]))
    return {
        "n": n,
        "passed": passed,
        "pass_at_1": round(passed / n, 4) if n else 0.0,
        "reasons": reasons,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "mean_eval_count": round(sum(eval_counts) / len(eval_counts), 1) if eval_counts else 0.0,
    }


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
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
    ap.add_argument("--limit", type=int, default=None, help="debug: limit number of tasks")
    ap.add_argument("--modes", nargs="+", default=None, help="override modes list (e.g. no_test with_test)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    results_dir = ROOT / paths["results_dir"]
    dataset_path = ROOT / paths["dataset_file"]
    logs_dir = ROOT / paths["logs_dir"]
    logger = setup_logger(logs_dir)

    logger.info("loading dataset %s", dataset_path)
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
        num_predict=model_cfg.get("num_predict", 512),
        timeout_seconds=model_cfg.get("timeout_seconds", 60.0),
        max_retries=model_cfg.get("max_retries", 3),
        retry_backoff_seconds=model_cfg.get("retry_backoff_seconds", 2.0),
    )

    mode_names = args.modes or [m["name"] for m in cfg["modes"]]
    mode_cfgs = {m["name"]: m for m in cfg["modes"]}
    concurrency = cfg["runner"].get("concurrency", 4)

    # Build MCP client once if any mode needs it
    mcp: MCPClient | None = None
    if any(mode_cfgs[m].get("use_mcp") for m in mode_names if m in mode_cfgs):
        corpus_rel = cfg.get("mcp", {}).get("corpus_path", "../CODEFORGE/codeforge/kb/skills_corpus.jsonl")
        corpus_path = (ROOT / corpus_rel).resolve()
        task_level = cfg.get("mcp", {}).get("task_level", "hard")
        top_k = int(cfg.get("mcp", {}).get("top_k", 3))
        logger.info("initializing CodeForge MCP client (corpus=%s, task_level=%s, top_k=%d)",
                    corpus_path, task_level, top_k)
        mcp = MCPClient(corpus_path, task_level=task_level, top_k=top_k)
        mcp._ensure_server()  # noqa: SLF001  — fail fast if corpus missing
        logger.info("MCP client ready")

    metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode in mode_names:
        mode_cfg = mode_cfgs.get(mode, {"name": mode, "use_mcp": False})
        logger.info("=== running mode=%s on %d tasks (concurrency=%d, mcp=%s) ===",
                    mode, len(tasks), concurrency, bool(mode_cfg.get("use_mcp")))
        records = asyncio.run(
            run_mode(tasks, mode, mode_cfg, ollama, cfg["sandbox"], results_dir, concurrency, logger, mcp)
        )
        summary = summarize(records)
        summary["mode"] = mode
        summary["model"] = model_cfg["name"]
        summary["endpoint"] = model_cfg["endpoint"]
        summary["dataset_sha256"] = digest
        metrics_by_mode[mode] = summary
        out = results_dir / f"metrics_{mode}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("mode=%s pass@1=%.4f (%d/%d)", mode, summary["pass_at_1"], summary["passed"], summary["n"])

    logger.info("=== done ===")
    for m, s in metrics_by_mode.items():
        logger.info("%s: pass@1=%.4f n=%d", m, s["pass_at_1"], s["n"])


if __name__ == "__main__":
    main()
