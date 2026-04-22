from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_loader import load_jsonl  # noqa: E402


def load_raw(results_dir: Path, mode: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    d = results_dir / "raw" / mode
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            r = json.load(f)
        out[int(r["task_id"])] = r
    return out


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
        "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "mean_eval_count": round(sum(eval_counts) / max(len(eval_counts), 1), 1),
    }


def mcnemar(b: int, c: int) -> dict[str, float]:
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # chi-square p-value for df=1 (continuity-corrected McNemar)
    # survival function of chi2 df=1 = erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"chi2": round(chi2, 4), "p_value": round(p, 6)}


def difficulty_bucket(ref_code: str) -> str:
    lines = len([ln for ln in (ref_code or "").splitlines() if ln.strip()])
    if lines <= 5:
        return "easy"
    if lines <= 10:
        return "medium"
    return "hard"


def main() -> None:
    results_dir = ROOT / "results"
    dataset_path = ROOT / "dataset" / "mbpp.jsonl"
    tasks = {int(t["task_id"]): t for t in load_jsonl(dataset_path)} if dataset_path.exists() else {}

    no = load_raw(results_dir, "without_mcp")
    wt = load_raw(results_dir, "with_mcp")

    both_ids = sorted(set(no.keys()) & set(wt.keys()))
    records_no = [no[i] for i in both_ids]
    records_wt = [wt[i] for i in both_ids]

    s_no = summarize(records_no)
    s_wt = summarize(records_wt)

    only_no = sum(1 for i in both_ids if no[i]["passed"] and not wt[i]["passed"])
    only_wt = sum(1 for i in both_ids if not no[i]["passed"] and wt[i]["passed"])
    both_pass = sum(1 for i in both_ids if no[i]["passed"] and wt[i]["passed"])
    both_fail = sum(1 for i in both_ids if not no[i]["passed"] and not wt[i]["passed"])

    mc = mcnemar(only_no, only_wt)

    # difficulty buckets
    buckets_no: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    buckets_wt: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    if tasks:
        for i in both_ids:
            b = difficulty_bucket(tasks.get(i, {}).get("code", ""))
            buckets_no[b].append(1 if no[i]["passed"] else 0)
            buckets_wt[b].append(1 if wt[i]["passed"] else 0)

    def _pass_rate(xs: list[int]) -> str:
        return f"{sum(xs) / len(xs):.4f} ({sum(xs)}/{len(xs)})" if xs else "n/a"

    # write comparison csv
    csv_path = results_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "without_mcp_pass", "with_mcp_pass", "without_mcp_reason", "with_mcp_reason"])
        for i in both_ids:
            w.writerow([
                i,
                int(no[i]["passed"]),
                int(wt[i]["passed"]),
                no[i]["reason"],
                wt[i]["reason"],
            ])

    # overwrite metrics files
    (results_dir / "metrics_without_mcp.json").write_text(json.dumps(s_no, indent=2), encoding="utf-8")
    (results_dir / "metrics_with_mcp.json").write_text(json.dumps(s_wt, indent=2), encoding="utf-8")

    lift_pp = round((s_wt["pass_at_1"] - s_no["pass_at_1"]) * 100, 2)

    # pull model name from config
    model_name = "unknown"
    try:
        import yaml

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        model_name = cfg.get("model", {}).get("name", "unknown")
    except Exception:
        pass

    md: list[str] = []
    md.append(f"# Benchmark Report — MBPP on {model_name}\n")
    md.append(f"- Samples evaluated: **{len(both_ids)}**\n")
    md.append(f"- Model: {model_name}\n")
    md.append("- Endpoint: Ollama (http://172.22.2.151:7021)\n")
    md.append("- Dataset: MBPP full (all splits concatenated)\n")
    md.append("\n## Headline\n")
    md.append("| Mode | pass@1 | Passed / N | Mean latency (s) | Mean eval_count |\n")
    md.append("|---|---|---|---|---|\n")
    md.append(f"| without_mcp   | {s_no['pass_at_1']:.4f} | {s_no['passed']}/{s_no['n']} | {s_no['mean_latency_s']} | {s_no['mean_eval_count']} |\n")
    md.append(f"| with_mcp | {s_wt['pass_at_1']:.4f} | {s_wt['passed']}/{s_wt['n']} | {s_wt['mean_latency_s']} | {s_wt['mean_eval_count']} |\n")
    md.append(f"\n**Lift** (with_mcp − without_mcp): **{lift_pp} pp**\n")
    md.append(f"\n**McNemar** (paired): chi² = {mc['chi2']}, p = {mc['p_value']}\n")

    md.append("\n## Flip matrix\n")
    md.append("|  | with_mcp pass | with_mcp fail |\n")
    md.append("|---|---|---|\n")
    md.append(f"| without_mcp pass | {both_pass} | {only_no} |\n")
    md.append(f"| without_mcp fail | {only_wt} | {both_fail} |\n")

    md.append("\n## Failure breakdown\n")
    md.append("| Reason | without_mcp | with_mcp |\n")
    md.append("|---|---|---|\n")
    all_reasons = sorted(set(list(s_no["reasons"].keys()) + list(s_wt["reasons"].keys())))
    for rr in all_reasons:
        md.append(f"| {rr} | {s_no['reasons'].get(rr, 0)} | {s_wt['reasons'].get(rr, 0)} |\n")

    if tasks:
        md.append("\n## Difficulty buckets (by reference-solution length)\n")
        md.append("| Bucket | without_mcp pass@1 | with_mcp pass@1 |\n")
        md.append("|---|---|---|\n")
        for b in ("easy", "medium", "hard"):
            md.append(f"| {b} | {_pass_rate(buckets_no[b])} | {_pass_rate(buckets_wt[b])} |\n")

    md.append("\n## Artifacts\n")
    md.append("- `results/metrics_without_mcp.json`\n- `results/metrics_with_mcp.json`\n- `results/comparison.csv`\n- `results/raw/without_mcp/*.json`\n- `results/raw/with_mcp/*.json`\n")

    (results_dir / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {results_dir / 'report.md'}")
    print(f"pass@1 without_mcp={s_no['pass_at_1']} with_mcp={s_wt['pass_at_1']} lift={lift_pp}pp")


if __name__ == "__main__":
    main()
