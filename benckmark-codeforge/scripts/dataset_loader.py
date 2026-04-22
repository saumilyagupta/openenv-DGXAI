from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def fetch_mbpp(hf_repo: str, hf_config: str, split: str) -> list[dict[str, Any]]:
    from datasets import concatenate_datasets, load_dataset

    if split == "all":
        parts = []
        for s in ("train", "test", "validation", "prompt"):
            try:
                parts.append(load_dataset(hf_repo, hf_config, split=s))
            except (ValueError, KeyError):
                pass
        ds = concatenate_datasets(parts)
    else:
        ds = load_dataset(hf_repo, hf_config, split=split)
    rows: list[dict[str, Any]] = []
    for r in ds:
        rows.append(
            {
                "task_id": int(r["task_id"]),
                "text": r["text"],
                "code": r["code"],
                "test_list": list(r["test_list"]),
                "test_setup_code": r.get("test_setup_code", "") or "",
                "challenge_test_list": list(r.get("challenge_test_list", []) or []),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_dataset(
    out: Path,
    hf_repo: str = "google-research-datasets/mbpp",
    hf_config: str = "full",
    split: str = "test",
) -> tuple[list[dict[str, Any]], str]:
    if out.exists():
        rows = load_jsonl(out)
        h = hashlib.sha256()
        for r in rows:
            h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return rows, h.hexdigest()
    rows = fetch_mbpp(hf_repo, hf_config, split)
    digest = write_jsonl(rows, out)
    return rows, digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/mbpp.jsonl")
    ap.add_argument("--hf-repo", default="google-research-datasets/mbpp")
    ap.add_argument("--hf-config", default="full")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows, digest = ensure_dataset(Path(args.out), args.hf_repo, args.hf_config, args.split)
    print(f"wrote {len(rows)} samples to {args.out}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
