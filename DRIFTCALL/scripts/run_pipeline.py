"""DriftCall pipeline orchestrator (DESIGN.md §10, training.md §2, evaluation.md §6.1).

Single CLI entrypoint that wires the cell-level ``train`` / ``eval_*`` /
``probe`` / ``render_plots`` / ``print_summary_table`` / ``push_lora_to_hub``
callables. ``scripts/train_full.sh`` invokes this script per stage; the cells
themselves do not ship ``__main__`` blocks (they are jupytext-built notebook
cells), so this script is the production glue.

Subcommands match the contract in ``scripts/train_full.sh`` exactly:

    stage1, stage2, stage3   GRPO curriculum stages (cells 15 / 16 / 17)
    eval-baseline            frozen-greedy eval against untrained Gemma 4 E2B
    eval-final               paired eval against the trained LoRA
    probe                    reward-hacking probe (200 held-out episodes)
    plots                    render the 4 evaluation plot panels
    summary                  emit the markdown before/after table
    deploy                   push the trained LoRA + eval reports to HF Hub

Heavy imports (cells, torch, unsloth, peft, trl, wandb, huggingface_hub) are
deferred inside subcommand handlers so ``--help`` works on CPU-only CI without
pulling the training stack.
"""

from __future__ import annotations

# Unsloth must import before transformers/peft so it can monkey-patch them.
# Set torch.compile/dynamo disable env vars first (Unsloth reads them at
# import time). Wrap in try/except so CPU-only CI / --help still works.
import os as _os_pre
_os_pre.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
_os_pre.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
_os_pre.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
_os_pre.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
try:
    import unsloth as _unsloth  # noqa: F401
except ImportError:
    pass

import argparse
import dataclasses
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence


__all__ = [
    "build_arg_parser",
    "build_briefs",
    "build_env_factory",
    "build_rollout_group_fn",
    "build_task_gen",
    "build_training_eval",
    "deserialize_eval_report",
    "main",
    "serialize_eval_report",
]


# ---------------------------------------------------------------------------
# Logging helpers — single output style so log-grep stays simple
# ---------------------------------------------------------------------------


def _info(subcmd: str, msg: str) -> None:
    print(f"[run_pipeline] starting {subcmd} {msg}", flush=True)


def _done(subcmd: str, output: Path | str) -> None:
    print(f"[run_pipeline] DONE {subcmd} output={output}", flush=True)


# ---------------------------------------------------------------------------
# Factories — real callables built from the cells
# ---------------------------------------------------------------------------


def build_task_gen() -> Callable[..., Any]:
    """Return ``cells.step_07_task_generator.generate`` (training.md §2.2)."""
    from cells.step_07_task_generator import generate

    return generate


def build_env_factory(
    *,
    curriculum_stage: int,
    language_weights: dict[str, float],
) -> Callable[[], Any]:
    """Return a zero-arg factory that builds a fresh ``DriftCallEnv``.

    Stage and language mix are bound at construction time so each call
    returns an env with identical config (training.md §3.2).
    """
    from cells.step_10_env import DriftCallEnv

    config: dict[str, Any] = {
        "curriculum_stage": curriculum_stage,
        "language_weights": dict(language_weights),
        "audio_boundary_enabled": False,
        "max_turns_override": None,
    }

    def factory() -> Any:
        return DriftCallEnv(config=dict(config))

    return factory


def build_rollout_group_fn() -> Callable[..., Any]:
    """Return the multi-turn rollout helper (training.md §2.2, §3.2).

    Matches :class:`cells.step_14_custom_trainer.RolloutGroupFn`. The actual
    rollout loop lives in ``cells.step_14_custom_trainer.rollout_group`` so
    the orchestrator stays glue-only.
    """
    from cells.step_14_custom_trainer import rollout_group

    return rollout_group


def build_training_eval() -> Callable[..., Any]:
    """Return the production ``TrainingEvalCallable`` delegate (evaluation.md §6.1).

    The delegate lives under ``scripts._training_eval`` so the heavy model
    loading + rollout path is decoupled from the CLI plumbing.
    """
    from scripts._training_eval import training_eval

    return training_eval


# ---------------------------------------------------------------------------
# Briefs loader — published JSONL preferred, deterministic synthetic fallback
# ---------------------------------------------------------------------------


_PUBLICATION_VAL_BRIEFS: Path = Path("data/publication/val/briefs.jsonl")


def _load_published_briefs(path: Path) -> list[Any]:
    """Load ``data/publication/val/briefs.jsonl`` rows.

    Each line is a canonical JSON object (datasets.md §4.7). When the
    full ``BriefRow`` dataclass is not yet shipped, we deserialize each
    row into a :class:`types.SimpleNamespace` so eval/probe cells can
    read ``.episode_id``, ``.catalogue_hash``, etc.
    """
    from types import SimpleNamespace

    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(SimpleNamespace(**payload))
    return rows


def _build_synthetic_briefs(count: int) -> list[Any]:
    """Build a deterministic synthetic brief set via ``enumerate_variants``.

    Used when the published HF dataset bundle is not present locally so
    eval/probe still execute end-to-end.
    """
    from types import SimpleNamespace

    from cells.step_07_task_generator import enumerate_variants

    rows: list[Any] = []
    for idx, goal in enumerate(enumerate_variants(limit=count, stage=3)):
        rows.append(
            SimpleNamespace(
                episode_id=f"s3_ep_{idx:08d}",
                seed=idx,
                stage=3,
                language=goal.language,
                domain=goal.domain,
                template_id="synthetic",
                goal=goal,
                catalogue_hash="synthetic",
                templates_sha256="synthetic",
                i18n_sha256="synthetic",
                generator_version="driftcall-synthetic",
                created_ts_ist="1970-01-01T00:00:00+05:30",
            ),
        )
    return rows


def build_briefs(min_rows: int) -> list[Any]:
    """Return at least ``min_rows`` briefs.

    Prefer the published bundle when it exists; otherwise build a
    deterministic synthetic set via the task generator.
    """
    if _PUBLICATION_VAL_BRIEFS.exists():
        rows = _load_published_briefs(_PUBLICATION_VAL_BRIEFS)
        if len(rows) >= min_rows:
            return rows
    return _build_synthetic_briefs(min_rows)


# ---------------------------------------------------------------------------
# EvalReport <-> JSON
# ---------------------------------------------------------------------------


def _encode_float(value: float) -> Any:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return value


def _decode_float(value: Any) -> float:
    if isinstance(value, str):
        if value == "NaN":
            return float("nan")
        if value == "Inf":
            return float("inf")
        if value == "-Inf":
            return float("-inf")
    return float(value)


def serialize_eval_report(report: Any) -> dict[str, Any]:
    """Convert a frozen ``EvalReport`` (or any nested dataclass) to a JSON-safe dict.

    NaN / Inf floats are encoded as the strings ``"NaN"`` / ``"Inf"`` /
    ``"-Inf"`` so :func:`deserialize_eval_report` can round-trip them.
    """

    def _coerce(value: Any) -> Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {k: _coerce(v) for k, v in dataclasses.asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_coerce(v) for v in value]
        if isinstance(value, float):
            return _encode_float(value)
        return value

    coerced = _coerce(report)
    if not isinstance(coerced, dict):
        raise TypeError(
            f"serialize_eval_report expected a dataclass; got {type(report).__name__}",
        )
    return coerced


def deserialize_eval_report(payload: dict[str, Any]) -> Any:
    """Rebuild :class:`cells.step_18_eval_baseline.EvalReport` from a JSON dict."""
    from cells.step_18_eval_baseline import (
        DriftDetectionLatency,
        EvalReport,
        PerLanguageReport,
    )

    def _ci_triple(raw: Any) -> tuple[float, float, float]:
        a, b, c = raw
        return _decode_float(a), _decode_float(b), _decode_float(c)

    latency_raw = payload["drift_detection_latency"]
    latency = DriftDetectionLatency(
        stage2_mean=_decode_float(latency_raw["stage2_mean"]),
        stage2_median=_decode_float(latency_raw["stage2_median"]),
        stage2_p95=_decode_float(latency_raw["stage2_p95"]),
        stage3_mean=_decode_float(latency_raw["stage3_mean"]),
        stage3_median=_decode_float(latency_raw["stage3_median"]),
        stage3_p95=_decode_float(latency_raw["stage3_p95"]),
        undetected_count=int(latency_raw["undetected_count"]),
    )
    per_language = tuple(
        PerLanguageReport(
            language=row["language"],
            n_episodes=int(row["n_episodes"]),
            reward_mean=_decode_float(row["reward_mean"]),
            r1_mean=_decode_float(row["r1_mean"]),
            r2_mean=_decode_float(row["r2_mean"]),
            r3_mean=_decode_float(row["r3_mean"]),
            r4_mean=_decode_float(row["r4_mean"]),
            r5_mean=_decode_float(row["r5_mean"]),
        )
        for row in payload.get("per_language", ())
    )
    curves_raw = payload.get("curves", {}) or {}
    curves: dict[str, tuple[tuple[int, float], ...]] = {
        str(k): tuple((int(step), _decode_float(value)) for step, value in pairs)
        for k, pairs in curves_raw.items()
    }
    return EvalReport(
        model_path=str(payload["model_path"]),
        n_episodes=int(payload["n_episodes"]),
        reward_mean_ci=_ci_triple(payload["reward_mean_ci"]),
        r1_mean_ci=_ci_triple(payload["r1_mean_ci"]),
        r2_mean_ci=_ci_triple(payload["r2_mean_ci"]),
        r3_mean_ci=_ci_triple(payload["r3_mean_ci"]),
        r4_mean_ci=_ci_triple(payload["r4_mean_ci"]),
        r5_mean_ci=_ci_triple(payload["r5_mean_ci"]),
        brier_mean=_decode_float(payload["brier_mean"]),
        floor_applied_rate=_decode_float(payload["floor_applied_rate"]),
        hallucinated_field_rate=_decode_float(payload["hallucinated_field_rate"]),
        reward_hacking_offenses={
            str(k): int(v) for k, v in payload.get("reward_hacking_offenses", {}).items()
        },
        drift_detection_latency=latency,
        per_language=per_language,
        curves=curves,
        breakdown=dict(payload.get("breakdown", {})),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        result: Any = json.load(fh)
    if not isinstance(result, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(result).__name__}")
    return result


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _run_stage(
    *,
    subcmd: str,
    train_module: str,
    num_steps: int,
    hardware: str,
    output_dir: Path,
    resume_from: Path | None,
) -> Path:
    """Shared dispatcher for ``stage1`` / ``stage2`` / ``stage3``."""
    _info(
        subcmd,
        (
            f"num_steps={num_steps} hardware={hardware} "
            f"output_dir={output_dir} resume_from={resume_from}"
        ),
    )

    import importlib

    module = importlib.import_module(train_module)
    train: Callable[..., Path] = module.train
    stage_int = int(module.STAGE)
    language_weights = dict(module.LANGUAGE_WEIGHTS)

    task_gen = build_task_gen()
    env_factory = build_env_factory(
        curriculum_stage=stage_int,
        language_weights=language_weights,
    )
    rollout_group_fn = build_rollout_group_fn()

    final_dir = output_dir / "final"
    kwargs: dict[str, Any] = {
        "num_steps": num_steps,
        "output_dir": final_dir,
        "task_gen": task_gen,
        "env_factory": env_factory,
        "rollout_group_fn": rollout_group_fn,
    }
    if resume_from is not None:
        kwargs["resume_from"] = resume_from

    final_path = Path(train(**kwargs))
    _done(subcmd, final_path)
    return final_path


def stage1(args: argparse.Namespace) -> int:
    _run_stage(
        subcmd="stage1",
        train_module="cells.step_15_train_stage1",
        num_steps=args.num_steps,
        hardware=args.hardware,
        output_dir=Path(args.output_dir),
        resume_from=None,
    )
    return 0


def stage2(args: argparse.Namespace) -> int:
    _run_stage(
        subcmd="stage2",
        train_module="cells.step_16_train_stage2",
        num_steps=args.num_steps,
        hardware=args.hardware,
        output_dir=Path(args.output_dir),
        resume_from=Path(args.resume_from),
    )
    return 0


def stage3(args: argparse.Namespace) -> int:
    _run_stage(
        subcmd="stage3",
        train_module="cells.step_17_train_stage3",
        num_steps=args.num_steps,
        hardware=args.hardware,
        output_dir=Path(args.output_dir),
        resume_from=Path(args.resume_from),
    )
    return 0


def eval_baseline_cmd(args: argparse.Namespace) -> int:
    _info("eval-baseline", f"episodes={args.episodes} output={args.output}")
    from cells.step_18_eval_baseline import eval_baseline as _eval_baseline

    briefs = build_briefs(min_rows=max(int(args.episodes), 50))
    training_eval = build_training_eval()
    report = _eval_baseline(
        "base",
        args.episodes,
        training_eval=training_eval,
        briefs=briefs,
    )
    output = Path(args.output)
    _write_json(output, serialize_eval_report(report))
    _done("eval-baseline", output)
    return 0


def eval_final_cmd(args: argparse.Namespace) -> int:
    _info(
        "eval-final",
        (
            f"episodes={args.episodes} checkpoint={args.checkpoint} "
            f"output={args.output}"
        ),
    )
    from cells.step_19_eval_final import eval_final as _eval_final

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    baseline_path = output.with_name("baseline.json")
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"eval-final requires {baseline_path} (run eval-baseline first)",
        )

    briefs = build_briefs(min_rows=max(int(args.episodes), 50))
    training_eval = build_training_eval()
    baseline = deserialize_eval_report(_read_json(baseline_path))

    report = _eval_final(
        checkpoint,
        args.episodes,
        baseline=baseline,
        training_eval=training_eval,
        briefs=briefs,
    )
    _write_json(output, serialize_eval_report(report))
    _done("eval-final", output)
    return 0


def probe_cmd(args: argparse.Namespace) -> int:
    _info(
        "probe",
        (
            f"episodes={args.episodes} checkpoint={args.checkpoint} "
            f"output={args.output}"
        ),
    )
    from cells.step_20_probe import probe_reward_hacking, render_probe_report_md

    checkpoint = Path(args.checkpoint)
    briefs = build_briefs(min_rows=50 + int(args.episodes))
    training_eval = build_training_eval()
    report = probe_reward_hacking(
        checkpoint,
        args.episodes,
        training_eval=training_eval,
        briefs=briefs,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_path": report.model_path,
        "n_episodes": report.n_episodes,
        "git_sha": report.git_sha,
        "timestamp_ist": report.timestamp_ist,
        "total_hits": report.total_hits,
        "novel_classes": list(report.novel_classes),
        "per_class": [dataclasses.asdict(row) for row in report.per_class],
    }
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
    md_path = output.with_suffix(".md")
    render_probe_report_md(report, md_path)
    _done("probe", output)
    return 0


def plots_cmd(args: argparse.Namespace) -> int:
    _info(
        "plots",
        (
            f"baseline={args.baseline} final={args.final} "
            f"output_dir={args.output_dir}"
        ),
    )
    from cells.step_21_plots import render_plots

    baseline = deserialize_eval_report(_read_json(Path(args.baseline)))
    final = deserialize_eval_report(_read_json(Path(args.final)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    paths = render_plots(baseline, final, wandb_run_id, out_dir)
    _info("plots", f"rendered={sorted(paths)}")
    _done("plots", out_dir)
    return 0


def summary_cmd(args: argparse.Namespace) -> int:
    _info(
        "summary",
        (
            f"baseline={args.baseline} final={args.final} probe={args.probe} "
            f"output={args.output}"
        ),
    )
    from cells.step_22_summary import print_summary_table

    baseline = deserialize_eval_report(_read_json(Path(args.baseline)))
    final = deserialize_eval_report(_read_json(Path(args.final)))
    md = print_summary_table(baseline, final)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md + "\n", encoding="utf-8")
    _done("summary", output)
    return 0


def deploy_cmd(args: argparse.Namespace) -> int:
    _info(
        "deploy",
        (
            f"checkpoint={args.checkpoint} eval_reports={args.eval_reports} "
            f"repo={args.repo}"
        ),
    )
    from cells.step_24_deploy_hf import push_lora_to_hub

    checkpoint = Path(args.checkpoint)
    eval_reports_dir = Path(args.eval_reports)
    if not eval_reports_dir.exists():
        raise FileNotFoundError(f"eval_reports dir not found: {eval_reports_dir}")

    token = os.environ.get("HF_TOKEN")
    result = push_lora_to_hub(
        checkpoint_path=checkpoint,
        repo_id=args.repo,
        token=token,
    )
    if not result.success:
        print(
            (
                f"[run_pipeline] deploy push_lora_to_hub failed "
                f"(rc={result.return_code}): {result.stderr}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return result.return_code or 1
    _done("deploy", args.repo)
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _add_stage_args(
    parser: argparse.ArgumentParser,
    *,
    default_steps: int,
    requires_resume: bool,
) -> None:
    parser.add_argument("--num-steps", type=int, default=default_steps)
    parser.add_argument("--hardware", choices=("v100", "h100"), default="h100")
    parser.add_argument("--output-dir", required=True, type=str)
    if requires_resume:
        parser.add_argument("--resume-from", required=True, type=str)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="DriftCall training/eval/deploy pipeline orchestrator.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p1 = sub.add_parser("stage1", help="Stage-1 GRPO (warmup, no drift)")
    _add_stage_args(p1, default_steps=150, requires_resume=False)

    p2 = sub.add_parser("stage2", help="Stage-2 GRPO (single drift)")
    _add_stage_args(p2, default_steps=200, requires_resume=True)

    p3 = sub.add_parser("stage3", help="Stage-3 GRPO (compound drift)")
    _add_stage_args(p3, default_steps=150, requires_resume=True)

    pb = sub.add_parser(
        "eval-baseline", help="Frozen-greedy eval against untrained Gemma 4 E2B"
    )
    pb.add_argument("--episodes", type=int, default=50)
    pb.add_argument("--output", required=True, type=str)

    pf = sub.add_parser("eval-final", help="Paired eval against the trained LoRA")
    pf.add_argument("--episodes", type=int, default=50)
    pf.add_argument("--checkpoint", required=True, type=str)
    pf.add_argument("--output", required=True, type=str)

    pr = sub.add_parser("probe", help="Reward-hacking probe (200 held-out episodes)")
    pr.add_argument("--episodes", type=int, default=200)
    pr.add_argument("--checkpoint", required=True, type=str)
    pr.add_argument("--output", required=True, type=str)

    pp = sub.add_parser("plots", help="Render the 4 evaluation plot panels")
    pp.add_argument("--baseline", required=True, type=str)
    pp.add_argument("--final", required=True, type=str)
    pp.add_argument("--output-dir", required=True, type=str)

    ps = sub.add_parser("summary", help="Markdown before/after summary")
    ps.add_argument("--baseline", required=True, type=str)
    ps.add_argument("--final", required=True, type=str)
    ps.add_argument("--probe", required=True, type=str)
    ps.add_argument("--output", required=True, type=str)

    pd = sub.add_parser("deploy", help="Push trained LoRA + eval reports to HF Hub")
    pd.add_argument("--checkpoint", required=True, type=str)
    pd.add_argument("--eval-reports", required=True, type=str)
    pd.add_argument("--repo", required=True, type=str)

    return parser


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "stage1": stage1,
    "stage2": stage2,
    "stage3": stage3,
    "eval-baseline": eval_baseline_cmd,
    "eval-final": eval_final_cmd,
    "probe": probe_cmd,
    "plots": plots_cmd,
    "summary": summary_cmd,
    "deploy": deploy_cmd,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.subcommand]
    try:
        return handler(args)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — top-level CLI surface; print + exit
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
