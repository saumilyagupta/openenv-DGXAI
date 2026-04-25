"""Tests for ``scripts/run_pipeline.py``.

CI-safe: every heavy path (``boot_gemma``, ``DriftCallGRPOTrainer.train``,
``huggingface_hub`` upload) is patched. Each subcommand is exercised through
:func:`scripts.run_pipeline.main` so argparse + dispatch + handler all run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_pipeline
from scripts.run_pipeline import (
    build_arg_parser,
    build_briefs,
    build_env_factory,
    build_rollout_group_fn,
    build_task_gen,
    deserialize_eval_report,
    main,
    serialize_eval_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _eval_report(model_path: str = "base") -> Any:
    from cells.step_18_eval_baseline import (
        DriftDetectionLatency,
        EvalReport,
        PerLanguageReport,
    )

    return EvalReport(
        model_path=model_path,
        n_episodes=50,
        reward_mean_ci=(0.30, 0.20, 0.40),
        r1_mean_ci=(0.10, 0.05, 0.15),
        r2_mean_ci=(0.20, 0.15, 0.25),
        r3_mean_ci=(0.30, 0.25, 0.35),
        r4_mean_ci=(0.40, 0.35, 0.45),
        r5_mean_ci=(-0.10, -0.15, -0.05),
        brier_mean=0.40,
        floor_applied_rate=0.05,
        hallucinated_field_rate=0.10,
        reward_hacking_offenses={"hallucinated_field": 3},
        drift_detection_latency=DriftDetectionLatency(
            stage2_mean=float("nan"),
            stage2_median=float("nan"),
            stage2_p95=float("nan"),
            stage3_mean=1.5,
            stage3_median=1.0,
            stage3_p95=2.0,
            undetected_count=2,
        ),
        per_language=(
            PerLanguageReport(
                language="hi",
                n_episodes=10,
                reward_mean=0.30,
                r1_mean=0.10,
                r2_mean=0.20,
                r3_mean=0.30,
                r4_mean=0.40,
                r5_mean=-0.10,
            ),
        ),
        curves={"train/reward_mean": ((0, 0.10), (50, 0.20))},
        breakdown={"episode_ids": ("ep_0",)},
    )


def _write_baseline_and_final(tmp_path: Path) -> tuple[Path, Path]:
    base = _eval_report("base")
    final = _eval_report("/tmp/ckpt")
    base_path = tmp_path / "baseline.json"
    final_path = tmp_path / "final.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(serialize_eval_report(base)), encoding="utf-8")
    final_path.write_text(json.dumps(serialize_eval_report(final)), encoding="utf-8")
    return base_path, final_path


def _fake_briefs(n: int) -> list[Any]:
    return [
        SimpleNamespace(
            episode_id=f"ep_{i:04d}",
            seed=i,
            catalogue_hash="x",
            templates_sha256="y",
            i18n_sha256="z",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Argparse / dispatch
# ---------------------------------------------------------------------------


def test_dispatch_unknown_subcommand_exits_2() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["bogus"])
    assert excinfo.value.code == 2


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_build_arg_parser_exposes_all_subcommands() -> None:
    parser = build_arg_parser()
    sub_choices: set[str] = set()
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            sub_choices.update(action.choices.keys())
    expected = {
        "stage1",
        "stage2",
        "stage3",
        "eval-baseline",
        "eval-final",
        "probe",
        "plots",
        "summary",
        "deploy",
    }
    assert expected.issubset(sub_choices), f"missing: {expected - sub_choices}"


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------


def _patch_train_call(monkeypatch: pytest.MonkeyPatch, module_path: str) -> MagicMock:
    """Replace ``module_path.train`` with a captured mock returning the output dir."""
    import importlib

    module = importlib.import_module(module_path)
    train_mock = MagicMock(side_effect=lambda **kwargs: kwargs["output_dir"])
    monkeypatch.setattr(module, "train", train_mock)
    return train_mock


def test_stage1_calls_step_15_train_with_built_factories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_mock = _patch_train_call(monkeypatch, "cells.step_15_train_stage1")
    rc = main(
        [
            "stage1",
            "--num-steps",
            "3",
            "--hardware",
            "v100",
            "--output-dir",
            str(tmp_path / "stage1"),
        ]
    )
    assert rc == 0
    assert train_mock.called
    kwargs = train_mock.call_args.kwargs
    assert kwargs["num_steps"] == 3
    assert callable(kwargs["task_gen"])
    assert callable(kwargs["env_factory"])
    assert callable(kwargs["rollout_group_fn"])
    assert kwargs["output_dir"] == tmp_path / "stage1" / "final"
    assert "resume_from" not in kwargs


def test_stage2_requires_resume_from(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "stage2",
                "--num-steps",
                "5",
                "--output-dir",
                str(tmp_path / "stage2"),
            ]
        )
    assert excinfo.value.code == 2


def test_stage2_passes_resume_from_to_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_mock = _patch_train_call(monkeypatch, "cells.step_16_train_stage2")
    resume = tmp_path / "stage1" / "final"
    resume.mkdir(parents=True)
    rc = main(
        [
            "stage2",
            "--num-steps",
            "5",
            "--hardware",
            "h100",
            "--resume-from",
            str(resume),
            "--output-dir",
            str(tmp_path / "stage2"),
        ]
    )
    assert rc == 0
    assert train_mock.call_args.kwargs["resume_from"] == resume


def test_stage3_requires_resume_from(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "stage3",
                "--num-steps",
                "5",
                "--output-dir",
                str(tmp_path / "stage3"),
            ]
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Eval handlers
# ---------------------------------------------------------------------------


def test_eval_baseline_writes_json_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_report = _eval_report("base")
    monkeypatch.setattr(
        run_pipeline,
        "build_training_eval",
        lambda: MagicMock(return_value=fake_report),
    )
    monkeypatch.setattr(
        run_pipeline, "build_briefs", lambda min_rows: _fake_briefs(60)
    )

    import cells.step_18_eval_baseline as eb

    monkeypatch.setattr(eb, "eval_baseline", lambda *a, **k: fake_report)

    output = tmp_path / "baseline.json"
    rc = main(["eval-baseline", "--episodes", "50", "--output", str(output)])
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_path"] == "base"
    assert payload["n_episodes"] == 50


def test_eval_final_requires_checkpoint(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            ["eval-final", "--episodes", "50", "--output", str(tmp_path / "f.json")]
        )
    assert excinfo.value.code == 2


def test_eval_final_requires_baseline_alongside_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_pipeline, "build_training_eval", lambda: MagicMock())
    monkeypatch.setattr(
        run_pipeline, "build_briefs", lambda min_rows: _fake_briefs(60)
    )
    rc = main(
        [
            "eval-final",
            "--episodes",
            "50",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--output",
            str(tmp_path / "final.json"),
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# Probe handler
# ---------------------------------------------------------------------------


def test_probe_requires_checkpoint(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            ["probe", "--episodes", "200", "--output", str(tmp_path / "p.json")]
        )
    assert excinfo.value.code == 2


def test_probe_writes_json_and_md(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cells.step_20_probe import ProbeReport

    fake = ProbeReport(
        model_path=str(tmp_path / "ckpt"),
        n_episodes=200,
        git_sha="abc",
        timestamp_ist="1970-01-01T00:00:00+05:30",
        per_class=(),
        raw_hits=(),
        total_hits=0,
        novel_classes=(),
    )
    monkeypatch.setattr(run_pipeline, "build_training_eval", lambda: MagicMock())
    monkeypatch.setattr(
        run_pipeline, "build_briefs", lambda min_rows: _fake_briefs(min_rows)
    )

    import cells.step_20_probe as probe_mod

    monkeypatch.setattr(probe_mod, "probe_reward_hacking", lambda *a, **k: fake)
    monkeypatch.setattr(
        probe_mod,
        "render_probe_report_md",
        lambda report, path: Path(path).write_text("# probe\n"),
    )

    output = tmp_path / "probe.json"
    rc = main(
        [
            "probe",
            "--episodes",
            "200",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_episodes"] == 200
    assert (tmp_path / "probe.md").exists()


# ---------------------------------------------------------------------------
# Plots / Summary
# ---------------------------------------------------------------------------


def test_plots_requires_baseline_and_final(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["plots", "--output-dir", str(tmp_path)])
    assert excinfo.value.code == 2


def test_plots_handler_invokes_render_plots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_path, final_path = _write_baseline_and_final(tmp_path)
    out_dir = tmp_path / "plots"

    import cells.step_21_plots as plots_mod

    capture: dict[str, Any] = {}

    def fake_render(
        baseline: Any, final: Any, run_id: Any, dest: Path
    ) -> dict[str, Path]:
        capture["called"] = True
        capture["dest"] = dest
        return {"per_language_bars": dest / "per_language_bars.png"}

    monkeypatch.setattr(plots_mod, "render_plots", fake_render)

    rc = main(
        [
            "plots",
            "--baseline",
            str(base_path),
            "--final",
            str(final_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert capture["called"] is True
    assert capture["dest"] == out_dir


def test_summary_writes_md_to_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_path, final_path = _write_baseline_and_final(tmp_path)
    probe_path = tmp_path / "probe.json"
    probe_path.write_text("{}", encoding="utf-8")

    import cells.step_22_summary as summary_mod

    monkeypatch.setattr(
        summary_mod,
        "print_summary_table",
        lambda baseline, final: "# DriftCall\n",
    )

    output = tmp_path / "summary.md"
    rc = main(
        [
            "summary",
            "--baseline",
            str(base_path),
            "--final",
            str(final_path),
            "--probe",
            str(probe_path),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.read_text(encoding="utf-8").startswith("# DriftCall")


# ---------------------------------------------------------------------------
# Deploy handler
# ---------------------------------------------------------------------------


def test_deploy_requires_repo(tmp_path: Path) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "deploy",
                "--checkpoint",
                str(tmp_path / "ckpt"),
                "--eval-reports",
                str(tmp_path),
            ]
        )
    assert excinfo.value.code == 2


def test_deploy_invokes_push_lora(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    eval_reports = tmp_path / "eval_reports"
    eval_reports.mkdir()

    import cells.step_24_deploy_hf as deploy_mod

    push_mock = MagicMock(
        return_value=SimpleNamespace(
            success=True,
            return_code=0,
            stderr="",
        )
    )
    monkeypatch.setattr(deploy_mod, "push_lora_to_hub", push_mock)
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    rc = main(
        [
            "deploy",
            "--checkpoint",
            str(ckpt),
            "--eval-reports",
            str(eval_reports),
            "--repo",
            "team/driftcall-lora",
        ]
    )
    assert rc == 0
    assert push_mock.called


# ---------------------------------------------------------------------------
# Factory smoke test
# ---------------------------------------------------------------------------


def test_factories_construct_real_callables() -> None:
    task_gen = build_task_gen()
    env_factory = build_env_factory(
        curriculum_stage=1,
        language_weights={
            "en": 0.50,
            "hinglish": 0.30,
            "hi": 0.20,
            "ta": 0.0,
            "kn": 0.0,
        },
    )
    rollout_group_fn = build_rollout_group_fn()

    assert callable(task_gen)
    assert callable(env_factory)
    assert callable(rollout_group_fn)

    goal = task_gen(
        seed=0,
        stage=1,
        language_weights={
            "en": 0.50,
            "hinglish": 0.30,
            "hi": 0.20,
            "ta": 0.0,
            "kn": 0.0,
        },
    )
    from cells.step_04_models import GoalSpec

    assert isinstance(goal, GoalSpec)

    from cells.step_10_env import DriftCallEnv

    env_a = env_factory()
    env_b = env_factory()
    assert isinstance(env_a, DriftCallEnv)
    assert isinstance(env_b, DriftCallEnv)
    assert env_a is not env_b


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_eval_report_json_roundtrip() -> None:
    import math

    original = _eval_report("base")
    payload = serialize_eval_report(original)
    restored = deserialize_eval_report(payload)
    assert restored.model_path == original.model_path
    assert restored.n_episodes == original.n_episodes
    assert restored.reward_mean_ci == original.reward_mean_ci
    assert restored.per_language == original.per_language
    assert math.isnan(restored.drift_detection_latency.stage2_mean)


def test_serialize_eval_report_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        serialize_eval_report({"not": "a dataclass"})


# ---------------------------------------------------------------------------
# build_briefs fallback
# ---------------------------------------------------------------------------


def test_build_briefs_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_pipeline, "_PUBLICATION_VAL_BRIEFS", Path("/nonexistent/briefs.jsonl")
    )
    rows = build_briefs(min_rows=10)
    assert len(rows) >= 10
    assert all(hasattr(r, "episode_id") for r in rows)
