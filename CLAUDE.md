# CLAUDE.md — DriftCall Phase C: Numbered-Cell Code Implementation

**Last updated:** 2026-04-25
**Phase:** C (code implementation)
**Predecessor:** Phase D complete — 14 module specs + 14 test plans, all critic-gated, cross-doc consistent.
**Target artifact:** Colab-runnable notebook `notebooks/train_driftcall.ipynb` built by concatenating numbered cell `.py` files from `cells/`.
**Event deadline:** Apr 26 2026, pitch day.
**Compute:** 1× V100 32GB (local training) + $30 HF Space credit (deployment).
**Base model:** `unsloth/gemma-3n-E2B-it`.

---

## 0. Prime Directive

Implement DriftCall as **25 numbered Python cells** in `cells/step_NN_<name>.py` that each becomes one cell of the Colab notebook, and that each is *also* an importable module used by the FastAPI env server, the demo Space, and the test suite. Every cell's content is spec'd by a Phase D module doc; every test is spec'd by a Phase D test plan doc.

**The docs are the contract. Your job is to ship code that passes the contract.**

**Stop conditions:**
1. All batch checklists green + `notebooks/train_driftcall.ipynb` builds + runs end-to-end on Colab T4 / V100 + demo Space deploys.
2. Any attempt to modify a Phase D doc without orchestrator approval → halt and escalate.
3. Gemma 3n E2B fails to boot in Unsloth → halt and escalate (no fallback model — Gemma 3n E2B is the pinned choice).

Everything else: proceed without interruption.

---

## 1. Source of Truth Hierarchy (frozen)

```
DESIGN.md                     ← master spec, v1.0 LOCKED (5 post-gate corrections applied)
docs/modules/*.md  (14)       ← per-module specs, ≥2 clean critic passes each
docs/tests/*.md    (14)       ← per-module test plans with test IDs, fixtures, coverage targets
   │
   ▼  implementation derives from these
cells/step_NN_<name>.py       ← numbered notebook cells = importable modules
tests/test_step_NN_<name>.py  ← pytest per cell, mirrors docs/tests/*.md
app.py                        ← FastAPI env server (imports from cells/)
demo/app_gradio.py            ← Gradio demo Space (imports from cells/)
notebooks/build_notebook.py   ← jupytext-based concatenator
notebooks/train_driftcall.ipynb  ← generated Colab artifact (committed)
data/                         ← task briefs, drift patterns, API schemas, audio fixtures
Dockerfile, pyproject.toml, openenv.yaml, requirements.txt
```

**Rule — never touch Phase D docs without escalation.** If a doc is wrong, stop code work, update the doc, re-run critic, then resume.

---

## 2. The 25 Numbered Cells

Every `.py` file in `cells/` is named `step_NN_<snake_name>.py`, where `NN` is a zero-padded two-digit sequence. Each file:

1. **Is one notebook cell** (top-level code only; no `if __name__ == "__main__":`).
2. **Is an importable module** — other cells import by `from cells.step_NN_name import X`.
3. **Has a twin test file** `tests/test_step_NN_<snake_name>.py` matching the Phase D test plan.
4. **Has a matching markdown preamble** `cells/step_NN_<snake_name>.md` (short — 1-3 sentences of context before the code).

| # | File | Phase D doc | Purpose |
|---|---|---|---|
| 01 | `step_01_install.py` | deploy_env_space.md §4 | `pip install` pinned deps + HF auth |
| 02 | `step_02_imports.py` | — | Consolidated imports |
| 03 | `step_03_fixtures.py` | datasets.md | Download / unpack `data/` |
| 04 | `step_04_models.py` | models.md | 7 frozen dataclasses + ActionType |
| 05 | `step_05_vendors.py` | vendors.md | 5 mock vendor modules + helpers |
| 06 | `step_06_drift_injector.py` | drift_injector.md | 20-pattern catalogue + build_schedule + apply_drift |
| 07 | `step_07_task_generator.py` | task_generator.md | generate() with blake2b seeding |
| 08 | `step_08_rewards.py` | rewards.md | 5 reward functions + Brier + uncertain-floor |
| 09 | `step_09_audio.py` | audio.md | TTSEngine + ASREngine + AudioTrace |
| 10 | `step_10_env.py` | env.md | DriftCallEnv class |
| 11 | `step_11_smoke_env.py` | env.md §8 | Run one Stage-1 episode locally |
| 12 | `step_12_gemma_boot.py` | training.md §3.1 | Load Gemma 3n E2B (Unsloth, 4-bit, hardware-aware precision, dtype-slippage assert) |
| 13 | `step_13_grpo_config.py` | training.md §2.4 | `build_grpo_config(stage)` + reward_fn wiring |
| 14 | `step_14_custom_trainer.py` | training.md §3.2.3 | `DriftCallGRPOTrainer` + `EpisodeDatasetAdapter` |
| 15 | `step_15_train_stage1.py` | training.md §3.5 | Stage 1: 150 GRPO steps, no drift |
| 16 | `step_16_train_stage2.py` | training.md §3.5 | Stage 2: 200 steps, single drift |
| 17 | `step_17_train_stage3.py` | training.md §3.5 | Stage 3: 150 steps, compound drift |
| 18 | `step_18_eval_baseline.py` | evaluation.md §2.1 | Baseline eval on untrained E2B (50 eps) |
| 19 | `step_19_eval_final.py` | evaluation.md §2.1 | Final eval on trained LoRA (50 eps, same 50 seeds) |
| 20 | `step_20_probe.py` | evaluation.md §2.1 | 200-episode reward-hacking probe |
| 21 | `step_21_plots.py` | evaluation.md §3.4 | 4 target curves (per-reward, drift-latency, per-lang, before/after) |
| 22 | `step_22_summary.py` | pitch_demo.md | Before/after metrics table + narrative |
| 23 | `step_23_demo_gradio.py` | deploy_demo_space.md | Inline Gradio demo for Colab |
| 24 | `step_24_deploy_hf.py` | deploy_env_space.md + deploy_demo_space.md | Push trained LoRA + env Space + demo Space |
| 25 | `step_25_conclusion.py` | pitch_demo.md | Final metrics, HF links, closing |

**Companion non-cell files (not numbered, live outside `cells/`):**

- `app.py` — FastAPI env server (imports from `cells/`)
- `demo/app_gradio.py` — standalone demo Space (imports from `cells/`, not the inline `step_23` variant)
- `notebooks/build_notebook.py` — concatenates `cells/step_NN_*.py` + `cells/step_NN_*.md` into `notebooks/train_driftcall.ipynb`
- `data/` — authored fixtures (YAML, JSON)
- `tests/conftest.py` — shared pytest fixtures
- `scripts/smoke_gemma3n_boot.py` — one-off smoke test
- `Dockerfile`, `pyproject.toml`, `requirements.txt`, `openenv.yaml`, `.gitignore`, `README.md`

---

## 3. Methodology: TDD → 2 Fresh Critics → Merge

Every cell follows this rigid loop:

```
1. READ   (module doc + test plan doc + DESIGN.md section)
2. RED    (write tests first in tests/test_step_NN_<name>.py, verify they fail)
3. GREEN  (implement cells/step_NN_<name>.py minimally to pass)
4. GATES  (pytest, coverage ≥ 85% line, ruff, mypy --strict)
5. CRITIC ≥ 2 fresh critics on the (.py + test) pair return NOTHING_FURTHER
6. MERGE  (orchestrator merges worktree → main)
```

**Rule — every numbered cell file goes through ≥ 2 fresh critic agents returning `NOTHING_FURTHER`**, identical to the Phase D gate. A third round fires if any prior returns `NEEDS_CHANGES`. Critics read main tree; authors use `isolation: "worktree"`.

### 3.1 Quality gates (every commit)

| Gate | Command | Threshold |
|---|---|---|
| pytest | `python3 -m pytest tests/ -v` | 100% pass |
| coverage | `python3 -m pytest tests/ --cov=cells --cov-branch` | line ≥ 85%, branch ≥ 75% on touched files |
| ruff | `ruff check cells/ tests/ app.py demo/ notebooks/` | 0 warnings |
| mypy | `mypy --strict cells/ app.py demo/` | 0 errors |
| openenv | `openenv validate .` | passes (C3 onward) |
| notebook build | `python3 notebooks/build_notebook.py` | produces valid .ipynb |

### 3.2 TDD author briefing (mandatory prefix for every implementer agent)

```
You are implementing cells/step_NN_<name>.py + tests/test_step_NN_<name>.py for DriftCall Phase C.

MANDATORY READS (in order):
  1. DRIFTCALL/DESIGN.md §<relevant section>
  2. DRIFTCALL/docs/modules/<module>.md   (the design contract)
  3. DRIFTCALL/docs/tests/<module>_tests.md  (the test contract)
  4. Existing cells/ merged from earlier batches (if any)
  5. DRIFTCALL/CLAUDE.md §2 (cell structure + 25-cell list + naming)

TDD workflow:
  1. Write tests/test_step_NN_<name>.py FIRST from the test plan doc — verify they fail (RED).
  2. Implement cells/step_NN_<name>.py minimally — verify tests pass (GREEN).
  3. Refactor only if quality gates demand it.
  4. Never add functions, classes, or features not in the module doc.

Cell-structure rules:
  - File is top-level code only (no `if __name__ == "__main__":` block).
  - File must be importable: `from cells.step_NN_<name> import X` works.
  - `from __future__ import annotations` on every file.
  - Frozen dataclasses only (frozen=True).
  - Imports at top of file (other cells use `from cells.step_NN_... import`).
  - Short markdown preamble at `cells/step_NN_<name>.md` (1–3 sentences).

Hard rules:
  - No placeholders, no TODOs, no `# type: ignore`, no `# noqa`
  - No hardcoded secrets
  - No LLM judge in reward pipeline (ever)
  - Pure functions where the doc says "pure"
  - If the doc is wrong, STOP — do not diverge
  - Quality gates ALL must pass before marking task complete
```

---

## 4. Phase C Batch Plan

5 batches. Each implementer owns a non-overlapping set of step_NN files.

### Batch C1 — Leaves (parallel worktrees, ~4h)

| Agent | Step files owned | Companion |
|---|---|---|
| **coder-A1** | `step_04_models.py` | `tests/test_step_04_models.py` |
| **coder-A2** | `step_05_vendors.py` | `tests/test_step_05_vendors.py` |
| **coder-C1** | `step_09_audio.py` | `tests/test_step_09_audio.py` |
| **coder-D1** | `step_01_install.py` + `step_02_imports.py` + `Dockerfile` + `pyproject.toml` + `requirements.txt` + `openenv.yaml` + `.gitignore` + `README.md` skeleton | (no tests — config files; README gets a basic smoke test) |

**Gate:** every agent pytest green on their subset, coverage ≥ 85%, ruff + mypy clean; ≥ 2 fresh critics each → NOTHING_FURTHER.

### Batch C2 — Drift + Task-Gen + Rewards + Env (sequenced)

Depends on C1 merge.

| Order | Agent | Files |
|---|---|---|
| 1 (parallel) | coder-A2 | `step_06_drift_injector.py` + `tests/test_step_06_drift_injector.py` |
| 1 (parallel) | coder-A3 | `step_07_task_generator.py` + `tests/test_step_07_task_generator.py` |
| 1 (parallel) | coder-C2 | `step_03_fixtures.py` + `data/` authored (templates.yaml, i18n.yaml, drift_patterns/*.yaml, api_schemas/*) |
| 2 | coder-B1 | `step_08_rewards.py` + `tests/test_step_08_rewards.py` |
| 3 | coder-A4 | `step_10_env.py` + `tests/test_step_10_env.py` (composes 4–9) |
| 4 | coder-B2 | `step_11_smoke_env.py` + `tests/test_step_11_smoke_env.py` |

### Batch C3 — Server + Data Space Push (parallel)

| Agent | Files |
|---|---|
| **coder-A5** | `app.py` (FastAPI + OpenEnv REST) + `tests/test_app.py` |
| **coder-B3** | `tests/test_e2e.py` (end-to-end episode) |
| **coder-D2** | Deploy env Space to HF (CPU basic, free tier) + `openenv validate` green against live Space |
| **coder-D3** | `notebooks/build_notebook.py` (jupytext concatenator, reads `cells/step_NN_*.py` + `.md` in numeric order, produces `.ipynb`) |

### Batch C4 — Training + Eval + Demo (parallel)

| Agent | Files |
|---|---|
| **coder-C3** | `step_12_gemma_boot.py` + `step_13_grpo_config.py` + `step_14_custom_trainer.py` + tests |
| **coder-C4** | `step_15_train_stage1.py` + `step_16_train_stage2.py` + `step_17_train_stage3.py` + tests |
| **coder-C5** | `step_18_eval_baseline.py` + `step_19_eval_final.py` + `step_20_probe.py` + `step_21_plots.py` + `step_22_summary.py` + tests |
| **coder-D4** | `step_23_demo_gradio.py` + `demo/app_gradio.py` (Space variant) + `step_24_deploy_hf.py` + `step_25_conclusion.py` + tests |

### Batch C5 — Ship (serial, ~4h + training compute)

1. coder-C3 + V100: run Stage 1/2/3 GRPO (500 steps total, ~14h wall-clock)
2. coder-C5: final eval on 50 held-out → EvalReport + 4 curves
3. coder-B1: reward-hacking probe 200 eps → `docs/probe_report.md`
4. coder-D4: push trained LoRA to HF Hub + demo Space with LoRA loaded
5. coder-D3: rebuild `notebooks/train_driftcall.ipynb` with real training numbers, commit
6. coder-D4: blog post + YouTube video + pitch deck
7. All hands: pitch rehearsal × 3

---

## 5. Notebook Builder Contract (`notebooks/build_notebook.py`)

```python
# Pseudocode for the builder; concrete impl in C3 batch.
# Reads cells/ in numeric order, emits notebooks/train_driftcall.ipynb.

import jupytext
from pathlib import Path

def build() -> None:
    cells_dir = Path("cells")
    ipynb_path = Path("notebooks/train_driftcall.ipynb")

    # 1. Sort cells/step_NN_*.py by NN.
    py_files = sorted(cells_dir.glob("step_[0-9][0-9]_*.py"))

    # 2. For each .py, find matching .md sibling (step_NN_<name>.md). Use as markdown cell.
    notebook = jupytext.new_notebook()
    for py in py_files:
        md_sibling = py.with_suffix(".md")
        if md_sibling.exists():
            notebook.cells.append(jupytext.new_markdown_cell(md_sibling.read_text()))
        notebook.cells.append(jupytext.new_code_cell(py.read_text()))

    # 3. Write .ipynb.
    jupytext.write(notebook, ipynb_path, fmt="ipynb")

if __name__ == "__main__":
    build()
```

Invariants:
- Rebuilding is idempotent: same inputs → byte-identical `.ipynb`.
- No cell in the notebook duplicates code that's already in a prior cell.
- Every numbered .py file in `cells/` becomes exactly one code cell, in numeric order.
- Every `cells/step_NN_<name>.md` becomes a markdown cell preceding its code twin.

---

## 6. TeamCreate Dispatch Patterns

### 6.1 Parallel implementer batch

Send ONE message with 4 Agent calls, `isolation: "worktree"`, `run_in_background: true`. Each gets the TDD briefing prefix (§3.2) + file list + module + test plan doc paths.

### 6.2 Critic round (≥2 per cell, main tree, read-only)

Critics:
- Round 1: one critic per numbered cell, parallel
- Round 2: fresh critics verify fixes OR confirm clean
- Round 3: only if any round returned `NEEDS_CHANGES`

### 6.3 Rules

1. **Max 4 concurrent agents**
2. **Never reuse a critic** — fresh each round
3. **Merge conflicts = orchestrator-only**
4. **If two agents need the same cell, redesign the batch**
5. **Quality gates are binding**

---

## 7. Commands

Run from `DRIFTCALL/` unless noted.

| Purpose | Command |
|---|---|
| Install dev | `uv pip install -e '.[dev]'` |
| Tests | `python3 -m pytest tests/ -v` |
| Coverage | `python3 -m pytest tests/ --cov=cells --cov-branch --cov-report=term-missing` |
| Lint | `ruff check cells/ tests/ app.py demo/ notebooks/` |
| Format | `ruff format cells/ tests/ app.py demo/` |
| Types | `mypy --strict cells/ app.py demo/` |
| OpenEnv validate | `openenv validate .` |
| Env server local | `uvicorn app:app --host 0.0.0.0 --port 8000` |
| Train Stage N | `python3 cells/step_15_train_stage1.py` (run standalone; same as notebook cell) |
| Notebook build | `python3 notebooks/build_notebook.py` |
| Docker build | `docker build -t driftcall-env .` |
| HF Space env | `hf upload <team>/driftcall-env . --repo-type space` |
| HF Hub model | `model.push_to_hub("<team>/gemma-3n-e2b-driftcall-lora", safe_serialization=True)` |
| Gemma 3n smoke | `python3 scripts/smoke_gemma3n_boot.py` |

---

## 8. Sanity Checks (before every commit)

1. Current batch's tests green
2. Coverage ≥ 85% line on touched cells
3. `ruff check` clean
4. `mypy --strict` clean
5. `from __future__ import annotations` in every `.py`
6. Frozen dataclasses only
7. **Code matches module doc + test plan exactly** — if diverges, escalate
8. No `# type: ignore`, `# noqa`, TODOs
9. No hardcoded secrets
10. Commit message: `feat(driftcall-c<N>): step_NN_<name> (docs/modules/<x>.md §N)`

---

## 9. Phase C Kickoff Checklist

Before Batch C1:

- [ ] Gemma 3n E2B smoke test passes on V100 (DESIGN.md §16.A.1)
- [ ] Unsloth 2026.4.5+, TRL 0.23+, PyTorch 2.5+ installed
- [ ] Kokoro-82M + faster-whisper-small in local HF cache
- [ ] HF CLI authenticated (`hf auth login`)
- [ ] Team `driftcall-code` created ✅
- [ ] Team roles assigned: coder-A (env/vendors/models/env), coder-B (rewards/tests), coder-C (audio/training/eval), coder-D (deploy/demo/notebook)
- [ ] `pyproject.toml` draft reviewed
- [ ] `.gitignore` excludes `checkpoints/`, `.cache/`, `*.pyc`, `__pycache__/`, `*.ipynb_checkpoints/`

---

## 10. What NOT To Do

- ❌ Modify any Phase D doc without orchestrator approval
- ❌ Improvise architecture — every class/function is in a module doc
- ❌ Add an LLM judge anywhere in the reward pipeline
- ❌ Put TTS/ASR in the training loop (text-only GRPO)
- ❌ Load Gemma 4 E4B on V100 for training (E2B only)
- ❌ Merge 4-bit → 16-bit naively (DESIGN.md §10.5)
- ❌ Write features not in the docs
- ❌ Skip tests (TDD is rigid)
- ❌ Run more than 4 concurrent implementer agents
- ❌ Touch files outside your batch assignment
- ❌ Force-resolve merge conflicts via agent
- ❌ Use deprecated `huggingface-cli upload` — use `hf upload`
- ❌ Commit notebook without regenerating (`python3 notebooks/build_notebook.py`)
- ❌ Create `step_NN_*.py` files outside the 25-cell list without orchestrator approval

---

## 11. Escalation & Stop Conditions

**Escalate (pause dispatch) if:**
- Gemma 3n E2B smoke fails on V100 → block; escalate to user (no fallback model)
- `openenv validate` fails after 3 attempts → block
- Stage-1 training R1 < 0.4 at step 100 → block; reward/curriculum revisit
- Critic finds consistent Phase D doc flaw → update spec first
- Merge conflict across differently-owned cells → ownership was wrong
- V100 unavailable > 8h → block; Colab T4 fallback

**Hard stop (terminate, re-plan with user) if:**
- HF Hub/Spaces outage > 2h (risk_book.md R06-STOP)
- Team drops below 3 active members
- Reward hacking in training that probe doesn't catch

---

## 12. Change Log

| Date | Change | By |
|---|---|---|
| 2026-04-24 | Phase D doc-first methodology | Orchestrator |
| 2026-04-25 | Phase C numbered-cell structure. 25 cells in `cells/step_NN_<name>.py`, each = one notebook cell AND importable module. Builder concatenates in numeric order. Same critic-gate discipline as Phase D. | Orchestrator |

---

**Phase D = specs. Phase C = 25 numbered cells that ARE both the notebook and the package. Read the module doc. Read the test plan. Write tests first. Implement the cell. Pass the gates. 2 critics clean. Ship.**
