# evaluation_tests.md — Test Plan for `docs/modules/evaluation.md`

**Target modules:** `training/eval_baseline.py`, `training/eval_final.py`, `training/probe_reward_hacking.py` (aliased `training/probe.py` in test imports), `training/plots.py`
**Spec doc:** `DRIFTCALL/docs/modules/evaluation.md` (final sealed, 2026-04-26)
**Cross-refs:** `DRIFTCALL/docs/modules/training.md` §2.1 (`eval()` contract, §4.2 `EvalReport`), `DRIFTCALL/docs/modules/rewards.md` §3.1 (purity), §3.6 (exploit classes), §4.2 (`Rewards.breakdown`), `DRIFTCALL/docs/modules/datasets.md` §4.7 (val split), §5 (catalogue hashes), `DRIFTCALL/CLAUDE.md` §3.1 (nine-section test-plan doc).
**Framework:** `pytest` + `hypothesis` + `unittest.mock` + `pytest-mpl` (plot image-compare, tolerance-based).
**Owner:** Person B (Rewards & Tests), co-signed by Person C (Training) for the `training.eval` delegation path.
**CUDA policy:** **Model inference is mocked by default.** Every test that would touch a real LoRA adapter or base weight goes through the `stubbed_training_eval` fixture (§5.3), which monkeypatches `training.train.eval` to return a hand-crafted `EvalReport`. Zero CUDA calls in CI. A single `@pytest.mark.cuda` integration test exercises the real delegation path and is skipped when `torch.cuda.is_available() is False` (the default laptop / CI environment).
**Deterministic RNG:** `numpy.random.default_rng(20260426)` for baseline + final bootstrap, `numpy.random.default_rng(20260427)` for the probe CI, `numpy.random.default_rng(20260428)` for paired-difference bootstrap. Seeds are frozen in `evaluation.md` §2.4 and re-asserted at every call site.
**Numeric tolerance:** `math.isclose(a, b, abs_tol=1e-9, rel_tol=0.0)` for scalar floats; `numpy.testing.assert_allclose(..., atol=1e-6, rtol=0.0)` for sample arrays; byte-exact for serialized JSON (`sort_keys=True, separators=(",", ":")`); image diff tolerance `rms < 0.5` pixel for plot snapshots.

This plan specifies **100% line coverage** and **≥ 95% branch coverage** on `training/eval_baseline.py`, `training/eval_final.py`, `training/probe.py`, and `training/plots.py`. Every function signature in evaluation.md §2, every behavior clause in §3.1–§3.8, every error mode in §5, every data-structure invariant in §4, and every edge case in §7 has at least one dedicated test.

**Fixtures contract (§5).** This plan is the **source of truth** for all eval-related fixtures. Bidirectionally-consistent sharing map (see §5.6 for the authoritative truth-verified table):

- `eval_50_episodes_val_slice` — defined here (§5.1). **Imported** by `training_tests.md` §3.4 only (via cross-reference at `training_tests.md §5.6`).
- `baseline_eval_report_fixture`, `final_eval_report_fixture` — defined here (§5.2, §5.3). **Eval-only**; not consumed by any other plan.
- `probe_report_no_exploits` — defined here (§5.4). **Imported** by `pitch_demo_tests.md` §5.5 (blog badge I12) only.
- `probe_report_with_novel_class` — defined here (§5.5). Eval-only.

Any change to fixture bodies MUST be mirrored in `tests/conftest.py` in lockstep. PR reviewers for any consuming plan must verify content parity against this plan's §5.

---

## 1. Unit tests

**Organisation:** one `pytest` module per behavior cluster. Zero real inference — every test either uses `stubbed_training_eval` or asserts against pure-Python helpers.

File layout under `tests/test_evaluation/`:

```
tests/test_evaluation/
  __init__.py
  test_run_eval_signature.py              # run_eval signature + delegation (§2.1, §3.2)
  test_episode_selection_deterministic.py # val[0:50], seed hashing, leak guard (§3.1)
  test_sampling_policy_frozen.py          # T=0, num_gen=1, eval(), no_grad (§3.2)
  test_aggregation_bootstrap_ci.py        # per-reward CI store (§3.3)
  test_per_language_cohort.py             # cohort means + low-n rendering (§3.4)
  test_drift_detection_latency.py         # §3.5 latency compute + Stage-1 NaN
  test_probe_scanner_mechanics.py         # §3.6 scan + Counter aggregation
  test_probe_novel_class.py               # §3.6 unknown offense code path
  test_probe_on_base_guard.py             # ProbeOnBaseModelError (§5)
  test_probe_insufficient_samples.py      # ProbeInsufficientSamplesError (§5)
  test_episode_set_leak_error.py          # EpisodeSetLeakError on mismatch (§5)
  test_catalogue_hash_mismatch.py         # CatalogueHashMismatchError (§5)
  test_eval_budget_exceeded.py            # EvalBudgetExceededError all three ceilings (§3.8, §5)
  test_bootstrap_edge_cases.py            # len 0 / 1 / all-identical samples (§2.4)
  test_zero_success_baseline.py           # warning + CI undefined breakdown (§7.1)
  test_plot_rendering.py                  # all 4 curves, PNG exists + shape (§2.1, §3.5)
  test_plot_graceful_degrade.py           # WandB unavailable skips 2 plots (§3.5, §7.6)
  test_render_probe_report_md.py          # fixed 35-line template (§2.3, §4.5)
  test_probe_report_json_roundtrip.py     # sort_keys serialization (§4.4)
  test_exploit_classes_always_emitted.py  # all 5 in report even when zero (§3.6)
```

**Unit test case inventory — 27 cases total (exceeds the ≥ 25 requirement).**

### 1.1 `run_eval` signature + delegation — `test_run_eval_signature.py`

**Scope:** `run_eval(model_path, episodes=50) -> EvalReport` is a thin wrapper over `training.train.eval`; it must pass `model_path` through verbatim, default `episodes=50`, and return the delegate's `EvalReport` unchanged.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U1 | `test_run_eval_default_episodes_50` | Stub `training.train.eval` to capture kwargs; call `run_eval("base")`. | Delegate invoked with `model_path="base"` and `episodes=50`. |
| U2 | `test_run_eval_accepts_literal_base` | `run_eval("base", 50)`. | No error; returns stubbed `EvalReport` with `model_path == "base"`. |
| U3 | `test_run_eval_accepts_path_object` | `run_eval(Path("/tmp/ckpt"), 50)`. | Delegate invoked with `model_path=Path("/tmp/ckpt")`. |
| U4 | `test_run_eval_propagates_model_load_error` | Stub raises `EvalModelLoadError("adapter missing")`. | `run_eval` re-raises `EvalModelLoadError`; no silent fallback. |

### 1.2 Episode selection determinism — `test_episode_selection_deterministic.py`

**Scope:** evaluation.md §3.1 — baseline and final both read `val/briefs.jsonl[0:50]` in file order; probe reads `[50:250]`; env seed = `hash((episode_id, "eval")) & 0xFFFFFFFF`; paired `(episode_id, seed)` tuples must match between baseline and final.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U5 | `test_eval_reads_first_50_rows_in_file_order` | `eval_50_episodes_val_slice` fixture + mock `load_briefs` recording `.take(50)`. Call `run_eval("base", 50)`. | `load_briefs` received exactly the first 50 `BriefRow`s; episode_ids order matches file order. |
| U6 | `test_probe_reads_rows_50_to_250` | Mock `load_briefs` with 500-row fixture. Call `probe_reward_hacking(Path("/ckpt"), 200)`. | The 200 `BriefRow`s passed to `training.eval` are rows `[50:250]` — disjoint from the paired 50, confirmed by episode_id set intersection == ∅. |
| U7 | `test_env_seed_is_hash_tuple_episode_id_eval` | Mock `env.reset` to record seed. | For each episode, recorded `seed == hash((episode_id, "eval")) & 0xFFFFFFFF`. |
| U8 | `test_baseline_and_final_share_same_seeds` | `stubbed_training_eval` records seeds per run. Run baseline then final. | `baseline.breakdown["episode_ids"] == final.breakdown["episode_ids"]`; zipped seeds are pairwise identical. |
| U9 | `test_episode_set_leak_error_raised_on_mismatch` | Baseline fixture with episode_ids `[a,b,c,...]`; final fixture with `[a,b,X,...]`. Call post-run guard. | Raises `EpisodeSetLeakError` with substring `"paired-comparison invariant"`. |

### 1.3 Sampling policy frozen — `test_sampling_policy_frozen.py`

**Scope:** evaluation.md §3.2 — greedy `temperature=0.0`, `top_k=1`, `num_generations=1`, `model.eval()`, `torch.no_grad()`, all dropouts OFF. `run_eval` re-asserts these at entry.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U10 | `test_run_eval_enforces_temperature_zero` | Stub `training.eval` with a capture-then-assert that records sampling kwargs. | Captured `temperature == 0.0`; `top_k == 1`; `num_generations == 1`. |
| U11 | `test_run_eval_wraps_in_no_grad_and_eval_mode` | Mock torch context; assert `model.eval()` called and `torch.no_grad()` context entered before first forward. | `model.eval` call_count ≥ 1; `no_grad.__enter__` called before first forward. |
| U12 | `test_run_eval_dropouts_off` | Stub model with `.train()` recorded; assert `.eval()` wins and dropout modules report `.training is False`. | All `nn.Dropout` / LoRA-dropout modules have `training is False` at sample time. |

### 1.4 Aggregation + bootstrap CI — `test_aggregation_bootstrap_ci.py`

**Scope:** evaluation.md §3.3 — `bootstrap_ci(samples, n_boot=10_000, alpha=0.05, rng_seed=20260426)` called once per reward channel; results stored on `EvalReport.r{i}_mean_ci` tuple.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U13 | `test_bootstrap_ci_default_n_boot_10000` | Call `bootstrap_ci(tuple(range(50)), n_boot=10_000, rng_seed=20260426)`. | Returned `(mean, lo, hi)` triple; `lo < mean < hi`; mean within `abs_tol=1e-9` of arithmetic mean. |
| U14 | `test_bootstrap_ci_deterministic_with_seed` | Call twice with identical args. | Byte-identical `(mean, lo, hi)` on both calls (re-run determinism). |
| U15 | `test_paired_difference_ci_uses_seed_20260428` | `paired_difference_ci(baseline, final)` with mocked rng capture. | `numpy.random.default_rng` called with `20260428`; output tuple reproducible. |

### 1.5 Bootstrap edge cases — `test_bootstrap_edge_cases.py`

**Scope:** evaluation.md §2.4 — len 0 → `(nan, nan, nan)`; len 1 → `(v, v, v)`; all-identical → `(v, v, v)` no variance.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U16 | `test_bootstrap_ci_len0_returns_all_nan` | `bootstrap_ci(tuple(), n_boot=10_000)`. | All three outputs `math.isnan(...)`. |
| U17 | `test_bootstrap_ci_len1_returns_triple_v` | `bootstrap_ci((0.42,), n_boot=10_000)`. | Returns `(0.42, 0.42, 0.42)` exactly. |

### 1.6 Per-language cohort rendering — `test_per_language_cohort.py`

**Scope:** evaluation.md §3.4 — cohorts with `n_episodes >= 5` render numeric mean + 95% CI; `1 <= n <= 4` renders striped low-n bar with label `(low-n)`; `n == 0` renders empty slot labelled `(no episodes)`. No CI computed for low-n or empty cohorts.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U18 | `test_cohort_ge_5_renders_numeric_with_ci` | Fixture cohort of 11 episodes. | Rendered bar payload has `n_episodes == 11`, numeric `mean`, `ci != None`. |
| U19 | `test_cohort_low_n_1_to_4_renders_striped` | Fixture cohort of 3 episodes. | Rendered bar carries `style == "striped"`, `label.endswith("(low-n)")`, `ci is None`. |
| U20 | `test_cohort_empty_renders_empty_slot` | Fixture cohort of 0 episodes. | Rendered entry has `n_episodes == 0`, `label.endswith("(no episodes)")`, `mean is float("nan")`. |

### 1.7 Probe scanner + novel class — `test_probe_scanner_mechanics.py`, `test_probe_novel_class.py`

**Scope:** evaluation.md §3.6 — all 5 exploit classes always enumerated in the report (count ≥ 0); any `offense.code ∉ EXPLOIT_CLASSES` surfaces as a Novel exploit (threshold = 1 occurrence).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U21 | `test_probe_emits_all_5_known_classes_even_when_zero` | Fixture rewards with zero anti-hack offenses. Call `probe_reward_hacking`. | `len(report.per_class) >= 5`; every member of `EXPLOIT_CLASSES` present; each has `count == 0`, `rate == 0.0`, `example_episode_id is None`. |
| U22 | `test_probe_novel_class_surfaced_on_single_instance` | Fixture: one episode with `offense.code == "zero_width_evasion"`. | `report.novel_classes == ("zero_width_evasion",)`; summary row for it has `count == 1`; markdown contains substring `"UNKNOWN EXPLOIT CLASS"`. |

### 1.8 Probe guards — `test_probe_on_base_guard.py`, `test_probe_insufficient_samples.py`

**Scope:** evaluation.md §5 — `ProbeOnBaseModelError` when `model_path=="base"`; `ProbeInsufficientSamplesError` when `episodes < 50`.

| # | Name | Setup | Assertion |
|---|---|---|---|
| U23 | `test_probe_on_base_raises` | `probe_reward_hacking("base", 200)`. | Raises `ProbeOnBaseModelError`; no call to `training.eval`. |
| U24 | `test_probe_insufficient_samples_raises` | `probe_reward_hacking(Path("/ckpt"), 49)`. | Raises `ProbeInsufficientSamplesError` with substring `"n < 50"`. |

### 1.9 Catalogue hash + budget — `test_catalogue_hash_mismatch.py`, `test_eval_budget_exceeded.py`

**Scope:** evaluation.md §3.1 (catalogue pinning) and §3.8 (wall-clock ceilings 20 min / 60 min / 2 min, raising `EvalBudgetExceededError`).

| # | Name | Setup | Assertion |
|---|---|---|---|
| U25 | `test_catalogue_hash_mismatch_blocks_eval` | Fixture `BriefRow.catalogue_hash` set to `"stale"`; currently-loaded yaml hashes to `"current"`. Call `run_eval`. | Raises `CatalogueHashMismatchError` before any rollout; no `training.eval` call. |
| U26 | `test_run_eval_budget_20min_exceeded_raises` | Monkeypatch `time.monotonic` to simulate 20 min 1 s elapsed. | Raises `EvalBudgetExceededError` with substring `"run_eval"` and `"20 min"`. |
| U27 | `test_probe_budget_60min_and_plot_budget_2min` | Parametrized across `(probe_reward_hacking, 60*60+1)` and `(render_plots, 120+1)`. | Both raise `EvalBudgetExceededError` naming their respective ceiling. |

---

## 2. Property tests

Property tests use `hypothesis.strategies` to generate arbitrary-but-bounded inputs and assert invariants evaluation.md commits to. Each strategy is seeded (`hypothesis.seed(20260426)`) so failures are reproducible.

**Property inventory — 6 properties total (exceeds the ≥ 5 requirement).**

### 2.1 Eval purity — same (model, 50-ep) produces byte-identical `EvalReport`

```python
@given(checkpoint=st.sampled_from([Path("/ckpt/a"), Path("/ckpt/b")]))
@settings(max_examples=10, deadline=None)
def test_eval_is_pure(checkpoint, stubbed_training_eval):
    r1 = run_eval(checkpoint, episodes=50)
    r2 = run_eval(checkpoint, episodes=50)
    assert serialize(r1) == serialize(r2)
```

**Invariant:** For the same checkpoint and the same 50-row slice, two back-to-back `run_eval` calls produce `EvalReport` records whose canonical JSON (`sort_keys=True, separators=(",", ":")`) byte-compares equal, including every `r{i}_mean_ci` tuple and every WandB-independent `curves` entry. This is evaluation.md §1 "Deterministic on re-run" invariant, bound to the fixed `rng_seed=20260426` in `bootstrap_ci`.

### 2.2 Bootstrap CI convergence — jitter ≤ 0.001 at `n_boot=10_000`

```python
@given(samples=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                         min_size=50, max_size=50))
@settings(max_examples=20, deadline=None)
def test_bootstrap_ci_converges(samples):
    m1, lo1, hi1 = bootstrap_ci(tuple(samples), n_boot=10_000, rng_seed=20260426)
    m2, lo2, hi2 = bootstrap_ci(tuple(samples), n_boot=10_000, rng_seed=20260426 + 1)
    assert abs(lo1 - lo2) <= 0.001
    assert abs(hi1 - hi2) <= 0.001
```

**Invariant:** At `n_boot=10_000` the 2.5th / 97.5th percentile estimates differ by at most 0.001 across distinct bootstrap seeds on the same underlying samples — the Monte-Carlo jitter ceiling referenced by evaluation.md §3.3. This property defines "convergent enough" and guards against anyone lowering `n_boot` silently.

### 2.3 Paired-diff CI equals final − baseline per episode

```python
@given(b=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=50, max_size=50),
       f=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=50, max_size=50))
@settings(max_examples=30, deadline=None)
def test_paired_diff_ci_is_paired(b, f):
    pd_mean, _, _ = paired_difference_ci(tuple(b), tuple(f),
                                          n_boot=10_000, rng_seed=20260428)
    expected = sum(fi - bi for bi, fi in zip(b, f)) / 50
    assert math.isclose(pd_mean, expected, abs_tol=1e-9)
```

**Invariant:** The paired-difference mean is the per-index `final[i] - baseline[i]` arithmetic mean — not the difference of independent means. Guards against anyone accidentally computing `mean(final) - mean(baseline)` via two unlinked samples (evaluation.md §2.4 `paired_difference_ci` docstring + §7 edge case 7).

### 2.4 Paired-diff CI requires equal lengths

```python
@given(n_b=st.integers(min_value=1, max_value=100),
       n_f=st.integers(min_value=1, max_value=100))
def test_paired_diff_requires_equal_lengths(n_b, n_f):
    if n_b == n_f:
        return  # happy path covered elsewhere
    with pytest.raises(EpisodeSetLeakError):
        paired_difference_ci(tuple([0.1] * n_b), tuple([0.2] * n_f))
```

**Invariant:** Unequal lengths raise `EpisodeSetLeakError` — pairing is strictly index-aligned (evaluation.md §2.4).

### 2.5 Bootstrap CI bracketing — `lo ≤ mean ≤ hi`

```python
@given(samples=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                         min_size=2, max_size=200))
def test_bootstrap_ci_brackets_mean(samples):
    m, lo, hi = bootstrap_ci(tuple(samples), n_boot=10_000, rng_seed=20260426)
    assert lo <= m <= hi
```

**Invariant:** The percentile bootstrap CI always brackets the point estimate. Guards against transposed percentile extraction (e.g., accidentally returning 97.5 as `lo`).

### 2.6 Probe class closure — every emitted class is either known or appears in `novel_classes`

```python
@given(offense_codes=st.lists(st.text(min_size=1, max_size=32), min_size=0, max_size=40))
def test_probe_class_closure(offense_codes, stubbed_training_eval_with_offenses):
    report = probe_reward_hacking(Path("/ckpt"), episodes=200)
    for row in report.per_class:
        assert row.exploit_class in EXPLOIT_CLASSES or row.exploit_class in report.novel_classes
```

**Invariant:** Every `ProbeExploitClassSummary.exploit_class` value in `report.per_class` is either one of the 5 known classes or is listed verbatim in `report.novel_classes`. No summary row exists outside this closure — protects the discovery channel.

---

## 3. Integration tests

Integration tests exercise cross-module wiring end-to-end. Real `training.train.eval` is **stubbed** (no CUDA), but the full evaluation pipeline — `run_eval → EvalReport → render_plots` and `probe_reward_hacking → scan → render_probe_report_md` — runs on actual dataclass instances.

File: `tests/test_evaluation/test_integration.py`.

### 3.1 Baseline eval on base model (stubbed)

```
test_integration_baseline_on_base_model
  Setup:   stubbed_training_eval returns baseline_eval_report_fixture.
           eval_50_episodes_val_slice loaded from fixtures/val_briefs_50.jsonl.
  Action:  run_eval("base", 50)
  Assert:  EvalReport.model_path == "base"
           EvalReport.n_episodes == 50
           len(r1_mean_ci) == 3 and lo <= mean <= hi for every r{i}_mean_ci
           breakdown["episode_ids"] == tuple of 50 episode_ids from val[0:50]
           JSON round-trip through canonical serializer is byte-stable.
```

### 3.2 Final eval on trained LoRA (stubbed)

```
test_integration_final_on_trained_lora
  Setup:   stubbed_training_eval returns final_eval_report_fixture.
           baseline.json already present on disk (from §3.1).
  Action:  run_eval(Path("/fake/ckpt/stage3_final"), 50)
           then post-run guard: assert baseline.episode_ids == final.episode_ids
  Assert:  EvalReport.model_path == "/fake/ckpt/stage3_final"
           EvalReport.reward_mean_ci[0] > baseline.reward_mean_ci[0]
           paired_difference_ci stored under breakdown["paired_ci"]
           No EpisodeSetLeakError raised.
```

### 3.3 Probe 200 episodes → markdown report

```
test_integration_probe_200_episodes_produces_markdown
  Setup:   stubbed_training_eval returns 200 Rewards records with:
             - 2 hallucinated_field offenses
             - 1 bare_drift_claim offense
             - 0 of the other 3 classes
  Action:  report = probe_reward_hacking(Path("/fake/ckpt"), 200)
           md_path = render_probe_report_md(report, tmp_path / "probe_report.md")
  Assert:  md_path.exists()
           md_text.count("### ") == 5            # all 5 known classes present
           "Novel exploit classes: none" in md_text
           "**Total offenses:** 3" in md_text
           json_round_trip(report) bytes-stable
           probe_report.json passes schema validation.
```

### 3.4 Plot rendering for all 4 target curves

```
test_integration_render_all_4_plots
  Setup:   baseline_eval_report_fixture, final_eval_report_fixture,
           stubbed WandB run-history returning R{1..5}_mean per step and
           eval/drift_latency_p50+p95 at steps {50,100,150,200,300,400,500}.
  Action:  paths = render_plots(baseline, final, wandb_run_id="stub-run",
                                 out_dir=tmp_path)
  Assert:  set(paths.keys()) == {
               "per_reward_stack", "drift_latency_vs_step",
               "per_language_bars",  "before_after_bars"
           }
           every path .exists() and .stat().st_size > 1024 bytes
           PIL.Image.open(path).size == (1600, 900)   # canonical figsize
           pytest-mpl snapshot compare passes with rms < 0.5 per plot.
```

### 3.5 WandB unavailable — graceful degrade (2 plots)

```
test_integration_render_plots_without_wandb
  Setup:   render_plots(..., wandb_run_id=None)
  Assert:  set(paths.keys()) == {"per_language_bars", "before_after_bars"}
           WandBHistoryUnavailableWarning emitted (captured via pytest.warns)
           no PlotRenderError raised
           returned dict omits the two history-driven plots.
```

### 3.6 GPU delegation path (skipped on CPU-only CI)

```
@pytest.mark.cuda
test_integration_real_training_eval_delegation
  Setup:   real Gemma 4 E2B + toy LoRA adapter on a 2-episode smoke slice.
  Assert:  run_eval returns a valid EvalReport; no exception.
```

Skipped automatically when `torch.cuda.is_available() is False`.

---

## 4. Coverage target

**Line coverage:** 100% on each of:
- `training/eval_baseline.py`
- `training/eval_final.py`
- `training/probe.py` (aliased for `training/probe_reward_hacking.py`)
- `training/plots.py`

**Branch coverage:** ≥ 95% on the same files. Exclusions (via `# pragma: no cover` with justification comment):

1. `if TYPE_CHECKING:` import blocks.
2. Real-CUDA-only branches inside `training.eval` delegation that the `stubbed_training_eval` fixture bypasses (marked with `# pragma: no cover-stubbed`).
3. `matplotlib` backend-selection dead code paths (`backend == "Agg"` already forced at module import).

**Verification command:**
```
pytest tests/test_evaluation/ \
  --cov=training.eval_baseline \
  --cov=training.eval_final \
  --cov=training.probe \
  --cov=training.plots \
  --cov-branch \
  --cov-fail-under=100 \
  --cov-report=term-missing
```

Branch coverage is independently enforced via `--cov-branch` against a local threshold file `.coveragerc` that sets `fail_under_branch = 95`.

**Per-file test → line mapping (authoritative):**

| File | Covering test file(s) | Targeted lines |
|---|---|---|
| `training/eval_baseline.py` | `test_run_eval_signature.py`, `test_episode_selection_deterministic.py`, `test_sampling_policy_frozen.py`, `test_zero_success_baseline.py`, `test_eval_budget_exceeded.py` | CLI argparse, `run_eval("base", …)` call, baseline.json write, ZeroSuccessBaselineWarning path. |
| `training/eval_final.py` | `test_run_eval_signature.py`, `test_episode_set_leak_error.py`, `test_aggregation_bootstrap_ci.py`, `test_plot_rendering.py`, `test_plot_graceful_degrade.py` | CLI, `run_eval(ckpt, …)`, paired-diff CI store, `render_plots` call, EpisodeSetLeakError guard at exit. |
| `training/probe.py` | `test_probe_scanner_mechanics.py`, `test_probe_novel_class.py`, `test_probe_on_base_guard.py`, `test_probe_insufficient_samples.py`, `test_render_probe_report_md.py`, `test_probe_report_json_roundtrip.py`, `test_exploit_classes_always_emitted.py` | `probe_reward_hacking`, `scan_episode_for_exploits`, `render_probe_report_md`, JSON serializer, all 5 guard branches. |
| `training/plots.py` | `test_plot_rendering.py`, `test_plot_graceful_degrade.py` | All 4 plot functions, WandB history fetcher, graceful-degrade branching, `PlotRenderError` path. |

---

## 5. Fixtures

All fixtures live in `tests/conftest.py` (repo-root shared scope) so the 4 shared names are identical content across `training_tests.md`, `pitch_demo_tests.md`, `risk_book_tests.md`, and this plan. Fixture IDs are the function names registered via `@pytest.fixture`.

### 5.1 `eval_50_episodes_val_slice`

**Shape:** `tuple[BriefRow, ...]` of length 50. Sourced from `tests/fixtures/val_briefs_50.jsonl` — the first 50 rows of a publication seed of `val/briefs.jsonl` (datasets.md §4.7), committed verbatim (≤ 120 KiB). Each row carries `episode_id`, `seed`, `catalogue_hash`, `templates_sha256`, `i18n_sha256`, `goal.language ∈ {hi, ta, kn, en, hinglish}`, and the embedded `GoalSpec`.

**Shared with:** `training_tests.md` (imported under the same canonical name `eval_50_episodes_val_slice` for `EpisodeDatasetAdapter` iteration in integration test 3.4 — see `training_tests.md §5` footer cross-reference), `pitch_demo_tests.md` (demo rollout harness), `risk_book_tests.md` (risk-book sample plots). **This plan is the sole definition site**; all consumers import from `tests/conftest.py`.

### 5.2 `baseline_eval_report_fixture`

**Shape:** A hand-built `EvalReport` mirroring the numbers in evaluation.md §8.1 verbatim:
- `model_path = "base"`, `n_episodes = 50`
- `reward_mean_ci = (0.118, 0.086, 0.152)`, `r1_mean_ci = (0.100, 0.040, 0.180)`, etc.
- 5 `PerLanguageReport`s (hi n=11, ta n=10, kn n=9, en n=10, hinglish n=10)
- `drift_detection_latency` with stage2/stage3 all NaN, `undetected_count=27`
- `breakdown["episode_ids"] = tuple_of_50_ids` deterministic
- `reward_hacking_offenses = {"hallucinated_field": 7, ...}` per §8.1

**Used by:** integration §3.1, §3.4, §3.5. **Eval-only** — not shared with `training_tests.md` (which constructs `EvalReport`s inline via stubbed `training.eval`).

### 5.3 `final_eval_report_fixture`

**Shape:** Hand-built `EvalReport` matching evaluation.md §8.2:
- `model_path = "/abs/path/checkpoints/stage3_final"`, `n_episodes = 50`
- `reward_mean_ci = (0.542, 0.480, 0.604)`, etc.
- `drift_detection_latency` with stage2_mean=1.2, stage3_mean=1.6, undetected_count=9
- `curves` dict with the 4 keys enumerated in §8.2
- `breakdown["paired_ci"]` populated with ΔR1, ΔR2, Δreward_mean, Δdrift_latency triples
- Shares `episode_ids` tuple with §5.2 (paired-comparison invariant)

**Used by:** integration §3.2, §3.4, §3.5. **Eval-only** — not shared with `training_tests.md`.

### 5.4 `probe_report_no_exploits`

**Shape:** `ProbeReport` with `n_episodes=200`, `per_class` containing all 5 known classes at `count=0`, `rate=0.0`, `example_episode_id=None`, `total_hits=0`, `novel_classes=()`. Generated from a stubbed `training.eval` that returns 200 `Rewards` records with empty `anti_hack.offenses` lists.

**Used by:** integration §3.3 (happy-path markdown rendering), `pitch_demo_tests.md` (probe-artefact badge I12). Not consumed by `risk_book_tests.md` (risk-book's domain is `Risk.triage` / risk register, not exploit reports).

### 5.5 `probe_report_with_novel_class`

**Shape:** `ProbeReport` with `n_episodes=200`, `per_class` containing all 5 known classes (zeros) **plus** a sixth `ProbeExploitClassSummary` for `exploit_class="zero_width_evasion"` with `count=1`, `rate=0.005`, `example_episode_id="s3_ep_00000131"`. `novel_classes=("zero_width_evasion",)`. Generated from a stubbed `training.eval` that seeds a single `offense.code="zero_width_evasion"` into one episode's `Rewards.breakdown.anti_hack.offenses`.

**Used by:** novel-class unit test (§1.7 U22). Eval-only — not consumed by other plans.

### 5.6 Cross-plan sharing contract

This plan (`evaluation_tests.md`) is the **sole definition site** for all 5 fixtures below. Consumers import from `tests/conftest.py`; they MUST NOT redefine. `✅` = consumed (imported from conftest and actually referenced in the consuming plan). `—` = not consumed by that plan.

| Fixture | Defined in | evaluation_tests.md | training_tests.md | pitch_demo_tests.md | risk_book_tests.md |
|---|---|---|---|---|---|
| `eval_50_episodes_val_slice` | evaluation_tests.md §5.1 | ✅ consumed (definer) | ✅ consumed (integration §3.4 paired eval; see training_tests.md §5.6 footer cross-reference) | — (no eval slice needed by pitch demo tests) | — (risk-register domain, no eval slice needed) |
| `baseline_eval_report_fixture` | evaluation_tests.md §5.2 | ✅ consumed (definer) | — (eval-only; training stubs `training.eval` directly) | — | — |
| `final_eval_report_fixture` | evaluation_tests.md §5.3 | ✅ consumed (definer) | — (eval-only) | — | — |
| `probe_report_no_exploits` | evaluation_tests.md §5.4 | ✅ consumed (definer) | — (probe entry-point tested via direct mocks in U45) | ✅ consumed (I12 blog badge) | — (risk register is about `Risk.triage`, not exploit reports) |
| `probe_report_with_novel_class` | evaluation_tests.md §5.5 | ✅ consumed (definer) | — | — | — |

**Truth-verification rule:** every `✅ consumed` above is bidirectionally consistent — the consuming plan's own §5 either (a) does not re-define the fixture, AND (b) explicitly cross-references this section as the definition site. The only cross-plan consumer rows are `training_tests.md` for `eval_50_episodes_val_slice` (see `training_tests.md §5.6`) and `pitch_demo_tests.md` for `probe_report_no_exploits` (see `pitch_demo_tests.md §5.5`). All other cells are `—`.

If a downstream plan needs to mutate any of these fixtures, it must define a derived fixture (e.g., `@pytest.fixture def probe_report_no_exploits_for_demo(probe_report_no_exploits): ...`) rather than editing the shared body. Enforced by a `tests/conftest_lock.py` sha256 check over the 5 fixture sources.

---

**End of evaluation_tests.md.** This plan is sealed pending ≥ 2 fresh critic `NOTHING_FURTHER` returns per `DRIFTCALL/CLAUDE.md` §3.4.
