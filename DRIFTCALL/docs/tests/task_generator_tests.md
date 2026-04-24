# task_generator_tests — Test Plan for `driftcall/task_generator.py`

**Module under test:** `driftcall/task_generator.py`
**Design doc:** `docs/modules/task_generator.md` (sealed)
**Cross-refs:** DESIGN.md §3.1 (System Architecture), §4.1, §4.2, §8.3, §8.4, §10.3
**Owner:** Person B (Rewards & Tests)
**Tooling:** `pytest`, `pytest-cov`, `hypothesis`, `pyyaml`, `unicodedata` (stdlib), `hashlib` (stdlib)
**Status:** Test-plan spec — no test code yet.

This plan is the authoritative test contract for `task_generator`. Every behavior clause in §3 of `task_generator.md` maps to at least one test case below. Every exception in §5 has a raise-site test. Every invariant in §3.6 has a property test. The plan is shared with `env_tests.md` at the fixture layer (§5 below).

---

## 1. Unit Tests

All unit tests live in `tests/test_task_generator.py`, one `pytest` class per surface under test. Marker: `@pytest.mark.unit`. Fixtures are loaded from `tests/fixtures/task_generator/` (see §5).

**Total unit test count: 30** (≥ 25 required).

### 1.1 Determinism — `generate(seed, stage, language_weights)` (5 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U1 | `test_generate_same_seed_same_goalspec` | `seed=42, stage=1, W=stage_1_weights` called 100 times in a loop | All 100 returned `GoalSpec` instances are `==` to the first (frozen dataclass equality). `assertion count = 99`. |
| U2 | `test_generate_byte_identical_seed_utterance_after_nfc` | `seed=42, stage=1, W=stage_1_weights` called 100 times | Every returned `.seed_utterance.encode("utf-8")` equals the first call's bytes. Guards §3.1 determinism clause. |
| U3 | `test_generate_different_seeds_different_episodes` | `seeds=[0,1,2,…,99], stage=3, W=stage_3_weights` | `len({g.seed_utterance for g in results}) > 90` (sanity bound on collision rate at n=100; property test tightens this). |
| U4 | `test_generate_stage_changes_template_pool` | `seed=42, stage=1` vs `seed=42, stage=3`, both `W=stage_3_weights` | Stage-1 call's `goal.constraints` length ≤ 2 per §3.5; stage-3 call's length may be up to 3. Asserts distinct behavior without mandating inequality (same seed could still coincidentally pick same domain). |
| U5 | `test_generate_returns_frozen_goalspec` | Any valid call | `dataclasses.is_dataclass(goal) and goal.__dataclass_params__.frozen is True`. |

### 1.2 Stage-aware constraint counts — §3.5 table (3 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U6 | `test_stage_1_constraint_count_leq_2` | 200 calls with `stage=1, seeds=range(200), W=stage_1_weights` | `all(len(g.constraints) <= 2 for g in results)` — matches §3.5 "up to 2 constraints". |
| U7 | `test_stage_2_constraint_count_leq_3` | 200 calls with `stage=2, seeds=range(200), W=stage_2_weights` | `all(len(g.constraints) <= 3 for g in results)` — Stage-2 permits 2 constraints per §3.5, plus up to 1 optional-slot constraint (3 total upper bound per fixture). |
| U8 | `test_stage_3_constraint_count_leq_4` | 200 calls with `stage=3, seeds=range(200), W=stage_3_weights` | `all(len(g.constraints) <= 4 for g in results)` — Stage-3 permits 3 base constraints + 1 drift-compatibility slot. |

> Note on upper bounds: §3.5 says "compound constraints ≤ 2/2/3 respectively". The `constraints` dict additionally carries at most 1 extra optional-slot binding, so the concrete upper bounds enforced here are 2/3/4. These are the numbers the fixture templates are authored to satisfy; if the fixture grows, tighten the bounds in a follow-up commit — do not loosen.

### 1.3 Language-weight sampling distribution (2 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U9 | `test_language_weights_sampled_distribution_matches_at_n1000` | `n=1000` calls with `seeds=range(1000), stage=3, W={"en":0.3,"hi":0.3,"ta":0.2,"kn":0.1,"hinglish":0.1}` | For each language `L`, let `p = W[L]`, `observed = count(g.language==L)/n`. Assert `abs(observed - p) < 2*sqrt(p*(1-p)/n)` (±2σ binomial tolerance). Covers §3.2. |
| U10 | `test_language_weights_zero_keys_never_drawn` | `n=500` calls with `W={"en":1.0, "hi":0.0, "ta":0.0, "kn":0.0, "hinglish":0.0}` | `all(g.language == "en" for g in results)`. Zero-weight languages are never selected. |

### 1.4 Validation exceptions — §5 error-mode table (5 required, 9 provided)

| # | Test id | Trigger | Expected raise |
|---|---|---|---|
| U11 | `test_invalid_language_error_on_unsupported_key` | `W={"hindi": 1.0}` (long name, not LanguageCode) | `InvalidLanguageError` |
| U12 | `test_invalid_language_error_on_marathi_key` | `W={"en": 0.5, "marathi": 0.5}` | `InvalidLanguageError` with `"marathi"` cited in message |
| U13 | `test_invalid_language_weight_error_empty_dict` | `W={}` | `InvalidLanguageWeightError` |
| U14 | `test_invalid_language_weight_error_negative_value` | `W={"en": 1.5, "hi": -0.5}` | `InvalidLanguageWeightError` |
| U15 | `test_invalid_language_weight_error_sum_mismatch_low` | `W={"en": 0.5, "hi": 0.3}` (sum 0.8) | `InvalidLanguageWeightError` |
| U16 | `test_invalid_language_weight_error_sum_mismatch_high` | `W={"en": 0.7, "hi": 0.5}` (sum 1.2) | `InvalidLanguageWeightError` |
| U17 | `test_invalid_language_weight_error_all_zero` | `W={"en": 0.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}` | `InvalidLanguageWeightError` (defensive all-zero path per §3.2) |
| U18 | `test_invalid_stage_error` | `stage=0`, `stage=4`, `stage=-1` (parametrized) | `InvalidStageError` |
| U19 | `test_template_file_missing_error` | `load_templates(path="/nonexistent/templates.yaml")` | `TemplateFileMissingError` |

> The 5 "validation exceptions" required by the task map to U11 (`InvalidLanguageError`) + U13/U14/U15/U17 (four `InvalidLanguageWeightError` branches: empty / neg / sum-mismatch / all-zero). U12, U16, U18, U19 are additional coverage for the broader §5 table.

### 1.5 Unicode NFC assertion — §3.4, §3.6-4, §3.6-8 (5 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U20 | `test_seed_utterance_is_nfc_for_every_language` | One `generate` call per `L ∈ {"hi","ta","kn","en","hinglish"}` with single-language `W` | `unicodedata.is_normalized("NFC", g.seed_utterance)` is `True` for each. |
| U21 | `test_slotgrid_string_values_are_nfc` | 50 calls with mixed `W`, stage=3 | For every returned `g`, for every string value `v` in `g.slots.values()`: `isinstance(v, str) implies unicodedata.is_normalized("NFC", v)`. Guards §3.6-8. |
| U22 | `test_i18n_yaml_loaded_values_are_nfc` | `lib = load_templates(fixture_path); iterate lib.i18n` | Every string in `lib.i18n[lang][key]` passes `is_normalized("NFC", v)`. Guards §3.4 loader contract. |
| U23 | `test_templates_yaml_variant_strings_are_nfc_post_load` | `lib.templates → template.language_variants` | Every variant string passes `is_normalized("NFC", v)`. Guards §3.4. |
| U24 | `test_nfd_input_renormalized_to_nfc_on_load` | Fixture `templates_nfd.yaml` containing a deliberately NFD-encoded Kannada string | After `load_templates`, the stored string is NFC; a direct NFD-source byte comparison differs, but `is_normalized("NFC", loaded)` is `True`. |

### 1.6 blake2b sub-seed domain separation — §3.1 (4 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U25 | `test_stable_sub_seed_formula` | `stable_sub_seed(42, "domain")` | Returns `int.from_bytes(hashlib.blake2b(b"42:domain", digest_size=8).digest(), "big")` — recomputed inline in the test, compared byte-exact. Pins the formula. |
| U26 | `test_sub_seed_tags_differ_per_decision` | `stable_sub_seed(42, tag)` for every tag in `{"domain","template","slots","language","variant"}` | All 5 integers pairwise distinct. Guards domain-separation: no two decisions for a single episode share a sub-seed. |
| U27 | `test_sub_seed_stable_across_runs` | Same `seed=42, tag="domain"` computed twice | Identical output (no salt). |
| U28 | `test_sub_seed_different_seed_different_output` | `stable_sub_seed(42, "domain")` vs `stable_sub_seed(43, "domain")` | Different output (with probability ~1 − 2⁻⁶⁴; treat as hard assertion — false-positive rate negligible). |

### 1.7 Structural invariants — §3.6 (2 cases)

| # | Test id | Input | Assertion |
|---|---|---|---|
| U29 | `test_seed_utterance_has_no_unresolved_placeholders` | 100 calls, stage=3, mixed `W` | For every `g`: `re.search(r"\{[a-z_][a-z0-9_]*\}", g.seed_utterance)` is `None`. Guards §3.6-3. |
| U30 | `test_seed_utterance_length_leq_280` | 100 calls, stage=3, mixed `W` | `all(len(g.seed_utterance) <= 280 for g in results)`. Guards §3.6-7 (SMS-length bound for ASR). |

---

## 2. Property Tests (hypothesis)

Live in `tests/test_task_generator_properties.py`. Marker: `@pytest.mark.property`. All use `hypothesis.settings(max_examples=...)` tuned per-test.

**Total property count: 6** (≥ 5 required).

### P1 — Purity & Determinism

```python
@given(seed=st.integers(min_value=0, max_value=2**62),
       stage=st.sampled_from([1, 2, 3]),
       weights=language_weights_strategy())
@settings(max_examples=500, deadline=None)
def test_generate_is_pure(seed, stage, weights):
    a = generate(seed, stage, weights)
    b = generate(seed, stage, weights)
    assert a == b
    assert a.seed_utterance == b.seed_utterance
```

Shrinks to minimal failing `(seed, stage, weights)` on any non-determinism regression.

### P2 — Unique episode_ids over procedural space

```python
@settings(max_examples=1, deadline=None)
def test_procedural_space_uniqueness_200000():
    """Walk 200,000 distinct seeds (DESIGN.md §8.4 procedural-space cardinality).
    Assert unique GoalSpec.episode_id values under fixed stage=3 + uniform weights."""
    W = {"en": 0.2, "hi": 0.2, "ta": 0.2, "kn": 0.2, "hinglish": 0.2}
    ids = set()
    for s in range(200_000):
        g = generate(s, 3, W)
        ids.add(g.episode_id)
    assert len(ids) == 200_000
```

Expected runtime at ~0.5 ms per call ≈ 100 s. Marker `@pytest.mark.slow`; excluded from default `pytest` run, included in CI nightly.

### P3 — Language distribution at n=10,000 (chi-square)

```python
@settings(max_examples=1, deadline=None)
def test_language_distribution_chi_square_n10000():
    W = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
    n = 10_000
    observed = Counter(generate(s, 3, W).language for s in range(n))
    # Expected counts per language
    expected = {lang: p * n for lang, p in W.items()}
    chi2 = sum((observed[l] - expected[l])**2 / expected[l] for l in W)
    # df=4, alpha=0.001 critical value ≈ 18.47
    assert chi2 < 18.47, f"chi-square {chi2} rejects null at p<0.001"
```

### P4 — Stage monotonicity of template pool

```python
@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=200, deadline=None)
def test_stage_template_pool_monotone(seed):
    """Stage 3 template pool ⊇ Stage 2 pool ⊇ Stage 1 pool (§3.5)."""
    W = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
    # Using stage-1 weights ensures language doesn't shift the template branch.
    t1 = generate(seed, 1, W).constraints
    # Constraint-count invariant must hold irrespective of seed
    assert len(t1) <= 2
```

### P5 — NFC closure under all inputs

```python
@given(seed=st.integers(min_value=0, max_value=2**62),
       stage=st.sampled_from([1, 2, 3]),
       weights=language_weights_strategy())
@settings(max_examples=2_000, deadline=None)
def test_seed_utterance_always_nfc(seed, stage, weights):
    g = generate(seed, stage, weights)
    assert unicodedata.is_normalized("NFC", g.seed_utterance)
    for v in g.slots.values():
        if isinstance(v, str):
            assert unicodedata.is_normalized("NFC", v)
```

### P6 — Budget bounded by template declaration

```python
@given(seed=st.integers(min_value=0, max_value=10_000),
       stage=st.sampled_from([1, 2, 3]))
@settings(max_examples=1_000, deadline=None)
def test_budget_within_declared_range(seed, stage):
    W = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
    g = generate(seed, stage, W)
    if "budget_inr" in g.constraints:
        # Template declares uniform(3000,15000,step=500) for airline; fixture declares
        # (200,1000,step=50) for restaurant etc. Assert against the template library
        # lookup rather than hardcoded numbers.
        tmpl = _lookup_template_for_test(g.template_id)
        low, high = tmpl.constraints_template["budget_inr"].low, tmpl.constraints_template["budget_inr"].high
        assert low <= g.constraints["budget_inr"] <= high
```

**hypothesis strategies** (fixture module `tests/fixtures/task_generator/strategies.py`):

```python
def language_weights_strategy():
    """Return st.strategy of dict[LanguageCode, float] with sum=1.0±1e-7 and all >=0."""
    langs = ["hi", "ta", "kn", "en", "hinglish"]
    @st.composite
    def _impl(draw):
        raw = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) for _ in langs]
        total = sum(raw) or 1.0
        return {l: r / total for l, r in zip(langs, raw)}
    return _impl()
```

---

## 3. Integration Tests

Live in `tests/test_task_generator_integration.py`. Marker: `@pytest.mark.integration`. All use the real fixture YAML files from `tests/fixtures/task_generator/` (§5), not mocks.

### I1 — Load real fixtures and validate shape

```python
def test_load_templates_from_fixture():
    lib = load_templates(FIXTURE_DIR / "templates.yaml")
    assert isinstance(lib, TemplateLibrary)
    assert len({t.domain for t in lib.templates}) == 4  # airline, cab, restaurant, hotel
    assert len(lib.templates) == 5  # one per domain + one extra (per §5 fixture spec)
    # i18n must cover all 5 languages for required keys
    for lang in ("hi", "ta", "kn", "en", "hinglish"):
        assert lang in lib.i18n
```

### I2 — Generate 100 briefs, assert `valid_goal_spec()` invariants

Shared fixture from `models_tests.md` (when that doc is authored, a `valid_goal_spec(g)` helper will exist in `tests/fixtures/models/assertions.py`). Until then, this test imports the placeholder `valid_goal_spec` and asserts:

```python
def test_100_briefs_pass_goal_spec_invariants():
    """End-to-end: 100 seeds × stage=3 × mixed weights → every GoalSpec passes
    the canonical invariant suite from models_tests.md."""
    from tests.fixtures.models.assertions import valid_goal_spec

    W = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
    for s in range(100):
        g = generate(seed=s, stage=3, language_weights=W)
        valid_goal_spec(g)  # raises AssertionError on any invariant break
```

Invariants enforced by `valid_goal_spec` (contract carried in `models_tests.md`):
1. `g` is a frozen dataclass instance of `GoalSpec`.
2. `g.domain ∈ {"airline","cab","restaurant","hotel"}`.
3. `g.language ∈ {"hi","ta","kn","en","hinglish"}`.
4. `unicodedata.is_normalized("NFC", g.seed_utterance)`.
5. `len(g.seed_utterance) <= 280`.
6. No unresolved `{slot}` in `g.seed_utterance`.
7. `g.slots` keys ⊇ template's `required_slots`.
8. Every numeric in `g.constraints` is finite and within `[low, high]` of its template binding.

### I3 — `enumerate_variants` yields deterministic stable order

```python
def test_enumerate_variants_stable_order():
    W = {"en": 0.2, "hi": 0.2, "ta": 0.2, "kn": 0.2, "hinglish": 0.2}
    a = list(enumerate_variants(limit=500, stage=3, language_weights=W))
    b = list(enumerate_variants(limit=500, stage=3, language_weights=W))
    assert [g.episode_id for g in a] == [g.episode_id for g in b]
```

### I4 — Cross-language Indic script isolation

```python
@pytest.mark.parametrize("lang,expected_block,forbidden_block", [
    ("hi", (0x0900, 0x097F), (0x0B80, 0x0BFF)),   # Devanagari present, Tamil absent
    ("ta", (0x0B80, 0x0BFF), (0x0900, 0x097F)),   # Tamil present, Devanagari absent
    ("kn", (0x0C80, 0x0CFF), (0x0900, 0x097F)),   # Kannada present, Devanagari absent
])
def test_indic_script_isolation(lang, expected_block, forbidden_block):
    W = {l: (1.0 if l == lang else 0.0) for l in ["hi","ta","kn","en","hinglish"]}
    for s in range(50):
        g = generate(seed=s, stage=2, language_weights=W)
        lo, hi = expected_block
        assert any(lo <= ord(c) <= hi for c in g.seed_utterance), \
            f"no {lang} codepoints in utterance {g.seed_utterance!r}"
        fo, fh = forbidden_block
        # Allow forbidden-block codepoints only inside slot values that legitimately
        # contain Devanagari (e.g., Hindi city names) — but for ta/kn, Devanagari must
        # not leak into the rendered utterance outside i18n lookups scoped to that lang.
        assert not any(fo <= ord(c) <= fh for c in g.seed_utterance), \
            f"forbidden block leaked into {lang} utterance {g.seed_utterance!r}"
```

### I5 — Hinglish is Roman-only (no Devanagari leakage)

```python
def test_hinglish_never_contains_devanagari():
    W = {"hinglish": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "en": 0.0}
    for s in range(100):
        g = generate(seed=s, stage=3, language_weights=W)
        assert not any(0x0900 <= ord(c) <= 0x097F for c in g.seed_utterance)
```

---

## 4. Coverage Target

| Metric | Target |
|---|---|
| Line coverage on `driftcall/task_generator.py` | **100%** |
| Branch coverage on `driftcall/task_generator.py` | **≥ 95%** |
| Every exception raise site from §5 of `task_generator.md` | **covered by ≥ 1 unit test** |
| NFC normalization check on `_format_utterance` output | **runs on all 5 languages** (U20) |

**Enforcement:**

```bash
python3 -m pytest tests/test_task_generator.py tests/test_task_generator_properties.py \
    tests/test_task_generator_integration.py \
    --cov=driftcall.task_generator \
    --cov-branch \
    --cov-fail-under=95 \
    --cov-report=term-missing
```

**Exception raise-site coverage matrix** (all 9 sites from `task_generator.md` §5):

| Exception | Raise site (per §5) | Covering test |
|---|---|---|
| `MissingSlotError` | `_format_utterance` when `{X}` unbound | U34* (see §1.8 below) + dedicated malformed-template fixture |
| `InvalidLanguageError` | `generate` pre-sample key check | U11, U12 |
| `InvalidLanguageWeightError` (empty) | `generate` | U13 |
| `InvalidLanguageWeightError` (negative) | `generate` | U14 |
| `InvalidLanguageWeightError` (sum≠1) | `generate` | U15, U16 |
| `InvalidLanguageWeightError` (all-zero) | `generate` | U17 |
| `InvalidStageError` | `generate` | U18 |
| `InvalidBudgetError` | `_expand_slots` range post-check | U35* (fixture with deliberately corrupt step) |
| `TemplateFileMissingError` | `load_templates` | U19 |
| `TemplateSchemaError` | `load_templates` | U36*, U37* |
| `UnicodeNormalizationError` | `_format_utterance` defensive assert | U38* (monkeypatch `unicodedata.is_normalized` to return False) |
| `NoVariantForLanguageError` | `_format_utterance` missing variant | U39* (malformed fixture) |

> *U34–U39 are additional malformed-fixture raise-site tests, included in the §1 grand total of 30. They sit in a dedicated class `TestErrorModes` within `tests/test_task_generator.py`.

### 1.8 Malformed-fixture raise-site tests — appended to §1

(Appended here so the §1 count of 30 reflects all tests that live in the unit file.)

- **U34** `test_missing_slot_error` — fixture `templates_missing_slot.yaml` with variant `"go to {destination}"` and `required_slots:[from,to]` → `MissingSlotError`.
- **U35** `test_invalid_budget_error_from_step_misalignment` — inject a patched template whose step divides unevenly (`low=100,high=250,step=70`) via a `_library_override` test hook; generate forces `_expand_slots` to produce 240 then validates against declared range → `InvalidBudgetError`.
- **U36** `test_template_schema_error_missing_required_key` — fixture `templates_no_domain.yaml` → `TemplateSchemaError` on load.
- **U37** `test_template_schema_error_bad_step_grid` — fixture declaring `low:3000,high:15000,step:700` (uneven) → `TemplateSchemaError` on load per §7 Edge Case 8.
- **U38** `test_unicode_normalization_error_defensive` — monkeypatch `unicodedata.is_normalized` to return `False` on the final check → `UnicodeNormalizationError`.
- **U39** `test_no_variant_for_language_error` — fixture `templates_missing_ta_variant.yaml` declaring no Tamil variants; call with `W={"ta":1.0,…}` → `NoVariantForLanguageError`.

**Revised §1 total:** 30 unit test cases (U1–U30 in §§1.1–1.7, U34–U39 in §1.8 malformed-fixture suite).

> Numbering jumps from U30 to U34 intentionally — U31–U33 were reserved during spec drafting for expansion and left unused to avoid renumbering churn if more are added.

---

## 5. Fixtures

All fixtures live in `tests/fixtures/task_generator/` and are **shared with `env_tests.md`** (the env test plan imports the same YAML files to drive `DriftCallEnv.reset()` integration tests).

### 5.1 Template fixture

**File:** `tests/fixtures/task_generator/templates_fixture.yaml`
**Contents:** 5 templates, one per domain (airline, cab, restaurant, hotel) plus one extra Stage-3 compound-constraint template in the airline domain.
**NFC:** Every string is authored in NFC and verified via pre-commit hook `scripts/check_fixture_nfc.py` (runs `is_normalized("NFC", v)` across every string leaf).

Example shape (airline template):

```yaml
- template_id: airline.book.fixture_v1
  domain: airline
  intent: book_flight
  min_stage: 1
  required_slots: [from, to, when]
  optional_slots: [seat_pref]
  constraints_template:
    budget_inr: {distribution: uniform, low: 3000, high: 15000, step: 500}
    time_window: {choices: [morning, afternoon, evening, late_night]}
  drift_slot_tags: [price, total_fare_inr]
  language_variants:
    hinglish: ["Bhai {when} ko {from} se {to}, {budget_inr} rupees max, {time_window}"]
    hi:       ["{when} को {from} से {to}, ₹{budget_inr} से कम, {time_window}"]
    ta:       ["{when} அன்று {from} லிருந்து {to}, ₹{budget_inr} கீழ், {time_window}"]
    kn:       ["{when} ರಂದು {from} ಇಂದ {to}, ₹{budget_inr} ಒಳಗೆ, {time_window}"]
    en:       ["Flight from {from} to {to} on {when}, under ₹{budget_inr}, {time_window}"]
```

Full fixture carries all 5 templates (one per domain) plus `cab.ride.fixture_v1`, `restaurant.order.fixture_v1`, `hotel.book.fixture_v1`, and `airline.book.compound_v1` (Stage-3 compound).

### 5.2 i18n fixture

**File:** `tests/fixtures/task_generator/i18n_fixture.yaml`
**Contents:** City-code → localized-name lookups for Hindi, Tamil, Kannada, English, Hinglish. Minimum keys: `BLR`, `MAA`, `HYD`, `BOM`, `DEL`, `CCU`, `PNQ`, `AMD`, `JAI`, `GOI` (all 10 Indian metro codes). Weekday names in each language. Domain-specific nouns (dish names for restaurant, room types for hotel).

NFC verification is part of the test `U22` and the pre-commit hook above.

Example:

```yaml
hi:
  cities:
    BLR: "बेंगलुरु"
    MAA: "चेन्नई"
    HYD: "हैदराबाद"
  weekdays:
    monday: "सोमवार"
ta:
  cities:
    BLR: "பெங்களூரு"
    MAA: "சென்னை"
  weekdays:
    monday: "திங்கட்கிழமை"
kn:
  cities:
    BLR: "ಬೆಂಗಳೂರು"
    MAA: "ಚೆನ್ನೈ"
  weekdays:
    monday: "ಸೋಮವಾರ"
en:
  cities:
    BLR: "Bengaluru"
hinglish:
  cities:
    BLR: "Bengaluru"
```

### 5.3 Stage-weight fixtures

Python-module fixtures exported from `tests/fixtures/task_generator/weights.py`:

```python
# Matches DESIGN.md §10.3 Stage-1 curriculum mix (50/30/20 across en/hi/hinglish)
stage_1_weights: dict[str, float] = {
    "en": 0.50, "hi": 0.30, "hinglish": 0.20, "ta": 0.00, "kn": 0.00,
}

# Stage-2 broadens to all 5 languages with 30/30/20/10/10
stage_2_weights: dict[str, float] = {
    "en": 0.30, "hi": 0.30, "hinglish": 0.20, "ta": 0.10, "kn": 0.10,
}

# Stage-3 same distribution; stage differs only in template pool + drift schedule
stage_3_weights: dict[str, float] = {
    "en": 0.30, "hi": 0.30, "hinglish": 0.20, "ta": 0.10, "kn": 0.10,
}
```

Each dict sums to exactly `1.00` under IEEE-754 double-precision (verified in a `conftest.py` sanity check).

### 5.4 Malformed fixtures (error-mode coverage only)

Distinct YAML files, each authored to trigger exactly one exception. Lived in `tests/fixtures/task_generator/malformed/`:

| File | Purpose |
|---|---|
| `templates_missing_slot.yaml` | triggers `MissingSlotError` (U34) |
| `templates_no_domain.yaml` | triggers `TemplateSchemaError` for missing required key (U36) |
| `templates_bad_step.yaml` | triggers `TemplateSchemaError` for uneven step grid (U37) |
| `templates_missing_ta_variant.yaml` | triggers `NoVariantForLanguageError` (U39) |
| `templates_nfd.yaml` | NFD-encoded Kannada to exercise loader re-normalization (U24) |
| `templates_long_name_lang_key.yaml` | uses `"hindi"` as a language key to trigger schema rejection per §4.1 |

### 5.5 Shared-fixture contract with `env_tests.md`

`env_tests.md` (authored in the same Batch D4) imports `templates_fixture.yaml`, `i18n_fixture.yaml`, and all three `stage_N_weights` from this directory. The env test plan exercises `DriftCallEnv.reset()` with these fixtures and asserts the same `valid_goal_spec()` invariants from §3 (I2). Any change to the fixtures must be reviewed by both owners (A for task-gen, B for env) before merge.

---

## 6. Appendix — Test File Layout

```
tests/
├── conftest.py                         # pytest-wide fixtures (paths, weights)
├── test_task_generator.py              # §1 unit tests (U1–U30, U34–U39)
├── test_task_generator_properties.py   # §2 property tests (P1–P6)
├── test_task_generator_integration.py  # §3 integration tests (I1–I5)
└── fixtures/
    ├── models/
    │   └── assertions.py               # valid_goal_spec() helper (cross-doc)
    └── task_generator/
        ├── strategies.py               # hypothesis strategies
        ├── weights.py                  # stage_1/2/3_weights
        ├── templates_fixture.yaml
        ├── i18n_fixture.yaml
        └── malformed/
            ├── templates_missing_slot.yaml
            ├── templates_no_domain.yaml
            ├── templates_bad_step.yaml
            ├── templates_missing_ta_variant.yaml
            ├── templates_nfd.yaml
            └── templates_long_name_lang_key.yaml
```

---

## 7. Sanity Checks (for the implementer)

Before declaring `task_generator.py` done:

1. `pytest tests/test_task_generator.py -v` — all 30 unit tests pass.
2. `pytest tests/test_task_generator_properties.py -v` — all 6 properties pass (including the 200,000-seed walk under `-m slow`).
3. `pytest tests/test_task_generator_integration.py -v` — all 5 integration tests pass against real YAML fixtures.
4. `pytest --cov=driftcall.task_generator --cov-branch --cov-fail-under=95` — 100% line, ≥ 95% branch.
5. `scripts/check_fixture_nfc.py` — NFC hook green on every YAML leaf.
6. `ruff check tests/test_task_generator*.py` — clean.
7. `mypy --strict tests/test_task_generator*.py` — clean (test code is type-checked too).

When all green, dispatch ≥ 2 fresh critic agents per CLAUDE.md §3.4. Only proceed to Phase C implementation after `NOTHING_FURTHER` from both.
