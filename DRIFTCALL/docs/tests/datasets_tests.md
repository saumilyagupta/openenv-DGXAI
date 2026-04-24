# datasets_tests.md — Test Plan for `docs/modules/datasets.md`

**Owner:** Person B (Rewards & Tests), co-authored with Person C (Training & Data)
**Target module:** `DRIFTCALL/docs/modules/datasets.md` (final sealed)
**Implements coverage for:** DESIGN.md §8 (§§8.1, 8.2, 8.3, 8.4, 8.5, 8.6) and CLAUDE.md §3.1
**Frameworks:** `pytest`, `hypothesis`, `pytest-cov`
**Status:** DRAFT — pending ≥ 1 fresh critic round (test-plan gate is lighter per CLAUDE.md §3.2 Batch D4)

---

## 0. Scope & Non-goals

`datasets.md` specifies four on-disk data layers (L1 templates/i18n, L2 drift-patterns + api-schemas, L3 audio manifest, L4 SFT warmup) plus a one-shot HF Hub publication contract. Every loader is a lazy singleton that NFC-normalizes on read, validates against a frozen dataclass schema, and raises a typed `DatasetError` subclass on any shape / license / lineage / leak violation.

This plan covers:

1. **Constructibility + immutability** of every frozen dataclass declared in datasets.md §4 (§4.1 – §4.7).
2. **Canonical JSON serialization** — byte-identical output of `json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",",":"))` across Python / libc versions (datasets.md §3.1 invariant #6).
3. **Lineage hash triple** — every `BriefRow` carries `catalogue_hash` / `templates_sha256` / `i18n_sha256`, and any mismatch at eval-load raises `CatalogueHashMismatchError` (datasets.md §3.5 invariant #9, §5).
4. **Size + contents invariants** — `TemplateLibrary` has exactly 20 templates at v1.0, `DriftPatternLibrary` has exactly 20 patterns, `APISchemaRegistry` exactly 14 schemas over 5 domains (datasets.md §3.5 invariants #2 / #3 / #4).
5. **License bundle integrity** — root `LICENSE` contains the full verbatim Apache-2.0 text (byte length ≥ 11 000, canonical header string present, SHA pinned in fixture), `LICENSES.md` markdown table parses (datasets.md §3.4).
6. **Audio manifest provenance** — `AudioClip.source` only accepts the single `Literal["real_indicvoices_r"]`; the string `"synth_kokoro"` is rejected at dataclass-construction time (datasets.md §4.5).
7. **Publication determinism** — `random.Random(20260425).sample(range(0, 20_000_000), 15_000)` is byte-identical across re-runs; val seeds are `list(range(20_000_000, 20_000_500))`; train ∩ val = ∅ (datasets.md §2.4, §3.1, §3.5 invariants #5 / #6).
8. **SFT restart recovery** — `training/sft_generator.py` appends one canonical-JSON line + `os.fsync(fd)` per trajectory, rehydrates `generation_batch_id` on restart, emits monotonic `generation_index`, and raises `PartialSFTCorpusError` on final-count mismatch (datasets.md §4.6, §7 edge 11).
9. **License-cache FTS5 schema** — `scripts/build_license_cache.py` produces `data/.license_cache/{sgd,mtop}.idx` with `CREATE VIRTUAL TABLE licensed_text USING fts5(chunk_text, source_id);` and 5-gram tokenization (datasets.md §9.1).
10. **HF dataset-card frontmatter** — `README.md` YAML frontmatter parses via the HF `datasets` loader (datasets.md §8.6).

Every test below maps to one numbered clause in `datasets.md`. Clause references are embedded in each test docstring as `datasets.md §X.Y` / `datasets.md §7 edge N`.

---

## 1. Unit tests

All unit tests live in `DRIFTCALL/tests/data/`. Import surface under test:

```python
from driftcall.data.models import (
    TemplateLibrary, I18nLibrary,
    DriftPatternLibrary, APISchemaRegistry, APISchema,
    AudioManifest, AudioClip,
    SFTCorpus, SFTTrajectory,
    BriefRow,
)
from driftcall.data.loaders import (
    load_templates, load_i18n,
    load_drift_patterns, load_api_schemas,
    load_audio_manifest, load_sft_corpus,
)
from driftcall.data.errors import (
    DatasetError, DatasetFileMissingError, MalformedYAMLError, MalformedJSONError,
    DatasetSchemaError, UnknownLanguageKeyError, LicenseConflictError,
    TrainValLeakError, DriftPatternOrphanError, ChecksumMismatchError,
    UnicodeNFDError, PIIDetectedError, DuplicateDriftPatternIdError,
    CatalogueHashMismatchError, PartialSFTCorpusError,
)
from training.data_export import canonical_dumps, sample_train_seeds, val_seeds
from training.sft_generator import append_trajectory, resume_batch
from scripts.build_license_cache import build_index, FTS5_SCHEMA_DDL
```

Fixtures (§5) come from `tests/data/conftest.py` and `tests/conftest.py`.

### 1.1 `BriefRow` — frozen dataclass + 13-field contract

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U1 | `test_brief_row_has_exactly_thirteen_fields` | `len(dataclasses.fields(BriefRow)) == 13`; field names equal the ordered tuple `("episode_id","seed","stage","language","domain","template_id","goal","drift_schedule","catalogue_hash","templates_sha256","i18n_sha256","generator_version","created_ts_ist")`. | datasets.md §4.7 |
| U2 | `test_brief_row_is_frozen` | Building from `brief_row_happy` fixture, every attempted assignment (`row.seed = 7`, `row.episode_id = "x"`, etc.) raises `dataclasses.FrozenInstanceError`. Parametrized over all 13 fields. | datasets.md §3.5 invariant #1 (immutability), §4.7 |
| U3 | `test_brief_row_happy_construct_roundtrip` | `brief_row_happy` constructs; `dataclasses.asdict(row)` returns a dict with 13 keys matching the spec; all string fields are NFC. | datasets.md §4.7 |
| U4 | `test_brief_row_missing_required_field_raises` | `BriefRow()` raises `TypeError` (no defaults — every field is required). Parametrized: supplying 12 of 13 fields also raises. | datasets.md §4.7 |
| U5 | `test_brief_row_stage_literal_enforced` | `BriefRow(..., stage=4)` is statically illegal; at runtime a `DatasetSchemaError` is raised by `load_*` on a `stage` value ∉ `{1,2,3}`. | datasets.md §4.7 |
| U6 | `test_brief_row_domain_literal_enforced` | A `domain="payment"` row is rejected at load time (`BriefRow.domain` is the 4-value primary-domain literal — payment is L2-only and does not appear in publication). | datasets.md §4.7, §3.5 invariant #4 |
| U7 | `test_brief_row_created_ts_ist_must_carry_plus0530_offset` | `load_briefs("train/briefs.jsonl")` rejects a row whose `created_ts_ist` does not end in `+05:30`. | datasets.md §4.7 |

### 1.2 Canonical JSON ordering

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U8 | `test_canonical_dumps_sorts_keys` | `canonical_dumps({"b":1,"a":2}) == '{"a":2,"b":1}'`; output contains no spaces; no trailing newline. | datasets.md §3.1 canonical-JSON block |
| U9 | `test_canonical_dumps_preserves_devanagari` | `canonical_dumps({"city":"बेंगलुरु"})` contains the literal Devanagari bytes (UTF-8), NOT `क…` escapes. Exact bytes asserted with `==`. | datasets.md §3.1 (`ensure_ascii=False`) |
| U10 | `test_canonical_dumps_exact_separators` | The serialized form of `{"a":1,"b":2}` equals `b'{"a":1,"b":2}'` byte-for-byte; no whitespace between `,` / `:` and neighbours. | datasets.md §3.1 canonical-JSON block |
| U11 | `test_canonical_dumps_brief_row_matches_golden` | `canonical_dumps(asdict(brief_row_happy))` equals the golden line in §8.5 of datasets.md byte-for-byte. | datasets.md §8.5 |
| U12 | `test_canonical_dumps_is_idempotent` | `canonical_dumps(json.loads(canonical_dumps(row)))` == `canonical_dumps(row)` for 100 random fixture perturbations (fuzzed via hypothesis — see §2). | datasets.md §3.1, §3.5 invariant #6 |

### 1.3 Lineage hashes (`catalogue_hash`, `templates_sha256`, `i18n_sha256`)

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U13 | `test_catalogue_hash_matches_drifts_yaml_bytes` | `catalogue_hash == hashlib.sha256(Path("data/drift_patterns/drifts.yaml").read_bytes()).hexdigest()`; length 64 hex chars; lowercase. | datasets.md §4.7, §3.5 invariant #9 |
| U14 | `test_templates_sha256_matches_templates_yaml_bytes` | `templates_sha256 == sha256(templates.yaml)` byte-for-byte. | datasets.md §4.7 |
| U15 | `test_i18n_sha256_matches_i18n_yaml_bytes` | `i18n_sha256 == sha256(i18n.yaml)` byte-for-byte. | datasets.md §4.7 |
| U16 | `test_catalogue_hash_mismatch_raises_at_load` | Given `brief_row_happy` serialized with `catalogue_hash="deadbeef…"` (wrong), `load_briefs(path)` raises `CatalogueHashMismatchError` naming the offending field(s). | datasets.md §3.5 invariant #9, §5 (`CatalogueHashMismatchError`) |
| U17 | `test_hash_computation_is_stable_across_reruns` | Computing the three hashes twice in the same process returns identical values; computing across two subprocesses (via `subprocess.check_output`) also identical. | datasets.md §3.5 invariant #6 |

### 1.4 `AudioClip.source` excludes synth

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U18 | `test_audio_clip_source_accepts_real_only` | `AudioClip(..., source="real_indicvoices_r", ...)` constructs; `AudioClip(..., source="synth_kokoro", ...)` raises `DatasetSchemaError` at load. | datasets.md §4.5 |
| U19 | `test_audio_manifest_rejects_synth_row` | A `MANIFEST.jsonl` line containing `"source":"synth_kokoro"` causes `load_audio_manifest` to raise `DatasetSchemaError("source must be 'real_indicvoices_r'")`. | datasets.md §4.5 |
| U20 | `test_audio_manifest_duration_upper_bound` | `AudioClip(..., duration_s=20.01)` loads raise `DatasetSchemaError`; `20.00` OK (DESIGN.md §9 upper bound). | datasets.md §4.5 |

### 1.5 `TemplateLibrary.size == 20` at v1.0

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U21 | `test_template_library_size_is_exactly_twenty_at_v1` | `len(load_templates().templates) == 20`; `len(templates) % 5 == 0`; `generator_version.startswith("driftcall-1.0")`. | datasets.md §3.5 invariant #4, §4.1 |
| U22 | `test_template_library_four_domains_five_each` | Grouped by `template.domain`, exactly 4 primary domains are present (airline, cab, restaurant, hotel), 5 templates each. Payment is NOT a primary-domain template owner. | datasets.md §4.1, §3.5 invariant #4 |
| U23 | `test_template_library_every_language_every_template` | For every template, `set(template.language_variants.keys()) == {"hi","ta","kn","en","hinglish"}`; missing key raises `DatasetSchemaError` at load. | datasets.md §3.5 invariant #4, §7 edge 1 |
| U24 | `test_template_library_future_version_monotonic_growth` | Synthesize a mock `templates.yaml` with 25 entries tagged `generator_version="driftcall-1.1.0"`; `load_templates` accepts it (monotonic growth invariant holds: `len >= 20` and `len % 5 == 0`). | datasets.md §4.1 |

### 1.6 `LICENSES.md` schema parse + `LICENSE` verbatim Apache-2.0

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U25 | `test_root_license_byte_length_at_least_11000` | `Path("LICENSE").read_bytes().__len__() >= 11_000`. | datasets.md §3.4 |
| U26 | `test_root_license_contains_apache_canonical_header` | The bytes `b"Apache License\n                           Version 2.0, January 2004"` appear at the top of `LICENSE`. | datasets.md §3.4 |
| U27 | `test_root_license_sha256_pinned` | `sha256(LICENSE bytes) == APACHE_2_0_CANONICAL_SHA` (pinned constant `8a0d778…`; exact value locked in `tests/data/fixtures/license_hashes.py`). | datasets.md §3.4 |
| U28 | `test_audio_licenses_md_embeds_full_apache_text` | `data/audio/LICENSES.md` byte length ≥ 11 000 AND contains the canonical Apache header. | datasets.md §3.4 |
| U29 | `test_sft_licenses_md_embeds_full_apache_text` | Same check for `data/sft_warmup/LICENSES.md`. | datasets.md §3.4 |
| U30 | `test_licenses_md_table_schema_parses` | The markdown table in each `LICENSES.md` parses with columns in exact order `["utterance_id"|"trajectory_id", "upstream_source", "upstream_license", "attribution_required", "notes"]`; every row has 5 cells; `attribution_required ∈ {"yes","no"}`. | datasets.md §3.4 |

### 1.7 Seed selection — deterministic, byte-identical

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U31 | `test_train_seed_sampling_is_deterministic` | `sample_train_seeds() == random.Random(20260425).sample(range(0, 20_000_000), 15_000)`; re-running the function twice yields identical lists (element-wise equal + ordering identical). | datasets.md §2.4 |
| U32 | `test_train_seed_count_is_fifteen_thousand` | `len(sample_train_seeds()) == 15_000`; all elements in `[0, 20_000_000)`; no duplicates. | datasets.md §2.4, §3.1 |
| U33 | `test_val_seeds_are_exact_contiguous_slice` | `val_seeds() == list(range(20_000_000, 20_000_500))`; `len == 500`; first == 20_000_000; last == 20_000_499. | datasets.md §2.4 |
| U34 | `test_train_val_disjoint` | `set(sample_train_seeds()).isdisjoint(set(val_seeds()))`; assert raises `TrainValLeakError` if injected seed `20_000_050` is spliced into train output. | datasets.md §3.5 invariant #5, §7 edge 5 |

### 1.8 SFT restart recovery + `PartialSFTCorpusError`

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U35 | `test_sft_append_one_line_fsyncs` | `append_trajectory(fd, traj)` writes exactly one canonical-JSON line ending `\n` and invokes `os.fsync(fd)` once per call (verified via `unittest.mock.patch("os.fsync")`). | datasets.md §4.6 |
| U36 | `test_sft_generation_batch_id_monotonic_within_batch` | Batch generates N=10 trajectories; all carry identical `generation_batch_id` (uuid4); `generation_index` values are `[0..9]` strictly monotonic and contiguous. | datasets.md §4.6 |
| U37 | `test_sft_restart_rehydrates_batch_id` | Given `trajectories.jsonl` pre-populated with 3 rows (batch_id `B`), call `resume_batch(path)`; returned `(batch_id, next_index) == (B, 3)`. New rows appended reuse `B`. | datasets.md §4.6, §7 edge 11 |
| U38 | `test_sft_partial_corpus_error_on_resume_count_mismatch` | Corpus file has 298 rows with `target_count=300` recorded in corpus metadata; `load_sft_corpus` raises `PartialSFTCorpusError("expected 300, got 298")`. | datasets.md §4.6, §5 (`PartialSFTCorpusError`), §7 edge 11 |
| U39 | `test_sft_generator_final_count_validation` | `training/sft_generator.py` run with `--target-count 5` but Sarvam-M drops 1 response → generator raises `PartialSFTCorpusError` post-loop (not silently). | datasets.md §4.6 |

### 1.9 License-cache FTS5 schema

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U40 | `test_license_cache_schema_ddl_is_exact` | `FTS5_SCHEMA_DDL == "CREATE VIRTUAL TABLE licensed_text USING fts5(chunk_text, source_id);"` — byte-for-byte. | datasets.md §9.1 |
| U41 | `test_license_cache_uses_5gram_tokenizer` | Introspect sqlite `PRAGMA fts5_integrity_check` + the `CREATE VIRTUAL TABLE` statement records `tokenize = 'unicode61'` with 5-gram config; `build_index(tokenizer="trigram")` raises `ValueError`. | datasets.md §9.1 |
| U42 | `test_license_cache_built_is_read_only_in_ci` | In CI mode (`DRIFTCALL_CI=1`), invoking `build_index` raises `RuntimeError("license cache is read-only in CI")`. | datasets.md §9.1 |

### 1.10 `README.md` YAML frontmatter — HF dataset loader parse

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U43 | `test_readme_frontmatter_parses_with_pyyaml` | `yaml.safe_load(frontmatter_block)` yields a dict with keys `{license, language, size_categories, task_categories, pretty_name, configs, dataset_info}`; `license == "apache-2.0"`; `language == ["hi","ta","kn","en"]`. | datasets.md §8.6 |
| U44 | `test_readme_frontmatter_loads_via_hf_datasets` | `datasets.load_dataset(str(bundle_dir))` (HF loader) returns a `DatasetDict` with `{"train","val"}` splits; `train.num_rows == 15_000`; `val.num_rows == 500`. Skipped if `datasets` not installed. | datasets.md §8.6 |
| U45 | `test_readme_frontmatter_features_flat_columns_only` | The `features` block lists only the 6 flat columns `{episode_id, seed, stage, language, domain, template_id}`; nested `goal`/`drift_schedule` are NOT pre-declared (auto-inferred). | datasets.md §8.6 |

### 1.11 Miscellaneous spec wiring

| # | Test name | Asserts | Maps to |
|---|---|---|---|
| U46 | `test_load_drift_patterns_count_equals_twenty` | `len(load_drift_patterns().patterns) == 20`; every `drift_type ∈ {"schema","policy","tnc","pricing","auth"}`. | datasets.md §3.5 invariant #2 |
| U47 | `test_load_api_schemas_count_equals_fourteen_across_five_domains` | `APISchemaRegistry` reports exactly 14 schemas keyed `{airline:{v1,v2,v3}, cab:{v1,v2,v3}, restaurant:{v1,v2,v3}, hotel:{v1,v2,v3}, payment:{v1,v2}}`. | datasets.md §3.5 invariant #3 |
| U48 | `test_drift_pattern_orphan_raises` | YAML with `from_version="v5"` (nonexistent) raises `DriftPatternOrphanError`. | datasets.md §5, §7 edge 6 |
| U49 | `test_duplicate_drift_pattern_id_raises` | YAML with two entries sharing `id: airline.price_rename` raises `DuplicateDriftPatternIdError` citing both line numbers. | datasets.md §5, §7 edge 10 |
| U50 | `test_nfc_normalization_applied_at_load` | A templates YAML authored with NFD Kannada weekday normalizes to NFC on load; `unicodedata.is_normalized("NFC", v) is True` for every loaded string. | datasets.md §3.5 invariant #1, §7 edge 2 |
| U51 | `test_pii_10_digit_run_raises` | An authored string containing `"9876543210"` outside IATA / timestamp contexts raises `PIIDetectedError`. | datasets.md §3.5 invariant #8, §3.1 |
| U52 | `test_license_header_missing_raises` | A YAML file without the `# SPDX-License-Identifier:` leading comment raises `DatasetSchemaError` at load. | datasets.md §3.5 invariant #7 |
| U53 | `test_audio_manifest_sha256_mismatch_raises` | Corrupt a wav byte; `load_audio_manifest` raises `ChecksumMismatchError` citing expected vs actual. | datasets.md §5, §7 edge 8 |
| U54 | `test_sft_trajectory_val_seed_raises` | `SFTTrajectory(goal_seed=20_000_042, …)` on load raises `TrainValLeakError`. | datasets.md §5, §7 edge 9 |
| U55 | `test_loader_is_singleton_per_path` | `load_templates()` twice returns the same object by identity (`is`); called with a different `path=` yields a distinct instance cached separately. | datasets.md §3.2 |

**Total unit tests: 55** (target ≥ 35).

---

## 2. Property tests

All property tests live in `DRIFTCALL/tests/data/test_properties.py` using `hypothesis`.

| # | Property | Strategy | Maps to |
|---|---|---|---|
| P1 | **Byte-identical re-runs of `data_export`.** For any seed `s == 20260425`, two invocations of `data_export.main(seed=s)` produce byte-identical `train/briefs.jsonl` + `val/briefs.jsonl` (SHA-256 hashes match). | Fixed seed + hypothesis-generated minor perturbations (run order, tmpdir path). | datasets.md §3.5 invariant #6 |
| P2 | **`BriefRow` is frozen.** For any `BriefRow` instance generated by `brief_row_strategy()`, assigning to any of its 13 fields raises `FrozenInstanceError`. Hypothesis enumerates field name and value type. | `st.builds(BriefRow, …)` + `st.sampled_from(fields_of(BriefRow))`. | datasets.md §3.5 invariant #1, §4.7 |
| P3 | **Seed-range disjointness.** For any pair `(t, v)` where `t ∈ [0, 20_000_000)` and `v ∈ [20_000_000, 20_000_500)`, `t != v` and both sets generated by the spec are disjoint. Hypothesis samples 10 000 pairs. | `st.tuples(st.integers(min_value=0, max_value=19_999_999), st.integers(min_value=20_000_000, max_value=20_000_499))`. | datasets.md §3.5 invariant #5, §2.4 |
| P4 | **Canonical JSON determinism under key permutation.** For any dict `d` and any permutation `d'` of its keys, `canonical_dumps(d) == canonical_dumps(d')` byte-for-byte. | `st.dictionaries(st.text(), st.one_of(st.integers(), st.text()))` + `.map(shuffle_keys)`. | datasets.md §3.1, §3.5 invariant #6 |
| P5 | **NFC idempotence.** For any string `s`, `nfc(nfc(s)) == nfc(s)`; `load_templates` applied twice yields the same library by hash. | `st.text(alphabet=st.characters(min_codepoint=0x0900, max_codepoint=0x0DFF))` — Devanagari + Tamil + Kannada ranges. | datasets.md §3.5 invariant #1, §7 edge 2 |
| P6 | **Catalogue-hash round-trip.** For any `brief_row_happy`-shaped row with a synthetic YAML `y`, `sha256(y)` computed by `load_*` equals `hashlib.sha256(y.encode("utf-8")).hexdigest()` (i.e., loader uses the same algorithm as the spec). | `st.text(alphabet=st.characters(whitelist_categories=("Ll","Lu","Nd")))`. | datasets.md §3.5 invariant #9, §4.7 |

**Total properties: 6** (target ≥ 5).

---

## 3. Integration tests

All integration tests live in `DRIFTCALL/tests/data/test_integration.py`. Marked `@pytest.mark.integration` — run by CI and by `pytest -m integration` locally.

| # | Test name | Scenario | Maps to |
|---|---|---|---|
| I1 | `test_full_data_export_writes_train_and_val_jsonl` | Invoke `training/data_export.main(--out-train, --out-val, --n-train 15000, --n-val 500, --seed 20260425)` in a tmpdir; assert both files exist, each line parses as canonical JSON, `train` has 15 000 rows, `val` has 500, and the set of `(seed)` values across both splits equals `set(train_seeds) ∪ set(val_seeds)`. | datasets.md §2.4 |
| I2 | `test_full_data_export_round_trip_hashes` | Re-run I1 a second time in a separate tmpdir; assert `sha256(train/briefs.jsonl)` and `sha256(val/briefs.jsonl)` match the first-run hashes byte-for-byte. | datasets.md §3.5 invariant #6 |
| I3 | `test_hf_upload_dry_run` | Run `hf upload <org>/driftcall-indic-briefs data/publication/ . --repo-type dataset --dry-run` (via subprocess). Assert exit 0; stdout lists exactly the files enumerated in datasets.md §2.1 publication tree; no network request fires (use `HF_HUB_OFFLINE=1`). | datasets.md §2.4 |
| I4 | `test_round_trip_load_json_dumps_load` | For every row in `train/briefs.jsonl`: `row_dict_a = json.loads(line)`; `line_b = canonical_dumps(row_dict_a)`; `row_dict_b = json.loads(line_b)`; assert `row_dict_a == row_dict_b` AND `line_b == line.rstrip("\n")`. 15 000 rows checked; fails on first discrepancy. | datasets.md §3.1, §3.5 invariant #6 |
| I5 | `test_verbatim_contamination_detector_sgd_mtop` | Build `corpus_snapshot_20260425` license cache (sqlite FTS5 over SGD + MTOP exports). For every `seed_utterance` in `train/briefs.jsonl` + `val/briefs.jsonl`, query the FTS5 index for a ≥ 10-token verbatim suffix match. Assert zero hits. Any hit → `LicenseConflictError` is raised by the CI wrapper. | datasets.md §3.4, §7 edge 3, §9.1 |
| I6 | `test_loader_cross_consistency_templates_vs_drift_patterns` | Load both libraries; assert every primary-domain pattern's `mutation` keys ⊆ union of `drift_slot_tags` across its domain's templates (the two transversal payment-auth patterns exempted). | datasets.md §3.5 invariant #4 |
| I7 | `test_loader_cross_consistency_drifts_vs_api_schemas` | For every pattern, `from_version` and `to_version` exist under `data/api_schemas/<pattern.domain>/`. | datasets.md §3.3 |
| I8 | `test_eval_load_raises_on_catalogue_hash_mismatch` | Publish a bundle with current catalogue; mutate `drifts.yaml` by one byte; invoke consumer-side `load_briefs(path)` → raises `CatalogueHashMismatchError` before any row is consumed. | datasets.md §3.5 invariant #9, §5 |
| I9 | `test_sft_generator_restart_end_to_end` | Generate 5 trajectories; `kill -9` the process after row 3 (simulated via `subprocess` + `signal.SIGKILL`); restart; assert final file has 5 rows, a single shared `generation_batch_id`, `generation_index` == `[0,1,2,3,4]` strictly, and no `PartialSFTCorpusError`. | datasets.md §4.6, §7 edge 11 |
| I10 | `test_bundle_immutability_after_publish` | Publish v1.0; attempt to re-publish without changes; assert `sha256(train/briefs.jsonl)` matches v1.0. Mutate one byte in an authored template → re-publication fails CI with `DatasetSchemaError` (version bump required). | datasets.md §3.5 invariant #10, §2.4 |

**Total integration tests: 10.**

---

## 4. Coverage target

**100% line + 95% branch** on:

- `driftcall/data/models.py` (dataclass definitions — trivial to hit 100%)
- `driftcall/data/loaders.py` (every loader, every validation branch, every error raise)
- `driftcall/data/errors.py` (every `DatasetError` subclass constructed in at least one test)
- `training/data_export.py` (seed sampling, canonical dumps, write path, disjointness assertion)
- `training/sft_generator.py` (append + fsync, batch-id rehydration, partial-count validation, Sarvam-M error paths mocked)
- `scripts/build_license_cache.py` (FTS5 schema DDL, 5-gram tokenizer wiring, CI read-only guard)

**Branch coverage ≥ 95%** — every error-mode `if` / `raise` pair is exercised. The remaining 5% allowance covers unreachable `else` branches defensively guarding against enum exhaustion (Python has no exhaustive-match static guarantee).

Enforced by:

```bash
python3 -m pytest tests/data/ \
  --cov=driftcall.data \
  --cov=training.data_export \
  --cov=training.sft_generator \
  --cov=scripts.build_license_cache \
  --cov-branch \
  --cov-fail-under=100 \
  --cov-report=term-missing
```

Any PR that drops line coverage below 100% or branch coverage below 95% on these modules fails CI.

---

## 5. Fixtures

Fixtures live in `DRIFTCALL/tests/data/conftest.py` and are shared verbatim with `evaluation_tests.md` and `training_tests.md`. All fixtures are `@pytest.fixture(scope="session")` unless noted; they are pure-read and return frozen dataclasses or bytes.

### 5.1 `brief_row_happy`

A canonical Stage-2 airline-booking `BriefRow` with hinglish `seed_utterance`, drifted at turn 4 via `airline.price_rename`. Matches the JSONL example in datasets.md §8.5 exactly (its `canonical_dumps(asdict(row))` equals the §8.5 golden line byte-for-byte). All three lineage hashes are pinned to the `corpus_snapshot_20260425` fixture (§5.5).

### 5.2 `brief_row_stage3_compound`

A Stage-3 compound row (hotel + payment, bilingual code-switch between `hi` and `en`) with two drift events in `drift_schedule` — `hotel.tnc_change` at turn 3 and `payment.auth_scope_upgrade` at turn 6. Exercises the transversal-payment-auth-exempt branch of datasets.md §3.5 invariant #4.

### 5.3 `manifest_ok`

An `AudioManifest` built from 20 curated `AudioClip` rows (4 per language × 5 languages) pulled from IndicVoices-R — all `source="real_indicvoices_r"`, every `sha256` matching the on-disk WAV under `tests/data/fixtures/audio/real/`, every `duration_s ≤ 20.0`. Used to verify the happy-path of `load_audio_manifest`.

### 5.4 `manifest_with_orphan`

Same as `manifest_ok` but with one row whose `path` references `kn/iv_r_kn_9999.wav` (absent on disk). Drives the `ChecksumMismatchError` + `DatasetFileMissingError` error-mode tests.

### 5.5 `corpus_snapshot_20260425`

A byte-frozen snapshot of the entire `data/` tree as of `2026-04-25` (publication seed date). Contains:

- `templates.yaml` + its pinned `sha256`
- `i18n.yaml` + its pinned `sha256`
- `drifts.yaml` + its pinned `sha256`
- All 14 `api_schemas/*/*.json` files
- The 20-row `MANIFEST.jsonl`
- License-cache sqlite files (`.license_cache/sgd.idx`, `.license_cache/mtop.idx`)
- Pinned Apache-2.0 `LICENSE` SHA constant (`APACHE_2_0_CANONICAL_SHA`)

Loaded once per session via `pytest.fixture(scope="session")`. Every lineage-hash test compares against constants frozen in this fixture, so a corpus-file byte mutation anywhere under `data/` causes the hash-pinning tests (U13 – U17, U27) to fail loudly — the intended canary for silent catalogue drift.

This fixture is **shared verbatim** with:

- `DRIFTCALL/docs/tests/evaluation_tests.md` (consumer-side `CatalogueHashMismatchError` coverage)
- `DRIFTCALL/docs/tests/training_tests.md` (GRPO warmup corpus lineage, `sft_warmup/` happy-path + restart coverage)

Authors of those test plans import via `from tests.data.conftest import corpus_snapshot_20260425` rather than re-deriving snapshots — single source of truth prevents divergence.

---

**This test plan implements the full verification surface for `docs/modules/datasets.md`. It does not exist until ≥ 1 fresh critic returns `NOTHING_FURTHER` per CLAUDE.md §3.2 Batch D4. Fixtures are locked to `corpus_snapshot_20260425` and shared with `evaluation_tests.md` + `training_tests.md` — any change here must be mirrored there in the same PR.**
