# datasets — Four-Layer Dataset Strategy + HF Hub Publication

**Module path:** `driftcall/data/` (loaders) + `data/` (on-disk artifacts)
**Owner:** Person C (Training & Data)
**Implements:** DESIGN.md §8 (Dataset Strategy — §§8.1, 8.2, 8.3, 8.4, 8.5, 8.6)
**Consumed by:** `driftcall/task_generator.py` (L1), `driftcall/drift_injector.py` (L2 drift patterns), `driftcall/vendors/*.py` (L2 API schemas), `driftcall/audio/*.py` (L3 audio), `training/train_grpo.py` (L4 SFT warmup).
**Status:** Design spec — no code yet.

---

## 1. Purpose

`datasets` is the **authoring-and-loading contract** for every piece of static data DriftCall depends on. It is *not* a training-data-pipeline module (that is `training.md`) and it does *not* compose rewards (that is `rewards.md`). It does exactly four things:

1. **Defines** the four dataset layers per DESIGN.md §8.2 — task-brief templates, vendor API schemas + drift patterns, voice audio, and the optional SFT warmup corpus — as on-disk files with frozen YAML/JSON schemas.
2. **Loads** each file through a deterministic, lazy, singleton loader that NFC-normalizes all Indic strings at load time (cross-references `docs/modules/task_generator.md` §3.4 invariant #8).
3. **Validates** each file at load time: schema shape, type constraints, license header presence, train/val leakage, and consistency cross-references (e.g., every `drift_slot_tags` token in templates.yaml is targetable by ≥ 1 pattern in drift_patterns.yaml).
4. **Publishes** the public-facing bundle `<team>/driftcall-indic-briefs` to the Hugging Face Hub per DESIGN.md §8.6, packaged from a deterministic `enumerate_variants()` walk (see `docs/modules/task_generator.md` §2.2).

**No file in `data/` is ever written at runtime.** All four layers are authored before Phase C ships. Runtime only reads. The only exception is the dataset-packaging script (`training/data_export.py`) which writes `train/briefs.jsonl` + `val/briefs.jsonl` once, pre-publication.

Supervision for GRPO comes from the 5 reward functions (DESIGN.md §8.1) — these files exist to parameterize the environment, not to teach the policy. L4 (SFT warmup) is optional per DESIGN.md §8.2 row 4.

---

## 2. Interface

### 2.1 Directory layout (on disk, shipped inside the env Docker image per DESIGN.md §11.1)

```
data/
├── task_briefs/
│   ├── templates.yaml                # L1 — hand-authored + procedural expansion source (§8.3)
│   └── i18n.yaml                     # L1 — Indic localized strings (cities, weekdays, dish names)
├── drift_patterns/
│   └── drifts.yaml                   # L2 — 20 drift patterns (§6.3, §8.2 row 2)
├── api_schemas/                      # L2 — frozen JSON Schema per vendor per version
│   ├── airline/
│   │   ├── v1.json
│   │   ├── v2.json
│   │   └── v3.json
│   ├── cab/
│   │   ├── v1.json
│   │   ├── v2.json
│   │   └── v3.json
│   ├── restaurant/
│   │   ├── v1.json
│   │   ├── v2.json
│   │   └── v3.json
│   ├── hotel/
│   │   ├── v1.json
│   │   ├── v2.json
│   │   └── v3.json
│   └── payment/
│       ├── v1.json
│       └── v2.json
├── audio/                            # L3 — synthesized + real voice clips (§8.2 row 3, §9)
│   ├── synth/                        # Kokoro-82M output, generated lazily; gitignored
│   │   └── .gitkeep
│   ├── real/                         # AI4Bharat IndicVoices-R held-out subset for pitch demo
│   │   └── MANIFEST.jsonl            # (utterance_id, path, language, license, sha256)
│   └── LICENSES.md                   # per-clip license attribution
└── sft_warmup/                       # L4 — optional Sarvam-M synthesized trajectories (§8.2 row 4)
    ├── trajectories.jsonl            # 200–500 correct rollouts
    └── LICENSES.md
```

**Publication structure** (HF Hub dataset repo `<team>/driftcall-indic-briefs`, DESIGN.md §8.6):

```
driftcall-indic-briefs/
├── README.md                         # model card — provenance, license, stats, reward caveats
├── train/briefs.jsonl                # 15,000 sampled episodes (seed, stage, language_weights, GoalSpec)
├── val/briefs.jsonl                  #    500 held-out episodes — seeds disjoint from train
├── drift_patterns.yaml               # exact copy of data/drift_patterns/drifts.yaml (20 patterns)
├── api_schemas/                      # exact copy of data/api_schemas/
└── LICENSE                           # bundle license (Apache 2.0 by default; see §3.4)
```

### 2.2 Per-file contracts

| File | Format | Authored by | Runtime writer | Schema anchor |
|---|---|---|---|---|
| `data/task_briefs/templates.yaml` | YAML | Hand (20 seeds) | none | `Template` (§4.1 task_generator.md) |
| `data/task_briefs/i18n.yaml` | YAML | Hand | none | `Mapping[LanguageCode, Mapping[str, str]]` |
| `data/drift_patterns/drifts.yaml` | YAML | Hand | none | `DriftPattern` (§4.2 drift_injector.md) |
| `data/api_schemas/<domain>/v<N>.json` | JSON Schema 2020-12 | Hand | none | `APISchema` (§4.4 below) |
| `data/audio/real/MANIFEST.jsonl` | JSONL | Hand (curated from IndicVoices-R) | none | `AudioClipManifest` (§4.5) |
| `data/audio/synth/*.wav` | WAV 16kHz mono | `audio/tts_kokoro.py` (lazy) | `audio/tts_kokoro.py` | n/a — generated |
| `data/sft_warmup/trajectories.jsonl` | JSONL | Sarvam-M via HF Inference (offline) | `training/sft_generator.py` (one-shot) | `SFTTrajectory` (§4.6) |

### 2.3 Loaders (all return frozen dataclasses, all NFC-normalize Indic strings)

```python
from __future__ import annotations
from pathlib import Path
from driftcall.data.models import (
    TemplateLibrary, I18nLibrary,
    DriftPatternLibrary, APISchemaRegistry,
    AudioManifest, SFTCorpus,
)

# L1 — task briefs
def load_templates(path: Path | str = "data/task_briefs/templates.yaml") -> TemplateLibrary: ...
def load_i18n(path: Path | str = "data/task_briefs/i18n.yaml") -> I18nLibrary: ...

# L2 — drift patterns + api schemas
def load_drift_patterns(path: Path | str = "data/drift_patterns/drifts.yaml") -> DriftPatternLibrary: ...
def load_api_schemas(root: Path | str = "data/api_schemas") -> APISchemaRegistry: ...

# L3 — audio manifest (paths + licenses only; actual WAVs resolved on-demand)
def load_audio_manifest(path: Path | str = "data/audio/real/MANIFEST.jsonl") -> AudioManifest: ...

# L4 — optional SFT warmup
def load_sft_corpus(path: Path | str = "data/sft_warmup/trajectories.jsonl") -> SFTCorpus: ...
```

Each loader is implemented as a **module-level lazy singleton** — the first call reads + validates + freezes; subsequent calls return the same instance. Not thread-safe for write (there is no write); safe for concurrent read.

### 2.4 HF Hub publication commands

Packaging runs *once*, pre-event. The script is `training/data_export.py` (see `docs/modules/training.md` for its interface — this module only defines the on-disk shape of what it writes).

**Immutability.** The published bundle is IMMUTABLE after publication. Re-running `hf upload` against the same `data/publication/` tree produces a byte-identical bundle (invariant #6). Adding rows to `val/briefs.jsonl` requires a MINOR-version bump (v1.1, v1.2, …) and a new publication seed; the `train/` split NEVER silently mutates between versions — a version bump either adds disjoint val rows or re-publishes train+val together, never partial mutation of train.

**Seed selection (deterministic, locked).** Train and val seeds are drawn by `training/data_export.py` using these two exact expressions:

```python
import random
# Train: 15,000 seeds sampled without replacement from [0, 20_000_000).
train_seeds = random.Random(20260425).sample(range(0, 20_000_000), 15_000)
# Val: deterministic slice of 500 contiguous seeds in the reserved range.
val_seeds = list(range(20_000_000, 20_000_500))
```

Both lists are byte-identical across re-runs. The publication meta-seed `20260425` is locked; changing it requires a major-version bump and a new repo name or subfolder.

```bash
# Generate the sampled briefs by walking enumerate_variants() (see task_generator.md §2.2)
python3 training/data_export.py \
    --out-train data/publication/train/briefs.jsonl \
    --out-val   data/publication/val/briefs.jsonl \
    --n-train   15000 \
    --n-val     500 \
    --seed      20260425        # frozen publication seed; NOT a training seed

# Copy the static L2 artifacts verbatim
cp  data/drift_patterns/drifts.yaml  data/publication/drift_patterns.yaml
cp -r  data/api_schemas              data/publication/api_schemas

# Upload (see DRIFTCALL/CLAUDE.md §6 command table and huggingface-skills:hf-cli).
# NOTE: `hf` is the modern CLI replacing the deprecated `huggingface-cli`.
hf upload <org>/driftcall-indic-briefs \
    data/publication/ . \
    --repo-type dataset \
    --hf-org <org> \
    --commit-message "v1.0 publication — locked 2026-04-25"
```

The publication seed `20260425` is fixed and recorded in the README. Re-running the script produces a byte-identical bundle (determinism contract inherited from `task_generator.generate` per `docs/modules/task_generator.md` §3.1).

> **Doc-sync flag:** `DRIFTCALL/CLAUDE.md` §6 still lists the deprecated `huggingface-cli upload` command; update that table to `hf upload` in the same PR that lands this doc (captured as Open Question #1 / CLAUDE.md sync item).

---

## 3. Behavior Spec

### 3.1 Authoring conventions

**NFC normalization.** Every string value in every YAML/JSON file is NFC-normalized before it is committed. The loaders re-normalize defensively at load time (invariant #8, `docs/modules/task_generator.md` §3.4). Editors used during authoring (VS Code, vim) must be configured to save NFC — a pre-commit hook (`ruff`-adjacent script) runs `python -c "import unicodedata, sys; ..."` to reject NFD commits.

**License headers.** Every hand-authored file begins with a YAML comment block declaring SPDX identifier, author, year, and upstream attribution if the content is derived from a public dataset (§8.5):

```yaml
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DriftCall Team
# Derived-from: AmazonScience/MASSIVE (intent taxonomy, Apache-2.0)
# See data/LICENSES.md for full attribution chain.
```

JSON files carry the same metadata in a `$comment` field at root (JSON Schema 2020-12 permits `$comment` per RFC 7159 conventions).

**Seed determinism.** Every numeric or stochastic sampling decision in template/drift authoring threads through a fixed seed: the publication seed `20260425`, the template-expansion seed `42`, or the curriculum-language seeds declared in DESIGN.md §10.3. No wall-clock, no `random.random()`, no host-machine entropy.

**No PII.** Authored strings never contain real names, phone numbers, email addresses, booking reference numbers, card PANs, or IP addresses. The `from` / `to` fields use IATA codes; the `pickup` / `drop` fields use fictional neighborhood landmarks. A CI lint (`grep -En '[0-9]{10}' data/`) runs before every commit and fails on any 10-digit run outside the allowed IATA/timestamp contexts.

**Eval-set held out from training.** The 500-episode val set uses seeds drawn from a reserved range (seed ∈ `[20_000_000, 20_000_500)`); training always draws seeds from `[0, 20_000_000)`. The publication script asserts disjointness at write time (see §5 leak detection). The exact seed-selection expressions are specified in §2.4.

**Canonical JSON key ordering.** Every row in `train/briefs.jsonl` and `val/briefs.jsonl` is serialized with:

```python
json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

This is enforced as an invariant precondition for byte-identical re-runs (see §3.5 invariant #6). `ensure_ascii=False` preserves Devanagari / Tamil / Kannada script without `\uXXXX` escaping; `sort_keys=True` canonicalizes key order; `separators=(",", ":")` eliminates whitespace variance across Python/libc versions.

**Per-row data lineage.** Every `BriefRow` carries the full six-tuple `(template_id, seed, stage, language, domain, generator_version)` plus three corpus-version hashes (`catalogue_hash`, `templates_sha256`, `i18n_sha256`). This is enforced as an invariant (§3.5 invariant #9) so that any published row is re-derivable from the triple `(seed, stage, library@hash)` alone.

### 3.2 Lazy singleton loaders

```python
# sketch of the module-level pattern, mirrored in every loader
_LIBRARY: TemplateLibrary | None = None
_LIBRARY_LOCK = threading.Lock()

def load_templates(path: Path | str = "data/task_briefs/templates.yaml") -> TemplateLibrary:
    global _LIBRARY
    if _LIBRARY is None:
        with _LIBRARY_LOCK:
            if _LIBRARY is None:
                _LIBRARY = _load_and_validate_templates(Path(path))
    return _LIBRARY
```

The singleton is **path-keyed** — if a test passes a different `path`, a fresh instance is built (still cached in a per-path dict). Production callers always use the default path.

### 3.3 Schema validation at load time

Each loader does three passes:

1. **YAML/JSON parse.** Failure → `MalformedYAMLError` / `MalformedJSONError` with line/column.
2. **Type + shape validation** against the dataclass schema in §4. Failure → `DatasetSchemaError` naming the offending key.
3. **Cross-file consistency** check (loader-specific):
   - `load_drift_patterns` asserts `pattern.id` values are unique, exactly 20 patterns total, `drift_type ∈ {schema,policy,tnc,pricing,auth}`, and every `from_version`/`to_version` references an existing schema file in `data/api_schemas/<domain>/`.
   - `load_templates` asserts every `drift_slot_tags` token is matched by ≥ 1 `DriftPattern.mutation` key or value (`airline.total_fare_inr` must be targetable, else why tag it).
   - `load_api_schemas` asserts each `v<N>.json` validates as JSON Schema 2020-12 against the meta-schema via `jsonschema.Draft202012Validator.check_schema`.
   - `load_audio_manifest` asserts every referenced `path` exists on disk and its sha256 matches the recorded hash.

Failures here abort env startup; HTTP 503 is served until the data is fixed (mirrors `DriftCatalogueError` handling in `docs/modules/drift_injector.md` §5).

### 3.4 License compatibility check

Per §8.5 the public datasets we reference carry mixed licenses:

| Upstream | License | Redistributable in our bundle? |
|---|---|---|
| AI4Bharat IndicVoices-R | Apache-2.0 | Yes, with attribution |
| MASSIVE (Amazon) | Apache-2.0 | Yes, with attribution |
| Schema-Guided Dialogue (SGD) | CC-BY-SA | Inspiration only — derived schema patterns, not verbatim rows |
| MTOP (Facebook) | MIT-style (see original repo) | Inspiration only — derived Hindi task phrasings, not verbatim rows |
| APIs.guru | CC0 | Yes, no attribution required but recorded |

The bundle license (`LICENSE` at the root of `<team>/driftcall-indic-briefs`) is **Apache-2.0**. Because CC-BY-SA is copyleft-adjacent, we never copy SGD or MTOP rows verbatim — only *inspiration* (intent labels, schema shapes). A CI check enforces that no string in `train/briefs.jsonl` or `val/briefs.jsonl` appears verbatim (≥ 10-token suffix match) in a cached SGD/MTOP export. See §5 for the error mode and §7 edge case #3 for the exact detection rule.

**Full verbatim license text (MANDATORY).** The root `LICENSE` file MUST contain the **full verbatim Apache 2.0 license text** as published at https://www.apache.org/licenses/LICENSE-2.0.txt — NOT a URL, NOT a one-line SPDX identifier, NOT a summary. The same requirement applies to `data/audio/LICENSES.md` and `data/sft_warmup/LICENSES.md` (both must embed the full Apache-2.0 text plus per-clip / per-trajectory attribution rows). CI check `tests/data/test_license_text.py` verifies that the byte length of each `LICENSE` file is ≥ 11,000 (Apache-2.0 full text is ~11,357 bytes) and that the canonical "Apache License / Version 2.0, January 2004" header string is present.

**`LICENSES.md` schema (L3 audio + L4 SFT warmup).** Both `data/audio/LICENSES.md` and `data/sft_warmup/LICENSES.md` follow the same markdown format:

1. A preamble (5–15 lines) naming the bundle and linking back to the root `LICENSE`.
2. The full verbatim Apache-2.0 text (as above).
3. A single markdown table with exactly these columns, one row per clip (L3) or per trajectory (L4):

```markdown
| utterance_id | upstream_source      | upstream_license | attribution_required | notes                         |
|--------------|----------------------|------------------|----------------------|-------------------------------|
| iv_r_kn_0451 | IndicVoices-R        | Apache-2.0       | yes                  | speaker consent verified      |
| sft_00042    | Sarvam-M (synthesis) | Apache-2.0       | no                   | rollout seed 42, stage 2      |
```

For L4 the `utterance_id` column is replaced by `trajectory_id` but the other four columns are identical. Loaders do not parse these tables at runtime; they are human-audit artifacts enforced only by pre-commit schema check `scripts/check_licenses_md.py`.

### 3.5 Invariants (enforced by tests)

1. Every string value in every loaded library is NFC (`unicodedata.is_normalized("NFC", s) == True`).
2. `load_drift_patterns()` returns exactly 20 patterns (matches `docs/modules/drift_injector.md` §4.4 and DESIGN.md §6.3).
3. `load_api_schemas()` returns exactly `{airline:v1,v2,v3 + cab:v1,v2,v3 + restaurant:v1,v2,v3 + hotel:v1,v2,v3 + payment:v1,v2}` = **14 schemas across 5 domains** (matches DESIGN.md §8.6 bundle enumeration and §5 vendor catalogue).
4. `load_templates()` library satisfies: every template has ≥ 1 variant in every `LanguageCode` (`hi`, `ta`, `kn`, `en`, `hinglish`); every **primary-domain** pattern's `mutation` field set is a subset of the union of `drift_slot_tags` across that domain's templates. The two transversal payment-auth patterns (`payment.auth_scope_upgrade`, `payment.mfa_required`) are EXEMPT from this subset check — they mutate shared payment fields (`token`, `scope`, `mfa_code`) that are intentionally not present in primary-domain goal templates and therefore cannot appear in `drift_slot_tags`.
5. Publication invariant: train seed set ∩ val seed set = ∅.
6. Publication invariant: running `data_export.py` twice with the same seed produces byte-identical `train/briefs.jsonl` + `val/briefs.jsonl` (SHA-256 match). Enforced via canonical JSON dump (§3.1): `json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
7. Every file in `data/` begins with an SPDX license header (YAML comment or JSON `$comment`).
8. No 10-digit digit-run in any authored string outside the timestamp / IATA allowed contexts (PII guard).
9. **Per-row data lineage.** Every `BriefRow` (§4.7) in the published `train/` and `val/` splits carries all of: `template_id`, `seed`, `stage`, `language`, `domain`, `generator_version`, `catalogue_hash`, `templates_sha256`, `i18n_sha256`. At eval-load time, `catalogue_hash` / `templates_sha256` / `i18n_sha256` must match the currently-loaded library hashes, else `CatalogueHashMismatchError` is raised (§5).
10. **Bundle immutability.** After publication (§2.4), the train split SHA-256 MUST match across all future re-runs of `hf upload`; adding val rows requires a minor-version bump, never a silent train-split mutation.

---

## 4. Data Structures

All types are frozen dataclasses, immutable after load. Mappings are wrapped in `types.MappingProxyType`.

### 4.1 `TemplateLibrary` (re-exported from `task_generator.models` — single source of truth)

```python
@dataclass(frozen=True)
class TemplateLibrary:
    templates: tuple[Template, ...]                                      # exactly 20 at v1.0
                                                                         # (4 domains × 5 templates);
                                                                         # ≥ 20 after minor-version bumps
    cities_by_domain: Mapping[Domain, tuple[str, ...]]                   # 10 per domain
    i18n: Mapping[LanguageCode, Mapping[str, str]]                       # merged from i18n.yaml
    source_sha256: str                                                   # hash of templates.yaml bytes
```

The `templates` tuple length is **exactly 20** at v1.0 publication (4 domains × 5 templates per domain). Post-v1.0 minor-version bumps may grow this count monotonically; the invariant `len(templates) >= 20` and `len(templates) % 5 == 0` holds across all future versions. `load_templates` asserts `len(templates) == 20` at v1.0 via the `generator_version` check.

Authoritative schema lives in `docs/modules/task_generator.md` §4. This module re-exports the type so callers of `load_templates` receive the same object that `task_generator.generate` consumes.

### 4.2 `I18nLibrary`

```python
@dataclass(frozen=True)
class I18nLibrary:
    strings: Mapping[LanguageCode, Mapping[str, str]]
    # e.g., strings["hi"]["BLR"] = "बेंगलुरु"
    # strings["ta"]["Monday"] = "திங்கள்"
    source_sha256: str
```

Merged into `TemplateLibrary.i18n` by `load_templates`, but exposed independently for pure-i18n use cases (e.g., the Gradio demo UI localizing labels).

### 4.3 `DriftPatternLibrary`

```python
@dataclass(frozen=True)
class DriftPatternLibrary:
    patterns: Mapping[str, DriftPattern]                                 # keyed by DriftPattern.id
    by_domain: Mapping[str, tuple[str, ...]]                             # domain → pattern_ids
    by_type:   Mapping[str, tuple[str, ...]]                             # drift_type → pattern_ids
    source_sha256: str
```

`DriftPattern` itself is defined in `docs/modules/drift_injector.md` §4.2 (see the `DriftPattern` dataclass snippet). This module owns *loading*, `drift_injector` owns *applying*.

### 4.4 `APISchemaRegistry`

```python
@dataclass(frozen=True)
class APISchema:
    domain: str                       # "airline" | "cab" | "restaurant" | "hotel" | "payment"
    version: str                      # "v1" | "v2" | "v3"
    schema: Mapping[str, Any]         # parsed JSON Schema 2020-12 document
    source_sha256: str

@dataclass(frozen=True)
class APISchemaRegistry:
    schemas: Mapping[str, Mapping[str, APISchema]]
    # schemas["airline"]["v2"] = APISchema(...)

    def get(self, domain: str, version: str) -> APISchema: ...
    def versions(self, domain: str) -> tuple[str, ...]: ...              # ordered v1,v2,v3
```

Each `v<N>.json` is a valid JSON Schema 2020-12 document describing the tool-response shape for that domain at that drift version. Vendors (DESIGN.md §5) validate outgoing responses against these schemas at test time; the injector (`docs/modules/drift_injector.md` §3) consults version transitions via these files.

### 4.5 `AudioManifest`

```python
@dataclass(frozen=True)
class AudioClip:
    utterance_id: str                 # stable; matches a curated IndicVoices-R clip id
    path: Path                        # relative to data/audio/
    language: LanguageCode
    source: Literal["real_indicvoices_r"]   # manifest is authored-only; synth clips
                                            # are lazily generated and NEVER recorded here
    license: str                      # SPDX identifier
    sha256: str
    duration_s: float                 # ≤ 20.0 (DESIGN.md §9 upper bound)

@dataclass(frozen=True)
class AudioManifest:
    clips: tuple[AudioClip, ...]
    source_sha256: str                # hash of MANIFEST.jsonl bytes
```

The `source` field is a single-value `Literal` — the manifest is **authored-only**. Synth clips generated on-demand by `audio/tts_kokoro.py` are **never** recorded in the manifest (they are transient, gitignored under `data/audio/synth/`). This keeps the manifest auditable and its SHA-256 stable across Kokoro model-weight updates.

### 4.6 `SFTCorpus` (L4, optional)

```python
@dataclass(frozen=True)
class SFTTrajectory:
    episode_id: int
    goal_seed: int                    # same seed space as train/; NEVER a val seed (§3.1)
    turns: tuple[Mapping[str, Any], ...]   # role/content pairs, JSON-serializable
    stage: Literal[1, 2, 3]
    reward_breakdown: Mapping[str, float]  # R1..R5 + total, from the env at synthesis time
    generation_batch_id: str          # uuid4 per invocation of sft_generator.py
    generation_index: int             # monotonic within a batch, 0..N-1

@dataclass(frozen=True)
class SFTCorpus:
    trajectories: tuple[SFTTrajectory, ...]
    generator: Literal["sarvam-m-hf-inference"]
    generation_seed: int
    target_count: int                 # from --target-count CLI flag
    source_sha256: str
```

Consumed by `training/train_grpo.py` only when `--sft-warmup-steps > 0` is passed. Absent by default; loader raises a non-fatal warning if the file is missing (training proceeds without warmup).

**Atomic append + restart recovery (`training/sft_generator.py`):**

- Each trajectory is appended to `data/sft_warmup/trajectories.jsonl` as a single canonical-JSON line followed by `os.fsync(fd)` on the file descriptor, ensuring durability before the next Sarvam-M API call. Partial-write recovery is therefore line-granular.
- Every row carries `generation_batch_id` (uuid4, generated once per invocation of `sft_generator.py`) and `generation_index` (monotonic integer 0..N-1 within that batch).
- On restart, `sft_generator.py` reads the existing `trajectories.jsonl`, reconstructs `(seed, generation_index)` pairs already completed, and resumes from the next uncompleted seed in its deterministic seed list. This tolerates Sarvam-M rate-limit drops, OOM kills, and SIGKILL.
- After all generation completes, the script performs a **final count validation**: if `len(trajectories) != target_count`, it raises `PartialSFTCorpusError` (§5). The loader `load_sft_corpus` also performs this check at load time and raises the same error if the on-disk row count does not match the `target_count` field.
- Edge case #11 (§7) walks through a concrete partial-generation-recovery scenario.

### 4.7 `BriefRow` — canonical publication-row contract

Every line of `train/briefs.jsonl` and `val/briefs.jsonl` in the published HF Hub bundle is exactly one serialized `BriefRow`. This dataclass is the single-source-of-truth schema for everything an offline consumer (a judge re-running eval, a third party reproducing our experiments) needs to re-derive the episode from `(seed, library@hash)` alone.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from driftcall.models import GoalSpec, DriftEvent, LanguageCode

@dataclass(frozen=True)
class BriefRow:
    episode_id: str                    # deterministic from seed + stage (e.g. "s2_ep_00000042")
    seed: int                          # original episode seed (train: [0, 20_000_000),
                                       #                         val:   [20_000_000, 20_000_500))
    stage: Literal[1, 2, 3]            # curriculum stage at publication time
    language: LanguageCode             # "hi" | "ta" | "kn" | "en" | "hinglish"
    domain: Literal["airline", "cab", "restaurant", "hotel"]
    template_id: str                   # e.g. "airline.book.budget_timewindow"
    goal: GoalSpec                     # full GoalSpec (slots + constraints + seed_utterance)
    drift_schedule: tuple[DriftEvent, ...]   # schedule pre-computed by drift_injector
    catalogue_hash: str                # sha256(drift_patterns/drifts.yaml bytes)
    templates_sha256: str              # sha256(task_briefs/templates.yaml bytes)
    i18n_sha256: str                   # sha256(task_briefs/i18n.yaml bytes)
    generator_version: str             # e.g. "driftcall-1.0.0" — semver of the generator
    created_ts_ist: str                # ISO 8601 with +05:30 offset, e.g. "2026-04-25T10:30:00+05:30"
```

Serialization is always canonical: `json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))`. A concrete JSONL line example is given in §8.5.

At eval-load time, the loader re-hashes the currently-loaded `drifts.yaml` / `templates.yaml` / `i18n.yaml` and compares against `catalogue_hash` / `templates_sha256` / `i18n_sha256`. Any mismatch raises `CatalogueHashMismatchError` (§5) — this prevents silent semantic drift where a consumer runs `train/briefs.jsonl` against a newer catalogue and gets different episodes.

---

## 5. Error Modes

All exceptions subclass `DatasetError(Exception)`. Each is raised exactly once and unit-tested.

| Exception | Trigger | Where raised |
|---|---|---|
| `DatasetFileMissingError` | `data/<path>` absent on disk | every loader |
| `MalformedYAMLError` | YAML parse failure (syntax) | `load_templates`, `load_i18n`, `load_drift_patterns` |
| `MalformedJSONError` | JSON parse failure (syntax) | `load_api_schemas`, `load_audio_manifest`, `load_sft_corpus` |
| `DatasetSchemaError` | type/shape validation failure (missing required key, wrong type, extra unknown key) | every loader |
| `UnknownLanguageKeyError` | a language key ∉ `LanguageCode = {"hi","ta","kn","en","hinglish"}` appears in `templates.yaml` or `i18n.yaml` | `load_templates`, `load_i18n` |
| `LicenseConflictError` | a CC-BY-SA or GPL-licensed row appears in publication bundle while bundle is Apache-2.0; or verbatim ≥ 10-token suffix matches a CC-BY-SA upstream row | publication script (see §3.4) |
| `TrainValLeakError` | train and val seed sets intersect; or an `SFTTrajectory.goal_seed` sits in the val reserved range `[20_000_000, 20_000_500)` | publication script, `load_sft_corpus` |
| `DriftPatternOrphanError` | `drift_patterns.yaml` references a `from_version`/`to_version` not present in `data/api_schemas/<domain>/` | `load_drift_patterns` |
| `ChecksumMismatchError` | `AudioClip.sha256` does not match the on-disk file's hash | `load_audio_manifest` |
| `UnicodeNFDError` | any loaded string fails `unicodedata.is_normalized("NFC", s)` | every loader |
| `PIIDetectedError` | a 10-digit run appears outside allowed contexts in authored text | every text-bearing loader; also CI lint |
| `DuplicateDriftPatternIdError` | two entries in `drifts.yaml` share an `id` | `load_drift_patterns` |
| `CatalogueHashMismatchError` | a `BriefRow` in `train/briefs.jsonl` or `val/briefs.jsonl` carries `catalogue_hash` / `templates_sha256` / `i18n_sha256` that does not match the currently-loaded library (drifts.yaml / templates.yaml / i18n.yaml) hashes | eval-load path (consumers of published bundle) |
| `PartialSFTCorpusError` | `len(SFTCorpus.trajectories) != target_count` at final-count validation; raised by `training/sft_generator.py` post-generation and by `load_sft_corpus` at load time | `load_sft_corpus`, `training/sft_generator.py` |

**No silent fallbacks.** If `data/sft_warmup/trajectories.jsonl` is missing, `load_sft_corpus` raises `DatasetFileMissingError`; the training script is the one that treats this as non-fatal (falls back to no-SFT warmup). Loaders themselves never substitute defaults.

---

## 6. Dependencies

### 6.1 Reads

- `data/task_briefs/templates.yaml`, `data/task_briefs/i18n.yaml`
- `data/drift_patterns/drifts.yaml`
- `data/api_schemas/**/*.json`
- `data/audio/real/MANIFEST.jsonl` + the `.wav` files it references
- `data/sft_warmup/trajectories.jsonl` (optional)

### 6.2 Imports

- `driftcall.models` — `GoalSpec`, `LanguageCode`, `Domain`
- Python stdlib: `json`, `hashlib`, `pathlib`, `unicodedata`, `threading`, `dataclasses`, `typing`, `types`
- Third-party: `PyYAML`, `jsonschema` (for JSON Schema 2020-12 meta-validation)

### 6.3 Consumers

Consuming modules and the exact function they call:

- `docs/modules/task_generator.md` — `load_templates()` in `task_generator.generate()`'s lazy-singleton `_get_library()`.
- `docs/modules/drift_injector.md` — `load_drift_patterns()` in the injector's module-level registry; consults DESIGN.md §6.3 pattern catalogue.
- `docs/modules/vendors.md` — `load_api_schemas()` at vendor import time; each vendor asserts its own response shape against the schema in test fixtures.
- `docs/modules/audio.md` — `load_audio_manifest()` for the pitch demo (§9.5 IndicVoices-R clip playback).
- `docs/modules/training.md` — `load_sft_corpus()` behind `--sft-warmup-steps` flag; also invokes `training/data_export.py` which calls `task_generator.enumerate_variants()` to produce the publication briefs.

### 6.4 Publishes to

- HF Hub dataset repo `<team>/driftcall-indic-briefs` (one-time, pre-event, Phase C5 per `DRIFTCALL/CLAUDE.md` §4.1).

### 6.5 Non-dependencies (explicit)

- Does **not** import from `env.py`, `rewards.py`, `app.py`, or the training entrypoint. Pure data layer.
- Does **not** hit the network at runtime. Every file is local. Publication script is a separate, explicitly-invoked entrypoint.
- Does **not** depend on GPU, CUDA, or PyTorch. CPU-only.

---

## 7. Edge Cases

1. **Missing template variant for a rare language.** `templates.yaml` is authored with `hinglish` + `hi` + `en` + `ta` but an author forgets `kn` for one template. `load_templates` runs per-template check `set(variants.keys()) == LanguageCode.values` and raises `DatasetSchemaError: template 'restaurant.order.veg' missing language_variants['kn']`. The generator's `NoVariantForLanguageError` (task_generator.md §5) never has a chance to fire because loading fails first. Fix: author supplies the missing variant; loader re-runs.

2. **Unicode NFD in author contribution.** A collaborator pastes a Kannada weekday name from macOS clipboard (NFD by default for composed characters). `load_i18n` re-normalizes to NFC *before* equality/hashing; the assertion `unicodedata.is_normalized("NFC", value)` fires post-normalization as a defense against Python/ICU bugs. In practice the round-trip succeeds and NFC is stored. The pre-commit hook separately catches NFD at commit time so CI never sees it.

3. **License incompatibility (CC-BY-SA row smuggled into an Apache-2.0 bundle).** An author, inspired by an SGD row, copies 20 tokens verbatim into a template variant. Publication CI runs a suffix-array check over cached SGD/MTOP exports looking for ≥ 10-token verbatim matches; on hit, `LicenseConflictError("variant in 'airline.book.budget_timewindow' matches SGD row sgd_5432:0 (≥ 10 tokens)")` raises. Fix: rewrite the variant. We keep only *inspiration*, never verbatim text, from CC-BY-SA sources. The threshold (10 tokens) is a pragmatic choice: below that length we treat overlap as incidental linguistic reuse; at or above we flag.

4. **Empty language cohort in a stage mix.** A future curriculum config passes `language_weights = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}`. This is valid at the task-generator level (task_generator.md §3.2 — non-negative weights summing to 1 are legal). `datasets` does not re-validate curriculum config; it only asserts the *library* has variants for all 5 languages. Downstream (`task_generator`) will simply never draw `hi`/`ta`/`kn`/`hinglish`. No error in this module.

5. **Train/val episode-id collision at publication time.** `data_export.py` draws 15,000 seeds for train and 500 for val. If the RNG accidentally maps a train seed into `[20_000_000, 20_000_500)` (the val reserved range) — which cannot happen given the seed-space partitioning in §3.1 — the assertion `train_seeds.isdisjoint(val_seeds)` raises `TrainValLeakError` with the offending seed. Safeguard: train seeds are drawn from `[0, 20_000_000)` and val seeds from `[20_000_000, 20_000_500)`. The two ranges are non-overlapping by construction; the assertion is a defense against future range edits.

6. **Drift-pattern-id orphan (trace references pattern not in YAML).** A test fixture or cached trace references `drift_pattern_id='airline.mysterious_fee'` but `drifts.yaml` has no such entry (it was renamed or removed). `load_drift_patterns` does not look at traces — it only checks internal consistency. The *trace consumer* (`rewards.r2_drift_detection` in `docs/modules/rewards.md`) raises `UnknownDriftPatternError` at scoring time per drift_injector.md §5. If the orphan is discovered during dataset publication, the publication script emits `DriftPatternOrphanError` and aborts.

7. **JSON Schema file that is valid JSON but not valid JSON Schema 2020-12.** `data/api_schemas/cab/v3.json` is hand-edited and accidentally drops the `$schema` keyword or uses an unknown keyword. `load_api_schemas` runs `jsonschema.Draft202012Validator.check_schema(schema)` and on failure raises `DatasetSchemaError("cab/v3.json: not a valid JSON Schema 2020-12: <error>")`. The env refuses to serve `reset()` until fixed.

8. **Audio clip on disk does not match manifest sha256.** `data/audio/real/MANIFEST.jsonl` lists `kn_greeting_03.wav` with `sha256=abc...`. The file gets re-encoded (e.g., by a well-intentioned ffmpeg pass). `load_audio_manifest` re-hashes every referenced WAV and raises `ChecksumMismatchError("kn_greeting_03.wav: expected abc..., got def...")`. Fix: either revert the WAV, or regenerate the manifest after an audit trail commit.

9. **SFT corpus contains a val-reserved seed.** Sarvam-M synthesis inadvertently uses a seed in `[20_000_000, 20_000_500)`. `load_sft_corpus` raises `TrainValLeakError`. The training script may be configured to treat this as fatal (default) or to filter out those trajectories (`--sft-tolerate-leak`); the loader itself always raises.

10. **PyYAML silently deduplicating keys.** If `drifts.yaml` has two entries with the same `id`, the YAML parse is valid but one wins. `load_drift_patterns` builds a set of ids during validation and raises `DuplicateDriftPatternIdError` on collision, with both source line numbers.

11. **Partial SFT corpus recovery (L4 restart).** `training/sft_generator.py` is mid-run at trajectory 137 of a target 300 when the host OOM-kills the process (Sarvam-M inference peak memory). On restart, the script re-opens `data/sft_warmup/trajectories.jsonl`, reads the existing 137 rows (each fsync'd atomically per §4.6), reconstructs the completed `(generation_batch_id, generation_index)` pairs, and resumes from index 137 of the same batch. It does NOT start a new `generation_batch_id` — batch id is rehydrated from the last row. When generation finally reaches 300, the script validates `len(rows) == target_count`; if a Sarvam-M response was silently truncated (say, only 298 rows written), `PartialSFTCorpusError("expected 300, got 298")` is raised and the operator must decide whether to re-run the missing two or ship a corpus with a smaller `target_count`. `load_sft_corpus` performs the same count check at load time.

---

## 8. Examples

### 8.1 Full `templates.yaml` entry for `airline.book.budget_timewindow`

```yaml
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DriftCall Team
# Derived-from: AmazonScience/MASSIVE (intent taxonomy, Apache-2.0)

- template_id: airline.book.budget_timewindow
  domain: airline
  intent: book_flight
  min_stage: 1
  required_slots: [from, to, when]
  optional_slots: [seat_pref]
  constraints_template:
    budget_inr:
      distribution: uniform
      low: 3000
      high: 15000
      step: 500
    time_window:
      choices: [morning, afternoon, evening, late_night]
  drift_slot_tags: [price, total_fare_inr]
  # Language keys: ISO short codes matching LanguageCode = Literal["hi","ta","kn","en","hinglish"]
  language_variants:
    hinglish:
      - "Bhai {when} ko {to} jaana hai, cheapest flight {time_window} mein, {budget_inr} rupees max"
      - "{when} ko {from} se {to} ka ticket book kar de, under {budget_inr}, {time_window} ke baad"
    hi:
      - "मुझे {when} को {from} से {to} जाना है, {budget_inr} रुपये से कम में"
    ta:
      - "{when} அன்று {from} லிருந்து {to} க்கு டிக்கெட் வேண்டும், {budget_inr} ரூபாய்க்கு கீழ்"
    kn:
      - "{when} ರಂದು {from} ಇಂದ {to} ಗೆ ಅಗ್ಗದ ವಿಮಾನ ಟಿಕೆಟ್ ಬೇಕು, {budget_inr} ರೂಪಾಯಿಗಳ ಒಳಗೆ"
    en:
      - "Book the cheapest flight from {from} to {to} on {when}, budget under ₹{budget_inr}, departing {time_window}"
```

This is the single source-of-truth entry for the Stage-1 airline booking template; mirror of DESIGN.md §8.3 and `docs/modules/task_generator.md` §4.1.

### 8.2 Full `drift_patterns.yaml` entry for `airline.price_rename`

```yaml
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DriftCall Team

- id: airline.price_rename
  drift_type: schema
  domain: airline
  from_version: v1
  to_version: v2
  description: "field 'price' renamed to 'total_fare_inr'; 'currency' removed"
  mutation:
    rename: {price: total_fare_inr}
    remove: [currency]
  detection_hints:
    - "total_fare_inr"
    - "price"
    - "rename"
```

`load_drift_patterns` will (a) parse this, (b) check `id` uniqueness, (c) confirm `from_version=v1` + `to_version=v2` both exist as `data/api_schemas/airline/v1.json` + `data/api_schemas/airline/v2.json`, (d) confirm `detection_hints` is non-empty, (e) wrap `mutation` in `MappingProxyType`. Matches `docs/modules/drift_injector.md` §4.3 byte-for-byte.

### 8.3 `data/api_schemas/airline/v2.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://driftcall.dev/schemas/airline/v2.json",
  "$comment": "SPDX-License-Identifier: Apache-2.0. v2 = post-price_rename drift (DESIGN.md §5.1).",
  "title": "Airline search result (v2)",
  "type": "object",
  "required": ["flight_id", "from", "to", "depart", "total_fare_inr", "seats_left"],
  "additionalProperties": false,
  "properties": {
    "flight_id":       {"type": "string", "pattern": "^[0-9A-Z]{2}-[0-9]{4}$"},
    "from":            {"type": "string", "pattern": "^[A-Z]{3}$"},
    "to":              {"type": "string", "pattern": "^[A-Z]{3}$"},
    "depart":          {"type": "string", "format": "date-time"},
    "total_fare_inr":  {"type": "integer", "minimum": 0},
    "seats_left":      {"type": "integer", "minimum": 0}
  }
}
```

Note that `price` and `currency` from v1 are absent (drift `airline.price_rename` applied). Vendors (`docs/modules/vendors.md`) validate their emitted `airline.search` responses against whichever version the injector has installed in `state.schema_versions['airline']`. This schema also serves as the R2 structural detection surface: a tool call that keys into `price` after drift returns `KeyError` / 422, which is a detection-positive signal per DESIGN.md §7.1 R2.

### 8.4 `MANIFEST.jsonl` row for a curated IndicVoices-R clip (L3)

```json
{"utterance_id": "iv_r_kn_0451", "path": "real/kn/iv_r_kn_0451.wav", "language": "kn", "source": "real_indicvoices_r", "license": "Apache-2.0", "sha256": "b7f1a9c2e5d4...", "duration_s": 4.82}
```

Referenced only by the pitch demo. Training never touches this file — DRIFTCALL/CLAUDE.md §9 "Do not put TTS/ASR in the training loop".

### 8.5 Canonical `BriefRow` JSONL line (single row from `train/briefs.jsonl`)

One line from the published bundle — canonical JSON (sorted keys, no whitespace, UTF-8 preserved for Devanagari):

```json
{"catalogue_hash":"3f9a8e7c2b1d4e5f6a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f","created_ts_ist":"2026-04-25T10:30:00+05:30","domain":"airline","drift_schedule":[{"description":"'price' field renamed to 'total_fare_inr'","domain":"airline","drift_type":"schema","from_version":"v1","pattern_id":"airline.price_rename","to_version":"v2","turn":4}],"episode_id":"s2_ep_00000042","generator_version":"driftcall-1.0.0","goal":{"constraints":{"budget_inr":8000,"time_window":"evening"},"domain":"airline","intent":"book_flight","language":"hinglish","seed_utterance":"Bhai Friday ko Bangalore jaana hai, cheapest flight evening mein, 8000 rupees max","slots":{"from":"HYD","to":"BLR","when":"2026-04-30"}},"i18n_sha256":"a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef","language":"hinglish","seed":42,"stage":2,"template_id":"airline.book.budget_timewindow","templates_sha256":"b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef12"}
```

Note: keys are alphabetically sorted (`catalogue_hash`, `created_ts_ist`, `domain`, …), strings are NFC-normalized, no embedded spaces. The 64-hex hashes are full sha256 hex digests.

### 8.6 `README.md` YAML frontmatter (HF Hub dataset card)

The published `<org>/driftcall-indic-briefs/README.md` begins with the following YAML frontmatter. The HF Dataset Viewer reads this block to auto-configure splits, license, and task tags.

```yaml
---
license: apache-2.0
language: [hi, ta, kn, en]
size_categories: [10K<n<100K]
task_categories: [conversational, text-generation]
pretty_name: DriftCall Indic Briefs
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/briefs.jsonl
      - split: val
        path: val/briefs.jsonl
dataset_info:
  features:
    - { name: episode_id, dtype: string }
    - { name: seed, dtype: int64 }
    - { name: stage, dtype: int32 }
    - { name: language, dtype: string }
    - { name: domain, dtype: string }
    - { name: template_id, dtype: string }
  splits:
    - { name: train, num_examples: 15000 }
    - { name: val, num_examples: 500 }
---
```

The body of `README.md` follows below the frontmatter: dataset description, licensing chain (full Apache-2.0 text is in the separate `LICENSE` file per §3.4), provenance (`generator_version`, `catalogue_hash`), reward-caveat paragraph, and usage example. The frontmatter's `features` block lists only the top-level flat columns; nested structs (`goal`, `drift_schedule`) are auto-inferred by the HF Datasets library on first load.

---

## 9. Open Questions

1. **HF org name not yet finalized.** `<org>` placeholder in `<org>/driftcall-indic-briefs` depends on `DRIFTCALL/CLAUDE.md` §8 kickoff-checklist item "HF org name locked". The publication script parameterizes the org via `--hf-org`; no code change needed once locked, just a CLI arg at publication time. Does not block Phase D. **Sync note:** `DRIFTCALL/CLAUDE.md` §6 command table still lists the deprecated `huggingface-cli upload` — when the org name is locked, update that table to the modern `hf upload` in the same PR.

2. **SFT warmup corpus size — 200 vs 500 trajectories.** DESIGN.md §8.2 row 4 quotes the range "200–500". The exact count depends on Sarvam-M's cost/latency budget during one-shot synthesis. Recommend 200 as a floor (sufficient for format priming per §10 training convergence target) and 500 as a ceiling if inference time permits. Resolution: Person C chooses during Phase C4; does not affect loader or schema.

3. **Audio manifest curation count.** DESIGN.md §9 implies a handful of real IndicVoices-R clips for pitch demo realism, but does not specify exact count. Recommend 20 curated clips (4 per language × 5 languages), balanced by speaker gender and dialect region. Resolution: Person D curates during Phase C5; this module only ensures the manifest format is stable.

### 9.1 Resolved

- **License-cache implementation (previously Open Q #4).** `data/.license_cache/{sgd,mtop}.idx` is a sqlite3 FTS5 index built by `scripts/build_license_cache.py` at dev time. Schema: `CREATE VIRTUAL TABLE licensed_text USING fts5(chunk_text, source_id);` with 5-gram tokenization. CI invokes this index (read-only) on each PR to verify that no `seed_utterance` or template variant in the publication bundle substring-matches any upstream CC-BY-SA text (≥ 10-token threshold, §3.4). The index is built once per upstream corpus version and committed to the repo so re-builds are only needed when SGD or MTOP themselves publish a new version. Determinism + reviewability win over per-PR rebuild cost.

---

**This doc tells you HOW the four dataset layers are shaped, loaded, validated, and published. Do not write loaders before a fresh critic returns `NOTHING_FURTHER`. Do not commit `data/*.yaml` without the pre-commit NFC + PII + license-header guards running. Do not ship the HF Hub bundle without the train/val disjointness and verbatim-match checks green.**
