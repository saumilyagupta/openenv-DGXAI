# task_generator — Procedural Task-Brief Generator

**Module path:** `driftcall/task_generator.py`
**Owner:** Person A (Environment)
**Implements:** DESIGN.md §4.2 (`reset()` semantics), §8 (Dataset Strategy — §8.2, §8.3, §8.4), §10.3 (curriculum language mix)
**Consumed by:** `driftcall/env.py` (`DriftCallEnv.reset()`)
**Status:** Design spec — no code yet.

---

## 1. Purpose

`task_generator` is the deterministic, seeded source of every `GoalSpec` consumed by `DriftCallEnv.reset()`. It expands a small hand-authored template library (4 domains × 5 templates × 10 source cities × 10 destinations × 5 languages × 20 drift-compatible slot combinations = **200,000 distinct episode variants**, DESIGN.md §8.4) into concrete per-episode briefs.

One call — `generate(seed, stage, language_weights)` — returns a single fully-populated `GoalSpec` with:

1. A domain (`airline` | `cab` | `restaurant` | `hotel`) chosen deterministically from `seed`.
2. A template variant for that domain, filled with sampled slots (cities, dates, budgets, time windows, dietary flags, etc.).
3. A language picked from the caller-supplied `language_weights` distribution.
4. A `seed_utterance` — the natural-language voice brief in the chosen language, with Unicode-correct Devanagari / Tamil / Kannada script and Hinglish Roman transliteration.
5. `slots` and `constraints` dicts suitable for the reward graders (R1 task completion, R3 constraint adherence — DESIGN.md §7.1).

**Determinism is the contract.** Identical `(seed, stage, language_weights)` triples always produce identical `GoalSpec`s, byte-for-byte after NFC normalization. This enables reproducible training, reproducible evals, and reproducible drift scheduling downstream (DESIGN.md §6.2 — drift schedules are themselves seeded off the same episode ID).

The generator owns **no random global state**. Every stochastic choice threads through `random.Random(seed_for_this_decision)` where the sub-seed is derived from `(seed, decision_tag)` via a stable hash. It does **not** own drift selection — that belongs to `drift_injector` (DESIGN.md §6), which receives the same `seed` and composes its own schedule against the `GoalSpec.domain`.

---

## 2. Interface

All types are imported from `driftcall.models` (see `docs/modules/models.md`). All dataclasses are frozen.

### 2.1 Primary entry point

```python
from __future__ import annotations
from driftcall.models import GoalSpec, LanguageCode

def generate(
    seed: int,
    stage: Literal[1, 2, 3],
    language_weights: dict[LanguageCode, float],
) -> GoalSpec:
    """
    Produce a single fully-populated GoalSpec for episode ``seed`` at curriculum ``stage``.

    Determinism: identical (seed, stage, language_weights) ⇒ identical GoalSpec
    after Unicode NFC normalization of ``seed_utterance``.

    :param seed:              non-negative int, episode identifier; also the root
                              seed for all sub-choices (domain, template, slots,
                              language, utterance variant).
    :param stage:             curriculum stage ∈ {1, 2, 3}; affects allowed
                              template complexity (stage 1 uses simple templates
                              only; stage 3 enables drift-compatible slots).
    :param language_weights:  normalized distribution over LanguageCode keys;
                              values must be non-negative and sum to 1.0 ± 1e-6.

    :returns: GoalSpec whose .seed_utterance is NFC-normalized UTF-8.

    :raises InvalidLanguageWeightError: weights empty, negative, or sum ≠ 1.0.
    :raises InvalidStageError:          stage ∉ {1, 2, 3}.
    :raises InvalidBudgetError:         sampled budget outside template's declared
                                        [low, high] range (indicates corrupt template).
    :raises MissingSlotError:           template variant references a {slot}
                                        placeholder not present in the filled slot dict.
    :raises TemplateFileMissingError:   ``data/task_briefs/templates.yaml`` not found
                                        or malformed.
    :raises UnicodeNormalizationError:  rendered utterance fails NFC round-trip check
                                        (raised defensively — should never fire in practice).
    """
```

### 2.2 Helper signatures (all module-private except where noted)

```python
# --- template loader (public for tests + corpus packaging) ---
def load_templates(path: Path | str = "data/task_briefs/templates.yaml") -> TemplateLibrary:
    """
    Parse the YAML template file, validate the schema (§4 below),
    and return an in-memory TemplateLibrary.

    Called once at module import via a lazy singleton; callers should use
    ``_get_library()`` inside the module. Exposed publicly for unit tests
    and the dataset-packaging script that writes ``train/briefs.jsonl``
    (DESIGN.md §8.6).

    :raises TemplateFileMissingError:   path does not exist.
    :raises TemplateSchemaError:        YAML present but fails schema validation
                                        (missing required key, wrong type, etc.).
    """

# --- domain + template picker ---
def _pick_domain(seed: int) -> Literal["airline", "cab", "restaurant", "hotel"]:
    """Uniform over 4 domains, seeded by hash(seed, 'domain')."""

def _pick_template(seed: int, stage: int, domain: str, library: TemplateLibrary) -> Template:
    """
    Uniform over templates for ``domain`` whose ``min_stage`` ≤ ``stage``.
    Seeded by hash(seed, 'template').
    """

# --- slot expander ---
def _expand_slots(seed: int, template: Template) -> SlotGrid:
    """
    For each slot in the template's required_slots + optional_slots + constraints_template,
    sample one concrete value per the slot's declared distribution.

    Returns a SlotGrid: a frozen mapping of slot-name -> concrete value.

    Handles:
      - enum slots (``choices: [...]``)
      - uniform numeric ranges (``distribution: uniform, low, high, step``)
      - city slots (from the 10×10 city/destination grid, domain-filtered)
      - date slots (relative to a fixed reference date, DESIGN.md §11.1 — deterministic)
      - boolean slots (veg_only, etc.)
    """

# --- language picker ---
def _pick_language(seed: int, language_weights: dict[LanguageCode, float]) -> LanguageCode:
    """
    Weighted draw from ``language_weights`` seeded by hash(seed, 'language').
    ``language_weights`` is validated by ``generate()`` before this is called.
    """

# --- utterance formatter ---
def _format_utterance(
    seed: int,
    template: Template,
    slots: SlotGrid,
    language: LanguageCode,
) -> str:
    """
    Pick one of the template.language_variants[language] strings (uniform,
    seeded by hash(seed, 'variant')), substitute every {slot} placeholder
    with the Unicode-correct rendering of slots[slot], and return the
    NFC-normalized result.

    :raises MissingSlotError:          format string references {X} but X not in slots.
    :raises UnicodeNormalizationError: NFC round-trip fails.
    """

# --- public helper: list all (seed, stage, lang_weights) combos for dataset packaging ---
def enumerate_variants(
    limit: int | None = None,
    stage: int = 3,
    language_weights: dict[LanguageCode, float] | None = None,
) -> Iterator[GoalSpec]:
    """
    Deterministic walk over the procedural grid, yielding up to ``limit``
    GoalSpecs. Used by DESIGN.md §8.6 to produce ``train/briefs.jsonl``
    and ``val/briefs.jsonl``. Not called from env.reset().

    Walk order: domain (4) → template (5) → from×to (10×10) → language (5)
    → utterance variant. Stable across runs.
    """
```

---

## 3. Behavior Spec

### 3.1 Determinism via seed (DESIGN.md §4.2, §8.4)

- Every sub-decision uses `random.Random(stable_sub_seed(seed, tag))` where `stable_sub_seed` is `int.from_bytes(hashlib.blake2b(f"{seed}:{tag}".encode(), digest_size=8).digest(), "big")`.
- Valid tags: `"domain"`, `"template"`, `"slots"`, `"language"`, `"variant"`, plus per-slot tags `f"slot:{slot_name}"`.
- Never call `random.random()` (global state) or `time.time()` anywhere in the module.
- `generate(42, 1, W)` on two machines with identical Python versions returns byte-identical `GoalSpec.seed_utterance` after NFC normalization.

### 3.2 Language-weight sampling

- `language_weights` is the caller's contract for the curriculum mix (DESIGN.md §10.3 defines Stage-1 50/30/20 and Stage-2/3 30/30/20/10/10 splits).
- `generate()` **validates** weights before sampling. Each check binds to exactly one exception:
  - **Unsupported key** — any key ∉ `{"hi", "ta", "kn", "en", "hinglish"}` (LanguageCode) → raises `InvalidLanguageError`.
  - **Empty weights dict** — `len(language_weights) == 0` → raises `InvalidLanguageWeightError`.
  - **Negative weight value** — any `w < 0` → raises `InvalidLanguageWeightError`.
  - **Sum outside tolerance** — `|sum(weights) − 1.0| > 1e-6` → raises `InvalidLanguageWeightError`.
  - **All weights zero** — defensive assertion; redundant with the sum-check (if sum = 1 ± 1e-6 and all ≥ 0, at least one must be > 0) but kept as an explicit invariant that guards against floating-point edge cases where sum rounds to 1 via noise while every entry is 0. Raises `InvalidLanguageWeightError`.
- `_pick_language` uses `random.Random(sub_seed).choices(population, weights=w, k=1)[0]`.

### 3.3 Slot combinatorial grid (DESIGN.md §8.4)

- Each template declares `required_slots`, `optional_slots`, and `constraints_template` (§4 below).
- The source × destination city grid is **domain-scoped**: airline + hotel draw from inter-city pairs; cab + restaurant draw from intra-city locations. Both lists are 10 entries each per domain (40 total unique cities across domains, deduped in the YAML).
- Optional slots are included with probability 0.5 (seeded).
- Date slots are sampled relative to a fixed reference date `2026-04-25` from a 60-day forward window (so train/val sets are temporally stable).
- Budget slots sample on the declared `step` grid: e.g., `uniform 3000..15000 step 500` yields one of `{3000, 3500, …, 15000}`.
- Stage 1 uses templates flagged `min_stage: 1`; stages 2–3 also admit `min_stage: 2` and `min_stage: 3` (more complex compound-constraint templates with drift-compatible slot layouts).

### 3.4 Unicode handling for Hindi, Tamil, Kannada

- Template YAML is authored in NFC (Unicode Normalization Form C). The loader **re-normalizes** on read (defensive).
- After slot substitution, `_format_utterance` calls `unicodedata.normalize("NFC", s)` and asserts `unicodedata.is_normalized("NFC", s)` — if not, raises `UnicodeNormalizationError`.
- City names, dish names, and day-of-week translations for Hindi / Tamil / Kannada live in a static lookup table (`data/task_briefs/i18n.yaml`, loaded by `load_templates`). English + Hinglish share Roman script with Devanagari-free glyphs (ASCII + `₹`).
- **`i18n.yaml` is NFC-normalized at load time.** `load_templates` applies `unicodedata.normalize("NFC", v)` to every string value parsed out of `data/task_briefs/i18n.yaml` (city names, weekday names, dish names, domain-specific nouns — across `hi`, `ta`, `kn`, `en`, `hinglish`) before those strings are stored in `TemplateLibrary.i18n`. The same NFC pass is applied to every string inside `templates.yaml` (variant strings, choices enums, slot labels). Consequence: every string that `_expand_slots` pulls into a `SlotGrid` is already NFC, so downstream consumers — `_format_utterance`, reward R1 string-equality comparisons (DESIGN.md §7.1), and audit logging — may assume NFC without re-normalizing.
- Hinglish is **always Roman-script** (no mixed scripts); Hindi is **always Devanagari-script**. A template that tries to mix the two in a single variant is rejected at load time.

### 3.5 Stage-aware complexity

| Stage | Templates allowed | Compound constraints | Drift-compatible slot layout |
|---|---|---|---|
| 1 | `min_stage: 1` only (simple: domain + 1 required slot + up to 2 constraints) | No | No — slots chosen from v1-schema-compatible fields only |
| 2 | `min_stage` ≤ 2 | Up to 2 constraints | Slots cover fields likely to be renamed (`price`, `fare_inr`) so drift is observable |
| 3 | all templates | Up to 3 constraints | Slots must include ≥ 1 field that a Stage-3 compound drift will touch |

"Drift-compatible slot layout" is a static property of the template (declared in YAML via `drift_slot_tags: [price, passenger_count, …]`) — the generator does **not** itself pick drifts; it only guarantees the slot surface is rich enough for `drift_injector` to have something meaningful to mutate.

### 3.6 Invariants (enforced by tests)

1. `generate(s, k, w) == generate(s, k, w)` for any valid `(s, k, w)`.
2. The returned `GoalSpec.language` appears in `language_weights` with weight > 0.
3. Every `{slot}` placeholder in `seed_utterance` is resolved — no literal `{…}` survives in the output.
4. `GoalSpec.seed_utterance` is in NFC.
5. Stage 1 never yields a template with `min_stage > 1`.
6. Numeric constraints (e.g., `budget_inr`) fall in the template's declared `[low, high]` range.
7. `seed_utterance` length ≤ 280 characters (one SMS; keeps ASR inputs bounded at deploy time — DESIGN.md §9).
8. Every string value in `SlotGrid.values` is NFC-normalized before `generate()` returns (guaranteed by the `i18n.yaml` + `templates.yaml` NFC pass in `load_templates`, §3.4). Reward R1 (string equality) and other downstream consumers may assume NFC on every slot string — they do not need to re-normalize.

---

## 4. Data Structures

### 4.1 Template YAML schema (matches DESIGN.md §8.3 exactly)

```yaml
# data/task_briefs/templates.yaml
- template_id: airline.book.budget_timewindow
  domain: airline                 # {airline, cab, restaurant, hotel}
  intent: book_flight             # free string; mirrored into GoalSpec.intent
  min_stage: 1                    # 1 | 2 | 3
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
  drift_slot_tags: [price, total_fare_inr]  # used by drift_injector for targeting
  # Language keys are ISO short codes matching LanguageCode = Literal["hi","ta","kn","en","hinglish"].
  # Long names (hindi/tamil/kannada/english) are NOT accepted — loader rejects them via TemplateSchemaError.
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

### 4.2 In-memory types

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Mapping

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]
Domain = Literal["airline", "cab", "restaurant", "hotel"]

@dataclass(frozen=True)
class SlotDistribution:
    """Either an enum (``choices``) or a uniform numeric grid (``low``, ``high``, ``step``)."""
    kind: Literal["choices", "uniform"]
    choices: tuple[str, ...] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None

@dataclass(frozen=True)
class Template:
    template_id: str
    domain: Domain
    intent: str
    min_stage: Literal[1, 2, 3]
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    constraints_template: Mapping[str, SlotDistribution]
    drift_slot_tags: tuple[str, ...]
    language_variants: Mapping[LanguageCode, tuple[str, ...]]  # ≥ 1 string per language

@dataclass(frozen=True)
class TemplateLibrary:
    templates: tuple[Template, ...]
    cities_by_domain: Mapping[Domain, tuple[str, ...]]
    i18n: Mapping[LanguageCode, Mapping[str, str]]  # e.g., {"hi": {"BLR": "बेंगलुरु", …}}

@dataclass(frozen=True)
class SlotGrid:
    """Concrete slot values after expansion. Keys are slot names; values are already
    localized to the chosen language (e.g., city rendered in Devanagari for 'hi')."""
    values: Mapping[str, object]  # str | int | float | bool

@dataclass(frozen=True)
class RawBrief:
    """Intermediate product: slots filled, language chosen, utterance not yet rendered.
    Used internally for testability — generate() returns a GoalSpec, not a RawBrief."""
    template_id: str
    domain: Domain
    intent: str
    slots: SlotGrid
    constraints: Mapping[str, object]
    language: LanguageCode
```

`GoalSpec` itself is defined in `driftcall/models.py` (DESIGN.md §4.1) and is the final product of `generate()`. The generator copies `RawBrief` fields into `GoalSpec` and adds the rendered `seed_utterance`.

---

## 5. Error Modes

All exceptions subclass `TaskGeneratorError(Exception)`. Each is raised exactly once in the module and has a test asserting it.

| Exception | Trigger | Where raised |
|---|---|---|
| `MissingSlotError` | template variant references `{X}` but X not in filled `SlotGrid` | `_format_utterance` |
| `InvalidLanguageError` | `language_weights` contains a key ∉ LanguageCode (e.g., `"hindi"`, `"marathi"`) | `generate` (pre-sample validation) |
| `InvalidLanguageWeightError` | empty dict, OR any value < 0, OR sum ∉ [1−1e-6, 1+1e-6], OR all weights = 0 (defensive, redundant with sum-check) | `generate` |
| `InvalidStageError` | `stage ∉ {1, 2, 3}` | `generate` |
| `InvalidBudgetError` | sampled numeric falls outside declared `[low, high]` (indicates corrupt template or step misalignment) | `_expand_slots` |
| `TemplateFileMissingError` | `data/task_briefs/templates.yaml` absent or unreadable | `load_templates` |
| `TemplateSchemaError` | YAML present but fails required-key / type / shape validation | `load_templates` |
| `UnicodeNormalizationError` | NFC round-trip check fails on rendered utterance (defensive) | `_format_utterance` |
| `NoVariantForLanguageError` | chosen template has no `language_variants[chosen_language]` entry | `_format_utterance` |

**No silent fallbacks.** The generator never substitutes a default city, a default language, or a default template on failure — it raises. The env's `reset()` is expected to let these propagate (callers catch and restart with a different seed, never mask).

---

## 6. Dependencies

### 6.1 Reads

- `data/task_briefs/templates.yaml` — the template library (§4.1 schema). Authored by hand in Phase D; never modified at runtime. NFC-normalized at load time (§3.4).
- `data/task_briefs/i18n.yaml` — localized strings for city names, weekdays, domain-specific nouns, in Hindi / Tamil / Kannada. Same load path as templates; separate file for readability. `load_templates` applies `unicodedata.normalize("NFC", v)` to every string value (§3.4) so that `TemplateLibrary.i18n` is NFC-clean before any slot expansion runs.

Both files ship inside the Docker image for the env Space (DESIGN.md §11.1).

### 6.2 Imports

- `driftcall.models` — `GoalSpec`, `LanguageCode`, `Domain`. The generator does **not** import from `env.py`, `rewards.py`, `drift_injector.py`, or any vendor module. Strict one-way dependency.
- Python stdlib: `random`, `hashlib`, `unicodedata`, `dataclasses`, `pathlib`, `typing`.
- Third-party: `PyYAML` (already in `requirements.txt` per DESIGN.md §11.1).

### 6.3 Produces

- `GoalSpec` instance returned to `DriftCallEnv.reset()` (DESIGN.md §4.2).
- `Iterator[GoalSpec]` via `enumerate_variants` for the dataset-packaging script that writes `train/briefs.jsonl` and `val/briefs.jsonl` (DESIGN.md §8.6).

### 6.4 Consumers

- `driftcall/env.py::DriftCallEnv.reset` — the single production caller of `generate()`.
- `training/data_export.py` (Phase C4) — batch-calls `enumerate_variants()` to build the HF Hub dataset artifact.
- `tests/test_task_generator.py` — exercises every branch + every error mode.

### 6.5 Non-dependencies (explicit)

- Does **not** depend on the drift injector. The generator never picks a drift; it only declares `drift_slot_tags` on the template so the injector can target slots later.
- Does **not** depend on audio pipeline. All output is text; TTS happens at the env boundary (DESIGN.md §9.4).

---

## 7. Edge Cases

1. **Missing slot placeholder in a template variant.** YAML author writes `"Bhai {when} ko {destination} jaana hai"` but declares `required_slots: [from, to, when]` — `{destination}` has no fill source. Detected in `_format_utterance` which iterates `string.Formatter().parse()` over the variant; raises `MissingSlotError` naming both the template_id and the missing slot. Also caught earlier if possible — `load_templates` does a static scan and raises `TemplateSchemaError` at load time so runtime failures are rare.

2. **Invalid language code in `language_weights`.** Caller passes `{"marathi": 1.0}`. `generate` validates keys against the `LanguageCode` literal before any sampling and raises `InvalidLanguageError` listing the unsupported keys. No partial `GoalSpec` is constructed.

3. **Budget out of declared range.** Template declares `uniform 3000..15000 step 500`. An implementation bug rounds to `step 1000` and yields `16000`. `_expand_slots` post-condition-checks every numeric against `[low, high]` and raises `InvalidBudgetError`. This should never fire with the spec implementation but exists as a defense — catching corrupt templates or future implementation regressions during unit tests.

4. **Unicode NFC / NFD collision in Kannada or Tamil.** Author pastes a Kannada string copied from macOS (NFD) into `templates.yaml`. `load_templates` re-normalizes to NFC on read; `_format_utterance` final-normalizes the substituted string. A direct byte comparison against the input YAML may differ, but the rendered `seed_utterance` is guaranteed NFC. `UnicodeNormalizationError` only fires if the round-trip assertion itself fails (indicates a Python/ICU bug, not a data bug).

5. **Seed collision across episodes.** Training loop calls `generate(seed=42, …)` twice across two different training epochs. Both calls return identical `GoalSpec`s — that is the contract. Upstream training code is responsible for using non-colliding seeds (e.g., `seed = epoch * 10_000 + step`); the generator does not deduplicate. Documented in the training spec (`docs/modules/training.md`, not here).

6. **Language weights sum ≠ 1.0.** Caller passes `{"en": 0.5, "hi": 0.3}` (sum 0.8). `generate` raises `InvalidLanguageWeightError`. Rationale: silent renormalization would mask curriculum-config bugs where a language is silently dropped. Caller must normalize explicitly.

7. **Template with zero variants for requested language.** `_pick_language` picks `"ta"` but the chosen template has no `language_variants["ta"]`. The generator **does not** resample language — that would bias the distribution. Instead it raises `NoVariantForLanguageError`. The template library invariant (enforced at `load_templates`) is **every template has ≥ 1 variant in every LanguageCode**; this exception is defense against YAML authoring regressions and is tested via a malformed fixture.

8. **Step-misaligned uniform range.** Template declares `low: 3000, high: 15000, step: 700`. `(15000-3000) % 700 ≠ 0` — the grid doesn't cleanly terminate at `high`. `load_templates` detects this at load time and raises `TemplateSchemaError`, preventing runtime surprise.

9. **Negative seed.** `generate(seed=-1, …)` — stable hash handles negatives fine (blake2b accepts any UTF-8 bytes), but by convention the env passes non-negative episode IDs. The generator does not reject negatives; it just uses them verbatim. Documented in the interface docstring.

10. **Very large seed (> 2^63).** Same as #9 — blake2b handles arbitrary strings. No overflow.

---

## 8. Examples

### 8.1 Stage-1 airline, English

```python
>>> W = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
>>> goal = generate(seed=42, stage=1, language_weights=W)
>>> goal.domain
'airline'
>>> goal.intent
'book_flight'
>>> goal.language
'en'
>>> goal.slots
{'from': 'HYD', 'to': 'BLR', 'when': '2026-05-02'}
>>> goal.constraints
{'budget_inr': 7500, 'time_window': 'evening'}
>>> goal.seed_utterance
'Book the cheapest flight from HYD to BLR on 2026-05-02, budget under ₹7500, departing evening'
```

Determinism check:

```python
>>> generate(42, 1, W) == generate(42, 1, W)
True
>>> generate(42, 1, W).seed_utterance == generate(42, 1, W).seed_utterance
True
```

### 8.2 Stage-3 restaurant, Hinglish, drift-compatible slot layout

```python
>>> W = {"en": 0.3, "hi": 0.2, "ta": 0.1, "kn": 0.1, "hinglish": 0.3}
>>> goal = generate(seed=42, stage=3, language_weights=W)
>>> goal.domain
'restaurant'
>>> goal.language
'hinglish'
>>> goal.slots
{'city': 'Mumbai', 'cuisine': 'Biryani', 'when': '2026-05-10T20:00'}
>>> goal.constraints
{'budget_inr': 400, 'veg_only': True, 'min_order_buffer': 100}
>>> goal.seed_utterance
"Bhai tonight Mumbai mein Biryani order karna hai, 400 rupees se kam, veg option chahiye"
```

This brief's slot surface (`budget_inr` + `veg_only`) overlaps the drift patterns `restaurant.min_order_bump` and `restaurant.veg_filter_semantic` (DESIGN.md §5.3) — so when `drift_injector` selects a Stage-3 compound drift, the agent's goal is genuinely affected. That is what "drift-compatible slot layout" means.

### 8.3 Kannada utterance (Unicode-correct Kannada script, U+0C80–U+0CFF)

```python
>>> W = {"kn": 1.0, "en": 0.0, "hi": 0.0, "ta": 0.0, "hinglish": 0.0}
>>> goal = generate(seed=7, stage=2, language_weights=W)
>>> goal.domain
'airline'
>>> goal.language
'kn'
>>> goal.slots
{'from': 'BLR', 'to': 'MAA', 'when': '2026-05-08'}
>>> goal.constraints
{'budget_inr': 5500}
>>> goal.seed_utterance
'2026-05-08 ರಂದು BLR ಇಂದ MAA ಗೆ ಅಗ್ಗದ ವಿಮಾನ ಟಿಕೆಟ್ ಬೇಕು, 5500 ರೂಪಾಯಿಗಳ ಒಳಗೆ'
>>> import unicodedata
>>> unicodedata.is_normalized("NFC", goal.seed_utterance)
True
>>> # At least one codepoint in the Kannada block (U+0C80–U+0CFF)
>>> any(0x0C80 <= ord(c) <= 0x0CFF for c in goal.seed_utterance)
True
>>> # No Devanagari codepoints leaked in (U+0900–U+097F)
>>> any(0x0900 <= ord(c) <= 0x097F for c in goal.seed_utterance)
False
```

This example uses the genuine-Kannada-script variant declared in §4.1. City codes (`BLR`, `MAA`) remain in Roman because IATA/AAI airport codes are canonical identifiers in every language; full Kannada place names (`ಬೆಂಗಳೂರು`, `ಚೆನ್ನೈ`) are available in `i18n.yaml` and used by variants that reference `{from_city_local}` instead of `{from}`.

### 8.4 Tamil utterance with Devanagari-free script

```python
>>> W = {"ta": 1.0, "en": 0.0, "hi": 0.0, "kn": 0.0, "hinglish": 0.0}
>>> goal = generate(seed=101, stage=2, language_weights=W)
>>> goal.language
'ta'
>>> goal.seed_utterance
'2026-05-04 அன்று HYD லிருந்து BLR க்கு டிக்கெட் வேண்டும், 6500 ரூபாய்க்கு கீழ்'
>>> unicodedata.is_normalized("NFC", goal.seed_utterance)
True
>>> # No Devanagari codepoints (U+0900–U+097F) present
>>> any(0x0900 <= ord(c) <= 0x097F for c in goal.seed_utterance)
False
```

### 8.5 Hindi utterance (Devanagari)

```python
>>> W = {"hi": 1.0, "en": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
>>> goal = generate(seed=5, stage=1, language_weights=W)
>>> goal.language
'hi'
>>> goal.seed_utterance
'मुझे 2026-05-01 को DEL से BOM जाना है, 6000 रुपये से कम में'
```

---

## 9. Open Questions

None — spec is complete.

All decisions referenced in §§1–8 follow DESIGN.md §4.1, §4.2, §8.3, §8.4, §10.3 without extension. The generator is a pure function of its inputs; no side effects, no mutable global state, no dependencies on drift or reward subsystems. Edge cases 1–10 cover the full error surface identified during review.

Cross-doc references established:

- `docs/modules/models.md` — `GoalSpec`, `LanguageCode`, `Domain` definitions
- `docs/modules/drift_injector.md` — consumes `GoalSpec.domain` and template `drift_slot_tags` to schedule drifts
- `docs/modules/env.md` — calls `generate()` from `DriftCallEnv.reset()`
- `docs/modules/rewards.md` — consumes `GoalSpec.slots` + `GoalSpec.constraints` for R1 and R3
- `docs/modules/datasets.md` — calls `enumerate_variants()` to package HF Hub dataset
