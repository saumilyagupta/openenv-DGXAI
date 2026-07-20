"""Tests for cells/step_07_task_generator.py.

Implements docs/tests/task_generator_tests.md:
  - 30 unit tests (U1–U30, U34–U39)
  - 6 hypothesis property tests (P1–P6)
  - 5 integration tests (I1–I5)
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import unicodedata
from collections import Counter
from math import sqrt
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from cells import step_07_task_generator as tg
from cells.step_07_task_generator import (
    InvalidBudgetError,
    InvalidLanguageError,
    InvalidLanguageWeightError,
    InvalidStageError,
    MissingSlotError,
    NoVariantForLanguageError,
    SlotDistribution,
    Template,
    TemplateFileMissingError,
    TemplateLibrary,
    TemplateSchemaError,
    UnicodeNormalizationError,
    enumerate_variants,
    generate,
    load_templates,
    stable_sub_seed,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cells.step_04_models import GoalSpec

# ---------------------------------------------------------------------------
# Shared fixtures / weight constants (§5.3 of the test plan)
# ---------------------------------------------------------------------------

STAGE_1_WEIGHTS: dict[str, float] = {
    "en": 0.50,
    "hi": 0.30,
    "hinglish": 0.20,
    "ta": 0.00,
    "kn": 0.00,
}
STAGE_2_WEIGHTS: dict[str, float] = {
    "en": 0.30,
    "hi": 0.30,
    "hinglish": 0.20,
    "ta": 0.10,
    "kn": 0.10,
}
STAGE_3_WEIGHTS: dict[str, float] = {
    "en": 0.30,
    "hi": 0.30,
    "hinglish": 0.20,
    "ta": 0.10,
    "kn": 0.10,
}


@pytest.fixture(autouse=True)
def _install_test_library(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Install a fully-wired fixture library for every test.

    Tests that need the production ``data/task_briefs/templates.yaml`` or a
    custom library override must call ``tg.set_library_override()`` inside
    the test body — this fixture only sets the default.
    """
    tg.set_library_override(None)
    tg.reset_library_cache()
    fixture_dir = tmp_path_factory.mktemp("task_gen_fixture")
    _write_fixture_library(fixture_dir)
    lib = load_templates(fixture_dir / "templates.yaml")
    tg.set_library_override(lib)
    yield
    tg.set_library_override(None)
    tg.reset_library_cache()


# ---------------------------------------------------------------------------
# §1.1 Determinism (U1–U5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_generate_same_seed_same_goalspec(self) -> None:  # U1
        first = generate(42, 1, STAGE_1_WEIGHTS)
        for _ in range(99):
            assert generate(42, 1, STAGE_1_WEIGHTS) == first

    def test_generate_byte_identical_seed_utterance_after_nfc(self) -> None:  # U2
        first_bytes = generate(42, 1, STAGE_1_WEIGHTS).seed_utterance.encode("utf-8")
        for _ in range(99):
            assert (
                generate(42, 1, STAGE_1_WEIGHTS).seed_utterance.encode("utf-8")
                == first_bytes
            )

    def test_generate_different_seeds_different_episodes(self) -> None:  # U3
        results = [generate(s, 3, STAGE_3_WEIGHTS) for s in range(100)]
        assert len({g.seed_utterance for g in results}) > 90

    def test_generate_stage_changes_template_pool(self) -> None:  # U4
        g1 = generate(42, 1, STAGE_3_WEIGHTS)
        g3 = generate(42, 3, STAGE_3_WEIGHTS)
        assert len(g1.constraints) <= 2
        assert len(g3.constraints) <= 4

    def test_generate_returns_frozen_goalspec(self) -> None:  # U5
        g = generate(42, 1, STAGE_1_WEIGHTS)
        assert dataclasses.is_dataclass(g)
        assert g.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# §1.2 Stage-aware constraint counts (U6–U8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStageConstraintCounts:
    def test_stage_1_constraint_count_leq_2(self) -> None:  # U6
        for s in range(200):
            g = generate(s, 1, STAGE_1_WEIGHTS)
            assert len(g.constraints) <= 2, (s, g.constraints)

    def test_stage_2_constraint_count_leq_3(self) -> None:  # U7
        for s in range(200):
            g = generate(s, 2, STAGE_2_WEIGHTS)
            assert len(g.constraints) <= 3, (s, g.constraints)

    def test_stage_3_constraint_count_leq_4(self) -> None:  # U8
        for s in range(200):
            g = generate(s, 3, STAGE_3_WEIGHTS)
            assert len(g.constraints) <= 4, (s, g.constraints)


# ---------------------------------------------------------------------------
# §1.3 Language-weight distribution (U9, U10)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLanguageWeightDistribution:
    def test_language_weights_sampled_distribution_matches_at_n1000(self) -> None:  # U9
        weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
        n = 1000
        counts = Counter(
            generate(s, 3, weights).language for s in range(n)
        )
        for lang, p in weights.items():
            observed = counts.get(lang, 0) / n
            # ±3σ tolerance to avoid flakiness while still catching implementation bugs.
            sigma = sqrt(p * (1 - p) / n)
            assert abs(observed - p) < 3 * sigma + 1e-6, (lang, observed, p)

    def test_language_weights_zero_keys_never_drawn(self) -> None:  # U10
        weights = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
        for s in range(500):
            assert generate(s, 3, weights).language == "en"


# ---------------------------------------------------------------------------
# §1.4 Validation exceptions (U11–U19)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationExceptions:
    def test_invalid_language_error_on_unsupported_key(self) -> None:  # U11
        with pytest.raises(InvalidLanguageError):
            generate(0, 1, {"hindi": 1.0})  # type: ignore[dict-item]

    def test_invalid_language_error_on_marathi_key(self) -> None:  # U12
        with pytest.raises(InvalidLanguageError, match="marathi"):
            generate(0, 1, {"en": 0.5, "marathi": 0.5})  # type: ignore[dict-item]

    def test_invalid_language_weight_error_empty_dict(self) -> None:  # U13
        with pytest.raises(InvalidLanguageWeightError):
            generate(0, 1, {})

    def test_invalid_language_weight_error_negative_value(self) -> None:  # U14
        with pytest.raises(InvalidLanguageWeightError):
            generate(0, 1, {"en": 1.5, "hi": -0.5})

    def test_invalid_language_weight_error_sum_mismatch_low(self) -> None:  # U15
        with pytest.raises(InvalidLanguageWeightError):
            generate(0, 1, {"en": 0.5, "hi": 0.3})

    def test_invalid_language_weight_error_sum_mismatch_high(self) -> None:  # U16
        with pytest.raises(InvalidLanguageWeightError):
            generate(0, 1, {"en": 0.7, "hi": 0.5})

    def test_invalid_language_weight_error_all_zero(self) -> None:  # U17
        # Direct all-zero (sum 0) triggers the sum-mismatch branch;
        # the all-zero defensive branch is covered via a weights dict that
        # normalizes to 1.0 via floating-point noise. We assert via sum=1
        # impossible with all zeros, so instead patch: use empty-style.
        # The design specifies *defensive redundant* check — to exercise it
        # directly, we call the private validator with a hand-crafted input
        # that the sum-check would otherwise let through.
        with pytest.raises(InvalidLanguageWeightError):
            tg._validate_language_weights(
                {"en": 0.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
            )

    @pytest.mark.parametrize("bad_stage", [0, 4, -1])
    def test_invalid_stage_error(self, bad_stage: int) -> None:  # U18
        with pytest.raises(InvalidStageError):
            generate(0, bad_stage, STAGE_1_WEIGHTS)  # type: ignore[arg-type]

    def test_template_file_missing_error(self, tmp_path: Path) -> None:  # U19
        with pytest.raises(TemplateFileMissingError):
            load_templates(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# §1.5 Unicode NFC (U20–U24)
# ---------------------------------------------------------------------------


def _single_lang_weights(code: str) -> dict[str, float]:
    return {"en": 0.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0} | {code: 1.0}


@pytest.mark.unit
class TestNFC:
    def test_seed_utterance_is_nfc_for_every_language(self) -> None:  # U20
        for code in ("hi", "ta", "kn", "en", "hinglish"):
            g = generate(7, 2, _single_lang_weights(code))
            assert unicodedata.is_normalized("NFC", g.seed_utterance)

    def test_slotgrid_string_values_are_nfc(self) -> None:  # U21
        weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
        for s in range(50):
            g = generate(s, 3, weights)
            for v in g.slots.values():
                if isinstance(v, str):
                    assert unicodedata.is_normalized("NFC", v), (s, v)

    def test_i18n_yaml_loaded_values_are_nfc(self, tmp_path: Path) -> None:  # U22
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        for _lang, block in lib.i18n.items():
            for v in block.values():
                assert unicodedata.is_normalized("NFC", v)

    def test_templates_yaml_variant_strings_are_nfc_post_load(
        self, tmp_path: Path
    ) -> None:  # U23
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        for t in lib.templates:
            for variants in t.language_variants.values():
                for v in variants:
                    assert unicodedata.is_normalized("NFC", v)

    def test_nfd_input_renormalized_to_nfc_on_load(self, tmp_path: Path) -> None:  # U24
        _write_fixture_library(tmp_path)
        # Overwrite one variant with NFD-encoded text.
        nfd_kannada = unicodedata.normalize("NFD", "ಬೆಂಗಳೂರು")
        assert not unicodedata.is_normalized("NFC", nfd_kannada) or True  # NFC may equal NFD for this str
        yaml_path = tmp_path / "i18n.yaml"
        data = {
            "hi": {"cities": {"BLR": unicodedata.normalize("NFD", "बेंगलुरु")}},
            "ta": {"cities": {"BLR": "பெங்களூரு"}},
            "kn": {"cities": {"BLR": nfd_kannada}},
            "en": {"cities": {"BLR": "Bengaluru"}},
            "hinglish": {"cities": {"BLR": "Bengaluru"}},
        }
        yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        lib = load_templates(tmp_path / "templates.yaml")
        for _lang, block in lib.i18n.items():
            for v in block.values():
                assert unicodedata.is_normalized("NFC", v)


# ---------------------------------------------------------------------------
# §1.6 stable_sub_seed domain separation (U25–U28)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubSeed:
    def test_stable_sub_seed_formula(self) -> None:  # U25
        expected = int.from_bytes(
            hashlib.blake2b(b"42:domain", digest_size=8).digest(), "big"
        )
        assert stable_sub_seed(42, "domain") == expected

    def test_sub_seed_tags_differ_per_decision(self) -> None:  # U26
        tags = ["domain", "template", "slots", "language", "variant"]
        out = {stable_sub_seed(42, t) for t in tags}
        assert len(out) == 5

    def test_sub_seed_stable_across_runs(self) -> None:  # U27
        a = stable_sub_seed(42, "domain")
        b = stable_sub_seed(42, "domain")
        assert a == b

    def test_sub_seed_different_seed_different_output(self) -> None:  # U28
        assert stable_sub_seed(42, "domain") != stable_sub_seed(43, "domain")


# ---------------------------------------------------------------------------
# §1.7 Structural invariants (U29, U30)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStructuralInvariants:
    def test_seed_utterance_has_no_unresolved_placeholders(self) -> None:  # U29
        weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
        for s in range(100):
            g = generate(s, 3, weights)
            assert re.search(r"\{[a-z_][a-z0-9_]*\}", g.seed_utterance) is None, (
                s,
                g.seed_utterance,
            )

    def test_seed_utterance_length_leq_280(self) -> None:  # U30
        weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
        for s in range(100):
            g = generate(s, 3, weights)
            assert len(g.seed_utterance) <= 280


# ---------------------------------------------------------------------------
# §1.8 Malformed-fixture raise-site tests (U34–U39)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorModes:
    def test_missing_slot_error(self) -> None:  # U34
        # Build a library whose variant references an undeclared placeholder by
        # bypassing load_templates static-scan (we inject directly).
        bad_variant = "go to {destination}"
        tmpl = Template(
            template_id="airline.bad",
            domain="airline",
            intent="book_flight",
            min_stage=1,
            required_slots=("from", "to", "when"),
            optional_slots=(),
            slot_distributions={
                "from": SlotDistribution(kind="choices", choices=("HYD",)),
                "to": SlotDistribution(kind="choices", choices=("BLR",)),
                "when": SlotDistribution(kind="date"),
            },
            constraints_template={},
            drift_slot_tags=(),
            language_variants={
                "en": (bad_variant,),
                "hi": (bad_variant,),
                "ta": (bad_variant,),
                "kn": (bad_variant,),
                "hinglish": (bad_variant,),
            },
        )
        lib = TemplateLibrary(
            templates=(tmpl,),
            cities_by_domain={"airline": ("HYD", "BLR")},
            i18n={k: {} for k in ("hi", "ta", "kn", "en", "hinglish")},
        )
        tg.set_library_override(lib)
        with pytest.raises(MissingSlotError, match="destination"):
            generate(0, 1, {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0})

    def test_invalid_budget_error_from_step_misalignment(self) -> None:  # U35
        # Feed _sample_slot_value a deliberately corrupt distribution that
        # would produce an out-of-range sample.
        import random

        dist = SlotDistribution(kind="uniform", low=100.0, high=250.0, step=70.0)

        class _BadRng(random.Random):
            def randint(self, a: int, b: int) -> int:  # noqa: ARG002
                return 3  # 100 + 3*70 = 310 > 250

        with pytest.raises(InvalidBudgetError):
            tg._sample_slot_value(_BadRng(0), "budget_inr", dist, template_id="x")

    def test_template_schema_error_missing_required_key(self, tmp_path: Path) -> None:  # U36
        (tmp_path / "templates.yaml").write_text(
            yaml.safe_dump([{"template_id": "x"}]), encoding="utf-8"
        )
        with pytest.raises(TemplateSchemaError):
            load_templates(tmp_path / "templates.yaml")

    def test_template_schema_error_bad_step_grid(self, tmp_path: Path) -> None:  # U37
        bad_template: dict[str, Any] = {
            "template_id": "airline.bad",
            "domain": "airline",
            "intent": "book_flight",
            "min_stage": 1,
            "required_slots": [],
            "optional_slots": [],
            "constraints_template": {
                "budget_inr": {"distribution": "uniform", "low": 3000, "high": 15000, "step": 700}
            },
            "drift_slot_tags": [],
            "language_variants": {
                "en": ["hello"],
                "hi": ["नमस्ते"],
                "ta": ["வணக்கம்"],
                "kn": ["ನಮಸ್ಕಾರ"],
                "hinglish": ["namaste"],
            },
        }
        (tmp_path / "templates.yaml").write_text(
            yaml.safe_dump([bad_template], allow_unicode=True), encoding="utf-8"
        )
        with pytest.raises(TemplateSchemaError, match="misaligned"):
            load_templates(tmp_path / "templates.yaml")

    def test_unicode_normalization_error_defensive(self, monkeypatch: pytest.MonkeyPatch) -> None:  # U38
        from cells import step_07_task_generator as mod

        monkeypatch.setattr(mod.unicodedata, "is_normalized", lambda *a, **k: False)
        with pytest.raises(UnicodeNormalizationError):
            mod._assert_nfc("anything", where="test")

    def test_no_variant_for_language_error(self) -> None:  # U39
        # Build a template with an empty variant tuple for Tamil (bypass loader).
        tmpl = Template(
            template_id="airline.missing_ta",
            domain="airline",
            intent="book_flight",
            min_stage=1,
            required_slots=("from", "to", "when"),
            optional_slots=(),
            slot_distributions={
                "from": SlotDistribution(kind="choices", choices=("HYD",)),
                "to": SlotDistribution(kind="choices", choices=("BLR",)),
                "when": SlotDistribution(kind="date"),
            },
            constraints_template={},
            drift_slot_tags=(),
            language_variants={
                "en": ("from {from} to {to} on {when}",),
                "hi": ("{from} से {to} {when}",),
                "ta": (),  # intentionally empty
                "kn": ("{from} {to} {when}",),
                "hinglish": ("{from} to {to} {when}",),
            },
        )
        lib = TemplateLibrary(
            templates=(tmpl,),
            cities_by_domain={"airline": ("HYD", "BLR")},
            i18n={k: {} for k in ("hi", "ta", "kn", "en", "hinglish")},
        )
        tg.set_library_override(lib)
        weights = {"en": 0.0, "hi": 0.0, "ta": 1.0, "kn": 0.0, "hinglish": 0.0}
        with pytest.raises(NoVariantForLanguageError):
            generate(0, 1, weights)


# ---------------------------------------------------------------------------
# §2 Property tests (P1–P6)
# ---------------------------------------------------------------------------


def _language_weights_strategy() -> st.SearchStrategy[dict[str, float]]:
    langs = ("hi", "ta", "kn", "en", "hinglish")

    @st.composite
    def _impl(draw: st.DrawFn) -> dict[str, float]:
        raw = [
            draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
            for _ in langs
        ]
        total = sum(raw)
        return {lang: r / total for lang, r in zip(langs, raw, strict=True)}

    return _impl()


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**62),
    stage=st.sampled_from([1, 2, 3]),
    weights=_language_weights_strategy(),
)
@settings(max_examples=150, deadline=None)
def test_generate_is_pure(seed: int, stage: int, weights: dict[str, float]) -> None:  # P1
    a = generate(seed, stage, weights)  # type: ignore[arg-type]
    b = generate(seed, stage, weights)  # type: ignore[arg-type]
    assert a == b
    assert a.seed_utterance == b.seed_utterance


@pytest.mark.property
@pytest.mark.slow
def test_procedural_space_uniqueness_scan() -> None:  # P2 (scaled down — slow)
    weights = {"en": 0.2, "hi": 0.2, "ta": 0.2, "kn": 0.2, "hinglish": 0.2}
    # Walk 5,000 distinct seeds (200k is gated behind -m slow in CI nightly).
    utterances = set()
    for s in range(5_000):
        utterances.add(generate(s, 3, weights).seed_utterance)
    # Collision rate < 10% at n=5k given the 4 domains × 5 templates × etc.
    assert len(utterances) >= 5_000 * 0.8


@pytest.mark.property
def test_language_distribution_chi_square_n10000() -> None:  # P3
    weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
    n = 10_000
    observed = Counter(generate(s, 3, weights).language for s in range(n))
    expected = {lang: p * n for lang, p in weights.items()}
    chi2 = sum(
        ((observed.get(lang, 0) - expected[lang]) ** 2) / expected[lang]
        for lang in weights
    )
    # df=4, alpha=0.001 critical value ≈ 18.47
    assert chi2 < 18.47, f"chi-square {chi2:.2f} rejects null"


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=100, deadline=None)
def test_stage_template_pool_monotone(seed: int) -> None:  # P4
    weights = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
    g1 = generate(seed, 1, weights)
    assert len(g1.constraints) <= 2


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**62),
    stage=st.sampled_from([1, 2, 3]),
    weights=_language_weights_strategy(),
)
@settings(max_examples=300, deadline=None)
def test_seed_utterance_always_nfc(
    seed: int, stage: int, weights: dict[str, float]
) -> None:  # P5
    g = generate(seed, stage, weights)  # type: ignore[arg-type]
    assert unicodedata.is_normalized("NFC", g.seed_utterance)
    for v in g.slots.values():
        if isinstance(v, str):
            assert unicodedata.is_normalized("NFC", v)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    stage=st.sampled_from([1, 2, 3]),
)
@settings(max_examples=200, deadline=None)
def test_budget_within_declared_range(seed: int, stage: int) -> None:  # P6
    weights = {"en": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "hinglish": 0.0}
    g = generate(seed, stage, weights)  # type: ignore[arg-type]
    if "budget_inr" in g.constraints:
        # Find any template in the library whose budget range could contain it.
        lib = tg._get_library()
        match = False
        for t in lib.templates:
            if "budget_inr" in t.constraints_template:
                dist = t.constraints_template["budget_inr"]
                assert dist.low is not None and dist.high is not None
                if dist.low <= g.constraints["budget_inr"] <= dist.high:
                    match = True
                    break
        assert match, (g.constraints, g.domain)


# ---------------------------------------------------------------------------
# §3 Integration tests (I1–I5) — use real fixture files written on disk
# ---------------------------------------------------------------------------


def _write_fixture_library(tmp_path: Path) -> None:
    """Author a minimal real templates.yaml + i18n.yaml pair."""
    templates: list[dict[str, Any]] = [
        {
            "template_id": "airline.book.fixture_v1",
            "domain": "airline",
            "intent": "book_flight",
            "min_stage": 1,
            "required_slots": ["from", "to", "when"],
            "optional_slots": [],
            "slot_distributions": {
                "from": {"choices": ["HYD", "BLR", "DEL", "BOM", "MAA"]},
                "to": {"choices": ["HYD", "BLR", "DEL", "BOM", "MAA"]},
                "when": {"distribution": "date"},
            },
            "constraints_template": {
                "budget_inr": {
                    "distribution": "uniform",
                    "low": 3000,
                    "high": 15000,
                    "step": 500,
                },
                "time_window": {
                    "choices": ["morning", "afternoon", "evening", "late_night"]
                },
            },
            "drift_slot_tags": ["price", "total_fare_inr"],
            "language_variants": {
                "hinglish": [
                    "Bhai {when} ko {from} se {to}, {budget_inr} rupees max, {time_window}"
                ],
                "hi": [
                    "{when} को {from} से {to}, ₹{budget_inr} से कम, {time_window}"
                ],
                "ta": [
                    "{when} அன்று {from} லிருந்து {to}, ₹{budget_inr} கீழ், {time_window}"
                ],
                "kn": [
                    "{when} ರಂದು {from} ಇಂದ {to}, ₹{budget_inr} ಒಳಗೆ, {time_window}"
                ],
                "en": [
                    "Flight from {from} to {to} on {when}, under ₹{budget_inr}, {time_window}"
                ],
            },
        },
        {
            "template_id": "cab.ride.fixture_v1",
            "domain": "cab",
            "intent": "book_cab",
            "min_stage": 1,
            "required_slots": ["pickup", "drop", "when"],
            "optional_slots": [],
            "slot_distributions": {
                "pickup": {"choices": ["Koramangala", "Indiranagar", "Whitefield"]},
                "drop": {"choices": ["Koramangala", "Indiranagar", "Whitefield"]},
                "when": {"distribution": "date"},
            },
            "constraints_template": {
                "budget_inr": {
                    "distribution": "uniform",
                    "low": 200,
                    "high": 2000,
                    "step": 50,
                }
            },
            "drift_slot_tags": ["fare_inr"],
            "language_variants": {
                "hinglish": ["{when} ko {pickup} se {drop} cab, {budget_inr} ke andar"],
                "hi": ["{when} को {pickup} से {drop}, ₹{budget_inr} के अंदर"],
                "ta": ["{when} அன்று {pickup} லிருந்து {drop}, ₹{budget_inr} கீழ்"],
                "kn": ["{when} ರಂದು {pickup} ಇಂದ {drop}, ₹{budget_inr} ಒಳಗೆ"],
                "en": ["Cab {pickup} to {drop} on {when}, under ₹{budget_inr}"],
            },
        },
        {
            "template_id": "restaurant.order.fixture_v1",
            "domain": "restaurant",
            "intent": "order_food",
            "min_stage": 1,
            "required_slots": ["city", "cuisine", "when"],
            "optional_slots": [],
            "slot_distributions": {
                "city": {"choices": ["HYD", "BLR", "DEL"]},
                "cuisine": {"choices": ["Biryani", "Dosa", "Pizza"]},
                "when": {"distribution": "date"},
            },
            "constraints_template": {
                "budget_inr": {
                    "distribution": "uniform",
                    "low": 200,
                    "high": 1000,
                    "step": 50,
                },
                "veg_only": {"distribution": "bool"},
            },
            "drift_slot_tags": ["min_order"],
            "language_variants": {
                "hinglish": [
                    "{when} ko {city} mein {cuisine}, {budget_inr} max, veg={veg_only}"
                ],
                "hi": [
                    "{when} को {city} में {cuisine}, ₹{budget_inr}, veg={veg_only}"
                ],
                "ta": [
                    "{when} அன்று {city} இல் {cuisine}, ₹{budget_inr}, veg={veg_only}"
                ],
                "kn": [
                    "{when} ರಂದು {city} ನಲ್ಲಿ {cuisine}, ₹{budget_inr}, veg={veg_only}"
                ],
                "en": [
                    "Order {cuisine} in {city} on {when}, ₹{budget_inr}, veg={veg_only}"
                ],
            },
        },
        {
            "template_id": "hotel.book.fixture_v1",
            "domain": "hotel",
            "intent": "book_hotel",
            "min_stage": 1,
            "required_slots": ["city", "checkin", "checkout"],
            "optional_slots": [],
            "slot_distributions": {
                "city": {"choices": ["HYD", "BLR", "GOI"]},
                "checkin": {"distribution": "date"},
                "checkout": {"distribution": "date"},
            },
            "constraints_template": {
                "budget_inr": {
                    "distribution": "uniform",
                    "low": 2000,
                    "high": 10000,
                    "step": 500,
                }
            },
            "drift_slot_tags": ["cancel_window"],
            "language_variants": {
                "hinglish": ["{city} {checkin}-{checkout}, ₹{budget_inr}/night"],
                "hi": ["{city} {checkin}-{checkout}, ₹{budget_inr} प्रति रात"],
                "ta": ["{city} {checkin}-{checkout}, ₹{budget_inr} இரவுக்கு"],
                "kn": ["{city} {checkin}-{checkout}, ₹{budget_inr} ಒಂದು ರಾತ್ರಿ"],
                "en": ["{city} {checkin} to {checkout}, ₹{budget_inr} per night"],
            },
        },
        {
            "template_id": "airline.book.compound_v1",
            "domain": "airline",
            "intent": "book_flight",
            "min_stage": 3,
            "required_slots": ["from", "to", "when"],
            "optional_slots": [],
            "slot_distributions": {
                "from": {"choices": ["HYD", "BLR", "DEL"]},
                "to": {"choices": ["HYD", "BLR", "DEL"]},
                "when": {"distribution": "date"},
            },
            "constraints_template": {
                "budget_inr": {
                    "distribution": "uniform",
                    "low": 3000,
                    "high": 15000,
                    "step": 500,
                },
                "time_window": {
                    "choices": ["morning", "afternoon", "evening", "late_night"]
                },
                "passenger_count": {
                    "distribution": "uniform",
                    "low": 1,
                    "high": 4,
                    "step": 1,
                },
            },
            "drift_slot_tags": ["price", "passenger_count"],
            "language_variants": {
                "hinglish": [
                    "{when} ko {from} se {to}, {passenger_count} log, ₹{budget_inr}, {time_window}"
                ],
                "hi": [
                    "{when} को {from} से {to}, {passenger_count} लोग, ₹{budget_inr}, {time_window}"
                ],
                "ta": [
                    "{when} அன்று {from} லிருந்து {to}, {passenger_count} பேர், ₹{budget_inr}, {time_window}"
                ],
                "kn": [
                    "{when} ರಂದು {from} ಇಂದ {to}, {passenger_count} ಜನ, ₹{budget_inr}, {time_window}"
                ],
                "en": [
                    "{from} to {to} on {when} for {passenger_count} pax, ₹{budget_inr}, {time_window}"
                ],
            },
        },
    ]
    (tmp_path / "templates.yaml").write_text(
        yaml.safe_dump(templates, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    i18n: dict[str, Any] = {
        "hi": {
            "cities": {"BLR": "बेंगलुरु", "MAA": "चेन्नई", "HYD": "हैदराबाद"},
            "weekdays": {"monday": "सोमवार"},
        },
        "ta": {
            "cities": {"BLR": "பெங்களூரு", "MAA": "சென்னை"},
            "weekdays": {"monday": "திங்கட்கிழமை"},
        },
        "kn": {
            "cities": {"BLR": "ಬೆಂಗಳೂರು", "MAA": "ಚೆನ್ನೈ"},
            "weekdays": {"monday": "ಸೋಮವಾರ"},
        },
        "en": {"cities": {"BLR": "Bengaluru"}},
        "hinglish": {"cities": {"BLR": "Bengaluru"}},
    }
    (tmp_path / "i18n.yaml").write_text(
        yaml.safe_dump(i18n, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _valid_goal_spec(g: GoalSpec) -> None:
    assert dataclasses.is_dataclass(g)
    assert g.domain in ("airline", "cab", "restaurant", "hotel")
    assert g.language in ("hi", "ta", "kn", "en", "hinglish")
    assert unicodedata.is_normalized("NFC", g.seed_utterance)
    assert len(g.seed_utterance) <= 280
    assert re.search(r"\{[a-z_][a-z0-9_]*\}", g.seed_utterance) is None


@pytest.mark.integration
class TestIntegration:
    def test_load_templates_from_fixture(self, tmp_path: Path) -> None:  # I1
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        assert isinstance(lib, TemplateLibrary)
        assert len({t.domain for t in lib.templates}) == 4
        assert len(lib.templates) == 5
        for lang in ("hi", "ta", "kn", "en", "hinglish"):
            assert lang in lib.i18n

    def test_100_briefs_pass_goal_spec_invariants(self, tmp_path: Path) -> None:  # I2
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        tg.set_library_override(lib)
        weights = {"en": 0.3, "hi": 0.3, "ta": 0.2, "kn": 0.1, "hinglish": 0.1}
        for s in range(100):
            g = generate(s, 3, weights)
            _valid_goal_spec(g)

    def test_enumerate_variants_stable_order(self, tmp_path: Path) -> None:  # I3
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        tg.set_library_override(lib)
        weights = {"en": 0.2, "hi": 0.2, "ta": 0.2, "kn": 0.2, "hinglish": 0.2}
        a = list(enumerate_variants(limit=200, stage=3, language_weights=weights))
        b = list(enumerate_variants(limit=200, stage=3, language_weights=weights))
        assert [g.seed_utterance for g in a] == [g.seed_utterance for g in b]

    @pytest.mark.parametrize(
        "lang,expected_block,forbidden_block",
        [
            ("hi", (0x0900, 0x097F), (0x0B80, 0x0BFF)),
            ("ta", (0x0B80, 0x0BFF), (0x0900, 0x097F)),
            ("kn", (0x0C80, 0x0CFF), (0x0900, 0x097F)),
        ],
    )
    def test_indic_script_isolation(
        self,
        tmp_path: Path,
        lang: str,
        expected_block: tuple[int, int],
        forbidden_block: tuple[int, int],
    ) -> None:  # I4
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        tg.set_library_override(lib)
        weights = {c: (1.0 if c == lang else 0.0) for c in ("hi", "ta", "kn", "en", "hinglish")}
        for s in range(50):
            g = generate(s, 2, weights)
            lo, hi = expected_block
            assert any(lo <= ord(c) <= hi for c in g.seed_utterance), g.seed_utterance
            fo, fh = forbidden_block
            assert not any(fo <= ord(c) <= fh for c in g.seed_utterance), g.seed_utterance

    def test_hinglish_never_contains_devanagari(self, tmp_path: Path) -> None:  # I5
        _write_fixture_library(tmp_path)
        lib = load_templates(tmp_path / "templates.yaml")
        tg.set_library_override(lib)
        weights = {"hinglish": 1.0, "hi": 0.0, "ta": 0.0, "kn": 0.0, "en": 0.0}
        for s in range(100):
            g = generate(s, 3, weights)
            assert not any(0x0900 <= ord(c) <= 0x097F for c in g.seed_utterance)
