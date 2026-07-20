"""Tests for cells/step_03_fixtures.py and on-disk data artifacts.

Covers datasets.md §3.3 schema validation, §3.5 invariants, and the error
modes in §5.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
import yaml

from cells.step_03_fixtures import (
    APISchemaRegistry,
    DatasetFileMissingError,
    DatasetSchemaError,
    DriftPatternLibrary,
    DriftPatternOrphanError,
    DuplicateDriftPatternIdError,
    I18nLibrary,
    MalformedJSONError,
    MalformedYAMLError,
    SlotDistribution,
    TemplateLibrary,
    UnknownLanguageKeyError,
    _reset_caches,
    load_api_schemas,
    load_drift_patterns,
    load_i18n,
    load_templates,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _REPO_ROOT / "data"
_EXPECTED_LANGUAGES = {"hi", "ta", "kn", "en", "hinglish"}
_EXPECTED_DOMAINS = {"airline", "cab", "restaurant", "hotel"}
_EXPECTED_PATTERN_IDS = {
    # 5 schema
    "airline.price_rename",
    "airline.pax_required",
    "cab.fare_breakdown",
    "restaurant.items_shape_bump",
    "hotel.gst_field",
    # 5 policy
    "airline.booking_window_shrink",
    "cab.school_hours_mini_reject",
    "restaurant.min_order_bump",
    "hotel.cancel_window_shrink",
    "cab.vehicle_class_expand",
    # 5 T&C
    "airline.baggage_tnc_rewrite",
    "cab.surge_policy_tnc",
    "restaurant.veg_filter_semantic",
    "hotel.early_checkin_tnc",
    "airline.reschedule_tnc",
    # 3 pricing
    "airline.convenience_fee_append",
    "cab.toll_unbundle",
    "hotel.resort_fee_append",
    # 2 transversal payment-auth
    "payment.auth_scope_upgrade",
    "payment.mfa_required",
}


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    _reset_caches()


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_load_templates_smoke() -> None:
    lib = load_templates(_DATA_ROOT / "task_briefs" / "templates.yaml")
    assert isinstance(lib, TemplateLibrary)
    assert len(lib.templates) >= 5
    domains = {t.domain for t in lib.templates}
    assert _EXPECTED_DOMAINS.issubset(domains)
    for tpl in lib.templates:
        assert set(tpl.language_variants.keys()) == _EXPECTED_LANGUAGES
        for variants in tpl.language_variants.values():
            assert len(variants) >= 1
    assert len(lib.source_sha256) == 64


def test_templates_contain_compound_stage3() -> None:
    lib = load_templates(_DATA_ROOT / "task_briefs" / "templates.yaml")
    stages = {t.min_stage for t in lib.templates}
    assert 3 in stages


def test_templates_distributions_well_formed() -> None:
    lib = load_templates(_DATA_ROOT / "task_briefs" / "templates.yaml")
    for tpl in lib.templates:
        assert len(tpl.constraints_template) >= 1
        for slot_name, dist in tpl.constraints_template.items():
            assert isinstance(dist, SlotDistribution)
            assert slot_name.strip() != ""
            if dist.kind == "uniform":
                assert dist.low is not None and dist.high is not None and dist.step is not None
                assert dist.high >= dist.low
                assert dist.step > 0
            else:
                assert dist.choices is not None and len(dist.choices) >= 1


def test_templates_all_strings_nfc() -> None:
    lib = load_templates(_DATA_ROOT / "task_briefs" / "templates.yaml")
    for tpl in lib.templates:
        for variants in tpl.language_variants.values():
            for v in variants:
                assert unicodedata.is_normalized("NFC", v), v


def test_templates_hash_stable() -> None:
    path = _DATA_ROOT / "task_briefs" / "templates.yaml"
    lib1 = load_templates(path)
    _reset_caches()
    lib2 = load_templates(path)
    assert lib1.source_sha256 == lib2.source_sha256


def test_templates_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetFileMissingError):
        load_templates(tmp_path / "nope.yaml")


def test_templates_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- template_id: foo\n  domain: [\n")
    with pytest.raises(MalformedYAMLError):
        load_templates(bad)


def test_templates_unknown_language_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    doc = [
        {
            "template_id": "airline.book.x",
            "domain": "airline",
            "intent": "book_flight",
            "min_stage": 1,
            "required_slots": ["from", "to", "when"],
            "optional_slots": [],
            "constraints_template": {
                "budget_inr": {"distribution": "uniform", "low": 100, "high": 200, "step": 10}
            },
            "drift_slot_tags": ["price"],
            "language_variants": {
                "marathi": ["test"],
            },
        }
    ]
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(UnknownLanguageKeyError):
        load_templates(bad)


def test_templates_missing_language_variant_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    doc = [
        {
            "template_id": "airline.book.x",
            "domain": "airline",
            "intent": "book_flight",
            "min_stage": 1,
            "required_slots": ["from", "to", "when"],
            "optional_slots": [],
            "constraints_template": {
                "budget_inr": {"distribution": "uniform", "low": 100, "high": 200, "step": 10}
            },
            "drift_slot_tags": ["price"],
            "language_variants": {
                "hi": ["x"],
                "ta": ["y"],
                "kn": ["z"],
                "en": ["a"],
                # missing hinglish
            },
        }
    ]
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(DatasetSchemaError):
        load_templates(bad)


def test_templates_unknown_domain_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    doc = [
        {
            "template_id": "foo.bar.x",
            "domain": "spaceship",
            "intent": "book_flight",
            "min_stage": 1,
            "required_slots": ["from"],
            "optional_slots": [],
            "constraints_template": {
                "budget_inr": {"distribution": "uniform", "low": 100, "high": 200, "step": 10}
            },
            "drift_slot_tags": ["price"],
            "language_variants": {k: ["x"] for k in _EXPECTED_LANGUAGES},
        }
    ]
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(DatasetSchemaError):
        load_templates(bad)


# ---------------------------------------------------------------------------
# I18n
# ---------------------------------------------------------------------------


def test_load_i18n_smoke() -> None:
    lib = load_i18n(_DATA_ROOT / "task_briefs" / "i18n.yaml")
    assert isinstance(lib, I18nLibrary)
    assert set(lib.strings.keys()) == _EXPECTED_LANGUAGES
    for entries in lib.strings.values():
        assert "BLR" in entries
        assert "Monday" in entries
    assert len(lib.source_sha256) == 64


def test_i18n_strings_nfc() -> None:
    lib = load_i18n(_DATA_ROOT / "task_briefs" / "i18n.yaml")
    for per_lang in lib.strings.values():
        for k, v in per_lang.items():
            assert unicodedata.is_normalized("NFC", k)
            assert unicodedata.is_normalized("NFC", v)


def test_i18n_hindi_devanagari_range() -> None:
    lib = load_i18n(_DATA_ROOT / "task_briefs" / "i18n.yaml")
    hi_blr = lib.strings["hi"]["BLR"]
    assert any(0x0900 <= ord(c) <= 0x097F for c in hi_blr)


def test_i18n_unknown_language_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"marathi": {"BLR": "x"}}))
    with pytest.raises(UnknownLanguageKeyError):
        load_i18n(bad)


def test_i18n_hash_stable() -> None:
    path = _DATA_ROOT / "task_briefs" / "i18n.yaml"
    lib1 = load_i18n(path)
    _reset_caches()
    lib2 = load_i18n(path)
    assert lib1.source_sha256 == lib2.source_sha256


def test_i18n_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetFileMissingError):
        load_i18n(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Drift patterns
# ---------------------------------------------------------------------------


def test_load_drift_patterns_smoke() -> None:
    lib = load_drift_patterns(_DATA_ROOT / "drift_patterns" / "drifts.yaml")
    assert isinstance(lib, DriftPatternLibrary)
    assert len(lib.patterns) == 20
    assert set(lib.patterns.keys()) == _EXPECTED_PATTERN_IDS


def test_drift_type_counts_match_spec() -> None:
    lib = load_drift_patterns(_DATA_ROOT / "drift_patterns" / "drifts.yaml")
    counts = {k: len(v) for k, v in lib.by_type.items()}
    assert counts["schema"] == 5
    assert counts["policy"] == 5
    assert counts["tnc"] == 5
    assert counts["pricing"] == 3
    assert counts["auth"] == 2


def test_drift_patterns_detection_hints_non_empty() -> None:
    lib = load_drift_patterns(_DATA_ROOT / "drift_patterns" / "drifts.yaml")
    for p in lib.patterns.values():
        assert len(p.detection_hints) >= 1
        for h in p.detection_hints:
            assert h == unicodedata.normalize("NFC", h)
            assert h.strip() != ""


def test_drift_patterns_reference_real_schemas() -> None:
    lib = load_drift_patterns(_DATA_ROOT / "drift_patterns" / "drifts.yaml")
    registry = load_api_schemas(_DATA_ROOT / "api_schemas")
    for p in lib.patterns.values():
        for ver in (p.from_version, p.to_version):
            assert p.domain in registry.schemas
            assert ver in registry.schemas[p.domain]


def test_drift_patterns_orphan_raises(tmp_path: Path) -> None:
    bad = tmp_path / "drifts.yaml"
    doc = [
        {
            "id": "airline.nonexistent",
            "drift_type": "schema",
            "domain": "airline",
            "from_version": "v1",
            "to_version": "v99",
            "description": "bogus",
            "mutation": {"rename": {"x": "y"}},
            "detection_hints": ["x"],
        }
    ] + [
        {
            "id": f"filler.{i}",
            "drift_type": "schema",
            "domain": "airline",
            "from_version": "v1",
            "to_version": "v2",
            "description": "d",
            "mutation": {"rename": {"x": "y"}},
            "detection_hints": ["x"],
        }
        for i in range(19)
    ]
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(DriftPatternOrphanError):
        load_drift_patterns(bad)


def test_drift_patterns_wrong_count_raises(tmp_path: Path) -> None:
    bad = tmp_path / "drifts.yaml"
    bad.write_text(yaml.safe_dump([]))
    with pytest.raises(DatasetSchemaError):
        load_drift_patterns(bad)


def test_drift_patterns_duplicate_id_raises(tmp_path: Path) -> None:
    bad = tmp_path / "drifts.yaml"
    entry = {
        "id": "airline.price_rename",
        "drift_type": "schema",
        "domain": "airline",
        "from_version": "v1",
        "to_version": "v2",
        "description": "dup",
        "mutation": {"rename": {"a": "b"}},
        "detection_hints": ["a"],
    }
    doc = [dict(entry) for _ in range(20)]
    bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(DuplicateDriftPatternIdError):
        load_drift_patterns(bad)


def test_drift_patterns_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "drifts.yaml"
    bad.write_text("- id: foo\n  mutation: [unterminated\n")
    with pytest.raises(MalformedYAMLError):
        load_drift_patterns(bad)


def test_drift_patterns_hash_stable() -> None:
    path = _DATA_ROOT / "drift_patterns" / "drifts.yaml"
    lib1 = load_drift_patterns(path)
    _reset_caches()
    lib2 = load_drift_patterns(path)
    assert lib1.source_sha256 == lib2.source_sha256


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


def test_load_api_schemas_smoke() -> None:
    registry = load_api_schemas(_DATA_ROOT / "api_schemas")
    assert isinstance(registry, APISchemaRegistry)
    total = sum(len(v) for v in registry.schemas.values())
    assert total == 14
    assert set(registry.schemas.keys()) == _EXPECTED_DOMAINS | {"payment"}
    for domain in ("airline", "cab", "restaurant", "hotel"):
        assert set(registry.schemas[domain].keys()) == {"v1", "v2", "v3"}
    assert set(registry.schemas["payment"].keys()) == {"v1", "v2"}


def test_api_schema_has_schema_field() -> None:
    registry = load_api_schemas(_DATA_ROOT / "api_schemas")
    s = registry.get("airline", "v1")
    assert s.schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
    assert len(s.source_sha256) == 64


def test_api_schema_get_unknown_raises() -> None:
    registry = load_api_schemas(_DATA_ROOT / "api_schemas")
    with pytest.raises(DatasetSchemaError):
        registry.get("airline", "v99")
    with pytest.raises(DatasetSchemaError):
        registry.versions("bogus")


def test_api_schemas_malformed_json_raises(tmp_path: Path) -> None:
    root = tmp_path / "api_schemas"
    (root / "airline").mkdir(parents=True)
    (root / "cab").mkdir()
    (root / "restaurant").mkdir()
    (root / "hotel").mkdir()
    (root / "payment").mkdir()
    (root / "airline" / "v1.json").write_text("{not json")
    with pytest.raises(MalformedJSONError):
        load_api_schemas(root)


def test_api_schemas_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetFileMissingError):
        load_api_schemas(tmp_path / "does_not_exist")


def test_api_schemas_missing_domain_dir_raises(tmp_path: Path) -> None:
    root = tmp_path / "api_schemas"
    (root / "airline").mkdir(parents=True)
    with pytest.raises(DatasetFileMissingError):
        load_api_schemas(root)


def test_api_schemas_invalid_json_schema_raises(tmp_path: Path) -> None:
    root = tmp_path / "api_schemas"
    for d in ("airline", "cab", "restaurant", "hotel", "payment"):
        (root / d).mkdir(parents=True)
    # Valid JSON, invalid JSON Schema 2020-12 (type is wrong kind)
    bad = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": 123}
    for d in ("airline", "cab", "restaurant", "hotel"):
        for v in ("v1", "v2", "v3"):
            (root / d / f"{v}.json").write_text(json.dumps(bad))
    for v in ("v1", "v2"):
        (root / "payment" / f"{v}.json").write_text(json.dumps(bad))
    with pytest.raises(DatasetSchemaError):
        load_api_schemas(root)


# ---------------------------------------------------------------------------
# Cross-artifact invariants
# ---------------------------------------------------------------------------


def test_singleton_same_instance_on_second_call() -> None:
    path = _DATA_ROOT / "drift_patterns" / "drifts.yaml"
    lib1 = load_drift_patterns(path)
    lib2 = load_drift_patterns(path)
    assert lib1 is lib2


def test_cross_artifact_drift_hints_reference_fields_or_codes() -> None:
    lib = load_drift_patterns(_DATA_ROOT / "drift_patterns" / "drifts.yaml")
    # Every pattern must have at least one hint that is a non-trivial token (>=2 chars).
    for p in lib.patterns.values():
        assert any(len(h) >= 2 for h in p.detection_hints), p.id
