"""Cross-file data-consistency invariants for DriftCall fixtures.

These eleven invariants guard the contracts between
``data/api_schemas/``, ``data/drift_patterns/drifts.yaml``,
``data/task_briefs/templates.yaml``, and ``data/task_briefs/i18n.yaml``.
A failure here means an authoring drift between files — never weaken the
test, fix the data.

References:
- ``docs/modules/datasets.md`` §3.3, §3.5 invariants #2/#3/#4
- ``docs/modules/drift_injector.md`` §4.4 (20-pattern catalogue)
- ``docs/modules/task_generator.md`` (templates contract)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from cells.step_03_fixtures import (
    _LANGUAGE_CODES,
    _PRIMARY_DOMAINS,
    DriftPattern,
    Template,
    _reset_caches,
    load_drift_patterns,
    load_i18n,
    load_templates,
)
from cells.step_06_drift_injector import _OPERATOR_DISPATCH

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data"
_TEMPLATES = _DATA / "task_briefs" / "templates.yaml"
_I18N = _DATA / "task_briefs" / "i18n.yaml"
_DRIFTS = _DATA / "drift_patterns" / "drifts.yaml"
_SCHEMAS = _DATA / "api_schemas"

# Auth drifts mutate transversal payment fields that are intentionally not
# present in primary-domain `drift_slot_tags` (datasets.md §3.5 invariant #4).
_TRANSVERSAL_AUTH_DRIFTS: frozenset[str] = frozenset(
    {"payment.auth_scope_upgrade", "payment.mfa_required"}
)

# The known set of mutation op keys recognised by the drift injector
# dispatcher. Kept in lock-step with cells/step_06_drift_injector.py.
_KNOWN_OPS: frozenset[str] = frozenset(_OPERATOR_DISPATCH.keys())


@pytest.fixture(autouse=True)
def _isolate_loader_caches() -> None:
    _reset_caches()


def _flatten_op_targets(payload: Any) -> set[str]:
    """Collect string keys + scalar string values from a mutation op payload.

    A field is considered "targeted" by a drift mutation if its name appears
    either as a key (e.g. ``rename: {price: total_fare_inr}`` targets both
    ``price`` and ``total_fare_inr``) or as a scalar string value of the
    payload mapping (rename targets), or as an item of a payload list (e.g.
    ``remove: [currency]`` targets ``currency``; ``require_new_field:
    [passenger_count]`` targets ``passenger_count``).
    """

    targets: set[str] = set()
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(k, str):
                targets.add(k)
            if isinstance(v, str):
                targets.add(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        targets.add(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                targets.add(item)
    elif isinstance(payload, str):
        targets.add(payload)
    return targets


def _drift_mutation_targets(pattern: DriftPattern) -> set[str]:
    targets: set[str] = set()
    for op_payload in pattern.mutation.values():
        targets |= _flatten_op_targets(op_payload)
    return targets


def _templates_in_domain(domain: str) -> tuple[Template, ...]:
    lib = load_templates(_TEMPLATES)
    return tuple(t for t in lib.templates if t.domain == domain)


# ---------------------------------------------------------------------------
# 1. No self-referential drifts
# ---------------------------------------------------------------------------


def test_no_self_referential_drifts() -> None:
    """Every drift must move from an earlier schema version to a later one."""

    lib = load_drift_patterns(_DRIFTS)
    offenders = [
        p.id for p in lib.patterns.values() if p.from_version == p.to_version
    ]
    assert offenders == [], (
        f"self-referential drifts forbidden, but found: {offenders}. "
        "Every drift's to_version must advance past from_version."
    )


# ---------------------------------------------------------------------------
# 2. Every drift_slot_tag is targetable by ≥ 1 drift mutation
# ---------------------------------------------------------------------------


def test_every_drift_slot_tag_is_targetable() -> None:
    """Each entry in ``drift_slot_tags`` must appear as a key/value of some
    drift's mutation payload (across all known op types). Auth drifts are
    transversal and therefore exempt from per-domain targetability."""

    lib = load_drift_patterns(_DRIFTS)
    primary_drifts = [
        p for p in lib.patterns.values() if p.id not in _TRANSVERSAL_AUTH_DRIFTS
    ]

    targets_by_domain: dict[str, set[str]] = {}
    for p in primary_drifts:
        targets_by_domain.setdefault(p.domain, set()).update(
            _drift_mutation_targets(p)
        )

    template_lib = load_templates(_TEMPLATES)
    untargeted: list[tuple[str, str]] = []
    for tpl in template_lib.templates:
        domain_targets = targets_by_domain.get(tpl.domain, set())
        for tag in tpl.drift_slot_tags:
            if tag not in domain_targets:
                untargeted.append((tpl.template_id, tag))

    assert untargeted == [], (
        "every drift_slot_tag must be targetable by ≥ 1 drift mutation "
        "in the same domain (datasets.md §3.5 invariant #4); "
        f"untargeted (template_id, tag) pairs: {untargeted}"
    )


# ---------------------------------------------------------------------------
# 3. Drift schema files exist
# ---------------------------------------------------------------------------


def test_every_drift_schema_exists() -> None:
    """Every drift's ``from_version`` and ``to_version`` JSON schema files
    must exist on disk (not just be referenced)."""

    lib = load_drift_patterns(_DRIFTS)
    missing: list[str] = []
    for p in lib.patterns.values():
        for ver in (p.from_version, p.to_version):
            schema_path = _SCHEMAS / p.domain / f"{ver}.json"
            if not schema_path.is_file():
                missing.append(f"{p.id}: {schema_path.relative_to(_REPO_ROOT)}")
    assert missing == [], (
        f"drift patterns reference non-existent schema files: {missing}"
    )


# ---------------------------------------------------------------------------
# 4. Drift schema files validate against the JSON Schema 2020-12 meta-schema
# ---------------------------------------------------------------------------


def test_every_drift_schema_valid() -> None:
    """Every schema referenced by a drift parses and validates as a
    well-formed Draft 2020-12 JSON Schema."""

    import json

    lib = load_drift_patterns(_DRIFTS)
    bad: list[str] = []
    for p in lib.patterns.values():
        for ver in (p.from_version, p.to_version):
            schema_path = _SCHEMAS / p.domain / f"{ver}.json"
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                Draft202012Validator.check_schema(schema)
            except (OSError, ValueError, SchemaError) as exc:
                bad.append(f"{p.id}: {schema_path.name}: {exc}")
    assert bad == [], f"invalid JSON Schema 2020-12 files: {bad}"


# ---------------------------------------------------------------------------
# 5. Every template covers all 5 LanguageCode values
# ---------------------------------------------------------------------------


def test_every_template_has_all_5_languages() -> None:
    lib = load_templates(_TEMPLATES)
    missing: list[tuple[str, set[str]]] = []
    for tpl in lib.templates:
        absent = _LANGUAGE_CODES - tpl.language_variants.keys()
        empty = {
            lang
            for lang, variants in tpl.language_variants.items()
            if len(variants) == 0
        }
        bad = absent | empty
        if bad:
            missing.append((tpl.template_id, bad))
    assert missing == [], (
        f"templates missing or empty for languages: {missing}"
    )


# ---------------------------------------------------------------------------
# 6. ≥ 2 prose variants per (template, lang) for hi / ta / kn
# ---------------------------------------------------------------------------


def test_every_template_has_min_2_variants_per_indic_lang() -> None:
    """Each Indic language (hi, ta, kn) must have ≥ 2 distinct prose
    variants per template (hinglish/en already meet this floor and must
    keep it)."""

    lib = load_templates(_TEMPLATES)
    floor_langs = ("hi", "ta", "kn", "hinglish", "en")
    too_few: list[tuple[str, str, int]] = []
    for tpl in lib.templates:
        for lang in floor_langs:
            variants = tpl.language_variants.get(lang, ())
            if len(set(variants)) < 2:
                too_few.append((tpl.template_id, lang, len(variants)))
    assert too_few == [], (
        f"templates with < 2 distinct variants for required languages: {too_few}"
    )


# ---------------------------------------------------------------------------
# 7. i18n keys complete across every language
# ---------------------------------------------------------------------------


def test_i18n_keys_complete_across_langs() -> None:
    """Every key that appears in any language's i18n.yaml block must be
    present in all 5 languages — partial localization is a bug."""

    lib = load_i18n(_I18N)
    union: set[str] = set()
    for entries in lib.strings.values():
        union |= set(entries.keys())

    missing: list[tuple[str, list[str]]] = []
    for lang in sorted(_LANGUAGE_CODES):
        absent = sorted(union - set(lib.strings[lang].keys()))
        if absent:
            missing.append((lang, absent))
    assert missing == [], (
        f"i18n.yaml has language-specific gaps (must mirror across all 5 langs): {missing}"
    )


# ---------------------------------------------------------------------------
# 8. payment is a transversal-only domain — never a primary template domain
# ---------------------------------------------------------------------------


def test_payment_is_transversal_only() -> None:
    """No template may declare ``domain: payment``. Payment is a
    transversal vendor used as a 2nd-leg auth side-effect only."""

    lib = load_templates(_TEMPLATES)
    offenders = [t.template_id for t in lib.templates if t.domain == "payment"]
    assert offenders == [], (
        f"payment domain is transversal-only but found primary templates: {offenders}"
    )

    # Defensive: also confirm the loader's primary-domain frozenset
    # excludes payment.
    assert "payment" not in _PRIMARY_DOMAINS


# ---------------------------------------------------------------------------
# 9. All mutation op keys are in the dispatcher's known-ops set
# ---------------------------------------------------------------------------


def test_all_drift_op_types_known() -> None:
    """Every mutation op key authored in drifts.yaml must be in the
    drift-injector's dispatch table; an unknown op silently no-ops at
    runtime, which would mask reward drift."""

    lib = load_drift_patterns(_DRIFTS)
    unknown: list[tuple[str, str]] = []
    for p in lib.patterns.values():
        for op_key in p.mutation:
            if op_key not in _KNOWN_OPS:
                unknown.append((p.id, op_key))
    assert unknown == [], (
        f"unknown mutation op keys (would silently no-op at runtime): {unknown}. "
        f"Known ops: {sorted(_KNOWN_OPS)}"
    )


# ---------------------------------------------------------------------------
# 10. Template count floor (diversity guard against overfitting)
# ---------------------------------------------------------------------------


def test_template_count_floor() -> None:
    """At least 15 templates total (DESIGN.md §6 task diversity guarantee)."""

    lib = load_templates(_TEMPLATES)
    assert len(lib.templates) >= 15, (
        f"need ≥ 15 templates for rollout diversity; got {len(lib.templates)}"
    )


# ---------------------------------------------------------------------------
# 11. Stage coverage — at least one template per min_stage ∈ {1,2,3}
# ---------------------------------------------------------------------------


def test_stage_coverage() -> None:
    """Each curriculum stage must have ≥ 1 dedicated template."""

    lib = load_templates(_TEMPLATES)
    stages_present = {t.min_stage for t in lib.templates}
    missing = {1, 2, 3} - stages_present
    assert missing == set(), (
        f"missing dedicated templates for min_stage(s): {sorted(missing)}"
    )
