"""Cell 07 — Procedural task-brief generator.

Implements docs/modules/task_generator.md. Pure, seeded, deterministic
expansion of a YAML template library into concrete ``GoalSpec`` briefs
for ``DriftCallEnv.reset()`` (DESIGN.md §4.2, §8.3, §8.4).

Contract: identical ``(seed, stage, language_weights)`` triples always
produce byte-identical ``GoalSpec.seed_utterance`` after NFC
normalization. No global mutable state; no ``random.random()``; no
``time.time()``; no ``hash()``. All stochastic choices thread through
``random.Random(stable_sub_seed(seed, tag))`` where ``stable_sub_seed``
uses ``hashlib.blake2b(digest_size=8)``.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from cells.step_04_models import GoalSpec

# ---------------------------------------------------------------------------
# Public literal types
# ---------------------------------------------------------------------------

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]
Domain = Literal["airline", "cab", "restaurant", "hotel"]

_LANGUAGE_CODES: frozenset[str] = frozenset({"hi", "ta", "kn", "en", "hinglish"})
_DOMAINS: frozenset[str] = frozenset({"airline", "cab", "restaurant", "hotel"})
_VALID_STAGES: frozenset[int] = frozenset({1, 2, 3})

# Fixed reference date for deterministic date sampling (task_generator.md §3.3).
_REFERENCE_DATE: date = date(2026, 4, 25)
_DATE_WINDOW_DAYS: int = 60

# SMS-length bound for ASR input (§3.6 invariant 7).
_MAX_UTTERANCE_LEN: int = 280

# Built-in slot conventions — §3.3 of task_generator.md. Templates may
# override by declaring slot_distributions explicitly; otherwise these
# name-based defaults apply.
_DATE_SLOT_NAMES: frozenset[str] = frozenset(
    {
        "when",
        "checkin",
        "checkout",
        "date",
        "departure",
        "arrival",
        "return_when",
        "new_when",
    }
)
_INTER_CITY_SLOT_NAMES: frozenset[str] = frozenset(
    {"from", "to", "city", "origin", "destination"}
)
_INTRA_CITY_SLOT_NAMES: frozenset[str] = frozenset({"pickup", "drop"})

# Default domain → city-code tuples (IATA-style). Authored here so the
# generator is self-contained without requiring the YAML library to
# declare a cities_by_domain block.
_DEFAULT_INTER_CITIES: tuple[str, ...] = (
    "HYD",
    "BLR",
    "DEL",
    "BOM",
    "MAA",
    "CCU",
    "PNQ",
    "AMD",
    "JAI",
    "GOI",
)
_DEFAULT_INTRA_CITIES: tuple[str, ...] = (
    "Koramangala",
    "Indiranagar",
    "Whitefield",
    "Andheri",
    "Bandra",
    "Powai",
    "Gurgaon",
    "Saket",
    "Banjara Hills",
    "Salt Lake",
)
_DEFAULT_CITIES_BY_DOMAIN: Mapping[Domain, tuple[str, ...]] = {
    "airline": _DEFAULT_INTER_CITIES,
    "hotel": _DEFAULT_INTER_CITIES,
    "restaurant": _DEFAULT_INTER_CITIES,
    "cab": _DEFAULT_INTRA_CITIES,
}


# ---------------------------------------------------------------------------
# Exception hierarchy (task_generator.md §5)
# ---------------------------------------------------------------------------


class TaskGeneratorError(Exception):
    """Base class for every failure raised by :mod:`step_07_task_generator`."""


class MissingSlotError(TaskGeneratorError):
    """Template variant references a ``{slot}`` placeholder not present in the filled SlotGrid."""


class InvalidLanguageError(TaskGeneratorError):
    """``language_weights`` contains a key outside :data:`LanguageCode`."""


class InvalidLanguageWeightError(TaskGeneratorError):
    """``language_weights`` is empty, has a negative value, sums off 1.0, or is all zero."""


class InvalidStageError(TaskGeneratorError):
    """``stage`` is not one of ``{1, 2, 3}``."""


class InvalidBudgetError(TaskGeneratorError):
    """Sampled numeric constraint falls outside the template's declared ``[low, high]`` range."""


class TemplateFileMissingError(TaskGeneratorError):
    """Template YAML file not found or unreadable."""


class TemplateSchemaError(TaskGeneratorError):
    """Template YAML present but fails schema validation."""


class UnicodeNormalizationError(TaskGeneratorError):
    """Rendered utterance fails NFC round-trip check (defensive)."""


class NoVariantForLanguageError(TaskGeneratorError):
    """Chosen template has no ``language_variants`` entry for the chosen language."""


# ---------------------------------------------------------------------------
# In-memory types (task_generator.md §4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotDistribution:
    """Either an enum (``choices``) or a uniform numeric grid (``low``, ``high``, ``step``)."""

    kind: Literal["choices", "uniform", "date", "bool"]
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
    slot_distributions: Mapping[str, SlotDistribution]
    constraints_template: Mapping[str, SlotDistribution]
    drift_slot_tags: tuple[str, ...]
    language_variants: Mapping[LanguageCode, tuple[str, ...]]


@dataclass(frozen=True)
class TemplateLibrary:
    templates: tuple[Template, ...]
    cities_by_domain: Mapping[Domain, tuple[str, ...]]
    i18n: Mapping[LanguageCode, Mapping[str, str]]


@dataclass(frozen=True)
class SlotGrid:
    """Concrete slot values after expansion."""

    values: Mapping[str, object]


@dataclass(frozen=True)
class RawBrief:
    template_id: str
    domain: Domain
    intent: str
    slots: SlotGrid
    constraints: Mapping[str, object]
    language: LanguageCode


# ---------------------------------------------------------------------------
# Sub-seed helper (task_generator.md §3.1)
# ---------------------------------------------------------------------------


def stable_sub_seed(seed: int, tag: str) -> int:
    """Return a stable 64-bit integer derived from ``(seed, tag)``.

    Uses blake2b with ``digest_size=8`` so the formula is pinned and
    domain-separated across decision tags.
    """
    digest = hashlib.blake2b(f"{seed}:{tag}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# NFC helpers
# ---------------------------------------------------------------------------


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _assert_nfc(text: str, *, where: str) -> None:
    if not unicodedata.is_normalized("NFC", text):
        raise UnicodeNormalizationError(
            f"string at {where} failed NFC round-trip: {text!r}"
        )


# ---------------------------------------------------------------------------
# Template loader (task_generator.md §2.2, §3.4, §7 edge cases 1 & 8)
# ---------------------------------------------------------------------------


def _parse_distribution(raw: Mapping[str, Any], *, where: str) -> SlotDistribution:
    """Parse a single slot/constraint distribution block."""
    if "choices" in raw:
        choices = raw["choices"]
        if not isinstance(choices, list) or not choices:
            raise TemplateSchemaError(f"{where}: 'choices' must be non-empty list")
        norm_choices = tuple(_nfc(str(c)) for c in choices)
        return SlotDistribution(kind="choices", choices=norm_choices)
    if raw.get("distribution") == "uniform":
        for key in ("low", "high", "step"):
            if key not in raw:
                raise TemplateSchemaError(f"{where}: uniform missing '{key}'")
        low = float(raw["low"])
        high = float(raw["high"])
        step = float(raw["step"])
        if step <= 0:
            raise TemplateSchemaError(f"{where}: step must be > 0 (got {step})")
        if low > high:
            raise TemplateSchemaError(f"{where}: low > high ({low} > {high})")
        span = high - low
        # Grid must terminate cleanly at ``high`` (§7 edge case 8).
        # Use integer step check avoiding floating-point drift.
        ratio = span / step
        if abs(ratio - round(ratio)) > 1e-9:
            raise TemplateSchemaError(
                f"{where}: step grid misaligned "
                f"(low={low}, high={high}, step={step}) — (high-low) not divisible by step"
            )
        return SlotDistribution(kind="uniform", low=low, high=high, step=step)
    if raw.get("distribution") == "date":
        return SlotDistribution(kind="date")
    if raw.get("distribution") == "bool":
        return SlotDistribution(kind="bool")
    raise TemplateSchemaError(
        f"{where}: unrecognized distribution descriptor {dict(raw)!r}"
    )


def _parse_template(raw: Mapping[str, Any], *, where: str) -> Template:
    required_keys = (
        "template_id",
        "domain",
        "intent",
        "min_stage",
        "required_slots",
        "optional_slots",
        "constraints_template",
        "drift_slot_tags",
        "language_variants",
    )
    for key in required_keys:
        if key not in raw:
            raise TemplateSchemaError(f"{where}: missing required key {key!r}")

    template_id = _nfc(str(raw["template_id"]))
    domain_raw = str(raw["domain"])
    if domain_raw not in _DOMAINS:
        raise TemplateSchemaError(
            f"{where}: domain {domain_raw!r} not in {sorted(_DOMAINS)}"
        )
    min_stage = int(raw["min_stage"])
    if min_stage not in _VALID_STAGES:
        raise TemplateSchemaError(
            f"{where}: min_stage {min_stage} not in {sorted(_VALID_STAGES)}"
        )

    required_slots = tuple(_nfc(str(s)) for s in raw["required_slots"])
    optional_slots = tuple(_nfc(str(s)) for s in raw["optional_slots"])
    drift_slot_tags = tuple(_nfc(str(s)) for s in raw["drift_slot_tags"])

    slot_distributions_raw = raw.get("slot_distributions", {}) or {}
    slot_distributions: dict[str, SlotDistribution] = {}
    for name, block in slot_distributions_raw.items():
        slot_distributions[_nfc(str(name))] = _parse_distribution(
            block, where=f"{where}.slot_distributions.{name}"
        )

    constraints_template: dict[str, SlotDistribution] = {}
    for name, block in raw["constraints_template"].items():
        constraints_template[_nfc(str(name))] = _parse_distribution(
            block, where=f"{where}.constraints_template.{name}"
        )

    language_variants_raw = raw["language_variants"]
    if not isinstance(language_variants_raw, dict):
        raise TemplateSchemaError(f"{where}: language_variants must be a mapping")
    language_variants: dict[LanguageCode, tuple[str, ...]] = {}
    for lang, variants in language_variants_raw.items():
        if lang not in _LANGUAGE_CODES:
            raise TemplateSchemaError(
                f"{where}: language key {lang!r} not in {sorted(_LANGUAGE_CODES)}"
            )
        if not isinstance(variants, list) or not variants:
            raise TemplateSchemaError(
                f"{where}.language_variants.{lang}: must be non-empty list"
            )
        language_variants[cast("LanguageCode", lang)] = tuple(
            _nfc(str(v)) for v in variants
        )

    # Every template must have ≥ 1 variant per LanguageCode (§7 edge case 7).
    for code in _LANGUAGE_CODES:
        if code not in language_variants:
            raise TemplateSchemaError(
                f"{where}: language_variants missing required code {code!r}"
            )

    # Static placeholder scan (§7 edge case 1).
    declared_placeholders = (
        set(required_slots)
        | set(optional_slots)
        | set(constraints_template.keys())
    )
    for lang, variants in language_variants.items():
        for variant in variants:
            for placeholder in _iter_placeholders(variant):
                if placeholder not in declared_placeholders:
                    raise TemplateSchemaError(
                        f"{where}.language_variants.{lang}: variant references "
                        f"undeclared placeholder {placeholder!r} in {variant!r}"
                    )

    return Template(
        template_id=template_id,
        domain=cast("Domain", domain_raw),
        intent=_nfc(str(raw["intent"])),
        min_stage=cast("Literal[1, 2, 3]", min_stage),
        required_slots=required_slots,
        optional_slots=optional_slots,
        slot_distributions=slot_distributions,
        constraints_template=constraints_template,
        drift_slot_tags=drift_slot_tags,
        language_variants=language_variants,
    )


def _iter_placeholders(fmt: str) -> Iterator[str]:
    """Yield placeholder names in a format string (ignores literals)."""
    for _literal, field_name, _spec, _conv in string.Formatter().parse(fmt):
        if field_name is not None and field_name != "":
            yield field_name


def load_templates(
    path: str | Path = "data/task_briefs/templates.yaml",
    i18n_path: str | Path | None = None,
) -> TemplateLibrary:
    """Parse the template YAML file and return an in-memory :class:`TemplateLibrary`.

    ``i18n_path`` defaults to ``data/task_briefs/i18n.yaml`` alongside
    ``path``. All strings are NFC-normalized on read (§3.4).
    """
    templates_path = Path(path)
    if not templates_path.exists():
        raise TemplateFileMissingError(f"templates YAML not found: {templates_path}")

    if i18n_path is None:
        i18n_path = templates_path.parent / "i18n.yaml"
    i18n_path = Path(i18n_path)

    try:
        with templates_path.open("r", encoding="utf-8") as fh:
            raw_templates = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise TemplateSchemaError(f"templates YAML malformed: {exc}") from exc

    if raw_templates is None:
        raise TemplateSchemaError("templates YAML is empty")

    parsed_templates: list[Template] = []
    cities_by_domain: dict[Domain, tuple[str, ...]] = {}

    if isinstance(raw_templates, dict):
        tmpl_list = raw_templates.get("templates", [])
        raw_cities = raw_templates.get("cities_by_domain", {}) or {}
        for dom, lst in raw_cities.items():
            if dom not in _DOMAINS:
                raise TemplateSchemaError(f"cities_by_domain: bad domain {dom!r}")
            cities_by_domain[cast("Domain", dom)] = tuple(_nfc(str(c)) for c in lst)
    elif isinstance(raw_templates, list):
        tmpl_list = raw_templates
    else:
        raise TemplateSchemaError(
            f"templates YAML root must be list or mapping, got {type(raw_templates).__name__}"
        )

    if not isinstance(tmpl_list, list) or not tmpl_list:
        raise TemplateSchemaError("templates YAML must contain a non-empty list")

    for idx, raw in enumerate(tmpl_list):
        if not isinstance(raw, dict):
            raise TemplateSchemaError(
                f"templates[{idx}]: entry must be a mapping, got {type(raw).__name__}"
            )
        parsed_templates.append(_parse_template(raw, where=f"templates[{idx}]"))

    # i18n file is optional; if absent we use an empty mapping.
    _LANG_CODES: tuple[LanguageCode, ...] = ("hi", "ta", "kn", "en", "hinglish")
    i18n_data: dict[LanguageCode, dict[str, str]] = {code: {} for code in _LANG_CODES}
    if i18n_path.exists():
        try:
            with i18n_path.open("r", encoding="utf-8") as fh:
                raw_i18n = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise TemplateSchemaError(f"i18n YAML malformed: {exc}") from exc
        if not isinstance(raw_i18n, dict):
            raise TemplateSchemaError("i18n YAML root must be a mapping")
        for lang, block in raw_i18n.items():
            if lang not in _LANGUAGE_CODES:
                raise TemplateSchemaError(
                    f"i18n: language key {lang!r} not in {sorted(_LANGUAGE_CODES)}"
                )
            if not isinstance(block, dict):
                raise TemplateSchemaError(f"i18n.{lang}: must be a mapping")
            flat: dict[str, str] = {}
            _flatten_i18n(block, prefix="", out=flat)
            i18n_data[cast("LanguageCode", lang)] = {
                _nfc(str(k)): _nfc(str(v)) for k, v in flat.items()
            }

    return TemplateLibrary(
        templates=tuple(parsed_templates),
        cities_by_domain=cities_by_domain,
        i18n=i18n_data,
    )


def _flatten_i18n(block: Mapping[str, Any], *, prefix: str, out: dict[str, str]) -> None:
    """Flatten nested i18n dicts into dotted keys, NFC everything."""
    for k, v in block.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten_i18n(v, prefix=key, out=out)
        else:
            out[key] = str(v)


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_library_cache: TemplateLibrary | None = None
_library_override: TemplateLibrary | None = None


def _get_library() -> TemplateLibrary:
    """Return the process-wide TemplateLibrary, loading lazily."""
    if _library_override is not None:
        return _library_override
    global _library_cache
    if _library_cache is None:
        _library_cache = _load_default_library()
    return _library_cache


def _load_default_library() -> TemplateLibrary:
    """Try the production path, then fall back to the packaged inline library."""
    default_path = Path("data/task_briefs/templates.yaml")
    if default_path.exists():
        return load_templates(default_path)
    return _builtin_library()


def set_library_override(library: TemplateLibrary | None) -> None:
    """Test hook: pin :func:`_get_library` to a specific library (or clear)."""
    global _library_override
    _library_override = library


def reset_library_cache() -> None:
    """Test hook: clear the lazy cache so the next call reloads."""
    global _library_cache
    _library_cache = None


# ---------------------------------------------------------------------------
# Built-in library (fallback when data/ isn't authored yet)
# ---------------------------------------------------------------------------


def _builtin_library() -> TemplateLibrary:
    """Minimal 5-template library so the generator is self-contained during dev."""
    # Shared numeric grids.
    budget_flight = SlotDistribution(kind="uniform", low=3000.0, high=15000.0, step=500.0)
    budget_hotel = SlotDistribution(kind="uniform", low=2000.0, high=10000.0, step=500.0)
    budget_cab = SlotDistribution(kind="uniform", low=200.0, high=2000.0, step=50.0)
    budget_food = SlotDistribution(kind="uniform", low=200.0, high=1000.0, step=50.0)
    time_window = SlotDistribution(
        kind="choices", choices=("morning", "afternoon", "evening", "late_night")
    )
    date_dist = SlotDistribution(kind="date")
    veg_only = SlotDistribution(kind="bool")
    pax = SlotDistribution(kind="uniform", low=1.0, high=4.0, step=1.0)

    cities_inter = (
        "HYD",
        "BLR",
        "DEL",
        "BOM",
        "MAA",
        "CCU",
        "PNQ",
        "AMD",
        "JAI",
        "GOI",
    )
    cities_intra = (
        "Koramangala",
        "Indiranagar",
        "Whitefield",
        "Andheri",
        "Bandra",
        "Powai",
        "Gurgaon",
        "Saket",
        "Banjara Hills",
        "Salt Lake",
    )

    airline = Template(
        template_id="airline.book.fixture_v1",
        domain="airline",
        intent="book_flight",
        min_stage=1,
        required_slots=("from", "to", "when"),
        optional_slots=(),
        slot_distributions={
            "from": SlotDistribution(kind="choices", choices=cities_inter),
            "to": SlotDistribution(kind="choices", choices=cities_inter),
            "when": date_dist,
        },
        constraints_template={
            "budget_inr": budget_flight,
            "time_window": time_window,
        },
        drift_slot_tags=("price", "total_fare_inr"),
        language_variants={
            "hinglish": (
                "Bhai {when} ko {from} se {to} jaana hai, {budget_inr} rupees max, {time_window}",
            ),
            "hi": (
                "{when} को {from} से {to} जाना है, {budget_inr} रुपये से कम, {time_window}",
            ),
            "ta": (
                "{when} அன்று {from} லிருந்து {to} டிக்கெட் வேண்டும், {budget_inr} ரூபாய் கீழ், {time_window}",
            ),
            "kn": (
                "{when} ರಂದು {from} ಇಂದ {to} ಗೆ ಟಿಕೆಟ್ ಬೇಕು, {budget_inr} ರೂಪಾಯಿ ಒಳಗೆ, {time_window}",
            ),
            "en": (
                "Flight from {from} to {to} on {when}, under ₹{budget_inr}, {time_window}",
            ),
        },
    )

    cab = Template(
        template_id="cab.book.fixture_v1",
        domain="cab",
        intent="book_cab",
        min_stage=1,
        required_slots=("pickup", "drop", "when"),
        optional_slots=(),
        slot_distributions={
            "pickup": SlotDistribution(kind="choices", choices=cities_intra),
            "drop": SlotDistribution(kind="choices", choices=cities_intra),
            "when": date_dist,
        },
        constraints_template={
            "budget_inr": budget_cab,
            "vehicle_class": SlotDistribution(
                kind="choices", choices=("mini", "sedan", "suv")
            ),
        },
        drift_slot_tags=("fare_inr", "fare_breakdown"),
        language_variants={
            "hinglish": (
                "{when} ko {pickup} se {drop} cab chahiye, {budget_inr} ke andar, {vehicle_class}",
            ),
            "hi": (
                "{when} को {pickup} से {drop} कैब चाहिए, {budget_inr} के अंदर, {vehicle_class}",
            ),
            "ta": (
                "{when} அன்று {pickup} லிருந்து {drop} கேப், {budget_inr} கீழ், {vehicle_class}",
            ),
            "kn": (
                "{when} ರಂದು {pickup} ಇಂದ {drop} ಟ್ಯಾಕ್ಸಿ, {budget_inr} ಒಳಗೆ, {vehicle_class}",
            ),
            "en": (
                "Cab from {pickup} to {drop} on {when}, under ₹{budget_inr}, {vehicle_class}",
            ),
        },
    )

    restaurant = Template(
        template_id="restaurant.order.fixture_v1",
        domain="restaurant",
        intent="order_food",
        min_stage=2,
        required_slots=("city", "cuisine", "when"),
        optional_slots=(),
        slot_distributions={
            "city": SlotDistribution(kind="choices", choices=cities_inter),
            "cuisine": SlotDistribution(
                kind="choices", choices=("Biryani", "Dosa", "Pizza", "Thali", "Noodles")
            ),
            "when": date_dist,
        },
        constraints_template={
            "budget_inr": budget_food,
            "veg_only": veg_only,
        },
        drift_slot_tags=("min_order", "veg_filter"),
        language_variants={
            "hinglish": (
                "Bhai {when} ko {city} mein {cuisine} order karna hai, {budget_inr} ke andar, veg_only={veg_only}",
            ),
            "hi": (
                "{when} को {city} में {cuisine} ऑर्डर करना है, {budget_inr} के अंदर, veg_only={veg_only}",
            ),
            "ta": (
                "{when} அன்று {city} இல் {cuisine} ஆர்டர், {budget_inr} கீழ், veg_only={veg_only}",
            ),
            "kn": (
                "{when} ರಂದು {city} ನಲ್ಲಿ {cuisine} ಆರ್ಡರ್, {budget_inr} ಒಳಗೆ, veg_only={veg_only}",
            ),
            "en": (
                "Order {cuisine} in {city} on {when}, under ₹{budget_inr}, veg_only={veg_only}",
            ),
        },
    )

    hotel = Template(
        template_id="hotel.book.fixture_v1",
        domain="hotel",
        intent="book_hotel",
        min_stage=2,
        required_slots=("city", "checkin", "checkout"),
        optional_slots=(),
        slot_distributions={
            "city": SlotDistribution(kind="choices", choices=cities_inter),
            "checkin": date_dist,
            "checkout": date_dist,
        },
        constraints_template={
            "budget_inr": budget_hotel,
            "room_type": SlotDistribution(
                kind="choices", choices=("single", "double", "suite")
            ),
        },
        drift_slot_tags=("cancel_window", "gst_number"),
        language_variants={
            "hinglish": (
                "{city} mein {checkin} se {checkout} tak hotel chahiye, {budget_inr} per night, {room_type}",
            ),
            "hi": (
                "{city} में {checkin} से {checkout} तक होटल चाहिए, {budget_inr} प्रति रात, {room_type}",
            ),
            "ta": (
                "{city} இல் {checkin} முதல் {checkout} வரை ஹோட்டல், {budget_inr} ஒரு இரவு, {room_type}",
            ),
            "kn": (
                "{city} ನಲ್ಲಿ {checkin} ಇಂದ {checkout} ವರೆಗೆ ಹೋಟೆಲ್, {budget_inr} ಒಂದು ರಾತ್ರಿ, {room_type}",
            ),
            "en": (
                "Hotel in {city} from {checkin} to {checkout}, ₹{budget_inr} per night, {room_type}",
            ),
        },
    )

    # Stage-3 compound-constraint airline template — adds a third constraint.
    airline_compound = Template(
        template_id="airline.book.compound_v1",
        domain="airline",
        intent="book_flight",
        min_stage=3,
        required_slots=("from", "to", "when"),
        optional_slots=(),
        slot_distributions={
            "from": SlotDistribution(kind="choices", choices=cities_inter),
            "to": SlotDistribution(kind="choices", choices=cities_inter),
            "when": date_dist,
        },
        constraints_template={
            "budget_inr": budget_flight,
            "time_window": time_window,
            "passenger_count": pax,
        },
        drift_slot_tags=("price", "total_fare_inr", "passenger_count"),
        language_variants={
            "hinglish": (
                "{when} ko {from} se {to}, {passenger_count} log, {budget_inr} max, {time_window}",
            ),
            "hi": (
                "{when} को {from} से {to}, {passenger_count} लोग, {budget_inr} रुपये, {time_window}",
            ),
            "ta": (
                "{when} அன்று {from} லிருந்து {to}, {passenger_count} பேர், {budget_inr} ரூபாய், {time_window}",
            ),
            "kn": (
                "{when} ರಂದು {from} ಇಂದ {to}, {passenger_count} ಜನ, {budget_inr} ರೂಪಾಯಿ, {time_window}",
            ),
            "en": (
                "Flight {from} to {to} on {when} for {passenger_count} pax, ₹{budget_inr}, {time_window}",
            ),
        },
    )

    return TemplateLibrary(
        templates=(airline, cab, restaurant, hotel, airline_compound),
        cities_by_domain={
            "airline": cities_inter,
            "hotel": cities_inter,
            "cab": cities_intra,
            "restaurant": cities_inter,
        },
        i18n={
            "hi": {"cities.BLR": "बेंगलुरु", "cities.MAA": "चेन्नई"},
            "ta": {"cities.BLR": "பெங்களூரு", "cities.MAA": "சென்னை"},
            "kn": {"cities.BLR": "ಬೆಂಗಳೂರು", "cities.MAA": "ಚೆನ್ನೈ"},
            "en": {"cities.BLR": "Bengaluru"},
            "hinglish": {"cities.BLR": "Bengaluru"},
        },
    )


# ---------------------------------------------------------------------------
# Picker + expander (task_generator.md §2.2, §3.2, §3.3)
# ---------------------------------------------------------------------------


def _pick_domain(seed: int, library: TemplateLibrary, stage: int) -> Domain:
    """Pick uniformly from domains that have ≥ 1 eligible template at ``stage``."""
    available = sorted({t.domain for t in library.templates if t.min_stage <= stage})
    if not available:
        raise TemplateSchemaError(
            f"library has no templates eligible at stage={stage}"
        )
    rng = random.Random(stable_sub_seed(seed, "domain"))
    return rng.choice(available)


def _eligible_templates(
    library: TemplateLibrary,
    stage: int,
    domain: Domain,
) -> tuple[Template, ...]:
    return tuple(
        t for t in library.templates if t.domain == domain and t.min_stage <= stage
    )


def _pick_template(
    seed: int,
    stage: int,
    domain: Domain,
    library: TemplateLibrary,
) -> Template:
    eligible = _eligible_templates(library, stage, domain)
    if not eligible:
        raise TemplateSchemaError(
            f"no eligible templates for domain={domain!r} stage={stage}"
        )
    rng = random.Random(stable_sub_seed(seed, "template"))
    # Use sorted template_ids for deterministic ordering.
    ordered = tuple(sorted(eligible, key=lambda t: t.template_id))
    return rng.choice(ordered)


def _sample_slot_value(
    rng: random.Random,
    name: str,
    dist: SlotDistribution,
    *,
    template_id: str,
) -> object:
    if dist.kind == "choices":
        if not dist.choices:
            raise TemplateSchemaError(
                f"{template_id}.{name}: empty choices list"
            )
        return rng.choice(dist.choices)
    if dist.kind == "uniform":
        assert dist.low is not None and dist.high is not None and dist.step is not None
        steps = int(round((dist.high - dist.low) / dist.step))
        pick = rng.randint(0, steps)
        value = dist.low + pick * dist.step
        # Integer-ify when step + bounds are integral.
        if float(int(dist.step)) == dist.step and float(int(dist.low)) == dist.low:
            value = int(round(value))
        # Post-check (§7 edge case 3).
        lo = int(dist.low) if isinstance(value, int) else dist.low
        hi = int(dist.high) if isinstance(value, int) else dist.high
        if not (lo <= value <= hi):
            raise InvalidBudgetError(
                f"{template_id}.{name}: sampled {value} outside [{dist.low}, {dist.high}]"
            )
        return value
    if dist.kind == "date":
        offset = rng.randint(0, _DATE_WINDOW_DAYS - 1)
        return (_REFERENCE_DATE + timedelta(days=offset)).isoformat()
    if dist.kind == "bool":
        return bool(rng.getrandbits(1))
    raise TemplateSchemaError(
        f"{template_id}.{name}: unknown distribution kind {dist.kind!r}"
    )


def _resolve_slot_distribution(
    template: Template,
    name: str,
    library: TemplateLibrary,
) -> SlotDistribution | None:
    """Resolve a slot's distribution, preferring explicit declaration then conventions."""
    explicit = template.slot_distributions.get(name)
    if explicit is not None:
        return explicit
    # Constraints block can also declare slot distributions that double as fills.
    constraint = template.constraints_template.get(name)
    if constraint is not None:
        return constraint
    # Conventional fills by slot name.
    if name in _DATE_SLOT_NAMES:
        return SlotDistribution(kind="date")
    if name in _INTER_CITY_SLOT_NAMES:
        pool = library.cities_by_domain.get(template.domain) or _DEFAULT_CITIES_BY_DOMAIN.get(
            template.domain, _DEFAULT_INTER_CITIES
        )
        return SlotDistribution(kind="choices", choices=pool)
    if name in _INTRA_CITY_SLOT_NAMES:
        pool = library.cities_by_domain.get(template.domain) or _DEFAULT_INTRA_CITIES
        return SlotDistribution(kind="choices", choices=pool)
    return None


def _expand_slots(
    seed: int,
    template: Template,
    *,
    stage: int,
    library: TemplateLibrary,
) -> tuple[SlotGrid, dict[str, object]]:
    """Sample one concrete value per required slot; stage-aware constraint pick.

    Returns ``(SlotGrid, constraints_dict)``.
    """
    values: dict[str, object] = {}

    # Required slots — always sampled.
    for name in template.required_slots:
        dist = _resolve_slot_distribution(template, name, library)
        if dist is None:
            raise TemplateSchemaError(
                f"{template.template_id}: required slot {name!r} has no distribution "
                f"(declare in slot_distributions or use a conventional name)"
            )
        rng = random.Random(stable_sub_seed(seed, f"slot:{name}"))
        values[name] = _sample_slot_value(rng, name, dist, template_id=template.template_id)

    # Optional slots — included with probability 0.5 (seeded). Silently
    # skipped if no distribution resolves (template declares the slot as
    # available but does not wire a fill source).
    for name in template.optional_slots:
        dist = _resolve_slot_distribution(template, name, library)
        if dist is None:
            continue
        rng = random.Random(stable_sub_seed(seed, f"opt:{name}"))
        if rng.random() < 0.5:
            sub_rng = random.Random(stable_sub_seed(seed, f"slot:{name}"))
            values[name] = _sample_slot_value(
                sub_rng, name, dist, template_id=template.template_id
            )

    # Constraints — stage-aware sub-selection (§3.5).
    max_constraints = {1: 2, 2: 3, 3: 4}[stage]
    constraint_names = list(template.constraints_template.keys())
    # Stage 1: keep only the first max_constraints deterministically.
    # Stage 2/3: include all declared constraints up to max.
    kept = constraint_names[:max_constraints]
    constraints: dict[str, object] = {}
    for name in kept:
        dist = template.constraints_template[name]
        rng = random.Random(stable_sub_seed(seed, f"constraint:{name}"))
        value = _sample_slot_value(
            rng, name, dist, template_id=template.template_id
        )
        constraints[name] = value
        # Also mirror into slots so variant-format can reference {budget_inr}.
        values[name] = value

    # NFC-normalize any string leaves.
    for k, v in list(values.items()):
        if isinstance(v, str):
            values[k] = _nfc(v)
    for k, v in list(constraints.items()):
        if isinstance(v, str):
            constraints[k] = _nfc(v)

    return SlotGrid(values=values), constraints


# ---------------------------------------------------------------------------
# Language picker
# ---------------------------------------------------------------------------


def _validate_language_weights(language_weights: Mapping[str, float]) -> None:
    """Raise on any malformed input per §3.2."""
    if not isinstance(language_weights, Mapping) or len(language_weights) == 0:
        raise InvalidLanguageWeightError("language_weights is empty")

    bad_keys = [k for k in language_weights if k not in _LANGUAGE_CODES]
    if bad_keys:
        raise InvalidLanguageError(
            f"unsupported language key(s): {bad_keys} "
            f"(allowed: {sorted(_LANGUAGE_CODES)})"
        )

    for k, v in language_weights.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise InvalidLanguageWeightError(
                f"language_weights[{k!r}] must be numeric, got {type(v).__name__}"
            )
        if v < 0:
            raise InvalidLanguageWeightError(
                f"language_weights[{k!r}]={v} is negative"
            )

    total = sum(float(v) for v in language_weights.values())
    if abs(total - 1.0) > 1e-6:
        raise InvalidLanguageWeightError(
            f"language_weights sum {total!r} outside [1-1e-6, 1+1e-6]"
        )

    # Defensive all-zero check (§3.2 last bullet).
    if all(float(v) == 0.0 for v in language_weights.values()):
        raise InvalidLanguageWeightError(
            "language_weights are all zero (would have no population to sample)"
        )


def _pick_language(
    seed: int,
    language_weights: Mapping[LanguageCode, float],
) -> LanguageCode:
    rng = random.Random(stable_sub_seed(seed, "language"))
    # Deterministic ordering of keys for reproducibility across dict insertion orders.
    codes = sorted(language_weights.keys())
    weights = [float(language_weights[c]) for c in codes]
    chosen = rng.choices(codes, weights=weights, k=1)[0]
    return chosen


# ---------------------------------------------------------------------------
# Utterance formatter
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _format_utterance(
    seed: int,
    template: Template,
    slots: SlotGrid,
    language: LanguageCode,
) -> str:
    variants = template.language_variants.get(language)
    if not variants:
        raise NoVariantForLanguageError(
            f"template {template.template_id!r} has no variants for language {language!r}"
        )
    rng = random.Random(stable_sub_seed(seed, "variant"))
    chosen = rng.choice(tuple(variants))

    # Render by placeholder-by-placeholder substitution so a missing slot
    # raises MissingSlotError with the exact field name rather than whatever
    # ``str.format`` would surface.
    def _repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in slots.values:
            raise MissingSlotError(
                f"template {template.template_id!r} variant references {{{name}}} "
                f"but slot is unbound (slots={sorted(slots.values)})"
            )
        value = slots.values[name]
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            # Trim trailing zeros for cleanness, but keep determinism.
            if value.is_integer():
                return str(int(value))
            return str(value)
        return str(value)

    rendered = _PLACEHOLDER_RE.sub(_repl, chosen)
    normalized = _nfc(rendered)
    _assert_nfc(normalized, where=f"utterance({template.template_id}, {language})")
    return normalized


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def generate(
    seed: int,
    stage: Literal[1, 2, 3],
    language_weights: Mapping[LanguageCode, float],
) -> GoalSpec:
    """Produce one :class:`GoalSpec` for episode ``seed`` at curriculum ``stage``.

    Determinism: identical ``(seed, stage, language_weights)`` ⇒ identical
    ``GoalSpec`` after NFC normalization of ``seed_utterance``.
    """
    # Stage validation (cheapest first).
    if stage not in _VALID_STAGES:
        raise InvalidStageError(
            f"stage must be in {sorted(_VALID_STAGES)}, got {stage!r}"
        )

    _validate_language_weights(cast("Mapping[str, float]", language_weights))

    library = _get_library()

    domain = _pick_domain(seed, library, int(stage))
    template = _pick_template(seed, int(stage), domain, library)
    slot_grid, constraints = _expand_slots(
        seed, template, stage=int(stage), library=library
    )
    language = _pick_language(seed, language_weights)
    utterance = _format_utterance(seed, template, slot_grid, language)

    if len(utterance) > _MAX_UTTERANCE_LEN:
        # Truncate is incorrect (breaks determinism/meaning). Raise so the
        # template author shortens the variant.
        raise TemplateSchemaError(
            f"rendered utterance exceeds {_MAX_UTTERANCE_LEN} chars "
            f"({len(utterance)}): {utterance!r}"
        )

    # Slot dict exposed on GoalSpec should exclude constraint-named entries —
    # those live in ``constraints``. ``required_slots`` + included optionals only.
    slot_keys = set(template.required_slots) | set(template.optional_slots)
    slots_out = {k: v for k, v in slot_grid.values.items() if k in slot_keys}

    return GoalSpec(
        domain=template.domain,
        intent=template.intent,
        slots=slots_out,
        constraints=constraints,
        language=language,
        seed_utterance=utterance,
    )


# ---------------------------------------------------------------------------
# Variant enumerator (task_generator.md §2.2)
# ---------------------------------------------------------------------------


def enumerate_variants(
    limit: int | None = None,
    stage: int = 3,
    language_weights: Mapping[LanguageCode, float] | None = None,
) -> Iterator[GoalSpec]:
    """Deterministic walk over the procedural grid."""
    if stage not in _VALID_STAGES:
        raise InvalidStageError(f"stage must be in {sorted(_VALID_STAGES)}, got {stage!r}")
    if language_weights is None:
        language_weights = {
            "en": 0.2,
            "hi": 0.2,
            "ta": 0.2,
            "kn": 0.2,
            "hinglish": 0.2,
        }
    count = 0
    seed = 0
    while limit is None or count < limit:
        yield generate(seed, cast("Literal[1, 2, 3]", stage), language_weights)
        count += 1
        seed += 1


# ---------------------------------------------------------------------------
# Test helpers (public so test modules can look up templates)
# ---------------------------------------------------------------------------


def _lookup_template_for_test(template_id: str) -> Template:
    """Public-for-tests helper to resolve a template by ID."""
    lib = _get_library()
    for t in lib.templates:
        if t.template_id == template_id:
            return t
    raise KeyError(template_id)


__all__ = [
    "Domain",
    "InvalidBudgetError",
    "InvalidLanguageError",
    "InvalidLanguageWeightError",
    "InvalidStageError",
    "LanguageCode",
    "MissingSlotError",
    "NoVariantForLanguageError",
    "RawBrief",
    "SlotDistribution",
    "SlotGrid",
    "TaskGeneratorError",
    "Template",
    "TemplateFileMissingError",
    "TemplateLibrary",
    "TemplateSchemaError",
    "UnicodeNormalizationError",
    "_lookup_template_for_test",
    "enumerate_variants",
    "generate",
    "load_templates",
    "reset_library_cache",
    "set_library_override",
    "stable_sub_seed",
]
