"""Cell 03 — Static fixture loaders for DriftCall data artifacts.

Implements the loader contract in ``docs/modules/datasets.md`` §§2–5. Each
loader is a lazy path-keyed singleton that reads, NFC-normalizes, and validates
a single on-disk artifact, then returns a frozen dataclass wrapped in
``MappingProxyType`` where mappings appear.

Artifacts covered:

  * ``data/task_briefs/templates.yaml``    — TemplateLibrary
  * ``data/task_briefs/i18n.yaml``         — I18nLibrary
  * ``data/drift_patterns/drifts.yaml``    — DriftPatternLibrary
  * ``data/api_schemas/<domain>/v<N>.json`` — APISchemaRegistry

Loaders raise one of the ``DatasetError`` subclasses declared below on any
authoring error — malformed YAML/JSON, schema violation, NFC failure, or the
21 cross-file consistency assertions enumerated in datasets.md §3.3.
"""

from __future__ import annotations

import hashlib
import json
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

if TYPE_CHECKING:
    from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LanguageCode = Literal["hi", "ta", "kn", "en", "hinglish"]
Domain = Literal["airline", "cab", "restaurant", "hotel"]

_LANGUAGE_CODES: frozenset[str] = frozenset({"hi", "ta", "kn", "en", "hinglish"})
_PRIMARY_DOMAINS: frozenset[str] = frozenset({"airline", "cab", "restaurant", "hotel"})
_VENDOR_DOMAINS: frozenset[str] = frozenset(
    {"airline", "cab", "restaurant", "hotel", "payment"}
)
_DRIFT_TYPES: frozenset[str] = frozenset(
    {"schema", "policy", "tnc", "pricing", "auth"}
)
_EXPECTED_PATTERN_COUNT = 20
_EXPECTED_SCHEMA_VERSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "airline": ("v1", "v2", "v3"),
        "cab": ("v1", "v2", "v3"),
        "restaurant": ("v1", "v2", "v3"),
        "hotel": ("v1", "v2", "v3"),
        "payment": ("v1", "v2"),
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Base class for every fixture loader error."""


class DatasetFileMissingError(DatasetError):
    """Raised when an authored data file is absent from disk."""


class MalformedYAMLError(DatasetError):
    """Raised when a YAML file fails to parse (file path + line preserved)."""


class MalformedJSONError(DatasetError):
    """Raised when a JSON file fails to parse (file path + line preserved)."""


class DatasetSchemaError(DatasetError):
    """Raised on type / shape / required-key violations of an authored file."""


class UnknownLanguageKeyError(DatasetError):
    """Raised when a language key ∉ LanguageCode appears in a YAML file."""


class UnicodeNFDError(DatasetError):
    """Raised when a loaded string is not NFC-normalized after defensive pass."""


class DriftPatternOrphanError(DatasetError):
    """Raised when a drift pattern references an API schema version that is missing."""


class DuplicateDriftPatternIdError(DatasetError):
    """Raised when drifts.yaml contains two entries sharing the same id."""


# ---------------------------------------------------------------------------
# Frozen dataclasses (library types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotDistribution:
    kind: Literal["choices", "uniform"]
    choices: tuple[str, ...] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class Template:
    template_id: str
    domain: str
    intent: str
    min_stage: Literal[1, 2, 3]
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    constraints_template: Mapping[str, SlotDistribution]
    drift_slot_tags: tuple[str, ...]
    language_variants: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class TemplateLibrary:
    templates: tuple[Template, ...]
    source_sha256: str


@dataclass(frozen=True)
class I18nLibrary:
    strings: Mapping[str, Mapping[str, str]]
    source_sha256: str


@dataclass(frozen=True)
class DriftPattern:
    id: str
    drift_type: str
    domain: str
    from_version: str
    to_version: str
    description: str
    mutation: Mapping[str, Any]
    detection_hints: tuple[str, ...]


@dataclass(frozen=True)
class DriftPatternLibrary:
    patterns: Mapping[str, DriftPattern]
    by_domain: Mapping[str, tuple[str, ...]]
    by_type: Mapping[str, tuple[str, ...]]
    source_sha256: str


@dataclass(frozen=True)
class APISchema:
    domain: str
    version: str
    schema: Mapping[str, Any]
    source_sha256: str


@dataclass(frozen=True)
class APISchemaRegistry:
    schemas: Mapping[str, Mapping[str, APISchema]]

    def get(self, domain: str, version: str) -> APISchema:
        try:
            return self.schemas[domain][version]
        except KeyError as exc:
            raise DatasetSchemaError(
                f"no schema registered for domain={domain!r} version={version!r}"
            ) from exc

    def versions(self, domain: str) -> tuple[str, ...]:
        try:
            return tuple(self.schemas[domain].keys())
        except KeyError as exc:
            raise DatasetSchemaError(f"unknown domain {domain!r}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nfc(value: str) -> str:
    """NFC-normalize ``value``; raise on post-normalization non-NFC (defensive)."""

    normalized = unicodedata.normalize("NFC", value)
    if not unicodedata.is_normalized("NFC", normalized):
        raise UnicodeNFDError(
            f"string failed NFC round-trip: {value!r}"
        )
    return normalized


def _nfc_deep(value: Any) -> Any:
    """Recursively NFC-normalize every string inside nested dict/list structures."""

    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, list):
        return [_nfc_deep(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_nfc_deep(v) for v in value)
    if isinstance(value, dict):
        return {_nfc(k) if isinstance(k, str) else k: _nfc_deep(v) for k, v in value.items()}
    return value


def _file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise DatasetFileMissingError(f"{path} not found") from exc
    except OSError as exc:
        raise DatasetFileMissingError(f"{path}: {exc}") from exc


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_yaml(path: Path) -> Any:
    data = _file_bytes(path)
    try:
        return yaml.safe_load(data)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else -1
        raise MalformedYAMLError(f"{path}:{line}: {exc}") from exc


def _parse_json(path: Path) -> Any:
    data = _file_bytes(path)
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise MalformedJSONError(f"{path}:{exc.lineno}: {exc.msg}") from exc


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DatasetSchemaError(msg)


def _as_tuple_of_str(value: Any, field: str, *, path: Path) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{path}: {field!r} must be a list")
    for item in value:
        _require(isinstance(item, str), f"{path}: {field!r} items must be strings")
    return tuple(_nfc(v) for v in value)


# ---------------------------------------------------------------------------
# Path-keyed singleton caches
# ---------------------------------------------------------------------------

_TEMPLATE_CACHE: dict[Path, TemplateLibrary] = {}
_I18N_CACHE: dict[Path, I18nLibrary] = {}
_DRIFT_CACHE: dict[Path, DriftPatternLibrary] = {}
_SCHEMA_CACHE: dict[Path, APISchemaRegistry] = {}
_CACHE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Templates loader
# ---------------------------------------------------------------------------


def _build_slot_distribution(raw: Any, slot_name: str, path: Path) -> SlotDistribution:
    _require(
        isinstance(raw, dict),
        f"{path}: slot {slot_name!r} definition must be a mapping",
    )
    if "choices" in raw:
        choices = _as_tuple_of_str(raw["choices"], f"{slot_name}.choices", path=path)
        _require(
            len(choices) >= 1,
            f"{path}: slot {slot_name!r} choices must be non-empty",
        )
        return SlotDistribution(kind="choices", choices=choices)
    if raw.get("distribution") == "uniform":
        for req in ("low", "high", "step"):
            _require(
                req in raw,
                f"{path}: slot {slot_name!r} uniform dist missing {req!r}",
            )
            _require(
                isinstance(raw[req], (int, float)),
                f"{path}: slot {slot_name!r} {req!r} must be numeric",
            )
        low = float(raw["low"])
        high = float(raw["high"])
        step = float(raw["step"])
        _require(
            high >= low and step > 0,
            f"{path}: slot {slot_name!r} invalid uniform range",
        )
        return SlotDistribution(kind="uniform", low=low, high=high, step=step)
    raise DatasetSchemaError(
        f"{path}: slot {slot_name!r} must declare either 'choices' or 'distribution: uniform'"
    )


def _build_template(raw: Any, path: Path) -> Template:
    _require(isinstance(raw, dict), f"{path}: each template must be a mapping")
    for req in (
        "template_id",
        "domain",
        "intent",
        "min_stage",
        "required_slots",
        "optional_slots",
        "constraints_template",
        "drift_slot_tags",
        "language_variants",
    ):
        _require(req in raw, f"{path}: template missing required key {req!r}")

    template_id = _nfc(str(raw["template_id"]))
    domain = _nfc(str(raw["domain"]))
    intent = _nfc(str(raw["intent"]))
    min_stage = raw["min_stage"]

    _require(
        domain in _PRIMARY_DOMAINS,
        f"{path}: template {template_id!r} has unknown domain {domain!r}",
    )
    _require(
        min_stage in (1, 2, 3),
        f"{path}: template {template_id!r} min_stage must be 1|2|3, got {min_stage!r}",
    )

    required_slots = _as_tuple_of_str(
        raw["required_slots"], f"{template_id}.required_slots", path=path
    )
    optional_slots = _as_tuple_of_str(
        raw["optional_slots"], f"{template_id}.optional_slots", path=path
    )
    drift_slot_tags = _as_tuple_of_str(
        raw["drift_slot_tags"], f"{template_id}.drift_slot_tags", path=path
    )

    raw_constraints = raw["constraints_template"]
    _require(
        isinstance(raw_constraints, dict),
        f"{path}: template {template_id!r} constraints_template must be a mapping",
    )
    constraints = {
        _nfc(slot_name): _build_slot_distribution(slot_def, slot_name, path)
        for slot_name, slot_def in raw_constraints.items()
    }

    raw_variants = raw["language_variants"]
    _require(
        isinstance(raw_variants, dict),
        f"{path}: template {template_id!r} language_variants must be a mapping",
    )
    variants: dict[str, tuple[str, ...]] = {}
    for lang_key, utterances in raw_variants.items():
        _require(
            isinstance(lang_key, str),
            f"{path}: template {template_id!r} language key must be string",
        )
        if lang_key not in _LANGUAGE_CODES:
            raise UnknownLanguageKeyError(
                f"{path}: template {template_id!r} has unknown language key {lang_key!r}"
            )
        _require(
            isinstance(utterances, list) and len(utterances) >= 1,
            f"{path}: template {template_id!r} variants[{lang_key!r}] must be non-empty list",
        )
        for u in utterances:
            _require(
                isinstance(u, str),
                f"{path}: template {template_id!r} variants[{lang_key!r}] items must be strings",
            )
        variants[lang_key] = tuple(_nfc(u) for u in utterances)

    missing_langs = _LANGUAGE_CODES - variants.keys()
    _require(
        not missing_langs,
        f"{path}: template {template_id!r} missing language_variants for {sorted(missing_langs)}",
    )

    return Template(
        template_id=template_id,
        domain=domain,
        intent=intent,
        min_stage=min_stage,
        required_slots=required_slots,
        optional_slots=optional_slots,
        constraints_template=MappingProxyType(constraints),
        drift_slot_tags=drift_slot_tags,
        language_variants=MappingProxyType(variants),
    )


def load_templates(
    path: Path | str = "data/task_briefs/templates.yaml",
) -> TemplateLibrary:
    """Load + validate the task-brief template library (datasets.md §3.3)."""

    resolved = Path(path).resolve()
    cached = _TEMPLATE_CACHE.get(resolved)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _TEMPLATE_CACHE.get(resolved)
        if cached is not None:
            return cached
        raw = _parse_yaml(resolved)
        _require(
            isinstance(raw, list) and len(raw) >= 1,
            f"{resolved}: templates.yaml must be a non-empty list",
        )
        templates = tuple(_build_template(entry, resolved) for entry in raw)

        seen_ids = set()
        seen_domains = set()
        for tpl in templates:
            _require(
                tpl.template_id not in seen_ids,
                f"{resolved}: duplicate template_id {tpl.template_id!r}",
            )
            seen_ids.add(tpl.template_id)
            seen_domains.add(tpl.domain)
        missing_primary = _PRIMARY_DOMAINS - seen_domains
        _require(
            not missing_primary,
            f"{resolved}: missing templates for domains {sorted(missing_primary)}",
        )

        library = TemplateLibrary(
            templates=templates,
            source_sha256=_sha256_hex(_file_bytes(resolved)),
        )
        _TEMPLATE_CACHE[resolved] = library
        return library


# ---------------------------------------------------------------------------
# I18n loader
# ---------------------------------------------------------------------------


def load_i18n(path: Path | str = "data/task_briefs/i18n.yaml") -> I18nLibrary:
    """Load + NFC-normalize the i18n lookup (datasets.md §4.2)."""

    resolved = Path(path).resolve()
    cached = _I18N_CACHE.get(resolved)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _I18N_CACHE.get(resolved)
        if cached is not None:
            return cached
        raw = _parse_yaml(resolved)
        _require(
            isinstance(raw, dict) and len(raw) >= 1,
            f"{resolved}: i18n.yaml must be a non-empty mapping",
        )

        strings: dict[str, Mapping[str, str]] = {}
        for lang_key, entries in raw.items():
            if lang_key not in _LANGUAGE_CODES:
                raise UnknownLanguageKeyError(
                    f"{resolved}: unknown language key {lang_key!r}"
                )
            _require(
                isinstance(entries, dict),
                f"{resolved}: i18n[{lang_key!r}] must be a mapping",
            )
            inner: dict[str, str] = {}
            for k, v in entries.items():
                _require(
                    isinstance(k, str) and isinstance(v, str),
                    f"{resolved}: i18n[{lang_key!r}] entries must be string→string",
                )
                inner[_nfc(k)] = _nfc(v)
            strings[lang_key] = MappingProxyType(inner)

        missing = _LANGUAGE_CODES - strings.keys()
        _require(
            not missing,
            f"{resolved}: i18n.yaml missing languages {sorted(missing)}",
        )

        library = I18nLibrary(
            strings=MappingProxyType(strings),
            source_sha256=_sha256_hex(_file_bytes(resolved)),
        )
        _I18N_CACHE[resolved] = library
        return library


# ---------------------------------------------------------------------------
# Drift patterns loader
# ---------------------------------------------------------------------------


def _build_drift_pattern(raw: Any, path: Path) -> DriftPattern:
    _require(isinstance(raw, dict), f"{path}: each drift entry must be a mapping")
    for req in (
        "id",
        "drift_type",
        "domain",
        "from_version",
        "to_version",
        "description",
        "mutation",
        "detection_hints",
    ):
        _require(req in raw, f"{path}: drift entry missing required key {req!r}")

    pid = _nfc(str(raw["id"]))
    drift_type = _nfc(str(raw["drift_type"]))
    domain = _nfc(str(raw["domain"]))
    from_version = _nfc(str(raw["from_version"]))
    to_version = _nfc(str(raw["to_version"]))
    description = _nfc(str(raw["description"]))

    _require(
        drift_type in _DRIFT_TYPES,
        f"{path}: drift {pid!r} has unknown drift_type {drift_type!r}",
    )
    _require(
        domain in _VENDOR_DOMAINS,
        f"{path}: drift {pid!r} has unknown domain {domain!r}",
    )

    mutation_raw = raw["mutation"]
    _require(
        isinstance(mutation_raw, dict) and len(mutation_raw) >= 1,
        f"{path}: drift {pid!r} mutation must be a non-empty mapping",
    )
    mutation = _nfc_deep(mutation_raw)

    hints_raw = raw["detection_hints"]
    _require(
        isinstance(hints_raw, list) and len(hints_raw) >= 1,
        f"{path}: drift {pid!r} detection_hints must be a non-empty list",
    )
    for h in hints_raw:
        _require(
            isinstance(h, str) and h.strip() != "",
            f"{path}: drift {pid!r} detection_hints entries must be non-empty strings",
        )
    hints = tuple(_nfc(h) for h in hints_raw)

    return DriftPattern(
        id=pid,
        drift_type=drift_type,
        domain=domain,
        from_version=from_version,
        to_version=to_version,
        description=description,
        mutation=MappingProxyType(dict(mutation)),
        detection_hints=hints,
    )


def load_drift_patterns(
    path: Path | str = "data/drift_patterns/drifts.yaml",
    *,
    schema_registry: APISchemaRegistry | None = None,
) -> DriftPatternLibrary:
    """Load + validate the 20-pattern drift catalogue (datasets.md §3.3, drift_injector.md §4.4)."""

    resolved = Path(path).resolve()
    cached = _DRIFT_CACHE.get(resolved)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _DRIFT_CACHE.get(resolved)
        if cached is not None:
            return cached
        raw = _parse_yaml(resolved)
        _require(
            isinstance(raw, list),
            f"{resolved}: drifts.yaml must be a list",
        )
        _require(
            len(raw) == _EXPECTED_PATTERN_COUNT,
            f"{resolved}: expected {_EXPECTED_PATTERN_COUNT} drift patterns, got {len(raw)}",
        )

        patterns_list = [_build_drift_pattern(entry, resolved) for entry in raw]

        ids_seen: dict[str, int] = {}
        for idx, p in enumerate(patterns_list):
            if p.id in ids_seen:
                raise DuplicateDriftPatternIdError(
                    f"{resolved}: duplicate drift pattern id {p.id!r} at entries {ids_seen[p.id]} and {idx}"
                )
            ids_seen[p.id] = idx

        registry = schema_registry if schema_registry is not None else load_api_schemas()
        for p in patterns_list:
            for ver in (p.from_version, p.to_version):
                if p.domain not in registry.schemas or ver not in registry.schemas[p.domain]:
                    raise DriftPatternOrphanError(
                        f"{resolved}: drift {p.id!r} references missing schema "
                        f"{p.domain}/{ver}"
                    )

        patterns = MappingProxyType({p.id: p for p in patterns_list})
        by_domain: dict[str, list[str]] = {}
        by_type: dict[str, list[str]] = {}
        for p in patterns_list:
            by_domain.setdefault(p.domain, []).append(p.id)
            by_type.setdefault(p.drift_type, []).append(p.id)

        library = DriftPatternLibrary(
            patterns=patterns,
            by_domain=MappingProxyType({k: tuple(v) for k, v in by_domain.items()}),
            by_type=MappingProxyType({k: tuple(v) for k, v in by_type.items()}),
            source_sha256=_sha256_hex(_file_bytes(resolved)),
        )
        _DRIFT_CACHE[resolved] = library
        return library


# ---------------------------------------------------------------------------
# API schema loader
# ---------------------------------------------------------------------------


def _load_single_schema(domain: str, version: str, path: Path) -> APISchema:
    data = _parse_json(path)
    _require(
        isinstance(data, dict),
        f"{path}: JSON Schema must be an object",
    )
    try:
        Draft202012Validator.check_schema(data)
    except SchemaError as exc:
        raise DatasetSchemaError(
            f"{path}: not a valid JSON Schema 2020-12: {exc.message}"
        ) from exc
    return APISchema(
        domain=domain,
        version=version,
        schema=MappingProxyType(_nfc_deep(data)),
        source_sha256=_sha256_hex(_file_bytes(path)),
    )


def load_api_schemas(
    root: Path | str = "data/api_schemas",
) -> APISchemaRegistry:
    """Load every ``<domain>/v<N>.json`` file under ``root`` (datasets.md §4.4)."""

    resolved = Path(root).resolve()
    cached = _SCHEMA_CACHE.get(resolved)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _SCHEMA_CACHE.get(resolved)
        if cached is not None:
            return cached
        if not resolved.is_dir():
            raise DatasetFileMissingError(f"{resolved} is not a directory")

        schemas: dict[str, dict[str, APISchema]] = {}
        for domain, expected_versions in _EXPECTED_SCHEMA_VERSIONS.items():
            domain_dir = resolved / domain
            if not domain_dir.is_dir():
                raise DatasetFileMissingError(
                    f"{resolved}: expected domain directory {domain_dir}"
                )
            per_version: dict[str, APISchema] = {}
            for version in expected_versions:
                file_path = domain_dir / f"{version}.json"
                per_version[version] = _load_single_schema(domain, version, file_path)
            schemas[domain] = per_version

        registry = APISchemaRegistry(
            schemas=MappingProxyType(
                {d: MappingProxyType(v) for d, v in schemas.items()}
            ),
        )
        _SCHEMA_CACHE[resolved] = registry
        return registry


# ---------------------------------------------------------------------------
# Cache-reset helper (tests only)
# ---------------------------------------------------------------------------


def _reset_caches() -> None:
    """Clear every loader cache. Intended for use by tests only."""

    with _CACHE_LOCK:
        _TEMPLATE_CACHE.clear()
        _I18N_CACHE.clear()
        _DRIFT_CACHE.clear()
        _SCHEMA_CACHE.clear()


__all__ = [
    "APISchema",
    "APISchemaRegistry",
    "DatasetError",
    "DatasetFileMissingError",
    "DatasetSchemaError",
    "Domain",
    "DriftPattern",
    "DriftPatternLibrary",
    "DriftPatternOrphanError",
    "DuplicateDriftPatternIdError",
    "I18nLibrary",
    "LanguageCode",
    "MalformedJSONError",
    "MalformedYAMLError",
    "SlotDistribution",
    "Template",
    "TemplateLibrary",
    "UnicodeNFDError",
    "UnknownLanguageKeyError",
    "load_api_schemas",
    "load_drift_patterns",
    "load_i18n",
    "load_templates",
]
