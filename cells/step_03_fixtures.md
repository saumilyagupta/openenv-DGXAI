# Load static fixtures

Lazy, NFC-normalized, validated loaders for the four authored data artifacts: `task_briefs/templates.yaml`, `task_briefs/i18n.yaml`, `drift_patterns/drifts.yaml`, and the per-domain `api_schemas/*` JSON registries. Loaders raise typed `DatasetError` subclasses on any authoring drift, schema break, or cross-file consistency violation (datasets.md §3.3).
