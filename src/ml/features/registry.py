from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


FEATURE_REGISTRY_VERSION = 1
FEATURE_REGISTRY_PATH = Path(__file__).with_name("feature_registry.json")


def load_feature_registry(path: Path | str | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else FEATURE_REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_loaded_registry(payload)
    return payload


def validate_loaded_registry(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Feature registry payload must be a JSON object")
    if int(payload.get("registry_version", 0)) != FEATURE_REGISTRY_VERSION:
        raise ValueError(
            f"Unsupported feature registry version: {payload.get('registry_version')}"
        )
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Feature registry must contain a non-empty 'features' list")
    required_fields = {
        "feature_name",
        "category",
        "provenance",
        "formula_logic",
        "expected_availability",
        "leakage_note",
        "usable_for_forecast",
        "usable_for_regime",
        "usable_for_risk",
        "status",
    }
    for entry in features:
        missing = required_fields - set(entry)
        if missing:
            raise ValueError(f"Feature registry entry missing required fields: {sorted(missing)}")
    approved_sets = payload.get("approved_feature_sets", {})
    if not isinstance(approved_sets, dict):
        raise ValueError("Feature registry must contain an 'approved_feature_sets' object")
    final_sets = payload.get("final_task_feature_sets", {})
    if final_sets and not isinstance(final_sets, dict):
        raise ValueError("Feature registry 'final_task_feature_sets' must be an object when present")
    evidence = payload.get("feature_selection_evidence", {})
    if evidence and not isinstance(evidence, dict):
        raise ValueError("Feature registry 'feature_selection_evidence' must be an object when present")
    sentiment = payload.get("sentiment_policy", {})
    if sentiment and not isinstance(sentiment, dict):
        raise ValueError("Feature registry 'sentiment_policy' must be an object when present")
    build_modes = payload.get("feature_build_modes", {})
    if build_modes and not isinstance(build_modes, dict):
        raise ValueError("Feature registry 'feature_build_modes' must be an object when present")


def feature_entries(path: Path | str | None = None) -> list[dict[str, Any]]:
    return list(load_feature_registry(path).get("features", []))


def feature_lookup(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    return {
        str(entry["feature_name"]): entry
        for entry in feature_entries(path)
    }


def approved_feature_sets(path: Path | str | None = None) -> dict[str, list[str]]:
    registry = load_feature_registry(path)
    return {
        str(name): [str(column) for column in columns]
        for name, columns in registry.get("approved_feature_sets", {}).items()
    }


def final_task_feature_sets(path: Path | str | None = None) -> dict[str, list[str]]:
    registry = load_feature_registry(path)
    return {
        str(name): [str(column) for column in columns]
        for name, columns in registry.get("final_task_feature_sets", {}).items()
    }


def feature_selection_evidence(path: Path | str | None = None) -> dict[str, Any]:
    return dict(load_feature_registry(path).get("feature_selection_evidence", {}))


def sentiment_policy(path: Path | str | None = None) -> dict[str, Any]:
    return dict(load_feature_registry(path).get("sentiment_policy", {}))


def feature_build_modes(path: Path | str | None = None) -> dict[str, Any]:
    return dict(load_feature_registry(path).get("feature_build_modes", {}))


def price_reference_semantics(path: Path | str | None = None) -> dict[str, Any]:
    return dict(load_feature_registry(path).get("price_reference_semantics", {}))


def resolve_feature_set(
    set_names: str | Iterable[str],
    *,
    available_columns: Iterable[str],
    registry: dict[str, Any] | None = None,
) -> list[str]:
    loaded = registry or load_feature_registry()
    approved = {
        str(name): [str(column) for column in columns]
        for name, columns in loaded.get("approved_feature_sets", {}).items()
    }
    if isinstance(set_names, str):
        names = [set_names]
    else:
        names = [str(name) for name in set_names]

    available = set(str(column) for column in available_columns)
    selected: list[str] = []
    for name in names:
        for column in approved.get(name, []):
            if column in available and column not in selected:
                selected.append(column)
    return selected


def resolve_task_feature_set(
    task_name: str,
    *,
    available_columns: Iterable[str],
    registry: dict[str, Any] | None = None,
) -> list[str]:
    loaded = registry or load_feature_registry()
    final_sets = {
        str(name): [str(column) for column in columns]
        for name, columns in loaded.get("final_task_feature_sets", {}).items()
    }
    available = set(str(column) for column in available_columns)
    selected: list[str] = []
    for column in final_sets.get(str(task_name), []):
        if column in available and column not in selected:
            selected.append(column)
    return selected


def validate_feature_registry_against_columns(
    columns: Iterable[str],
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = registry or load_feature_registry()
    lookup = {
        str(entry["feature_name"]): entry
        for entry in loaded.get("features", [])
    }
    observed = [str(column) for column in columns]
    missing_from_registry = sorted(column for column in observed if column not in lookup)

    invalid_approved_features: dict[str, list[str]] = {}
    approved = loaded.get("approved_feature_sets", {})
    for set_name, feature_names in approved.items():
        invalid = [
            str(feature_name)
            for feature_name in feature_names
            if feature_name not in lookup
        ]
        if invalid:
            invalid_approved_features[str(set_name)] = sorted(invalid)

    invalid_final_task_features: dict[str, list[str]] = {}
    final_sets = loaded.get("final_task_feature_sets", {})
    for set_name, feature_names in final_sets.items():
        invalid = [
            str(feature_name)
            for feature_name in feature_names
            if feature_name not in lookup
        ]
        if invalid:
            invalid_final_task_features[str(set_name)] = sorted(invalid)

    return {
        "registry_feature_count": int(len(lookup)),
        "observed_feature_count": int(len(observed)),
        "missing_from_registry": missing_from_registry,
        "invalid_approved_features": invalid_approved_features,
        "invalid_final_task_features": invalid_final_task_features,
    }
