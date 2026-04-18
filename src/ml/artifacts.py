"""Artifact naming, persistence helpers, and manifests for ML models."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable


ARTIFACT_SCHEMA_VERSION = 3
MANIFEST_COMPATIBILITY_VERSION = "dual_model_manifest_v1"
ARTIFACT_CREATED_BY = "src.ml.trainer.DualModelTrainer"
MODEL_EXTENSIONS = {
    "cart": ".joblib",
    "lstm": ".pt",
    "bilstm": ".pt",
    "sarimax": ".joblib",
    "ets": ".joblib",
    "xgboost": ".joblib",
    "lightgbm": ".joblib",
    "stacking": ".joblib",
}
TASK_PREFIXES = {
    "trend": "trend_classifier",
    "return": "return_regressor",
    "profit": "profit_classifier",
}


def model_extension(algorithm: str) -> str:
    algo = algorithm.lower()
    if algo not in MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")
    return MODEL_EXTENSIONS[algo]


def artifact_filename(task: str, algorithm: str, horizon: str) -> str:
    if task not in TASK_PREFIXES:
        raise ValueError(f"Unsupported task '{task}'")
    return f"{TASK_PREFIXES[task]}_{algorithm.lower()}_{horizon.lower()}{model_extension(algorithm)}"


def artifact_path(model_root: Path, ticker: str, task: str, algorithm: str, horizon: str) -> Path:
    return model_root / ticker.upper() / artifact_filename(task=task, algorithm=algorithm, horizon=horizon)


def meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.joblib")


def scaler_path(model_path: Path) -> Path:
    return model_path.with_suffix(".scaler.joblib")


def manifest_path(model_root: Path, ticker: str) -> Path:
    return model_root / ticker.upper() / "manifest.json"


def ensure_ticker_dir(model_root: Path, ticker: str) -> Path:
    ticker_dir = model_root / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def _default_prediction_output_semantics() -> dict:
    return {
        "risk_output_field": "heuristic_scenario_risk",
        "risk_output_aliases": ["risk_assessment"],
        "risk_semantics": "heuristic_scenario_risk_not_calibrated_confidence",
        "uncertainty_methodology": "residual_based_normal_scenario_simulation_using_validation_error_scale",
        "calibration_status": "heuristic_not_calibrated",
        "interpretation_warning": (
            "Scenario risk uses residual-based simulated tails around the point forecast. "
            "It is not calibrated forecast confidence or a guaranteed loss bound."
        ),
        "deprecated_output_aliases": {
            "risk_assessment": {
                "alias_for": "heuristic_scenario_risk",
                "status": "deprecated_backward_compat_alias",
            }
        },
    }


def _default_evaluation_semantics(manifest: dict) -> dict:
    split_config = manifest.get("split_config", {})
    return {
        "evaluation_split_name": split_config.get("evaluation_split_name"),
        "metric_source": split_config.get("metric_source"),
        "validation_method": split_config.get("validation_method"),
    }


def _default_feature_governance() -> dict:
    return {
        "registry_path": "src/ml/features/feature_registry.json",
        "approved_feature_sets": {},
        "final_task_feature_sets": {},
        "feature_selection_evidence": {},
        "feature_build_mode": "full_research_mode",
        "price_reference_semantics": {
            "model_close_reference_column": "model_close_reference",
            "raw_close_column": "raw_close",
            "deprecated_raw_close_alias": "close_raw",
            "adjusted_close_available_from_live_vnstock_data": False,
        },
    }


def _normalize_risk_config(risk_config: dict | None) -> dict:
    config = dict(risk_config or {})
    if "scenario_confidence_levels" not in config and "risk_confidence_levels" in config:
        config["scenario_confidence_levels"] = list(config["risk_confidence_levels"])
    if "risk_output_field" not in config:
        config["risk_output_field"] = "heuristic_scenario_risk"
    config.setdefault(
        "uncertainty_methodology",
        "residual_based_normal_scenario_simulation_using_validation_error_scale",
    )
    config.setdefault("calibration_status", "heuristic_not_calibrated")
    config.setdefault(
        "interpretation_warning",
        "Residual-based scenario VaR/CVaR are heuristic tail summaries, not calibrated risk guarantees.",
    )
    config.setdefault(
        "deprecated_output_aliases",
        {
            "risk_assessment": {
                "alias_for": "heuristic_scenario_risk",
                "status": "deprecated_backward_compat_alias",
            }
        },
    )
    return config


def _normalize_calibration_payload(payload: dict | None) -> dict:
    calibration = dict(payload or {})
    calibration.setdefault(
        "uncertainty_methodology",
        "validation_residual_quantiles_not_probability_calibration",
    )
    calibration.setdefault("calibration_status", "not_probability_calibration")
    calibration.setdefault(
        "interpretation_warning",
        "This legacy calibration payload stores residual-error quantiles only; it is not a calibrated probability model.",
    )
    calibration.setdefault("deprecated_field_name", "calibration")
    return calibration


def normalize_manifest(manifest: dict) -> dict:
    normalized = deepcopy(manifest)

    stored_schema = int(
        normalized.get(
            "manifest_schema_version",
            normalized.get("schema_version", ARTIFACT_SCHEMA_VERSION),
        )
    )
    normalized.setdefault("schema_version", stored_schema)
    normalized.setdefault("manifest_schema_version", stored_schema)
    normalized.setdefault("compatibility_version", MANIFEST_COMPATIBILITY_VERSION)
    normalized.setdefault("artifact_created_by", ARTIFACT_CREATED_BY)
    semantics = _default_prediction_output_semantics()
    semantics.update(normalized.get("prediction_output_semantics", {}))
    normalized["prediction_output_semantics"] = semantics

    evaluation_semantics = _default_evaluation_semantics(normalized)
    evaluation_semantics.update(normalized.get("evaluation_semantics", {}))
    normalized["evaluation_semantics"] = evaluation_semantics

    feature_governance = _default_feature_governance()
    feature_governance.update(normalized.get("feature_governance", {}))
    normalized["feature_governance"] = feature_governance

    for horizon_info in normalized.get("horizons", {}).values():
        algorithms = horizon_info.get("algorithms", {})
        for algorithm_info in algorithms.values():
            merged_prediction_semantics = deepcopy(normalized["prediction_output_semantics"])
            merged_prediction_semantics.update(algorithm_info.get("prediction_output_semantics", {}))
            algorithm_info["prediction_output_semantics"] = merged_prediction_semantics

            merged_evaluation_metadata = deepcopy(normalized["evaluation_semantics"])
            merged_evaluation_metadata.update(algorithm_info.get("evaluation_metadata", {}))
            algorithm_info["evaluation_metadata"] = merged_evaluation_metadata
            if "risk_config" in algorithm_info:
                algorithm_info["risk_config"] = _normalize_risk_config(algorithm_info.get("risk_config"))
            if "calibration" in algorithm_info:
                algorithm_info["calibration"] = _normalize_calibration_payload(algorithm_info.get("calibration"))
            if "schema_version" not in algorithm_info:
                algorithm_info["schema_version"] = normalized["manifest_schema_version"]

    return normalized


def write_manifest(model_root: Path, ticker: str, manifest: dict) -> Path:
    path = manifest_path(model_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = deepcopy(manifest)
    manifest_payload.setdefault("schema_version", ARTIFACT_SCHEMA_VERSION)
    manifest_payload.setdefault("manifest_schema_version", ARTIFACT_SCHEMA_VERSION)
    manifest_payload.setdefault("compatibility_version", MANIFEST_COMPATIBILITY_VERSION)
    manifest_payload.setdefault("artifact_created_by", ARTIFACT_CREATED_BY)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
    return path


def load_manifest(model_root: Path, ticker: str) -> dict:
    path = manifest_path(model_root, ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest found for {ticker}. Expected {path}. Run training first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return normalize_manifest(json.load(handle))


def cleanup_ticker_dir(model_root: Path, ticker: str) -> None:
    """Delete stale model artifacts for a ticker before writing a fresh bundle."""

    ticker_dir = ensure_ticker_dir(model_root, ticker)
    for pattern in ("*.joblib", "*.pt", "*.json", "*.csv"):
        for file_path in ticker_dir.glob(pattern):
            try:
                file_path.unlink()
            except PermissionError:
                pass


def relative_paths(paths: Iterable[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]
