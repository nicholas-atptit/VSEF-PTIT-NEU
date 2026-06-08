"""Protected evidence registry and artifact path guards."""

from __future__ import annotations

from pathlib import Path

PROTECTED_EVIDENCE_PATTERNS = (
    "reports/project_review",
    "reports/paper/qml_kernel_feature_vn30",
    "reports/generated/vn30_qml_forecasting",
    "reports/generated/vn30_model_universe_direction_price",
    "reports/generated/vn30_index_group_range_forecast",
    "reports/generated/vn_forecast_engine_v1",
    "reports/results/VN30_QML",
    "reports/results/VN30_MODEL_UNIVERSE",
    "reports/results/VN30_INDEX_GROUP_PRICE_RANGE",
    "reports/results/VN_FORECAST_ENGINE",
    "reports/claims/VN30_QML",
    "reports/claims/VN30_MODEL_UNIVERSE",
    "reports/claims/VN30_INDEX_GROUP_PRICE_RANGE",
    "reports/claims/VN_FORECAST_ENGINE",
    "reports/protocols/VN30_QML",
    "reports/protocols/VN30_HIGH_ACCURACY_69_FUTURE_BLIND_PROTOCOL.md",
    "data",
    "outputs",
    "archive",
    "paper_evidence_export",
    "paper_evidence_raw_full_export",
)

GENERATED_ARTIFACT_ROOTS = (
    "reports/generated",
    "reports/results",
    "reports/claims",
    "reports/protocols",
    "outputs",
)


def normalized_repo_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def is_protected_evidence_path(path: str | Path) -> bool:
    candidate = normalized_repo_path(path)
    return any(candidate == prefix or candidate.startswith(prefix + "/") or candidate.startswith(prefix) for prefix in PROTECTED_EVIDENCE_PATTERNS)


def assert_deletion_allowed(path: str | Path) -> None:
    if is_protected_evidence_path(path):
        raise PermissionError(f"Protected evidence path cannot be deleted: {normalized_repo_path(path)}")


def active_evidence_registry() -> tuple[str, ...]:
    return PROTECTED_EVIDENCE_PATTERNS
