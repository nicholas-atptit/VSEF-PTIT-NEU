from __future__ import annotations

from src.core.model_governance import RUN_MODES
from src.forecast.registry import forecast_model_governance_table, supported_forecast_models
from src.reporting.quant_core import build_quant_core_manifest


def test_supported_forecast_models_filter_by_run_mode() -> None:
    assert supported_forecast_models(run_mode="full_forecast") == [
        "lightgbm",
        "xgboost",
        "random_forest",
        "ets",
        "sarimax",
        "naive",
        "moving_average",
        "linear",
        "ridge",
        "lasso",
    ]
    assert supported_forecast_models(run_mode="research_core") == [
        "lightgbm",
        "xgboost",
        "random_forest",
        "ets",
        "sarimax",
    ]
    assert supported_forecast_models(run_mode="decision_core") == [
        "lightgbm",
        "xgboost",
        "random_forest",
    ]
    assert supported_forecast_models(run_mode="baseline_only") == [
        "naive",
        "moving_average",
    ]


def test_governance_table_schema_is_consistent_across_run_modes() -> None:
    schemas = [tuple(forecast_model_governance_table(run_mode=run_mode).columns) for run_mode in RUN_MODES]
    assert all(schema == schemas[0] for schema in schemas[1:])


def test_quant_core_manifest_records_run_mode_and_governance() -> None:
    governance = forecast_model_governance_table(run_mode="research_core")
    manifest = build_quant_core_manifest(
        git_metadata={"branch": "test", "commit_hash": "abc123", "is_dirty": False},
        runtime={"python_executable": "python"},
        dependency_versions={"pandas": "3.0.1"},
        command="python scripts/run_quant_core.py --run-mode research_core",
        requested_models=["lightgbm", "xgboost"],
        evaluated_models=["lightgbm", "xgboost"],
        skipped_models=[],
        seed=42,
        matrix_config={"preset": "smoke"},
        run_counts={"scenario_count": 1, "forecast_rows": 10},
        artifact_paths={"model_governance": "artifacts/quant_core/model_governance.csv"},
        started_at="2026-04-19T00:00:00+00:00",
        completed_at="2026-04-19T00:05:00+00:00",
        run_mode="research_core",
        requested_model_roles=["primary_research"],
        governance_frame=governance,
    )

    assert manifest["manifest_type"] == "quant_core_run_manifest_v1"
    assert manifest["run_mode"] == "research_core"
    assert manifest["run_mode_spec"]["enablement_field"] == "enabled_for_research_core"
    assert manifest["requested_model_roles"] == ["primary_research"]
    assert manifest["governance_output"]["model_count"] == len(governance)
