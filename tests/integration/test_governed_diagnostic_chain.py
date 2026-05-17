from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_quant_core as runner
import src.evaluation.quant_core as quant_core
from src.core.runtime_mode import RuntimeMode, build_data_provenance, ensure_mock_allowed
from src.evaluation.walkforward import WalkForwardEvaluator
from src.ml.feature_engineering import FeatureEngineer
from src.ml.features.registry import resolve_task_feature_set
from src.phase3_router.schema import ROUTE_DECISIONS


RECOMMENDATION_TOKEN_RE = re.compile(r"\b(?:BUY|SELL|STRONG_BUY|STRONG_SELL)\b")
AUTHORITY_TOKEN_RE = re.compile(
    r"\b(?:broker|order|execute_order|submit_order|execution_authority|"
    r"final_trade_authority|trade_authority|final_recommendation)\b",
    flags=re.IGNORECASE,
)

GOVERNED_ARTIFACT_KEYS = {
    "full_model_predictions",
    "forecast_summary",
    "risk_summary",
    "analysis_packets",
    "scenario_probability",
    "scenario_dominance_summary",
    "risk_governance_summary",
    "risk_adjusted_candidates",
    "risk_override_log",
    "decision_lane_enriched_candidates",
    "portfolio_allocation",
    "portfolio_summary",
    "portfolio_risk_summary",
    "portfolio_decision_cards",
    "allocator_manifest",
    "router_decisions",
    "router_summary",
    "router_manifest",
}


def _fixture_ohlcv(ticker: str, rows: int = 96) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    index = np.arange(rows, dtype=float)
    close = 100.0 + index * 0.50 + np.sin(index / 4.0) * 0.20
    open_ = close * (1.0 + np.cos(index / 5.0) * 0.001)
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = 900_000 + (index.astype(int) % 11) * 12_500
    daily_return = pd.Series(close).pct_change().fillna(0.0)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "m_ret": daily_return * 0.70,
            "s_ret": daily_return * 0.85,
            "market_breadth": 0.55 + np.sin(index / 8.0) * 0.03,
            "foreign_net_volume": np.sin(index / 6.0) * 1_500.0,
            "foreign_net_value": np.sin(index / 6.0) * 150_000.0,
            "fx_usdvnd": 24_000 + np.cos(index / 13.0) * 15.0,
            "interest_rate": 0.045 + np.sin(index / 17.0) * 0.001,
        }
    )


def _write_feature_fixture(prepared_dir: Path, ticker: str) -> dict[str, object]:
    provenance = build_data_provenance(
        source="phase8_deterministic_feature_fixture",
        uses_mock_data=False,
        fallback_triggered=False,
        runtime_mode=RuntimeMode.RESEARCH,
        reason="controlled fixture data for governed diagnostic-chain integration test",
    )
    raw = _fixture_ohlcv(ticker)
    features = FeatureEngineer().transform(raw, drop_na=False, build_mode="fast_core_mode")
    features = pd.concat(
        [
            features.copy(),
            pd.DataFrame(
                {
                    "source_provenance": provenance["source"],
                    "fixture_runtime_mode": provenance["runtime_mode"],
                },
                index=features.index,
            ),
        ],
        axis=1,
    )

    selected_features = resolve_task_feature_set(
        "regression_forecasting",
        available_columns=features.columns,
    )
    assert selected_features, "fixture feature generation produced no governed forecast features"

    prepared_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(prepared_dir / f"{ticker}.csv", index=False)
    return provenance | {"selected_feature_count": len(selected_features)}


def _patch_feature_fixture_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    prepared_dir: Path,
    source_label: str,
) -> None:
    class FixtureWalkForwardEvaluator(WalkForwardEvaluator):
        def __init__(self, config):
            super().__init__(
                replace(
                    config,
                    prepared_dir=str(prepared_dir),
                    raw_dir=str(prepared_dir),
                )
            )

        def _load_prepared_frame(self, ticker: str) -> tuple[pd.DataFrame, str]:
            frame, _ = super()._load_prepared_frame(ticker)
            return frame, source_label

    monkeypatch.setattr(quant_core, "WalkForwardEvaluator", FixtureWalkForwardEvaluator)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_paths(run_manifest: dict[str, object], output_dir: Path) -> list[Path]:
    artifact_paths = run_manifest["artifact_paths"]
    assert isinstance(artifact_paths, dict)
    paths: list[Path] = []
    for name, value in artifact_paths.items():
        if name in GOVERNED_ARTIFACT_KEYS:
            path = Path(str(value))
            if not path.is_absolute():
                path = output_dir / path
            assert path.exists(), f"missing governed artifact: {name}"
            paths.append(path)
    paths.extend(
        [
            output_dir / "run_manifest.json",
            output_dir / "risk_manifest.json",
            output_dir / "allocator_manifest.json",
            output_dir / "router_manifest.json",
        ]
    )
    return paths


def _assert_no_forbidden_authority_terms(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert RECOMMENDATION_TOKEN_RE.search(text) is None, path.name
        assert AUTHORITY_TOKEN_RE.search(text) is None, path.name


def _assert_boolean_column(frame: pd.DataFrame, column: str) -> None:
    assert column in frame.columns
    assert frame[column].astype(bool).all()


def test_governed_diagnostic_chain_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ticker = "P8A"
    output_dir = tmp_path / "governed_chain_output"
    prepared_dir = tmp_path / "prepared_features"
    fixture_provenance = _write_feature_fixture(prepared_dir, ticker)
    _patch_feature_fixture_evaluator(
        monkeypatch,
        prepared_dir,
        source_label=str(fixture_provenance["source"]),
    )
    monkeypatch.setitem(
        runner.PRESET_CONFIGS,
        "phase8_fixture",
        {
            "group_names": ["phase8_fixture"],
            "horizons": [5],
            "target_names": ["forward_return"],
            "evaluation_config": {
                "train_size": 45,
                "test_size": 8,
                "step_size": 8,
                "gap_size": 0,
                "max_windows": 1,
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_quant_core.py",
            "--preset",
            "phase8_fixture",
            "--run-mode",
            "research_core",
            "--tickers",
            ticker,
            "--horizons",
            "5",
            "--target-types",
            "forward_return",
            "--models",
            "random_forest",
            "--no-ensemble",
            "--output-dir",
            str(output_dir),
            "--enable-scenario-engine",
            "--scenario-calibration-lookback",
            "0",
            "--enable-risk-governance",
            "--enable-portfolio-allocator",
            "--enable-phase3-router",
        ],
    )
    monkeypatch.setattr(
        runner,
        "collect_git_metadata",
        lambda path: {"branch": "test", "commit_hash": "phase8", "is_dirty": False},
    )
    monkeypatch.setattr(
        runner,
        "collect_runtime_metadata",
        lambda: {"python_executable": sys.executable, "runtime_mode": RuntimeMode.RESEARCH.value},
    )
    monkeypatch.setattr(
        runner,
        "collect_dependency_versions",
        lambda packages: {str(package): "test" for package in packages},
    )

    assert runner.main() == 0

    forecast_summary = pd.read_csv(output_dir / "forecast_summary.csv")
    risk_summary = pd.read_csv(output_dir / "risk_summary.csv")
    allocation = pd.read_csv(output_dir / "portfolio_allocation.csv")
    portfolio_summary = pd.read_csv(output_dir / "portfolio_summary.csv")
    router_decisions = pd.read_csv(output_dir / "router_decisions.csv")
    predictions = pd.read_csv(output_dir / "full_model_predictions.csv")
    run_manifest = _load_json(output_dir / "run_manifest.json")
    risk_manifest = _load_json(output_dir / "risk_manifest.json")
    allocator_manifest = _load_json(output_dir / "allocator_manifest.json")
    router_manifest = _load_json(output_dir / "router_manifest.json")

    assert not forecast_summary.empty
    assert not risk_summary.empty
    assert not portfolio_summary.empty
    if allocation.empty:
        assert portfolio_summary.loc[0, "portfolio_status"] == "all_cash"
        assert set(router_decisions["route_decision"]) == {"no_candidate"}
    else:
        assert set(allocation["allocation_status"]) <= {"allocation_candidate", "no_allocation"}
    assert not router_decisions.empty

    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "risk_manifest.json").exists()
    assert (output_dir / "allocator_manifest.json").exists()
    assert (output_dir / "router_manifest.json").exists()
    assert risk_manifest["manifest_type"] == "risk_governance_layer_v1_manifest"
    assert allocator_manifest["manifest_type"] == "portfolio_allocator_v1_manifest"
    assert router_manifest["manifest_type"] == "phase3_router_v1_manifest"

    assert run_manifest["run_mode"] == "research_core"
    assert run_manifest["runtime"]["runtime_mode"] == RuntimeMode.RESEARCH.value
    assert forecast_summary["run_mode"].eq("research_core").all()
    assert risk_summary["run_mode"].eq("research_core").all()
    if not allocation.empty:
        assert allocation["run_mode"].astype(str).replace({"nan": ""}).isin(["research_core", ""]).all()
    assert set(predictions["source"]) == {fixture_provenance["source"]}
    assert fixture_provenance["uses_mock_data"] is False
    assert fixture_provenance["fallback_triggered"] is False
    assert fixture_provenance["runtime_mode"] == RuntimeMode.RESEARCH.value
    with pytest.raises(RuntimeError, match="Mock data disabled for audit mode"):
        ensure_mock_allowed(RuntimeMode.AUDIT, explicit_mock=True)

    _assert_boolean_column(portfolio_summary, "diagnostic_only_authority")
    _assert_boolean_column(portfolio_summary, "no_buy_sell_recommendation_authority")
    assert allocator_manifest["diagnostic_only_authority"] is True
    assert allocator_manifest["no_buy_sell_recommendation_authority"] is True
    _assert_boolean_column(router_decisions, "diagnostic_only_authority")
    _assert_boolean_column(router_decisions, "no_buy_sell_recommendation_authority")

    assert set(router_decisions["route_decision"]) <= set(ROUTE_DECISIONS)
    assert router_manifest["route_decisions"] == list(ROUTE_DECISIONS)
    assert set(router_manifest["route_decision_counts"]) == set(ROUTE_DECISIONS)

    artifact_paths = _artifact_paths(run_manifest, output_dir)
    _assert_no_forbidden_authority_terms(artifact_paths)
