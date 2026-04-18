from __future__ import annotations

import pandas as pd

from src.ml.data_loader import generate_mock_data
from src.ml.feature_engineering import FeatureEngineer
from src.ml.features.registry import (
    approved_feature_sets,
    feature_build_modes,
    feature_selection_evidence,
    final_task_feature_sets,
    load_feature_registry,
    price_reference_semantics,
    resolve_task_feature_set,
    sentiment_policy,
    validate_feature_registry_against_columns,
)


def _build_registry_probe_frame() -> pd.DataFrame:
    base = generate_mock_data(ticker="TEST", num_days=320, seed=7)
    base["m_ret"] = pd.Series(base["close"]).pct_change().fillna(0.0) * 0.8
    base["m_ret_5d"] = pd.Series(base["m_ret"]).rolling(5).mean().fillna(0.0)
    base["m_ret_20d"] = pd.Series(base["m_ret"]).rolling(20).mean().fillna(0.0)
    base["s_ret"] = pd.Series(base["close"]).pct_change().fillna(0.0) * 0.6
    base["s_ret_5d"] = pd.Series(base["s_ret"]).rolling(5).mean().fillna(0.0)
    base["rel_to_market"] = pd.Series(base["close"]).pct_change().fillna(0.0) - base["m_ret"]
    base["rel_to_sector"] = pd.Series(base["close"]).pct_change().fillna(0.0) - base["s_ret"]
    base["advancers"] = 120.0
    base["decliners"] = 80.0
    base["unchanged"] = 10.0
    base["net_advancers"] = 40.0
    base["market_breadth"] = 0.2
    base["advance_decline_ratio"] = 1.1
    base["advancing_share"] = 0.6
    base["pct_above_ma20"] = 0.55
    base["pct_above_ma50"] = 0.50
    base["new_highs_252"] = 15.0
    base["new_lows_252"] = 4.0
    base["new_high_low_spread"] = 0.11
    base["up_volume"] = 1_500_000.0
    base["down_volume"] = 900_000.0
    base["sector_dispersion"] = 0.02
    base["foreign_net_volume"] = pd.Series(base["volume"]).mul(0.05)
    base["foreign_net_value"] = pd.Series(base["close"] * base["volume"]).mul(0.03)
    base["fx_usdvnd"] = 24_000 + pd.Series(range(len(base))) * 2
    base["interest_rate"] = 4.0 + pd.Series(range(len(base))) * 0.0005
    base["gold_price"] = 2_000 + pd.Series(range(len(base))) * 0.3
    base["oil_price"] = 80 + pd.Series(range(len(base))) * 0.02
    return base


class TestFeatureRegistry:
    def test_registry_loads(self):
        registry = load_feature_registry()
        assert registry["registry_version"] >= 1
        assert registry["features"]

    def test_registry_covers_current_feature_universe(self):
        engineer = FeatureEngineer()
        sentiment = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=320, freq="B"),
                "sentiment_avg": [0.1] * 320,
                "news_volume": [1.0] * 320,
                "sentiment_std": [0.0] * 320,
            }
        )
        feature_frame = engineer.build_feature_frame(
            _build_registry_probe_frame(),
            sentiment_df=sentiment,
        )
        columns = engineer.get_registry_candidate_columns(feature_frame) + [
            "var_q",
            "cvar_q",
            "covar_q",
            "delta_covar",
            "rolling_drawdown",
            "regime_label",
            "regime_probability",
        ]

        validation = validate_feature_registry_against_columns(columns)

        assert validation["missing_from_registry"] == []
        assert validation["invalid_approved_features"] == {}
        assert validation["invalid_final_task_features"] == {}

    def test_approved_feature_sets_only_reference_active_features(self):
        registry = load_feature_registry()
        lookup = {entry["feature_name"]: entry for entry in registry["features"]}

        for set_name, feature_names in approved_feature_sets().items():
            assert feature_names, f"{set_name} should not be empty"
            for feature_name in feature_names:
                assert feature_name in lookup
                assert lookup[feature_name]["status"] == "active"

    def test_registry_declares_adjusted_close_limitations(self):
        semantics = price_reference_semantics()
        assert semantics["model_close_reference_column"] == "model_close_reference"
        assert semantics["raw_close_column"] == "raw_close"
        assert semantics["deprecated_raw_close_alias"] == "close_raw"
        assert semantics["adjusted_close_available_from_live_vnstock_data"] is False

    def test_registry_includes_final_task_feature_sets(self):
        registry = load_feature_registry()
        lookup = {entry["feature_name"]: entry for entry in registry["features"]}

        task_sets = final_task_feature_sets()
        assert set(task_sets) >= {
            "regression_forecasting",
            "directional_classification",
            "regime_detection",
            "risk_layer",
        }
        for task_name, feature_names in task_sets.items():
            assert feature_names, f"{task_name} should not be empty"
            for feature_name in feature_names:
                assert feature_name in lookup

    def test_registry_includes_feature_selection_evidence(self):
        evidence = feature_selection_evidence()
        assert "selection_scope" in evidence
        assert evidence["selection_scope"] == "walk_forward_train_only"

    def test_registry_declares_sentiment_policy_and_excludes_it_from_approved_defaults(self):
        policy = sentiment_policy()
        approved = approved_feature_sets()

        assert policy["enabled_by_default"] is False
        assert policy["requires_validated_source"] is True
        assert policy["approved_for_main_pipeline"] is False

        approved_default_columns = set(approved["forecast_core_features"]) | set(approved["classification_signal_features"])
        assert "sentiment_avg" not in approved_default_columns
        assert "news_volume" not in approved_default_columns

    def test_resolve_task_feature_set_filters_to_available_columns(self):
        available = ["sma_20", "rsi_14", "rolling_volatility_20", "market_regime_code"]
        resolved = resolve_task_feature_set(
            "regression_forecasting",
            available_columns=available,
        )
        assert set(resolved) <= set(available)

    def test_registry_includes_regime_and_corporate_action_notes(self):
        registry = load_feature_registry()
        assert "regime_definitions" in registry
        assert "corporate_action_diagnostics" in registry
        assert "market_regime" in registry["regime_definitions"]

    def test_registry_declares_feature_build_modes(self):
        modes = feature_build_modes()
        assert modes["default_mode"] == "full_research_mode"
        assert set(modes["modes"]) >= {
            "fast_core_mode",
            "full_research_mode",
            "regime_risk_mode",
        }

    def test_inventory_marks_sentiment_features_as_experimental(self):
        engineer = FeatureEngineer()
        sentiment = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=320, freq="B"),
                "sentiment_avg": [0.1] * 320,
                "news_volume": [1.0] * 320,
                "sentiment_std": [0.0] * 320,
            }
        )
        feature_frame = engineer.build_feature_frame(
            _build_registry_probe_frame(),
            sentiment_df=sentiment,
        )
        inventory = engineer.build_feature_inventory(feature_frame)
        sentiment_rows = inventory[inventory["feature_name"].isin(["sentiment_avg", "news_volume", "sentiment_std"])]

        assert not sentiment_rows.empty
        assert set(sentiment_rows["status"]) == {"experimental"}
