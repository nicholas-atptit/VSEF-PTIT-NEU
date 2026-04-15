from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.backtest.regime_aware_analysis import RegimeAwareAnalysisConfig, RegimeAwareAnalysisRunner


def _market_proxy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2025-12-22",
                "2025-12-23",
                "2025-12-24",
                "2025-12-25",
                "2025-12-26",
                "2025-12-29",
                "2025-12-30",
                "2025-12-31",
            ],
            "m_ret": [0.0, 0.01, 0.01, 0.02, 0.02, -0.06, 0.0, 0.01],
        }
    )


def _build_artifacts(tmp_path: Path, horizon: str) -> None:
    dual_task_regression = tmp_path / "artifacts" / "dual_task" / "regression" / horizon
    dual_task_classification = tmp_path / "artifacts" / "dual_task" / "classification" / horizon
    combined_signal = tmp_path / "artifacts" / "combined_signal" / horizon
    dual_task_regression.mkdir(parents=True, exist_ok=True)
    dual_task_classification.mkdir(parents=True, exist_ok=True)
    combined_signal.mkdir(parents=True, exist_ok=True)

    rows = [
        ("2025-12-31", "2025-12-26", "AAA", 0.04, 0.05, 1, 1, 0.70, "strong_positive"),
        ("2025-12-31", "2025-12-26", "BBB", 0.01, 0.02, 1, 1, 0.55, "moderate_positive"),
        ("2025-12-31", "2025-12-26", "CCC", -0.02, -0.01, 0, 0, 0.40, "reject"),
        ("2026-01-02", "2025-12-30", "AAA", -0.04, -0.03, 0, 0, 0.30, "reject"),
        ("2026-01-02", "2025-12-30", "BBB", -0.01, 0.01, 0, 1, 0.60, "weak_or_uncertain"),
        ("2026-01-02", "2025-12-30", "CCC", 0.03, 0.02, 1, 1, 0.65, "strong_positive"),
        ("2026-01-05", "2025-12-31", "AAA", 0.005, 0.01, 1, 1, 0.55, "moderate_positive"),
        ("2026-01-05", "2025-12-31", "BBB", -0.002, -0.005, 0, 0, 0.45, "reject"),
        ("2026-01-05", "2025-12-31", "CCC", 0.04, 0.03, 1, 1, 0.72, "strong_positive"),
    ]

    regression_rows = []
    classification_rows = []
    combined_rows = []
    for target_date, prediction_date, ticker, actual_return, predicted_return, actual_label, predicted_label, prob, signal_label in rows:
        target_ts = pd.Timestamp(target_date)
        prediction_ts = pd.Timestamp(prediction_date)
        entry_ts = prediction_ts + pd.tseries.offsets.BDay(1)
        if predicted_return <= -0.005:
            norm_return = 0.0
        elif predicted_return <= 0.015:
            norm_return = 0.5
        else:
            norm_return = 1.0
        regression_rows.append(
            {
                "date": target_date,
                "ticker": ticker,
                "model_name": "cart",
                "prediction_date": prediction_date,
                "target_date": target_date,
                "horizon": horizon,
                "horizon_days": int(horizon.replace("d", "")),
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "absolute_error": abs(predicted_return - actual_return),
                "pct_error": abs(predicted_return - actual_return) / max(abs(actual_return), 1e-9),
            }
        )
        classification_rows.append(
            {
                "date": target_date,
                "ticker": ticker,
                "model_name": "cart",
                "prediction_date": prediction_date,
                "entry_date": str(entry_ts.date()),
                "target_date": target_date,
                "horizon": horizon,
                "horizon_days": int(horizon.replace("d", "")),
                "actual_net_trade_return": actual_return - 0.01,
                "actual_profit_label": actual_label,
                "predicted_profit_label": predicted_label,
                "predicted_profit_probability": prob,
            }
        )
        combined_rows.append(
            {
                "date": target_date,
                "ticker": ticker,
                "horizon": horizon,
                "model_name": "cart",
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "actual_profit_label": actual_label,
                "predicted_profit_label": predicted_label,
                "predicted_profit_probability": prob,
                "ranking_group": target_date,
                "normalized_predicted_return": norm_return,
                "return_strength": norm_return,
                "profit_confidence": prob,
                "combined_score": 0.5 * norm_return + 0.5 * prob,
                "gated_valid_signal": bool(predicted_return > 0 and prob > 0.5),
                "return_only_signal_label": signal_label if predicted_return > 0 else "reject",
                "probability_only_signal_label": signal_label if prob > 0.5 else "reject",
                "combined_signal_label": signal_label,
                "return_rank_score": norm_return,
                "profit_probability_rank_score": prob,
                "rank_based_joint_score": 0.5 * norm_return + 0.5 * prob,
            }
        )

    pd.DataFrame(regression_rows).to_csv(dual_task_regression / "predicted_vs_actual.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(dual_task_classification / "predicted_vs_actual.csv", index=False)
    pd.DataFrame(combined_rows).to_csv(combined_signal / "combined_signal_table.csv", index=False)


def test_regime_assignment_uses_prediction_date_without_leakage(tmp_path) -> None:
    _build_artifacts(tmp_path, "3d")
    market_proxy_path = tmp_path / "market_proxy.csv"
    _market_proxy_frame().to_csv(market_proxy_path, index=False)

    runner = RegimeAwareAnalysisRunner(
        RegimeAwareAnalysisConfig(
            dual_task_dir=str(tmp_path / "artifacts" / "dual_task"),
            combined_signal_dir=str(tmp_path / "artifacts" / "combined_signal"),
            output_dir=str(tmp_path / "artifacts" / "regime"),
            horizons=["3d"],
            benchmark_source="market_proxy",
            benchmark_path=str(market_proxy_path),
            regime_lookback_days=2,
            bull_threshold=0.03,
            bear_threshold=-0.03,
        )
    )

    result = runner.run()
    labeled = result["horizons"]["3d"]["regime_labeled_signal_table"]
    regimes = labeled.groupby("prediction_date")["regime"].first().to_dict()

    assert regimes[pd.Timestamp("2025-12-26")] == "bull"
    assert regimes[pd.Timestamp("2025-12-30")] == "bear"
    assert regimes[pd.Timestamp("2025-12-31")] == "sideway"
    assert not (labeled["benchmark_date"] > labeled["prediction_date"]).any()


def test_regime_runner_writes_outputs_and_summary_tables(tmp_path) -> None:
    _build_artifacts(tmp_path, "3d")
    _build_artifacts(tmp_path, "5d")
    market_proxy_path = tmp_path / "market_proxy.csv"
    _market_proxy_frame().to_csv(market_proxy_path, index=False)

    runner = RegimeAwareAnalysisRunner(
        RegimeAwareAnalysisConfig(
            dual_task_dir=str(tmp_path / "artifacts" / "dual_task"),
            combined_signal_dir=str(tmp_path / "artifacts" / "combined_signal"),
            output_dir=str(tmp_path / "artifacts" / "regime"),
            horizons=["3d", "5d"],
            benchmark_source="market_proxy",
            benchmark_path=str(market_proxy_path),
            regime_lookback_days=2,
            bull_threshold=0.03,
            bear_threshold=-0.03,
        )
    )
    result = runner.run()

    ranking_path = Path(result["horizons"]["3d"]["paths"]["ranking_by_regime"])
    calibration_path = Path(result["horizons"]["3d"]["paths"]["calibration_by_regime"])
    summary_path = Path(result["summary_paths"]["overall_regime_summary"])
    method_ranking_path = Path(result["summary_paths"]["regime_combined_method_ranking"])
    assert ranking_path.exists()
    assert calibration_path.exists()
    assert summary_path.exists()
    assert method_ranking_path.exists()

    ranking_df = pd.read_csv(ranking_path)
    calibration_df = pd.read_csv(calibration_path)
    overall_summary = pd.read_csv(summary_path)

    assert {"regime", "ranking_method", "top_k", "average_actual_return", "profit_rate"} <= set(ranking_df.columns)
    assert {"regime", "probability_bucket", "realized_profit_rate", "calibration_gap"} <= set(calibration_df.columns)
    assert {"regime", "best_regression_model", "best_classification_model", "best_combined_method", "best_horizon"} <= set(overall_summary.columns)

    run_config = json.loads((ranking_path.parent / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["benchmark_alignment_safe"] is True
    assert run_config["analysis_only"] is True
