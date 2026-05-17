"""A/B benchmark framework for legacy vs risk/regime-aware ML system modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.benchmark.acceptance import evaluate_benchmark_acceptance
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.portfolio.allocation import RiskAwareAllocator
from src.ml.regime.regime_detector import REGIME_TO_CODE
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS, SEQUENCE_ALGORITHMS


@dataclass(frozen=True)
class BenchmarkModeSpec:
    name: str
    description: str
    risk_config: dict[str, Any]


def default_benchmark_modes() -> list[BenchmarkModeSpec]:
    return [
        BenchmarkModeSpec(
            name="legacy_forecast_only",
            description="Forecasting only, no risk/regime/allocation extensions.",
            risk_config={
                "risk_enabled": False,
                "enable_covar": False,
                "enable_risk_engine": False,
                "enable_regime_detection": False,
                "enable_regime_switching": False,
                "enable_risk_allocation": False,
            },
        ),
        BenchmarkModeSpec(
            name="forecast_plus_risk_features",
            description="Forecasting with rolling VaR/CVaR/CoVaR/Drawdown features.",
            risk_config={
                "risk_enabled": False,
                "enable_covar": True,
                "enable_risk_engine": True,
                "enable_regime_detection": False,
                "enable_regime_switching": False,
                "enable_risk_allocation": False,
            },
        ),
        BenchmarkModeSpec(
            name="forecast_plus_risk_and_regime",
            description="Forecasting with risk features and regime-aware inputs.",
            risk_config={
                "risk_enabled": False,
                "enable_covar": True,
                "enable_risk_engine": True,
                "enable_regime_detection": True,
                "enable_regime_switching": True,
                "enable_risk_allocation": False,
            },
        ),
        BenchmarkModeSpec(
            name="full_system",
            description="Forecasting with risk, regime, and allocation overlays.",
            risk_config={
                "risk_enabled": True,
                "enable_covar": True,
                "enable_risk_engine": True,
                "enable_regime_detection": True,
                "enable_regime_switching": True,
                "enable_risk_allocation": True,
            },
        ),
    ]


class SystemBenchmarkRunner:
    """Benchmark runner that preserves identical training/test splits across modes."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        evaluator: MetricsEvaluator | None = None,
        fee: float = 0.0015,
        slippage: float = 0.002,
    ) -> None:
        self.model_root = Path(model_root)
        self.evaluator = evaluator or MetricsEvaluator()
        self.eval_config = {"fee": fee, "slippage": slippage}

    @staticmethod
    def _reverse_regime_map() -> dict[int, str]:
        return {value: key for key, value in REGIME_TO_CODE.items()}

    def _build_signal(
        self,
        mode: BenchmarkModeSpec,
        ticker: str,
        predicted_direction: np.ndarray,
        test_feature_frame: pd.DataFrame,
    ) -> np.ndarray:
        signal = np.asarray(predicted_direction, dtype=float)
        if mode.name != "full_system":
            return signal

        allocator = RiskAwareAllocator(
            risk_penalty_strength=float(mode.risk_config.get("risk_penalty_strength", 1.0)),
            high_vol_exposure_cut=float(mode.risk_config.get("high_vol_exposure_cut", 0.6)),
            crisis_exposure_cut=float(mode.risk_config.get("crisis_exposure_cut", 0.25)),
        )
        reverse_regime = self._reverse_regime_map()
        scaled_signal = np.zeros(len(signal), dtype=float)
        for idx, row in test_feature_frame.reset_index(drop=True).iterrows():
            if signal[idx] <= 0:
                continue
            risk_frame = pd.DataFrame(
                [
                    {
                        "var_q": float(pd.to_numeric(row.get("var_q"), errors="coerce") if pd.notna(row.get("var_q")) else 0.0),
                        "delta_covar": float(pd.to_numeric(row.get("delta_covar"), errors="coerce") if pd.notna(row.get("delta_covar")) else 0.0),
                        "rolling_drawdown": float(pd.to_numeric(row.get("rolling_drawdown"), errors="coerce") if pd.notna(row.get("rolling_drawdown")) else 0.0),
                    }
                ],
                index=[ticker],
            )
            regime_labels = None
            regime_value = row.get("regime_label")
            if pd.notna(regime_value):
                regime_labels = pd.Series({ticker: reverse_regime.get(int(regime_value), "NORMAL")})
            allocation = allocator.allocate(
                risk_frame=risk_frame,
                regime_labels=regime_labels,
                base_weights=pd.Series({ticker: 1.0}),
            )
            scaled_signal[idx] = float(allocation.weights.get(ticker, 0.0))
        return scaled_signal

    def _evaluate_trained_models(
        self,
        *,
        trainer: DualModelTrainer,
        ticker: str,
        df: pd.DataFrame,
        mode: BenchmarkModeSpec,
        sequence_length: int,
    ) -> list[dict[str, Any]]:
        prepared = trainer.prepare_ticker_data(
            ticker=ticker,
            df=df,
            max_sequence_length=sequence_length,
            risk_config=mode.risk_config,
        )
        labeled = trainer._add_targets(prepared.feature_frame)
        manifest = trainer._manifests[ticker]
        rows: list[dict[str, Any]] = []

        for horizon, horizon_info in manifest.get("horizons", {}).items():
            for algorithm, algorithm_info in horizon_info.get("algorithms", {}).items():
                feature_columns = algorithm_info.get("feature_columns", manifest.get("feature_columns", []))
                feature_columns_by_task = algorithm_info.get("feature_columns_by_task", {})
                trend_feature_columns = feature_columns_by_task.get("trend", feature_columns)
                return_feature_columns = feature_columns_by_task.get("return", feature_columns)
                algo_sequence_length = int(algorithm_info.get("sequence_length") or sequence_length)
                trend_problem = trainer._build_horizon_problem(
                    labeled,
                    trend_feature_columns,
                    horizon,
                    algo_sequence_length,
                )
                return_problem = trainer._build_horizon_problem(
                    labeled,
                    return_feature_columns,
                    horizon,
                    algo_sequence_length,
                )
                if trend_problem is None or return_problem is None:
                    continue

                use_sequence = algorithm in SEQUENCE_ALGORITHMS
                trend_inputs = trend_problem["sequence" if use_sequence else "tabular"]
                return_inputs = return_problem["sequence" if use_sequence else "tabular"]
                x_test = trend_inputs["X_test"]
                return_x_test = return_inputs["X_test"]
                if len(x_test) == 0 or len(return_x_test) == 0:
                    continue

                trend_model = trainer._get_loaded_model(ticker, algorithm, horizon, "trend")
                return_model = trainer._get_loaded_model(ticker, algorithm, horizon, "return")
                predicted_direction = np.asarray(trend_model.predict(x_test), dtype=int)
                predicted_returns = np.asarray(return_model.predict(return_x_test), dtype=float).reshape(-1)
                realized_returns = np.asarray(return_inputs["y_test_return"], dtype=float)
                realized_direction = np.asarray(trend_inputs["y_test_direction"], dtype=int)
                test_feature_frame = trend_inputs["test_feature_frame"].reset_index(drop=True)

                signal = self._build_signal(mode, ticker, predicted_direction, test_feature_frame)
                strategy_eval = DualModelTrainer.evaluate_strategy_for_horizon(
                    signal,
                    realized_returns,
                    HORIZON_DAYS[horizon],
                    evaluator=self.evaluator,
                    config=self.eval_config,
                )
                prediction_eval = self.evaluator.evaluate_prediction_quality(
                    predicted_returns,
                    realized_returns,
                    predicted_direction,
                    realized_direction,
                )

                rows.append(
                    {
                        "benchmark_mode": mode.name,
                        "ticker": ticker,
                        "horizon": horizon,
                        "algorithm": algorithm,
                        "mode_description": mode.description,
                        "train_rows": int(len(trend_inputs["X_train"])),
                        "val_rows": int(len(trend_inputs["X_val"])),
                        "test_rows": int(len(trend_inputs["X_test"])),
                        "cumulative_return": strategy_eval["metrics"]["cumulative_return"],
                        "cagr": strategy_eval["metrics"]["cagr"],
                        "volatility": strategy_eval["metrics"]["volatility"],
                        "sharpe": strategy_eval["metrics"]["sharpe"],
                        "sortino": strategy_eval["metrics"]["sortino"],
                        "calmar": strategy_eval["metrics"]["calmar"],
                        "max_drawdown": strategy_eval["metrics"]["max_drawdown"],
                        "avg_drawdown": strategy_eval["metrics"]["avg_drawdown"],
                        "tail_loss": strategy_eval["metrics"]["tail_loss"],
                        "turnover": strategy_eval["trade_stats"]["turnover"],
                        "exposure": strategy_eval["trade_stats"]["exposure"],
                        "trade_count": strategy_eval["trade_stats"]["trade_count"],
                        **prediction_eval,
                    }
                )

        return rows

    @staticmethod
    def _summary(detail_df: pd.DataFrame) -> pd.DataFrame:
        if detail_df.empty:
            return pd.DataFrame()
        summary = (
            detail_df.groupby("benchmark_mode", dropna=False)[
                [
                    "cumulative_return",
                    "cagr",
                    "volatility",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "max_drawdown",
                    "avg_drawdown",
                    "tail_loss",
                    "turnover",
                    "exposure",
                    "trade_count",
                    "rmse",
                    "mae",
                    "directional_accuracy",
                    "test_rows",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        if "legacy_forecast_only" in summary["benchmark_mode"].values:
            legacy = summary.loc[summary["benchmark_mode"] == "legacy_forecast_only"].iloc[0]
            summary["delta_sharpe_vs_legacy"] = summary["sharpe"] - float(legacy["sharpe"])
            summary["delta_cagr_vs_legacy"] = summary["cagr"] - float(legacy["cagr"])
            summary["delta_mdd_vs_legacy"] = summary["max_drawdown"] - float(legacy["max_drawdown"])
        return SystemBenchmarkRunner._append_acceptance_status(summary)

    @staticmethod
    def _append_acceptance_status(summary_df: pd.DataFrame) -> pd.DataFrame:
        if summary_df.empty:
            return summary_df
        result = summary_df.copy().reset_index(drop=True)
        legacy_rows = result[result["benchmark_mode"] == "legacy_forecast_only"]
        legacy = legacy_rows.iloc[0] if not legacy_rows.empty else None
        comparison_count = max(int(len(result) - (0 if legacy is None else 1)), 0)
        acceptance_rows: list[dict[str, Any]] = []
        for _, row in result.iterrows():
            benchmark_mode = str(row.get("benchmark_mode") or "")
            if legacy is None or benchmark_mode == "legacy_forecast_only":
                acceptance = evaluate_benchmark_acceptance(
                    prediction_metric_delta=None,
                    economic_metric_delta=None,
                    bootstrap_ci=None,
                    dm_p_value=None,
                    turnover_delta=None,
                    cost_adjusted_delta=None,
                    sample_size=_safe_int(row.get("test_rows")),
                    comparison_count=comparison_count,
                )
            else:
                prediction_delta = _safe_float(row.get("directional_accuracy")) - _safe_float(legacy.get("directional_accuracy"))
                economic_delta = _safe_float(row.get("delta_sharpe_vs_legacy"))
                cost_adjusted_delta = _safe_float(row.get("delta_cagr_vs_legacy"))
                turnover_delta = _safe_float(row.get("turnover")) - _safe_float(legacy.get("turnover"))
                acceptance = evaluate_benchmark_acceptance(
                    prediction_metric_delta=prediction_delta,
                    economic_metric_delta=economic_delta,
                    bootstrap_ci=None,
                    dm_p_value=None,
                    turnover_delta=turnover_delta,
                    cost_adjusted_delta=cost_adjusted_delta,
                    sample_size=_safe_int(row.get("test_rows")),
                    comparison_count=comparison_count,
                )
            payload = acceptance.to_dict()
            acceptance_rows.append(
                {
                    "accepted": payload["accepted"],
                    "status": payload["status"],
                    "effect_size": payload["effect_size"],
                    "bootstrap_ci": payload["bootstrap_ci"],
                    "dm_p_value": payload["dm_p_value"],
                    "warnings": payload["warnings"],
                    "economic_metric_delta": payload["economic_metric_delta"],
                    "turnover_penalty": payload["turnover_penalty"],
                    "cost_adjusted_delta": payload["cost_adjusted_delta"],
                    "sample_size": payload["sample_size"],
                    "comparison_count": payload["comparison_count"],
                    "policy_version": payload["policy_version"],
                    "decision_reasons": payload["decision_reasons"],
                    "acceptance_interpretation": payload["interpretation"],
                }
            )
        return pd.concat([result, pd.DataFrame(acceptance_rows)], axis=1)

    @staticmethod
    def _markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows generated._"
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    def _write_outputs(
        self,
        *,
        detail_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        detail_path: Path,
        summary_path: Path,
        json_path: Path,
        markdown_path: Path,
    ) -> None:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        json_payload = {
            "detail_rows": detail_df.to_dict(orient="records"),
            "summary_rows": summary_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        best_rows = (
            detail_df.sort_values(["sharpe", "calmar", "directional_accuracy"], ascending=False)
            .groupby("benchmark_mode", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        acceptance_columns = [
            "benchmark_mode",
            "accepted",
            "status",
            "effect_size",
            "bootstrap_ci",
            "dm_p_value",
            "warnings",
            "acceptance_interpretation",
        ]
        acceptance_df = (
            summary_df[[column for column in acceptance_columns if column in summary_df.columns]]
            if not summary_df.empty
            else pd.DataFrame()
        )
        markdown = "\n".join(
            [
                "# System Benchmark",
                "",
                "## Summary",
                self._markdown_table(summary_df.round(6)),
                "",
                "## Acceptance Governance",
                "Leaderboard position alone is not benchmark promotion. Rows without bootstrap CI and DM evidence remain `exploratory_only` or lower.",
                self._markdown_table(acceptance_df),
                "",
                "## Best Rows By Mode (Exploratory Ranking)",
                self._markdown_table(best_rows.round(6)),
                "",
                "## Output Files",
                f"- Detail CSV: `{detail_path}`",
                f"- Summary CSV: `{summary_path}`",
                f"- JSON: `{json_path}`",
            ]
        )
        markdown_path.write_text(markdown, encoding="utf-8")

    def run(
        self,
        *,
        files: list[Path],
        algorithms: list[str],
        output_root: str | Path,
        report_path: str | Path,
        primary_algorithm: str | None = None,
        horizons: list[str] | None = None,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 30,
        patience: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str | None = None,
    ) -> dict[str, Any]:
        output_root = Path(output_root)
        report_path = Path(report_path)
        detail_rows: list[dict[str, Any]] = []

        for mode in default_benchmark_modes():
            mode_model_root = output_root / mode.name
            trainer = DualModelTrainer(model_dir=mode_model_root)
            for csv_path in files:
                ticker = csv_path.stem.upper()
                df = pd.read_csv(csv_path)
                trainer.train(
                    ticker=ticker,
                    df=df,
                    algorithms=algorithms,
                    primary_algorithm=primary_algorithm,
                    horizons=horizons,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    risk_config=mode.risk_config,
                )
                detail_rows.extend(
                    self._evaluate_trained_models(
                        trainer=trainer,
                        ticker=ticker,
                        df=df,
                        mode=mode,
                        sequence_length=sequence_length,
                    )
                )

        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            detail_df = detail_df.sort_values(
                ["benchmark_mode", "ticker", "horizon", "algorithm"]
            ).reset_index(drop=True)
        summary_df = self._summary(detail_df)

        detail_csv = report_path if report_path.suffix.lower() == ".csv" else report_path.with_suffix(".csv")
        summary_csv = detail_csv.with_name(f"{detail_csv.stem}_summary.csv")
        json_path = detail_csv.with_suffix(".json")
        markdown_path = detail_csv.parent / "system_benchmark.md"
        self._write_outputs(
            detail_df=detail_df,
            summary_df=summary_df,
            detail_path=detail_csv,
            summary_path=summary_csv,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        return {
            "detail": detail_df,
            "summary": summary_df,
            "detail_path": detail_csv,
            "summary_path": summary_csv,
            "json_path": json_path,
            "markdown_path": markdown_path,
        }


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def _safe_int(value: Any) -> int | None:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None
