"""Deterministic crisis stress testing for benchmarked ML trading modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.benchmark.acceptance import evaluate_benchmark_acceptance
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.benchmark.system_benchmark import BenchmarkModeSpec, SystemBenchmarkRunner, default_benchmark_modes
from src.ml.regime.regime_detector import REGIME_TO_CODE, RegimeDetector
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS, SEQUENCE_ALGORITHMS


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str


def default_stress_scenarios() -> list[StressScenario]:
    return [
        StressScenario("volatility_shock", "Amplify realized volatility and risk metrics after the midpoint."),
        StressScenario("drawdown_shock", "Inject a concentrated negative block loss and deeper drawdown."),
        StressScenario("liquidity_cost_shock", "Increase fees and slippage to simulate thinner liquidity."),
        StressScenario("regime_persistence_shock", "Force crisis-like regime persistence after the first shock."),
    ]


class StressTestRunner:
    """Run deterministic stress scenarios over identical held-out benchmark splits."""

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
        self.base_eval_config = {"fee": fee, "slippage": slippage}
        self.signal_builder = SystemBenchmarkRunner(model_root=model_root, evaluator=self.evaluator, fee=fee, slippage=slippage)

    @staticmethod
    def _recompute_regime(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        detector = RegimeDetector()
        result = detector.detect_from_frame(frame)
        updated = frame.copy()
        updated["regime_label"] = result.encoded_labels.reindex(updated.index)
        updated["regime_probability"] = result.probabilities.max(axis=1).reindex(updated.index)
        return updated, result.labels.reindex(updated.index)

    def _apply_scenario(
        self,
        *,
        scenario: StressScenario,
        feature_frame: pd.DataFrame,
        realized_returns: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
        stressed_frame = feature_frame.copy().reset_index(drop=True)
        stressed_returns = np.asarray(realized_returns, dtype=float).copy()
        n_rows = len(stressed_returns)
        shock_start = n_rows // 2
        metadata: dict[str, Any] = {
            "shock_start": int(shock_start),
            "fee_multiplier": 1.0,
            "slippage_multiplier": 1.0,
        }

        def _scale_column(column: str, factor: float, preserve_negative: bool = False) -> None:
            if column not in stressed_frame.columns:
                return
            values = pd.to_numeric(stressed_frame[column], errors="coerce")
            if preserve_negative:
                stressed_frame[column] = np.where(values < 0, values * factor, values)
            else:
                stressed_frame[column] = values * factor

        if scenario.name == "volatility_shock":
            stressed_returns[shock_start:] *= 2.0
            for column in ("var_q", "cvar_q", "covar_q", "delta_covar", "rolling_volatility_20"):
                _scale_column(column, 2.0)
            _scale_column("rolling_drawdown", 1.5, preserve_negative=True)
        elif scenario.name == "drawdown_shock":
            shock_end = min(shock_start + 5, n_rows)
            stressed_returns[shock_start:shock_end] -= 0.08
            stressed_returns = np.clip(stressed_returns, -0.95, None)
            if "rolling_drawdown" in stressed_frame.columns:
                values = pd.to_numeric(stressed_frame["rolling_drawdown"], errors="coerce").fillna(0.0)
                values.iloc[shock_start:] = np.minimum(values.iloc[shock_start:] - 0.1, -0.15)
                stressed_frame["rolling_drawdown"] = values
            _scale_column("delta_covar", 1.75)
            _scale_column("var_q", 1.5, preserve_negative=True)
        elif scenario.name == "liquidity_cost_shock":
            metadata["fee_multiplier"] = 3.0
            metadata["slippage_multiplier"] = 4.0
        elif scenario.name == "regime_persistence_shock":
            stressed_returns[shock_start:] -= 0.01
            stressed_returns = np.clip(stressed_returns, -0.95, None)
            _scale_column("rolling_volatility_20", 1.8)
            _scale_column("delta_covar", 2.0)
            _scale_column("rolling_drawdown", 1.6, preserve_negative=True)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported stress scenario {scenario.name}")

        stressed_frame, stressed_labels = self._recompute_regime(stressed_frame)
        if scenario.name == "regime_persistence_shock" and "regime_label" in stressed_frame.columns:
            reverse = {value: key for key, value in REGIME_TO_CODE.items()}
            horizon = min(shock_start + 10, len(stressed_frame))
            stressed_frame.loc[shock_start:horizon - 1, "regime_label"] = float(REGIME_TO_CODE["CRISIS"])
            stressed_frame.loc[shock_start:horizon - 1, "regime_probability"] = 1.0
            stressed_labels = stressed_frame["regime_label"].map(lambda x: reverse.get(int(x), "NORMAL") if pd.notna(x) else "NORMAL")

        reaction_speed = None
        if len(stressed_labels) > shock_start:
            after = stressed_labels.iloc[shock_start:]
            non_normal = np.where(after.to_numpy(dtype=object) != "NORMAL")[0]
            reaction_speed = None if len(non_normal) == 0 else int(non_normal[0])
        metadata["regime_reaction_speed"] = reaction_speed
        return stressed_frame, stressed_returns, metadata

    def _evaluate_contexts(
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
                if len(x_test) == 0 or len(return_inputs["X_test"]) == 0:
                    continue

                trend_model = trainer._get_loaded_model(ticker, algorithm, horizon, "trend")
                return_model = trainer._get_loaded_model(ticker, algorithm, horizon, "return")
                predicted_direction = np.asarray(trend_model.predict(x_test), dtype=int)
                predicted_returns = np.asarray(return_model.predict(return_inputs["X_test"]), dtype=float).reshape(-1)
                realized_returns = np.asarray(return_inputs["y_test_return"], dtype=float)
                realized_direction = np.asarray(trend_inputs["y_test_direction"], dtype=int)
                base_frame = trend_inputs["test_feature_frame"].reset_index(drop=True)
                baseline_signal = self.signal_builder._build_signal(mode, ticker, predicted_direction, base_frame)
                baseline_eval = DualModelTrainer.evaluate_strategy_for_horizon(
                    baseline_signal,
                    realized_returns,
                    HORIZON_DAYS[horizon],
                    evaluator=self.evaluator,
                    config=self.base_eval_config,
                )

                for scenario in default_stress_scenarios():
                    stressed_frame, stressed_returns, metadata = self._apply_scenario(
                        scenario=scenario,
                        feature_frame=base_frame,
                        realized_returns=realized_returns,
                    )
                    eval_config = {
                        "fee": self.base_eval_config["fee"] * metadata["fee_multiplier"],
                        "slippage": self.base_eval_config["slippage"] * metadata["slippage_multiplier"],
                    }
                    stressed_signal = self.signal_builder._build_signal(
                        mode,
                        ticker,
                        predicted_direction,
                        stressed_frame,
                    )
                    stressed_eval = DualModelTrainer.evaluate_strategy_for_horizon(
                        stressed_signal,
                        stressed_returns,
                        HORIZON_DAYS[horizon],
                        evaluator=self.evaluator,
                        config=eval_config,
                    )
                    rows.append(
                        {
                            "benchmark_mode": mode.name,
                            "stress_scenario": scenario.name,
                            "ticker": ticker,
                            "horizon": horizon,
                            "algorithm": algorithm,
                            "baseline_sharpe": baseline_eval["metrics"]["sharpe"],
                            "stressed_sharpe": stressed_eval["metrics"]["sharpe"],
                            "baseline_max_drawdown": baseline_eval["metrics"]["max_drawdown"],
                            "stressed_max_drawdown": stressed_eval["metrics"]["max_drawdown"],
                            "baseline_tail_loss": baseline_eval["metrics"]["tail_loss"],
                            "stressed_tail_loss": stressed_eval["metrics"]["tail_loss"],
                            "baseline_exposure": baseline_eval["trade_stats"]["exposure"],
                            "stressed_exposure": stressed_eval["trade_stats"]["exposure"],
                            "baseline_turnover": baseline_eval["trade_stats"]["turnover"],
                            "stressed_turnover": stressed_eval["trade_stats"]["turnover"],
                            "baseline_trade_count": baseline_eval["trade_stats"]["trade_count"],
                            "stressed_trade_count": stressed_eval["trade_stats"]["trade_count"],
                            "stress_fee": eval_config["fee"],
                            "stress_slippage": eval_config["slippage"],
                            "regime_reaction_speed": metadata["regime_reaction_speed"],
                            "delta_sharpe": stressed_eval["metrics"]["sharpe"] - baseline_eval["metrics"]["sharpe"],
                            "delta_tail_loss": stressed_eval["metrics"]["tail_loss"] - baseline_eval["metrics"]["tail_loss"],
                            "delta_drawdown": stressed_eval["metrics"]["max_drawdown"] - baseline_eval["metrics"]["max_drawdown"],
                            "delta_exposure": stressed_eval["trade_stats"]["exposure"] - baseline_eval["trade_stats"]["exposure"],
                        }
                    )
        return rows

    @staticmethod
    def _summary(detail_df: pd.DataFrame) -> pd.DataFrame:
        if detail_df.empty:
            return pd.DataFrame()
        summary = (
            detail_df.groupby(["stress_scenario", "benchmark_mode"], dropna=False)[
                [
                    "stressed_sharpe",
                    "stressed_max_drawdown",
                    "stressed_tail_loss",
                    "stressed_exposure",
                    "stressed_turnover",
                    "delta_sharpe",
                    "delta_tail_loss",
                    "delta_drawdown",
                    "delta_exposure",
                    "regime_reaction_speed",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        return StressTestRunner._append_acceptance_status(summary)

    @staticmethod
    def _append_acceptance_status(summary_df: pd.DataFrame) -> pd.DataFrame:
        if summary_df.empty:
            return summary_df
        acceptance_rows: list[dict[str, Any]] = []
        comparison_count = int(len(summary_df))
        for _ in summary_df.itertuples(index=False):
            acceptance = evaluate_benchmark_acceptance(
                prediction_metric_delta=None,
                economic_metric_delta=None,
                bootstrap_ci=None,
                dm_p_value=None,
                sample_size=None,
                comparison_count=comparison_count,
            ).to_dict()
            acceptance_rows.append(
                {
                    "accepted": acceptance["accepted"],
                    "status": acceptance["status"],
                    "effect_size": acceptance["effect_size"],
                    "bootstrap_ci": acceptance["bootstrap_ci"],
                    "dm_p_value": acceptance["dm_p_value"],
                    "warnings": acceptance["warnings"],
                    "policy_version": acceptance["policy_version"],
                    "decision_reasons": acceptance["decision_reasons"],
                    "acceptance_interpretation": (
                        "Stress scenarios are exploratory_only diagnostics and do not promote benchmark claims."
                    ),
                }
            )
        return pd.concat([summary_df.reset_index(drop=True), pd.DataFrame(acceptance_rows)], axis=1)

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
            trainer = DualModelTrainer(model_dir=output_root / mode.name)
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
                    self._evaluate_contexts(
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
                ["stress_scenario", "benchmark_mode", "ticker", "horizon", "algorithm"]
            ).reset_index(drop=True)
        summary_df = self._summary(detail_df)

        detail_csv = report_path if report_path.suffix.lower() == ".csv" else report_path.with_suffix(".csv")
        summary_csv = detail_csv.with_name(f"{detail_csv.stem}_summary.csv")
        json_path = detail_csv.with_suffix(".json")
        markdown_path = detail_csv.parent / "stress_test_report.md"

        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "detail_rows": detail_df.to_dict(orient="records"),
                    "summary_rows": summary_df.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        acceptance_columns = [
            "stress_scenario",
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
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            "\n".join(
                [
                    "# Stress Test Report",
                    "",
                    "## Scenario Assumptions",
                    *[f"- `{scenario.name}`: {scenario.description}" for scenario in default_stress_scenarios()],
                    "",
                    "## Summary",
                    self._markdown_table(summary_df.round(6)),
                    "",
                    "## Acceptance Governance",
                    "Stress-test outputs are scenario diagnostics, not benchmark promotion claims.",
                    self._markdown_table(acceptance_df),
                    "",
                    "## Output Files",
                    f"- Detail CSV: `{detail_csv}`",
                    f"- Summary CSV: `{summary_csv}`",
                    f"- JSON: `{json_path}`",
                ]
            ),
            encoding="utf-8",
        )

        return {
            "detail": detail_df,
            "summary": summary_df,
            "detail_path": detail_csv,
            "summary_path": summary_csv,
            "json_path": json_path,
            "markdown_path": markdown_path,
        }
