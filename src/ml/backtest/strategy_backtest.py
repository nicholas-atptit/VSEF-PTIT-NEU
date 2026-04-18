"""Strategy backtesting layer built on forward-return forecast artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.backtest.forward_return import (
    BASELINE_MODEL_NAMES,
    FORWARD_RETURN_HORIZONS,
    MOMENTUM_BASELINE_NAME,
    ForwardReturnBacktestConfig,
    ForwardReturnBacktestRunner,
)
from src.ml.backtest.real_data import STANDARD_COLUMNS
from src.ml.metrics import (
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_weight_turnover as canonical_compute_weight_turnover,
)
from src.utils.logging import get_logger
from src.validators.data_quality import DataQualityValidator

logger = get_logger(__name__)

DEFAULT_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
BUY_AND_HOLD_MODEL_NAME = "buy_and_hold"
NAIVE_FLAT_STRATEGY_NAME = "naive_flat_strategy"
PORTFOLIO_TICKER = "PORTFOLIO_EQUAL_WEIGHT"
MODEL_STRATEGY_TYPE = "model"
BENCHMARK_STRATEGY_TYPE = "benchmark"
PORTFOLIO_CAPITAL_MODEL = "equal_weight_active_positions_with_cash_when_flat"
PORTFOLIO_TURNOVER_DEFINITION = (
    "gross sum of absolute changes in risky-asset portfolio weights, including opening and terminal liquidation"
)
PORTFOLIO_TRADE_DIAGNOSTIC_BASIS = (
    "raw constituent trade diagnostics; not portfolio-weighted equity metrics"
)


@dataclass(slots=True)
class StrategyBacktestConfig(ForwardReturnBacktestConfig):
    output_dir: str = "artifacts/strategy_backtest"
    forecast_output_dir: str = "artifacts/backtest_forward_return"
    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    transaction_fee_bps: float = 15.0
    slippage_bps: float = 20.0
    entry_price_field: str = "open"
    exit_price_field: str = "close"
    non_overlapping: bool = True
    rerun_forecasts_if_missing: bool = True


def normalize_thresholds(values: list[float] | tuple[float, ...] | None) -> list[float]:
    numeric_values = []
    for value in (values or DEFAULT_THRESHOLDS):
        numeric = float(value)
        if numeric < 0:
            raise ValueError(f"Thresholds must be non-negative. Got {numeric}")
        numeric_values.append(numeric)
    if not numeric_values:
        raise ValueError("At least one threshold must be provided")
    return list(dict.fromkeys(numeric_values))


def generate_signal_label(predicted_return: float | int | None, threshold: float) -> str:
    if predicted_return is None or pd.isna(predicted_return):
        return "invalid"
    predicted = float(predicted_return)
    if predicted > threshold:
        return "buy"
    if abs(predicted) <= threshold:
        return "hold"
    return "stay_out"


def calculate_net_trade_return(entry_price: float, exit_price: float, side_cost_rate: float) -> float:
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Entry and exit prices must be strictly positive")
    effective_entry = float(entry_price) * (1.0 + side_cost_rate)
    effective_exit = float(exit_price) * (1.0 - side_cost_rate)
    return float((effective_exit / effective_entry) - 1.0)


def build_trade_daily_returns(history_slice: pd.DataFrame, side_cost_rate: float) -> pd.Series:
    if history_slice.empty:
        return pd.Series(dtype=float)

    working = history_slice.sort_values("date").reset_index(drop=True)
    if len(working) == 1:
        single_return = calculate_net_trade_return(
            float(working.loc[0, "open"]),
            float(working.loc[0, "close"]),
            side_cost_rate,
        )
        return pd.Series([single_return], index=working["date"], dtype=float)

    returns: list[float] = []
    for idx, row in working.iterrows():
        if idx == 0:
            daily_return = (float(row["close"]) / (float(row["open"]) * (1.0 + side_cost_rate))) - 1.0
        elif idx == len(working) - 1:
            previous_close = float(working.loc[idx - 1, "close"])
            daily_return = ((float(row["close"]) * (1.0 - side_cost_rate)) / previous_close) - 1.0
        else:
            previous_close = float(working.loc[idx - 1, "close"])
            daily_return = (float(row["close"]) / previous_close) - 1.0
        returns.append(float(daily_return))
    return pd.Series(returns, index=working["date"], dtype=float)


def compute_strategy_metrics(
    daily_returns: pd.Series,
    positions: pd.Series,
    trade_returns: pd.Series,
    *,
    weight_history: pd.DataFrame | None = None,
) -> dict[str, float]:
    clean_daily = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0).astype(float)
    clean_positions = pd.to_numeric(positions, errors="coerce").fillna(0.0).astype(float)
    clean_trade_returns = pd.to_numeric(trade_returns, errors="coerce").dropna().astype(float)

    if clean_daily.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "number_of_trades": 0,
            "average_trade_return": 0.0,
            "exposure_ratio": 0.0,
            "turnover": 0.0,
        }

    equity_curve = (1.0 + clean_daily).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0)
    annualized_return = (
        float(np.power(equity_curve.iloc[-1], 252.0 / len(clean_daily)) - 1.0)
        if equity_curve.iloc[-1] > 0
        else 0.0
    )
    sharpe_ratio = compute_sharpe_ratio(clean_daily)
    sortino_ratio = compute_sortino_ratio(clean_daily)
    max_drawdown = compute_max_drawdown(equity_curve)
    profit_factor = compute_profit_factor(clean_trade_returns)

    win_rate = float((clean_trade_returns > 0.0).mean()) if not clean_trade_returns.empty else 0.0
    if weight_history is not None and not weight_history.empty:
        turnover = compute_weight_turnover(weight_history)
    else:
        position_states = clean_positions.round().astype(int).to_numpy()
        turnover = float(np.abs(np.diff(np.r_[0, position_states, 0])).sum())
    exposure_ratio = float(clean_positions.mean()) if not clean_positions.empty else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "number_of_trades": int(len(clean_trade_returns)),
        "average_trade_return": float(clean_trade_returns.mean()) if not clean_trade_returns.empty else 0.0,
        "exposure_ratio": exposure_ratio,
        "turnover": turnover,
    }


def aggregate_active_position_portfolio(
    pivot_return: pd.DataFrame,
    pivot_position: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Use active-position-only equal weights and keep idle days in cash."""
    weights, active_count, cash_weight = build_active_position_weight_frame(pivot_position)
    sanitized_returns = pivot_return.fillna(0.0).astype(float)
    portfolio_daily_return = (sanitized_returns * weights).sum(axis=1).astype(float)
    portfolio_position = weights.sum(axis=1).astype(float)
    return portfolio_daily_return, portfolio_position, active_count.astype(int), cash_weight


def build_active_position_weight_frame(
    pivot_position: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build risky-asset weights for the active-position equal-weight capital model."""
    active_mask = pivot_position.fillna(0.0).astype(float) > 0.0
    active_count = active_mask.sum(axis=1)
    divisor = active_count.where(active_count > 0, 1).astype(float)
    weights = active_mask.astype(float).div(divisor, axis=0).where(active_mask, 0.0).astype(float)
    cash_weight = (1.0 - weights.sum(axis=1)).clip(lower=0.0, upper=1.0).astype(float)
    return weights, active_count.astype(int), cash_weight


def compute_weight_turnover(weight_history: pd.DataFrame) -> float:
    """Compatibility wrapper around the canonical risky-asset weight turnover helper."""
    return canonical_compute_weight_turnover(weight_history)


class StrategyBacktestRunner:
    """Convert forward-return forecasts into threshold-based trading strategies."""

    def __init__(self, config: StrategyBacktestConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.forecast_output_dir = Path(config.forecast_output_dir)
        self.summary_dir = self.output_dir / "summary"
        self.charts_dir = self.output_dir / "charts"
        self.adapter = VnstockAdapter(symbol_list=[ticker.upper() for ticker in config.tickers])
        self._resolved_horizons = {
            name: FORWARD_RETURN_HORIZONS[name]
            for name in dict.fromkeys(
                str(value).strip().lower()
                for value in (config.horizons or list(FORWARD_RETURN_HORIZONS))
                if str(value).strip()
            )
        }
        invalid_horizons = [name for name in self._resolved_horizons if name not in FORWARD_RETURN_HORIZONS]
        if invalid_horizons:
            raise ValueError(
                f"Unsupported horizons: {invalid_horizons}. Available: {sorted(FORWARD_RETURN_HORIZONS)}"
            )
        if not self._resolved_horizons:
            raise ValueError("At least one horizon must be specified")
        self._thresholds = normalize_thresholds(config.thresholds)

    @staticmethod
    def _normalize_dates(config: StrategyBacktestConfig) -> dict[str, pd.Timestamp]:
        return {
            "train_start": pd.Timestamp(config.train_start).normalize(),
            "train_end": pd.Timestamp(config.train_end).normalize(),
            "eval_start": pd.Timestamp(config.eval_start).normalize(),
            "eval_end": pd.Timestamp(config.eval_end).normalize(),
        }

    def _forecast_config(self) -> ForwardReturnBacktestConfig:
        return ForwardReturnBacktestConfig(
            tickers=[ticker.upper().strip() for ticker in self.config.tickers],
            train_start=self.config.train_start,
            train_end=self.config.train_end,
            eval_start=self.config.eval_start,
            eval_end=self.config.eval_end,
            output_dir=str(self.forecast_output_dir),
            algorithms=list(self.config.algorithms),
            primary_algorithm=self.config.primary_algorithm,
            interval=self.config.interval,
            horizon_name=self.config.horizon_name,
            horizon_days=self.config.horizon_days,
            sequence_length=self.config.sequence_length,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            patience=self.config.patience,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            criterion=self.config.criterion,
            validation_fraction=self.config.validation_fraction,
            validation_min_rows=self.config.validation_min_rows,
            min_train_rows=self.config.min_train_rows,
            clean_model_dir=self.config.clean_model_dir,
            horizons=list(self._resolved_horizons),
            task_type=self.config.task_type,
            target_type=self.config.target_type,
            include_momentum_baseline=self.config.include_momentum_baseline,
            beats_baseline_rule=self.config.beats_baseline_rule,
        )

    def _load_existing_forecasts(self) -> dict[str, Any] | None:
        horizon_results: dict[str, dict[str, Any]] = {}
        available_algorithms: list[str] | None = None
        skipped_algorithms: list[dict[str, str]] | None = None

        for horizon_name in self._resolved_horizons:
            horizon_dir = self.forecast_output_dir / horizon_name
            required_paths = {
                "predicted_vs_actual": horizon_dir / "predicted_vs_actual.csv",
                "metrics_summary": horizon_dir / "metrics_summary.csv",
                "run_config": horizon_dir / "run_config.json",
                "fetch_summary": horizon_dir / "fetch_summary.csv",
                "training_summary": horizon_dir / "training_summary.csv",
            }
            if not all(path.exists() for path in required_paths.values()):
                return None

            run_config = json.loads(required_paths["run_config"].read_text(encoding="utf-8"))
            expected_tickers = sorted(ticker.upper().strip() for ticker in self.config.tickers)
            if sorted(run_config.get("tickers", [])) != expected_tickers:
                return None
            if run_config.get("train_start") != self.config.train_start:
                return None
            if run_config.get("train_end") != self.config.train_end:
                return None
            if run_config.get("eval_start") != self.config.eval_start:
                return None
            if run_config.get("eval_end") != self.config.eval_end:
                return None
            if run_config.get("target_type") != "forward_return":
                return None

            comparison_df = pd.read_csv(required_paths["predicted_vs_actual"])
            metrics_df = pd.read_csv(required_paths["metrics_summary"])
            fetch_summary_df = pd.read_csv(required_paths["fetch_summary"])
            training_df = pd.read_csv(required_paths["training_summary"])

            required_columns = {
                "ticker",
                "model_name",
                "prediction_date",
                "target_date",
                "actual_return",
                "predicted_return",
            }
            if not required_columns.issubset(comparison_df.columns):
                return None

            horizon_results[horizon_name] = {
                "comparison": comparison_df,
                "metrics": metrics_df,
                "fetch_summary": fetch_summary_df,
                "training_summary": training_df,
                "run_config": run_config,
                "paths": {name: str(path) for name, path in required_paths.items()},
            }
            available_algorithms = list(run_config.get("available_algorithms", []))
            skipped_algorithms = list(run_config.get("skipped_algorithms", []))

        overall_paths = {
            "overall_horizon_summary": str(self.forecast_output_dir / "overall_horizon_summary.csv"),
            "overall_horizon_ranking": str(self.forecast_output_dir / "overall_horizon_ranking.csv"),
        }
        return {
            "horizons": horizon_results,
            "overall_paths": overall_paths,
            "available_algorithms": available_algorithms or [],
            "skipped_algorithms": skipped_algorithms or [],
        }

    def _ensure_forecasts(self) -> dict[str, Any]:
        existing = self._load_existing_forecasts()
        if existing is not None:
            logger.info("strategy_backtest_using_existing_forecasts", output_dir=str(self.forecast_output_dir))
            return existing
        if not self.config.rerun_forecasts_if_missing:
            raise FileNotFoundError(
                f"Forward-return artifacts are missing or mismatched in {self.forecast_output_dir}"
            )
        logger.info("strategy_backtest_rerunning_forecasts", output_dir=str(self.forecast_output_dir))
        return ForwardReturnBacktestRunner(self._forecast_config()).run()

    def _fetch_execution_history(
        self,
        ticker: str,
        *,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        history = self.adapter.get_ohlcv(
            ticker,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            interval=self.config.interval,
        )
        if history.empty:
            raise ValueError(f"No vnstock OHLCV data returned for {ticker} in the execution window")
        standardized = history.copy()[STANDARD_COLUMNS]
        standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce").dt.normalize()
        standardized = standardized.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        DataQualityValidator(ticker=ticker).validate_ohlcv(standardized)
        return standardized

    def _fetch_execution_histories(self, dates: dict[str, pd.Timestamp]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        histories: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, Any]] = []
        for raw_ticker in self.config.tickers:
            ticker = raw_ticker.upper().strip()
            history = self._fetch_execution_history(
                ticker,
                start_ts=dates["eval_start"],
                end_ts=dates["eval_end"],
            )
            fetched_min = pd.Timestamp(history["date"].min()).normalize()
            fetched_max = pd.Timestamp(history["date"].max()).normalize()
            eval_rows = int(((history["date"] >= dates["eval_start"]) & (history["date"] <= dates["eval_end"])).sum())
            if eval_rows == 0:
                raise ValueError(f"{ticker} has no tradable execution rows inside the evaluation window")
            histories[ticker] = history
            rows.append(
                {
                    "ticker": ticker,
                    "source": "vnstock",
                    "rows": int(len(history)),
                    "requested_eval_start": str(dates["eval_start"].date()),
                    "requested_eval_end": str(dates["eval_end"].date()),
                    "first_tradable_date": str(fetched_min.date()),
                    "last_tradable_date": str(fetched_max.date()),
                    "fetched_min_date": str(fetched_min.date()),
                    "fetched_max_date": str(fetched_max.date()),
                    "eval_rows": eval_rows,
                }
            )
        return histories, pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    @staticmethod
    def _strategy_candidates(comparison_df: pd.DataFrame, dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
        working = comparison_df.copy()
        working["prediction_date"] = pd.to_datetime(working["prediction_date"], errors="coerce").dt.normalize()
        working["target_date"] = pd.to_datetime(working["target_date"], errors="coerce").dt.normalize()
        filtered = working[
            (working["prediction_date"] >= dates["eval_start"])
            & (working["prediction_date"] <= dates["eval_end"])
            & (working["target_date"] >= dates["eval_start"])
            & (working["target_date"] <= dates["eval_end"])
        ].copy()
        return filtered.sort_values(["ticker", "model_name", "prediction_date", "target_date"]).reset_index(drop=True)

    @staticmethod
    def _history_date_index(history: pd.DataFrame) -> dict[pd.Timestamp, int]:
        return {
            pd.Timestamp(value).normalize(): idx
            for idx, value in enumerate(pd.to_datetime(history["date"], errors="coerce").dt.normalize())
        }

    def _build_trades_for_group(
        self,
        *,
        prediction_rows: pd.DataFrame,
        history: pd.DataFrame,
        ticker: str,
        horizon_name: str,
        model_name: str,
        strategy_type: str,
        threshold: float,
        predicted_column: str,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if prediction_rows.empty:
            return pd.DataFrame(), {"candidate_signals": 0, "buy_signals": 0, "skipped_overlap_signals": 0}

        working = prediction_rows.copy()
        working["predicted_signal_return"] = pd.to_numeric(working[predicted_column], errors="coerce")
        working = working.dropna(subset=["predicted_signal_return"]).sort_values(
            ["prediction_date", "target_date"]
        ).reset_index(drop=True)
        history_lookup = self._history_date_index(history)
        side_cost_rate = (float(self.config.transaction_fee_bps) + float(self.config.slippage_bps)) / 10000.0
        last_exit_date: pd.Timestamp | None = None
        trade_rows: list[dict[str, Any]] = []
        buy_signals = 0
        skipped_overlap_signals = 0

        for raw_idx, row in enumerate(working.itertuples(index=False), start=1):
            prediction_date = pd.Timestamp(row.prediction_date).normalize()
            target_date = pd.Timestamp(row.target_date).normalize()
            predicted_return = float(row.predicted_signal_return)
            signal_label = generate_signal_label(predicted_return, threshold)
            if signal_label != "buy":
                continue

            buy_signals += 1
            if prediction_date not in history_lookup or target_date not in history_lookup:
                raise ValueError(
                    f"Missing execution prices for {ticker}: prediction_date={prediction_date.date()} target_date={target_date.date()}"
                )

            prediction_index = history_lookup[prediction_date]
            target_index = history_lookup[target_date]
            entry_index = prediction_index + 1
            if entry_index >= len(history) or entry_index > target_index:
                continue

            entry_date = pd.Timestamp(history.loc[entry_index, "date"]).normalize()
            exit_date = pd.Timestamp(history.loc[target_index, "date"]).normalize()
            if self.config.non_overlapping and last_exit_date is not None and entry_date <= last_exit_date:
                skipped_overlap_signals += 1
                continue

            trade_history = history.iloc[entry_index : target_index + 1][["date", "open", "close"]].copy()
            trade_daily_returns = build_trade_daily_returns(trade_history, side_cost_rate)
            gross_return = float(
                (float(trade_history["close"].iloc[-1]) / float(trade_history["open"].iloc[0])) - 1.0
            )
            net_return = float((1.0 + trade_daily_returns).prod() - 1.0)

            trade_rows.append(
                {
                    "horizon": horizon_name,
                    "ticker": ticker,
                    "model_name": model_name,
                    "strategy_type": strategy_type,
                    "threshold": threshold,
                    "prediction_date": str(prediction_date.date()),
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(exit_date.date()),
                    "target_date": str(target_date.date()),
                    "trade_id": f"{horizon_name}_{ticker}_{model_name}_{threshold:.4f}_{raw_idx}",
                    "predicted_return": predicted_return,
                    "actual_forward_return": float(pd.to_numeric(getattr(row, "actual_return"), errors="coerce")),
                    "entry_price": float(trade_history["open"].iloc[0]),
                    "exit_price": float(trade_history["close"].iloc[-1]),
                    "gross_trade_return": gross_return,
                    "net_trade_return": net_return,
                    "holding_sessions": int(len(trade_history)),
                    "round_trip_cost_bps": float(2.0 * (self.config.transaction_fee_bps + self.config.slippage_bps)),
                    "entry_index": int(entry_index),
                    "exit_index": int(target_index),
                }
            )
            last_exit_date = exit_date

        trades_df = (
            pd.DataFrame(trade_rows).sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
            if trade_rows
            else pd.DataFrame()
        )
        diagnostics = {
            "candidate_signals": int(len(working)),
            "buy_signals": int(buy_signals),
            "skipped_overlap_signals": int(skipped_overlap_signals),
        }
        return trades_df, diagnostics

    def _build_ticker_equity_curve(
        self,
        *,
        history: pd.DataFrame,
        trades_df: pd.DataFrame,
        ticker: str,
        horizon_name: str,
        model_name: str,
        strategy_type: str,
        threshold: float | None,
    ) -> pd.DataFrame:
        eval_history = history[
            (history["date"] >= pd.Timestamp(self.config.eval_start).normalize())
            & (history["date"] <= pd.Timestamp(self.config.eval_end).normalize())
        ][["date"]].copy()
        eval_history["daily_return"] = 0.0
        eval_history["position"] = 0.0
        eval_history["active_trade_count"] = 0
        date_index = pd.Index(pd.to_datetime(eval_history["date"], errors="coerce").dt.normalize())

        if not trades_df.empty:
            side_cost_rate = (float(self.config.transaction_fee_bps) + float(self.config.slippage_bps)) / 10000.0
            for trade in trades_df.itertuples(index=False):
                trade_history = history.iloc[int(trade.entry_index) : int(trade.exit_index) + 1][["date", "open", "close"]].copy()
                trade_daily_returns = build_trade_daily_returns(trade_history, side_cost_rate)
                trade_dates = pd.Index(pd.to_datetime(trade_history["date"], errors="coerce").dt.normalize())
                selection = date_index.isin(trade_dates)
                eval_history.loc[selection, "daily_return"] = trade_daily_returns.reindex(date_index[selection]).to_numpy(dtype=float)
                eval_history.loc[selection, "position"] = 1.0
                eval_history.loc[selection, "active_trade_count"] = 1

        eval_history["equity_curve"] = (1.0 + eval_history["daily_return"]).cumprod()
        eval_history["running_peak"] = eval_history["equity_curve"].cummax()
        eval_history["drawdown"] = (eval_history["equity_curve"] / eval_history["running_peak"]) - 1.0
        eval_history.insert(0, "horizon", horizon_name)
        eval_history.insert(1, "ticker", ticker)
        eval_history.insert(2, "model_name", model_name)
        eval_history.insert(3, "strategy_type", strategy_type)
        eval_history.insert(4, "threshold", threshold)
        eval_history["date"] = pd.to_datetime(eval_history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return eval_history[
            [
                "horizon",
                "ticker",
                "model_name",
                "strategy_type",
                "threshold",
                "date",
                "daily_return",
                "equity_curve",
                "drawdown",
                "position",
                "active_trade_count",
            ]
        ].copy()

    def _build_buy_and_hold_trade(
        self,
        *,
        history: pd.DataFrame,
        ticker: str,
        horizon_name: str,
    ) -> pd.DataFrame:
        eval_history = history[
            (history["date"] >= pd.Timestamp(self.config.eval_start).normalize())
            & (history["date"] <= pd.Timestamp(self.config.eval_end).normalize())
        ].reset_index(drop=True)
        if eval_history.empty:
            raise ValueError(f"{ticker} has no execution rows in the evaluation window")
        trade = {
            "horizon": horizon_name,
            "ticker": ticker,
            "model_name": BUY_AND_HOLD_MODEL_NAME,
            "strategy_type": BENCHMARK_STRATEGY_TYPE,
            "threshold": np.nan,
            "prediction_date": str(pd.Timestamp(eval_history.loc[0, "date"]).date()),
            "entry_date": str(pd.Timestamp(eval_history.loc[0, "date"]).date()),
            "exit_date": str(pd.Timestamp(eval_history.loc[len(eval_history) - 1, "date"]).date()),
            "target_date": str(pd.Timestamp(eval_history.loc[len(eval_history) - 1, "date"]).date()),
            "trade_id": f"{horizon_name}_{ticker}_{BUY_AND_HOLD_MODEL_NAME}",
            "predicted_return": np.nan,
            "actual_forward_return": np.nan,
            "entry_price": float(eval_history.loc[0, "open"]),
            "exit_price": float(eval_history.loc[len(eval_history) - 1, "close"]),
            "gross_trade_return": float(
                (float(eval_history.loc[len(eval_history) - 1, "close"]) / float(eval_history.loc[0, "open"])) - 1.0
            ),
            "net_trade_return": calculate_net_trade_return(
                float(eval_history.loc[0, "open"]),
                float(eval_history.loc[len(eval_history) - 1, "close"]),
                (float(self.config.transaction_fee_bps) + float(self.config.slippage_bps)) / 10000.0,
            ),
            "holding_sessions": int(len(eval_history)),
            "round_trip_cost_bps": float(2.0 * (self.config.transaction_fee_bps + self.config.slippage_bps)),
            "entry_index": int(eval_history.index[0]),
            "exit_index": int(eval_history.index[-1]),
        }
        return pd.DataFrame([trade])

    def _build_flat_equity_curve(
        self,
        *,
        history: pd.DataFrame,
        ticker: str,
        horizon_name: str,
    ) -> pd.DataFrame:
        eval_history = history[
            (history["date"] >= pd.Timestamp(self.config.eval_start).normalize())
            & (history["date"] <= pd.Timestamp(self.config.eval_end).normalize())
        ][["date"]].copy()
        eval_history["daily_return"] = 0.0
        eval_history["equity_curve"] = 1.0
        eval_history["drawdown"] = 0.0
        eval_history["position"] = 0.0
        eval_history["active_trade_count"] = 0
        eval_history.insert(0, "horizon", horizon_name)
        eval_history.insert(1, "ticker", ticker)
        eval_history.insert(2, "model_name", NAIVE_FLAT_STRATEGY_NAME)
        eval_history.insert(3, "strategy_type", BENCHMARK_STRATEGY_TYPE)
        eval_history.insert(4, "threshold", np.nan)
        eval_history["date"] = pd.to_datetime(eval_history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return eval_history[
            [
                "horizon",
                "ticker",
                "model_name",
                "strategy_type",
                "threshold",
                "date",
                "daily_return",
                "equity_curve",
                "drawdown",
                "position",
                "active_trade_count",
            ]
        ].copy()

    def _metrics_row(
        self,
        *,
        horizon_name: str,
        ticker: str,
        model_name: str,
        strategy_type: str,
        threshold: float | None,
        equity_curve_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        diagnostics: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        metrics = compute_strategy_metrics(
            pd.to_numeric(equity_curve_df["daily_return"], errors="coerce"),
            pd.to_numeric(equity_curve_df["position"], errors="coerce"),
            pd.to_numeric(trades_df.get("net_trade_return", pd.Series(dtype=float)), errors="coerce"),
        )
        return {
            "horizon": horizon_name,
            "ticker": ticker,
            "model_name": model_name,
            "strategy_type": strategy_type,
            "threshold": threshold,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "number_of_trades": metrics["number_of_trades"],
            "average_trade_return": metrics["average_trade_return"],
            "exposure_ratio": metrics["exposure_ratio"],
            "turnover": metrics["turnover"],
            "turnover_definition": "binary position-state transitions including entry and terminal exit",
            "candidate_signals": int((diagnostics or {}).get("candidate_signals", 0)),
            "buy_signals": int((diagnostics or {}).get("buy_signals", 0)),
            "skipped_overlap_signals": int((diagnostics or {}).get("skipped_overlap_signals", 0)),
            "positive_net_return_after_costs": bool(metrics["total_return"] > 0.0),
        }

    def _build_portfolio_artifacts(
        self,
        *,
        horizon_name: str,
        ticker_equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        portfolio_rows: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        if ticker_equity_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        for (model_name, strategy_type, threshold), group in ticker_equity_df.groupby(
            ["model_name", "strategy_type", "threshold"],
            dropna=False,
            sort=True,
        ):
            pivot_return = group.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="first").fillna(0.0)
            pivot_position = group.pivot_table(index="date", columns="ticker", values="position", aggfunc="first").fillna(0.0)
            portfolio_weights, active_trade_count, cash_weight = build_active_position_weight_frame(pivot_position)
            portfolio_daily_return, portfolio_position, active_trade_count, cash_weight = (
                aggregate_active_position_portfolio(pivot_return, pivot_position)
            )
            portfolio_equity = (1.0 + portfolio_daily_return).cumprod()
            portfolio_drawdown = (portfolio_equity / portfolio_equity.cummax()) - 1.0

            portfolio_frame = pd.DataFrame(
                {
                    "horizon": horizon_name,
                    "ticker": PORTFOLIO_TICKER,
                    "model_name": model_name,
                    "strategy_type": strategy_type,
                    "threshold": threshold if not pd.isna(threshold) else np.nan,
                    "date": portfolio_daily_return.index,
                    "daily_return": portfolio_daily_return.to_numpy(dtype=float),
                    "equity_curve": portfolio_equity.to_numpy(dtype=float),
                    "drawdown": portfolio_drawdown.to_numpy(dtype=float),
                    "position": portfolio_position.to_numpy(dtype=float),
                    "active_trade_count": active_trade_count.to_numpy(dtype=int),
                    "cash_weight": cash_weight.to_numpy(dtype=float),
                    "capital_model": PORTFOLIO_CAPITAL_MODEL,
                    "scope": "portfolio",
                }
            )
            portfolio_rows.append(portfolio_frame)

            matching_trades = trades_df[
                (trades_df["horizon"] == horizon_name)
                & (trades_df["model_name"] == model_name)
                & (trades_df["strategy_type"] == strategy_type)
            ].copy()
            if pd.isna(threshold):
                matching_trades = matching_trades[matching_trades["threshold"].isna()]
            else:
                matching_trades = matching_trades[matching_trades["threshold"] == threshold]

            metrics = compute_strategy_metrics(
                portfolio_daily_return,
                portfolio_position,
                pd.to_numeric(matching_trades.get("net_trade_return", pd.Series(dtype=float)), errors="coerce"),
                weight_history=portfolio_weights,
            )
            metric_rows.append(
                {
                    "horizon": horizon_name,
                    "ticker": PORTFOLIO_TICKER,
                    "model_name": model_name,
                    "strategy_type": strategy_type,
                    "threshold": threshold if not pd.isna(threshold) else np.nan,
                    "total_return": metrics["total_return"],
                    "annualized_return": metrics["annualized_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "sortino_ratio": metrics["sortino_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "exposure_ratio": metrics["exposure_ratio"],
                    "turnover": metrics["turnover"],
                    "positive_net_return_after_costs": bool(metrics["total_return"] > 0.0),
                    "capital_model": PORTFOLIO_CAPITAL_MODEL,
                    "turnover_definition": PORTFOLIO_TURNOVER_DEFINITION,
                    "average_active_positions": float(active_trade_count.mean()) if len(active_trade_count) else 0.0,
                    "cash_days_ratio": float(cash_weight.mean()) if len(cash_weight) else 0.0,
                    "trade_diagnostic_basis": PORTFOLIO_TRADE_DIAGNOSTIC_BASIS,
                    "trade_diagnostic_win_rate": metrics["win_rate"],
                    "trade_diagnostic_profit_factor": metrics["profit_factor"],
                    "trade_diagnostic_count": metrics["number_of_trades"],
                    "trade_diagnostic_average_trade_return": metrics["average_trade_return"],
                }
            )

        portfolio_equity_df = pd.concat(portfolio_rows, ignore_index=True).sort_values(
            ["model_name", "threshold", "date"]
        ).reset_index(drop=True)
        portfolio_metrics_df = pd.DataFrame(metric_rows).sort_values(["model_name", "threshold"]).reset_index(drop=True)
        return portfolio_equity_df, portfolio_metrics_df

    @staticmethod
    def _annotate_benchmark_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
        if metrics_df.empty:
            return metrics_df
        annotated = metrics_df.copy()
        annotated["beats_buy_and_hold"] = False
        annotated["beats_naive_flat_strategy"] = False
        for horizon_name, group in annotated.groupby("horizon", sort=True):
            portfolio_rows = group[group["ticker"] == PORTFOLIO_TICKER]
            buy_and_hold = portfolio_rows[portfolio_rows["model_name"] == BUY_AND_HOLD_MODEL_NAME]
            flat = portfolio_rows[portfolio_rows["model_name"] == NAIVE_FLAT_STRATEGY_NAME]
            buy_and_hold_return = float(buy_and_hold["total_return"].iloc[0]) if not buy_and_hold.empty else np.nan
            flat_return = float(flat["total_return"].iloc[0]) if not flat.empty else 0.0
            model_mask = (annotated["horizon"] == horizon_name) & (annotated["strategy_type"] == MODEL_STRATEGY_TYPE)
            if not np.isnan(buy_and_hold_return):
                annotated.loc[model_mask, "beats_buy_and_hold"] = annotated.loc[model_mask, "total_return"] > buy_and_hold_return
            annotated.loc[model_mask, "beats_naive_flat_strategy"] = annotated.loc[model_mask, "total_return"] > flat_return
        return annotated

    def _render_charts(
        self,
        *,
        horizon_name: str,
        portfolio_equity_df: pd.DataFrame,
        portfolio_metrics_df: pd.DataFrame,
        ticker_metrics_df: pd.DataFrame,
    ) -> dict[str, str]:
        chart_paths: dict[str, str] = {}
        horizon_metrics = portfolio_metrics_df[
            (portfolio_metrics_df["horizon"] == horizon_name)
            & (portfolio_metrics_df["strategy_type"] == MODEL_STRATEGY_TYPE)
        ].copy()
        if horizon_metrics.empty:
            return chart_paths

        benchmark_curves = portfolio_equity_df[
            (portfolio_equity_df["horizon"] == horizon_name)
            & (portfolio_equity_df["ticker"] == PORTFOLIO_TICKER)
            & (portfolio_equity_df["model_name"].isin([BUY_AND_HOLD_MODEL_NAME, NAIVE_FLAT_STRATEGY_NAME]))
        ].copy()

        for model_name, model_group in horizon_metrics.groupby("model_name", sort=True):
            best_row = model_group.sort_values(["total_return", "sharpe_ratio", "max_drawdown"], ascending=[False, False, False]).iloc[0]
            threshold = best_row["threshold"]
            curve = portfolio_equity_df[
                (portfolio_equity_df["horizon"] == horizon_name)
                & (portfolio_equity_df["ticker"] == PORTFOLIO_TICKER)
                & (portfolio_equity_df["model_name"] == model_name)
            ].copy()
            if pd.isna(threshold):
                curve = curve[curve["threshold"].isna()]
            else:
                curve = curve[curve["threshold"] == threshold]
            curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
            curve = curve.sort_values("date")

            equity_chart_path = self.charts_dir / f"{horizon_name}_{model_name}_equity_curve.png"
            plt.figure(figsize=(10, 5))
            plt.plot(curve["date"], curve["equity_curve"], label=f"{model_name} thr={threshold:.3f}", linewidth=2.0)
            for benchmark_name in (BUY_AND_HOLD_MODEL_NAME, NAIVE_FLAT_STRATEGY_NAME):
                benchmark_curve = benchmark_curves[benchmark_curves["model_name"] == benchmark_name].copy()
                if benchmark_curve.empty:
                    continue
                benchmark_curve["date"] = pd.to_datetime(benchmark_curve["date"], errors="coerce")
                benchmark_curve = benchmark_curve.sort_values("date")
                plt.plot(benchmark_curve["date"], benchmark_curve["equity_curve"], label=benchmark_name, linewidth=1.5, linestyle="--")
            plt.title(f"{horizon_name.upper()} Portfolio Equity Curve - {model_name}")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.xticks(rotation=30)
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(equity_chart_path, dpi=140)
            plt.close()
            chart_paths[f"{horizon_name}_{model_name}_equity_curve"] = str(equity_chart_path)

            drawdown_chart_path = self.charts_dir / f"{horizon_name}_{model_name}_drawdown.png"
            plt.figure(figsize=(10, 5))
            plt.plot(curve["date"], curve["drawdown"], label=f"{model_name} drawdown", linewidth=2.0)
            buy_hold_curve = benchmark_curves[benchmark_curves["model_name"] == BUY_AND_HOLD_MODEL_NAME].copy()
            if not buy_hold_curve.empty:
                buy_hold_curve["date"] = pd.to_datetime(buy_hold_curve["date"], errors="coerce")
                buy_hold_curve = buy_hold_curve.sort_values("date")
                plt.plot(buy_hold_curve["date"], buy_hold_curve["drawdown"], label="buy_and_hold drawdown", linewidth=1.5, linestyle="--")
            plt.title(f"{horizon_name.upper()} Portfolio Drawdown - {model_name}")
            plt.xlabel("Date")
            plt.ylabel("Drawdown")
            plt.xticks(rotation=30)
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(drawdown_chart_path, dpi=140)
            plt.close()
            chart_paths[f"{horizon_name}_{model_name}_drawdown"] = str(drawdown_chart_path)

            trade_count_chart_path = self.charts_dir / f"{horizon_name}_{model_name}_trade_count.png"
            trade_counts = ticker_metrics_df[
                (ticker_metrics_df["horizon"] == horizon_name)
                & (ticker_metrics_df["strategy_type"] == MODEL_STRATEGY_TYPE)
                & (ticker_metrics_df["ticker"] != PORTFOLIO_TICKER)
                & (ticker_metrics_df["model_name"] == model_name)
            ].copy()
            if pd.isna(threshold):
                trade_counts = trade_counts[trade_counts["threshold"].isna()]
            else:
                trade_counts = trade_counts[trade_counts["threshold"] == threshold]
            if not trade_counts.empty:
                plt.figure(figsize=(8.5, 4.8))
                plt.bar(trade_counts["ticker"], trade_counts["number_of_trades"], color="#3b82f6")
                plt.title(f"{horizon_name.upper()} Trade Count by Ticker - {model_name}")
                plt.xlabel("Ticker")
                plt.ylabel("Trades")
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                plt.savefig(trade_count_chart_path, dpi=140)
                plt.close()
                chart_paths[f"{horizon_name}_{model_name}_trade_count"] = str(trade_count_chart_path)

        return chart_paths

    def _write_horizon_artifacts(
        self,
        *,
        horizon_name: str,
        horizon_days: int,
        trades_df: pd.DataFrame,
        ticker_metrics_df: pd.DataFrame,
        equity_curve_df: pd.DataFrame,
        portfolio_metrics_df: pd.DataFrame,
        portfolio_equity_df: pd.DataFrame,
        fetch_summary_df: pd.DataFrame,
        forecast_metadata: dict[str, Any],
        chart_paths: dict[str, str],
    ) -> dict[str, Any]:
        horizon_dir = self.output_dir / horizon_name
        horizon_dir.mkdir(parents=True, exist_ok=True)

        equity_output = pd.concat([equity_curve_df, portfolio_equity_df], ignore_index=True).sort_values(
            ["ticker", "model_name", "threshold", "date"]
        ).reset_index(drop=True)

        paths = {
            "trades": horizon_dir / "trades.csv",
            "strategy_metrics": horizon_dir / "strategy_metrics.csv",
            "equity_curve": horizon_dir / "equity_curve.csv",
            "portfolio_metrics": horizon_dir / "portfolio_metrics.csv",
            "fetch_summary": horizon_dir / "fetch_summary.csv",
            "run_config": horizon_dir / "run_config.json",
        }
        trades_df.to_csv(paths["trades"], index=False)
        ticker_metrics_df.to_csv(paths["strategy_metrics"], index=False)
        equity_output.to_csv(paths["equity_curve"], index=False)
        portfolio_metrics_df.to_csv(paths["portfolio_metrics"], index=False)
        fetch_summary_df.to_csv(paths["fetch_summary"], index=False)

        forecast_trades = (
            trades_df[trades_df["model_name"] != BUY_AND_HOLD_MODEL_NAME].copy()
            if not trades_df.empty
            else trades_df
        )
        leakage_checks = {
            "prediction_dates_only_in_eval_window": bool(
                pd.to_datetime(forecast_trades.get("prediction_date", pd.Series(dtype=str)), errors="coerce").dropna().between(
                    pd.Timestamp(self.config.eval_start).normalize(),
                    pd.Timestamp(self.config.eval_end).normalize(),
                ).all()
            )
            if not forecast_trades.empty
            else True,
            "entry_dates_after_prediction_dates": bool(
                (
                    pd.to_datetime(forecast_trades.get("entry_date", pd.Series(dtype=str)), errors="coerce")
                    > pd.to_datetime(forecast_trades.get("prediction_date", pd.Series(dtype=str)), errors="coerce")
                ).all()
            )
            if not forecast_trades.empty
            else True,
            "exit_dates_not_after_target_dates": bool(
                (
                    pd.to_datetime(forecast_trades.get("exit_date", pd.Series(dtype=str)), errors="coerce")
                    <= pd.to_datetime(forecast_trades.get("target_date", pd.Series(dtype=str)), errors="coerce")
                ).all()
            )
            if not forecast_trades.empty
            else True,
            "no_overlapping_trades_per_ticker_model_threshold": self._validate_non_overlapping(trades_df),
            "cost_model_applied_positive": bool(
                float(self.config.transaction_fee_bps) >= 0.0 and float(self.config.slippage_bps) >= 0.0
            ),
        }

        run_config = {
            **asdict(self.config),
            "horizon": horizon_name,
            "horizon_days": horizon_days,
            "execution_assumptions": {
                "signal_time": "prediction_date close",
                "entry_convention": "next trading session open after prediction_date",
                "exit_convention": "target_date close",
                "evaluation_window_for_strategy": "prediction_date between eval_start and eval_end; target_date no later than eval_end",
                "non_overlapping": bool(self.config.non_overlapping),
            },
            "signal_rules": {
                "buy": "predicted_return > threshold",
                "hold": "abs(predicted_return) <= threshold",
                "stay_out": "predicted_return < threshold",
                "thresholds": list(self._thresholds),
            },
            "cost_model": {
                "transaction_fee_bps": float(self.config.transaction_fee_bps),
                "slippage_bps": float(self.config.slippage_bps),
                "costs_applied_on": ["entry", "exit"],
            },
            "portfolio_construction": {
                "capital_model": PORTFOLIO_CAPITAL_MODEL,
                "turnover_definition": PORTFOLIO_TURNOVER_DEFINITION,
                "trade_diagnostic_basis": PORTFOLIO_TRADE_DIAGNOSTIC_BASIS,
            },
            "forecast_output_dir": str(self.forecast_output_dir),
            "forecast_metadata": forecast_metadata,
            "leakage_checks": leakage_checks,
            "output_files": {name: str(path) for name, path in paths.items()},
            "chart_files": chart_paths,
        }
        paths["run_config"].write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        return {
            "paths": {name: str(path) for name, path in paths.items()},
            "run_config": run_config,
        }

    @staticmethod
    def _validate_non_overlapping(trades_df: pd.DataFrame) -> bool:
        if trades_df.empty:
            return True
        working = trades_df.copy()
        working["entry_date"] = pd.to_datetime(working["entry_date"], errors="coerce").dt.normalize()
        working["exit_date"] = pd.to_datetime(working["exit_date"], errors="coerce").dt.normalize()
        for _, group in working.groupby(["ticker", "model_name", "threshold"], dropna=False, sort=True):
            group = group.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
            previous_exit: pd.Timestamp | None = None
            for row in group.itertuples(index=False):
                entry_date = pd.Timestamp(row.entry_date).normalize()
                exit_date = pd.Timestamp(row.exit_date).normalize()
                if previous_exit is not None and entry_date <= previous_exit:
                    return False
                previous_exit = exit_date
        return True

    @staticmethod
    def _build_overall_ranking(portfolio_summary_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for horizon_name, group in portfolio_summary_df.groupby("horizon", sort=True):
            portfolio_rows = group[group["ticker"] == PORTFOLIO_TICKER].copy()
            model_rows = portfolio_rows[portfolio_rows["strategy_type"] == MODEL_STRATEGY_TYPE].copy()
            if model_rows.empty:
                continue
            best_total = model_rows.sort_values(
                ["total_return", "sharpe_ratio", "max_drawdown", "model_name"],
                ascending=[False, False, False, True],
            ).iloc[0]
            best_sharpe = model_rows.sort_values(
                ["sharpe_ratio", "total_return", "model_name"],
                ascending=[False, False, True],
            ).iloc[0]
            best_drawdown = model_rows.sort_values(
                ["max_drawdown", "total_return", "model_name"],
                ascending=[False, False, True],
            ).iloc[0]
            rows.append(
                {
                    "horizon": horizon_name,
                    "best_model_by_total_return": str(best_total["model_name"]),
                    "best_threshold_by_total_return": float(best_total["threshold"]),
                    "best_total_return": float(best_total["total_return"]),
                    "best_model_by_sharpe_ratio": str(best_sharpe["model_name"]),
                    "best_threshold_by_sharpe_ratio": float(best_sharpe["threshold"]),
                    "best_sharpe_ratio": float(best_sharpe["sharpe_ratio"]),
                    "best_model_by_max_drawdown_control": str(best_drawdown["model_name"]),
                    "best_threshold_by_max_drawdown_control": float(best_drawdown["threshold"]),
                    "best_max_drawdown": float(best_drawdown["max_drawdown"]),
                    "any_model_beats_buy_and_hold": bool(model_rows["beats_buy_and_hold"].any()),
                    "any_model_beats_naive_flat_strategy": bool(model_rows["beats_naive_flat_strategy"].any()),
                    "any_model_positive_net_return_after_costs": bool(model_rows["positive_net_return_after_costs"].any()),
                }
            )
        return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

    def run(self) -> dict[str, Any]:
        dates = self._normalize_dates(self.config)
        if dates["train_end"] >= dates["eval_start"]:
            raise ValueError(
                f"train_end must be strictly earlier than eval_start. Got {dates['train_end'].date()} and {dates['eval_start'].date()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        forecast_result = self._ensure_forecasts()
        histories, execution_fetch_summary = self._fetch_execution_histories(dates)

        horizon_outputs: dict[str, dict[str, Any]] = {}
        portfolio_summary_frames: list[pd.DataFrame] = []

        for horizon_name, horizon_days in self._resolved_horizons.items():
            comparison_df = forecast_result["horizons"][horizon_name]["comparison"].copy()
            candidates = self._strategy_candidates(comparison_df, dates)
            if candidates.empty:
                raise ValueError(
                    f"No strategy candidates remain for {horizon_name} after filtering prediction_date into the evaluation window"
                )

            trades_frames: list[pd.DataFrame] = []
            ticker_metric_rows: list[dict[str, Any]] = []
            ticker_equity_frames: list[pd.DataFrame] = []
            fetch_rows: list[dict[str, Any]] = []

            for ticker in sorted(histories):
                ticker_history = histories[ticker]
                ticker_candidates = candidates[candidates["ticker"] == ticker].copy()
                unique_signal_rows = ticker_candidates[["prediction_date", "target_date"]].drop_duplicates()
                base_fetch_row = execution_fetch_summary[execution_fetch_summary["ticker"] == ticker].iloc[0].to_dict()
                base_fetch_row.update(
                    {
                        "horizon": horizon_name,
                        "strategy_signal_rows": int(len(unique_signal_rows)),
                        "strategy_signal_start": str(pd.to_datetime(unique_signal_rows["prediction_date"], errors="coerce").min().date()),
                        "strategy_signal_end": str(pd.to_datetime(unique_signal_rows["prediction_date"], errors="coerce").max().date()),
                    }
                )
                fetch_rows.append(base_fetch_row)

                model_names = sorted(name for name in ticker_candidates["model_name"].dropna().unique() if str(name) not in BASELINE_MODEL_NAMES)
                for model_name in model_names:
                    model_predictions = ticker_candidates[ticker_candidates["model_name"] == model_name].copy()
                    for threshold in self._thresholds:
                        trades_df, diagnostics = self._build_trades_for_group(
                            prediction_rows=model_predictions,
                            history=ticker_history,
                            ticker=ticker,
                            horizon_name=horizon_name,
                            model_name=model_name,
                            strategy_type=MODEL_STRATEGY_TYPE,
                            threshold=threshold,
                            predicted_column="predicted_return",
                        )
                        if not trades_df.empty:
                            trades_frames.append(trades_df)
                        equity_curve_df = self._build_ticker_equity_curve(
                            history=ticker_history,
                            trades_df=trades_df,
                            ticker=ticker,
                            horizon_name=horizon_name,
                            model_name=model_name,
                            strategy_type=MODEL_STRATEGY_TYPE,
                            threshold=threshold,
                        )
                        ticker_equity_frames.append(equity_curve_df)
                        ticker_metric_rows.append(
                            self._metrics_row(
                                horizon_name=horizon_name,
                                ticker=ticker,
                                model_name=model_name,
                                strategy_type=MODEL_STRATEGY_TYPE,
                                threshold=threshold,
                                equity_curve_df=equity_curve_df,
                                trades_df=trades_df,
                                diagnostics=diagnostics,
                            )
                        )

                if self.config.include_momentum_baseline and "momentum_predicted_return" in ticker_candidates.columns:
                    for threshold in self._thresholds:
                        momentum_trades_df, diagnostics = self._build_trades_for_group(
                            prediction_rows=ticker_candidates,
                            history=ticker_history,
                            ticker=ticker,
                            horizon_name=horizon_name,
                            model_name=MOMENTUM_BASELINE_NAME,
                            strategy_type=BENCHMARK_STRATEGY_TYPE,
                            threshold=threshold,
                            predicted_column="momentum_predicted_return",
                        )
                        if not momentum_trades_df.empty:
                            trades_frames.append(momentum_trades_df)
                        momentum_equity_df = self._build_ticker_equity_curve(
                            history=ticker_history,
                            trades_df=momentum_trades_df,
                            ticker=ticker,
                            horizon_name=horizon_name,
                            model_name=MOMENTUM_BASELINE_NAME,
                            strategy_type=BENCHMARK_STRATEGY_TYPE,
                            threshold=threshold,
                        )
                        ticker_equity_frames.append(momentum_equity_df)
                        ticker_metric_rows.append(
                            self._metrics_row(
                                horizon_name=horizon_name,
                                ticker=ticker,
                                model_name=MOMENTUM_BASELINE_NAME,
                                strategy_type=BENCHMARK_STRATEGY_TYPE,
                                threshold=threshold,
                                equity_curve_df=momentum_equity_df,
                                trades_df=momentum_trades_df,
                                diagnostics=diagnostics,
                            )
                        )

                buy_hold_trades_df = self._build_buy_and_hold_trade(
                    history=ticker_history,
                    ticker=ticker,
                    horizon_name=horizon_name,
                )
                trades_frames.append(buy_hold_trades_df)
                buy_hold_equity_df = self._build_ticker_equity_curve(
                    history=ticker_history,
                    trades_df=buy_hold_trades_df,
                    ticker=ticker,
                    horizon_name=horizon_name,
                    model_name=BUY_AND_HOLD_MODEL_NAME,
                    strategy_type=BENCHMARK_STRATEGY_TYPE,
                    threshold=np.nan,
                )
                ticker_equity_frames.append(buy_hold_equity_df)
                ticker_metric_rows.append(
                    self._metrics_row(
                        horizon_name=horizon_name,
                        ticker=ticker,
                        model_name=BUY_AND_HOLD_MODEL_NAME,
                        strategy_type=BENCHMARK_STRATEGY_TYPE,
                        threshold=np.nan,
                        equity_curve_df=buy_hold_equity_df,
                        trades_df=buy_hold_trades_df,
                        diagnostics={"candidate_signals": 1, "buy_signals": 1, "skipped_overlap_signals": 0},
                    )
                )

                flat_equity_df = self._build_flat_equity_curve(
                    history=ticker_history,
                    ticker=ticker,
                    horizon_name=horizon_name,
                )
                ticker_equity_frames.append(flat_equity_df)
                ticker_metric_rows.append(
                    self._metrics_row(
                        horizon_name=horizon_name,
                        ticker=ticker,
                        model_name=NAIVE_FLAT_STRATEGY_NAME,
                        strategy_type=BENCHMARK_STRATEGY_TYPE,
                        threshold=np.nan,
                        equity_curve_df=flat_equity_df,
                        trades_df=pd.DataFrame(),
                        diagnostics={"candidate_signals": 0, "buy_signals": 0, "skipped_overlap_signals": 0},
                    )
                )

            trades_df = pd.concat(trades_frames, ignore_index=True).sort_values(
                ["ticker", "model_name", "threshold", "entry_date", "exit_date"]
            ).reset_index(drop=True)
            ticker_equity_df = pd.concat(ticker_equity_frames, ignore_index=True).sort_values(
                ["ticker", "model_name", "threshold", "date"]
            ).reset_index(drop=True)
            ticker_metrics_df = pd.DataFrame(ticker_metric_rows).sort_values(
                ["ticker", "strategy_type", "model_name", "threshold"]
            ).reset_index(drop=True)

            portfolio_equity_df, portfolio_metrics_df = self._build_portfolio_artifacts(
                horizon_name=horizon_name,
                ticker_equity_df=ticker_equity_df,
                trades_df=trades_df,
            )
            portfolio_metrics_df = self._annotate_benchmark_rows(portfolio_metrics_df)
            ticker_metrics_df = self._annotate_benchmark_rows(ticker_metrics_df)

            horizon_fetch_summary_df = pd.DataFrame(fetch_rows).sort_values("ticker").reset_index(drop=True)
            chart_paths = self._render_charts(
                horizon_name=horizon_name,
                portfolio_equity_df=portfolio_equity_df,
                portfolio_metrics_df=portfolio_metrics_df,
                ticker_metrics_df=ticker_metrics_df,
            )
            artifact_info = self._write_horizon_artifacts(
                horizon_name=horizon_name,
                horizon_days=horizon_days,
                trades_df=trades_df,
                ticker_metrics_df=ticker_metrics_df,
                equity_curve_df=ticker_equity_df,
                portfolio_metrics_df=portfolio_metrics_df,
                portfolio_equity_df=portfolio_equity_df,
                fetch_summary_df=horizon_fetch_summary_df,
                forecast_metadata={
                    "available_algorithms": forecast_result.get("available_algorithms", []),
                    "skipped_algorithms": forecast_result.get("skipped_algorithms", []),
                    "source_paths": forecast_result["horizons"][horizon_name].get("paths", {}),
                },
                chart_paths=chart_paths,
            )
            horizon_outputs[horizon_name] = {
                "trades": trades_df,
                "strategy_metrics": ticker_metrics_df,
                "equity_curve": pd.concat([ticker_equity_df, portfolio_equity_df], ignore_index=True),
                "portfolio_metrics": portfolio_metrics_df,
                "paths": artifact_info["paths"],
                "run_config": artifact_info["run_config"],
                "chart_files": chart_paths,
            }
            portfolio_summary_frames.append(portfolio_metrics_df.copy())

        model_horizon_threshold_summary = pd.concat(portfolio_summary_frames, ignore_index=True).sort_values(
            ["horizon", "strategy_type", "model_name", "threshold"]
        ).reset_index(drop=True)
        overall_strategy_ranking = self._build_overall_ranking(model_horizon_threshold_summary)

        summary_paths = {
            "model_horizon_threshold_summary": self.summary_dir / "model_horizon_threshold_summary.csv",
            "overall_strategy_ranking": self.summary_dir / "overall_strategy_ranking.csv",
        }
        model_horizon_threshold_summary.to_csv(summary_paths["model_horizon_threshold_summary"], index=False)
        overall_strategy_ranking.to_csv(summary_paths["overall_strategy_ranking"], index=False)

        return {
            "horizons": horizon_outputs,
            "summary": model_horizon_threshold_summary,
            "overall_ranking": overall_strategy_ranking,
            "summary_paths": {name: str(path) for name, path in summary_paths.items()},
            "forecast_output_dir": str(self.forecast_output_dir),
            "available_algorithms": forecast_result.get("available_algorithms", []),
            "skipped_algorithms": forecast_result.get("skipped_algorithms", []),
        }
