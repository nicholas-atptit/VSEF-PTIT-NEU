"""Cost-aware strategy backtest helpers built on forecast outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.contracts import validate_position_frame
from src.ml.metrics import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)


TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class BacktestConfig:
    """Explicit execution and cost assumptions for the Phase 1 backtest."""

    horizon: int
    transaction_fee_bps: float = 15.0
    slippage_bps: float = 20.0
    allow_short: bool = False
    non_overlapping: bool = True


def side_cost_rate(
    *,
    transaction_fee_bps: float,
    slippage_bps: float,
) -> float:
    return (float(transaction_fee_bps) + float(slippage_bps)) / 10000.0


def calculate_net_trade_return(
    entry_open: float,
    exit_close: float,
    *,
    position_size: float = 1.0,
    direction: float = 1.0,
    transaction_fee_bps: float = 15.0,
    slippage_bps: float = 20.0,
) -> float:
    """Calculate net trade return after explicit entry/exit costs."""

    if entry_open <= 0 or exit_close <= 0:
        raise ValueError("Entry and exit prices must be strictly positive")
    cost_rate = side_cost_rate(
        transaction_fee_bps=transaction_fee_bps,
        slippage_bps=slippage_bps,
    )
    if float(direction) >= 0:
        effective_entry = float(entry_open) * (1.0 + cost_rate)
        effective_exit = float(exit_close) * (1.0 - cost_rate)
        gross_return = (effective_exit / effective_entry) - 1.0
    else:
        effective_entry = float(entry_open) * (1.0 - cost_rate)
        effective_exit = float(exit_close) * (1.0 + cost_rate)
        gross_return = (effective_entry / effective_exit) - 1.0
    return float(gross_return * float(position_size))


def build_trade_daily_returns(
    history_slice: pd.DataFrame,
    *,
    direction: float = 1.0,
    position_size: float = 1.0,
    transaction_fee_bps: float = 15.0,
    slippage_bps: float = 20.0,
) -> pd.Series:
    """Create a daily marked-to-market return series for one executed trade."""

    if history_slice.empty:
        return pd.Series(dtype=float)
    working = history_slice.sort_values("timestamp").reset_index(drop=True)
    cost_rate = side_cost_rate(
        transaction_fee_bps=transaction_fee_bps,
        slippage_bps=slippage_bps,
    )
    returns: list[float] = []
    for idx, row in working.iterrows():
        if idx == 0:
            base = (float(row["close"]) / (float(row["open"]) * (1.0 + cost_rate))) - 1.0
        elif idx == len(working) - 1:
            previous_close = float(working.loc[idx - 1, "close"])
            base = ((float(row["close"]) * (1.0 - cost_rate)) / previous_close) - 1.0
        else:
            previous_close = float(working.loc[idx - 1, "close"])
            base = (float(row["close"]) / previous_close) - 1.0
        returns.append(float(base) * float(direction) * float(position_size))
    return pd.Series(returns, index=pd.to_datetime(working["timestamp"]), dtype=float)


def compute_strategy_metrics(
    daily_returns: pd.Series,
    position_series: pd.Series,
    trade_returns: pd.Series,
) -> dict[str, float]:
    """Compute strategy metrics separately from the forecast metrics."""

    clean_daily = pd.to_numeric(pd.Series(daily_returns), errors="coerce").fillna(0.0).astype(float)
    clean_positions = pd.to_numeric(pd.Series(position_series), errors="coerce").fillna(0.0).astype(float)
    clean_trade_returns = pd.to_numeric(pd.Series(trade_returns), errors="coerce").dropna().astype(float)
    if clean_daily.empty:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "total_return": 0.0,
        }

    equity = (1.0 + clean_daily).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(np.power(equity.iloc[-1], TRADING_DAYS_PER_YEAR / len(clean_daily)) - 1.0) if equity.iloc[-1] > 0 else 0.0
    max_drawdown = compute_max_drawdown(equity)
    turnover = float(np.abs(np.diff(np.r_[0.0, clean_positions.to_numpy(), 0.0])).sum())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": compute_sharpe_ratio(clean_daily),
        "sortino": compute_sortino_ratio(clean_daily),
        "max_drawdown": max_drawdown,
        "calmar": compute_calmar_ratio(cagr, max_drawdown),
        "win_rate": compute_win_rate(clean_trade_returns, ignore_zero_returns=True),
        "turnover": turnover,
    }


class CostAwareBacktester:
    """Run a simple cost-aware backtest over sized strategy signals."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    @staticmethod
    def _history_lookup(frame: pd.DataFrame) -> dict[pd.Timestamp, int]:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dt.normalize()
        return {timestamp: idx for idx, timestamp in enumerate(timestamps)}

    def run(
        self,
        position_df: pd.DataFrame,
        market_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        positions = validate_position_frame(position_df)
        trades: list[dict[str, Any]] = []
        daily_returns_by_model: dict[str, pd.Series] = {}
        position_history_by_model: dict[str, pd.Series] = {}

        for model_name, model_group in positions.groupby("model_name", sort=True):
            model_daily_returns: list[pd.Series] = []
            model_position_points: list[tuple[pd.Timestamp, float]] = []

            for ticker, ticker_group in model_group.groupby("ticker", sort=True):
                history = market_data[ticker].copy()
                history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce").dt.normalize()
                history = history.sort_values("timestamp").reset_index(drop=True)
                lookup = self._history_lookup(history)
                next_eligible_ts: pd.Timestamp | None = None

                for row in ticker_group.sort_values("timestamp").itertuples(index=False):
                    signal_ts = pd.Timestamp(row.timestamp).normalize()
                    if self.config.non_overlapping and next_eligible_ts is not None and signal_ts < next_eligible_ts:
                        continue
                    if float(row.position_size) == 0.0 or float(row.signal) == 0.0:
                        model_position_points.append((signal_ts, 0.0))
                        continue

                    signal_idx = lookup.get(signal_ts)
                    if signal_idx is None or (signal_idx + 1) >= len(history):
                        continue
                    entry_idx = signal_idx + 1
                    if "target_timestamp" in ticker_group.columns and pd.notna(getattr(row, "target_timestamp", pd.NaT)):
                        target_ts = pd.Timestamp(getattr(row, "target_timestamp")).normalize()
                        exit_idx = lookup.get(target_ts)
                    else:
                        exit_idx = min(signal_idx + int(self.config.horizon), len(history) - 1)
                    if exit_idx is None or exit_idx <= signal_idx or exit_idx < entry_idx:
                        continue

                    trade_history = history.iloc[entry_idx : exit_idx + 1][["timestamp", "open", "close"]].copy()
                    trade_returns = build_trade_daily_returns(
                        trade_history,
                        direction=float(row.signal),
                        position_size=float(row.position_size),
                        transaction_fee_bps=self.config.transaction_fee_bps,
                        slippage_bps=self.config.slippage_bps,
                    )
                    model_daily_returns.append(trade_returns)
                    model_position_points.append((pd.Timestamp(history.loc[entry_idx, "timestamp"]), float(row.position_size)))
                    model_position_points.append((pd.Timestamp(history.loc[exit_idx, "timestamp"]), 0.0))
                    net_trade_return = calculate_net_trade_return(
                        entry_open=float(history.loc[entry_idx, "open"]),
                        exit_close=float(history.loc[exit_idx, "close"]),
                        position_size=float(row.position_size),
                        direction=float(row.signal),
                        transaction_fee_bps=self.config.transaction_fee_bps,
                        slippage_bps=self.config.slippage_bps,
                    )
                    trades.append(
                        {
                            "model_name": model_name,
                            "ticker": ticker,
                            "signal_timestamp": str(signal_ts.date()),
                            "entry_timestamp": str(pd.Timestamp(history.loc[entry_idx, "timestamp"]).date()),
                            "exit_timestamp": str(pd.Timestamp(history.loc[exit_idx, "timestamp"]).date()),
                            "signal": float(row.signal),
                            "position_size": float(row.position_size),
                            "entry_open": float(history.loc[entry_idx, "open"]),
                            "exit_close": float(history.loc[exit_idx, "close"]),
                            "net_trade_return": float(net_trade_return),
                        }
                    )
                    next_eligible_ts = pd.Timestamp(history.loc[exit_idx, "timestamp"]).normalize()

            combined_returns = (
                pd.concat(model_daily_returns, axis=1).fillna(0.0).sum(axis=1).sort_index()
                if model_daily_returns
                else pd.Series(dtype=float)
            )
            if model_position_points:
                position_series = (
                    pd.Series(
                        {timestamp: size for timestamp, size in model_position_points},
                        dtype=float,
                    )
                    .sort_index()
                    .groupby(level=0)
                    .last()
                )
            else:
                position_series = pd.Series(dtype=float)
            daily_returns_by_model[model_name] = combined_returns
            position_history_by_model[model_name] = position_series

        trade_df = pd.DataFrame(trades).sort_values(["model_name", "ticker", "entry_timestamp"]).reset_index(drop=True) if trades else pd.DataFrame(
            columns=[
                "model_name",
                "ticker",
                "signal_timestamp",
                "entry_timestamp",
                "exit_timestamp",
                "signal",
                "position_size",
                "entry_open",
                "exit_close",
                "net_trade_return",
            ]
        )
        metric_rows: list[dict[str, Any]] = []
        equity_rows: list[pd.DataFrame] = []
        for model_name, daily_returns in daily_returns_by_model.items():
            position_series = position_history_by_model.get(model_name, pd.Series(dtype=float)).reindex(daily_returns.index, method="ffill").fillna(0.0)
            model_trades = trade_df[trade_df["model_name"] == model_name]["net_trade_return"] if not trade_df.empty else pd.Series(dtype=float)
            metrics = compute_strategy_metrics(daily_returns, position_series, model_trades)
            metric_rows.append({"model_name": model_name, **metrics})
            if not daily_returns.empty:
                equity_rows.append(
                    pd.DataFrame(
                        {
                            "timestamp": daily_returns.index,
                            "model_name": model_name,
                            "daily_return": daily_returns.to_numpy(),
                            "equity_curve": (1.0 + daily_returns).cumprod().to_numpy(),
                            "position_size": position_series.to_numpy(),
                        }
                    )
                )

        metrics_df = pd.DataFrame(metric_rows).sort_values("model_name").reset_index(drop=True) if metric_rows else pd.DataFrame(
            columns=["model_name", "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "turnover"]
        )
        equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame(
            columns=["timestamp", "model_name", "daily_return", "equity_curve", "position_size"]
        )
        return {
            "trades": trade_df,
            "strategy_metrics": metrics_df,
            "equity_curve": equity_df,
        }
