"""Canonical metric helpers for the modern ML and backtest path.

These helpers centralize reusable formulas without forcing every caller into the
same output schema. Benchmark, trainer, and backtest layers can wrap these
functions and keep their existing public contracts where needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TRADING_DAYS_PER_YEAR = 252.0


def compute_prediction_error_metrics(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    *,
    actual_direction: pd.Series | np.ndarray | list[float] | None = None,
    predicted_direction: pd.Series | np.ndarray | list[float] | None = None,
    mape_denominator: pd.Series | np.ndarray | list[float] | None = None,
    mape_as_percent: bool = True,
    include_residual_std: bool = False,
    residual_std_source: str = "test_residuals_std",
) -> dict[str, float]:
    actual_numeric = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted_numeric = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = actual_numeric.notna() & predicted_numeric.notna()
    if not mask.any():
        metrics: dict[str, float] = {
            "observations": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "directional_accuracy": np.nan,
        }
        if include_residual_std:
            metrics["residual_std"] = np.nan
            metrics["volatility_proxy_source"] = residual_std_source
        return metrics

    actual_filtered = actual_numeric.loc[mask]
    predicted_filtered = predicted_numeric.loc[mask]
    errors = predicted_filtered - actual_filtered
    abs_errors = errors.abs()

    if mape_denominator is None:
        denominator = actual_filtered.abs()
    else:
        denominator_series = pd.to_numeric(pd.Series(mape_denominator), errors="coerce")
        denominator = denominator_series.loc[mask].abs()

    non_zero_denominator = denominator > 0
    mape = np.nan
    if non_zero_denominator.any():
        mape = float((abs_errors.loc[non_zero_denominator] / denominator.loc[non_zero_denominator]).mean())
        if mape_as_percent:
            mape *= 100.0

    if actual_direction is None:
        actual_sign = np.sign(actual_filtered)
    else:
        actual_sign = pd.to_numeric(pd.Series(actual_direction), errors="coerce").loc[mask]
    if predicted_direction is None:
        predicted_sign = np.sign(predicted_filtered)
    else:
        predicted_sign = pd.to_numeric(pd.Series(predicted_direction), errors="coerce").loc[mask]

    directional_mask = actual_sign.notna() & predicted_sign.notna()
    directional_accuracy = np.nan
    if directional_mask.any():
        directional_accuracy = float(
            (actual_sign.loc[directional_mask] == predicted_sign.loc[directional_mask]).mean()
        )

    metrics = {
        "observations": int(mask.sum()),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }
    if include_residual_std:
        metrics["residual_std"] = float(np.std(actual_filtered - predicted_filtered))
        metrics["volatility_proxy_source"] = residual_std_source
    return metrics


def compare_prediction_metric_sets(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    rule: str = "majority_of_metrics",
) -> tuple[bool, dict[str, bool]]:
    metric_wins = {
        "mae": bool(model_metrics["mae"] < baseline_metrics["mae"]),
        "rmse": bool(model_metrics["rmse"] < baseline_metrics["rmse"]),
        "mape": bool(model_metrics["mape"] < baseline_metrics["mape"])
        if not np.isnan(model_metrics["mape"]) and not np.isnan(baseline_metrics["mape"])
        else False,
        "directional_accuracy": bool(model_metrics["directional_accuracy"] > baseline_metrics["directional_accuracy"])
        if not np.isnan(model_metrics["directional_accuracy"])
        and not np.isnan(baseline_metrics["directional_accuracy"])
        else False,
    }
    if rule == "majority_of_metrics":
        beats = sum(metric_wins.values()) >= 3
    else:
        beats = metric_wins["rmse"]
    return beats, metric_wins


def compute_binary_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    true_array = np.asarray(y_true, dtype=int).reshape(-1)
    pred_array = np.asarray(y_pred, dtype=int).reshape(-1)
    if len(true_array) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_class_precision": 0.0,
            "roc_auc": np.nan,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 0,
        }

    metrics = {
        "accuracy": float(accuracy_score(true_array, pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(true_array, pred_array)),
        "precision": float(precision_score(true_array, pred_array, zero_division=0)),
        "recall": float(recall_score(true_array, pred_array, zero_division=0)),
        "f1": float(f1_score(true_array, pred_array, zero_division=0)),
        "positive_class_precision": float(precision_score(true_array, pred_array, zero_division=0)),
    }
    roc_auc = np.nan
    if y_prob is not None:
        probability_array = np.asarray(y_prob, dtype=float).reshape(-1)
        if len(np.unique(true_array)) > 1:
            try:
                roc_auc = float(roc_auc_score(true_array, probability_array))
            except ValueError:
                roc_auc = np.nan
    metrics["roc_auc"] = roc_auc
    tn, fp, fn, tp = confusion_matrix(true_array, pred_array, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    return metrics


def compute_brier_score(
    y_true: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series,
) -> float:
    true_series = pd.to_numeric(pd.Series(y_true), errors="coerce")
    prob_series = pd.to_numeric(pd.Series(y_prob), errors="coerce")
    mask = true_series.isin([0, 1]) & prob_series.notna()
    if not mask.any():
        return np.nan
    true_array = true_series.loc[mask].astype(float).to_numpy()
    prob_array = np.clip(prob_series.loc[mask].astype(float).to_numpy(), 0.0, 1.0)
    return float(np.mean(np.square(prob_array - true_array)))


def summarize_binary_probability_calibration(
    y_true: np.ndarray | list[int] | pd.Series,
    y_prob: np.ndarray | list[float] | pd.Series,
    *,
    num_bins: int = 5,
) -> dict[str, Any]:
    true_series = pd.to_numeric(pd.Series(y_true), errors="coerce")
    prob_series = pd.to_numeric(pd.Series(y_prob), errors="coerce")
    mask = true_series.isin([0, 1]) & prob_series.notna()
    if not mask.any():
        return {
            "available": False,
            "reason": "direction_probability_unavailable_or_invalid",
            "observations": 0,
            "brier_score": np.nan,
            "avg_abs_calibration_gap": np.nan,
            "max_abs_calibration_gap": np.nan,
            "bin_count": int(num_bins),
            "bins": [],
        }

    true_array = true_series.loc[mask].astype(float).to_numpy()
    prob_array = np.clip(prob_series.loc[mask].astype(float).to_numpy(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    assignments = np.digitize(prob_array, edges[1:-1], right=True)
    bins: list[dict[str, Any]] = []
    abs_gaps: list[float] = []
    weights: list[int] = []

    for idx in range(int(num_bins)):
        bin_mask = assignments == idx
        if not np.any(bin_mask):
            continue
        avg_probability = float(np.mean(prob_array[bin_mask]))
        empirical_rate = float(np.mean(true_array[bin_mask]))
        gap = float(empirical_rate - avg_probability)
        count = int(np.sum(bin_mask))
        bins.append(
            {
                "bin_index": idx,
                "range_start": float(edges[idx]),
                "range_end": float(edges[idx + 1]),
                "count": count,
                "avg_probability": avg_probability,
                "empirical_positive_rate": empirical_rate,
                "calibration_gap": gap,
            }
        )
        abs_gaps.append(abs(gap))
        weights.append(count)

    weighted_gap = np.average(abs_gaps, weights=weights) if abs_gaps else np.nan
    return {
        "available": True,
        "observations": int(mask.sum()),
        "positive_rate": float(np.mean(true_array)),
        "brier_score": compute_brier_score(true_array, prob_array),
        "avg_abs_calibration_gap": float(weighted_gap) if not np.isnan(weighted_gap) else np.nan,
        "max_abs_calibration_gap": float(max(abs_gaps)) if abs_gaps else np.nan,
        "bin_count": int(num_bins),
        "bins": bins,
        "interpretation": "Binary probability calibration summary for directional outputs only; not scenario-risk calibration.",
    }


def summarize_regression_residual_diagnostics(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    actual_numeric = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted_numeric = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = actual_numeric.notna() & predicted_numeric.notna()
    if not mask.any():
        return {
            "available": False,
            "reason": "no_valid_residuals",
            "observations": 0,
        }

    actual_array = actual_numeric.loc[mask].astype(float).to_numpy()
    predicted_array = predicted_numeric.loc[mask].astype(float).to_numpy()
    residuals = actual_array - predicted_array
    return {
        "available": True,
        "observations": int(mask.sum()),
        "residual_mean": float(np.mean(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_mae": float(np.mean(np.abs(residuals))),
        "residual_q10": float(np.quantile(residuals, 0.10)),
        "residual_q90": float(np.quantile(residuals, 0.90)),
        "underprediction_rate": float(np.mean(residuals > 0.0)),
        "overprediction_rate": float(np.mean(residuals < 0.0)),
        "interpretation": "Residual diagnostics summarize forecast error shape; they are not calibrated predictive intervals.",
    }


def compute_drawdown_series(equity_curve: pd.Series | np.ndarray | list[float]) -> pd.Series:
    equity = pd.to_numeric(pd.Series(equity_curve), errors="coerce").dropna().astype(float)
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return (equity / peak) - 1.0


def compute_max_drawdown(equity_curve: pd.Series | np.ndarray | list[float]) -> float:
    drawdown = compute_drawdown_series(equity_curve)
    return float(drawdown.min()) if not drawdown.empty else 0.0


def compute_average_drawdown(drawdown_series: pd.Series | np.ndarray | list[float]) -> float:
    drawdown = pd.to_numeric(pd.Series(drawdown_series), errors="coerce").dropna().astype(float)
    negative = drawdown[drawdown < 0.0]
    if negative.empty:
        return 0.0
    return float(negative.mean())


def compute_annualized_volatility(
    daily_returns: pd.Series | np.ndarray | list[float],
    *,
    annualization_factor: float = TRADING_DAYS_PER_YEAR,
) -> float:
    clean_daily = pd.to_numeric(pd.Series(daily_returns), errors="coerce").dropna().astype(float)
    if clean_daily.empty:
        return 0.0
    return float(clean_daily.std(ddof=0) * np.sqrt(annualization_factor))


def compute_sharpe_ratio(
    daily_returns: pd.Series | np.ndarray | list[float],
    *,
    annualization_factor: float = TRADING_DAYS_PER_YEAR,
) -> float:
    clean_daily = pd.to_numeric(pd.Series(daily_returns), errors="coerce").dropna().astype(float)
    if clean_daily.empty:
        return 0.0
    volatility = float(clean_daily.std(ddof=0))
    mean_return = float(clean_daily.mean())
    if volatility < 1e-12:
        return 1e6 if mean_return > 0 else (-1e6 if mean_return < 0 else 0.0)
    return float((mean_return / volatility) * np.sqrt(annualization_factor))


def compute_sortino_ratio(
    daily_returns: pd.Series | np.ndarray | list[float],
    *,
    annualization_factor: float = TRADING_DAYS_PER_YEAR,
) -> float:
    clean_daily = pd.to_numeric(pd.Series(daily_returns), errors="coerce").dropna().astype(float)
    if clean_daily.empty:
        return 0.0
    downside = clean_daily[clean_daily < 0.0]
    downside_vol = float(downside.std(ddof=0)) if not downside.empty else 0.0
    mean_return = float(clean_daily.mean())
    if downside_vol < 1e-12:
        return 1e6 if mean_return > 0 else (-1e6 if mean_return < 0 else 0.0)
    return float((mean_return / downside_vol) * np.sqrt(annualization_factor))


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    if abs(max_drawdown) < 1e-12:
        return 0.0 if cagr == 0 else 1e6
    return float(cagr / abs(max_drawdown))


def compute_tail_loss(
    daily_returns: pd.Series | np.ndarray | list[float],
    *,
    quantile: float = 0.05,
) -> float:
    clean_daily = pd.to_numeric(pd.Series(daily_returns), errors="coerce").dropna().astype(float)
    if clean_daily.empty:
        return 0.0
    cutoff = np.quantile(clean_daily, quantile)
    tail = clean_daily[clean_daily <= cutoff]
    if tail.empty:
        return float(cutoff)
    return float(tail.mean())


def compute_profit_factor(trade_returns: pd.Series | np.ndarray | list[float]) -> float:
    clean_trade_returns = pd.to_numeric(pd.Series(trade_returns), errors="coerce").dropna().astype(float)
    gains = clean_trade_returns[clean_trade_returns > 0.0]
    losses = clean_trade_returns[clean_trade_returns < 0.0]
    if losses.empty:
        return 10.0 if not gains.empty else 0.0
    return float(gains.sum() / abs(losses.sum()))


def compute_win_rate(trade_returns: pd.Series | np.ndarray | list[float], *, ignore_zero_returns: bool = False) -> float:
    clean_trade_returns = pd.to_numeric(pd.Series(trade_returns), errors="coerce").dropna().astype(float)
    if ignore_zero_returns:
        clean_trade_returns = clean_trade_returns[clean_trade_returns != 0.0]
    if clean_trade_returns.empty:
        return 0.0
    return float((clean_trade_returns > 0.0).mean())


def compute_signal_turnover(signal: pd.Series | np.ndarray | list[float]) -> float:
    sig = np.asarray(signal, dtype=float).reshape(-1)
    if len(sig) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(sig))))


def compute_weight_turnover(weight_history: pd.DataFrame) -> float:
    if weight_history is None or weight_history.empty:
        return 0.0
    weights = weight_history.sort_index().fillna(0.0).astype(float)
    if weights.empty:
        return 0.0
    opening = np.zeros((1, weights.shape[1]), dtype=float)
    terminal = np.zeros((1, weights.shape[1]), dtype=float)
    full_path = np.vstack([opening, weights.to_numpy(dtype=float), terminal])
    return float(np.abs(np.diff(full_path, axis=0)).sum())


def compute_exposure_ratio(positions: pd.Series | np.ndarray | list[float]) -> float:
    clean_positions = pd.to_numeric(pd.Series(positions), errors="coerce").fillna(0.0).astype(float)
    if clean_positions.empty:
        return 0.0
    return float(clean_positions.mean())
