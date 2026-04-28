"""Signal-effectiveness diagnostics for saved forecast outputs.

This module converts existing forecast CSV rows into transparent BUY/HOLD/AVOID
signals and evaluates whether BUY selections were useful after explicit cost
and slippage assumptions. It is intentionally diagnostic: it does not fetch live
data, train models, alter forecast artifacts, or claim tradable performance.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVALUATION_MODE_FRONTIER = "frontier"
EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION = "heldout_threshold_selection"
SUPPORTED_EVALUATION_MODES = {
    EVALUATION_MODE_FRONTIER,
    EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION,
}
SELECTION_CRITERION_MAX_PRECISION = "max_precision"
SUPPORTED_SELECTION_CRITERIA = {SELECTION_CRITERION_MAX_PRECISION}

SIGNAL_BUY = "BUY"
SIGNAL_HOLD = "HOLD"
SIGNAL_AVOID = "AVOID"

POLICY_RETURN_THRESHOLD = "return_threshold"
POLICY_DIRECTION_AND_RETURN_THRESHOLD = "direction_and_return_threshold"
POLICY_STRICT_BUY_PRECISION_PROBE = "strict_buy_precision_probe"
SUPPORTED_POLICIES = {
    POLICY_RETURN_THRESHOLD,
    POLICY_DIRECTION_AND_RETURN_THRESHOLD,
    POLICY_STRICT_BUY_PRECISION_PROBE,
}

SUCCESS_RAW_POSITIVE = "raw_positive"
SUCCESS_COST_ADJUSTED_POSITIVE = "cost_adjusted_positive"
SUCCESS_TARGET_RETURN = "target_return"
SUPPORTED_SUCCESS_DEFINITIONS = {
    SUCCESS_RAW_POSITIVE,
    SUCCESS_COST_ADJUSTED_POSITIVE,
    SUCCESS_TARGET_RETURN,
}

DEFAULT_PREDICTED_RETURN_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
DEFAULT_COST_PER_TRADE_VALUES = [0.001, 0.002, 0.003]
DEFAULT_SLIPPAGE_VALUES = [0.0005, 0.001]
DEFAULT_MINIMUM_SIGNAL_COUNTS = [30, 50, 100]
DEFAULT_PROBABILITY_UP_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
DEFAULT_PRECISION_TARGETS = [0.60, 0.65, 0.70]

DATE_CANDIDATES = ["prediction_date", "date", "forecast_date"]
TICKER_CANDIDATES = ["ticker", "symbol"]
MODEL_CANDIDATES = ["model_name", "model", "algorithm"]
PREDICTED_RETURN_CANDIDATES = [
    "predicted_return",
    "final_predicted_return",
    "forecast_return",
    "predicted_forward_return",
]
REALIZED_RETURN_CANDIDATES = [
    "actual_realized_forward_return",
    "realized_forward_return",
    "realized_return",
    "actual_return",
]
PREDICTED_DIRECTION_CANDIDATES = [
    "predicted_direction",
    "final_predicted_direction",
    "signal_direction",
]
PROBABILITY_UP_CANDIDATES = [
    "probability_up",
    "positive_probability",
    "predicted_positive_probability",
    "final_positive_probability",
    "direction_probability",
    "up_probability",
]

SIGNAL_ROW_COLUMNS = [
    "ticker",
    "prediction_date",
    "target_date",
    "model_name",
    "horizon",
    "predicted_return",
    "realized_forward_return",
    "predicted_direction",
    "probability_up",
    "policy",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "estimated_round_trip_cost",
    "success_definition",
    "target_return_threshold",
    "signal",
    "success_condition_met",
    "buy_success",
    "raw_win",
    "net_realized_return_after_costs",
    "source_predicted_return_column",
    "source_realized_return_column",
]

SUMMARY_COLUMNS = [
    "model_name",
    "horizon",
    "policy",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "estimated_round_trip_cost",
    "success_definition",
    "target_return_threshold",
    "minimum_signal_count",
    "passes_minimum_signal_count",
    "signal_count",
    "buy_signal_count",
    "hold_signal_count",
    "avoid_signal_count",
    "buy_precision",
    "buy_recall",
    "average_realized_return_after_buy",
    "median_realized_return_after_buy",
    "win_rate_after_buy",
    "gross_average_return_after_buy",
    "net_average_return_after_buy",
    "cumulative_simple_signal_return",
    "hit_rate",
    "profit_factor",
    "max_drawdown",
    "turnover_proxy",
    "buy_coverage",
]

FRONTIER_COLUMNS = [
    "model_name",
    "horizon",
    "policy",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "minimum_signal_count",
    "passes_minimum_signal_count",
    "signal_count",
    "buy_signal_count",
    "buy_coverage",
    "buy_precision",
    "buy_recall",
    "net_average_return_after_buy",
    "cumulative_simple_signal_return",
]

STRATEGY_PROXY_COLUMNS = [
    "model_name",
    "horizon",
    "policy",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "minimum_signal_count",
    "passes_minimum_signal_count",
    "gross_average_return_after_buy",
    "net_average_return_after_buy",
    "cumulative_simple_signal_return",
    "hit_rate",
    "profit_factor",
    "max_drawdown",
    "turnover_proxy",
    "buy_signal_count",
    "hold_signal_count",
    "avoid_signal_count",
]

BENCHMARK_COLUMNS = [
    "model_name",
    "horizon",
    "policy",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "minimum_signal_count",
    "passes_minimum_signal_count",
    "buy_signal_count",
    "benchmark_signal_count",
    "buy_average_realized_return",
    "benchmark_average_realized_return",
    "buy_win_rate",
    "benchmark_win_rate",
    "buy_success_rate",
    "benchmark_success_rate",
    "buy_precision_lift_over_benchmark",
]

SELECTED_THRESHOLD_COLUMNS = [
    "model_name",
    "horizon",
    "ticker_scope",
    "policy",
    "success_definition",
    "selected_predicted_return_threshold",
    "selected_probability_up_threshold",
    "selected_cost_per_trade",
    "selected_slippage",
    "selection_buy_precision",
    "selection_buy_count",
    "selection_net_avg_return",
    "selection_minimum_signal_count",
    "selection_buy_recall",
    "selection_signal_count",
    "selection_hold_count",
    "selection_avoid_count",
]

THRESHOLD_SELECTION_TRACE_COLUMNS = [
    "model_name",
    "horizon",
    "ticker_scope",
    "policy",
    "success_definition",
    "predicted_return_threshold",
    "probability_up_threshold",
    "cost_per_trade",
    "slippage",
    "minimum_signal_count",
    "passes_minimum_signal_count",
    "selection_candidate",
    "selection_rank",
    "selected",
    "buy_precision",
    "buy_signal_count",
    "net_average_return_after_buy",
    "buy_recall",
    "signal_count",
    "hold_signal_count",
    "avoid_signal_count",
]

HELDOUT_BUY_PRECISION_COLUMNS = [
    "model_name",
    "horizon",
    "ticker_scope",
    "policy",
    "success_definition",
    "selected_predicted_return_threshold",
    "selected_probability_up_threshold",
    "selected_cost_per_trade",
    "selected_slippage",
    "minimum_signal_count",
    "heldout_signal_count",
    "heldout_buy_precision",
    "heldout_buy_count",
    "heldout_successful_buy_count",
    "heldout_buy_recall",
    "heldout_avg_realized_return_after_buy",
    "heldout_net_avg_return_after_buy",
    "heldout_hit_rate",
    "heldout_profit_factor",
    "heldout_max_drawdown_proxy",
    "heldout_turnover_proxy",
    "heldout_cumulative_simple_signal_return",
]

PRECISION_TARGET_PASS_FAIL_COLUMNS = [
    "model_name",
    "horizon",
    "ticker_scope",
    "policy",
    "success_definition",
    "selected_predicted_return_threshold",
    "selected_probability_up_threshold",
    "selected_cost_per_trade",
    "selected_slippage",
    "minimum_signal_count",
    "precision_target",
    "heldout_buy_precision",
    "heldout_buy_count",
    "pass_fail",
]

HELDOUT_STRATEGY_PROXY_COLUMNS = [
    "model_name",
    "horizon",
    "ticker_scope",
    "policy",
    "success_definition",
    "selected_predicted_return_threshold",
    "selected_probability_up_threshold",
    "selected_cost_per_trade",
    "selected_slippage",
    "minimum_signal_count",
    "heldout_buy_count",
    "heldout_net_avg_return_after_buy",
    "heldout_hit_rate",
    "heldout_profit_factor",
    "heldout_max_drawdown_proxy",
    "heldout_turnover_proxy",
    "heldout_cumulative_simple_signal_return",
]


@dataclass(frozen=True, slots=True)
class PredictionColumnMapping:
    date_column: str
    ticker_column: str
    model_column: str
    predicted_return_column: str
    realized_return_column: str
    predicted_direction_column: str | None = None
    probability_up_column: str | None = None
    horizon_column: str | None = None
    target_date_column: str | None = None
    evaluation_eligible_column: str | None = None
    inferred_horizon: str | None = None


@dataclass(slots=True)
class SignalEffectivenessConfig:
    predictions_path: str | None = None
    output_dir: str = "artifacts/signal_effectiveness"
    evaluation_mode: str = EVALUATION_MODE_FRONTIER
    models: list[str] | None = None
    horizons: list[str] | None = None
    tickers: list[str] | None = None
    policy: str = POLICY_STRICT_BUY_PRECISION_PROBE
    predicted_return_thresholds: list[float] = field(
        default_factory=lambda: list(DEFAULT_PREDICTED_RETURN_THRESHOLDS)
    )
    cost_per_trade_values: list[float] = field(default_factory=lambda: list(DEFAULT_COST_PER_TRADE_VALUES))
    slippage_values: list[float] = field(default_factory=lambda: list(DEFAULT_SLIPPAGE_VALUES))
    probability_up_thresholds: list[float] = field(
        default_factory=lambda: list(DEFAULT_PROBABILITY_UP_THRESHOLDS)
    )
    success_definition: str = SUCCESS_COST_ADJUSTED_POSITIVE
    target_return_threshold: float = 0.01
    minimum_signal_counts: list[int] = field(default_factory=lambda: list(DEFAULT_MINIMUM_SIGNAL_COUNTS))
    selection_start: str | None = None
    selection_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    precision_targets: list[float] = field(default_factory=lambda: list(DEFAULT_PRECISION_TARGETS))
    selection_criterion: str = SELECTION_CRITERION_MAX_PRECISION


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_numeric_values(values: list[float] | tuple[float, ...], *, name: str) -> list[float]:
    result = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} values must be finite")
        if numeric < 0.0:
            raise ValueError(f"{name} values must be non-negative")
        result.append(numeric)
    if not result:
        raise ValueError(f"At least one {name} value is required")
    return sorted(dict.fromkeys(result))


def _stable_int_values(values: list[int] | tuple[int, ...], *, name: str) -> list[int]:
    result = []
    for value in values:
        numeric = int(value)
        if numeric < 0:
            raise ValueError(f"{name} values must be non-negative")
        result.append(numeric)
    if not result:
        raise ValueError(f"At least one {name} value is required")
    return sorted(dict.fromkeys(result))


def _detect_column(columns: pd.Index, candidates: list[str], *, required: bool = True) -> str | None:
    exact = set(columns)
    lower_lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        resolved = lower_lookup.get(candidate.lower())
        if resolved is not None:
            return resolved
    if required:
        raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")
    return None


def _infer_horizon_from_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    csv_path = Path(path)
    parent_name = csv_path.parent.name
    if parent_name and parent_name.lower() not in {"csv", "summary"}:
        return parent_name
    grandparent_name = csv_path.parent.parent.name if csv_path.parent.parent != csv_path.parent else ""
    return grandparent_name or None


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def detect_prediction_columns(frame: pd.DataFrame, *, source_path: str | Path | None = None) -> PredictionColumnMapping:
    horizon_column = "horizon" if "horizon" in frame.columns else None
    target_date_column = "target_date" if "target_date" in frame.columns else None
    evaluation_eligible_column = "evaluation_eligible" if "evaluation_eligible" in frame.columns else None
    probability_column = _detect_column(frame.columns, PROBABILITY_UP_CANDIDATES, required=False)
    if probability_column is not None and pd.to_numeric(frame[probability_column], errors="coerce").notna().sum() == 0:
        probability_column = None

    return PredictionColumnMapping(
        date_column=str(_detect_column(frame.columns, DATE_CANDIDATES)),
        ticker_column=str(_detect_column(frame.columns, TICKER_CANDIDATES)),
        model_column=str(_detect_column(frame.columns, MODEL_CANDIDATES)),
        predicted_return_column=str(_detect_column(frame.columns, PREDICTED_RETURN_CANDIDATES)),
        realized_return_column=str(_detect_column(frame.columns, REALIZED_RETURN_CANDIDATES)),
        predicted_direction_column=_detect_column(frame.columns, PREDICTED_DIRECTION_CANDIDATES, required=False),
        probability_up_column=probability_column,
        horizon_column=horizon_column,
        target_date_column=target_date_column,
        evaluation_eligible_column=evaluation_eligible_column,
        inferred_horizon=_infer_horizon_from_path(source_path),
    )


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return pd.DataFrame(columns=[
            "ticker",
            "prediction_date",
            "target_date",
            "model_name",
            "horizon",
            "predicted_return",
            "realized_forward_return",
            "predicted_direction",
            "probability_up",
            "evaluation_eligible",
        ]), {"input_rows": 0, "dropped_rows": 0, "column_mapping": None}

    mapping = detect_prediction_columns(frame, source_path=source_path)
    working = pd.DataFrame(
        {
            "ticker": frame[mapping.ticker_column].astype(str).str.upper().str.strip(),
            "prediction_date": pd.to_datetime(frame[mapping.date_column], errors="coerce").dt.normalize(),
            "model_name": frame[mapping.model_column].astype(str).str.strip(),
            "predicted_return": pd.to_numeric(frame[mapping.predicted_return_column], errors="coerce"),
            "realized_forward_return": pd.to_numeric(frame[mapping.realized_return_column], errors="coerce"),
        }
    )
    if mapping.horizon_column is not None:
        working["horizon"] = frame[mapping.horizon_column].astype(str).str.strip()
    else:
        working["horizon"] = str(mapping.inferred_horizon or "unknown")

    if mapping.target_date_column is not None:
        working["target_date"] = pd.to_datetime(frame[mapping.target_date_column], errors="coerce").dt.normalize()
    else:
        working["target_date"] = pd.NaT

    if mapping.predicted_direction_column is not None:
        working["predicted_direction"] = pd.to_numeric(frame[mapping.predicted_direction_column], errors="coerce")
        predicted_direction_source = mapping.predicted_direction_column
    else:
        working["predicted_direction"] = (working["predicted_return"] > 0.0).astype(float)
        predicted_direction_source = "derived_from_predicted_return"

    if mapping.probability_up_column is not None:
        working["probability_up"] = pd.to_numeric(frame[mapping.probability_up_column], errors="coerce")
    else:
        working["probability_up"] = np.nan

    if mapping.evaluation_eligible_column is not None:
        working["evaluation_eligible"] = _bool_series(frame[mapping.evaluation_eligible_column])
    else:
        working["evaluation_eligible"] = True

    working["source_predicted_return_column"] = mapping.predicted_return_column
    working["source_realized_return_column"] = mapping.realized_return_column
    before = int(len(working))
    working = working[
        (working["evaluation_eligible"] == True)
        & working["ticker"].ne("")
        & working["model_name"].ne("")
        & working["horizon"].ne("")
        & working["prediction_date"].notna()
        & working["predicted_return"].notna()
        & working["realized_forward_return"].notna()
    ].copy()
    working = working.sort_values(["model_name", "horizon", "ticker", "prediction_date"]).reset_index(drop=True)
    metadata = {
        "input_rows": before,
        "normalized_rows": int(len(working)),
        "dropped_rows": int(before - len(working)),
        "column_mapping": asdict(mapping),
        "predicted_direction_source": predicted_direction_source,
        "probability_up_available": bool(working["probability_up"].notna().any()),
        "realized_return_used_only_for_evaluation": True,
    }
    return working, metadata


def load_prediction_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    return normalize_prediction_frame(frame, source_path=csv_path)


def _success_condition(
    realized_return: pd.Series,
    *,
    success_definition: str,
    estimated_round_trip_cost: float,
    target_return_threshold: float,
) -> pd.Series:
    realized = pd.to_numeric(realized_return, errors="coerce")
    if success_definition == SUCCESS_RAW_POSITIVE:
        return realized > 0.0
    if success_definition == SUCCESS_COST_ADJUSTED_POSITIVE:
        return realized > float(estimated_round_trip_cost)
    if success_definition == SUCCESS_TARGET_RETURN:
        return realized >= float(target_return_threshold)
    raise ValueError(
        f"Unsupported success_definition={success_definition!r}. "
        f"Supported: {sorted(SUPPORTED_SUCCESS_DEFINITIONS)}"
    )


def generate_signal_rows(
    predictions: pd.DataFrame,
    *,
    policy: str,
    predicted_return_threshold: float,
    cost_per_trade: float,
    slippage: float,
    success_definition: str = SUCCESS_COST_ADJUSTED_POSITIVE,
    target_return_threshold: float = 0.01,
    probability_up_threshold: float | None = None,
) -> pd.DataFrame:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported signal policy={policy!r}. Supported: {sorted(SUPPORTED_POLICIES)}")

    working = predictions.copy()
    if working.empty:
        return pd.DataFrame(columns=SIGNAL_ROW_COLUMNS)

    threshold = float(predicted_return_threshold)
    if threshold < 0.0:
        raise ValueError("predicted_return_threshold must be non-negative")
    estimated_round_trip_cost = 2.0 * (float(cost_per_trade) + float(slippage))
    predicted = pd.to_numeric(working["predicted_return"], errors="coerce")
    direction = pd.to_numeric(working["predicted_direction"], errors="coerce")

    probability_gate = pd.Series(True, index=working.index)
    if probability_up_threshold is not None:
        probability = pd.to_numeric(working["probability_up"], errors="coerce")
        probability_gate = probability >= float(probability_up_threshold)

    if policy == POLICY_DIRECTION_AND_RETURN_THRESHOLD:
        buy_mask = (predicted >= threshold) & (direction > 0.0) & probability_gate
    else:
        buy_mask = (predicted >= threshold) & probability_gate

    avoid_mask = predicted <= -threshold
    signal = pd.Series(SIGNAL_HOLD, index=working.index, dtype=object)
    signal.loc[avoid_mask] = SIGNAL_AVOID
    signal.loc[buy_mask] = SIGNAL_BUY

    success = _success_condition(
        working["realized_forward_return"],
        success_definition=success_definition,
        estimated_round_trip_cost=estimated_round_trip_cost,
        target_return_threshold=target_return_threshold,
    )
    working["policy"] = policy
    working["predicted_return_threshold"] = threshold
    working["probability_up_threshold"] = np.nan if probability_up_threshold is None else float(probability_up_threshold)
    working["cost_per_trade"] = float(cost_per_trade)
    working["slippage"] = float(slippage)
    working["estimated_round_trip_cost"] = estimated_round_trip_cost
    working["success_definition"] = success_definition
    working["target_return_threshold"] = float(target_return_threshold)
    working["signal"] = signal
    working["success_condition_met"] = success.astype(bool)
    working["buy_success"] = (working["signal"] == SIGNAL_BUY) & working["success_condition_met"]
    working["raw_win"] = working["realized_forward_return"] > 0.0
    working["net_realized_return_after_costs"] = (
        pd.to_numeric(working["realized_forward_return"], errors="coerce") - estimated_round_trip_cost
    )

    return working.reindex(columns=SIGNAL_ROW_COLUMNS).sort_values(
        [
            "policy",
            "predicted_return_threshold",
            "probability_up_threshold",
            "cost_per_trade",
            "slippage",
            "model_name",
            "horizon",
            "ticker",
            "prediction_date",
        ],
        na_position="first",
    ).reset_index(drop=True)


def _cumulative_simple_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0
    if (clean <= -1.0).any():
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def _profit_factor(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0
    gains = float(clean[clean > 0.0].sum())
    losses = float(-clean[clean < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def _max_drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty or (clean <= -1.0).any():
        return 0.0 if clean.empty else float("nan")
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _metrics_for_group(group: pd.DataFrame, minimum_signal_count: int) -> dict[str, Any]:
    buy = group[group["signal"] == SIGNAL_BUY].copy()
    hold_count = int((group["signal"] == SIGNAL_HOLD).sum())
    avoid_count = int((group["signal"] == SIGNAL_AVOID).sum())
    buy_count = int(len(buy))
    total = int(len(group))
    success_count = int(group["success_condition_met"].sum())
    correct_buy = int(buy["success_condition_met"].sum()) if buy_count else 0
    buy_net = pd.to_numeric(buy["net_realized_return_after_costs"], errors="coerce")
    buy_gross = pd.to_numeric(buy["realized_forward_return"], errors="coerce")

    return {
        "model_name": str(group["model_name"].iloc[0]),
        "horizon": str(group["horizon"].iloc[0]),
        "policy": str(group["policy"].iloc[0]),
        "predicted_return_threshold": float(group["predicted_return_threshold"].iloc[0]),
        "probability_up_threshold": (
            float(group["probability_up_threshold"].iloc[0])
            if pd.notna(group["probability_up_threshold"].iloc[0])
            else np.nan
        ),
        "cost_per_trade": float(group["cost_per_trade"].iloc[0]),
        "slippage": float(group["slippage"].iloc[0]),
        "estimated_round_trip_cost": float(group["estimated_round_trip_cost"].iloc[0]),
        "success_definition": str(group["success_definition"].iloc[0]),
        "target_return_threshold": float(group["target_return_threshold"].iloc[0]),
        "minimum_signal_count": int(minimum_signal_count),
        "passes_minimum_signal_count": bool(buy_count >= int(minimum_signal_count)),
        "signal_count": total,
        "buy_signal_count": buy_count,
        "hold_signal_count": hold_count,
        "avoid_signal_count": avoid_count,
        "buy_precision": float(correct_buy / buy_count) if buy_count else np.nan,
        "buy_recall": float(correct_buy / success_count) if success_count else np.nan,
        "average_realized_return_after_buy": float(buy_gross.mean()) if buy_count else np.nan,
        "median_realized_return_after_buy": float(buy_gross.median()) if buy_count else np.nan,
        "win_rate_after_buy": float((buy_gross > 0.0).mean()) if buy_count else np.nan,
        "gross_average_return_after_buy": float(buy_gross.mean()) if buy_count else np.nan,
        "net_average_return_after_buy": float(buy_net.mean()) if buy_count else np.nan,
        "cumulative_simple_signal_return": _cumulative_simple_return(buy_net),
        "hit_rate": float(buy["success_condition_met"].mean()) if buy_count else np.nan,
        "profit_factor": _profit_factor(buy_net),
        "max_drawdown": _max_drawdown(buy_net),
        "turnover_proxy": float(buy_count / total) if total else np.nan,
        "buy_coverage": float(buy_count / total) if total else np.nan,
    }


def summarize_signal_effectiveness(
    signal_rows: pd.DataFrame,
    *,
    minimum_signal_counts: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    if signal_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    min_counts = _stable_int_values(list(minimum_signal_counts), name="minimum_signal_count")
    group_columns = [
        "model_name",
        "horizon",
        "policy",
        "predicted_return_threshold",
        "probability_up_threshold",
        "cost_per_trade",
        "slippage",
        "success_definition",
        "target_return_threshold",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in signal_rows.groupby(group_columns, dropna=False, sort=True):
        ordered = group.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)
        for minimum_signal_count in min_counts:
            rows.append(_metrics_for_group(ordered, minimum_signal_count))
    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS).sort_values(
        [
            "model_name",
            "horizon",
            "policy",
            "predicted_return_threshold",
            "probability_up_threshold",
            "cost_per_trade",
            "slippage",
            "minimum_signal_count",
        ],
        na_position="first",
    ).reset_index(drop=True)


def build_precision_coverage_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    return summary.reindex(columns=FRONTIER_COLUMNS).copy().reset_index(drop=True)


def build_strategy_proxy_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=STRATEGY_PROXY_COLUMNS)
    return summary.reindex(columns=STRATEGY_PROXY_COLUMNS).copy().reset_index(drop=True)


def build_buy_precision_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    filtered = summary[summary["passes_minimum_signal_count"] == True].copy()
    return filtered.reindex(columns=SUMMARY_COLUMNS).reset_index(drop=True)


def build_benchmark_comparison(signal_rows: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if signal_rows.empty or summary.empty:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)

    group_columns = [
        "model_name",
        "horizon",
        "policy",
        "predicted_return_threshold",
        "probability_up_threshold",
        "cost_per_trade",
        "slippage",
        "success_definition",
        "target_return_threshold",
    ]
    baseline_rows: list[dict[str, Any]] = []
    for _, group in signal_rows.groupby(group_columns, dropna=False, sort=True):
        realized = pd.to_numeric(group["realized_forward_return"], errors="coerce")
        buy = group[group["signal"] == SIGNAL_BUY].copy()
        buy_realized = pd.to_numeric(buy["realized_forward_return"], errors="coerce")
        buy_success_rate = float(buy["success_condition_met"].mean()) if not buy.empty else np.nan
        benchmark_success_rate = float(group["success_condition_met"].mean()) if not group.empty else np.nan
        baseline_rows.append(
            {
                "model_name": str(group["model_name"].iloc[0]),
                "horizon": str(group["horizon"].iloc[0]),
                "policy": str(group["policy"].iloc[0]),
                "predicted_return_threshold": float(group["predicted_return_threshold"].iloc[0]),
                "probability_up_threshold": (
                    float(group["probability_up_threshold"].iloc[0])
                    if pd.notna(group["probability_up_threshold"].iloc[0])
                    else np.nan
                ),
                "cost_per_trade": float(group["cost_per_trade"].iloc[0]),
                "slippage": float(group["slippage"].iloc[0]),
                "success_definition": str(group["success_definition"].iloc[0]),
                "target_return_threshold": float(group["target_return_threshold"].iloc[0]),
                "buy_signal_count": int(len(buy)),
                "benchmark_signal_count": int(len(group)),
                "buy_average_realized_return": float(buy_realized.mean()) if not buy.empty else np.nan,
                "benchmark_average_realized_return": float(realized.mean()) if not group.empty else np.nan,
                "buy_win_rate": float((buy_realized > 0.0).mean()) if not buy.empty else np.nan,
                "benchmark_win_rate": float((realized > 0.0).mean()) if not group.empty else np.nan,
                "buy_success_rate": buy_success_rate,
                "benchmark_success_rate": benchmark_success_rate,
                "buy_precision_lift_over_benchmark": (
                    buy_success_rate - benchmark_success_rate
                    if not pd.isna(buy_success_rate) and not pd.isna(benchmark_success_rate)
                    else np.nan
                ),
            }
        )
    baseline = pd.DataFrame(baseline_rows)
    if baseline.empty:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    join_columns = [
        "model_name",
        "horizon",
        "policy",
        "predicted_return_threshold",
        "probability_up_threshold",
        "cost_per_trade",
        "slippage",
        "success_definition",
        "target_return_threshold",
        "buy_signal_count",
    ]
    comparison = summary[
        [
            *join_columns,
            "minimum_signal_count",
            "passes_minimum_signal_count",
        ]
    ].merge(baseline, on=join_columns, how="left")
    return comparison.reindex(columns=BENCHMARK_COLUMNS).reset_index(drop=True)


def _filter_values(frame: pd.DataFrame, column: str, values: list[str] | None, *, upper: bool = False) -> pd.DataFrame:
    if not values:
        return frame
    requested = {str(value).upper().strip() if upper else str(value).strip() for value in values if str(value).strip()}
    if not requested:
        return frame
    series = frame[column].astype(str).str.upper() if upper else frame[column].astype(str)
    return frame[series.isin(requested)].copy()


def _parse_required_date(value: str | None, *, name: str) -> pd.Timestamp:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} is required for heldout_threshold_selection")
    parsed = pd.Timestamp(value).normalize()
    if pd.isna(parsed):
        raise ValueError(f"{name} must be a valid date")
    return parsed


def _filter_prediction_date_window(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if end < start:
        raise ValueError(f"Date window end {end.date()} is earlier than start {start.date()}")
    dates = pd.to_datetime(frame["prediction_date"], errors="coerce").dt.normalize()
    return frame[(dates >= start) & (dates <= end)].copy().reset_index(drop=True)


def _float_or_nan(value: Any) -> float:
    return float(value) if pd.notna(value) else np.nan


def _selected_thresholds_from_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame(columns=SELECTED_THRESHOLD_COLUMNS)
    selected = trace[trace["selected"] == True].copy()
    if selected.empty:
        return pd.DataFrame(columns=SELECTED_THRESHOLD_COLUMNS)
    selected["ticker_scope"] = "pooled"
    selected = selected.rename(
        columns={
            "predicted_return_threshold": "selected_predicted_return_threshold",
            "probability_up_threshold": "selected_probability_up_threshold",
            "cost_per_trade": "selected_cost_per_trade",
            "slippage": "selected_slippage",
            "buy_precision": "selection_buy_precision",
            "buy_signal_count": "selection_buy_count",
            "net_average_return_after_buy": "selection_net_avg_return",
            "minimum_signal_count": "selection_minimum_signal_count",
            "buy_recall": "selection_buy_recall",
            "signal_count": "selection_signal_count",
            "hold_signal_count": "selection_hold_count",
            "avoid_signal_count": "selection_avoid_count",
        }
    )
    return selected.reindex(columns=SELECTED_THRESHOLD_COLUMNS).sort_values(
        ["model_name", "horizon", "selection_minimum_signal_count"]
    ).reset_index(drop=True)


def select_thresholds_from_summary(
    summary: pd.DataFrame,
    *,
    selection_criterion: str = SELECTION_CRITERION_MAX_PRECISION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one threshold per model/horizon/minimum-count using selection-period metrics."""
    if selection_criterion != SELECTION_CRITERION_MAX_PRECISION:
        raise ValueError(
            f"Unsupported selection_criterion={selection_criterion!r}. "
            f"Supported: {sorted(SUPPORTED_SELECTION_CRITERIA)}"
        )
    if summary.empty:
        empty_trace = pd.DataFrame(columns=THRESHOLD_SELECTION_TRACE_COLUMNS)
        return pd.DataFrame(columns=SELECTED_THRESHOLD_COLUMNS), empty_trace

    trace = summary.copy()
    trace["ticker_scope"] = "pooled"
    trace["selection_candidate"] = (
        (trace["passes_minimum_signal_count"] == True)
        & pd.to_numeric(trace["buy_signal_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(trace["buy_precision"], errors="coerce").notna()
    )
    trace["selection_rank"] = np.nan
    trace["selected"] = False

    rank_group_columns = ["model_name", "horizon", "minimum_signal_count"]
    for _, group in trace.groupby(rank_group_columns, dropna=False, sort=True):
        candidate_index = group[group["selection_candidate"] == True].index
        if len(candidate_index) == 0:
            continue
        candidates = trace.loc[candidate_index].copy()
        candidates["_sort_precision"] = pd.to_numeric(candidates["buy_precision"], errors="coerce").fillna(-np.inf)
        candidates["_sort_net"] = pd.to_numeric(candidates["net_average_return_after_buy"], errors="coerce").fillna(-np.inf)
        candidates["_sort_buy_count"] = pd.to_numeric(candidates["buy_signal_count"], errors="coerce").fillna(-1)
        candidates["_sort_threshold"] = pd.to_numeric(candidates["predicted_return_threshold"], errors="coerce").fillna(np.inf)
        candidates["_sort_cost"] = pd.to_numeric(candidates["cost_per_trade"], errors="coerce").fillna(np.inf)
        candidates["_sort_slippage"] = pd.to_numeric(candidates["slippage"], errors="coerce").fillna(np.inf)
        ordered = candidates.sort_values(
            [
                "_sort_precision",
                "_sort_net",
                "_sort_buy_count",
                "_sort_threshold",
                "_sort_cost",
                "_sort_slippage",
            ],
            ascending=[False, False, False, True, True, True],
        )
        trace.loc[ordered.index, "selection_rank"] = list(range(1, len(ordered) + 1))
        trace.loc[ordered.index[0], "selected"] = True

    trace = trace.reindex(columns=THRESHOLD_SELECTION_TRACE_COLUMNS).sort_values(
        [
            "model_name",
            "horizon",
            "minimum_signal_count",
            "selection_candidate",
            "selection_rank",
            "predicted_return_threshold",
            "cost_per_trade",
            "slippage",
        ],
        ascending=[True, True, True, False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return _selected_thresholds_from_trace(trace), trace


def _empty_heldout_metric_row(selected_row: pd.Series) -> dict[str, Any]:
    return {
        "model_name": str(selected_row["model_name"]),
        "horizon": str(selected_row["horizon"]),
        "ticker_scope": str(selected_row["ticker_scope"]),
        "policy": str(selected_row["policy"]),
        "success_definition": str(selected_row["success_definition"]),
        "selected_predicted_return_threshold": _float_or_nan(selected_row["selected_predicted_return_threshold"]),
        "selected_probability_up_threshold": _float_or_nan(selected_row["selected_probability_up_threshold"]),
        "selected_cost_per_trade": _float_or_nan(selected_row["selected_cost_per_trade"]),
        "selected_slippage": _float_or_nan(selected_row["selected_slippage"]),
        "minimum_signal_count": int(selected_row["selection_minimum_signal_count"]),
        "heldout_signal_count": 0,
        "heldout_buy_precision": np.nan,
        "heldout_buy_count": 0,
        "heldout_successful_buy_count": 0,
        "heldout_buy_recall": np.nan,
        "heldout_avg_realized_return_after_buy": np.nan,
        "heldout_net_avg_return_after_buy": np.nan,
        "heldout_hit_rate": np.nan,
        "heldout_profit_factor": 0.0,
        "heldout_max_drawdown_proxy": 0.0,
        "heldout_turnover_proxy": np.nan,
        "heldout_cumulative_simple_signal_return": 0.0,
    }


def _heldout_metric_row(selected_row: pd.Series, signal_rows: pd.DataFrame) -> dict[str, Any]:
    if signal_rows.empty:
        return _empty_heldout_metric_row(selected_row)
    metrics = _metrics_for_group(
        signal_rows.sort_values(["prediction_date", "ticker"]).reset_index(drop=True),
        int(selected_row["selection_minimum_signal_count"]),
    )
    buy = signal_rows[signal_rows["signal"] == SIGNAL_BUY]
    return {
        "model_name": str(selected_row["model_name"]),
        "horizon": str(selected_row["horizon"]),
        "ticker_scope": str(selected_row["ticker_scope"]),
        "policy": str(selected_row["policy"]),
        "success_definition": str(selected_row["success_definition"]),
        "selected_predicted_return_threshold": _float_or_nan(selected_row["selected_predicted_return_threshold"]),
        "selected_probability_up_threshold": _float_or_nan(selected_row["selected_probability_up_threshold"]),
        "selected_cost_per_trade": _float_or_nan(selected_row["selected_cost_per_trade"]),
        "selected_slippage": _float_or_nan(selected_row["selected_slippage"]),
        "minimum_signal_count": int(selected_row["selection_minimum_signal_count"]),
        "heldout_signal_count": int(metrics["signal_count"]),
        "heldout_buy_precision": metrics["buy_precision"],
        "heldout_buy_count": int(metrics["buy_signal_count"]),
        "heldout_successful_buy_count": int(buy["success_condition_met"].sum()) if not buy.empty else 0,
        "heldout_buy_recall": metrics["buy_recall"],
        "heldout_avg_realized_return_after_buy": metrics["average_realized_return_after_buy"],
        "heldout_net_avg_return_after_buy": metrics["net_average_return_after_buy"],
        "heldout_hit_rate": metrics["hit_rate"],
        "heldout_profit_factor": metrics["profit_factor"],
        "heldout_max_drawdown_proxy": metrics["max_drawdown"],
        "heldout_turnover_proxy": metrics["turnover_proxy"],
        "heldout_cumulative_simple_signal_return": metrics["cumulative_simple_signal_return"],
    }


def build_precision_target_pass_fail(
    heldout_precision: pd.DataFrame,
    *,
    precision_targets: list[float] | tuple[float, ...],
) -> pd.DataFrame:
    if heldout_precision.empty:
        return pd.DataFrame(columns=PRECISION_TARGET_PASS_FAIL_COLUMNS)
    rows: list[dict[str, Any]] = []
    for heldout_row in heldout_precision.itertuples(index=False):
        precision = getattr(heldout_row, "heldout_buy_precision")
        buy_count = int(getattr(heldout_row, "heldout_buy_count"))
        for target in precision_targets:
            numeric_target = float(target)
            rows.append(
                {
                    "model_name": heldout_row.model_name,
                    "horizon": heldout_row.horizon,
                    "ticker_scope": heldout_row.ticker_scope,
                    "policy": heldout_row.policy,
                    "success_definition": heldout_row.success_definition,
                    "selected_predicted_return_threshold": heldout_row.selected_predicted_return_threshold,
                    "selected_probability_up_threshold": heldout_row.selected_probability_up_threshold,
                    "selected_cost_per_trade": heldout_row.selected_cost_per_trade,
                    "selected_slippage": heldout_row.selected_slippage,
                    "minimum_signal_count": int(heldout_row.minimum_signal_count),
                    "precision_target": numeric_target,
                    "heldout_buy_precision": precision,
                    "heldout_buy_count": buy_count,
                    "pass_fail": bool(buy_count > 0 and pd.notna(precision) and float(precision) >= numeric_target),
                }
            )
    return pd.DataFrame(rows).reindex(columns=PRECISION_TARGET_PASS_FAIL_COLUMNS).reset_index(drop=True)


class SignalEffectivenessRunner:
    """Run signal-effectiveness diagnostics on a saved prediction table."""

    def __init__(self, config: SignalEffectivenessConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.evaluation_mode not in SUPPORTED_EVALUATION_MODES:
            raise ValueError(
                f"Unsupported evaluation_mode={self.config.evaluation_mode!r}. "
                f"Supported: {sorted(SUPPORTED_EVALUATION_MODES)}"
            )
        if self.config.policy not in SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported policy={self.config.policy!r}. Supported: {sorted(SUPPORTED_POLICIES)}")
        if self.config.success_definition not in SUPPORTED_SUCCESS_DEFINITIONS:
            raise ValueError(
                f"Unsupported success_definition={self.config.success_definition!r}. "
                f"Supported: {sorted(SUPPORTED_SUCCESS_DEFINITIONS)}"
            )
        self.config.predicted_return_thresholds = _stable_numeric_values(
            self.config.predicted_return_thresholds,
            name="predicted_return_threshold",
        )
        self.config.cost_per_trade_values = _stable_numeric_values(
            self.config.cost_per_trade_values,
            name="cost_per_trade",
        )
        self.config.slippage_values = _stable_numeric_values(self.config.slippage_values, name="slippage")
        self.config.probability_up_thresholds = _stable_numeric_values(
            self.config.probability_up_thresholds,
            name="probability_up_threshold",
        )
        self.config.minimum_signal_counts = _stable_int_values(
            self.config.minimum_signal_counts,
            name="minimum_signal_count",
        )
        self.config.precision_targets = _stable_numeric_values(
            self.config.precision_targets,
            name="precision_target",
        )
        for target in self.config.precision_targets:
            if target > 1.0:
                raise ValueError("precision_target values must be between 0 and 1")
        if self.config.selection_criterion not in SUPPORTED_SELECTION_CRITERIA:
            raise ValueError(
                f"Unsupported selection_criterion={self.config.selection_criterion!r}. "
                f"Supported: {sorted(SUPPORTED_SELECTION_CRITERIA)}"
            )
        if float(self.config.target_return_threshold) < 0.0:
            raise ValueError("target_return_threshold must be non-negative")
        if self.config.evaluation_mode == EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION:
            selection_start = _parse_required_date(self.config.selection_start, name="selection_start")
            selection_end = _parse_required_date(self.config.selection_end, name="selection_end")
            test_start = _parse_required_date(self.config.test_start, name="test_start")
            test_end = _parse_required_date(self.config.test_end, name="test_end")
            if selection_end < selection_start:
                raise ValueError("selection_end must be on or after selection_start")
            if test_end < test_start:
                raise ValueError("test_end must be on or after test_start")
            if test_start <= selection_end:
                raise ValueError("test_start must be later than selection_end for held-out evaluation")

    def _load_predictions(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self.config.predictions_path is None:
            raise ValueError("predictions_path is required when no DataFrame is supplied")
        return load_prediction_csv(self.config.predictions_path)

    def _apply_filters(self, predictions: pd.DataFrame) -> pd.DataFrame:
        filtered = predictions.copy()
        filtered = _filter_values(filtered, "model_name", self.config.models)
        filtered = _filter_values(filtered, "horizon", self.config.horizons)
        filtered = _filter_values(filtered, "ticker", self.config.tickers, upper=True)
        return filtered.reset_index(drop=True)

    def _probability_thresholds(self, predictions: pd.DataFrame) -> tuple[list[float | None], dict[str, Any]]:
        probability_available = bool(not predictions.empty and predictions["probability_up"].notna().any())
        if probability_available:
            return list(self.config.probability_up_thresholds), {
                "probability_up_available": True,
                "probability_up_thresholds_used": list(self.config.probability_up_thresholds),
                "probability_rules_skipped_reason": None,
            }
        return [None], {
            "probability_up_available": False,
            "probability_up_thresholds_used": [],
            "probability_rules_skipped_reason": (
                "No usable upward probability column was found; probability calibration is a future task."
            ),
        }

    def _build_signal_grid(self, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        probability_thresholds, probability_metadata = self._probability_thresholds(predictions)
        frames: list[pd.DataFrame] = []
        for threshold in self.config.predicted_return_thresholds:
            for cost_per_trade in self.config.cost_per_trade_values:
                for slippage in self.config.slippage_values:
                    for probability_threshold in probability_thresholds:
                        frames.append(
                            generate_signal_rows(
                                predictions,
                                policy=self.config.policy,
                                predicted_return_threshold=threshold,
                                cost_per_trade=cost_per_trade,
                                slippage=slippage,
                                success_definition=self.config.success_definition,
                                target_return_threshold=self.config.target_return_threshold,
                                probability_up_threshold=probability_threshold,
                            )
                        )
        if not frames:
            return pd.DataFrame(columns=SIGNAL_ROW_COLUMNS), probability_metadata
        return pd.concat(frames, ignore_index=True).reindex(columns=SIGNAL_ROW_COLUMNS), probability_metadata

    def _load_filtered_predictions(self, predictions: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
        if predictions is None:
            normalized, load_metadata = self._load_predictions()
        else:
            normalized, load_metadata = normalize_prediction_frame(predictions)
        return self._apply_filters(normalized), load_metadata

    def _heldout_windows(self, filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
        selection_start = _parse_required_date(self.config.selection_start, name="selection_start")
        selection_end = _parse_required_date(self.config.selection_end, name="selection_end")
        test_start = _parse_required_date(self.config.test_start, name="test_start")
        test_end = _parse_required_date(self.config.test_end, name="test_end")
        selection = _filter_prediction_date_window(filtered, start=selection_start, end=selection_end)
        test = _filter_prediction_date_window(filtered, start=test_start, end=test_end)
        return selection, test, {
            "selection_start": str(selection_start.date()),
            "selection_end": str(selection_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        }

    def _build_heldout_outputs(
        self,
        *,
        selection_predictions: pd.DataFrame,
        test_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        selection_signal_rows, probability_metadata = self._build_signal_grid(selection_predictions)
        selection_summary = summarize_signal_effectiveness(
            selection_signal_rows,
            minimum_signal_counts=self.config.minimum_signal_counts,
        )
        selected_thresholds, trace = select_thresholds_from_summary(
            selection_summary,
            selection_criterion=self.config.selection_criterion,
        )
        heldout_frames: list[pd.DataFrame] = []
        heldout_metric_rows: list[dict[str, Any]] = []

        for selected_row in selected_thresholds.itertuples(index=False):
            group = test_predictions[
                (test_predictions["model_name"] == selected_row.model_name)
                & (test_predictions["horizon"] == selected_row.horizon)
            ].copy()
            probability_threshold = (
                None
                if pd.isna(selected_row.selected_probability_up_threshold)
                else float(selected_row.selected_probability_up_threshold)
            )
            heldout_signals = generate_signal_rows(
                group,
                policy=str(selected_row.policy),
                predicted_return_threshold=float(selected_row.selected_predicted_return_threshold),
                cost_per_trade=float(selected_row.selected_cost_per_trade),
                slippage=float(selected_row.selected_slippage),
                success_definition=str(selected_row.success_definition),
                target_return_threshold=float(self.config.target_return_threshold),
                probability_up_threshold=probability_threshold,
            )
            if not heldout_signals.empty:
                heldout_signals["ticker_scope"] = str(selected_row.ticker_scope)
                heldout_signals["selection_minimum_signal_count"] = int(selected_row.selection_minimum_signal_count)
                heldout_frames.append(heldout_signals)
            heldout_metric_rows.append(_heldout_metric_row(pd.Series(selected_row._asdict()), heldout_signals))

        heldout_signal_rows = (
            pd.concat(heldout_frames, ignore_index=True, sort=False)
            if heldout_frames
            else pd.DataFrame(columns=[*SIGNAL_ROW_COLUMNS, "ticker_scope", "selection_minimum_signal_count"])
        )
        heldout_precision = pd.DataFrame(heldout_metric_rows).reindex(columns=HELDOUT_BUY_PRECISION_COLUMNS)
        target_pass_fail = build_precision_target_pass_fail(
            heldout_precision,
            precision_targets=self.config.precision_targets,
        )
        strategy_proxy = heldout_precision.reindex(columns=HELDOUT_STRATEGY_PROXY_COLUMNS).copy()
        metadata = {
            "probability_rules": probability_metadata,
            "selection_rows": int(len(selection_predictions)),
            "selection_signal_rows": int(len(selection_signal_rows)),
            "test_rows": int(len(test_predictions)),
            "heldout_signal_rows": int(len(heldout_signal_rows)),
            "selected_threshold_count": int(len(selected_thresholds)),
            "selection_criterion": self.config.selection_criterion,
            "threshold_selection_rule": (
                "maximize BUY precision subject to minimum BUY signal count; "
                "tie-break by higher net average return, then larger BUY signal count"
            ),
            "heldout_realized_returns_used_for_threshold_selection": False,
        }
        return (
            selected_thresholds,
            heldout_precision,
            trace,
            heldout_signal_rows,
            target_pass_fail,
            strategy_proxy,
            metadata,
        )

    def run(self, predictions: pd.DataFrame | None = None) -> dict[str, Any]:
        if self.config.evaluation_mode == EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION:
            return self.run_heldout_threshold_selection(predictions)

        filtered, load_metadata = self._load_filtered_predictions(predictions)
        signal_rows, probability_metadata = self._build_signal_grid(filtered)
        summary = summarize_signal_effectiveness(
            signal_rows,
            minimum_signal_counts=self.config.minimum_signal_counts,
        )
        buy_precision = build_buy_precision_table(summary)
        frontier = build_precision_coverage_frontier(summary)
        strategy_metrics = build_strategy_proxy_metrics(summary)
        benchmark = build_benchmark_comparison(signal_rows, summary)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "signal_rows": self.output_dir / "signal_rows.csv",
            "buy_precision_by_model_horizon": self.output_dir / "buy_precision_by_model_horizon.csv",
            "precision_coverage_frontier": self.output_dir / "precision_coverage_frontier.csv",
            "signal_effectiveness_summary": self.output_dir / "signal_effectiveness_summary.csv",
            "strategy_proxy_metrics": self.output_dir / "strategy_proxy_metrics.csv",
            "benchmark_comparison": self.output_dir / "benchmark_comparison.csv",
            "run_metadata": self.output_dir / "run_metadata.json",
        }
        signal_rows.to_csv(paths["signal_rows"], index=False)
        buy_precision.to_csv(paths["buy_precision_by_model_horizon"], index=False)
        frontier.to_csv(paths["precision_coverage_frontier"], index=False)
        summary.to_csv(paths["signal_effectiveness_summary"], index=False)
        strategy_metrics.to_csv(paths["strategy_proxy_metrics"], index=False)
        benchmark.to_csv(paths["benchmark_comparison"], index=False)

        metadata = {
            "analysis_only": True,
            "live_execution_enabled": False,
            "trading_performance_proof": False,
            "evaluation_mode": EVALUATION_MODE_FRONTIER,
            "input_path": self.config.predictions_path,
            "config": asdict(self.config),
            "load_metadata": load_metadata,
            "filtered_rows": int(len(filtered)),
            "signal_rows": int(len(signal_rows)),
            "policies": {
                "return_threshold": "BUY if predicted_return >= threshold; AVOID if predicted_return <= -threshold; otherwise HOLD.",
                "direction_and_return_threshold": "BUY if predicted_return >= threshold and predicted_direction is positive; AVOID if predicted_return <= -threshold; otherwise HOLD.",
                "strict_buy_precision_probe": "Reports the full threshold frontier and does not select thresholds using realized returns.",
            },
            "success_definitions": {
                "raw_positive": "BUY is correct if realized_forward_return > 0.",
                "cost_adjusted_positive": "BUY is correct if realized_forward_return exceeds 2 * (cost_per_trade + slippage).",
                "target_return": "BUY is correct if realized_forward_return >= target_return_threshold.",
            },
            "probability_rules": probability_metadata,
            "leakage_guard": {
                "signal_columns": [
                    "predicted_return",
                    "predicted_direction",
                    "probability_up",
                    "policy thresholds",
                    "cost/slippage assumptions",
                ],
                "realized_return_used_for_signal_creation": False,
            },
            "output_files": {name: str(path) for name, path in paths.items()},
        }
        paths["run_metadata"].write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

        return {
            "signal_rows": signal_rows,
            "buy_precision_by_model_horizon": buy_precision,
            "precision_coverage_frontier": frontier,
            "signal_effectiveness_summary": summary,
            "strategy_proxy_metrics": strategy_metrics,
            "benchmark_comparison": benchmark,
            "run_metadata": metadata,
            "paths": {name: str(path) for name, path in paths.items()},
        }

    def run_heldout_threshold_selection(self, predictions: pd.DataFrame | None = None) -> dict[str, Any]:
        filtered, load_metadata = self._load_filtered_predictions(predictions)
        selection_predictions, test_predictions, window_metadata = self._heldout_windows(filtered)
        (
            selected_thresholds,
            heldout_precision,
            trace,
            heldout_signal_rows,
            target_pass_fail,
            strategy_proxy,
            heldout_metadata,
        ) = self._build_heldout_outputs(
            selection_predictions=selection_predictions,
            test_predictions=test_predictions,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "selected_thresholds": self.output_dir / "selected_thresholds.csv",
            "heldout_buy_precision": self.output_dir / "heldout_buy_precision.csv",
            "threshold_selection_trace": self.output_dir / "threshold_selection_trace.csv",
            "heldout_signal_rows": self.output_dir / "heldout_signal_rows.csv",
            "precision_target_pass_fail": self.output_dir / "precision_target_pass_fail.csv",
            "heldout_strategy_proxy_metrics": self.output_dir / "heldout_strategy_proxy_metrics.csv",
            "run_metadata": self.output_dir / "run_metadata.json",
        }
        selected_thresholds.to_csv(paths["selected_thresholds"], index=False)
        heldout_precision.to_csv(paths["heldout_buy_precision"], index=False)
        trace.to_csv(paths["threshold_selection_trace"], index=False)
        heldout_signal_rows.to_csv(paths["heldout_signal_rows"], index=False)
        target_pass_fail.to_csv(paths["precision_target_pass_fail"], index=False)
        strategy_proxy.to_csv(paths["heldout_strategy_proxy_metrics"], index=False)

        metadata = {
            "analysis_only": True,
            "live_execution_enabled": False,
            "trading_performance_proof": False,
            "evaluation_mode": EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION,
            "input_path": self.config.predictions_path,
            "config": asdict(self.config),
            "load_metadata": load_metadata,
            "filtered_rows": int(len(filtered)),
            "window_metadata": window_metadata,
            "precision_targets": list(self.config.precision_targets),
            "leakage_guard": {
                "threshold_selection_uses_period": "selection",
                "heldout_period_used_for_threshold_selection": False,
                "heldout_realized_return_used_for_signal_creation": False,
            },
            **heldout_metadata,
            "output_files": {name: str(path) for name, path in paths.items()},
        }
        paths["run_metadata"].write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
        return {
            "selected_thresholds": selected_thresholds,
            "heldout_buy_precision": heldout_precision,
            "threshold_selection_trace": trace,
            "heldout_signal_rows": heldout_signal_rows,
            "precision_target_pass_fail": target_pass_fail,
            "heldout_strategy_proxy_metrics": strategy_proxy,
            "run_metadata": metadata,
            "paths": {name: str(path) for name, path in paths.items()},
        }


__all__ = [
    "BENCHMARK_COLUMNS",
    "DEFAULT_COST_PER_TRADE_VALUES",
    "DEFAULT_MINIMUM_SIGNAL_COUNTS",
    "DEFAULT_PREDICTED_RETURN_THRESHOLDS",
    "DEFAULT_PRECISION_TARGETS",
    "DEFAULT_PROBABILITY_UP_THRESHOLDS",
    "DEFAULT_SLIPPAGE_VALUES",
    "EVALUATION_MODE_FRONTIER",
    "EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION",
    "FRONTIER_COLUMNS",
    "HELDOUT_BUY_PRECISION_COLUMNS",
    "HELDOUT_STRATEGY_PROXY_COLUMNS",
    "POLICY_DIRECTION_AND_RETURN_THRESHOLD",
    "POLICY_RETURN_THRESHOLD",
    "POLICY_STRICT_BUY_PRECISION_PROBE",
    "PRECISION_TARGET_PASS_FAIL_COLUMNS",
    "SELECTED_THRESHOLD_COLUMNS",
    "SELECTION_CRITERION_MAX_PRECISION",
    "SIGNAL_AVOID",
    "SIGNAL_BUY",
    "SIGNAL_HOLD",
    "SIGNAL_ROW_COLUMNS",
    "STRATEGY_PROXY_COLUMNS",
    "SUCCESS_COST_ADJUSTED_POSITIVE",
    "SUCCESS_RAW_POSITIVE",
    "SUCCESS_TARGET_RETURN",
    "SUMMARY_COLUMNS",
    "THRESHOLD_SELECTION_TRACE_COLUMNS",
    "SignalEffectivenessConfig",
    "SignalEffectivenessRunner",
    "build_benchmark_comparison",
    "build_buy_precision_table",
    "build_precision_target_pass_fail",
    "build_precision_coverage_frontier",
    "build_strategy_proxy_metrics",
    "detect_prediction_columns",
    "generate_signal_rows",
    "load_prediction_csv",
    "normalize_prediction_frame",
    "select_thresholds_from_summary",
    "summarize_signal_effectiveness",
]
