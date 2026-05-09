"""Standard metrics engine for Phase 1 experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


METRICS_COLUMNS = [
    "experiment_id",
    "run_id",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "metric_group",
    "metric_name",
    "metric_value",
    "sample_size",
    "start_date",
    "end_date",
    "notes",
]


@dataclass(frozen=True)
class MetricResult:
    """One standardized metric row."""

    experiment_id: str
    run_id: str
    ticker: str
    horizon: int
    model_name: str
    model_type: str
    metric_group: str
    metric_name: str
    metric_value: float | None
    sample_size: int
    start_date: str
    end_date: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "ticker": self.ticker,
            "horizon": self.horizon,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metric_group": self.metric_group,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "sample_size": self.sample_size,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "notes": self.notes,
        }


class MetricsEngine:
    """Compute auditable metrics from model and baseline prediction rows."""

    def compute(
        self,
        predictions: pd.DataFrame,
        *,
        experiment_id: str,
        run_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Compute standardized metrics for each ticker/horizon/model group."""
        if predictions is None or predictions.empty:
            return pd.DataFrame(columns=METRICS_COLUMNS)

        required = {"ticker", "horizon", "model_name", "model_type", "y_true", "y_pred"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Predictions are missing required metric columns: {sorted(missing)}")

        frame = predictions.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

        rows: list[MetricResult] = []
        group_keys = ["ticker", "horizon", "model_name", "model_type"]
        for (ticker, horizon, model_name, model_type), group in frame.groupby(group_keys, dropna=False):
            group_start = self._date_value(group, "min", fallback=start_date)
            group_end = self._date_value(group, "max", fallback=end_date)
            rows.extend(
                self._compute_group(
                    group,
                    experiment_id=str(experiment_id),
                    run_id=str(run_id),
                    ticker=str(ticker),
                    horizon=int(horizon),
                    model_name=str(model_name),
                    model_type=str(model_type),
                    start_date=str(group_start or ""),
                    end_date=str(group_end or ""),
                )
            )
        return pd.DataFrame([row.to_dict() for row in rows], columns=METRICS_COLUMNS)

    @staticmethod
    def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
        """Return a compact JSON-serializable metrics summary."""
        if metrics is None or metrics.empty:
            return {
                "metric_rows": 0,
                "models": [],
                "baselines": [],
                "tickers": [],
                "horizons": [],
            }
        frame = metrics.copy()
        baselines = frame.loc[frame["model_type"] == "baseline", "model_name"].dropna().unique().tolist()
        models = frame.loc[frame["model_type"] == "model", "model_name"].dropna().unique().tolist()
        return {
            "metric_rows": int(len(frame)),
            "models": sorted(str(value) for value in models),
            "baselines": sorted(str(value) for value in baselines),
            "tickers": sorted(str(value) for value in frame["ticker"].dropna().unique().tolist()),
            "horizons": sorted(int(value) for value in frame["horizon"].dropna().unique().tolist()),
        }

    def _compute_group(
        self,
        group: pd.DataFrame,
        *,
        experiment_id: str,
        run_id: str,
        ticker: str,
        horizon: int,
        model_name: str,
        model_type: str,
        start_date: str,
        end_date: str,
    ) -> list[MetricResult]:
        rows: list[MetricResult] = []

        def add(
            metric_group: str,
            metric_name: str,
            metric_value: float | None,
            sample_size: int,
            notes: str,
        ) -> None:
            rows.append(
                MetricResult(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    ticker=ticker,
                    horizon=horizon,
                    model_name=model_name,
                    model_type=model_type,
                    metric_group=metric_group,
                    metric_name=metric_name,
                    metric_value=self._clean_value(metric_value),
                    sample_size=int(sample_size),
                    start_date=start_date,
                    end_date=end_date,
                    notes=notes,
                )
            )

        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        y_pred = pd.to_numeric(group["y_pred"], errors="coerce")
        valid_mask = y_true.notna() & y_pred.notna()
        valid_count = int(valid_mask.sum())
        total_count = int(len(group))

        if valid_count == 0:
            for metric in ("mae", "rmse", "mape", "directional_accuracy"):
                add("forecast", metric, None, 0, "no_valid_y_true_y_pred_pairs")
        else:
            errors = y_pred.loc[valid_mask] - y_true.loc[valid_mask]
            add("forecast", "mae", float(errors.abs().mean()), valid_count, "computed")
            add("forecast", "rmse", float(np.sqrt(np.square(errors).mean())), valid_count, "computed")

            denominator = y_true.loc[valid_mask].abs()
            mape_mask = denominator > 1e-12
            if bool(mape_mask.any()):
                mape = (errors.loc[mape_mask].abs() / denominator.loc[mape_mask]).mean() * 100.0
                add("forecast", "mape", float(mape), int(mape_mask.sum()), "computed_on_non_zero_denominator")
            else:
                add("forecast", "mape", None, 0, "not_computed_zero_or_near_zero_denominator")

            directional = self._directional_accuracy(group, valid_mask)
            add(
                "forecast",
                "directional_accuracy",
                directional["value"],
                directional["sample_size"],
                directional["notes"],
            )

        signal = self._signal_accuracy(group)
        add("decision_diagnostic", "signal_accuracy", signal["value"], signal["sample_size"], signal["notes"])
        add("decision_diagnostic", "coverage_count", float(valid_count), total_count, "non_null_prediction_count")
        add("decision_diagnostic", "prediction_count", float(total_count), total_count, "row_count")

        realized_vol = self._realized_volatility(group)
        add("risk", "realized_volatility", realized_vol["value"], realized_vol["sample_size"], realized_vol["notes"])
        drawdown = self._max_drawdown(group)
        add("risk", "max_drawdown", drawdown["value"], drawdown["sample_size"], drawdown["notes"])
        for source_column, metric_name in (("var", "var"), ("cvar", "cvar"), ("VaR", "var"), ("CVaR", "cvar")):
            if source_column in group.columns:
                value = pd.to_numeric(group[source_column], errors="coerce").dropna()
                add(
                    "risk",
                    metric_name,
                    float(value.iloc[-1]) if not value.empty else None,
                    int(len(value)),
                    f"passthrough_from_{source_column}" if not value.empty else f"{source_column}_not_available",
                )

        prediction_std = float(y_pred.dropna().std(ddof=0)) if y_pred.notna().any() else None
        add(
            "stability",
            "prediction_std",
            prediction_std,
            int(y_pred.notna().sum()),
            "computed" if prediction_std is not None else "no_valid_predictions",
        )
        if valid_count > 0:
            add("stability", "error_std", float(errors.std(ddof=0)), valid_count, "computed")
        else:
            add("stability", "error_std", None, 0, "no_valid_y_true_y_pred_pairs")
        missing_rate = float(y_pred.isna().mean()) if total_count else None
        add("stability", "missing_prediction_rate", missing_rate, total_count, "computed")
        return rows

    @staticmethod
    def _directional_accuracy(group: pd.DataFrame, valid_mask: pd.Series) -> dict[str, Any]:
        if {"actual_direction", "predicted_direction"}.issubset(group.columns):
            actual = pd.to_numeric(group.loc[valid_mask, "actual_direction"], errors="coerce")
            predicted = pd.to_numeric(group.loc[valid_mask, "predicted_direction"], errors="coerce")
            mask = actual.notna() & predicted.notna()
            if bool(mask.any()):
                return {
                    "value": float((np.sign(actual.loc[mask]) == np.sign(predicted.loc[mask])).mean()),
                    "sample_size": int(mask.sum()),
                    "notes": "computed_from_direction_columns",
                }
        y_true = pd.to_numeric(group.loc[valid_mask, "y_true"], errors="coerce")
        y_pred = pd.to_numeric(group.loc[valid_mask, "y_pred"], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        if bool(mask.any()):
            return {
                "value": float((np.sign(y_true.loc[mask]) == np.sign(y_pred.loc[mask])).mean()),
                "sample_size": int(mask.sum()),
                "notes": "computed_from_prediction_signs",
            }
        return {"value": None, "sample_size": 0, "notes": "direction_columns_unavailable"}

    @staticmethod
    def _signal_accuracy(group: pd.DataFrame) -> dict[str, Any]:
        actual_col = "signal_label" if "signal_label" in group.columns else "actual_signal" if "actual_signal" in group.columns else None
        predicted_col = "predicted_signal" if "predicted_signal" in group.columns else None
        if actual_col is None or predicted_col is None:
            return {"value": None, "sample_size": 0, "notes": "signal_labels_unavailable"}
        actual = group[actual_col].astype(str)
        predicted = group[predicted_col].astype(str)
        mask = actual.notna() & predicted.notna()
        if not bool(mask.any()):
            return {"value": None, "sample_size": 0, "notes": "signal_labels_empty"}
        return {
            "value": float((actual.loc[mask] == predicted.loc[mask]).mean()),
            "sample_size": int(mask.sum()),
            "notes": f"computed_from_{actual_col}_and_{predicted_col}",
        }

    @staticmethod
    def _realized_volatility(group: pd.DataFrame) -> dict[str, Any]:
        source_col = None
        for candidate in ("realized_return", "return", "returns"):
            if candidate in group.columns:
                source_col = candidate
                break
        if source_col is None:
            return {"value": None, "sample_size": 0, "notes": "returns_unavailable"}
        returns = pd.to_numeric(group[source_col], errors="coerce").dropna()
        if returns.empty:
            return {"value": None, "sample_size": 0, "notes": f"{source_col}_empty"}
        return {
            "value": float(returns.std(ddof=0) * np.sqrt(252.0)),
            "sample_size": int(len(returns)),
            "notes": f"annualized_from_{source_col}",
        }

    @staticmethod
    def _max_drawdown(group: pd.DataFrame) -> dict[str, Any]:
        if "equity_curve" not in group.columns:
            return {"value": None, "sample_size": 0, "notes": "equity_curve_unavailable"}
        equity = pd.to_numeric(group["equity_curve"], errors="coerce").dropna()
        if equity.empty:
            return {"value": None, "sample_size": 0, "notes": "equity_curve_empty"}
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        return {"value": float(drawdown.min()), "sample_size": int(len(equity)), "notes": "computed_from_equity_curve"}

    @staticmethod
    def _date_value(group: pd.DataFrame, op: str, fallback: str | None = None) -> str | None:
        if "date" not in group.columns:
            return fallback
        dates = pd.to_datetime(group["date"], errors="coerce").dropna()
        if dates.empty:
            return fallback
        value = dates.min() if op == "min" else dates.max()
        return value.date().isoformat()

    @staticmethod
    def _clean_value(value: float | None) -> float | None:
        if value is None:
            return None
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
