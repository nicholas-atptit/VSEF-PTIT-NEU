"""Combined decision-support analysis for dual-task forecast outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

REQUIRED_JOINED_COLUMNS = {
    "date",
    "ticker",
    "horizon",
    "model_name",
    "actual_return",
    "predicted_return",
    "actual_profit_label",
    "predicted_profit_label",
    "predicted_profit_probability",
}


@dataclass(slots=True)
class CombinedSignalConfig:
    dual_task_dir: str = "artifacts/dual_task"
    output_dir: str = "artifacts/combined_signal"
    horizons: list[str] = field(default_factory=lambda: ["3d", "5d", "20d"])
    return_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02])
    probability_thresholds: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65])
    w_return: float = 0.5
    w_profit: float = 0.5
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    ranking_group: str = "date"


def _normalize_numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.replace([np.inf, -np.inf], np.nan)
    minimum = valid.min()
    maximum = valid.max()
    if pd.isna(minimum) or pd.isna(maximum) or float(maximum) == float(minimum):
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((valid - minimum) / (maximum - minimum)).astype(float)


def _descending_rank_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    mask = numeric.notna()
    if not mask.any():
        return result
    rank = numeric.loc[mask].rank(method="average", ascending=False)
    count = int(mask.sum())
    if count == 1:
        result.loc[mask] = 1.0
        return result
    result.loc[mask] = 1.0 - ((rank - 1.0) / float(count - 1))
    return result


def _derive_label_thresholds(values: list[float]) -> tuple[float, float, float]:
    cleaned = sorted({float(item) for item in values})
    if not cleaned:
        raise ValueError("At least one threshold is required")
    base = cleaned[0]
    low = cleaned[1] if len(cleaned) > 1 else cleaned[0]
    high = cleaned[-1]
    return base, low, high


def _label_single_signal(value: float, *, base: float, low: float, high: float) -> str:
    if pd.isna(value):
        return "reject"
    if value > high:
        return "strong_positive"
    if value > low:
        return "moderate_positive"
    if value > base:
        return "weak_or_uncertain"
    return "reject"


def _label_combined_signal(
    predicted_return: float,
    probability: float,
    *,
    return_base: float,
    return_low: float,
    return_high: float,
    probability_base: float,
    probability_low: float,
    probability_high: float,
) -> str:
    if pd.isna(predicted_return) or pd.isna(probability):
        return "reject"
    if predicted_return > return_high and probability > probability_high:
        return "strong_positive"
    if predicted_return > return_low and probability > probability_low:
        return "moderate_positive"
    if predicted_return > return_base or probability > probability_base:
        return "weak_or_uncertain"
    return "reject"


def _ranking_group_key(dates: pd.Series, mode: str) -> pd.Series:
    timestamp = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if mode == "date":
        return timestamp.dt.strftime("%Y-%m-%d")
    if mode == "week":
        return timestamp.dt.to_period("W-FRI").astype(str)
    raise ValueError(f"Unsupported ranking_group={mode!r}. Expected 'date' or 'week'.")


class CombinedSignalAnalysisRunner:
    """Research-only combined signal analysis on top of dual-task outputs."""

    def __init__(self, config: CombinedSignalConfig) -> None:
        self.config = config
        self.dual_task_dir = Path(config.dual_task_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.summary_root = self.output_dir / "summary"
        self.return_base, self.return_low, self.return_high = _derive_label_thresholds(config.return_thresholds)
        self.probability_base, self.probability_low, self.probability_high = _derive_label_thresholds(
            config.probability_thresholds
        )

    @property
    def joined_evaluation_path(self) -> Path:
        return self.dual_task_dir / "summary" / "joined_regression_classification_evaluation.csv"

    def _load_joined_evaluation(self) -> pd.DataFrame:
        if not self.joined_evaluation_path.exists():
            raise FileNotFoundError(
                f"Dual-task joined evaluation not found at {self.joined_evaluation_path}. "
                "Run the dual-task workflow first."
            )
        frame = pd.read_csv(self.joined_evaluation_path)
        missing = REQUIRED_JOINED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                "Joined evaluation table is missing required columns: "
                + ", ".join(sorted(missing))
            )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for column in [
            "actual_return",
            "predicted_return",
            "predicted_profit_probability",
            "actual_profit_label",
            "predicted_profit_label",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["horizon"] = frame["horizon"].astype(str).str.lower()
        frame["model_name"] = frame["model_name"].astype(str).str.lower()
        frame = frame.dropna(subset=["date", "ticker", "horizon", "model_name"]).copy()
        duplicate_mask = frame.duplicated(subset=["date", "ticker", "horizon", "model_name"], keep=False)
        if duplicate_mask.any():
            duplicates = frame.loc[duplicate_mask, ["date", "ticker", "horizon", "model_name"]]
            raise ValueError(
                "Joined evaluation contains duplicate date/ticker/horizon/model rows: "
                + duplicates.head(5).to_dict(orient="records").__repr__()
            )
        return frame.sort_values(["horizon", "date", "model_name", "ticker"]).reset_index(drop=True)

    def _prepare_horizon_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
        prepared["ranking_group"] = _ranking_group_key(prepared["date"], self.config.ranking_group)
        prepared["predicted_profit_probability"] = prepared["predicted_profit_probability"].clip(lower=0.0, upper=1.0)
        prepared["normalized_predicted_return"] = (
            prepared.groupby("model_name", group_keys=False)["predicted_return"].apply(_normalize_numeric_series)
        )
        prepared["return_strength"] = prepared["normalized_predicted_return"]
        prepared["profit_confidence"] = prepared["predicted_profit_probability"]
        prepared["combined_score"] = (
            (float(self.config.w_return) * prepared["return_strength"])
            + (float(self.config.w_profit) * prepared["profit_confidence"])
        )
        prepared["gated_valid_signal"] = (
            (prepared["predicted_return"] > self.return_base)
            & (prepared["predicted_profit_probability"] > self.probability_base)
        )
        prepared["return_only_signal_label"] = prepared["predicted_return"].apply(
            lambda value: _label_single_signal(
                float(value) if pd.notna(value) else np.nan,
                base=self.return_base,
                low=self.return_low,
                high=self.return_high,
            )
        )
        prepared["probability_only_signal_label"] = prepared["predicted_profit_probability"].apply(
            lambda value: _label_single_signal(
                float(value) if pd.notna(value) else np.nan,
                base=self.probability_base,
                low=self.probability_low,
                high=self.probability_high,
            )
        )
        prepared["combined_signal_label"] = prepared.apply(
            lambda row: _label_combined_signal(
                float(row["predicted_return"]) if pd.notna(row["predicted_return"]) else np.nan,
                float(row["predicted_profit_probability"]) if pd.notna(row["predicted_profit_probability"]) else np.nan,
                return_base=self.return_base,
                return_low=self.return_low,
                return_high=self.return_high,
                probability_base=self.probability_base,
                probability_low=self.probability_low,
                probability_high=self.probability_high,
            ),
            axis=1,
        )
        prepared["return_rank_score"] = prepared.groupby("ranking_group", group_keys=False)["predicted_return"].apply(
            _descending_rank_score
        )
        prepared["profit_probability_rank_score"] = prepared.groupby(
            "ranking_group",
            group_keys=False,
        )["predicted_profit_probability"].apply(_descending_rank_score)
        prepared["rank_based_joint_score"] = (
            prepared["return_rank_score"].fillna(0.0) + prepared["profit_probability_rank_score"].fillna(0.0)
        ) / 2.0
        prepared["date"] = prepared["date"].dt.strftime("%Y-%m-%d")
        return prepared.sort_values(["date", "model_name", "ticker"]).reset_index(drop=True)

    def _summarize_bucket_family(
        self,
        frame: pd.DataFrame,
        *,
        model_name: str,
        family_name: str,
        bucket_column: str,
    ) -> list[dict[str, Any]]:
        subset = frame[frame["model_name"] == model_name].copy()
        total_rows = len(subset)
        profitable_total = int(subset["actual_profit_label"].sum())
        base_profit_rate = float(subset["actual_profit_label"].mean()) if total_rows else np.nan
        rows: list[dict[str, Any]] = []
        label_order = ["strong_positive", "moderate_positive", "weak_or_uncertain", "reject"]
        for label in label_order:
            label_frame = subset[subset[bucket_column] == label]
            observations = len(label_frame)
            profit_count = int(label_frame["actual_profit_label"].sum()) if observations else 0
            profit_rate = float(label_frame["actual_profit_label"].mean()) if observations else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "bucket_family": family_name,
                    "signal_bucket": label,
                    "observations": observations,
                    "coverage_ratio": (observations / total_rows) if total_rows else np.nan,
                    "avg_actual_return": float(label_frame["actual_return"].mean()) if observations else np.nan,
                    "median_actual_return": float(label_frame["actual_return"].median()) if observations else np.nan,
                    "hit_rate": float((label_frame["actual_return"] > 0).mean()) if observations else np.nan,
                    "realized_profit_rate": profit_rate,
                    "positive_precision": profit_rate,
                    "recall_of_profitable_cases": (profit_count / profitable_total) if profitable_total else np.nan,
                    "lift_vs_base_rate": (profit_rate / base_profit_rate)
                    if observations and pd.notna(base_profit_rate) and base_profit_rate > 0
                    else np.nan,
                }
            )

        selected_positive = subset[subset[bucket_column].isin(["strong_positive", "moderate_positive"])]
        selected_profitable = int(selected_positive["actual_profit_label"].sum()) if not selected_positive.empty else 0
        profit_rate = (
            float(selected_positive["actual_profit_label"].mean()) if not selected_positive.empty else np.nan
        )
        rows.append(
            {
                "model_name": model_name,
                "bucket_family": family_name,
                "signal_bucket": "strong_or_moderate_positive",
                "observations": int(len(selected_positive)),
                "coverage_ratio": (len(selected_positive) / total_rows) if total_rows else np.nan,
                "avg_actual_return": float(selected_positive["actual_return"].mean())
                if not selected_positive.empty
                else np.nan,
                "median_actual_return": float(selected_positive["actual_return"].median())
                if not selected_positive.empty
                else np.nan,
                "hit_rate": float((selected_positive["actual_return"] > 0).mean())
                if not selected_positive.empty
                else np.nan,
                "realized_profit_rate": profit_rate,
                "positive_precision": profit_rate,
                "recall_of_profitable_cases": (selected_profitable / profitable_total) if profitable_total else np.nan,
                "lift_vs_base_rate": (profit_rate / base_profit_rate)
                if not selected_positive.empty and pd.notna(base_profit_rate) and base_profit_rate > 0
                else np.nan,
            }
        )
        return rows

    def _build_bucket_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for model_name in sorted(frame["model_name"].unique()):
            rows.extend(
                self._summarize_bucket_family(
                    frame,
                    model_name=model_name,
                    family_name="combined_signal",
                    bucket_column="combined_signal_label",
                )
            )
            rows.extend(
                self._summarize_bucket_family(
                    frame,
                    model_name=model_name,
                    family_name="predicted_return_only",
                    bucket_column="return_only_signal_label",
                )
            )
            rows.extend(
                self._summarize_bucket_family(
                    frame,
                    model_name=model_name,
                    family_name="profit_probability_only",
                    bucket_column="probability_only_signal_label",
                )
            )
        return pd.DataFrame(rows).sort_values(["model_name", "bucket_family", "signal_bucket"]).reset_index(drop=True)

    def _summarize_ranking_method(
        self,
        subset: pd.DataFrame,
        *,
        model_name: str,
        method_name: str,
        score_column: str,
        top_k_values: list[int],
        return_threshold: float | None = None,
        probability_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if subset.empty:
            return []
        rows: list[dict[str, Any]] = []
        group_count = subset["ranking_group"].nunique()
        total_profitable = int(subset["actual_profit_label"].sum())
        base_profit_rate = float(subset["actual_profit_label"].mean()) if len(subset) else np.nan
        for top_k in top_k_values:
            picked_groups: list[pd.DataFrame] = []
            for _, group in subset.groupby("ranking_group", sort=True):
                ranked_group = group.sort_values(
                    [score_column, "ticker", "date"],
                    ascending=[False, True, True],
                )
                picked_groups.append(ranked_group.head(int(top_k)))
            if picked_groups:
                selected = pd.concat(picked_groups, ignore_index=True)
            else:
                selected = pd.DataFrame(columns=subset.columns)
            observations = len(selected)
            profit_count = int(selected["actual_profit_label"].sum()) if observations else 0
            profit_rate = float(selected["actual_profit_label"].mean()) if observations else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "ranking_method": method_name,
                    "score_column": score_column,
                    "top_k": int(top_k),
                    "ranking_group_mode": self.config.ranking_group,
                    "return_threshold": return_threshold,
                    "probability_threshold": probability_threshold,
                    "groups_covered": int(group_count),
                    "observations": observations,
                    "coverage_ratio": (observations / float(group_count * int(top_k)))
                    if group_count and top_k
                    else np.nan,
                    "average_actual_return": float(selected["actual_return"].mean()) if observations else np.nan,
                    "median_actual_return": float(selected["actual_return"].median()) if observations else np.nan,
                    "profit_rate": profit_rate,
                    "precision_at_top_k": profit_rate,
                    "recall_of_profitable_cases": (profit_count / total_profitable) if total_profitable else np.nan,
                    "lift_vs_base_rate": (profit_rate / base_profit_rate)
                    if observations and pd.notna(base_profit_rate) and base_profit_rate > 0
                    else np.nan,
                }
            )
        return rows

    def _build_ranking_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for model_name in sorted(frame["model_name"].unique()):
            model_frame = frame[frame["model_name"] == model_name].copy()
            rows.extend(
                self._summarize_ranking_method(
                    model_frame,
                    model_name=model_name,
                    method_name="predicted_return",
                    score_column="predicted_return",
                    top_k_values=self.config.top_k_values,
                )
            )
            rows.extend(
                self._summarize_ranking_method(
                    model_frame,
                    model_name=model_name,
                    method_name="predicted_profit_probability",
                    score_column="predicted_profit_probability",
                    top_k_values=self.config.top_k_values,
                )
            )
            rows.extend(
                self._summarize_ranking_method(
                    model_frame,
                    model_name=model_name,
                    method_name="combined_weighted_linear",
                    score_column="combined_score",
                    top_k_values=self.config.top_k_values,
                )
            )
            rows.extend(
                self._summarize_ranking_method(
                    model_frame,
                    model_name=model_name,
                    method_name="combined_rank_based",
                    score_column="rank_based_joint_score",
                    top_k_values=self.config.top_k_values,
                )
            )
            for return_threshold in self.config.return_thresholds:
                for probability_threshold in self.config.probability_thresholds:
                    gated = model_frame[
                        (model_frame["predicted_return"] > float(return_threshold))
                        & (model_frame["predicted_profit_probability"] > float(probability_threshold))
                    ].copy()
                    if gated.empty:
                        continue
                    rows.extend(
                        self._summarize_ranking_method(
                            gated,
                            model_name=model_name,
                            method_name="combined_weighted_linear_gated",
                            score_column="combined_score",
                            top_k_values=self.config.top_k_values,
                            return_threshold=float(return_threshold),
                            probability_threshold=float(probability_threshold),
                        )
                    )
        return pd.DataFrame(rows).sort_values(
            ["model_name", "ranking_method", "top_k", "return_threshold", "probability_threshold"]
        ).reset_index(drop=True)

    def _build_calibration_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        bins = [-1e-9, 0.50, 0.55, 0.60, 0.65, 1.000001]
        labels = ["lt_0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65+"]
        calibration = frame.copy()
        calibration["probability_bucket"] = pd.cut(
            calibration["predicted_profit_probability"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        rows: list[dict[str, Any]] = []
        for model_name in sorted(calibration["model_name"].unique()):
            model_frame = calibration[calibration["model_name"] == model_name]
            for bucket in labels:
                bucket_frame = model_frame[model_frame["probability_bucket"] == bucket]
                if bucket == "lt_0.50":
                    bucket_min, bucket_max = 0.0, 0.50
                elif bucket == "0.50-0.55":
                    bucket_min, bucket_max = 0.50, 0.55
                elif bucket == "0.55-0.60":
                    bucket_min, bucket_max = 0.55, 0.60
                elif bucket == "0.60-0.65":
                    bucket_min, bucket_max = 0.60, 0.65
                else:
                    bucket_min, bucket_max = 0.65, 1.0
                rows.append(
                    {
                        "model_name": model_name,
                        "probability_bucket": bucket,
                        "bucket_min_probability": bucket_min,
                        "bucket_max_probability": bucket_max,
                        "observations": int(len(bucket_frame)),
                        "avg_predicted_probability": float(bucket_frame["predicted_profit_probability"].mean())
                        if not bucket_frame.empty
                        else np.nan,
                        "realized_profit_rate": float(bucket_frame["actual_profit_label"].mean())
                        if not bucket_frame.empty
                        else np.nan,
                        "calibration_gap": (
                            float(bucket_frame["actual_profit_label"].mean())
                            - float(bucket_frame["predicted_profit_probability"].mean())
                        )
                        if not bucket_frame.empty
                        else np.nan,
                    }
                )
        return pd.DataFrame(rows).sort_values(["model_name", "bucket_min_probability"]).reset_index(drop=True)

    def _select_primary_top_k(self) -> int:
        if 3 in self.config.top_k_values:
            return 3
        if not self.config.top_k_values:
            raise ValueError("top_k_values must not be empty")
        return int(self.config.top_k_values[min(1, len(self.config.top_k_values) - 1)])

    def _build_overall_summary(
        self,
        *,
        horizon: str,
        bucket_summary: pd.DataFrame,
        ranking_summary: pd.DataFrame,
        calibration_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        primary_top_k = self._select_primary_top_k()
        rows: list[dict[str, Any]] = []
        for model_name in sorted(ranking_summary["model_name"].unique()):
            model_ranking = ranking_summary[
                (ranking_summary["model_name"] == model_name)
                & (ranking_summary["top_k"] == primary_top_k)
            ].copy()
            if model_ranking.empty:
                continue
            combined_candidates = model_ranking[
                model_ranking["ranking_method"].isin(["combined_weighted_linear", "combined_rank_based"])
                | (model_ranking["ranking_method"] == "combined_weighted_linear_gated")
            ].copy()
            best_combined = combined_candidates.sort_values(
                ["profit_rate", "average_actual_return", "coverage_ratio"],
                ascending=[False, False, False],
            ).iloc[0]
            return_only = model_ranking[model_ranking["ranking_method"] == "predicted_return"].iloc[0]
            probability_only = model_ranking[
                model_ranking["ranking_method"] == "predicted_profit_probability"
            ].iloc[0]

            strong_positive = bucket_summary[
                (bucket_summary["model_name"] == model_name)
                & (bucket_summary["bucket_family"] == "combined_signal")
                & (bucket_summary["signal_bucket"] == "strong_positive")
            ]
            strong_or_moderate = bucket_summary[
                (bucket_summary["model_name"] == model_name)
                & (bucket_summary["bucket_family"] == "combined_signal")
                & (bucket_summary["signal_bucket"] == "strong_or_moderate_positive")
            ]
            calibration_model = calibration_summary[calibration_summary["model_name"] == model_name]
            calibration_mae = float(calibration_model["calibration_gap"].abs().mean()) if not calibration_model.empty else np.nan

            rows.append(
                {
                    "horizon": horizon,
                    "model_name": model_name,
                    "primary_top_k": primary_top_k,
                    "best_combined_method": best_combined["ranking_method"],
                    "best_combined_return_threshold": best_combined["return_threshold"],
                    "best_combined_probability_threshold": best_combined["probability_threshold"],
                    "best_combined_avg_return": best_combined["average_actual_return"],
                    "best_combined_profit_rate": best_combined["profit_rate"],
                    "return_only_avg_return": return_only["average_actual_return"],
                    "return_only_profit_rate": return_only["profit_rate"],
                    "probability_only_avg_return": probability_only["average_actual_return"],
                    "probability_only_profit_rate": probability_only["profit_rate"],
                    "combined_minus_return_only_avg_return": (
                        best_combined["average_actual_return"] - return_only["average_actual_return"]
                    ),
                    "combined_minus_return_only_profit_rate": (
                        best_combined["profit_rate"] - return_only["profit_rate"]
                    ),
                    "combined_minus_probability_only_avg_return": (
                        best_combined["average_actual_return"] - probability_only["average_actual_return"]
                    ),
                    "combined_minus_probability_only_profit_rate": (
                        best_combined["profit_rate"] - probability_only["profit_rate"]
                    ),
                    "strong_positive_precision": (
                        float(strong_positive["positive_precision"].iloc[0]) if not strong_positive.empty else np.nan
                    ),
                    "strong_positive_avg_return": (
                        float(strong_positive["avg_actual_return"].iloc[0]) if not strong_positive.empty else np.nan
                    ),
                    "strong_or_moderate_profit_recall": (
                        float(strong_or_moderate["recall_of_profitable_cases"].iloc[0])
                        if not strong_or_moderate.empty
                        else np.nan
                    ),
                    "combined_improves_over_return_only": bool(
                        (
                            best_combined["average_actual_return"] >= return_only["average_actual_return"]
                        )
                        and (best_combined["profit_rate"] >= return_only["profit_rate"])
                    ),
                    "combined_improves_over_probability_only": bool(
                        (
                            best_combined["average_actual_return"] >= probability_only["average_actual_return"]
                        )
                        and (best_combined["profit_rate"] >= probability_only["profit_rate"])
                    ),
                    "calibration_mae": calibration_mae,
                }
            )

        return pd.DataFrame(rows).sort_values(["horizon", "model_name"]).reset_index(drop=True)

    def _build_cross_horizon_ranking(self, overall_summary: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for horizon in sorted(overall_summary["horizon"].unique()):
            horizon_frame = overall_summary[overall_summary["horizon"] == horizon].copy()
            if horizon_frame.empty:
                continue
            best_return = horizon_frame.sort_values(
                ["best_combined_avg_return", "best_combined_profit_rate"],
                ascending=[False, False],
            ).iloc[0]
            best_profit = horizon_frame.sort_values(
                ["best_combined_profit_rate", "best_combined_avg_return"],
                ascending=[False, False],
            ).iloc[0]
            best_precision = horizon_frame.sort_values(
                ["strong_positive_precision", "strong_positive_avg_return"],
                ascending=[False, False],
            ).iloc[0]
            improvement_score = int(horizon_frame["combined_improves_over_return_only"].sum()) + int(
                horizon_frame["combined_improves_over_probability_only"].sum()
            )
            rows.append(
                {
                    "horizon": horizon,
                    "best_model_by_combined_topk_return": best_return["model_name"],
                    "best_model_by_combined_topk_profit_rate": best_profit["model_name"],
                    "best_model_by_strong_positive_precision": best_precision["model_name"],
                    "models_combined_improve_over_return_only": int(
                        horizon_frame["combined_improves_over_return_only"].sum()
                    ),
                    "models_combined_improve_over_probability_only": int(
                        horizon_frame["combined_improves_over_probability_only"].sum()
                    ),
                    "mean_combined_minus_return_only_avg_return": float(
                        horizon_frame["combined_minus_return_only_avg_return"].mean()
                    ),
                    "mean_combined_minus_probability_only_profit_rate": float(
                        horizon_frame["combined_minus_probability_only_profit_rate"].mean()
                    ),
                    "horizon_improvement_score": improvement_score,
                }
            )
        ranking = pd.DataFrame(rows)
        if ranking.empty:
            return ranking
        ranking = ranking.sort_values(
            ["horizon_improvement_score", "mean_combined_minus_return_only_avg_return"],
            ascending=[False, False],
        ).reset_index(drop=True)
        ranking["horizon_rank"] = np.arange(1, len(ranking) + 1)
        return ranking

    def _render_horizon_charts(
        self,
        *,
        horizon: str,
        combined_table: pd.DataFrame,
        bucket_summary: pd.DataFrame,
        calibration_summary: pd.DataFrame,
        ranking_summary: pd.DataFrame,
        horizon_dir: Path,
    ) -> None:
        charts_dir = horizon_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, subset in combined_table.groupby("model_name"):
            ax.scatter(
                subset["combined_score"],
                subset["actual_return"],
                alpha=0.6,
                s=18,
                label=model_name,
            )
        ax.set_title(f"{horizon.upper()} Combined Score vs Actual Return")
        ax.set_xlabel("Combined Score")
        ax.set_ylabel("Actual Forward Return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts_dir / "combined_score_vs_actual_return.png", dpi=150)
        plt.close(fig)

        bar_data = bucket_summary[
            (bucket_summary["bucket_family"] == "combined_signal")
            & (bucket_summary["signal_bucket"].isin(["strong_positive", "moderate_positive", "weak_or_uncertain", "reject"]))
        ].copy()
        if not bar_data.empty:
            pivot = bar_data.pivot(index="signal_bucket", columns="model_name", values="avg_actual_return").reindex(
                ["strong_positive", "moderate_positive", "weak_or_uncertain", "reject"]
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Average Actual Return by Combined Signal Bucket")
            ax.set_xlabel("Combined Signal Bucket")
            ax.set_ylabel("Average Actual Return")
            ax.legend(title="Model")
            fig.tight_layout()
            fig.savefig(charts_dir / "signal_bucket_average_return.png", dpi=150)
            plt.close(fig)

        calibration = calibration_summary.copy()
        if not calibration.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model_name, subset in calibration.groupby("model_name"):
                x_values = subset["avg_predicted_probability"].fillna(
                    (subset["bucket_min_probability"] + subset["bucket_max_probability"]) / 2.0
                )
                ax.plot(x_values, subset["realized_profit_rate"], marker="o", label=model_name)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_title(f"{horizon.upper()} Profit Probability Calibration")
            ax.set_xlabel("Average Predicted Profit Probability")
            ax.set_ylabel("Realized Profit Rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(charts_dir / "probability_calibration.png", dpi=150)
            plt.close(fig)

        topk = self._select_primary_top_k()
        ranking_plot = ranking_summary[
            (ranking_summary["top_k"] == topk)
            & (ranking_summary["ranking_method"] != "combined_weighted_linear_gated")
        ].copy()
        if not ranking_plot.empty:
            pivot = ranking_plot.pivot(index="model_name", columns="ranking_method", values="precision_at_top_k")
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Precision@Top{topk} by Ranking Method")
            ax.set_xlabel("Model")
            ax.set_ylabel("Precision / Profit Rate")
            ax.legend(title="Ranking Method")
            fig.tight_layout()
            fig.savefig(charts_dir / "topk_precision_comparison.png", dpi=150)
            plt.close(fig)

    def _render_summary_charts(
        self,
        overall_summary: pd.DataFrame,
        cross_horizon_ranking: pd.DataFrame,
    ) -> None:
        self.summary_root.mkdir(parents=True, exist_ok=True)
        topk = self._select_primary_top_k()

        if not overall_summary.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df = overall_summary.copy()
            plot_df["comparison_label"] = plot_df["horizon"] + ":" + plot_df["model_name"]
            ax.bar(
                plot_df["comparison_label"],
                plot_df["best_combined_profit_rate"],
                label="Combined",
                alpha=0.8,
            )
            ax.plot(plot_df["comparison_label"], plot_df["return_only_profit_rate"], marker="o", label="Return-only")
            ax.plot(
                plot_df["comparison_label"],
                plot_df["probability_only_profit_rate"],
                marker="s",
                label="Probability-only",
            )
            ax.set_title(f"Profit Rate@Top{topk}: Combined vs Single-Signal Ranking")
            ax.set_xlabel("Horizon / Model")
            ax.set_ylabel("Profit Rate")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.summary_root / "combined_vs_single_signal_comparison.png", dpi=150)
            plt.close(fig)

        if not cross_horizon_ranking.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(cross_horizon_ranking["horizon"], cross_horizon_ranking["horizon_improvement_score"])
            ax.set_title("Cross-Horizon Combined-Signal Improvement Score")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Improvement Score")
            fig.tight_layout()
            fig.savefig(self.summary_root / "cross_horizon_combined_ranking.png", dpi=150)
            plt.close(fig)

    def run(self) -> dict[str, Any]:
        joined = self._load_joined_evaluation()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        horizon_results: dict[str, dict[str, Any]] = {}
        overall_rows: list[pd.DataFrame] = []
        filtered = joined[joined["horizon"].isin(self.config.horizons)].copy()
        if filtered.empty:
            raise ValueError(
                f"No joined evaluation rows found for requested horizons: {', '.join(self.config.horizons)}"
            )

        for horizon in self.config.horizons:
            horizon_frame = filtered[filtered["horizon"] == horizon].copy()
            if horizon_frame.empty:
                continue
            combined_table = self._prepare_horizon_table(horizon_frame)
            bucket_summary = self._build_bucket_summary(combined_table)
            ranking_summary = self._build_ranking_summary(combined_table)
            calibration_summary = self._build_calibration_summary(combined_table)
            overall_summary = self._build_overall_summary(
                horizon=horizon,
                bucket_summary=bucket_summary,
                ranking_summary=ranking_summary,
                calibration_summary=calibration_summary,
            )
            overall_rows.append(overall_summary)

            horizon_dir = self.output_dir / horizon
            horizon_dir.mkdir(parents=True, exist_ok=True)
            combined_path = horizon_dir / "combined_signal_table.csv"
            bucket_path = horizon_dir / "combined_bucket_summary.csv"
            ranking_path = horizon_dir / "combined_ranking_summary.csv"
            calibration_path = horizon_dir / "probability_calibration_summary.csv"
            config_path = horizon_dir / "run_config.json"

            combined_table.to_csv(combined_path, index=False)
            bucket_summary.to_csv(bucket_path, index=False)
            ranking_summary.to_csv(ranking_path, index=False)
            calibration_summary.to_csv(calibration_path, index=False)

            run_config = {
                "analysis_only": True,
                "live_execution_enabled": False,
                "dual_task_source_dir": str(self.dual_task_dir),
                "joined_evaluation_path": str(self.joined_evaluation_path),
                "horizon": horizon,
                "ranking_group": self.config.ranking_group,
                "top_k_values": [int(value) for value in self.config.top_k_values],
                "return_thresholds": [float(value) for value in self.config.return_thresholds],
                "probability_thresholds": [float(value) for value in self.config.probability_thresholds],
                "w_return": float(self.config.w_return),
                "w_profit": float(self.config.w_profit),
                "label_thresholds": {
                    "return_base": self.return_base,
                    "return_low": self.return_low,
                    "return_high": self.return_high,
                    "probability_base": self.probability_base,
                    "probability_low": self.probability_low,
                    "probability_high": self.probability_high,
                },
                "rows_analyzed": int(len(combined_table)),
                "models": sorted(combined_table["model_name"].unique().tolist()),
                "tickers": sorted(combined_table["ticker"].unique().tolist()),
                "no_leakage_reused_from_dual_task_outputs": True,
            }
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, indent=2)

            self._render_horizon_charts(
                horizon=horizon,
                combined_table=combined_table,
                bucket_summary=bucket_summary,
                calibration_summary=calibration_summary,
                ranking_summary=ranking_summary,
                horizon_dir=horizon_dir,
            )

            horizon_results[horizon] = {
                "combined_signal_table": combined_table,
                "combined_bucket_summary": bucket_summary,
                "combined_ranking_summary": ranking_summary,
                "probability_calibration_summary": calibration_summary,
                "paths": {
                    "combined_signal_table": str(combined_path),
                    "combined_bucket_summary": str(bucket_path),
                    "combined_ranking_summary": str(ranking_path),
                    "probability_calibration_summary": str(calibration_path),
                    "run_config": str(config_path),
                },
            }

        if overall_rows:
            overall_summary = pd.concat(overall_rows, ignore_index=True)
        else:
            overall_summary = pd.DataFrame(
                columns=[
                    "horizon",
                    "model_name",
                    "best_combined_method",
                    "best_combined_avg_return",
                    "best_combined_profit_rate",
                ]
            )
        cross_horizon_ranking = self._build_cross_horizon_ranking(overall_summary)
        overall_summary_path = self.summary_root / "overall_combined_signal_summary.csv"
        cross_horizon_path = self.summary_root / "cross_horizon_combined_ranking.csv"
        overall_summary.to_csv(overall_summary_path, index=False)
        cross_horizon_ranking.to_csv(cross_horizon_path, index=False)
        self._render_summary_charts(overall_summary, cross_horizon_ranking)

        return {
            "horizons": horizon_results,
            "overall_combined_signal_summary": overall_summary,
            "cross_horizon_combined_ranking": cross_horizon_ranking,
            "summary_paths": {
                "overall_combined_signal_summary": str(overall_summary_path),
                "cross_horizon_combined_ranking": str(cross_horizon_path),
            },
        }
