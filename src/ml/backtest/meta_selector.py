"""Regime-conditioned meta-selector on top of walk-forward artifacts."""

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
from pandas.errors import EmptyDataError

CANDIDATE_KEY_COLUMNS = [
    "model_name",
    "horizon",
    "ranking_method",
    "return_threshold",
    "probability_threshold",
]
AGGREGATED_CANDIDATE_COLUMNS = CANDIDATE_KEY_COLUMNS + [
    "prior_fold_count",
    "sample_count",
    "combined_topk_avg_return",
    "combined_topk_profit_rate",
    "positive_class_precision",
    "directional_accuracy",
    "normalized_combined_topk_avg_return",
    "normalized_combined_topk_profit_rate",
    "normalized_positive_class_precision",
    "normalized_directional_accuracy",
    "utility_score",
    "rank_score_combined_topk_avg_return",
    "rank_score_combined_topk_profit_rate",
    "rank_score_positive_class_precision",
    "rank_score_directional_accuracy",
    "weighted_rank_score",
    "candidate_label",
]
DEFAULT_SELECTOR_MODES = ["simple_regime_lookup", "regime_weighted_rank", "fallback_global"]
DEFAULT_REGIMES = ["bull", "bear", "sideway"]


@dataclass(slots=True)
class MetaSelectorConfig:
    walk_forward_dir: str = "artifacts/walk_forward_regime_robustness"
    output_dir: str = "artifacts/meta_selector"
    selector_modes: list[str] = field(default_factory=lambda: DEFAULT_SELECTOR_MODES.copy())
    minimum_prior_folds_per_regime: int = 2
    minimum_samples_per_regime: int = 30
    primary_top_k: int = 3
    utility_weight_topk_avg_return: float = 0.40
    utility_weight_topk_profit_rate: float = 0.30
    utility_weight_positive_class_precision: float = 0.20
    utility_weight_directional_accuracy: float = 0.10
    regimes: list[str] = field(default_factory=lambda: DEFAULT_REGIMES.copy())


def _normalize_for_utility(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.dropna().empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum == minimum:
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((numeric - minimum) / (maximum - minimum)).fillna(0.0).astype(float)


def _normalized_rank_score(values: pd.Series, *, ascending: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(0.0, index=values.index, dtype=float)
    mask = numeric.notna()
    if not mask.any():
        return result
    rank = numeric.loc[mask].rank(method="average", ascending=ascending)
    count = int(mask.sum())
    if count == 1:
        result.loc[mask] = 1.0
        return result
    result.loc[mask] = 1.0 - ((rank - 1.0) / float(count - 1))
    return result.fillna(0.0)


def _candidate_label(row: pd.Series | dict[str, Any]) -> str:
    item = dict(row)
    label = f"{item['model_name']}+{item['horizon']}+{item['ranking_method']}"
    if item.get("ranking_method") == "combined_weighted_linear_gated":
        return_threshold = item.get("return_threshold")
        probability_threshold = item.get("probability_threshold")
        label += f"+rt={float(return_threshold):.3f}+pt={float(probability_threshold):.2f}"
    return label


def _threshold_value(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


class RegimeConditionedMetaSelectorRunner:
    """Select historically best candidate setups by regime using only prior folds."""

    def __init__(self, config: MetaSelectorConfig) -> None:
        self.config = config
        self.walk_forward_dir = Path(config.walk_forward_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.summary_root = self.output_dir / "summary"
        self.charts_root = self.summary_root / "charts"

    def _discover_folds(self) -> list[dict[str, Any]]:
        folds: list[dict[str, Any]] = []
        for fold_dir in sorted(self.walk_forward_dir.glob("fold_*")):
            fold_summary_path = fold_dir / "fold_summary.csv"
            if not fold_summary_path.exists():
                continue
            overview = pd.read_csv(fold_summary_path)
            if overview.empty:
                continue
            row = overview.iloc[0].to_dict()
            if str(row.get("status", "completed")).lower() != "completed":
                continue
            row["fold_dir"] = fold_dir
            row["fold_number"] = int(row["fold_number"])
            folds.append(row)
        folds.sort(key=lambda item: item["fold_number"])
        if not folds:
            raise ValueError(f"No completed walk-forward folds found in {self.walk_forward_dir}")
        return folds

    def _load_fold_rows(self, fold: dict[str, Any]) -> pd.DataFrame:
        fold_dir = Path(fold["fold_dir"])
        regime_root = fold_dir / "regime_aware"
        tables: list[pd.DataFrame] = []
        for horizon_dir in sorted(path for path in regime_root.iterdir() if path.is_dir() and path.name != "summary"):
            table_path = horizon_dir / "regime_labeled_signal_table.csv"
            if not table_path.exists():
                continue
            try:
                frame = pd.read_csv(table_path)
            except EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="coerce").dt.normalize()
            frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
            frame["model_name"] = frame["model_name"].astype(str).str.lower()
            frame["horizon"] = frame["horizon"].astype(str).str.lower()
            frame["regime"] = frame["regime"].astype(str).str.lower()
            for column in [
                "actual_return",
                "predicted_return",
                "predicted_profit_probability",
                "actual_profit_label",
                "combined_score",
                "rank_based_joint_score",
                "directional_accuracy",
                "positive_class_precision",
            ]:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["fold_id"] = str(fold["fold_id"])
            frame["fold_number"] = int(fold["fold_number"])
            tables.append(frame)
        if not tables:
            return pd.DataFrame()
        combined = pd.concat(tables, ignore_index=True)
        combined = combined.sort_values(["prediction_date", "ticker", "model_name", "horizon"]).reset_index(drop=True)
        duplicate_mask = combined.duplicated(
            subset=["prediction_date", "ticker", "model_name", "horizon"],
            keep=False,
        )
        if duplicate_mask.any():
            raise ValueError(
                f"Duplicate meta-selector rows detected in {fold['fold_id']}: "
                f"{combined.loc[duplicate_mask, ['prediction_date', 'ticker', 'model_name', 'horizon']].head(5).to_dict(orient='records')}"
            )
        return combined

    def _load_fold_histories(
        self,
        prior_folds: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        model_frames: list[pd.DataFrame] = []
        combined_frames: list[pd.DataFrame] = []
        for fold in prior_folds:
            fold_dir = Path(fold["fold_dir"])
            model_path = fold_dir / "model_ranking.csv"
            combined_path = fold_dir / "combined_method_ranking.csv"
            if not model_path.exists() or not combined_path.exists():
                continue
            model_frame = pd.read_csv(model_path)
            combined_frame = pd.read_csv(combined_path)
            if not model_frame.empty:
                model_frame["model_name"] = model_frame["model_name"].astype(str).str.lower()
                model_frame["horizon"] = model_frame["horizon"].astype(str).str.lower()
                model_frame["regime"] = model_frame["regime"].astype(str).str.lower()
                model_frame["fold_number"] = int(fold["fold_number"])
                model_frames.append(model_frame)
            if not combined_frame.empty:
                combined_frame["model_name"] = combined_frame["model_name"].astype(str).str.lower()
                combined_frame["horizon"] = combined_frame["horizon"].astype(str).str.lower()
                combined_frame["regime"] = combined_frame["regime"].astype(str).str.lower()
                combined_frame["ranking_method"] = combined_frame["ranking_method"].astype(str).str.lower()
                for column in ("return_threshold", "probability_threshold", "top_k"):
                    if column in combined_frame.columns:
                        combined_frame[column] = pd.to_numeric(combined_frame[column], errors="coerce")
                combined_frame["fold_number"] = int(fold["fold_number"])
                combined_frames.append(combined_frame)
        model_history = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
        combined_history = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
        return model_history, combined_history

    def _merge_candidate_history(
        self,
        model_history: pd.DataFrame,
        combined_history: pd.DataFrame,
    ) -> pd.DataFrame:
        if combined_history.empty:
            return pd.DataFrame()
        combined = combined_history.copy()
        if "top_k" in combined.columns:
            combined = combined[combined["top_k"].fillna(self.config.primary_top_k) == int(self.config.primary_top_k)].copy()
        if model_history.empty:
            merged = combined.copy()
            merged["directional_accuracy"] = np.nan
            merged["positive_class_precision"] = np.nan
            return merged
        model_subset = model_history.copy()
        for column in [
            "fold_id",
            "regime",
            "model_name",
            "horizon",
            "directional_accuracy",
            "positive_class_precision",
            "observations_regression",
            "observations_classification",
        ]:
            if column not in model_subset.columns:
                model_subset[column] = np.nan
        keep_columns = [
            "fold_id",
            "regime",
            "model_name",
            "horizon",
            "directional_accuracy",
            "positive_class_precision",
            "observations_regression",
            "observations_classification",
        ]
        model_subset = model_subset[keep_columns].drop_duplicates(
            subset=["fold_id", "regime", "model_name", "horizon"]
        )
        merged = combined.merge(
            model_subset,
            on=["fold_id", "regime", "model_name", "horizon"],
            how="left",
            validate="many_to_one",
        )
        return merged

    def _aggregate_candidate_pool(
        self,
        candidate_history: pd.DataFrame,
        *,
        regime: str | None,
    ) -> pd.DataFrame:
        if candidate_history.empty:
            return pd.DataFrame()
        pool = candidate_history.copy()
        if regime is not None:
            pool = pool[pool["regime"] == regime].copy()
        if pool.empty:
            return pd.DataFrame(columns=AGGREGATED_CANDIDATE_COLUMNS)
        aggregated = (
            pool.groupby(CANDIDATE_KEY_COLUMNS, dropna=False, as_index=False)
            .agg(
                prior_fold_count=("fold_id", "nunique"),
                sample_count=("observations", "sum"),
                combined_topk_avg_return=("average_actual_return", "mean"),
                combined_topk_profit_rate=("profit_rate", "mean"),
                positive_class_precision=("positive_class_precision", "mean"),
                directional_accuracy=("directional_accuracy", "mean"),
            )
        )
        aggregated["normalized_combined_topk_avg_return"] = _normalize_for_utility(aggregated["combined_topk_avg_return"])
        aggregated["normalized_combined_topk_profit_rate"] = _normalize_for_utility(aggregated["combined_topk_profit_rate"])
        aggregated["normalized_positive_class_precision"] = _normalize_for_utility(aggregated["positive_class_precision"])
        aggregated["normalized_directional_accuracy"] = _normalize_for_utility(aggregated["directional_accuracy"])
        aggregated["utility_score"] = (
            float(self.config.utility_weight_topk_avg_return) * aggregated["normalized_combined_topk_avg_return"]
            + float(self.config.utility_weight_topk_profit_rate) * aggregated["normalized_combined_topk_profit_rate"]
            + float(self.config.utility_weight_positive_class_precision) * aggregated["normalized_positive_class_precision"]
            + float(self.config.utility_weight_directional_accuracy) * aggregated["normalized_directional_accuracy"]
        )
        aggregated["rank_score_combined_topk_avg_return"] = _normalized_rank_score(
            aggregated["combined_topk_avg_return"],
            ascending=False,
        )
        aggregated["rank_score_combined_topk_profit_rate"] = _normalized_rank_score(
            aggregated["combined_topk_profit_rate"],
            ascending=False,
        )
        aggregated["rank_score_positive_class_precision"] = _normalized_rank_score(
            aggregated["positive_class_precision"],
            ascending=False,
        )
        aggregated["rank_score_directional_accuracy"] = _normalized_rank_score(
            aggregated["directional_accuracy"],
            ascending=False,
        )
        aggregated["weighted_rank_score"] = (
            float(self.config.utility_weight_topk_avg_return) * aggregated["rank_score_combined_topk_avg_return"]
            + float(self.config.utility_weight_topk_profit_rate) * aggregated["rank_score_combined_topk_profit_rate"]
            + float(self.config.utility_weight_positive_class_precision) * aggregated["rank_score_positive_class_precision"]
            + float(self.config.utility_weight_directional_accuracy) * aggregated["rank_score_directional_accuracy"]
        )
        aggregated["candidate_label"] = aggregated.apply(_candidate_label, axis=1)
        return aggregated.sort_values(["utility_score", "weighted_rank_score"], ascending=[False, False]).reset_index(drop=True)

    @staticmethod
    def _filter_available_candidates(
        candidate_pool: pd.DataFrame,
        current_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidate_pool.empty or current_rows.empty:
            return candidate_pool.iloc[0:0].copy()
        available_pairs = current_rows[["model_name", "horizon"]].drop_duplicates().copy()
        return candidate_pool.merge(available_pairs, on=["model_name", "horizon"], how="inner")

    @staticmethod
    def _pick_candidate(pool: pd.DataFrame, *, mode: str) -> pd.Series | None:
        if pool.empty:
            return None
        if mode == "regime_weighted_rank":
            ranked = pool.sort_values(
                ["weighted_rank_score", "utility_score", "prior_fold_count", "sample_count"],
                ascending=[False, False, False, False],
            )
            return ranked.iloc[0]
        ranked = pool.sort_values(
            ["utility_score", "weighted_rank_score", "prior_fold_count", "sample_count"],
            ascending=[False, False, False, False],
        )
        return ranked.iloc[0]

    def _select_candidate_for_regime(
        self,
        *,
        candidate_history: pd.DataFrame,
        current_rows: pd.DataFrame,
        regime: str,
        selector_mode: str,
    ) -> dict[str, Any]:
        if candidate_history.empty:
            return {
                "status": "no_prior_history",
                "selection_reason": "No prior folds are available yet.",
            }
        regime_pool = self._filter_available_candidates(
            self._aggregate_candidate_pool(candidate_history, regime=regime),
            current_rows,
        )
        global_pool = self._filter_available_candidates(
            self._aggregate_candidate_pool(candidate_history, regime=None),
            current_rows,
        )
        strict_regime_pool = regime_pool[
            (regime_pool["prior_fold_count"] >= int(self.config.minimum_prior_folds_per_regime))
            & (regime_pool["sample_count"] >= int(self.config.minimum_samples_per_regime))
        ].copy()

        if selector_mode == "fallback_global":
            fallback_pool = global_pool[global_pool["sample_count"] >= int(self.config.minimum_samples_per_regime)].copy()
            if fallback_pool.empty:
                fallback_pool = global_pool.copy()
            chosen = self._pick_candidate(fallback_pool, mode="simple_regime_lookup")
            if chosen is None:
                return {
                    "status": "no_candidate_history",
                    "selection_reason": "No global candidate history is available.",
                }
            chosen_dict = chosen.to_dict()
            chosen_dict.update(
                {
                    "status": "selected",
                    "fallback_used": False,
                    "selection_reason": (
                        f"fallback_global selected {_candidate_label(chosen_dict)} from "
                        f"{int(chosen_dict['prior_fold_count'])} prior folds and {int(chosen_dict['sample_count'])} samples"
                    ),
                }
            )
            return chosen_dict

        if not strict_regime_pool.empty:
            chosen = self._pick_candidate(strict_regime_pool, mode=selector_mode)
            if chosen is None:
                return {
                    "status": "no_candidate_history",
                    "selection_reason": f"No eligible {regime} candidates were available.",
                }
            chosen_dict = chosen.to_dict()
            chosen_dict.update(
                {
                    "status": "selected",
                    "fallback_used": False,
                    "selection_reason": (
                        f"selected {_candidate_label(chosen_dict)} because it ranked highest in {regime} "
                        f"using {int(chosen_dict['prior_fold_count'])} prior folds and {int(chosen_dict['sample_count'])} samples"
                    ),
                }
            )
            return chosen_dict

        fallback_pool = global_pool[global_pool["sample_count"] >= int(self.config.minimum_samples_per_regime)].copy()
        if fallback_pool.empty:
            fallback_pool = global_pool.copy()
        chosen = self._pick_candidate(fallback_pool, mode=selector_mode)
        if chosen is None:
            return {
                "status": "no_candidate_history",
                "selection_reason": (
                    f"{selector_mode} had no prior candidates for regime={regime}, and global fallback was also unavailable."
                ),
            }
        chosen_dict = chosen.to_dict()
        regime_fold_count = int(regime_pool["prior_fold_count"].max()) if not regime_pool.empty else 0
        regime_sample_count = int(regime_pool["sample_count"].max()) if not regime_pool.empty else 0
        chosen_dict.update(
            {
                "status": "selected",
                "fallback_used": True,
                "selection_reason": (
                    f"fallback_global used because {regime} had insufficient history "
                    f"({regime_fold_count} prior folds, {regime_sample_count} samples); "
                    f"selected {_candidate_label(chosen_dict)} from {int(chosen_dict['prior_fold_count'])} prior folds"
                ),
            }
        )
        return chosen_dict

    def _choose_global_component_baselines(
        self,
        *,
        candidate_history: pd.DataFrame,
        model_history: pd.DataFrame,
        current_rows: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        if candidate_history.empty:
            return {}
        global_pool = self._filter_available_candidates(
            self._aggregate_candidate_pool(candidate_history, regime=None),
            current_rows,
        )
        if global_pool.empty:
            return {}

        baselines: dict[str, dict[str, Any]] = {}

        fixed_best_global = self._pick_candidate(global_pool, mode="simple_regime_lookup")
        if fixed_best_global is not None:
            item = fixed_best_global.to_dict()
            item["selection_reason"] = "Fixed best global candidate by prior composite utility."
            baselines["fixed_best_global_setup"] = item

        fixed_ranked = self._pick_candidate(global_pool, mode="regime_weighted_rank")
        if fixed_ranked is not None:
            item = fixed_ranked.to_dict()
            item["selection_reason"] = "Regime-agnostic top-ranked candidate by prior weighted rank."
            baselines["regime_agnostic_top_ranked_setup"] = item

        return_only_pool = global_pool[global_pool["ranking_method"] == "predicted_return"].copy()
        fixed_return_only = self._pick_candidate(return_only_pool, mode="simple_regime_lookup")
        if fixed_return_only is None:
            synthetic_pool = global_pool.copy()
            if not model_history.empty:
                model_pool = model_history.copy()
                model_pool["model_name"] = model_pool["model_name"].astype(str).str.lower()
                model_pool["horizon"] = model_pool["horizon"].astype(str).str.lower()
                available_models = set(current_rows["model_name"].dropna().astype(str).str.lower())
                model_pool = model_pool[model_pool["model_name"].isin(available_models)]
                if not model_pool.empty:
                    best_regression_row = (
                        model_pool.groupby(["model_name", "horizon"], as_index=False)["rank_regression_rmse"]
                        .mean()
                        .sort_values(["rank_regression_rmse", "model_name", "horizon"], ascending=[True, True, True])
                        .iloc[0]
                    )
                    synthetic_pool = synthetic_pool[
                        (synthetic_pool["model_name"] == str(best_regression_row["model_name"]))
                        & (synthetic_pool["horizon"] == str(best_regression_row["horizon"]))
                    ].copy()
            if synthetic_pool.empty:
                synthetic_pool = global_pool.copy()
            if not synthetic_pool.empty:
                synthetic_pool = synthetic_pool.sort_values(
                    ["utility_score", "weighted_rank_score", "prior_fold_count", "sample_count"],
                    ascending=[False, False, False, False],
                ).copy()
                synthetic_pool["ranking_method"] = "predicted_return"
                synthetic_pool["return_threshold"] = np.nan
                synthetic_pool["probability_threshold"] = np.nan
                fixed_return_only = synthetic_pool.iloc[0]
        if fixed_return_only is not None:
            item = fixed_return_only.to_dict()
            item["selection_reason"] = "Naive global baseline using return-only ranking."
            baselines["naive_global_baseline"] = item

        if not model_history.empty:
            model_pool = model_history.copy()
            model_pool["model_name"] = model_pool["model_name"].astype(str).str.lower()
            available_models = set(current_rows["model_name"].dropna().astype(str).str.lower())
            model_pool = model_pool[model_pool["model_name"].isin(available_models)]
            if not model_pool.empty:
                regression_model = (
                    model_pool.groupby("model_name", as_index=False)["rank_regression_rmse"]
                    .mean()
                    .sort_values("rank_regression_rmse", ascending=True)
                    .iloc[0]["model_name"]
                )
                classifier_model = (
                    model_pool.groupby("model_name", as_index=False)["positive_class_precision"]
                    .mean()
                    .sort_values("positive_class_precision", ascending=False)
                    .iloc[0]["model_name"]
                )

                regression_candidates = global_pool[global_pool["model_name"] == regression_model]
                regression_choice = self._pick_candidate(regression_candidates, mode="simple_regime_lookup")
                if regression_choice is not None:
                    item = regression_choice.to_dict()
                    item["selection_reason"] = "Fixed best regression-model family chosen from prior folds."
                    baselines["fixed_best_regression_model"] = item

                classifier_candidates = global_pool[global_pool["model_name"] == classifier_model]
                classifier_choice = self._pick_candidate(classifier_candidates, mode="simple_regime_lookup")
                if classifier_choice is not None:
                    item = classifier_choice.to_dict()
                    item["selection_reason"] = "Fixed best classifier-model family chosen from prior folds."
                    baselines["fixed_best_classifier_model"] = item

        horizon_pool = (
            global_pool.groupby("horizon", as_index=False)["utility_score"]
            .mean()
            .sort_values("utility_score", ascending=False)
        )
        if not horizon_pool.empty:
            horizon_name = str(horizon_pool.iloc[0]["horizon"])
            horizon_candidates = global_pool[global_pool["horizon"] == horizon_name]
            horizon_choice = self._pick_candidate(horizon_candidates, mode="simple_regime_lookup")
            if horizon_choice is not None:
                item = horizon_choice.to_dict()
                item["selection_reason"] = "Fixed best horizon chosen from prior folds."
                baselines["fixed_best_horizon"] = item

        method_pool = (
            global_pool.groupby("ranking_method", as_index=False)["utility_score"]
            .mean()
            .sort_values("utility_score", ascending=False)
        )
        if not method_pool.empty:
            method_name = str(method_pool.iloc[0]["ranking_method"])
            method_candidates = global_pool[global_pool["ranking_method"] == method_name]
            method_choice = self._pick_candidate(method_candidates, mode="simple_regime_lookup")
            if method_choice is not None:
                item = method_choice.to_dict()
                item["selection_reason"] = "Fixed best combined-method family chosen from prior folds."
                baselines["fixed_best_combined_method"] = item

        return baselines

    @staticmethod
    def _candidate_score_from_row(row: pd.Series, candidate: dict[str, Any]) -> tuple[float, bool]:
        method = str(candidate["ranking_method"])
        if method == "predicted_return":
            return float(row["predicted_return"]), True
        if method == "predicted_profit_probability":
            return float(row["predicted_profit_probability"]), True
        if method == "combined_weighted_linear":
            return float(row["combined_score"]), True
        if method == "combined_rank_based":
            return float(row["rank_based_joint_score"]), True
        if method == "combined_weighted_linear_gated":
            return_threshold = _threshold_value(candidate.get("return_threshold")) or 0.0
            probability_threshold = _threshold_value(candidate.get("probability_threshold")) or 0.5
            active = (
                float(row["predicted_return"]) > return_threshold
                and float(row["predicted_profit_probability"]) > probability_threshold
            )
            return (float(row["combined_score"]) if active else -1.0), active
        return float(row.get("combined_score", np.nan)), True

    def _materialize_selection_rows(
        self,
        *,
        current_rows: pd.DataFrame,
        candidate: dict[str, Any],
        selector_mode: str,
        selection_family: str,
        selection_reason: str,
        selected_regime: str | None,
    ) -> pd.DataFrame:
        if current_rows.empty:
            return current_rows.iloc[0:0].copy()
        scoped = current_rows.copy()
        if selected_regime is not None:
            scoped = scoped[scoped["regime"] == selected_regime].copy()
        scoped = scoped[
            (scoped["model_name"] == str(candidate["model_name"]))
            & (scoped["horizon"] == str(candidate["horizon"]))
        ].copy()
        if scoped.empty:
            return scoped

        scores = scoped.apply(lambda row: self._candidate_score_from_row(row, candidate), axis=1)
        scoped["selector_score"] = [item[0] for item in scores]
        scoped["signal_active"] = [item[1] for item in scores]
        scoped["selector_mode"] = selector_mode
        scoped["selection_family"] = selection_family
        scoped["selected_model_name"] = str(candidate["model_name"])
        scoped["selected_horizon"] = str(candidate["horizon"])
        scoped["selected_combined_method"] = str(candidate["ranking_method"])
        scoped["selected_return_threshold"] = _threshold_value(candidate.get("return_threshold"))
        scoped["selected_probability_threshold"] = _threshold_value(candidate.get("probability_threshold"))
        scoped["selection_reason"] = selection_reason
        scoped["prior_utility_score"] = float(candidate.get("utility_score", np.nan))
        scoped["prior_weighted_rank_score"] = float(candidate.get("weighted_rank_score", np.nan))
        scoped["prior_fold_count"] = int(candidate.get("prior_fold_count", 0))
        scoped["prior_sample_count"] = int(candidate.get("sample_count", 0))
        scoped["fallback_used"] = bool(candidate.get("fallback_used", False))
        scoped["selection_regime"] = selected_regime if selected_regime is not None else "global"
        return scoped.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)

    def _compute_topk_metrics(self, frame: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for top_k in (1, 3, 5):
            metrics[f"top_{top_k}_avg_return"] = np.nan
            metrics[f"top_{top_k}_profit_rate"] = np.nan
        if frame.empty:
            return metrics
        scored = frame.copy()
        scored["prediction_date"] = pd.to_datetime(scored["prediction_date"], errors="coerce").dt.normalize()
        scored["selector_score"] = pd.to_numeric(scored["selector_score"], errors="coerce").fillna(-1.0)
        for top_k in (1, 3, 5):
            top_groups: list[pd.DataFrame] = []
            for _, group in scored.groupby("prediction_date", dropna=False):
                ranked = group.sort_values(["selector_score", "ticker"], ascending=[False, True]).head(top_k)
                if not ranked.empty:
                    top_groups.append(ranked)
            if not top_groups:
                continue
            merged = pd.concat(top_groups, ignore_index=True)
            metrics[f"top_{top_k}_avg_return"] = float(merged["actual_return"].mean())
            metrics[f"top_{top_k}_profit_rate"] = float(merged["actual_profit_label"].mean())
        return metrics

    def _evaluate_rows(
        self,
        frame: pd.DataFrame,
        *,
        fold: dict[str, Any],
        entity_name: str,
        selection_family: str,
    ) -> dict[str, Any]:
        topk_metrics = self._compute_topk_metrics(frame)
        selector_score = pd.to_numeric(frame["selector_score"], errors="coerce") if not frame.empty else pd.Series(dtype=float)
        return {
            "fold_id": str(fold["fold_id"]),
            "fold_number": int(fold["fold_number"]),
            "entity_name": entity_name,
            "selection_family": selection_family,
            "status": "evaluated" if not frame.empty else "no_rows_selected",
            "observations": int(len(frame)),
            "active_observations": int(frame["signal_active"].sum()) if "signal_active" in frame.columns else 0,
            "average_actual_return": float(frame["actual_return"].mean()) if not frame.empty else np.nan,
            "profit_label_hit_rate": float(frame["actual_profit_label"].mean()) if not frame.empty else np.nan,
            "average_predicted_return": float(frame["predicted_return"].mean()) if not frame.empty else np.nan,
            "average_predicted_profit_probability": (
                float(frame["predicted_profit_probability"].mean()) if not frame.empty else np.nan
            ),
            "average_selector_score": float(selector_score.mean()) if not selector_score.empty else np.nan,
            "selector_score_std": float(selector_score.std(ddof=0)) if not selector_score.empty else np.nan,
            **topk_metrics,
        }

    def _evaluate_by_regime(
        self,
        frame: pd.DataFrame,
        *,
        fold: dict[str, Any],
        entity_name: str,
        selection_family: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for regime, group in frame.groupby("regime", dropna=False):
            row = self._evaluate_rows(group, fold=fold, entity_name=entity_name, selection_family=selection_family)
            row["regime"] = regime
            rows.append(row)
        return pd.DataFrame(rows)

    def _empty_fold_outputs(self, fold: dict[str, Any], *, reason: str) -> dict[str, Any]:
        selected_candidates = pd.DataFrame(
            columns=[
                "date",
                "prediction_date",
                "ticker",
                "regime",
                "selector_mode",
                "selected_model_name",
                "selected_horizon",
                "selected_combined_method",
                "selection_reason",
                "actual_return",
                "predicted_return",
                "predicted_profit_probability",
                "combined_score",
                "actual_profit_label",
            ]
        )
        selector_performance = pd.DataFrame(
            [
                {
                    "fold_id": str(fold["fold_id"]),
                    "fold_number": int(fold["fold_number"]),
                    "entity_name": mode,
                    "selection_family": "selector",
                    "status": "insufficient_prior_history",
                    "selection_reason": reason,
                }
                for mode in self.config.selector_modes
            ]
        )
        selector_vs_baselines = selector_performance.copy()
        regime_selection_summary = pd.DataFrame(
            [
                {
                    "fold_id": str(fold["fold_id"]),
                    "fold_number": int(fold["fold_number"]),
                    "selector_mode": mode,
                    "regime": regime,
                    "status": "insufficient_prior_history",
                    "selection_reason": reason,
                }
                for mode in self.config.selector_modes
                for regime in self.config.regimes
            ]
        )
        return {
            "selected_candidates": selected_candidates,
            "selector_performance": selector_performance,
            "selector_vs_baselines": selector_vs_baselines,
            "regime_selection_summary": regime_selection_summary,
            "selector_regime_performance": pd.DataFrame(),
        }

    def _run_fold(
        self,
        *,
        fold: dict[str, Any],
        prior_folds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fold_dir = self.output_dir / str(fold["fold_id"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        current_rows = self._load_fold_rows(fold)
        model_history, combined_history = self._load_fold_histories(prior_folds)
        candidate_history = self._merge_candidate_history(model_history, combined_history)

        if candidate_history.empty:
            result = self._empty_fold_outputs(fold, reason="No prior folds are available for meta-selection.")
        else:
            selection_summaries: list[dict[str, Any]] = []
            selected_frames: list[pd.DataFrame] = []

            for selector_mode in self.config.selector_modes:
                for regime in self.config.regimes:
                    regime_rows = current_rows[current_rows["regime"] == regime].copy()
                    if regime_rows.empty:
                        continue
                    selected_candidate = self._select_candidate_for_regime(
                        candidate_history=candidate_history,
                        current_rows=regime_rows,
                        regime=regime,
                        selector_mode=selector_mode,
                    )
                    selection_summaries.append(
                        {
                            "fold_id": str(fold["fold_id"]),
                            "fold_number": int(fold["fold_number"]),
                            "selector_mode": selector_mode,
                            "regime": regime,
                            "status": selected_candidate.get("status", "unknown"),
                            "selected_model_name": selected_candidate.get("model_name"),
                            "selected_horizon": selected_candidate.get("horizon"),
                            "selected_combined_method": selected_candidate.get("ranking_method"),
                            "selected_return_threshold": _threshold_value(selected_candidate.get("return_threshold")),
                            "selected_probability_threshold": _threshold_value(selected_candidate.get("probability_threshold")),
                            "prior_fold_count": int(selected_candidate.get("prior_fold_count", 0) or 0),
                            "prior_sample_count": int(selected_candidate.get("sample_count", 0) or 0),
                            "utility_score": float(selected_candidate.get("utility_score", np.nan)),
                            "weighted_rank_score": float(selected_candidate.get("weighted_rank_score", np.nan)),
                            "fallback_used": bool(selected_candidate.get("fallback_used", False)),
                            "selection_reason": selected_candidate.get("selection_reason"),
                        }
                    )
                    if selected_candidate.get("status") != "selected":
                        continue
                    materialized = self._materialize_selection_rows(
                        current_rows=regime_rows,
                        candidate=selected_candidate,
                        selector_mode=selector_mode,
                        selection_family="selector",
                        selection_reason=str(selected_candidate["selection_reason"]),
                        selected_regime=regime,
                    )
                    if not materialized.empty:
                        selected_frames.append(materialized)

            selected_candidates = (
                pd.concat(selected_frames, ignore_index=True)
                if selected_frames
                else pd.DataFrame()
            )
            selector_performance_rows = []
            selector_regime_performance_frames = []
            for selector_mode in self.config.selector_modes:
                mode_rows = selected_candidates[selected_candidates["selector_mode"] == selector_mode].copy()
                selector_performance_rows.append(
                    self._evaluate_rows(
                        mode_rows,
                        fold=fold,
                        entity_name=selector_mode,
                        selection_family="selector",
                    )
                )
                regime_frame = self._evaluate_by_regime(
                    mode_rows,
                    fold=fold,
                    entity_name=selector_mode,
                    selection_family="selector",
                )
                if not regime_frame.empty:
                    regime_frame["selector_mode"] = selector_mode
                    selector_regime_performance_frames.append(regime_frame)

            selector_performance = pd.DataFrame(selector_performance_rows)
            selector_regime_performance = (
                pd.concat(selector_regime_performance_frames, ignore_index=True)
                if selector_regime_performance_frames
                else pd.DataFrame()
            )

            baselines = self._choose_global_component_baselines(
                candidate_history=candidate_history,
                model_history=model_history,
                current_rows=current_rows,
            )
            baseline_eval_rows = []
            for baseline_name, candidate in baselines.items():
                baseline_rows = self._materialize_selection_rows(
                    current_rows=current_rows,
                    candidate=candidate,
                    selector_mode=baseline_name,
                    selection_family="baseline",
                    selection_reason=str(candidate["selection_reason"]),
                    selected_regime=None,
                )
                baseline_eval_rows.append(
                    self._evaluate_rows(
                        baseline_rows,
                        fold=fold,
                        entity_name=baseline_name,
                        selection_family="baseline",
                    )
                )

            selector_vs_baselines = pd.concat(
                [
                    selector_performance,
                    pd.DataFrame(baseline_eval_rows),
                ],
                ignore_index=True,
            )
            fixed_best_global = selector_vs_baselines[
                selector_vs_baselines["entity_name"] == "fixed_best_global_setup"
            ]
            if not fixed_best_global.empty:
                fixed_row = fixed_best_global.iloc[0]
                selector_vs_baselines["avg_return_vs_fixed_best_global"] = (
                    selector_vs_baselines["average_actual_return"] - float(fixed_row["average_actual_return"])
                )
                selector_vs_baselines["profit_rate_vs_fixed_best_global"] = (
                    selector_vs_baselines["profit_label_hit_rate"] - float(fixed_row["profit_label_hit_rate"])
                )
            else:
                selector_vs_baselines["avg_return_vs_fixed_best_global"] = np.nan
                selector_vs_baselines["profit_rate_vs_fixed_best_global"] = np.nan

            result = {
                "selected_candidates": selected_candidates,
                "selector_performance": selector_performance,
                "selector_vs_baselines": selector_vs_baselines,
                "regime_selection_summary": pd.DataFrame(selection_summaries),
                "selector_regime_performance": selector_regime_performance,
            }

        fold_config = {
            **asdict(self.config),
            "fold_id": str(fold["fold_id"]),
            "fold_number": int(fold["fold_number"]),
            "train_start": str(fold["train_start"]),
            "train_end": str(fold["train_end"]),
            "eval_start": str(fold["eval_start"]),
            "eval_end": str(fold["eval_end"]),
            "prior_fold_ids": [str(item["fold_id"]) for item in prior_folds],
            "no_leakage": True,
            "analysis_only": True,
            "live_execution_enabled": False,
        }
        with (fold_dir / "fold_config.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_config, handle, indent=2)
        result["selected_candidates"].to_csv(fold_dir / "selected_candidates.csv", index=False)
        result["selector_performance"].to_csv(fold_dir / "selector_performance.csv", index=False)
        result["selector_vs_baselines"].to_csv(fold_dir / "selector_vs_baselines.csv", index=False)
        result["regime_selection_summary"].to_csv(fold_dir / "regime_selection_summary.csv", index=False)
        result["fold_id"] = str(fold["fold_id"])
        result["fold_number"] = int(fold["fold_number"])
        result["fold_dir"] = fold_dir
        return result

    def _build_selector_stability_summary(self, selected_candidates: pd.DataFrame) -> pd.DataFrame:
        if selected_candidates.empty:
            return pd.DataFrame()
        grouped = (
            selected_candidates.groupby(
                [
                    "selector_mode",
                    "regime",
                    "selected_model_name",
                    "selected_horizon",
                    "selected_combined_method",
                    "selected_return_threshold",
                    "selected_probability_threshold",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                folds_selected=("fold_id", "nunique"),
                rows_selected=("ticker", "size"),
                mean_prior_utility_score=("prior_utility_score", "mean"),
                utility_score_std=("prior_utility_score", lambda values: float(pd.Series(values).std(ddof=0))),
                fallback_count=("fallback_used", "sum"),
            )
        )
        grouped["candidate_label"] = grouped.apply(
            lambda row: _candidate_label(
                {
                    "model_name": row["selected_model_name"],
                    "horizon": row["selected_horizon"],
                    "ranking_method": row["selected_combined_method"],
                    "return_threshold": row["selected_return_threshold"],
                    "probability_threshold": row["selected_probability_threshold"],
                }
            ),
            axis=1,
        )
        return grouped.sort_values(["selector_mode", "regime", "folds_selected", "rows_selected"], ascending=[True, True, False, False]).reset_index(drop=True)

    def _build_selector_regime_summary(self, selected_candidates: pd.DataFrame) -> pd.DataFrame:
        if selected_candidates.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for (selector_mode, regime), group in selected_candidates.groupby(["selector_mode", "regime"], dropna=False):
            metrics = self._evaluate_rows(
                group,
                fold={"fold_id": "ALL", "fold_number": 0},
                entity_name=str(selector_mode),
                selection_family="selector",
            )
            metrics["selector_mode"] = selector_mode
            metrics["regime"] = regime
            rows.append(metrics)
        return pd.DataFrame(rows).sort_values(["selector_mode", "regime"]).reset_index(drop=True)

    def _build_vs_baselines_summary(self, selector_vs_baselines: pd.DataFrame) -> pd.DataFrame:
        if selector_vs_baselines.empty:
            return pd.DataFrame()
        summary = (
            selector_vs_baselines.groupby(["entity_name", "selection_family"], as_index=False)
            .agg(
                evaluated_folds=("observations", lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum())),
                observations=("observations", "sum"),
                average_actual_return=("average_actual_return", "mean"),
                profit_label_hit_rate=("profit_label_hit_rate", "mean"),
                top_1_avg_return=("top_1_avg_return", "mean"),
                top_3_avg_return=("top_3_avg_return", "mean"),
                top_1_profit_rate=("top_1_profit_rate", "mean"),
                top_3_profit_rate=("top_3_profit_rate", "mean"),
                avg_return_vs_fixed_best_global=("avg_return_vs_fixed_best_global", "mean"),
                profit_rate_vs_fixed_best_global=("profit_rate_vs_fixed_best_global", "mean"),
                selector_score_std=("selector_score_std", "mean"),
            )
        )
        fixed_best = summary[summary["entity_name"] == "fixed_best_global_setup"]
        if not fixed_best.empty:
            fixed_row = fixed_best.iloc[0]
            summary["beats_fixed_best_global_on_avg_return"] = (
                summary["average_actual_return"] > float(fixed_row["average_actual_return"])
            )
            summary["beats_fixed_best_global_on_profit_rate"] = (
                summary["profit_label_hit_rate"] > float(fixed_row["profit_label_hit_rate"])
            )
        else:
            summary["beats_fixed_best_global_on_avg_return"] = False
            summary["beats_fixed_best_global_on_profit_rate"] = False
        return summary.sort_values(["selection_family", "average_actual_return"], ascending=[True, False]).reset_index(drop=True)

    def _build_meta_selector_overview(
        self,
        fold_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in fold_results:
            performance = item["selector_performance"]
            if performance.empty:
                rows.append(
                    {
                        "fold_id": item["fold_id"],
                        "fold_number": item["fold_number"],
                        "selector_mode": "all",
                        "status": "insufficient_prior_history",
                        "rows_selected": 0,
                    }
                )
                continue
            for row in performance.to_dict(orient="records"):
                rows.append(
                    {
                        "fold_id": item["fold_id"],
                        "fold_number": item["fold_number"],
                        "selector_mode": row["entity_name"],
                        "status": row.get("status", "evaluated"),
                        "rows_selected": row.get("observations", 0),
                        "average_actual_return": row.get("average_actual_return"),
                        "profit_label_hit_rate": row.get("profit_label_hit_rate"),
                        "top_3_avg_return": row.get("top_3_avg_return"),
                    }
                )
        return pd.DataFrame(rows).sort_values(["fold_number", "selector_mode"]).reset_index(drop=True)

    def _build_overall_report(
        self,
        *,
        selector_stability_summary: pd.DataFrame,
        selector_regime_summary: pd.DataFrame,
        selector_vs_baselines_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        selector_rows = selector_vs_baselines_summary[
            selector_vs_baselines_summary["selection_family"] == "selector"
        ].copy()
        baseline_rows = selector_vs_baselines_summary[
            selector_vs_baselines_summary["selection_family"] == "baseline"
        ].copy()
        if not selector_rows.empty:
            best_selector = selector_rows.sort_values(
                ["average_actual_return", "profit_label_hit_rate"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    "component": "best selector mode",
                    "best_overall_choice": best_selector["entity_name"],
                    "stability_level": "medium" if float(best_selector["selector_score_std"]) < 0.15 else "low",
                    "supporting_evidence": (
                        f"{best_selector['entity_name']} avg_return={float(best_selector['average_actual_return']):.4f}, "
                        f"profit_rate={float(best_selector['profit_label_hit_rate']):.4f}"
                    ),
                    "caution_note": "Selector gains remain sample-sensitive across folds.",
                }
            )
            if not baseline_rows.empty:
                best_baseline = baseline_rows.sort_values(
                    ["average_actual_return", "profit_label_hit_rate"],
                    ascending=[False, False],
                ).iloc[0]
                selector_beats_baseline = float(best_selector["average_actual_return"]) > float(best_baseline["average_actual_return"])
                rows.append(
                    {
                        "component": "selector vs fixed baselines",
                        "best_overall_choice": "selector beats fixed baseline" if selector_beats_baseline else "fixed baseline still stronger",
                        "stability_level": "low",
                        "supporting_evidence": (
                            f"best_selector={best_selector['entity_name']} ({float(best_selector['average_actual_return']):.4f}) "
                            f"vs best_baseline={best_baseline['entity_name']} ({float(best_baseline['average_actual_return']):.4f})"
                        ),
                        "caution_note": "Treat any edge as provisional until more folds accumulate.",
                    }
                )
        if not selector_regime_summary.empty:
            for regime in self.config.regimes:
                regime_rows = selector_regime_summary[selector_regime_summary["regime"] == regime]
                if regime_rows.empty:
                    continue
                best_regime = regime_rows.sort_values(
                    ["average_actual_return", "profit_label_hit_rate"],
                    ascending=[False, False],
                ).iloc[0]
                rows.append(
                    {
                        "component": f"{regime} regime",
                        "best_overall_choice": best_regime["selector_mode"],
                        "stability_level": "medium" if float(best_regime["top_3_avg_return"]) > 0 else "low",
                        "supporting_evidence": (
                            f"{best_regime['selector_mode']} avg_return={float(best_regime['average_actual_return']):.4f}, "
                            f"top3={float(best_regime['top_3_avg_return']):.4f}"
                        ),
                        "caution_note": "Bear evidence is especially fragile when fold coverage is sparse." if regime == "bear" else "Regime benefit is not uniform across folds.",
                    }
                )
        if not selector_stability_summary.empty:
            top_frequency = selector_stability_summary.sort_values(
                ["folds_selected", "rows_selected"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    "component": "most frequent adaptive setup",
                    "best_overall_choice": top_frequency["candidate_label"],
                    "stability_level": "medium" if int(top_frequency["folds_selected"]) >= 3 else "low",
                    "supporting_evidence": (
                        f"selected in {int(top_frequency['folds_selected'])} folds and {int(top_frequency['rows_selected'])} rows"
                    ),
                    "caution_note": "Method-family stability is stronger than exact setup stability.",
                }
            )
        return pd.DataFrame(rows)

    def _render_summary_charts(
        self,
        *,
        selected_candidates: pd.DataFrame,
        selector_vs_baselines_summary: pd.DataFrame,
        meta_selector_overview: pd.DataFrame,
        regime_selection_summary: pd.DataFrame,
    ) -> None:
        self.charts_root.mkdir(parents=True, exist_ok=True)
        if not selected_candidates.empty:
            by_regime = (
                selected_candidates.groupby(["regime", "selected_horizon"], as_index=False)["ticker"]
                .count()
                .rename(columns={"ticker": "count"})
            )
            pivot = by_regime.pivot(index="selected_horizon", columns="regime", values="count").fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Selected Horizon Frequency By Regime")
            ax.set_ylabel("Rows Selected")
            fig.tight_layout()
            fig.savefig(self.charts_root / "horizon_switching_frequency_by_regime.png", dpi=150)
            plt.close(fig)

            by_model = (
                selected_candidates.groupby(["regime", "selected_model_name"], as_index=False)["ticker"]
                .count()
                .rename(columns={"ticker": "count"})
            )
            pivot = by_model.pivot(index="selected_model_name", columns="regime", values="count").fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Selected Model Frequency By Regime")
            ax.set_ylabel("Rows Selected")
            fig.tight_layout()
            fig.savefig(self.charts_root / "model_switching_frequency_by_regime.png", dpi=150)
            plt.close(fig)

        if not meta_selector_overview.empty and "average_actual_return" in meta_selector_overview.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for selector_mode, group in meta_selector_overview.groupby("selector_mode", dropna=False):
                if selector_mode == "all":
                    continue
                ax.plot(group["fold_number"], group["average_actual_return"], marker="o", label=selector_mode)
            ax.set_title("Selector Utility Over Folds")
            ax.set_xlabel("Fold Number")
            ax.set_ylabel("Average Actual Return")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.charts_root / "selector_utility_over_folds.png", dpi=150)
            plt.close(fig)

        if not selector_vs_baselines_summary.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(selector_vs_baselines_summary["entity_name"], selector_vs_baselines_summary["average_actual_return"])
            ax.set_title("Selector Vs Fixed Baselines")
            ax.set_ylabel("Average Actual Return")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(self.charts_root / "selector_vs_fixed_baseline_comparison.png", dpi=150)
            plt.close(fig)

        if not regime_selection_summary.empty and "fallback_used" in regime_selection_summary.columns:
            fallback = regime_selection_summary.groupby("selector_mode", as_index=False)["fallback_used"].sum()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(fallback["selector_mode"], fallback["fallback_used"])
            ax.set_title("Fallback Usage By Selector Mode")
            ax.set_ylabel("Fallback Count")
            fig.tight_layout()
            fig.savefig(self.charts_root / "fallback_usage_chart.png", dpi=150)
            plt.close(fig)

    def run(self) -> dict[str, Any]:
        folds = self._discover_folds()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            prior_folds = [item for item in folds if int(item["fold_number"]) < int(fold["fold_number"])]
            fold_results.append(self._run_fold(fold=fold, prior_folds=prior_folds))

        selected_candidates = pd.concat(
            [item["selected_candidates"] for item in fold_results if not item["selected_candidates"].empty],
            ignore_index=True,
        ) if any(not item["selected_candidates"].empty for item in fold_results) else pd.DataFrame()
        selector_vs_baselines = pd.concat(
            [item["selector_vs_baselines"] for item in fold_results if not item["selector_vs_baselines"].empty],
            ignore_index=True,
        ) if any(not item["selector_vs_baselines"].empty for item in fold_results) else pd.DataFrame()
        regime_selection_summary = pd.concat(
            [item["regime_selection_summary"] for item in fold_results if not item["regime_selection_summary"].empty],
            ignore_index=True,
        ) if any(not item["regime_selection_summary"].empty for item in fold_results) else pd.DataFrame()

        meta_selector_overview = self._build_meta_selector_overview(fold_results)
        selector_stability_summary = self._build_selector_stability_summary(selected_candidates)
        selector_regime_summary = self._build_selector_regime_summary(selected_candidates)
        selector_vs_baselines_summary = self._build_vs_baselines_summary(selector_vs_baselines)
        overall_meta_selector_report = self._build_overall_report(
            selector_stability_summary=selector_stability_summary,
            selector_regime_summary=selector_regime_summary,
            selector_vs_baselines_summary=selector_vs_baselines_summary,
        )

        summary_paths = {
            "meta_selector_overview": self.summary_root / "meta_selector_overview.csv",
            "selector_stability_summary": self.summary_root / "selector_stability_summary.csv",
            "selector_regime_summary": self.summary_root / "selector_regime_summary.csv",
            "selector_vs_baselines_summary": self.summary_root / "selector_vs_baselines_summary.csv",
            "overall_meta_selector_report": self.summary_root / "overall_meta_selector_report.csv",
        }
        meta_selector_overview.to_csv(summary_paths["meta_selector_overview"], index=False)
        selector_stability_summary.to_csv(summary_paths["selector_stability_summary"], index=False)
        selector_regime_summary.to_csv(summary_paths["selector_regime_summary"], index=False)
        selector_vs_baselines_summary.to_csv(summary_paths["selector_vs_baselines_summary"], index=False)
        overall_meta_selector_report.to_csv(summary_paths["overall_meta_selector_report"], index=False)

        self._render_summary_charts(
            selected_candidates=selected_candidates,
            selector_vs_baselines_summary=selector_vs_baselines_summary,
            meta_selector_overview=meta_selector_overview,
            regime_selection_summary=regime_selection_summary,
        )

        return {
            "fold_results": fold_results,
            "meta_selector_overview": meta_selector_overview,
            "selector_stability_summary": selector_stability_summary,
            "selector_regime_summary": selector_regime_summary,
            "selector_vs_baselines_summary": selector_vs_baselines_summary,
            "overall_meta_selector_report": overall_meta_selector_report,
            "summary_paths": {name: str(path) for name, path in summary_paths.items()},
        }
