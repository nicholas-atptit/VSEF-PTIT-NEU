"""Walk-forward regime-aware robustness orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ml.backtest.combined_signal import CombinedSignalAnalysisRunner, CombinedSignalConfig
from src.ml.backtest.dual_task import DualTaskBacktestConfig, DualTaskBacktestRunner
from src.ml.backtest.regime_aware_analysis import RegimeAwareAnalysisConfig, RegimeAwareAnalysisRunner


@dataclass(slots=True)
class WalkForwardRegimeRobustnessConfig:
    tickers: list[str] = field(default_factory=lambda: ["DGC", "ACB", "MWG", "HPG"])
    train_start: str = "2020-01-01"
    first_eval_start: str = "2023-01-01"
    last_eval_end: str = "2026-04-10"
    eval_window_days: int = 60
    step_size_days: int = 30
    max_folds: int = 4
    horizons: list[str] = field(default_factory=lambda: ["3d", "5d", "20d"])
    algorithms: list[str] = field(default_factory=lambda: ["cart", "xgboost", "lightgbm", "sarimax", "ets"])
    training_window_mode: str = "expanding"
    rolling_train_window_days: int | None = None
    output_dir: str = "artifacts/walk_forward_regime_robustness"
    benchmark_symbol: str = "VNINDEX"
    benchmark_source: str = "vnindex_or_market_proxy"
    benchmark_path: str | None = None
    regime_lookback_days: int = 20
    bull_threshold: float = 0.03
    bear_threshold: float = -0.03
    return_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02])
    probability_thresholds: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65])
    w_return: float = 0.5
    w_profit: float = 0.5
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    fold_retry_count: int = 2
    fold_retry_backoff_seconds: float = 2.0
    continue_on_fold_error: bool = True
    reuse_completed_folds: bool = True


def _stability_level_from_win_rate(win_rate: float) -> str:
    if pd.isna(win_rate):
        return "low"
    if float(win_rate) >= 0.70:
        return "high"
    if float(win_rate) >= 0.45:
        return "medium"
    return "low"


class WalkForwardRegimeRobustnessRunner:
    """Run multiple expanding/rolling folds through the existing stack and aggregate stability."""

    def __init__(self, config: WalkForwardRegimeRobustnessConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.summary_root = self.output_dir / "summary"
        self.charts_root = self.summary_root / "charts"

    def _resolve_primary_top_k(self) -> int:
        if 3 in self.config.top_k_values:
            return 3
        return int(self.config.top_k_values[0])

    def _generate_folds(self) -> list[dict[str, Any]]:
        train_start = pd.Timestamp(self.config.train_start).normalize()
        eval_start = pd.Timestamp(self.config.first_eval_start).normalize()
        last_eval_end = pd.Timestamp(self.config.last_eval_end).normalize()
        folds: list[dict[str, Any]] = []
        for fold_number in range(1, int(self.config.max_folds) + 1):
            eval_end = eval_start + pd.Timedelta(days=int(self.config.eval_window_days) - 1)
            if eval_end > last_eval_end:
                break
            if self.config.training_window_mode == "rolling":
                if not self.config.rolling_train_window_days:
                    raise ValueError("rolling_train_window_days must be provided for rolling mode")
                fold_train_start = max(
                    train_start,
                    eval_start - pd.Timedelta(days=int(self.config.rolling_train_window_days)),
                )
            else:
                fold_train_start = train_start
            train_end = eval_start - pd.Timedelta(days=1)
            if fold_train_start >= train_end:
                raise ValueError(
                    f"Invalid fold window: train_start={fold_train_start.date()} train_end={train_end.date()}"
                )
            folds.append(
                {
                    "fold_id": f"fold_{fold_number:03d}",
                    "fold_number": fold_number,
                    "train_start": str(fold_train_start.date()),
                    "train_end": str(train_end.date()),
                    "eval_start": str(eval_start.date()),
                    "eval_end": str(eval_end.date()),
                    "training_window_mode": self.config.training_window_mode,
                }
            )
            eval_start = eval_start + pd.Timedelta(days=int(self.config.step_size_days))
        if not folds:
            raise ValueError("No folds were generated from the requested walk-forward configuration")
        return folds

    def _fold_artifact_paths(self, fold_dir: Path) -> dict[str, Path]:
        return {
            "fold_config": fold_dir / "fold_config.json",
            "fold_summary": fold_dir / "fold_summary.csv",
            "regime_summary": fold_dir / "regime_summary.csv",
            "model_ranking": fold_dir / "model_ranking.csv",
            "combined_method_ranking": fold_dir / "combined_method_ranking.csv",
            "joined_sample": fold_dir / "joined_evaluation_sample.csv",
        }

    def _fold_is_complete(self, fold_dir: Path) -> bool:
        return all(path.exists() for path in self._fold_artifact_paths(fold_dir).values())

    def _load_existing_fold_result(self, fold: dict[str, Any]) -> dict[str, Any]:
        fold_dir = self.output_dir / fold["fold_id"]
        artifact_paths = self._fold_artifact_paths(fold_dir)
        fold_overview = pd.read_csv(artifact_paths["fold_summary"])
        fold_summary = pd.read_csv(artifact_paths["regime_summary"])
        model_ranking = pd.read_csv(artifact_paths["model_ranking"])
        combined_method_ranking = pd.read_csv(artifact_paths["combined_method_ranking"])
        joined_sample = pd.read_csv(artifact_paths["joined_sample"])
        overview_row = fold_overview.iloc[0].to_dict()
        overview_row.setdefault("status", "completed")
        overview_row.setdefault("attempts_used", 0)
        return {
            "status": str(overview_row["status"]),
            "fold": fold,
            "fold_dir": fold_dir,
            "fold_overview_row": overview_row,
            "fold_summary": fold_summary,
            "model_ranking": model_ranking,
            "combined_method_ranking": combined_method_ranking,
            "joined_sample": joined_sample,
        }

    def _build_failed_fold_result(
        self,
        fold: dict[str, Any],
        error: Exception,
        attempts_used: int,
    ) -> dict[str, Any]:
        fold_dir = self.output_dir / fold["fold_id"]
        fold_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = self._fold_artifact_paths(fold_dir)

        fold_overview_row = {
            **fold,
            "status": "failed",
            "attempts_used": int(attempts_used),
            "benchmark_source_used": "",
            "rows_analyzed": 0,
            "bull_count": 0,
            "sideway_count": 0,
            "bear_count": 0,
            "available_algorithms": "",
            "skipped_algorithms": "[]",
            "error_message": str(error),
        }
        empty_fold_summary = pd.DataFrame(
            columns=[
                "fold_id",
                "fold_number",
                "train_start",
                "train_end",
                "eval_start",
                "eval_end",
                "regime",
                "best_regression_model",
                "best_regression_horizon",
                "best_classification_model",
                "best_classification_horizon",
                "best_combined_method",
                "best_combined_model",
                "best_horizon",
                "key_takeaway",
            ]
        )
        empty_model_ranking = pd.DataFrame(
            columns=[
                "fold_id",
                "horizon",
                "regime",
                "model_name",
                "rank_regression_rmse",
                "rank_classification_f1",
                "overall_rank",
            ]
        )
        empty_combined_ranking = pd.DataFrame(
            columns=[
                "fold_id",
                "horizon",
                "regime",
                "model_name",
                "ranking_method",
                "profit_rate",
                "average_actual_return",
                "rank_in_regime",
            ]
        )
        empty_joined_sample = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon",
                "model_name",
                "prediction_date",
                "benchmark_date",
                "regime",
            ]
        )

        fold_config = {
            **asdict(self.config),
            **fold,
            "status": "failed",
            "attempts_used": int(attempts_used),
            "error_message": str(error),
            "analysis_only": True,
            "live_execution_enabled": False,
        }
        with artifact_paths["fold_config"].open("w", encoding="utf-8") as handle:
            json.dump(fold_config, handle, indent=2)
        pd.DataFrame([fold_overview_row]).to_csv(artifact_paths["fold_summary"], index=False)
        empty_fold_summary.to_csv(artifact_paths["regime_summary"], index=False)
        empty_model_ranking.to_csv(artifact_paths["model_ranking"], index=False)
        empty_combined_ranking.to_csv(artifact_paths["combined_method_ranking"], index=False)
        empty_joined_sample.to_csv(artifact_paths["joined_sample"], index=False)

        return {
            "status": "failed",
            "fold": fold,
            "fold_dir": fold_dir,
            "fold_overview_row": fold_overview_row,
            "fold_summary": empty_fold_summary,
            "model_ranking": empty_model_ranking,
            "combined_method_ranking": empty_combined_ranking,
            "joined_sample": empty_joined_sample,
        }

    def _run_fold_once(self, fold: dict[str, Any]) -> dict[str, Any]:
        fold_dir = self.output_dir / fold["fold_id"]
        dual_task_dir = fold_dir / "dual_task"
        combined_signal_dir = fold_dir / "combined_signal"
        regime_dir = fold_dir / "regime_aware"

        dual_task_config = DualTaskBacktestConfig(
            tickers=self.config.tickers,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            eval_start=fold["eval_start"],
            eval_end=fold["eval_end"],
            output_dir=str(dual_task_dir),
            horizons=self.config.horizons,
            algorithms=self.config.algorithms,
        )
        dual_task_result = DualTaskBacktestRunner(dual_task_config).run()

        combined_config = CombinedSignalConfig(
            dual_task_dir=str(dual_task_dir),
            output_dir=str(combined_signal_dir),
            horizons=self.config.horizons,
            return_thresholds=self.config.return_thresholds,
            probability_thresholds=self.config.probability_thresholds,
            w_return=self.config.w_return,
            w_profit=self.config.w_profit,
            top_k_values=self.config.top_k_values,
        )
        combined_result = CombinedSignalAnalysisRunner(combined_config).run()

        regime_config = RegimeAwareAnalysisConfig(
            dual_task_dir=str(dual_task_dir),
            combined_signal_dir=str(combined_signal_dir),
            output_dir=str(regime_dir),
            horizons=self.config.horizons,
            benchmark_symbol=self.config.benchmark_symbol,
            benchmark_source=self.config.benchmark_source,
            benchmark_path=self.config.benchmark_path,
            regime_lookback_days=self.config.regime_lookback_days,
            bull_threshold=self.config.bull_threshold,
            bear_threshold=self.config.bear_threshold,
            return_thresholds=self.config.return_thresholds,
            probability_thresholds=self.config.probability_thresholds,
            top_k_values=self.config.top_k_values,
        )
        regime_result = RegimeAwareAnalysisRunner(regime_config).run()

        regime_tables = [
            item["regime_labeled_signal_table"]
            for item in regime_result["horizons"].values()
        ]
        combined_regime_table = pd.concat(regime_tables, ignore_index=True).sort_values(
            ["horizon", "prediction_date", "model_name", "ticker"]
        ).reset_index(drop=True)
        regime_counts = (
            combined_regime_table["regime"].value_counts().reindex(["bull", "sideway", "bear"]).fillna(0).astype(int)
        )

        fold_overview_row = {
            **fold,
            "status": "completed",
            "benchmark_source_used": regime_result["benchmark_source_used"],
            "rows_analyzed": int(len(combined_regime_table)),
            "bull_count": int(regime_counts.get("bull", 0)),
            "sideway_count": int(regime_counts.get("sideway", 0)),
            "bear_count": int(regime_counts.get("bear", 0)),
            "available_algorithms": ",".join(dual_task_result.get("available_algorithms", [])),
            "skipped_algorithms": json.dumps(dual_task_result.get("skipped_algorithms", [])),
        }

        fold_summary = regime_result["overall_regime_summary"].copy()
        for key, value in fold.items():
            fold_summary.insert(0, key, value) if key not in fold_summary.columns else None
        fold_summary["benchmark_source_used"] = regime_result["benchmark_source_used"]
        fold_summary["bull_count"] = int(regime_counts.get("bull", 0))
        fold_summary["sideway_count"] = int(regime_counts.get("sideway", 0))
        fold_summary["bear_count"] = int(regime_counts.get("bear", 0))

        model_ranking = regime_result["regime_model_horizon_ranking"].copy()
        model_ranking.insert(0, "fold_id", fold["fold_id"])
        combined_method_ranking = regime_result["regime_combined_method_ranking"].copy()
        combined_method_ranking.insert(0, "fold_id", fold["fold_id"])
        joined_sample = combined_regime_table.head(200).copy()

        fold_config = {
            **asdict(self.config),
            **fold,
            "status": "completed",
            "analysis_only": True,
            "live_execution_enabled": False,
            "dual_task_output_dir": str(dual_task_dir),
            "combined_signal_output_dir": str(combined_signal_dir),
            "regime_aware_output_dir": str(regime_dir),
            "benchmark_source_used": regime_result["benchmark_source_used"],
        }
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "fold_config.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_config, handle, indent=2)
        pd.DataFrame([fold_overview_row]).to_csv(fold_dir / "fold_summary.csv", index=False)
        fold_summary.to_csv(fold_dir / "regime_summary.csv", index=False)
        model_ranking.to_csv(fold_dir / "model_ranking.csv", index=False)
        combined_method_ranking.to_csv(fold_dir / "combined_method_ranking.csv", index=False)
        joined_sample.to_csv(fold_dir / "joined_evaluation_sample.csv", index=False)

        return {
            "status": "completed",
            "fold": fold,
            "fold_dir": fold_dir,
            "fold_overview_row": fold_overview_row,
            "fold_summary": fold_summary,
            "model_ranking": model_ranking,
            "combined_method_ranking": combined_method_ranking,
            "joined_sample": joined_sample,
        }

    def _run_fold(self, fold: dict[str, Any]) -> dict[str, Any]:
        fold_dir = self.output_dir / fold["fold_id"]
        if self.config.reuse_completed_folds and self._fold_is_complete(fold_dir):
            existing_result = self._load_existing_fold_result(fold)
            if existing_result.get("status", "completed") == "completed":
                return existing_result

        max_attempts = int(self.config.fold_retry_count) + 1
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._run_fold_once(fold)
                result["fold_overview_row"]["attempts_used"] = attempt
                return result
            except Exception as error:
                last_error = error
                if attempt < max_attempts:
                    time.sleep(float(self.config.fold_retry_backoff_seconds) * attempt)
                else:
                    break

        if last_error is None:
            last_error = RuntimeError(f"Fold {fold['fold_id']} failed without an exception payload")
        failed_result = self._build_failed_fold_result(fold, last_error, attempts_used=max_attempts)
        if not self.config.continue_on_fold_error:
            raise last_error
        return failed_result

    def _build_model_stability_summary(
        self,
        fold_summaries: pd.DataFrame,
        model_rankings: pd.DataFrame,
    ) -> pd.DataFrame:
        regression_winners = fold_summaries[
            ["fold_id", "regime", "best_regression_model", "best_regression_horizon"]
        ].rename(
            columns={
                "best_regression_model": "model_name",
                "best_regression_horizon": "horizon",
            }
        )
        regression_winners["component"] = "regression_model"

        classification_winners = fold_summaries[
            ["fold_id", "regime", "best_classification_model", "best_classification_horizon"]
        ].rename(
            columns={
                "best_classification_model": "model_name",
                "best_classification_horizon": "horizon",
            }
        )
        classification_winners["component"] = "classification_model"

        winners = pd.concat([regression_winners, classification_winners], ignore_index=True)
        rows: list[dict[str, Any]] = []
        for (component, regime, model_name, horizon), winner_group in winners.groupby(
            ["component", "regime", "model_name", "horizon"],
            dropna=False,
        ):
            relevant_folds = winners[(winners["component"] == component) & (winners["regime"] == regime)]
            folds_observed = int(relevant_folds["fold_id"].nunique())
            rank_col = "rank_regression_rmse" if component == "regression_model" else "rank_classification_f1"
            ranking_subset = model_rankings[
                (model_rankings["regime"] == regime)
                & (model_rankings["model_name"] == model_name)
                & (model_rankings["horizon"] == horizon)
            ]
            rows.append(
                {
                    "component": component,
                    "regime": regime,
                    "model_name": model_name,
                    "horizon": horizon,
                    "fold_win_count": int(len(winner_group)),
                    "folds_observed": folds_observed,
                    "win_rate": (len(winner_group) / folds_observed) if folds_observed else np.nan,
                    "average_rank": float(ranking_subset[rank_col].mean()) if not ranking_subset.empty else np.nan,
                    "rank_std": float(ranking_subset[rank_col].std(ddof=0)) if not ranking_subset.empty else np.nan,
                }
            )
        summary = pd.DataFrame(rows)
        if summary.empty:
            return summary
        summary["stability_level"] = summary["win_rate"].apply(_stability_level_from_win_rate)
        return summary.sort_values(["component", "regime", "win_rate", "average_rank"], ascending=[True, True, False, True]).reset_index(drop=True)

    def _build_horizon_stability_summary(self, fold_summaries: pd.DataFrame) -> pd.DataFrame:
        horizon_choices = pd.concat(
            [
                fold_summaries[["fold_id", "regime", "best_regression_horizon"]]
                .rename(columns={"best_regression_horizon": "horizon"})
                .assign(component="regression_horizon"),
                fold_summaries[["fold_id", "regime", "best_classification_horizon"]]
                .rename(columns={"best_classification_horizon": "horizon"})
                .assign(component="classification_horizon"),
                fold_summaries[["fold_id", "regime", "best_horizon"]]
                .rename(columns={"best_horizon": "horizon"})
                .assign(component="combined_horizon"),
            ],
            ignore_index=True,
        )
        rows: list[dict[str, Any]] = []
        for (component, regime, horizon), group in horizon_choices.groupby(["component", "regime", "horizon"], dropna=False):
            total = int(
                horizon_choices[(horizon_choices["component"] == component) & (horizon_choices["regime"] == regime)]["fold_id"].nunique()
            )
            win_rate = (len(group) / total) if total else np.nan
            rows.append(
                {
                    "component": component,
                    "regime": regime,
                    "horizon": horizon,
                    "fold_win_count": int(len(group)),
                    "folds_observed": total,
                    "win_rate": win_rate,
                    "stability_level": _stability_level_from_win_rate(win_rate),
                }
            )
        return pd.DataFrame(rows).sort_values(["component", "regime", "win_rate"], ascending=[True, True, False]).reset_index(drop=True)

    def _build_regime_stability_summary(self, fold_summaries: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        component_maps = {
            "regression_model": "best_regression_model",
            "classification_model": "best_classification_model",
            "combined_method": "best_combined_method",
        }
        for regime in sorted(fold_summaries["regime"].unique()):
            regime_frame = fold_summaries[fold_summaries["regime"] == regime]
            fold_count = int(regime_frame["fold_id"].nunique())
            for component, column in component_maps.items():
                counts = regime_frame[column].value_counts()
                if counts.empty:
                    continue
                top_choice = counts.index[0]
                win_count = int(counts.iloc[0])
                win_rate = win_count / fold_count if fold_count else np.nan
                rows.append(
                    {
                        "regime": regime,
                        "component": component,
                        "most_frequent_choice": top_choice,
                        "win_count": win_count,
                        "fold_count": fold_count,
                        "unique_choices": int(counts.size),
                        "regime_consistency_score": win_rate,
                        "stability_level": _stability_level_from_win_rate(win_rate),
                    }
                )
        return pd.DataFrame(rows).sort_values(["regime", "component"]).reset_index(drop=True)

    def _build_combined_method_stability_summary(self, combined_rankings: pd.DataFrame) -> pd.DataFrame:
        winners = combined_rankings[combined_rankings["rank_in_regime"] == 1].copy()
        if winners.empty:
            return winners
        if "top_k" in winners.columns:
            winners = winners[winners["top_k"] == self._resolve_primary_top_k()].copy()
        slot_cols = ["fold_id", "regime", "horizon"]
        rows: list[dict[str, Any]] = []
        for (regime, ranking_method, model_name, horizon), group in winners.groupby(
            ["regime", "ranking_method", "model_name", "horizon"],
            dropna=False,
        ):
            relevant_slots = winners[winners["regime"] == regime][slot_cols].drop_duplicates()
            group_slots = group[slot_cols].drop_duplicates()
            folds_observed = int(len(relevant_slots))
            fold_win_count = int(len(group_slots))
            win_rate = (fold_win_count / folds_observed) if folds_observed else np.nan
            rows.append(
                {
                    "regime": regime,
                    "ranking_method": ranking_method,
                    "model_name": model_name,
                    "horizon": horizon,
                    "fold_win_count": fold_win_count,
                    "folds_observed": folds_observed,
                    "win_rate": win_rate,
                    "average_profit_rate": float(group.drop_duplicates(subset=slot_cols)["profit_rate"].mean()),
                    "average_actual_return": float(group.drop_duplicates(subset=slot_cols)["average_actual_return"].mean()),
                    "stability_level": _stability_level_from_win_rate(win_rate),
                }
            )
        return pd.DataFrame(rows).sort_values(["regime", "win_rate", "average_profit_rate"], ascending=[True, False, False]).reset_index(drop=True)

    def _build_overall_robustness_report(
        self,
        model_stability: pd.DataFrame,
        horizon_stability: pd.DataFrame,
        regime_stability: pd.DataFrame,
        combined_stability: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        model_specs = [
            ("regression model", model_stability[model_stability["component"] == "regression_model"]),
            ("classification model", model_stability[model_stability["component"] == "classification_model"]),
        ]
        for component, frame in model_specs:
            if frame.empty:
                continue
            aggregated = (
                frame.groupby("model_name", as_index=False)
                .agg(
                    fold_win_count=("fold_win_count", "sum"),
                    folds_observed=("folds_observed", "sum"),
                    average_rank=("average_rank", "mean"),
                )
            )
            aggregated["win_rate"] = aggregated["fold_win_count"] / aggregated["folds_observed"].replace(0, np.nan)
            ranked = aggregated.sort_values(["fold_win_count", "win_rate", "average_rank"], ascending=[False, False, True]).iloc[0]
            win_rate = float(ranked["win_rate"])
            rows.append(
                {
                    "component": component,
                    "best_overall_choice": ranked["model_name"],
                    "stability_level": _stability_level_from_win_rate(win_rate),
                    "supporting_evidence": f"Top choice won {int(ranked['fold_win_count'])} of {int(ranked['folds_observed'])} fold-regime slots",
                    "caution_note": "Results are mixed across folds; treat as provisional." if win_rate < 0.70 else "Still validate on future folds before claiming persistence.",
                }
            )

        if not combined_stability.empty:
            combined_agg = (
                combined_stability.groupby("ranking_method", as_index=False)
                .agg(
                    fold_win_count=("fold_win_count", "sum"),
                    folds_observed=("folds_observed", "sum"),
                    average_profit_rate=("average_profit_rate", "mean"),
                )
            )
            combined_agg["win_rate"] = combined_agg["fold_win_count"] / combined_agg["folds_observed"].replace(0, np.nan)
            ranked = combined_agg.sort_values(["fold_win_count", "win_rate", "average_profit_rate"], ascending=[False, False, False]).iloc[0]
            win_rate = float(ranked["win_rate"])
            rows.append(
                {
                    "component": "combined method",
                    "best_overall_choice": ranked["ranking_method"],
                    "stability_level": _stability_level_from_win_rate(win_rate),
                    "supporting_evidence": f"Top method won {int(ranked['fold_win_count'])} of {int(ranked['folds_observed'])} fold-regime-horizon slots",
                    "caution_note": "Results are mixed across folds; treat as provisional." if win_rate < 0.70 else "Still validate on future folds before claiming persistence.",
                }
            )

        combined_horizons = horizon_stability[horizon_stability["component"] == "combined_horizon"].copy()
        if not combined_horizons.empty:
            horizon_agg = (
                combined_horizons.groupby("horizon", as_index=False)
                .agg(
                    fold_win_count=("fold_win_count", "sum"),
                    folds_observed=("folds_observed", "sum"),
                )
            )
            horizon_agg["win_rate"] = horizon_agg["fold_win_count"] / horizon_agg["folds_observed"].replace(0, np.nan)
            ranked = horizon_agg.sort_values(["fold_win_count", "win_rate"], ascending=[False, False]).iloc[0]
            win_rate = float(ranked["win_rate"])
            rows.append(
                {
                    "component": "horizon",
                    "best_overall_choice": ranked["horizon"],
                    "stability_level": _stability_level_from_win_rate(win_rate),
                    "supporting_evidence": f"Top horizon won {int(ranked['fold_win_count'])} of {int(ranked['folds_observed'])} combined-horizon slots",
                    "caution_note": "Results are mixed across folds; treat as provisional." if win_rate < 0.70 else "Still validate on future folds before claiming persistence.",
                }
            )

        for regime in ["bull", "bear", "sideway"]:
            regime_rows = regime_stability[regime_stability["regime"] == regime]
            if regime_rows.empty:
                continue
            best = regime_rows.sort_values(["regime_consistency_score", "component"], ascending=[False, True]).iloc[0]
            rows.append(
                {
                    "component": f"{regime} regime setup",
                    "best_overall_choice": best["most_frequent_choice"],
                    "stability_level": best["stability_level"],
                    "supporting_evidence": f"{best['component']} repeated in {int(best['win_count'])}/{int(best['fold_count'])} fold summaries",
                    "caution_note": "Regime dependence is unstable." if best["regime_consistency_score"] < 0.60 else "Regime-specific pattern appears repeatable in this sample.",
                }
            )
        return pd.DataFrame(rows)

    def _render_summary_charts(
        self,
        fold_overview: pd.DataFrame,
        model_stability: pd.DataFrame,
        horizon_stability: pd.DataFrame,
        combined_stability: pd.DataFrame,
        model_rankings: pd.DataFrame,
    ) -> None:
        self.charts_root.mkdir(parents=True, exist_ok=True)

        if not model_stability.empty:
            winners = model_stability.groupby(["component", "model_name"], as_index=False)["fold_win_count"].sum()
            pivot = winners.pivot(index="model_name", columns="component", values="fold_win_count").fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Best Model Frequency Across Folds")
            ax.set_ylabel("Fold-Regime Wins")
            fig.tight_layout()
            fig.savefig(self.charts_root / "best_model_frequency_across_folds.png", dpi=150)
            plt.close(fig)

        if not horizon_stability.empty:
            winners = horizon_stability.groupby(["component", "horizon"], as_index=False)["fold_win_count"].sum()
            pivot = winners.pivot(index="horizon", columns="component", values="fold_win_count").fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Best Horizon Frequency Across Folds")
            ax.set_ylabel("Fold-Regime Wins")
            fig.tight_layout()
            fig.savefig(self.charts_root / "best_horizon_frequency_across_folds.png", dpi=150)
            plt.close(fig)

        if not model_rankings.empty:
            avg_rank = model_rankings.groupby("fold_id", as_index=False)["overall_rank"].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(avg_rank["fold_id"], avg_rank["overall_rank"], marker="o")
            ax.set_title("Average Rank by Fold")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Average Overall Rank")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(self.charts_root / "average_rank_by_fold.png", dpi=150)
            plt.close(fig)

        if not combined_stability.empty:
            combined_wins = combined_stability.groupby("ranking_method", as_index=False)["fold_win_count"].sum()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(combined_wins["ranking_method"], combined_wins["fold_win_count"])
            ax.set_title("Combined-Method Win Frequency")
            ax.set_ylabel("Fold-Regime Wins")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(self.charts_root / "combined_method_win_frequency.png", dpi=150)
            plt.close(fig)

        if not fold_overview.empty:
            stacked = fold_overview.set_index("fold_id")[["bull_count", "sideway_count", "bear_count"]]
            fig, ax = plt.subplots(figsize=(8, 4))
            stacked.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Regime Distribution by Fold")
            ax.set_ylabel("Observations")
            fig.tight_layout()
            fig.savefig(self.charts_root / "regime_distribution_by_fold.png", dpi=150)
            plt.close(fig)

    def run(self) -> dict[str, Any]:
        folds = self._generate_folds()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        fold_results = [self._run_fold(fold) for fold in folds]
        completed_results = [item for item in fold_results if item.get("status", "completed") == "completed"]
        if not completed_results:
            raise ValueError("All walk-forward folds failed; no completed folds are available for robustness summaries")

        fold_overview = pd.DataFrame([item["fold_overview_row"] for item in fold_results]).sort_values("fold_number").reset_index(drop=True)
        fold_summaries = pd.concat(
            [item["fold_summary"] for item in completed_results if not item["fold_summary"].empty],
            ignore_index=True,
        )
        model_rankings = pd.concat(
            [item["model_ranking"] for item in completed_results if not item["model_ranking"].empty],
            ignore_index=True,
        )
        combined_rankings = pd.concat(
            [item["combined_method_ranking"] for item in completed_results if not item["combined_method_ranking"].empty],
            ignore_index=True,
        )

        model_stability = self._build_model_stability_summary(fold_summaries, model_rankings)
        horizon_stability = self._build_horizon_stability_summary(fold_summaries)
        regime_stability = self._build_regime_stability_summary(fold_summaries)
        combined_stability = self._build_combined_method_stability_summary(combined_rankings)
        overall_report = self._build_overall_robustness_report(
            model_stability,
            horizon_stability,
            regime_stability,
            combined_stability,
        )

        summary_paths = {
            "fold_overview": self.summary_root / "fold_overview.csv",
            "model_stability_summary": self.summary_root / "model_stability_summary.csv",
            "horizon_stability_summary": self.summary_root / "horizon_stability_summary.csv",
            "regime_stability_summary": self.summary_root / "regime_stability_summary.csv",
            "combined_method_stability_summary": self.summary_root / "combined_method_stability_summary.csv",
            "overall_robustness_report": self.summary_root / "overall_robustness_report.csv",
        }
        fold_overview.to_csv(summary_paths["fold_overview"], index=False)
        model_stability.to_csv(summary_paths["model_stability_summary"], index=False)
        horizon_stability.to_csv(summary_paths["horizon_stability_summary"], index=False)
        regime_stability.to_csv(summary_paths["regime_stability_summary"], index=False)
        combined_stability.to_csv(summary_paths["combined_method_stability_summary"], index=False)
        overall_report.to_csv(summary_paths["overall_robustness_report"], index=False)
        self._render_summary_charts(fold_overview, model_stability, horizon_stability, combined_stability, model_rankings)

        return {
            "folds": fold_results,
            "fold_overview": fold_overview,
            "completed_folds": len(completed_results),
            "failed_folds": len(fold_results) - len(completed_results),
            "model_stability_summary": model_stability,
            "horizon_stability_summary": horizon_stability,
            "regime_stability_summary": regime_stability,
            "combined_method_stability_summary": combined_stability,
            "overall_robustness_report": overall_report,
            "summary_paths": {name: str(path) for name, path in summary_paths.items()},
        }
