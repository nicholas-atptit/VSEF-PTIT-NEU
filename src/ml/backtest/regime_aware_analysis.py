"""Regime-aware analysis for dual-task and combined-signal artifacts."""

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

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.backtest.dual_task import _compute_profit_classification_metrics
from src.ml.backtest.forward_return import _compute_error_metrics
from src.ml.data_loader import load_market_proxy
from src.ml.trainer import DualModelTrainer

PROBABILITY_BUCKET_BINS = [-1e-9, 0.50, 0.55, 0.60, 0.65, 1.000001]
PROBABILITY_BUCKET_LABELS = ["lt_0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65+"]


@dataclass(slots=True)
class RegimeAwareAnalysisConfig:
    dual_task_dir: str = "artifacts/dual_task"
    combined_signal_dir: str = "artifacts/combined_signal"
    output_dir: str = "artifacts/regime_aware_analysis"
    horizons: list[str] = field(default_factory=lambda: ["3d", "5d", "20d"])
    benchmark_symbol: str = "VNINDEX"
    benchmark_source: str = "vnindex_or_market_proxy"
    benchmark_path: str | None = None
    interval: str = "1D"
    regime_method: str = "rolling_return_threshold"
    regime_lookback_days: int = 20
    bull_threshold: float = 0.03
    bear_threshold: float = -0.03
    return_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02])
    probability_thresholds: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65])
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])


def _regime_from_return(value: float, *, bull_threshold: float, bear_threshold: float) -> str:
    if pd.isna(value):
        return "sideway"
    if float(value) > float(bull_threshold):
        return "bull"
    if float(value) < float(bear_threshold):
        return "bear"
    return "sideway"


class RegimeAwareAnalysisRunner:
    """Attach market regimes to saved evaluation outputs and summarize by regime."""

    def __init__(self, config: RegimeAwareAnalysisConfig) -> None:
        self.config = config
        self.dual_task_dir = Path(config.dual_task_dir).resolve()
        self.combined_signal_dir = Path(config.combined_signal_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.summary_root = self.output_dir / "summary"
        self.adapter = VnstockAdapter(symbol_list=[])

    def _resolve_primary_top_k(self) -> int:
        if 3 in self.config.top_k_values:
            return 3
        return int(self.config.top_k_values[0])

    def _load_horizon_inputs(
        self,
        horizon: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        regression_path = self.dual_task_dir / "regression" / horizon / "predicted_vs_actual.csv"
        classification_path = self.dual_task_dir / "classification" / horizon / "predicted_vs_actual.csv"
        combined_path = self.combined_signal_dir / horizon / "combined_signal_table.csv"
        for path in (regression_path, classification_path, combined_path):
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        regression_df = pd.read_csv(regression_path)
        classification_df = pd.read_csv(classification_path)
        combined_df = pd.read_csv(combined_path)

        for df in (regression_df, classification_df, combined_df):
            for date_col in [col for col in df.columns if "date" in col]:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
            for text_col in ("ticker", "model_name", "horizon"):
                if text_col in df.columns:
                    df[text_col] = df[text_col].astype(str).str.upper() if text_col == "ticker" else df[text_col].astype(str).str.lower()

        return regression_df, classification_df, combined_df

    def _load_joined_horizon(self, horizon: str) -> pd.DataFrame:
        regression_df, classification_df, combined_df = self._load_horizon_inputs(horizon)
        base_keys = ["date", "ticker", "model_name", "prediction_date", "target_date", "horizon", "horizon_days"]
        merged = regression_df.merge(
            classification_df,
            on=base_keys,
            how="inner",
            validate="1:1",
            suffixes=("", "_classification"),
        )
        merged = merged.merge(
            combined_df,
            on=["date", "ticker", "horizon", "model_name"],
            how="inner",
            validate="1:1",
            suffixes=("", "_combined"),
        )
        if merged.empty:
            raise ValueError(f"No merged rows found for horizon {horizon}")
        return merged.sort_values(["prediction_date", "model_name", "ticker"]).reset_index(drop=True)

    def _benchmark_fetch_bounds(self, tables: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
        prediction_dates = pd.concat(
            [table["prediction_date"] for table in tables.values()],
            ignore_index=True,
        )
        min_prediction = pd.Timestamp(prediction_dates.min()).normalize()
        max_prediction = pd.Timestamp(prediction_dates.max()).normalize()
        warmup_days = max(90, int(self.config.regime_lookback_days) * 5)
        return min_prediction - pd.Timedelta(days=warmup_days), max_prediction

    def _load_benchmark_history(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[pd.DataFrame, str]:
        benchmark_source_used = "market_proxy_csv"
        benchmark_df = pd.DataFrame()
        if self.config.benchmark_source in {"vnindex", "vnindex_or_market_proxy"}:
            try:
                fetched = self.adapter.get_index_ohlcv(
                    self.config.benchmark_symbol,
                    start_date=start_ts.strftime("%Y-%m-%d"),
                    end_date=end_ts.strftime("%Y-%m-%d"),
                    interval=self.config.interval,
                )
                if fetched is not None and not fetched.empty and "close" in fetched.columns:
                    benchmark_df = DualModelTrainer._normalize_ohlcv(fetched, ticker=self.config.benchmark_symbol)[
                        ["date", "close"]
                    ].copy()
                    benchmark_df["m_ret"] = benchmark_df["close"].pct_change().fillna(0.0)
                    benchmark_source_used = "vnstock_index"
            except Exception:
                benchmark_df = pd.DataFrame()

        if benchmark_df.empty:
            proxy = load_market_proxy(
                path=self.config.benchmark_path,
                start_date=start_ts.date(),
                end_date=end_ts.date(),
            )
            if proxy.empty:
                raise ValueError(
                    "Unable to load benchmark history from VNINDEX or market_proxy.csv"
                )
            proxy = proxy.copy()
            proxy["date"] = pd.to_datetime(proxy["date"], errors="coerce").dt.normalize()
            proxy["m_ret"] = pd.to_numeric(proxy["m_ret"], errors="coerce").fillna(0.0)
            proxy["close"] = (1.0 + proxy["m_ret"]).cumprod()
            benchmark_df = proxy[["date", "close", "m_ret"]].copy()

        benchmark_df = benchmark_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        benchmark_df["market_return_lookback"] = benchmark_df["close"] / benchmark_df["close"].shift(
            int(self.config.regime_lookback_days)
        ) - 1.0
        benchmark_df["market_volatility_lookback"] = benchmark_df["m_ret"].rolling(
            int(self.config.regime_lookback_days),
            min_periods=1,
        ).std()
        benchmark_df["regime"] = benchmark_df["market_return_lookback"].apply(
            lambda value: _regime_from_return(
                value,
                bull_threshold=self.config.bull_threshold,
                bear_threshold=self.config.bear_threshold,
            )
        )
        benchmark_df = benchmark_df.rename(columns={"date": "benchmark_date"})
        return benchmark_df, benchmark_source_used

    def _assign_regimes(self, frame: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        left = frame.sort_values("prediction_date").reset_index(drop=True).copy()
        right = benchmark_df.sort_values("benchmark_date").reset_index(drop=True).copy()
        left["prediction_date"] = pd.to_datetime(left["prediction_date"], errors="coerce").astype("datetime64[ns]")
        right["benchmark_date"] = pd.to_datetime(right["benchmark_date"], errors="coerce").astype("datetime64[ns]")
        merged = pd.merge_asof(
            left,
            right,
            left_on="prediction_date",
            right_on="benchmark_date",
            direction="backward",
        )
        merged["benchmark_stale_days"] = (
            pd.to_datetime(merged["prediction_date"], errors="coerce")
            - pd.to_datetime(merged["benchmark_date"], errors="coerce")
        ).dt.days
        merged["regime"] = merged["regime"].fillna("sideway")
        merged["benchmark_data_available"] = merged["benchmark_date"].notna()
        if ((merged["benchmark_date"].notna()) & (merged["benchmark_date"] > merged["prediction_date"])).any():
            raise ValueError("Regime assignment leakage detected: benchmark_date exceeded prediction_date")
        return merged.sort_values(["prediction_date", "model_name", "ticker"]).reset_index(drop=True)

    @staticmethod
    def _regression_by_regime(frame: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (regime, model_name), group in frame.groupby(["regime", "model_name"], dropna=False):
            metrics = _compute_error_metrics(group["actual_return"], group["predicted_return"])
            rows.append(
                {
                    "horizon": horizon,
                    "regime": regime,
                    "model_name": model_name,
                    "observations": int(len(group)),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "directional_accuracy": metrics.get("directional_accuracy"),
                    "average_actual_return": float(group["actual_return"].mean()),
                    "average_predicted_return": float(group["predicted_return"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(["regime", "model_name"]).reset_index(drop=True)

    @staticmethod
    def _classification_by_regime(frame: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (regime, model_name), group in frame.groupby(["regime", "model_name"], dropna=False):
            metrics = _compute_profit_classification_metrics(
                group["actual_profit_label"],
                group["predicted_profit_label"],
                group["predicted_profit_probability"],
            )
            rows.append(
                {
                    "horizon": horizon,
                    "regime": regime,
                    "model_name": model_name,
                    "observations": int(len(group)),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "positive_class_precision": metrics.get("positive_class_precision"),
                    "realized_profit_rate": float(group["actual_profit_label"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(["regime", "model_name"]).reset_index(drop=True)

    @staticmethod
    def _combined_signal_by_regime(frame: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        label_order = ["strong_positive", "moderate_positive", "weak_or_uncertain", "reject"]
        for (regime, model_name), model_frame in frame.groupby(["regime", "model_name"], dropna=False):
            profitable_total = int(model_frame["actual_profit_label"].sum())
            base_rate = float(model_frame["actual_profit_label"].mean()) if len(model_frame) else np.nan
            for label in label_order:
                bucket = model_frame[model_frame["combined_signal_label"] == label]
                observations = len(bucket)
                profit_count = int(bucket["actual_profit_label"].sum()) if observations else 0
                profit_rate = float(bucket["actual_profit_label"].mean()) if observations else np.nan
                rows.append(
                    {
                        "horizon": horizon,
                        "regime": regime,
                        "model_name": model_name,
                        "signal_bucket": label,
                        "observations": observations,
                        "avg_actual_return": float(bucket["actual_return"].mean()) if observations else np.nan,
                        "median_actual_return": float(bucket["actual_return"].median()) if observations else np.nan,
                        "hit_rate": float((bucket["actual_return"] > 0).mean()) if observations else np.nan,
                        "realized_profit_rate": profit_rate,
                        "positive_precision": profit_rate,
                        "recall_of_profitable_cases": (profit_count / profitable_total) if profitable_total else np.nan,
                        "lift_vs_base_rate": (profit_rate / base_rate)
                        if observations and pd.notna(base_rate) and base_rate > 0
                        else np.nan,
                    }
                )
            selected = model_frame[model_frame["combined_signal_label"].isin(["strong_positive", "moderate_positive"])]
            selected_profit = int(selected["actual_profit_label"].sum()) if not selected.empty else 0
            profit_rate = float(selected["actual_profit_label"].mean()) if not selected.empty else np.nan
            rows.append(
                {
                    "horizon": horizon,
                    "regime": regime,
                    "model_name": model_name,
                    "signal_bucket": "strong_or_moderate_positive",
                    "observations": int(len(selected)),
                    "avg_actual_return": float(selected["actual_return"].mean()) if not selected.empty else np.nan,
                    "median_actual_return": float(selected["actual_return"].median()) if not selected.empty else np.nan,
                    "hit_rate": float((selected["actual_return"] > 0).mean()) if not selected.empty else np.nan,
                    "realized_profit_rate": profit_rate,
                    "positive_precision": profit_rate,
                    "recall_of_profitable_cases": (selected_profit / profitable_total) if profitable_total else np.nan,
                    "lift_vs_base_rate": (profit_rate / base_rate)
                    if not selected.empty and pd.notna(base_rate) and base_rate > 0
                    else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values(["regime", "model_name", "signal_bucket"]).reset_index(drop=True)

    def _summarize_ranking(
        self,
        subset: pd.DataFrame,
        *,
        regime: str,
        model_name: str,
        ranking_method: str,
        score_column: str,
        horizon: str,
        top_k: int,
        return_threshold: float | None = None,
        probability_threshold: float | None = None,
    ) -> dict[str, Any]:
        picks: list[pd.DataFrame] = []
        for _, group in subset.groupby("ranking_group", sort=True):
            ranked = group.sort_values([score_column, "ticker", "date"], ascending=[False, True, True])
            picks.append(ranked.head(int(top_k)))
        if picks:
            selected = pd.concat(picks, ignore_index=True)
        else:
            selected = pd.DataFrame(columns=subset.columns)
        base_profit_rate = float(subset["actual_profit_label"].mean()) if len(subset) else np.nan
        profit_count = int(selected["actual_profit_label"].sum()) if not selected.empty else 0
        profitable_total = int(subset["actual_profit_label"].sum()) if len(subset) else 0
        profit_rate = float(selected["actual_profit_label"].mean()) if not selected.empty else np.nan
        return {
            "horizon": horizon,
            "regime": regime,
            "model_name": model_name,
            "ranking_method": ranking_method,
            "score_column": score_column,
            "top_k": int(top_k),
            "return_threshold": return_threshold,
            "probability_threshold": probability_threshold,
            "groups_covered": int(subset["ranking_group"].nunique()),
            "observations": int(len(selected)),
            "average_actual_return": float(selected["actual_return"].mean()) if not selected.empty else np.nan,
            "profit_rate": profit_rate,
            "precision_at_top_k": profit_rate,
            "recall_of_profitable_cases": (profit_count / profitable_total) if profitable_total else np.nan,
            "lift_vs_regime_base_rate": (profit_rate / base_profit_rate)
            if not selected.empty and pd.notna(base_profit_rate) and base_profit_rate > 0
            else np.nan,
        }

    def _ranking_by_regime(self, frame: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (regime, model_name), regime_frame in frame.groupby(["regime", "model_name"], dropna=False):
            for top_k in self.config.top_k_values:
                rows.append(
                    self._summarize_ranking(
                        regime_frame,
                        regime=regime,
                        model_name=model_name,
                        ranking_method="predicted_return",
                        score_column="predicted_return",
                        horizon=horizon,
                        top_k=int(top_k),
                    )
                )
                rows.append(
                    self._summarize_ranking(
                        regime_frame,
                        regime=regime,
                        model_name=model_name,
                        ranking_method="predicted_profit_probability",
                        score_column="predicted_profit_probability",
                        horizon=horizon,
                        top_k=int(top_k),
                    )
                )
                rows.append(
                    self._summarize_ranking(
                        regime_frame,
                        regime=regime,
                        model_name=model_name,
                        ranking_method="combined_weighted_linear",
                        score_column="combined_score",
                        horizon=horizon,
                        top_k=int(top_k),
                    )
                )
                rows.append(
                    self._summarize_ranking(
                        regime_frame,
                        regime=regime,
                        model_name=model_name,
                        ranking_method="combined_rank_based",
                        score_column="rank_based_joint_score",
                        horizon=horizon,
                        top_k=int(top_k),
                    )
                )
                for return_threshold in self.config.return_thresholds:
                    for probability_threshold in self.config.probability_thresholds:
                        gated = regime_frame[
                            (regime_frame["predicted_return"] > float(return_threshold))
                            & (regime_frame["predicted_profit_probability"] > float(probability_threshold))
                        ]
                        if gated.empty:
                            continue
                        rows.append(
                            self._summarize_ranking(
                                gated,
                                regime=regime,
                                model_name=model_name,
                                ranking_method="combined_weighted_linear_gated",
                                score_column="combined_score",
                                horizon=horizon,
                                top_k=int(top_k),
                                return_threshold=float(return_threshold),
                                probability_threshold=float(probability_threshold),
                            )
                        )
        return pd.DataFrame(rows).sort_values(
            ["regime", "model_name", "ranking_method", "top_k", "return_threshold", "probability_threshold"]
        ).reset_index(drop=True)

    @staticmethod
    def _calibration_by_regime(frame: pd.DataFrame, *, horizon: str) -> pd.DataFrame:
        working = frame.copy()
        working["probability_bucket"] = pd.cut(
            working["predicted_profit_probability"],
            bins=PROBABILITY_BUCKET_BINS,
            labels=PROBABILITY_BUCKET_LABELS,
            right=False,
            include_lowest=True,
        )
        rows: list[dict[str, Any]] = []
        for (regime, model_name), regime_frame in working.groupby(["regime", "model_name"], dropna=False):
            for bucket in PROBABILITY_BUCKET_LABELS:
                bucket_frame = regime_frame[regime_frame["probability_bucket"] == bucket]
                rows.append(
                    {
                        "horizon": horizon,
                        "regime": regime,
                        "model_name": model_name,
                        "probability_bucket": bucket,
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
        return pd.DataFrame(rows).sort_values(["regime", "model_name", "probability_bucket"]).reset_index(drop=True)

    def _render_horizon_charts(
        self,
        *,
        horizon: str,
        table: pd.DataFrame,
        regression_df: pd.DataFrame,
        classification_df: pd.DataFrame,
        combined_df: pd.DataFrame,
        ranking_df: pd.DataFrame,
        calibration_df: pd.DataFrame,
        horizon_dir: Path,
    ) -> None:
        charts_dir = horizon_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        regime_counts = table["regime"].value_counts().reindex(["bull", "sideway", "bear"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(regime_counts.index, regime_counts.values)
        ax.set_title(f"{horizon.upper()} Regime Distribution")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Observations")
        fig.tight_layout()
        fig.savefig(charts_dir / "regime_distribution.png", dpi=150)
        plt.close(fig)

        if not regression_df.empty:
            pivot = regression_df.pivot(index="regime", columns="model_name", values="rmse").reindex(
                ["bull", "sideway", "bear"]
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Regression RMSE by Regime")
            ax.set_ylabel("RMSE")
            ax.legend(title="Model")
            fig.tight_layout()
            fig.savefig(charts_dir / "regression_rmse_by_regime.png", dpi=150)
            plt.close(fig)

        if not classification_df.empty:
            pivot = classification_df.pivot(index="regime", columns="model_name", values="f1").reindex(
                ["bull", "sideway", "bear"]
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Classification F1 by Regime")
            ax.set_ylabel("F1")
            ax.legend(title="Model")
            fig.tight_layout()
            fig.savefig(charts_dir / "classification_f1_by_regime.png", dpi=150)
            plt.close(fig)

        strong = combined_df[combined_df["signal_bucket"] == "strong_positive"].copy()
        if not strong.empty:
            pivot = strong.pivot(index="regime", columns="model_name", values="avg_actual_return").reindex(
                ["bull", "sideway", "bear"]
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Strong-Positive Bucket Return by Regime")
            ax.set_ylabel("Average Actual Return")
            ax.legend(title="Model")
            fig.tight_layout()
            fig.savefig(charts_dir / "combined_signal_bucket_returns_by_regime.png", dpi=150)
            plt.close(fig)

        primary_top_k = self._resolve_primary_top_k()
        topk = ranking_df[
            (ranking_df["top_k"] == primary_top_k)
            & (ranking_df["ranking_method"] != "combined_weighted_linear_gated")
        ].copy()
        if not topk.empty:
            aggregated = (
                topk.groupby(["regime", "ranking_method"], as_index=False)["precision_at_top_k"].mean()
            )
            pivot = aggregated.pivot(index="regime", columns="ranking_method", values="precision_at_top_k").reindex(
                ["bull", "sideway", "bear"]
            )
            fig, ax = plt.subplots(figsize=(9, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{horizon.upper()} Precision@Top{primary_top_k} by Regime")
            ax.set_ylabel("Profit Rate")
            ax.legend(title="Ranking Method")
            fig.tight_layout()
            fig.savefig(charts_dir / "topk_ranking_comparison_by_regime.png", dpi=150)
            plt.close(fig)

        if not calibration_df.empty:
            aggregated = (
                calibration_df.groupby(["regime", "probability_bucket"], as_index=False)[
                    ["avg_predicted_probability", "realized_profit_rate"]
                ]
                .mean(numeric_only=True)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            for regime, subset in aggregated.groupby("regime"):
                ax.plot(
                    subset["avg_predicted_probability"],
                    subset["realized_profit_rate"],
                    marker="o",
                    label=regime,
                )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_title(f"{horizon.upper()} Probability Calibration by Regime")
            ax.set_xlabel("Average Predicted Probability")
            ax.set_ylabel("Realized Profit Rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(charts_dir / "calibration_by_regime.png", dpi=150)
            plt.close(fig)

    def _build_summary_tables(
        self,
        regression_frames: list[pd.DataFrame],
        classification_frames: list[pd.DataFrame],
        ranking_frames: list[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        regression_all = pd.concat(regression_frames, ignore_index=True) if regression_frames else pd.DataFrame()
        classification_all = pd.concat(classification_frames, ignore_index=True) if classification_frames else pd.DataFrame()
        ranking_all = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
        primary_top_k = self._resolve_primary_top_k()

        summary_rows: list[dict[str, Any]] = []
        model_horizon_rows: list[dict[str, Any]] = []
        combined_rows: list[dict[str, Any]] = []

        for regime in ["bull", "sideway", "bear"]:
            regression_regime = regression_all[regression_all["regime"] == regime].copy()
            classification_regime = classification_all[classification_all["regime"] == regime].copy()
            ranking_regime = ranking_all[
                (ranking_all["regime"] == regime)
                & (ranking_all["top_k"] == primary_top_k)
            ].copy()
            if regression_regime.empty or classification_regime.empty or ranking_regime.empty:
                continue

            best_regression = regression_regime.sort_values(["rmse", "directional_accuracy"], ascending=[True, False]).iloc[0]
            best_classification = classification_regime.sort_values(
                ["f1", "positive_class_precision", "realized_profit_rate"],
                ascending=[False, False, False],
            ).iloc[0]
            combined_candidates = ranking_regime[
                ranking_regime["ranking_method"].str.startswith("combined_")
            ].copy()
            best_combined = combined_candidates.sort_values(
                ["profit_rate", "average_actual_return"],
                ascending=[False, False],
            ).iloc[0]

            if best_combined["profit_rate"] >= 0.55:
                takeaway = "Combined ranking is selective and profitable in this regime."
            elif best_regression["rmse"] < regression_regime["rmse"].median():
                takeaway = "Return forecasts are relatively stable, but combined ranking remains mixed."
            else:
                takeaway = "Signals are unstable; rely on stricter filtering and probability checks."

            summary_rows.append(
                {
                    "regime": regime,
                    "best_regression_model": best_regression["model_name"],
                    "best_regression_horizon": best_regression["horizon"],
                    "best_classification_model": best_classification["model_name"],
                    "best_classification_horizon": best_classification["horizon"],
                    "best_combined_method": best_combined["ranking_method"],
                    "best_combined_model": best_combined["model_name"],
                    "best_horizon": best_combined["horizon"],
                    "key_takeaway": takeaway,
                }
            )

        if not regression_all.empty and not classification_all.empty and not ranking_all.empty:
            for regime in sorted(set(regression_all["regime"]).intersection(classification_all["regime"])):
                ranking_regime = ranking_all[
                    (ranking_all["regime"] == regime)
                    & (ranking_all["top_k"] == primary_top_k)
                ].copy()
                if ranking_regime.empty:
                    continue
                best_combined_per_model = (
                    ranking_regime[ranking_regime["ranking_method"].str.startswith("combined_")]
                    .sort_values(["profit_rate", "average_actual_return"], ascending=[False, False])
                    .drop_duplicates(["regime", "horizon", "model_name"])
                )
                merged = regression_all[regression_all["regime"] == regime].merge(
                    classification_all[classification_all["regime"] == regime],
                    on=["horizon", "regime", "model_name"],
                    how="inner",
                    suffixes=("_regression", "_classification"),
                ).merge(
                    best_combined_per_model[
                        [
                            "horizon",
                            "regime",
                            "model_name",
                            "ranking_method",
                            "profit_rate",
                            "average_actual_return",
                        ]
                    ],
                    on=["horizon", "regime", "model_name"],
                    how="left",
                )
                if merged.empty:
                    continue
                merged["rank_regression_rmse"] = merged["rmse"].rank(method="dense", ascending=True)
                merged["rank_classification_f1"] = merged["f1"].rank(method="dense", ascending=False)
                merged["rank_combined_profit_rate"] = merged["profit_rate"].rank(method="dense", ascending=False)
                merged["overall_rank"] = (
                    merged["rank_regression_rmse"]
                    + merged["rank_classification_f1"]
                    + merged["rank_combined_profit_rate"]
                ) / 3.0
                model_horizon_rows.extend(merged.to_dict(orient="records"))

                combined_ranked = ranking_regime.copy()
                combined_ranked["rank_in_regime"] = combined_ranked["profit_rate"].rank(
                    method="dense",
                    ascending=False,
                )
                combined_rows.extend(combined_ranked.to_dict(orient="records"))

        overall_regime_summary = pd.DataFrame(summary_rows)
        regime_model_horizon_ranking = pd.DataFrame(model_horizon_rows)
        if not regime_model_horizon_ranking.empty:
            regime_model_horizon_ranking = regime_model_horizon_ranking.sort_values(
                ["regime", "overall_rank", "horizon", "model_name"]
            ).reset_index(drop=True)
        regime_combined_method_ranking = pd.DataFrame(combined_rows)
        if not regime_combined_method_ranking.empty:
            regime_combined_method_ranking = regime_combined_method_ranking.sort_values(
                ["regime", "rank_in_regime", "horizon", "model_name", "ranking_method"]
            ).reset_index(drop=True)

        return overall_regime_summary, regime_model_horizon_ranking, regime_combined_method_ranking

    def run(self) -> dict[str, Any]:
        horizon_tables = {horizon: self._load_joined_horizon(horizon) for horizon in self.config.horizons}
        start_ts, end_ts = self._benchmark_fetch_bounds(horizon_tables)
        benchmark_df, benchmark_source_used = self._load_benchmark_history(start_ts, end_ts)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        horizon_results: dict[str, dict[str, Any]] = {}
        regression_frames: list[pd.DataFrame] = []
        classification_frames: list[pd.DataFrame] = []
        ranking_frames: list[pd.DataFrame] = []

        for horizon, horizon_table in horizon_tables.items():
            labeled = self._assign_regimes(horizon_table, benchmark_df)
            regression_by_regime = self._regression_by_regime(labeled, horizon=horizon)
            classification_by_regime = self._classification_by_regime(labeled, horizon=horizon)
            combined_signal_by_regime = self._combined_signal_by_regime(labeled, horizon=horizon)
            ranking_by_regime = self._ranking_by_regime(labeled, horizon=horizon)
            calibration_by_regime = self._calibration_by_regime(labeled, horizon=horizon)

            horizon_dir = self.output_dir / horizon
            horizon_dir.mkdir(parents=True, exist_ok=True)
            signal_path = horizon_dir / "regime_labeled_signal_table.csv"
            regression_path = horizon_dir / "regression_by_regime.csv"
            classification_path = horizon_dir / "classification_by_regime.csv"
            combined_path = horizon_dir / "combined_signal_by_regime.csv"
            ranking_path = horizon_dir / "ranking_by_regime.csv"
            calibration_path = horizon_dir / "calibration_by_regime.csv"
            config_path = horizon_dir / "run_config.json"

            labeled.to_csv(signal_path, index=False)
            regression_by_regime.to_csv(regression_path, index=False)
            classification_by_regime.to_csv(classification_path, index=False)
            combined_signal_by_regime.to_csv(combined_path, index=False)
            ranking_by_regime.to_csv(ranking_path, index=False)
            calibration_by_regime.to_csv(calibration_path, index=False)

            run_config = {
                "analysis_only": True,
                "live_execution_enabled": False,
                "horizon": horizon,
                "benchmark_symbol": self.config.benchmark_symbol,
                "benchmark_source_requested": self.config.benchmark_source,
                "benchmark_source_used": benchmark_source_used,
                "benchmark_path": self.config.benchmark_path,
                "benchmark_min_date": str(pd.Timestamp(benchmark_df["benchmark_date"].min()).date()),
                "benchmark_max_date": str(pd.Timestamp(benchmark_df["benchmark_date"].max()).date()),
                "regime_method": self.config.regime_method,
                "regime_lookback_days": int(self.config.regime_lookback_days),
                "bull_threshold": float(self.config.bull_threshold),
                "bear_threshold": float(self.config.bear_threshold),
                "top_k_values": [int(value) for value in self.config.top_k_values],
                "return_thresholds": [float(value) for value in self.config.return_thresholds],
                "probability_thresholds": [float(value) for value in self.config.probability_thresholds],
                "rows_analyzed": int(len(labeled)),
                "benchmark_alignment_safe": bool(
                    not ((labeled["benchmark_date"].notna()) & (labeled["benchmark_date"] > labeled["prediction_date"])).any()
                ),
            }
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, indent=2)

            self._render_horizon_charts(
                horizon=horizon,
                table=labeled,
                regression_df=regression_by_regime,
                classification_df=classification_by_regime,
                combined_df=combined_signal_by_regime,
                ranking_df=ranking_by_regime,
                calibration_df=calibration_by_regime,
                horizon_dir=horizon_dir,
            )

            regression_frames.append(regression_by_regime)
            classification_frames.append(classification_by_regime)
            ranking_frames.append(ranking_by_regime)

            horizon_results[horizon] = {
                "regime_labeled_signal_table": labeled,
                "regression_by_regime": regression_by_regime,
                "classification_by_regime": classification_by_regime,
                "combined_signal_by_regime": combined_signal_by_regime,
                "ranking_by_regime": ranking_by_regime,
                "calibration_by_regime": calibration_by_regime,
                "paths": {
                    "regime_labeled_signal_table": str(signal_path),
                    "regression_by_regime": str(regression_path),
                    "classification_by_regime": str(classification_path),
                    "combined_signal_by_regime": str(combined_path),
                    "ranking_by_regime": str(ranking_path),
                    "calibration_by_regime": str(calibration_path),
                    "run_config": str(config_path),
                },
            }

        overall_regime_summary, regime_model_horizon_ranking, regime_combined_method_ranking = self._build_summary_tables(
            regression_frames,
            classification_frames,
            ranking_frames,
        )
        overall_summary_path = self.summary_root / "overall_regime_summary.csv"
        model_horizon_path = self.summary_root / "regime_model_horizon_ranking.csv"
        combined_ranking_path = self.summary_root / "regime_combined_method_ranking.csv"
        overall_regime_summary.to_csv(overall_summary_path, index=False)
        regime_model_horizon_ranking.to_csv(model_horizon_path, index=False)
        regime_combined_method_ranking.to_csv(combined_ranking_path, index=False)

        return {
            "horizons": horizon_results,
            "overall_regime_summary": overall_regime_summary,
            "regime_model_horizon_ranking": regime_model_horizon_ranking,
            "regime_combined_method_ranking": regime_combined_method_ranking,
            "summary_paths": {
                "overall_regime_summary": str(overall_summary_path),
                "regime_model_horizon_ranking": str(model_horizon_path),
                "regime_combined_method_ranking": str(combined_ranking_path),
            },
            "benchmark_source_used": benchmark_source_used,
        }
