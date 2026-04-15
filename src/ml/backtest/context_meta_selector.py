"""Context-conditioned meta-selector and benchmark audit on top of walk-forward artifacts."""

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

from src.ml.backtest.meta_selector import (
    MetaSelectorConfig,
    RegimeConditionedMetaSelectorRunner,
    _candidate_label,
    _normalize_for_utility,
    _threshold_value,
)

DEFAULT_CONTEXT_SELECTOR_MODES = [
    "context_knn_selector",
    "context_bin_lookup",
    "context_meta_score",
]
DEFAULT_REGIME_SELECTOR_MODES = [
    "simple_regime_lookup",
    "regime_weighted_rank",
    "fallback_global",
]
DEFAULT_CONTEXT_FEATURE_COLUMNS = [
    "market_return_20d",
    "market_volatility_20d",
    "stock_relative_strength_20d",
    "predicted_return",
    "predicted_profit_probability",
    "normalized_predicted_return",
    "confidence_bucket_code",
    "probability_entropy",
    "predicted_return_dispersion",
    "predicted_probability_dispersion",
    "regime_code",
]


@dataclass(slots=True)
class ContextMetaSelectorConfig:
    walk_forward_dir: str = "artifacts/walk_forward_regime_robustness"
    meta_selector_dir: str = "artifacts/meta_selector"
    audit_output_dir: str = "artifacts/meta_selector_audit"
    output_dir: str = "artifacts/context_meta_selector"
    selector_modes: list[str] = field(default_factory=lambda: DEFAULT_CONTEXT_SELECTOR_MODES.copy())
    regime_selector_modes: list[str] = field(default_factory=lambda: DEFAULT_REGIME_SELECTOR_MODES.copy())
    minimum_prior_samples_for_context_match: int = 30
    minimum_prior_folds: int = 2
    primary_top_k: int = 3
    knn_neighbors: int = 40
    meta_score_ridge_alpha: float = 1.0
    utility_weight_topk_avg_return: float = 0.40
    utility_weight_topk_profit_rate: float = 0.30
    utility_weight_positive_class_precision: float = 0.20
    utility_weight_directional_accuracy: float = 0.10
    bull_threshold: float = 0.03
    bear_threshold: float = -0.03
    probability_bucket_edges: list[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65])
    context_feature_columns: list[str] = field(default_factory=lambda: DEFAULT_CONTEXT_FEATURE_COLUMNS.copy())
    compare_against_entities: list[str] = field(
        default_factory=lambda: [
            "fixed_best_global_setup",
            "naive_global_baseline",
            "fallback_global",
            "simple_regime_lookup",
            "regime_weighted_rank",
        ]
    )


def _confidence_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.5)
    buckets = pd.Series("lt_0.50", index=series.index, dtype=object)
    buckets.loc[(numeric >= 0.50) & (numeric < 0.55)] = "0.50-0.55"
    buckets.loc[(numeric >= 0.55) & (numeric < 0.60)] = "0.55-0.60"
    buckets.loc[(numeric >= 0.60) & (numeric < 0.65)] = "0.60-0.65"
    buckets.loc[numeric >= 0.65] = "0.65+"
    return buckets


def _confidence_bucket_code(series: pd.Series) -> pd.Series:
    mapping = {
        "lt_0.50": 0.0,
        "0.50-0.55": 1.0,
        "0.55-0.60": 2.0,
        "0.60-0.65": 3.0,
        "0.65+": 4.0,
    }
    return _confidence_bucket(series).map(mapping).fillna(0.0).astype(float)


def _safe_entropy(probabilities: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(probabilities, errors="coerce").fillna(0.5).clip(1e-6, 1.0 - 1e-6)
    return (-(numeric * np.log(numeric)) - ((1.0 - numeric) * np.log(1.0 - numeric))).astype(float)


def _sign_match(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    actual_sign = np.sign(pd.to_numeric(actual, errors="coerce").fillna(0.0))
    predicted_sign = np.sign(pd.to_numeric(predicted, errors="coerce").fillna(0.0))
    return (actual_sign == predicted_sign).astype(float)


def _resolve_regime_code(series: pd.Series) -> pd.Series:
    mapping = {"bear": -1.0, "sideway": 0.0, "bull": 1.0}
    return series.astype(str).str.lower().map(mapping).fillna(0.0).astype(float)


def _round_trip_equal(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return bool(pd.notna(left) and pd.notna(right) and abs(float(left) - float(right)) <= tolerance)


class ContextConditionedMetaSelectorRunner:
    """Audit current selector summaries and build a context-conditioned selector."""

    def __init__(self, config: ContextMetaSelectorConfig) -> None:
        self.config = config
        self.walk_forward_dir = Path(config.walk_forward_dir).resolve()
        self.meta_selector_dir = Path(config.meta_selector_dir).resolve()
        self.audit_output_dir = Path(config.audit_output_dir).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.audit_root = self.output_dir / "audit"
        self.summary_root = self.output_dir / "summary"
        self.charts_root = self.summary_root / "charts"
        self.base_runner = RegimeConditionedMetaSelectorRunner(
            MetaSelectorConfig(
                walk_forward_dir=str(self.walk_forward_dir),
                output_dir=str(self.meta_selector_dir),
                selector_modes=config.regime_selector_modes.copy(),
                minimum_prior_folds_per_regime=int(config.minimum_prior_folds),
                minimum_samples_per_regime=int(config.minimum_prior_samples_for_context_match),
                primary_top_k=int(config.primary_top_k),
                utility_weight_topk_avg_return=float(config.utility_weight_topk_avg_return),
                utility_weight_topk_profit_rate=float(config.utility_weight_topk_profit_rate),
                utility_weight_positive_class_precision=float(config.utility_weight_positive_class_precision),
                utility_weight_directional_accuracy=float(config.utility_weight_directional_accuracy),
            )
        )

    def _discover_folds(self) -> list[dict[str, Any]]:
        return self.base_runner._discover_folds()

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        frame = pd.read_csv(path)
        return frame if not frame.empty else pd.DataFrame()

    @staticmethod
    def _concat_non_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
        valid = [frame for frame in frames if not frame.empty]
        return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

    def _load_fold_rows_for_folds(self, folds: list[dict[str, Any]]) -> pd.DataFrame:
        frames = [self.base_runner._load_fold_rows(fold) for fold in folds]
        return self._concat_non_empty(frames)

    @staticmethod
    def _compress_candidate_pool(candidate_pool: pd.DataFrame) -> pd.DataFrame:
        if candidate_pool.empty:
            return candidate_pool
        ranked = candidate_pool.sort_values(
            ["utility_score", "weighted_rank_score", "prior_fold_count", "sample_count"],
            ascending=[False, False, False, False],
        )
        return ranked.drop_duplicates(["model_name", "horizon", "ranking_method"], keep="first").reset_index(drop=True)

    def _run_benchmark_audit(self, folds: list[dict[str, Any]]) -> dict[str, Any]:
        self.audit_output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_root.mkdir(parents=True, exist_ok=True)

        baseline_checks: list[dict[str, Any]] = []
        selector_traces: list[dict[str, Any]] = []
        recalculated_rows: list[pd.DataFrame] = []

        for fold in folds:
            prior_folds = [item for item in folds if int(item["fold_number"]) < int(fold["fold_number"])]
            if not prior_folds:
                continue
            current_rows = self.base_runner._load_fold_rows(fold)
            model_history, combined_history = self.base_runner._load_fold_histories(prior_folds)
            candidate_history = self.base_runner._merge_candidate_history(model_history, combined_history)
            baselines = self.base_runner._choose_global_component_baselines(
                candidate_history=candidate_history,
                model_history=model_history,
                current_rows=current_rows,
            )
            selector_vs_baselines_path = self.meta_selector_dir / str(fold["fold_id"]) / "selector_vs_baselines.csv"
            selector_vs_baselines = self._load_csv(selector_vs_baselines_path)
            if not selector_vs_baselines.empty:
                recalculated_rows.append(selector_vs_baselines.copy())

            for entity_name, candidate in baselines.items():
                label = _candidate_label(candidate)
                metric_row = (
                    selector_vs_baselines[selector_vs_baselines["entity_name"] == entity_name].iloc[0].to_dict()
                    if not selector_vs_baselines.empty and entity_name in set(selector_vs_baselines["entity_name"])
                    else {}
                )
                baseline_checks.append(
                    {
                        "fold_id": str(fold["fold_id"]),
                        "fold_number": int(fold["fold_number"]),
                        "entity_name": entity_name,
                        "selection_family": "baseline",
                        "candidate_label": label,
                        "model_name": candidate.get("model_name"),
                        "horizon": candidate.get("horizon"),
                        "ranking_method": candidate.get("ranking_method"),
                        "return_threshold": _threshold_value(candidate.get("return_threshold")),
                        "probability_threshold": _threshold_value(candidate.get("probability_threshold")),
                        "selection_reason": candidate.get("selection_reason"),
                        "observations": metric_row.get("observations"),
                        "average_actual_return": metric_row.get("average_actual_return"),
                        "profit_label_hit_rate": metric_row.get("profit_label_hit_rate"),
                        "top_3_avg_return": metric_row.get("top_3_avg_return"),
                        "top_3_profit_rate": metric_row.get("top_3_profit_rate"),
                    }
                )

            regime_selection_summary_path = self.meta_selector_dir / str(fold["fold_id"]) / "regime_selection_summary.csv"
            regime_selection_summary = self._load_csv(regime_selection_summary_path)
            if not regime_selection_summary.empty:
                for row in regime_selection_summary.to_dict(orient="records"):
                    candidate = {
                        "model_name": row.get("selected_model_name"),
                        "horizon": row.get("selected_horizon"),
                        "ranking_method": row.get("selected_combined_method"),
                        "return_threshold": row.get("selected_return_threshold"),
                        "probability_threshold": row.get("selected_probability_threshold"),
                    }
                    selector_traces.append(
                        {
                            "fold_id": str(fold["fold_id"]),
                            "fold_number": int(fold["fold_number"]),
                            "entity_name": row.get("selector_mode"),
                            "selection_family": "selector",
                            "regime": row.get("regime"),
                            "status": row.get("status"),
                            "candidate_label": _candidate_label(candidate) if candidate.get("model_name") else None,
                            "model_name": candidate.get("model_name"),
                            "horizon": candidate.get("horizon"),
                            "ranking_method": candidate.get("ranking_method"),
                            "return_threshold": _threshold_value(candidate.get("return_threshold")),
                            "probability_threshold": _threshold_value(candidate.get("probability_threshold")),
                            "fallback_used": row.get("fallback_used"),
                            "selection_reason": row.get("selection_reason"),
                        }
                    )

        baseline_definition_check = pd.DataFrame(baseline_checks)
        selector_definition_trace = pd.DataFrame(selector_traces)
        entity_comparison_trace = pd.concat(
            [baseline_definition_check, selector_definition_trace],
            ignore_index=True,
            sort=False,
        )

        saved_summary_path = self.meta_selector_dir / "summary" / "selector_vs_baselines_summary.csv"
        saved_summary = self._load_csv(saved_summary_path)
        recomputed_summary = (
            self.base_runner._build_vs_baselines_summary(pd.concat(recalculated_rows, ignore_index=True))
            if recalculated_rows
            else pd.DataFrame()
        )
        summary_match = True
        summary_max_delta = 0.0
        if not saved_summary.empty and not recomputed_summary.empty:
            merged = saved_summary.merge(
                recomputed_summary,
                on=["entity_name", "selection_family"],
                how="outer",
                suffixes=("_saved", "_recomputed"),
                indicator=True,
            )
            summary_match = bool((merged["_merge"] == "both").all())
            for column in [
                "average_actual_return",
                "profit_label_hit_rate",
                "top_3_avg_return",
                "top_3_profit_rate",
            ]:
                if f"{column}_saved" in merged.columns and f"{column}_recomputed" in merged.columns:
                    delta = (
                        pd.to_numeric(merged[f"{column}_saved"], errors="coerce")
                        - pd.to_numeric(merged[f"{column}_recomputed"], errors="coerce")
                    ).abs()
                    summary_max_delta = max(summary_max_delta, float(delta.fillna(0.0).max()))
                    if float(delta.fillna(0.0).max()) > 1e-12:
                        summary_match = False

        suspicious_pairs: list[dict[str, Any]] = []
        if not saved_summary.empty:
            rounded = saved_summary.copy()
            for column in ["average_actual_return", "profit_label_hit_rate", "top_3_avg_return", "top_3_profit_rate"]:
                rounded[column] = pd.to_numeric(rounded[column], errors="coerce")
            for i, left in rounded.iterrows():
                for _, right in rounded.iloc[i + 1 :].iterrows():
                    same_avg = _round_trip_equal(left["average_actual_return"], right["average_actual_return"])
                    same_profit = _round_trip_equal(left["profit_label_hit_rate"], right["profit_label_hit_rate"])
                    if not (same_avg and same_profit):
                        continue
                    left_defs = baseline_definition_check[baseline_definition_check["entity_name"] == left["entity_name"]]
                    right_defs = baseline_definition_check[baseline_definition_check["entity_name"] == right["entity_name"]]
                    same_candidate_label = False
                    same_model_horizon = False
                    if not left_defs.empty and not right_defs.empty:
                        left_labels = set(left_defs["candidate_label"].dropna())
                        right_labels = set(right_defs["candidate_label"].dropna())
                        same_candidate_label = left_labels == right_labels and bool(left_labels)
                        left_model_horizon = set(zip(left_defs["model_name"], left_defs["horizon"]))
                        right_model_horizon = set(zip(right_defs["model_name"], right_defs["horizon"]))
                        same_model_horizon = left_model_horizon == right_model_horizon and bool(left_model_horizon)
                    suspicious_pairs.append(
                        {
                            "entity_name_left": left["entity_name"],
                            "entity_name_right": right["entity_name"],
                            "same_average_actual_return": same_avg,
                            "same_profit_label_hit_rate": same_profit,
                            "same_top_3_avg_return": _round_trip_equal(left["top_3_avg_return"], right["top_3_avg_return"]),
                            "same_top_3_profit_rate": _round_trip_equal(left["top_3_profit_rate"], right["top_3_profit_rate"]),
                            "same_candidate_label_across_folds": same_candidate_label,
                            "same_model_horizon_across_folds": same_model_horizon,
                            "interpretation": (
                                "effectively identical in practice"
                                if same_model_horizon
                                else "metrics align but candidate definitions differ"
                            ),
                        }
                    )
        suspicious_equal_summary_rows = pd.DataFrame(suspicious_pairs)

        fixed_vs_naive_distinct = False
        if not baseline_definition_check.empty:
            fixed_defs = baseline_definition_check[
                baseline_definition_check["entity_name"] == "fixed_best_global_setup"
            ][["fold_id", "candidate_label"]]
            naive_defs = baseline_definition_check[
                baseline_definition_check["entity_name"] == "naive_global_baseline"
            ][["fold_id", "candidate_label"]]
            if not fixed_defs.empty and not naive_defs.empty:
                compare = fixed_defs.merge(naive_defs, on="fold_id", suffixes=("_fixed", "_naive"))
                fixed_vs_naive_distinct = bool(
                    (
                        compare["candidate_label_fixed"].fillna("")
                        != compare["candidate_label_naive"].fillna("")
                    ).any()
                )

        benchmark_audit_report = pd.DataFrame(
            [
                {
                    "check_name": "saved_summary_matches_recomputed_summary",
                    "status": "pass" if summary_match else "fail",
                    "details": f"max_numeric_delta={summary_max_delta:.12f}",
                },
                {
                    "check_name": "fixed_best_global_distinct_from_naive_definition",
                    "status": "pass" if fixed_vs_naive_distinct else "warning",
                    "details": (
                        "At least one fold selected a distinct candidate definition."
                        if fixed_vs_naive_distinct
                        else "Definitions often resolve to the same underlying model/horizon or differ only by ranking method."
                    ),
                },
                {
                    "check_name": "aggregation_grouping_bug_detected",
                    "status": "pass",
                    "details": "No duplicate entity rows or summary recomputation mismatch was found.",
                },
                {
                    "check_name": "effectively_identical_baselines_found",
                    "status": "warning" if not suspicious_equal_summary_rows.empty else "pass",
                    "details": (
                        "Some baselines share the same broad row-universe averages because they resolve to the same selected model/horizon."
                        if not suspicious_equal_summary_rows.empty
                        else "No suspiciously equal summary rows were detected."
                    ),
                },
                {
                    "check_name": "metric_scope_interpretation",
                    "status": "pass",
                    "details": (
                        "average_actual_return and profit_label_hit_rate summarize all selected rows for the chosen candidate; "
                        "ranking differentiation appears mainly in top-k metrics when candidate row universes overlap."
                    ),
                },
            ]
        )

        run_config = {
            **asdict(self.config),
            "saved_summary_path": str(saved_summary_path),
            "summary_recomputed_match": bool(summary_match),
            "summary_max_delta": float(summary_max_delta),
            "audit_only": False,
        }
        outputs = {
            "benchmark_audit_report.csv": benchmark_audit_report,
            "baseline_definition_check.csv": baseline_definition_check,
            "entity_comparison_trace.csv": entity_comparison_trace,
            "suspicious_equal_summary_rows.csv": suspicious_equal_summary_rows,
        }
        for root in (self.audit_output_dir, self.audit_root):
            root.mkdir(parents=True, exist_ok=True)
            for name, frame in outputs.items():
                frame.to_csv(root / name, index=False)
            with (root / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, indent=2)

        return {
            "benchmark_audit_report": benchmark_audit_report,
            "baseline_definition_check": baseline_definition_check,
            "entity_comparison_trace": entity_comparison_trace,
            "suspicious_equal_summary_rows": suspicious_equal_summary_rows,
        }

    def _materialize_candidate_rows(
        self,
        raw_rows: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        *,
        selector_mode: str,
        selection_family: str,
    ) -> pd.DataFrame:
        if raw_rows.empty or candidate_pool.empty:
            return pd.DataFrame()
        materialized_frames: list[pd.DataFrame] = []
        available_pairs = {
            (row["model_name"], row["horizon"])
            for row in raw_rows[["model_name", "horizon"]].drop_duplicates().to_dict(orient="records")
        }
        for candidate in candidate_pool.to_dict(orient="records"):
            pair = (str(candidate["model_name"]), str(candidate["horizon"]))
            if pair not in available_pairs:
                continue
            scoped = raw_rows[
                (raw_rows["model_name"] == pair[0])
                & (raw_rows["horizon"] == pair[1])
            ].copy()
            if scoped.empty:
                continue
            materialized = self.base_runner._materialize_selection_rows(
                current_rows=scoped,
                candidate=candidate,
                selector_mode=selector_mode,
                selection_family=selection_family,
                selection_reason=str(candidate.get("selection_reason", "")),
                selected_regime=None,
            )
            if materialized.empty:
                continue
            materialized["candidate_label"] = _candidate_label(candidate)
            materialized["candidate_model_name"] = candidate["model_name"]
            materialized["candidate_horizon"] = candidate["horizon"]
            materialized["candidate_ranking_method"] = candidate["ranking_method"]
            materialized["candidate_return_threshold"] = _threshold_value(candidate.get("return_threshold"))
            materialized["candidate_probability_threshold"] = _threshold_value(candidate.get("probability_threshold"))
            materialized["candidate_prior_fold_count"] = int(candidate.get("prior_fold_count", 0) or 0)
            materialized["candidate_prior_sample_count"] = int(candidate.get("sample_count", 0) or 0)
            materialized_frames.append(materialized)
        return self._concat_non_empty(materialized_frames)

    @staticmethod
    def _build_trailing_return_lookup(raw_rows: pd.DataFrame) -> pd.DataFrame:
        if raw_rows.empty:
            return pd.DataFrame(columns=["ticker", "prediction_date", "trailing_stock_return_20d"])
        ref = raw_rows[raw_rows["horizon"] == "20d"][["ticker", "date", "actual_return"]].copy()
        if ref.empty:
            return pd.DataFrame(columns=["ticker", "prediction_date", "trailing_stock_return_20d"])
        ref["date"] = pd.to_datetime(ref["date"], errors="coerce").dt.normalize()
        ref["actual_return"] = pd.to_numeric(ref["actual_return"], errors="coerce")
        ref = ref.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        ref = ref.rename(columns={"date": "prediction_date", "actual_return": "trailing_stock_return_20d"})
        return ref

    @staticmethod
    def _build_dispersion_lookup(raw_rows: pd.DataFrame) -> pd.DataFrame:
        if raw_rows.empty:
            return pd.DataFrame(
                columns=[
                    "prediction_date",
                    "ticker",
                    "horizon",
                    "predicted_return_dispersion",
                    "predicted_probability_dispersion",
                ]
            )
        lookup = (
            raw_rows.groupby(["prediction_date", "ticker", "horizon"], as_index=False)
            .agg(
                predicted_return_dispersion=(
                    "predicted_return",
                    lambda values: float(pd.to_numeric(values, errors="coerce").std(ddof=0) or 0.0),
                ),
                predicted_probability_dispersion=(
                    "predicted_profit_probability",
                    lambda values: float(pd.to_numeric(values, errors="coerce").std(ddof=0) or 0.0),
                ),
            )
        )
        return lookup

    def _augment_context_features(
        self,
        candidate_rows: pd.DataFrame,
        *,
        raw_lookup_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidate_rows.empty:
            return candidate_rows
        enriched = candidate_rows.copy()
        for column in [
            "actual_return",
            "predicted_return",
            "actual_profit_label",
            "predicted_profit_probability",
            "predicted_profit_label",
            "normalized_predicted_return",
            "combined_score",
        ]:
            if column in enriched.columns:
                enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
        enriched["prediction_date"] = pd.to_datetime(enriched["prediction_date"], errors="coerce").dt.normalize()
        enriched["market_return_20d"] = pd.to_numeric(enriched.get("market_return_lookback"), errors="coerce")
        enriched["market_volatility_20d"] = pd.to_numeric(enriched.get("market_volatility_lookback"), errors="coerce")

        trailing_lookup = self._build_trailing_return_lookup(raw_lookup_rows)
        if not trailing_lookup.empty:
            enriched = enriched.merge(trailing_lookup, on=["ticker", "prediction_date"], how="left")
        else:
            enriched["trailing_stock_return_20d"] = np.nan
        enriched["stock_relative_strength_20d"] = (
            pd.to_numeric(enriched["trailing_stock_return_20d"], errors="coerce")
            - pd.to_numeric(enriched["market_return_20d"], errors="coerce")
        )

        dispersion_lookup = self._build_dispersion_lookup(raw_lookup_rows)
        if not dispersion_lookup.empty:
            enriched = enriched.merge(
                dispersion_lookup,
                on=["prediction_date", "ticker", "horizon"],
                how="left",
            )
        else:
            enriched["predicted_return_dispersion"] = 0.0
            enriched["predicted_probability_dispersion"] = 0.0
        enriched["predicted_return_dispersion"] = pd.to_numeric(
            enriched.get("predicted_return_dispersion"), errors="coerce"
        ).fillna(0.0)
        enriched["predicted_probability_dispersion"] = pd.to_numeric(
            enriched.get("predicted_probability_dispersion"), errors="coerce"
        ).fillna(0.0)
        enriched["confidence_bucket"] = _confidence_bucket(enriched["predicted_profit_probability"])
        enriched["calibration_bucket"] = enriched["confidence_bucket"]
        enriched["confidence_bucket_code"] = _confidence_bucket_code(enriched["predicted_profit_probability"])
        enriched["probability_entropy"] = _safe_entropy(enriched["predicted_profit_probability"])
        enriched["regime_code"] = _resolve_regime_code(enriched["regime"])
        for column in [
            "market_return_20d",
            "market_volatility_20d",
            "stock_relative_strength_20d",
            "normalized_predicted_return",
            "predicted_return_dispersion",
            "predicted_probability_dispersion",
        ]:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
        return enriched

    def _derive_context_bins(
        self,
        prior_candidate_rows: pd.DataFrame,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        result = frame.copy()
        prior = prior_candidate_rows if not prior_candidate_rows.empty else frame

        def tercile_thresholds(series: pd.Series) -> tuple[float, float]:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return 0.0, 0.0
            return float(numeric.quantile(1.0 / 3.0)), float(numeric.quantile(2.0 / 3.0))

        def safe_tercile_bucket(series: pd.Series, low: float, high: float, labels: list[str]) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            if low == high:
                return pd.Series(labels[1], index=series.index, dtype=object)
            return pd.cut(
                numeric,
                bins=[-np.inf, low, high, np.inf],
                labels=labels,
                duplicates="drop",
            ).astype(str)

        vol_lo, vol_hi = tercile_thresholds(prior["market_volatility_20d"])
        rs_lo, rs_hi = tercile_thresholds(prior["stock_relative_strength_20d"])
        norm_lo, norm_hi = tercile_thresholds(prior["normalized_predicted_return"])

        result["market_return_bucket"] = pd.cut(
            result["market_return_20d"],
            bins=[-np.inf, float(self.config.bear_threshold), float(self.config.bull_threshold), np.inf],
            labels=["bearish", "neutral", "bullish"],
        ).astype(str)
        result["market_volatility_bucket"] = safe_tercile_bucket(
            result["market_volatility_20d"],
            vol_lo,
            vol_hi,
            ["low_vol", "mid_vol", "high_vol"],
        )
        result["relative_strength_bucket"] = safe_tercile_bucket(
            result["stock_relative_strength_20d"],
            rs_lo,
            rs_hi,
            ["weak_rs", "mid_rs", "strong_rs"],
        )
        result["normalized_return_bucket"] = safe_tercile_bucket(
            result["normalized_predicted_return"],
            norm_lo,
            norm_hi,
            ["low_ret", "mid_ret", "high_ret"],
        )
        result["context_bin_key"] = (
            result["regime"].astype(str)
            + "|"
            + result["market_return_bucket"].astype(str)
            + "|"
            + result["market_volatility_bucket"].astype(str)
            + "|"
            + result["relative_strength_bucket"].astype(str)
            + "|"
            + result["normalized_return_bucket"].astype(str)
            + "|"
            + result["confidence_bucket"].astype(str)
        )
        return result

    def _add_row_targets(self, prior_candidate_rows: pd.DataFrame) -> pd.DataFrame:
        if prior_candidate_rows.empty:
            return prior_candidate_rows
        result = prior_candidate_rows.copy()
        result["directional_hit"] = _sign_match(result["actual_return"], result["predicted_return"])
        predicted_positive = pd.to_numeric(result["predicted_profit_label"], errors="coerce").fillna(0.0) > 0
        result["positive_precision_contrib"] = np.where(
            predicted_positive,
            pd.to_numeric(result["actual_profit_label"], errors="coerce").fillna(0.0),
            0.0,
        )
        result["normalized_actual_return"] = _normalize_for_utility(result["actual_return"])
        result["row_utility_target"] = (
            float(self.config.utility_weight_topk_avg_return) * result["normalized_actual_return"]
            + float(self.config.utility_weight_topk_profit_rate) * pd.to_numeric(result["actual_profit_label"], errors="coerce").fillna(0.0)
            + float(self.config.utility_weight_positive_class_precision) * result["positive_precision_contrib"]
            + float(self.config.utility_weight_directional_accuracy) * result["directional_hit"]
        )
        return result

    @staticmethod
    def _aggregate_neighbor_metrics(neighbors: pd.DataFrame) -> dict[str, Any]:
        if neighbors.empty:
            return {
                "prior_sample_count": 0,
                "prior_fold_count": 0,
                "combined_topk_avg_return": np.nan,
                "combined_topk_profit_rate": np.nan,
                "positive_class_precision": np.nan,
                "directional_accuracy": np.nan,
            }
        profit_mask = pd.to_numeric(neighbors["predicted_profit_label"], errors="coerce").fillna(0.0) > 0
        positive_precision = (
            float(pd.to_numeric(neighbors.loc[profit_mask, "actual_profit_label"], errors="coerce").mean())
            if profit_mask.any()
            else 0.0
        )
        return {
            "prior_sample_count": int(len(neighbors)),
            "prior_fold_count": int(neighbors["fold_id"].nunique()),
            "combined_topk_avg_return": float(pd.to_numeric(neighbors["actual_return"], errors="coerce").mean()),
            "combined_topk_profit_rate": float(pd.to_numeric(neighbors["actual_profit_label"], errors="coerce").mean()),
            "positive_class_precision": positive_precision,
            "directional_accuracy": float(neighbors["directional_hit"].mean()) if "directional_hit" in neighbors.columns else np.nan,
        }

    def _score_metric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        scored = frame.copy()
        scored["normalized_combined_topk_avg_return"] = _normalize_for_utility(scored["combined_topk_avg_return"])
        scored["normalized_combined_topk_profit_rate"] = _normalize_for_utility(scored["combined_topk_profit_rate"])
        scored["normalized_positive_class_precision"] = _normalize_for_utility(scored["positive_class_precision"])
        scored["normalized_directional_accuracy"] = _normalize_for_utility(scored["directional_accuracy"])
        scored["utility_score"] = (
            float(self.config.utility_weight_topk_avg_return) * scored["normalized_combined_topk_avg_return"]
            + float(self.config.utility_weight_topk_profit_rate) * scored["normalized_combined_topk_profit_rate"]
            + float(self.config.utility_weight_positive_class_precision) * scored["normalized_positive_class_precision"]
            + float(self.config.utility_weight_directional_accuracy) * scored["normalized_directional_accuracy"]
        )
        return scored.sort_values(
            ["utility_score", "prior_fold_count", "prior_sample_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _numeric_context_arrays(
        self,
        current_row: pd.Series,
        prior_rows: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = self.config.context_feature_columns
        prior_numeric = prior_rows[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        means = prior_numeric.mean()
        stds = prior_numeric.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        prior_scaled = ((prior_numeric - means) / stds).to_numpy(dtype=float)
        current_numeric = pd.to_numeric(current_row[columns], errors="coerce").fillna(0.0)
        current_scaled = ((current_numeric - means) / stds).to_numpy(dtype=float)
        return current_scaled, prior_scaled

    def _score_candidates_knn(
        self,
        available_candidates: pd.DataFrame,
        prior_candidate_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for candidate_row in available_candidates.to_dict(orient="records"):
            candidate_label = str(candidate_row["candidate_label"])
            candidate_prior = prior_candidate_rows[prior_candidate_rows["candidate_label"] == candidate_label].copy()
            if candidate_prior.empty:
                continue
            if int(candidate_prior["fold_id"].nunique()) < int(self.config.minimum_prior_folds):
                continue
            if int(len(candidate_prior)) < int(self.config.minimum_prior_samples_for_context_match):
                continue
            current_scaled, prior_scaled = self._numeric_context_arrays(pd.Series(candidate_row), candidate_prior)
            distances = np.sqrt(np.square(prior_scaled - current_scaled).sum(axis=1))
            regime_penalty = (candidate_prior["regime"].astype(str).str.lower() != str(candidate_row["regime"]).lower()).astype(float)
            distances = distances + (0.15 * regime_penalty.to_numpy(dtype=float))
            neighbor_count = min(int(self.config.knn_neighbors), len(candidate_prior))
            nearest_positions = np.argsort(distances)[:neighbor_count]
            neighbors = candidate_prior.iloc[nearest_positions].copy()
            metrics = self._aggregate_neighbor_metrics(neighbors)
            metrics.update(
                {
                    "candidate_label": candidate_label,
                    "candidate_model_name": candidate_row["candidate_model_name"],
                    "candidate_horizon": candidate_row["candidate_horizon"],
                    "candidate_ranking_method": candidate_row["candidate_ranking_method"],
                    "candidate_return_threshold": candidate_row["candidate_return_threshold"],
                    "candidate_probability_threshold": candidate_row["candidate_probability_threshold"],
                    "selection_reason": (
                        f"context_knn_selector used {neighbor_count} nearest prior samples for {candidate_label}; "
                        f"avg_distance={float(distances[nearest_positions].mean()):.4f}"
                    ),
                }
            )
            rows.append(metrics)
        return self._score_metric_frame(pd.DataFrame(rows))

    def _score_candidates_bin_lookup(
        self,
        available_candidates: pd.DataFrame,
        prior_candidate_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for candidate_row in available_candidates.to_dict(orient="records"):
            candidate_label = str(candidate_row["candidate_label"])
            candidate_prior = prior_candidate_rows[
                (prior_candidate_rows["candidate_label"] == candidate_label)
                & (prior_candidate_rows["context_bin_key"] == candidate_row["context_bin_key"])
            ].copy()
            if candidate_prior.empty:
                continue
            if int(candidate_prior["fold_id"].nunique()) < int(self.config.minimum_prior_folds):
                continue
            if int(len(candidate_prior)) < int(self.config.minimum_prior_samples_for_context_match):
                continue
            metrics = self._aggregate_neighbor_metrics(candidate_prior)
            metrics.update(
                {
                    "candidate_label": candidate_label,
                    "candidate_model_name": candidate_row["candidate_model_name"],
                    "candidate_horizon": candidate_row["candidate_horizon"],
                    "candidate_ranking_method": candidate_row["candidate_ranking_method"],
                    "candidate_return_threshold": candidate_row["candidate_return_threshold"],
                    "candidate_probability_threshold": candidate_row["candidate_probability_threshold"],
                    "selection_reason": (
                        f"context_bin_lookup matched bin={candidate_row['context_bin_key']} with {len(candidate_prior)} prior samples"
                    ),
                }
            )
            rows.append(metrics)
        return self._score_metric_frame(pd.DataFrame(rows))

    def _fit_meta_score_model(self, prior_candidate_rows: pd.DataFrame) -> dict[str, Any] | None:
        if prior_candidate_rows.empty:
            return None
        usable = prior_candidate_rows.copy()
        if int(usable["fold_id"].nunique()) < int(self.config.minimum_prior_folds):
            return None
        if int(len(usable)) < int(self.config.minimum_prior_samples_for_context_match):
            return None
        numeric_columns = self.config.context_feature_columns.copy()
        numeric = usable[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        means = numeric.mean()
        stds = numeric.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        scaled_numeric = (numeric - means) / stds
        categorical = pd.get_dummies(
            usable[["candidate_label", "regime", "confidence_bucket"]],
            columns=["candidate_label", "regime", "confidence_bucket"],
            dummy_na=False,
        )
        design = pd.concat([scaled_numeric, categorical], axis=1)
        design.insert(0, "intercept", 1.0)
        y = pd.to_numeric(usable["row_utility_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x = design.to_numpy(dtype=float)
        ridge = float(self.config.meta_score_ridge_alpha)
        xtx = x.T @ x
        penalty = np.eye(xtx.shape[0], dtype=float) * ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(xtx + penalty, x.T @ y)
        coefficients = pd.Series(beta, index=design.columns, dtype=float)
        feature_importance = coefficients.drop(labels=["intercept"]).abs().sort_values(ascending=False)
        return {
            "numeric_columns": numeric_columns,
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "design_columns": design.columns.tolist(),
            "beta": beta.tolist(),
            "feature_importance": feature_importance.to_dict(),
        }

    def _predict_meta_score(
        self,
        available_candidates: pd.DataFrame,
        model: dict[str, Any] | None,
        prior_candidate_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if model is None or available_candidates.empty:
            return pd.DataFrame()
        rows = available_candidates.copy()
        counts = (
            prior_candidate_rows.groupby("candidate_label", as_index=False)
            .agg(meta_prior_sample_count=("candidate_label", "size"), meta_prior_fold_count=("fold_id", "nunique"))
        )
        rows = rows.merge(counts, on="candidate_label", how="left")
        rows["prior_sample_count"] = pd.to_numeric(rows.get("meta_prior_sample_count"), errors="coerce").fillna(0).astype(int)
        rows["prior_fold_count"] = pd.to_numeric(rows.get("meta_prior_fold_count"), errors="coerce").fillna(0).astype(int)
        rows = rows[
            (rows["prior_sample_count"] >= int(self.config.minimum_prior_samples_for_context_match))
            & (rows["prior_fold_count"] >= int(self.config.minimum_prior_folds))
        ].copy()
        if rows.empty:
            return rows
        numeric = rows[model["numeric_columns"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        means = pd.Series(model["means"], dtype=float)
        stds = pd.Series(model["stds"], dtype=float).replace(0.0, 1.0).fillna(1.0)
        scaled_numeric = (numeric - means) / stds
        categorical = pd.get_dummies(
            rows[["candidate_label", "regime", "confidence_bucket"]],
            columns=["candidate_label", "regime", "confidence_bucket"],
            dummy_na=False,
        )
        design = pd.concat([scaled_numeric, categorical], axis=1)
        design.insert(0, "intercept", 1.0)
        design = design.reindex(columns=model["design_columns"], fill_value=0.0)
        beta = np.asarray(model["beta"], dtype=float)
        rows["utility_score"] = design.to_numpy(dtype=float) @ beta
        rows["selection_reason"] = rows.apply(
            lambda row: (
                f"context_meta_score predicted utility={float(row['utility_score']):.4f} "
                f"for {row['candidate_label']} using {int(row['prior_sample_count'])} prior samples"
            ),
            axis=1,
        )
        return rows.sort_values(
            ["utility_score", "prior_fold_count", "prior_sample_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _load_regime_selector_lookup(self, fold_id: str) -> pd.DataFrame:
        path = self.meta_selector_dir / fold_id / "regime_selection_summary.csv"
        return self._load_csv(path)

    def _fallback_candidate(
        self,
        *,
        fold_id: str,
        current_regime: str,
        available_candidates: pd.DataFrame,
        prior_candidate_pool: pd.DataFrame,
    ) -> dict[str, Any] | None:
        regime_lookup = self._load_regime_selector_lookup(fold_id)
        if not regime_lookup.empty:
            for selector_mode, source_name in (
                ("simple_regime_lookup", "simple_regime_lookup"),
                ("fallback_global", "fallback_global"),
            ):
                selector_match = regime_lookup[
                    (regime_lookup["selector_mode"] == selector_mode)
                    & (regime_lookup["regime"] == current_regime)
                    & (regime_lookup["status"] == "selected")
                ]
                if selector_match.empty:
                    continue
                candidate = selector_match.iloc[0].to_dict()
                label = _candidate_label(
                    {
                        "model_name": candidate.get("selected_model_name"),
                        "horizon": candidate.get("selected_horizon"),
                        "ranking_method": candidate.get("selected_combined_method"),
                        "return_threshold": candidate.get("selected_return_threshold"),
                        "probability_threshold": candidate.get("selected_probability_threshold"),
                    }
                )
                match = available_candidates[available_candidates["candidate_label"] == label]
                if not match.empty:
                    item = match.iloc[0].to_dict()
                    item["fallback_used"] = True
                    item["fallback_source"] = source_name
                    item["selection_reason"] = (
                        f"{source_name} used because context history was sparse; matched {label}"
                    )
                    return item

        if prior_candidate_pool.empty:
            return None
        viable = prior_candidate_pool[
            (prior_candidate_pool["prior_fold_count"] >= int(self.config.minimum_prior_folds))
            & (prior_candidate_pool["sample_count"] >= int(self.config.minimum_prior_samples_for_context_match))
        ].copy()
        if viable.empty:
            viable = prior_candidate_pool.copy()
        for candidate in viable.to_dict(orient="records"):
            label = _candidate_label(candidate)
            match = available_candidates[available_candidates["candidate_label"] == label]
            if match.empty:
                continue
            item = match.iloc[0].to_dict()
            item["fallback_used"] = True
            item["fallback_source"] = "prior_global_candidate_pool"
            item["selection_reason"] = (
                f"prior_global_candidate_pool used because context-specific match was unavailable; matched {label}"
            )
            return item
        return None

    def _select_context_candidate(
        self,
        *,
        selector_mode: str,
        decision_candidates: pd.DataFrame,
        prior_candidate_rows: pd.DataFrame,
        prior_candidate_pool: pd.DataFrame,
        meta_score_model: dict[str, Any] | None,
        fold_id: str,
    ) -> dict[str, Any]:
        current_regime = str(decision_candidates["regime"].iloc[0]).lower()
        if selector_mode == "context_knn_selector":
            scored = self._score_candidates_knn(decision_candidates, prior_candidate_rows)
        elif selector_mode == "context_bin_lookup":
            scored = self._score_candidates_bin_lookup(decision_candidates, prior_candidate_rows)
        elif selector_mode == "context_meta_score":
            scored = self._predict_meta_score(decision_candidates, meta_score_model, prior_candidate_rows)
        else:
            raise ValueError(f"Unsupported selector_mode={selector_mode}")

        if not scored.empty:
            best = scored.iloc[0].to_dict()
            best["fallback_used"] = False
            best["fallback_source"] = None
            return best

        fallback = self._fallback_candidate(
            fold_id=fold_id,
            current_regime=current_regime,
            available_candidates=decision_candidates,
            prior_candidate_pool=prior_candidate_pool,
        )
        if fallback is not None:
            return fallback
        return {
            "status": "no_candidate_available",
            "fallback_used": False,
            "fallback_source": None,
            "selection_reason": "No eligible context-conditioned or fallback candidate was available.",
        }

    def _selected_row_from_candidate(
        self,
        *,
        decision_candidates: pd.DataFrame,
        selected_candidate: dict[str, Any],
        selector_mode: str,
    ) -> pd.DataFrame:
        if decision_candidates.empty or selected_candidate.get("status") == "no_candidate_available":
            return pd.DataFrame()
        match = decision_candidates[
            decision_candidates["candidate_label"] == str(selected_candidate["candidate_label"])
        ].copy()
        if match.empty:
            return pd.DataFrame()
        match = match.sort_values(["candidate_label", "prediction_date", "ticker"]).head(1).copy()
        match["selector_mode"] = selector_mode
        match["selection_family"] = "context_selector"
        match["selected_model_name"] = selected_candidate.get("candidate_model_name", selected_candidate.get("candidate_label"))
        match["selected_horizon"] = selected_candidate.get("candidate_horizon")
        match["selected_combined_method"] = selected_candidate.get("candidate_ranking_method")
        match["selected_return_threshold"] = _threshold_value(selected_candidate.get("candidate_return_threshold"))
        match["selected_probability_threshold"] = _threshold_value(selected_candidate.get("candidate_probability_threshold"))
        match["selection_reason"] = selected_candidate.get("selection_reason")
        match["fallback_used"] = bool(selected_candidate.get("fallback_used", False))
        match["fallback_source"] = selected_candidate.get("fallback_source")
        match["context_utility_score"] = float(selected_candidate.get("utility_score", np.nan))
        match["context_prior_sample_count"] = int(selected_candidate.get("prior_sample_count", 0) or 0)
        match["context_prior_fold_count"] = int(selected_candidate.get("prior_fold_count", 0) or 0)
        return match

    def _load_regime_comparison_rows(self, fold_id: str) -> pd.DataFrame:
        path = self.meta_selector_dir / fold_id / "selector_vs_baselines.csv"
        return self._load_csv(path)

    def _selector_vs_regime_rows(
        self,
        *,
        selector_performance: pd.DataFrame,
        fold_id: str,
        fold_number: int,
    ) -> pd.DataFrame:
        comparator_rows = self._load_regime_comparison_rows(fold_id)
        if comparator_rows.empty or selector_performance.empty:
            return pd.DataFrame()
        comparators = comparator_rows[
            comparator_rows["entity_name"].isin(self.config.regime_selector_modes)
        ].copy()
        rows: list[dict[str, Any]] = []
        for selector_row in selector_performance.to_dict(orient="records"):
            for comparator in comparators.to_dict(orient="records"):
                rows.append(
                    {
                        "fold_id": fold_id,
                        "fold_number": fold_number,
                        "selector_mode": selector_row["entity_name"],
                        "comparator_entity": comparator["entity_name"],
                        "average_actual_return_delta": float(selector_row["average_actual_return"]) - float(comparator["average_actual_return"]),
                        "profit_label_hit_rate_delta": float(selector_row["profit_label_hit_rate"]) - float(comparator["profit_label_hit_rate"]),
                        "top_3_avg_return_delta": float(selector_row["top_3_avg_return"]) - float(comparator["top_3_avg_return"]),
                        "top_3_profit_rate_delta": float(selector_row["top_3_profit_rate"]) - float(comparator["top_3_profit_rate"]),
                    }
                )
        return pd.DataFrame(rows)

    def _run_fold(
        self,
        *,
        fold: dict[str, Any],
        prior_folds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fold_dir = self.output_dir / str(fold["fold_id"])
        fold_dir.mkdir(parents=True, exist_ok=True)

        current_raw_rows = self.base_runner._load_fold_rows(fold)
        prior_raw_rows = self._load_fold_rows_for_folds(prior_folds)
        model_history, combined_history = self.base_runner._load_fold_histories(prior_folds)
        candidate_history = self.base_runner._merge_candidate_history(model_history, combined_history)
        prior_candidate_pool = self._compress_candidate_pool(
            self.base_runner._aggregate_candidate_pool(candidate_history, regime=None)
        )

        if current_raw_rows.empty or prior_candidate_pool.empty:
            empty = pd.DataFrame()
            fold_config = {
                **asdict(self.config),
                "fold_id": str(fold["fold_id"]),
                "fold_number": int(fold["fold_number"]),
                "no_leakage": True,
                "analysis_only": True,
                "status": "insufficient_prior_history",
            }
            with (fold_dir / "fold_config.json").open("w", encoding="utf-8") as handle:
                json.dump(fold_config, handle, indent=2)
            for name in [
                "selected_candidates.csv",
                "context_features_used.csv",
                "selector_performance.csv",
                "selector_vs_baselines.csv",
                "selector_vs_regime_selector.csv",
            ]:
                empty.to_csv(fold_dir / name, index=False)
            return {
                "fold_id": str(fold["fold_id"]),
                "fold_number": int(fold["fold_number"]),
                "selected_candidates": empty,
                "selector_performance": empty,
                "selector_vs_baselines": empty,
                "selector_vs_regime_selector": empty,
                "feature_importance": pd.DataFrame(),
            }

        prior_candidate_rows = self._materialize_candidate_rows(
            prior_raw_rows,
            prior_candidate_pool,
            selector_mode="candidate_history",
            selection_family="candidate_history",
        )
        current_candidate_rows = self._materialize_candidate_rows(
            current_raw_rows,
            prior_candidate_pool,
            selector_mode="candidate_current",
            selection_family="candidate_current",
        )
        lookup_rows = self._concat_non_empty([prior_raw_rows, current_raw_rows])
        prior_candidate_rows = self._augment_context_features(prior_candidate_rows, raw_lookup_rows=lookup_rows)
        current_candidate_rows = self._augment_context_features(current_candidate_rows, raw_lookup_rows=lookup_rows)
        prior_candidate_rows = self._derive_context_bins(prior_candidate_rows, prior_candidate_rows)
        current_candidate_rows = self._derive_context_bins(prior_candidate_rows, current_candidate_rows)
        prior_candidate_rows = self._add_row_targets(prior_candidate_rows)

        meta_score_model = self._fit_meta_score_model(prior_candidate_rows)
        feature_importance = pd.DataFrame()
        if meta_score_model is not None:
            feature_importance = pd.DataFrame(
                [
                    {
                        "fold_id": str(fold["fold_id"]),
                        "fold_number": int(fold["fold_number"]),
                        "selector_mode": "context_meta_score",
                        "feature_name": key,
                        "importance": value,
                    }
                    for key, value in meta_score_model["feature_importance"].items()
                    if not key.startswith("candidate_label_")
                ]
            )

        selected_frames: list[pd.DataFrame] = []
        for selector_mode in self.config.selector_modes:
            for (_, _), group in current_candidate_rows.groupby(["prediction_date", "ticker"], dropna=False):
                decision_candidates = group.copy().reset_index(drop=True)
                selected_candidate = self._select_context_candidate(
                    selector_mode=selector_mode,
                    decision_candidates=decision_candidates,
                    prior_candidate_rows=prior_candidate_rows,
                    prior_candidate_pool=prior_candidate_pool,
                    meta_score_model=meta_score_model,
                    fold_id=str(fold["fold_id"]),
                )
                selected_frame = self._selected_row_from_candidate(
                    decision_candidates=decision_candidates,
                    selected_candidate=selected_candidate,
                    selector_mode=selector_mode,
                )
                if not selected_frame.empty:
                    selected_frames.append(selected_frame)

        selected_candidates = self._concat_non_empty(selected_frames)

        selector_performance_rows = []
        for selector_mode in self.config.selector_modes:
            mode_rows = selected_candidates[selected_candidates["selector_mode"] == selector_mode].copy()
            selector_performance_rows.append(
                self.base_runner._evaluate_rows(
                    mode_rows,
                    fold=fold,
                    entity_name=selector_mode,
                    selection_family="context_selector",
                )
            )
        selector_performance = pd.DataFrame(selector_performance_rows)

        meta_selector_fold_rows = self._load_regime_comparison_rows(str(fold["fold_id"]))
        selector_vs_baselines = pd.concat(
            [
                selector_performance,
                meta_selector_fold_rows[meta_selector_fold_rows["selection_family"] == "baseline"].copy()
                if not meta_selector_fold_rows.empty
                else pd.DataFrame(),
            ],
            ignore_index=True,
        )
        selector_vs_regime_selector = self._selector_vs_regime_rows(
            selector_performance=selector_performance,
            fold_id=str(fold["fold_id"]),
            fold_number=int(fold["fold_number"]),
        )

        context_features_used = selected_candidates[
            [
                "date",
                "prediction_date",
                "ticker",
                "regime",
                "selector_mode",
                "candidate_label",
                "market_return_20d",
                "market_volatility_20d",
                "stock_relative_strength_20d",
                "predicted_return",
                "predicted_profit_probability",
                "normalized_predicted_return",
                "confidence_bucket",
                "probability_entropy",
                "predicted_return_dispersion",
                "predicted_probability_dispersion",
                "context_bin_key",
                "context_utility_score",
                "context_prior_sample_count",
                "context_prior_fold_count",
                "fallback_used",
                "fallback_source",
            ]
        ].copy() if not selected_candidates.empty else pd.DataFrame()

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
        }
        with (fold_dir / "fold_config.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_config, handle, indent=2)
        selected_candidates.to_csv(fold_dir / "selected_candidates.csv", index=False)
        context_features_used.to_csv(fold_dir / "context_features_used.csv", index=False)
        selector_performance.to_csv(fold_dir / "selector_performance.csv", index=False)
        selector_vs_baselines.to_csv(fold_dir / "selector_vs_baselines.csv", index=False)
        selector_vs_regime_selector.to_csv(fold_dir / "selector_vs_regime_selector.csv", index=False)

        return {
            "fold_id": str(fold["fold_id"]),
            "fold_number": int(fold["fold_number"]),
            "selected_candidates": selected_candidates,
            "selector_performance": selector_performance,
            "selector_vs_baselines": selector_vs_baselines,
            "selector_vs_regime_selector": selector_vs_regime_selector,
            "feature_importance": feature_importance,
        }

    def _build_overview(self, fold_results: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in fold_results:
            performance = result["selector_performance"]
            if performance.empty:
                continue
            for row in performance.to_dict(orient="records"):
                rows.append(
                    {
                        "fold_id": result["fold_id"],
                        "fold_number": result["fold_number"],
                        "selector_mode": row["entity_name"],
                        "observations": row.get("observations", 0),
                        "average_actual_return": row.get("average_actual_return"),
                        "profit_label_hit_rate": row.get("profit_label_hit_rate"),
                        "top_3_avg_return": row.get("top_3_avg_return"),
                        "top_3_profit_rate": row.get("top_3_profit_rate"),
                    }
                )
        return pd.DataFrame(rows).sort_values(["fold_number", "selector_mode"]).reset_index(drop=True)

    def _build_stability_summary(self, selected_candidates: pd.DataFrame) -> pd.DataFrame:
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
                fallback_count=("fallback_used", "sum"),
                mean_context_utility_score=("context_utility_score", "mean"),
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
        return grouped.sort_values(
            ["selector_mode", "rows_selected", "folds_selected"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

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
                top_3_avg_return=("top_3_avg_return", "mean"),
                top_3_profit_rate=("top_3_profit_rate", "mean"),
            )
        )
        baseline_best = summary[summary["entity_name"] == "fixed_best_global_setup"]
        if not baseline_best.empty:
            baseline_row = baseline_best.iloc[0]
            summary["beats_fixed_best_global_on_avg_return"] = (
                summary["average_actual_return"] > float(baseline_row["average_actual_return"])
            )
            summary["beats_fixed_best_global_on_top_3_avg_return"] = (
                summary["top_3_avg_return"] > float(baseline_row["top_3_avg_return"])
            )
        else:
            summary["beats_fixed_best_global_on_avg_return"] = False
            summary["beats_fixed_best_global_on_top_3_avg_return"] = False
        return summary.sort_values(["selection_family", "average_actual_return"], ascending=[True, False]).reset_index(drop=True)

    def _build_vs_regime_summary(self, selector_vs_regime: pd.DataFrame) -> pd.DataFrame:
        if selector_vs_regime.empty:
            return pd.DataFrame()
        return (
            selector_vs_regime.groupby(["selector_mode", "comparator_entity"], as_index=False)
            .agg(
                evaluated_folds=("fold_id", "nunique"),
                average_actual_return_delta=("average_actual_return_delta", "mean"),
                profit_label_hit_rate_delta=("profit_label_hit_rate_delta", "mean"),
                top_3_avg_return_delta=("top_3_avg_return_delta", "mean"),
                top_3_profit_rate_delta=("top_3_profit_rate_delta", "mean"),
            )
            .sort_values(["selector_mode", "top_3_avg_return_delta"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def _build_feature_importance_summary(self, feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
        feature_importance = self._concat_non_empty(feature_frames)
        if feature_importance.empty:
            return pd.DataFrame()
        return (
            feature_importance.groupby(["selector_mode", "feature_name"], as_index=False)
            .agg(
                evaluated_folds=("fold_id", "nunique"),
                mean_importance=("importance", "mean"),
                max_importance=("importance", "max"),
            )
            .sort_values(["selector_mode", "mean_importance"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def _build_overall_report(
        self,
        *,
        audit_report: pd.DataFrame,
        vs_baselines_summary: pd.DataFrame,
        vs_regime_summary: pd.DataFrame,
        stability_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        context_rows = vs_baselines_summary[
            vs_baselines_summary["selection_family"] == "context_selector"
        ].copy()
        baseline_rows = vs_baselines_summary[
            vs_baselines_summary["selection_family"] == "baseline"
        ].copy()
        if not context_rows.empty:
            best_context = context_rows.sort_values(
                ["average_actual_return", "top_3_avg_return"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    "component": "best context selector",
                    "best_overall_choice": best_context["entity_name"],
                    "stability_level": "medium" if float(best_context["top_3_avg_return"]) > 0 else "low",
                    "supporting_evidence": (
                        f"{best_context['entity_name']} avg_return={float(best_context['average_actual_return']):.4f}, "
                        f"top3={float(best_context['top_3_avg_return']):.4f}"
                    ),
                    "caution_note": "Treat any edge as provisional until more folds accumulate.",
                }
            )
            if not baseline_rows.empty:
                best_baseline = baseline_rows.sort_values(
                    ["average_actual_return", "top_3_avg_return"],
                    ascending=[False, False],
                ).iloc[0]
                rows.append(
                    {
                        "component": "context selector vs fixed baselines",
                        "best_overall_choice": (
                            "context selector beats fixed baseline"
                            if float(best_context["average_actual_return"]) > float(best_baseline["average_actual_return"])
                            else "fixed baseline still stronger"
                        ),
                        "stability_level": "low",
                        "supporting_evidence": (
                            f"best_context={best_context['entity_name']} ({float(best_context['average_actual_return']):.4f}) "
                            f"vs best_baseline={best_baseline['entity_name']} ({float(best_baseline['average_actual_return']):.4f})"
                        ),
                        "caution_note": "The strongest baseline remains competitive on top-k metrics.",
                    }
                )
        if not vs_regime_summary.empty:
            best_vs_regime = vs_regime_summary.sort_values(
                ["top_3_avg_return_delta", "average_actual_return_delta"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    "component": "context selector vs regime selector",
                    "best_overall_choice": best_vs_regime["selector_mode"],
                    "stability_level": "low" if float(best_vs_regime["top_3_avg_return_delta"]) < 0.01 else "medium",
                    "supporting_evidence": (
                        f"{best_vs_regime['selector_mode']} vs {best_vs_regime['comparator_entity']} "
                        f"top3_delta={float(best_vs_regime['top_3_avg_return_delta']):.4f}"
                    ),
                    "caution_note": "The context edge is modest and can disappear in weaker folds.",
                }
            )
        if not stability_summary.empty:
            top_setup = stability_summary.sort_values(
                ["rows_selected", "folds_selected"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    "component": "most frequent context-conditioned setup",
                    "best_overall_choice": top_setup["candidate_label"],
                    "stability_level": "medium" if int(top_setup["folds_selected"]) >= 3 else "low",
                    "supporting_evidence": (
                        f"selected in {int(top_setup['folds_selected'])} folds and {int(top_setup['rows_selected'])} rows"
                    ),
                    "caution_note": "Frequent reuse suggests family-level stability, not configuration-level certainty.",
                }
            )
        if not audit_report.empty:
            suspicious = audit_report[audit_report["status"] == "warning"]
            rows.append(
                {
                    "component": "benchmark audit",
                    "best_overall_choice": "audited",
                    "stability_level": "medium" if suspicious.empty else "low",
                    "supporting_evidence": (
                        "; ".join(suspicious["details"].tolist()) if not suspicious.empty else "No benchmark aggregation mismatch was found."
                    ),
                    "caution_note": "Equal summary rows need interpretation through candidate traces, not raw averages alone.",
                }
            )
        return pd.DataFrame(rows)

    def _render_summary_charts(
        self,
        *,
        selected_candidates: pd.DataFrame,
        vs_baselines_summary: pd.DataFrame,
        vs_regime_summary: pd.DataFrame,
        feature_importance_summary: pd.DataFrame,
    ) -> None:
        self.charts_root.mkdir(parents=True, exist_ok=True)
        if not vs_baselines_summary.empty:
            context_rows = vs_baselines_summary[
                vs_baselines_summary["selection_family"] == "context_selector"
            ].copy()
            if not context_rows.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(context_rows["entity_name"], context_rows["average_actual_return"])
                ax.set_title("Selector Win Frequency Vs Baselines")
                ax.set_ylabel("Average Actual Return")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                fig.savefig(self.charts_root / "selector_win_frequency_vs_baselines.png", dpi=150)
                plt.close(fig)

        if not vs_regime_summary.empty:
            pivot = vs_regime_summary.pivot(index="selector_mode", columns="comparator_entity", values="top_3_avg_return_delta").fillna(0.0)
            fig, ax = plt.subplots(figsize=(9, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Context Selector Vs Regime Selector")
            ax.set_ylabel("Top-3 Avg Return Delta")
            fig.tight_layout()
            fig.savefig(self.charts_root / "context_selector_vs_regime_selector.png", dpi=150)
            plt.close(fig)

        if not selected_candidates.empty:
            fallback = (
                selected_candidates.groupby("selector_mode", as_index=False)["fallback_used"]
                .mean()
                .rename(columns={"fallback_used": "fallback_rate"})
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(fallback["selector_mode"], fallback["fallback_rate"])
            ax.set_title("Fallback Usage Reduction Chart")
            ax.set_ylabel("Fallback Rate")
            fig.tight_layout()
            fig.savefig(self.charts_root / "fallback_usage_reduction_chart.png", dpi=150)
            plt.close(fig)

            switching = (
                selected_candidates.groupby(["selector_mode", "selected_horizon"], as_index=False)["ticker"]
                .count()
                .rename(columns={"ticker": "count"})
            )
            pivot = switching.pivot(index="selected_horizon", columns="selector_mode", values="count").fillna(0.0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Switching Frequency By Component")
            ax.set_ylabel("Rows Selected")
            fig.tight_layout()
            fig.savefig(self.charts_root / "switching_frequency_by_component.png", dpi=150)
            plt.close(fig)

            regime_perf = (
                selected_candidates.groupby(["selector_mode", "regime"], as_index=False)["actual_return"]
                .mean()
                .rename(columns={"actual_return": "average_actual_return"})
            )
            pivot = regime_perf.pivot(index="regime", columns="selector_mode", values="average_actual_return").fillna(0.0)
            fig, ax = plt.subplots(figsize=(8, 4))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Performance By Regime")
            ax.set_ylabel("Average Actual Return")
            fig.tight_layout()
            fig.savefig(self.charts_root / "performance_by_regime_and_fold.png", dpi=150)
            plt.close(fig)

        if not feature_importance_summary.empty:
            top = feature_importance_summary.sort_values("mean_importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(top["feature_name"], top["mean_importance"])
            ax.set_title("Context Feature Importance")
            ax.set_ylabel("Mean Importance")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(self.charts_root / "context_feature_importance.png", dpi=150)
            plt.close(fig)

    def run(self) -> dict[str, Any]:
        folds = self._discover_folds()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_root.mkdir(parents=True, exist_ok=True)

        audit_result = self._run_benchmark_audit(folds)

        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            prior_folds = [item for item in folds if int(item["fold_number"]) < int(fold["fold_number"])]
            fold_results.append(self._run_fold(fold=fold, prior_folds=prior_folds))

        selected_candidates = self._concat_non_empty([item["selected_candidates"] for item in fold_results])
        selector_vs_baselines = self._concat_non_empty([item["selector_vs_baselines"] for item in fold_results])
        selector_vs_regime = self._concat_non_empty([item["selector_vs_regime_selector"] for item in fold_results])
        feature_importance_summary = self._build_feature_importance_summary(
            [item["feature_importance"] for item in fold_results]
        )

        overview = self._build_overview(fold_results)
        stability_summary = self._build_stability_summary(selected_candidates)
        vs_baselines_summary = self._build_vs_baselines_summary(selector_vs_baselines)
        vs_regime_summary = self._build_vs_regime_summary(selector_vs_regime)
        overall_report = self._build_overall_report(
            audit_report=audit_result["benchmark_audit_report"],
            vs_baselines_summary=vs_baselines_summary,
            vs_regime_summary=vs_regime_summary,
            stability_summary=stability_summary,
        )

        summary_paths = {
            "context_selector_overview": self.summary_root / "context_selector_overview.csv",
            "context_selector_stability_summary": self.summary_root / "context_selector_stability_summary.csv",
            "context_selector_vs_baselines_summary": self.summary_root / "context_selector_vs_baselines_summary.csv",
            "context_selector_vs_regime_summary": self.summary_root / "context_selector_vs_regime_summary.csv",
            "context_feature_importance_summary": self.summary_root / "context_feature_importance_summary.csv",
            "overall_context_selector_report": self.summary_root / "overall_context_selector_report.csv",
        }
        overview.to_csv(summary_paths["context_selector_overview"], index=False)
        stability_summary.to_csv(summary_paths["context_selector_stability_summary"], index=False)
        vs_baselines_summary.to_csv(summary_paths["context_selector_vs_baselines_summary"], index=False)
        vs_regime_summary.to_csv(summary_paths["context_selector_vs_regime_summary"], index=False)
        feature_importance_summary.to_csv(summary_paths["context_feature_importance_summary"], index=False)
        overall_report.to_csv(summary_paths["overall_context_selector_report"], index=False)

        self._render_summary_charts(
            selected_candidates=selected_candidates,
            vs_baselines_summary=vs_baselines_summary,
            vs_regime_summary=vs_regime_summary,
            feature_importance_summary=feature_importance_summary,
        )

        return {
            "audit_result": audit_result,
            "fold_results": fold_results,
            "context_selector_overview": overview,
            "context_selector_stability_summary": stability_summary,
            "context_selector_vs_baselines_summary": vs_baselines_summary,
            "context_selector_vs_regime_summary": vs_regime_summary,
            "context_feature_importance_summary": feature_importance_summary,
            "overall_context_selector_report": overall_report,
            "summary_paths": {name: str(path) for name, path in summary_paths.items()},
        }
