"""Consensus and disagreement summaries derived from governed forecast outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def packet_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return the canonical packet grouping columns present in a frame."""

    ordered = [
        "timestamp",
        "ticker",
        "horizon",
        "target_type",
        "window_id",
        "target_timestamp",
        "core_run_id",
        "preset",
        "group_name",
        "target_name",
        "target_column",
        "target_family",
        "target_tradable",
        "ticker_count",
        "ticker_group_members",
        "run_mode",
    ]
    return [column for column in ordered if column in frame.columns]


def scenario_group_columns(frame: pd.DataFrame) -> list[str]:
    ordered = [
        "core_run_id",
        "preset",
        "group_name",
        "horizon",
        "target_name",
        "target_type",
        "target_column",
        "target_family",
        "target_tradable",
        "ticker_count",
        "ticker_group_members",
        "run_mode",
    ]
    return [column for column in ordered if column in frame.columns]


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def add_model_ranks(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Attach within-packet rank columns to forecast rows."""

    if forecasts_df.empty:
        return forecasts_df.copy()
    ranked = forecasts_df.copy()
    group_columns = packet_group_columns(ranked)
    ranked["model_rank"] = ranked.groupby(group_columns)["y_pred"].rank(method="dense", ascending=False)
    ranked["abs_signal_rank"] = ranked.groupby(group_columns)["y_pred"].transform(
        lambda values: values.abs().rank(method="dense", ascending=False)
    )
    return ranked.sort_values([*group_columns, "model_rank", "model_name"]).reset_index(drop=True)


def _agreement_bucket(agreement_score: float) -> str:
    if pd.isna(agreement_score):
        return "unknown"
    if agreement_score >= 0.75:
        return "high"
    if agreement_score >= 0.55:
        return "medium"
    return "low"


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _signal_disagreement_share(group: pd.DataFrame) -> float:
    if "signal" not in group.columns:
        return float("nan")
    forecast_sign = np.sign(pd.to_numeric(group["y_pred"], errors="coerce").fillna(0.0))
    signal_sign = np.sign(pd.to_numeric(group["signal"], errors="coerce").fillna(0.0))
    return float((forecast_sign != signal_sign).mean()) if len(group) else float("nan")


def build_model_consensus_summary(
    forecasts_df: pd.DataFrame,
    *,
    signals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build packet-level model agreement and disagreement summaries."""

    if forecasts_df.empty:
        return pd.DataFrame(columns=[*packet_group_columns(forecasts_df), "model_count"])

    ranked = add_model_ranks(forecasts_df)
    if signals_df is not None and not signals_df.empty:
        join_columns = _shared_columns(
            ranked,
            signals_df,
            [
                "timestamp",
                "ticker",
                "model_name",
                "target_type",
                "horizon",
                "window_id",
                "core_run_id",
                "run_mode",
            ],
        )
        ranked = ranked.merge(
            signals_df[join_columns + [column for column in ["signal", "position_size"] if column in signals_df.columns]],
            on=join_columns,
            how="left",
        )

    rows: list[dict[str, Any]] = []
    group_columns = packet_group_columns(ranked)
    for keys, group in ranked.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        predictions = pd.to_numeric(group["y_pred"], errors="coerce").astype(float)
        signs = np.sign(predictions)
        positive_count = int((signs > 0).sum())
        negative_count = int((signs < 0).sum())
        neutral_count = int((signs == 0).sum())
        model_count = int(len(group))
        agreement_score = float(max(positive_count, negative_count, neutral_count) / model_count) if model_count else float("nan")
        role_counts = group.get("model_role", pd.Series(dtype="object")).astype(str).value_counts()
        prediction_range = float(predictions.max() - predictions.min()) if not predictions.empty else float("nan")
        dispersion_score = float(predictions.std(ddof=0)) if len(predictions) else float("nan")
        primary_rows = group[group.get("model_role", pd.Series(dtype="object")).astype(str) == "primary_research"]
        comparator_rows = group[group.get("model_role", pd.Series(dtype="object")).astype(str) == "comparator"]
        baseline_rows = group[group.get("model_role", pd.Series(dtype="object")).astype(str) == "baseline_only"]
        ensemble_rows = group[group["model_name"].astype(str) == "weighted_ensemble"]
        row.update(
            {
                "model_count": model_count,
                "primary_model_count": int(role_counts.get("primary_research", 0)),
                "comparator_model_count": int(role_counts.get("comparator", 0)),
                "baseline_model_count": int(role_counts.get("baseline_only", 0)),
                "shadow_model_count": int(role_counts.get("shadow_only", 0)),
                "ensemble_model_count": int(len(ensemble_rows)),
                "agreement_score": agreement_score,
                "disagreement_score": float(1.0 - agreement_score) if pd.notna(agreement_score) else float("nan"),
                "agreement_bucket": _agreement_bucket(agreement_score),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "sign_conflict": bool(positive_count > 0 and negative_count > 0),
                "rank_spread": prediction_range,
                "dispersion_score": dispersion_score,
                "prediction_range": prediction_range,
                "primary_mean_prediction": _safe_mean(primary_rows.get("y_pred", pd.Series(dtype=float))),
                "comparator_mean_prediction": _safe_mean(comparator_rows.get("y_pred", pd.Series(dtype=float))),
                "baseline_mean_prediction": _safe_mean(baseline_rows.get("y_pred", pd.Series(dtype=float))),
                "ensemble_prediction": _safe_mean(ensemble_rows.get("y_pred", pd.Series(dtype=float))),
                "primary_vs_comparator_gap": abs(
                    _safe_mean(primary_rows.get("y_pred", pd.Series(dtype=float)))
                    - _safe_mean(comparator_rows.get("y_pred", pd.Series(dtype=float)))
                )
                if not primary_rows.empty and not comparator_rows.empty
                else float("nan"),
                "primary_vs_baseline_gap": abs(
                    _safe_mean(primary_rows.get("y_pred", pd.Series(dtype=float)))
                    - _safe_mean(baseline_rows.get("y_pred", pd.Series(dtype=float)))
                )
                if not primary_rows.empty and not baseline_rows.empty
                else float("nan"),
                "ensemble_vs_primary_gap": abs(
                    _safe_mean(ensemble_rows.get("y_pred", pd.Series(dtype=float)))
                    - _safe_mean(primary_rows.get("y_pred", pd.Series(dtype=float)))
                )
                if not ensemble_rows.empty and not primary_rows.empty
                else float("nan"),
                "policy_gate_disagreement_share": _signal_disagreement_share(group),
                "active_signal_share": float(
                    pd.to_numeric(group.get("signal", pd.Series(dtype=float)), errors="coerce").fillna(0.0).ne(0.0).mean()
                )
                if "signal" in group.columns
                else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
