"""Analysis-ready forecast packet builders for quant-core outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.consensus import add_model_ranks, packet_group_columns, scenario_group_columns


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


def _safe_timestamp(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _signal_strength_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    magnitude = abs(float(value))
    if magnitude >= 0.03:
        return "high"
    if magnitude >= 0.01:
        return "medium"
    return "low"


def _volatility_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.05:
        return "high"
    if value >= 0.02:
        return "medium"
    return "low"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _lookup_by_keys(frame: pd.DataFrame, group_columns: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if frame.empty:
        return {}
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = tuple(row.get(column) for column in group_columns)
        indexed[key] = row
    return indexed


def _lookup_groups(frame: pd.DataFrame, group_columns: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if frame.empty:
        return {}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        groups[keys] = group.to_dict(orient="records")
    return groups


def build_analysis_packets(
    forecasts_df: pd.DataFrame,
    consensus_summary: pd.DataFrame,
    *,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    signals_df: pd.DataFrame | None = None,
    positions_df: pd.DataFrame | None = None,
    strategy_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one explicit analysis packet per ticker/date/horizon/target slice."""

    if forecasts_df.empty:
        return pd.DataFrame(columns=["packet_id", "timestamp", "ticker", "horizon", "target_type", "run_mode"])

    ranked = add_model_ranks(forecasts_df)
    group_columns = packet_group_columns(ranked)
    risk_frame = risk_df if risk_df is not None else pd.DataFrame()
    regime_frame = regime_df if regime_df is not None else pd.DataFrame()
    signals_frame = signals_df if signals_df is not None else pd.DataFrame()
    positions_frame = positions_df if positions_df is not None else pd.DataFrame()
    strategy_metrics_frame = strategy_metrics_df if strategy_metrics_df is not None else pd.DataFrame()

    risk_lookup = _lookup_by_keys(risk_frame, _shared_columns(ranked, risk_frame, group_columns))
    regime_lookup = _lookup_by_keys(regime_frame, _shared_columns(ranked, regime_frame, group_columns))
    signal_groups = _lookup_groups(
        signals_frame,
        _shared_columns(
            ranked,
            signals_frame,
            [*group_columns, "model_name"],
        ),
    )
    position_groups = _lookup_groups(
        positions_frame,
        _shared_columns(
            ranked,
            positions_frame,
            [*group_columns, "model_name"],
        ),
    )
    scenario_metric_groups = _lookup_groups(
        strategy_metrics_frame,
        _shared_columns(
            ranked,
            strategy_metrics_frame,
            scenario_group_columns(ranked),
        ),
    )
    consensus_lookup = _lookup_by_keys(consensus_summary, group_columns)

    packet_rows: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for keys, group in ranked.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        packet = dict(zip(group_columns, keys))
        key_tuple = tuple(packet.get(column) for column in group_columns)
        scenario_key_columns = scenario_group_columns(group)
        scenario_key = tuple(packet.get(column) for column in scenario_key_columns)

        ordered = group.sort_values(["research_priority", "model_rank", "model_name"]).reset_index(drop=True)
        primary_rows = ordered[ordered.get("model_role", pd.Series(dtype="object")).astype(str) == "primary_research"]
        primary_row = primary_rows.iloc[0] if not primary_rows.empty else ordered.iloc[0]
        ensemble_rows = ordered[ordered["model_name"].astype(str) == "weighted_ensemble"]
        ensemble_row = ensemble_rows.iloc[0] if not ensemble_rows.empty else None
        consensus_row = consensus_lookup.get(key_tuple, {})
        risk_row = risk_lookup.get(tuple(packet.get(column) for column in _shared_columns(group, risk_frame, group_columns)), {})
        regime_row = regime_lookup.get(tuple(packet.get(column) for column in _shared_columns(group, regime_frame, group_columns)), {})

        model_records: list[dict[str, Any]] = []
        active_signal_count = 0
        long_signal_count = 0
        short_signal_count = 0
        position_sizes: list[float] = []
        for model_row in ordered.to_dict(orient="records"):
            signal_key = tuple(model_row.get(column) for column in _shared_columns(group, signals_frame, [*group_columns, "model_name"]))
            position_key = tuple(model_row.get(column) for column in _shared_columns(group, positions_frame, [*group_columns, "model_name"]))
            signal_record = (signal_groups.get(signal_key) or [None])[0]
            position_record = (position_groups.get(position_key) or [None])[0]
            signal_value = _safe_float(signal_record.get("signal")) if isinstance(signal_record, dict) else None
            position_value = _safe_float(position_record.get("position_size")) if isinstance(position_record, dict) else None
            if signal_value is not None and signal_value != 0.0:
                active_signal_count += 1
                if signal_value > 0:
                    long_signal_count += 1
                if signal_value < 0:
                    short_signal_count += 1
            if position_value is not None:
                position_sizes.append(position_value)
            model_records.append(
                {
                    "model_name": model_row.get("model_name"),
                    "model_family": model_row.get("model_family"),
                    "model_role": model_row.get("model_role"),
                    "model_status": model_row.get("model_status"),
                    "research_priority": int(model_row.get("research_priority", 0)),
                    "prediction": _safe_float(model_row.get("y_pred")),
                    "realized": _safe_float(model_row.get("y_true")),
                    "model_rank": _safe_float(model_row.get("model_rank")),
                    "abs_signal_rank": _safe_float(model_row.get("abs_signal_rank")),
                    "signal": signal_value,
                    "position_size": position_value,
                }
            )

        strategy_metric_rows = scenario_metric_groups.get(scenario_key, [])
        top_policy_row = None
        if strategy_metric_rows:
            top_policy_row = sorted(
                strategy_metric_rows,
                key=lambda row: (
                    -(_safe_float(row.get("sharpe")) or float("-inf")),
                    -(_safe_float(row.get("cagr")) or float("-inf")),
                ),
            )[0]

        primary_prediction = _safe_float(primary_row.get("y_pred"))
        regime_label = str(regime_row.get("regime_label")) if regime_row else None
        vol_forecast = _safe_float(risk_row.get("vol_forecast")) if risk_row else None
        agreement_score = _safe_float(consensus_row.get("agreement_score"))
        agreement_bucket = str(consensus_row.get("agreement_bucket", "unknown"))
        packet_id = "|".join(
            [
                str(packet.get("ticker")),
                str(pd.Timestamp(packet.get("timestamp")).date()),
                f"h{int(packet.get('horizon', 0)):02d}",
                str(packet.get("target_type")),
                str(packet.get("run_mode")),
                str(packet.get("core_run_id")),
            ]
        )

        packet_row = {
            **packet,
            "packet_id": packet_id,
            "packet_generated_at": generated_at,
            "primary_model_name": primary_row.get("model_name"),
            "primary_model_role": primary_row.get("model_role"),
            "primary_prediction": primary_prediction,
            "primary_prediction_summary": _json_dumps(
                {
                    "model_name": primary_row.get("model_name"),
                    "model_role": primary_row.get("model_role"),
                    "prediction": primary_prediction,
                    "research_priority": int(primary_row.get("research_priority", 0)),
                }
            ),
            "model_by_model_predictions": _json_dumps(model_records),
            "model_ranks": _json_dumps(
                {
                    record["model_name"]: {
                        "rank": record["model_rank"],
                        "abs_signal_rank": record["abs_signal_rank"],
                    }
                    for record in model_records
                }
            ),
            "model_agreement_score": agreement_score,
            "model_disagreement_score": _safe_float(consensus_row.get("disagreement_score")),
            "dispersion_score": _safe_float(consensus_row.get("dispersion_score")),
            "sign_conflict": bool(consensus_row.get("sign_conflict", False)),
            "rank_spread": _safe_float(consensus_row.get("rank_spread")),
            "primary_vs_comparator_gap": _safe_float(consensus_row.get("primary_vs_comparator_gap")),
            "primary_vs_baseline_gap": _safe_float(consensus_row.get("primary_vs_baseline_gap")),
            "ensemble_vs_primary_gap": _safe_float(consensus_row.get("ensemble_vs_primary_gap")),
            "policy_gate_disagreement_share": _safe_float(consensus_row.get("policy_gate_disagreement_share")),
            "agreement_bucket": agreement_bucket,
            "ensemble_summary": _json_dumps(
                {
                    "model_name": ensemble_row.get("model_name"),
                    "prediction": _safe_float(ensemble_row.get("y_pred")),
                    "component_models": ensemble_row.get("component_models"),
                    "component_count": _safe_float(ensemble_row.get("component_count")),
                }
            )
            if ensemble_row is not None
            else "",
            "regime_summary": _json_dumps(
                {
                    "regime_label": regime_label,
                    "regime_prob_bull": _safe_float(regime_row.get("regime_prob_bull")),
                    "regime_prob_bear": _safe_float(regime_row.get("regime_prob_bear")),
                    "regime_prob_sideway": _safe_float(regime_row.get("regime_prob_sideway")),
                    "source_model": regime_row.get("source_model"),
                }
            )
            if regime_row
            else "",
            "risk_summary": _json_dumps(
                {
                    "risk_model": risk_row.get("risk_model"),
                    "vol_forecast": vol_forecast,
                    "var_loss_95": _safe_float(risk_row.get("var_loss_95")),
                    "cvar_loss_95": _safe_float(risk_row.get("cvar_loss_95")),
                    "drawdown_state": risk_row.get("drawdown_state"),
                    "current_drawdown": _safe_float(risk_row.get("current_drawdown")),
                    "max_drawdown": _safe_float(risk_row.get("max_drawdown")),
                }
            )
            if risk_row
            else "",
            "policy_summary": _json_dumps(
                {
                    "active_signal_count": active_signal_count,
                    "long_signal_count": long_signal_count,
                    "short_signal_count": short_signal_count,
                    "mean_position_size": float(np.mean(position_sizes)) if position_sizes else None,
                    "top_policy_model": top_policy_row.get("model_name") if top_policy_row else None,
                    "top_policy_sharpe": _safe_float(top_policy_row.get("sharpe")) if top_policy_row else None,
                    "top_policy_cagr": _safe_float(top_policy_row.get("cagr")) if top_policy_row else None,
                }
            ),
            "active_signal_count": int(active_signal_count),
            "long_signal_count": int(long_signal_count),
            "short_signal_count": int(short_signal_count),
            "mean_position_size": float(np.mean(position_sizes)) if position_sizes else None,
            "top_policy_model": top_policy_row.get("model_name") if top_policy_row else None,
            "top_policy_sharpe": _safe_float(top_policy_row.get("sharpe")) if top_policy_row else None,
            "top_policy_cagr": _safe_float(top_policy_row.get("cagr")) if top_policy_row else None,
            "realized_y_true": _safe_float(primary_row.get("y_true")),
            "realized_available": _safe_float(primary_row.get("y_true")) is not None,
            "target_timestamp": _safe_timestamp(packet.get("target_timestamp")),
            "regime_label": regime_label,
            "vol_forecast": vol_forecast,
            "volatility_bucket": _volatility_bucket(vol_forecast),
            "signal_strength_bucket": _signal_strength_bucket(primary_prediction),
            "model_role_context": ",".join(sorted({str(value) for value in ordered.get("model_role", pd.Series(dtype="object")).astype(str).unique()})),
            "retrieval_metadata": _json_dumps(
                {
                    "ticker": packet.get("ticker"),
                    "ticker_group": packet.get("group_name"),
                    "horizon": packet.get("horizon"),
                    "target_type": packet.get("target_type"),
                    "run_mode": packet.get("run_mode"),
                    "regime_label": regime_label,
                    "volatility_bucket": _volatility_bucket(vol_forecast),
                    "signal_strength_bucket": _signal_strength_bucket(primary_prediction),
                    "agreement_bucket": agreement_bucket,
                    "model_role_context": ",".join(sorted({str(value) for value in ordered.get("model_role", pd.Series(dtype="object")).astype(str).unique()})),
                    "cost_mode": top_policy_row.get("cost_mode") if top_policy_row else None,
                }
            ),
        }
        packet_rows.append(packet_row)
    return pd.DataFrame(packet_rows).sort_values(group_columns).reset_index(drop=True)


def build_decision_lane_candidates(packets_df: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative candidate view for later analyst review."""

    if packets_df.empty:
        return pd.DataFrame(columns=["packet_id", "ticker", "timestamp", "primary_prediction", "agreement_bucket"])
    candidates = packets_df.copy()
    tradable = candidates.get("target_tradable", pd.Series(False, index=candidates.index)).fillna(False).astype(bool)
    agreement = candidates.get("agreement_bucket", pd.Series("unknown", index=candidates.index)).astype(str)
    prediction = pd.to_numeric(candidates.get("primary_prediction"), errors="coerce")
    active_signal_count = pd.to_numeric(candidates.get("active_signal_count"), errors="coerce").fillna(0.0)
    candidates = candidates[
        tradable
        & prediction.gt(0.0)
        & agreement.isin(["medium", "high"])
        & active_signal_count.gt(0.0)
    ].copy()
    candidates["candidate_score"] = (
        prediction.fillna(0.0)
        * pd.to_numeric(candidates.get("model_agreement_score"), errors="coerce").fillna(0.0)
    )
    ordered_columns = [
        "packet_id",
        "timestamp",
        "ticker",
        "group_name",
        "horizon",
        "target_type",
        "run_mode",
        "primary_model_name",
        "primary_prediction",
        "model_agreement_score",
        "agreement_bucket",
        "regime_label",
        "volatility_bucket",
        "active_signal_count",
        "top_policy_model",
        "top_policy_sharpe",
        "candidate_score",
    ]
    return candidates.sort_values(["candidate_score", "top_policy_sharpe"], ascending=[False, False]).reset_index(drop=True)[ordered_columns]


def write_analysis_packets_jsonl(
    output_dir: str | Path,
    packets_df: pd.DataFrame,
    *,
    filename: str = "analysis_packets.jsonl",
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    with path.open("w", encoding="utf-8") as handle:
        for record in packets_df.to_dict(orient="records"):
            handle.write(_json_dumps(record))
            handle.write("\n")
    return path
