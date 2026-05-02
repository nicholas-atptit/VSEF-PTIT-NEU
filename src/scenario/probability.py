"""Deterministic scenario probability scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scenario.schema import (
    PACKET_CONTEXT_COLUMNS,
    SCENARIO_LABELS,
    SCENARIO_REQUIRED_FIELDS,
    present_columns,
)


SOURCE_MODEL = "scenario_engine_v1:deterministic_v1"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric) or not np.isfinite(numeric):
        return default
    return float(numeric)


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return lower
    return float(min(max(value, lower), upper))


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame, preferred: list[str] | tuple[str, ...]) -> list[str]:
    return [column for column in preferred if column in left.columns and column in right.columns]


def _context_columns(frame: pd.DataFrame) -> list[str]:
    columns = present_columns(frame, PACKET_CONTEXT_COLUMNS)
    if columns:
        return columns
    fallback = [column for column in ("timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id") if column in frame.columns]
    return fallback or list(frame.columns[:1])


def _lookup_by_keys(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if frame.empty or not columns:
        return {}
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        indexed[tuple(row.get(column) for column in columns)] = row
    return indexed


def _lookup_groups(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], pd.DataFrame]:
    if frame.empty or not columns:
        return {}
    groups: dict[tuple[Any, ...], pd.DataFrame] = {}
    for keys, group in frame.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        groups[keys] = group.copy()
    return groups


def _key_from_row(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _target_centered_predictions(group: pd.DataFrame, target_type: str, target_family: str | None) -> pd.Series:
    predictions = pd.to_numeric(group.get("y_pred", pd.Series(dtype=float)), errors="coerce").astype(float)
    target_text = f"{target_type} {target_family or ''}".lower()
    if "direction" in target_text or "binary" in target_text:
        return predictions - 0.5
    return predictions


def _health_weight(status: str | None) -> float:
    return {
        "healthy": 1.0,
        "brittle": 0.75,
        "weak": 0.50,
        "failing": 0.20,
    }.get(str(status or "").lower(), 0.80)


def _model_health_factor(group: pd.DataFrame, health_lookup: dict[str, dict[str, Any]]) -> tuple[float, bool]:
    values: list[float] = []
    for model_name in group.get("model_name", pd.Series(dtype="object")).astype(str):
        row = health_lookup.get(model_name)
        if not row:
            continue
        status_weight = _health_weight(row.get("health_status"))
        run_success = _safe_float(row.get("run_success_rate"), default=float("nan"))
        if pd.notna(run_success):
            values.append(0.5 * status_weight + 0.5 * _bounded(run_success))
        else:
            values.append(status_weight)
    if not values:
        return 0.80, True
    return float(np.mean(values)), False


def _strategy_context(strategy_rows: pd.DataFrame, packet_row: dict[str, Any]) -> tuple[float, bool]:
    packet_sharpe = _safe_float(packet_row.get("top_policy_sharpe"), default=float("nan"))
    if pd.notna(packet_sharpe):
        return packet_sharpe, False
    if strategy_rows.empty or "sharpe" not in strategy_rows.columns:
        return 0.0, True
    sharpe = pd.to_numeric(strategy_rows["sharpe"], errors="coerce").dropna()
    if sharpe.empty:
        return 0.0, True
    return float(sharpe.mean()), False


def _risk_context(risk_row: dict[str, Any]) -> dict[str, float | str | bool]:
    if not risk_row:
        return {
            "missing": True,
            "vol_forecast": 0.0,
            "downside_risk": 0.0,
            "drawdown_pressure": 0.0,
            "drawdown_state": "",
        }
    vol = abs(_safe_float(risk_row.get("vol_forecast"), default=0.0))
    var_loss = abs(_safe_float(risk_row.get("var_loss_95"), default=0.0))
    cvar_loss = abs(_safe_float(risk_row.get("cvar_loss_95"), default=0.0))
    current_drawdown = abs(_safe_float(risk_row.get("current_drawdown"), default=0.0))
    max_drawdown = abs(_safe_float(risk_row.get("max_drawdown"), default=0.0))
    downside = max(var_loss, cvar_loss, current_drawdown, max_drawdown * 0.5, 0.0)
    state = str(risk_row.get("drawdown_state", "") or "").lower()
    state_pressure = {"normal": 0.0, "elevated": 0.55, "severe": 1.0}.get(state, 0.0)
    drawdown_pressure = max(_bounded(current_drawdown / 0.15), state_pressure)
    return {
        "missing": False,
        "vol_forecast": vol,
        "downside_risk": downside,
        "drawdown_pressure": drawdown_pressure,
        "drawdown_state": state,
    }


def _regime_context(regime_row: dict[str, Any]) -> dict[str, float | bool]:
    if not regime_row:
        return {
            "missing": True,
            "regime_prob_bull": 1.0 / 3.0,
            "regime_prob_bear": 1.0 / 3.0,
            "regime_prob_sideway": 1.0 / 3.0,
        }
    bull = _safe_float(regime_row.get("regime_prob_bull"), default=float("nan"))
    bear = _safe_float(regime_row.get("regime_prob_bear"), default=float("nan"))
    sideway = _safe_float(regime_row.get("regime_prob_sideway"), default=float("nan"))
    if any(pd.isna(value) for value in (bull, bear, sideway)):
        label = str(regime_row.get("regime_label", "") or "").lower()
        bull = 1.0 if label == "bull" else 0.0
        bear = 1.0 if label == "bear" else 0.0
        sideway = 1.0 if label == "sideway" else 0.0
    total = bull + bear + sideway
    if total <= 0.0:
        bull = bear = sideway = 1.0 / 3.0
    else:
        bull, bear, sideway = bull / total, bear / total, sideway / total
    return {
        "missing": False,
        "regime_prob_bull": _bounded(bull),
        "regime_prob_bear": _bounded(bear),
        "regime_prob_sideway": _bounded(sideway),
    }


def _signal_context(packet_row: dict[str, Any], model_count: int) -> dict[str, float]:
    active = _safe_float(packet_row.get("active_signal_count"), default=0.0)
    long_count = _safe_float(packet_row.get("long_signal_count"), default=0.0)
    short_count = _safe_float(packet_row.get("short_signal_count"), default=0.0)
    denominator = max(float(model_count), active, long_count + short_count, 1.0)
    return {
        "active_signal_share": _bounded(active / denominator),
        "long_signal_share": _bounded(long_count / denominator),
        "short_signal_share": _bounded(short_count / denominator),
    }


def _scenario_id(context: dict[str, Any], label: str) -> str:
    timestamp = context.get("timestamp")
    try:
        date_part = str(pd.Timestamp(timestamp).date())
    except Exception:
        date_part = str(timestamp)
    horizon = int(_safe_float(context.get("horizon"), default=0.0))
    return "|".join(
        [
            str(context.get("ticker")),
            date_part,
            f"h{horizon:02d}",
            str(context.get("target_type")),
            str(context.get("run_mode")),
            str(context.get("core_run_id")),
            label,
        ]
    )


def _expected_outcome(label: str, mean_prediction: float, dispersion: float, downside_risk: float, drawdown_pressure: float) -> float:
    magnitude = max(abs(mean_prediction), dispersion, 0.001)
    if label == "bull":
        return float(max(mean_prediction, 0.0) + 0.25 * magnitude)
    if label == "bear":
        return float(min(mean_prediction, 0.0) - 0.25 * magnitude)
    if label == "sideway":
        return 0.0
    if label == "high_volatility":
        return float(mean_prediction)
    if label == "drawdown":
        return float(-max(downside_risk, magnitude))
    if label == "recovery":
        return float(max(mean_prediction, 0.0) + (0.25 + 0.25 * drawdown_pressure) * magnitude)
    return float(mean_prediction)


def _interval(expected: float, dispersion: float, vol_forecast: float, downside_risk: float, label: str) -> tuple[float, float]:
    spread = max(dispersion, 0.0) + 0.5 * max(vol_forecast, 0.0) + 0.25 * max(downside_risk, 0.0)
    if label in {"high_volatility", "uncertain"}:
        spread *= 1.5
    if spread <= 0.0:
        spread = max(abs(expected) * 0.25, 0.001)
    low = expected - 1.64 * spread
    high = expected + 1.64 * spread
    return float(low), float(high)


def _score_scenarios(
    *,
    positive_share: float,
    negative_share: float,
    neutral_share: float,
    agreement_score: float,
    centered_mean: float,
    strength: float,
    dispersion_norm: float,
    regime: dict[str, float | bool],
    risk: dict[str, float | str | bool],
    strategy_sharpe: float,
    signals: dict[str, float],
    health_factor: float,
    missing_context_share: float,
) -> dict[str, float]:
    bull_regime = float(regime["regime_prob_bull"])
    bear_regime = float(regime["regime_prob_bear"])
    sideway_regime = float(regime["regime_prob_sideway"])
    vol_norm = _bounded(float(risk["vol_forecast"]) / 0.08)
    downside_norm = _bounded(float(risk["downside_risk"]) / 0.12)
    drawdown_pressure = _bounded(float(risk["drawdown_pressure"]))
    sharpe_positive = _bounded((strategy_sharpe + 1.0) / 3.0)
    sharpe_negative = _bounded((-strategy_sharpe + 1.0) / 3.0)
    positive_strength = strength if centered_mean > 0.0 else 0.0
    negative_strength = strength if centered_mean < 0.0 else 0.0
    low_direction_strength = 1.0 - strength
    disagreement = 1.0 - agreement_score

    base = 0.05
    scores = {
        "bull": (
            base
            + 1.20 * positive_share
            + 0.70 * bull_regime
            + 0.40 * positive_strength
            + 0.25 * signals["long_signal_share"]
            + 0.20 * sharpe_positive
            + 0.10 * health_factor
        ),
        "bear": (
            base
            + 1.20 * negative_share
            + 0.70 * bear_regime
            + 0.40 * negative_strength
            + 0.25 * signals["short_signal_share"]
            + 0.15 * sharpe_negative
            + 0.20 * downside_norm
        ),
        "sideway": (
            base
            + 0.90 * max(neutral_share, low_direction_strength)
            + 0.70 * sideway_regime
            + 0.35 * (1.0 - dispersion_norm)
            + 0.10 * (1.0 - signals["active_signal_share"])
        ),
        "high_volatility": (
            base
            + 0.95 * vol_norm
            + 0.75 * dispersion_norm
            + 0.45 * downside_norm
            + 0.25 * disagreement
        ),
        "drawdown": (
            base
            + 0.95 * drawdown_pressure
            + 0.60 * downside_norm
            + 0.45 * negative_share
            + 0.35 * bear_regime
        ),
        "recovery": (
            base
            + 0.70 * positive_share
            + 0.45 * drawdown_pressure
            + 0.40 * bull_regime
            + 0.25 * positive_strength
            + 0.20 * sharpe_positive
        ),
        "uncertain": (
            base
            + 0.85 * disagreement
            + 0.65 * dispersion_norm
            + 0.40 * (1.0 - health_factor)
            + 0.35 * missing_context_share
        ),
    }

    for label in ("bull", "bear", "sideway", "recovery"):
        scores[label] *= 0.75 + 0.25 * health_factor
    scores["uncertain"] *= 1.0 + 0.25 * (1.0 - health_factor)
    return {label: max(float(scores[label]), 1.0e-9) for label in SCENARIO_LABELS}


def build_scenario_probability_frame(
    forecasts_df: pd.DataFrame,
    *,
    consensus_df: pd.DataFrame | None = None,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    strategy_metrics_df: pd.DataFrame | None = None,
    analysis_packets_df: pd.DataFrame | None = None,
    model_health_df: pd.DataFrame | None = None,
    probability_method: str = "deterministic_v1",
) -> pd.DataFrame:
    """Score deterministic scenario probabilities from existing Quant Core outputs."""

    if probability_method != "deterministic_v1":
        raise ValueError(f"Unsupported scenario probability method '{probability_method}'")
    if forecasts_df.empty:
        return pd.DataFrame(columns=[*SCENARIO_REQUIRED_FIELDS, "realized_outcome", "realized_available"])

    forecasts = forecasts_df.copy()
    context_columns = _context_columns(forecasts)
    consensus = consensus_df if consensus_df is not None else pd.DataFrame()
    risk = risk_df if risk_df is not None else pd.DataFrame()
    regime = regime_df if regime_df is not None else pd.DataFrame()
    strategy = strategy_metrics_df if strategy_metrics_df is not None else pd.DataFrame()
    packets = analysis_packets_df if analysis_packets_df is not None else pd.DataFrame()
    health = model_health_df if model_health_df is not None else pd.DataFrame()

    consensus_columns = _shared_columns(forecasts, consensus, context_columns)
    risk_columns = _shared_columns(forecasts, risk, context_columns)
    regime_columns = _shared_columns(forecasts, regime, context_columns)
    packet_columns = _shared_columns(forecasts, packets, context_columns)
    strategy_columns = _shared_columns(
        forecasts,
        strategy,
        [
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
        ],
    )

    consensus_lookup = _lookup_by_keys(consensus, consensus_columns)
    risk_lookup = _lookup_by_keys(risk, risk_columns)
    regime_lookup = _lookup_by_keys(regime, regime_columns)
    packet_lookup = _lookup_by_keys(packets, packet_columns)
    strategy_groups = _lookup_groups(strategy, strategy_columns)
    health_lookup = {
        str(row.get("model_name")): row
        for row in health.to_dict(orient="records")
        if row.get("model_name") is not None
    }

    rows: list[dict[str, Any]] = []
    sort_columns = [*context_columns, *[column for column in ("model_name",) if column in forecasts.columns]]
    forecasts = forecasts.sort_values(sort_columns).reset_index(drop=True)
    for keys, group in forecasts.groupby(context_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(context_columns, keys))
        context_for_required = {column: context.get(column) for column in SCENARIO_REQUIRED_FIELDS if column in context}
        first = group.iloc[0].to_dict()
        for column in (
            "timestamp",
            "ticker",
            "horizon",
            "target_type",
            "run_mode",
            "core_run_id",
            "target_name",
            "target_family",
            "target_tradable",
            "group_name",
            "preset",
            "window_id",
            "target_timestamp",
        ):
            if column not in context_for_required and column in first:
                context_for_required[column] = first.get(column)

        consensus_row = consensus_lookup.get(_key_from_row(context, consensus_columns), {}) if consensus_columns else {}
        risk_row = risk_lookup.get(_key_from_row(context, risk_columns), {}) if risk_columns else {}
        regime_row = regime_lookup.get(_key_from_row(context, regime_columns), {}) if regime_columns else {}
        packet_row = packet_lookup.get(_key_from_row(context, packet_columns), {}) if packet_columns else {}
        strategy_rows = strategy_groups.get(_key_from_row(context, strategy_columns), pd.DataFrame()) if strategy_columns else pd.DataFrame()

        target_type = str(first.get("target_type", context.get("target_type", "")) or "")
        target_family = str(first.get("target_family", "") or "")
        centered = _target_centered_predictions(group, target_type, target_family).dropna()
        if centered.empty:
            centered = pd.Series([0.0])
        signs = np.sign(centered)
        model_count = max(int(len(centered)), 1)
        positive_share = float((signs > 0).sum() / model_count)
        negative_share = float((signs < 0).sum() / model_count)
        neutral_share = float((signs == 0).sum() / model_count)
        centered_mean = float(centered.mean())
        dispersion = _safe_float(consensus_row.get("dispersion_score"), default=float(centered.std(ddof=0)))
        if pd.isna(dispersion):
            dispersion = float(centered.std(ddof=0)) if len(centered) else 0.0
        agreement_score = _safe_float(
            consensus_row.get("agreement_score"),
            default=max(positive_share, negative_share, neutral_share),
        )
        agreement_score = _bounded(agreement_score)
        scale = max(float(centered.abs().quantile(0.75)), abs(centered_mean), 0.01)
        strength = _bounded(abs(centered_mean) / scale)
        dispersion_norm = _bounded(dispersion / (scale + dispersion + 1.0e-12))
        risk_context = _risk_context(risk_row)
        regime_context = _regime_context(regime_row)
        health_factor, missing_health = _model_health_factor(group, health_lookup)
        strategy_sharpe, missing_strategy = _strategy_context(strategy_rows, packet_row)
        signals = _signal_context(packet_row, model_count)
        missing_count = sum(
            [
                bool(risk_context["missing"]),
                bool(regime_context["missing"]),
                bool(missing_health),
                bool(missing_strategy),
            ]
        )
        missing_context_share = float(missing_count / 4.0)

        scores = _score_scenarios(
            positive_share=positive_share,
            negative_share=negative_share,
            neutral_share=neutral_share,
            agreement_score=agreement_score,
            centered_mean=centered_mean,
            strength=strength,
            dispersion_norm=dispersion_norm,
            regime=regime_context,
            risk=risk_context,
            strategy_sharpe=strategy_sharpe,
            signals=signals,
            health_factor=health_factor,
            missing_context_share=missing_context_share,
        )
        total_score = sum(scores.values())
        realized = _safe_float(group.get("y_true", pd.Series(dtype=float)).dropna().iloc[0], default=float("nan")) if "y_true" in group.columns and not group["y_true"].dropna().empty else float("nan")

        for label in SCENARIO_LABELS:
            probability = float(scores[label] / total_score) if total_score > 0.0 else float(1.0 / len(SCENARIO_LABELS))
            expected = _expected_outcome(
                label,
                centered_mean,
                dispersion,
                float(risk_context["downside_risk"]),
                float(risk_context["drawdown_pressure"]),
            )
            interval_low, interval_high = _interval(
                expected,
                dispersion,
                float(risk_context["vol_forecast"]),
                float(risk_context["downside_risk"]),
                label,
            )
            rows.append(
                {
                    **context_for_required,
                    "scenario_id": _scenario_id(context_for_required, label),
                    "scenario_label": label,
                    "scenario_probability": probability,
                    "confidence_adjusted_probability": probability,
                    "expected_outcome": expected,
                    "downside_risk": float(risk_context["downside_risk"]),
                    "confidence_interval_low": interval_low,
                    "confidence_interval_high": interval_high,
                    "uncertainty_score": float("nan"),
                    "dispersion_score": float(dispersion),
                    "dispersion_normalized": dispersion_norm,
                    "dominance_score": float("nan"),
                    "dominant_scenario_flag": False,
                    "calibration_error": float("nan"),
                    "historical_hit_rate": float("nan"),
                    "source_model": SOURCE_MODEL,
                    "realized_outcome": realized,
                    "realized_available": bool(pd.notna(realized)),
                    "model_agreement_score": agreement_score,
                    "missing_context_share": missing_context_share,
                    "model_health_factor": health_factor,
                    "regime_prob_bull": float(regime_context["regime_prob_bull"]),
                    "regime_prob_bear": float(regime_context["regime_prob_bear"]),
                    "regime_prob_sideway": float(regime_context["regime_prob_sideway"]),
                    "vol_forecast": float(risk_context["vol_forecast"]),
                    "drawdown_pressure": float(risk_context["drawdown_pressure"]),
                    "strategy_sharpe": float(strategy_sharpe),
                    "active_signal_share": signals["active_signal_share"],
                }
            )

    result = pd.DataFrame(rows)
    ordered_columns = [
        *SCENARIO_REQUIRED_FIELDS,
        *[column for column in result.columns if column not in SCENARIO_REQUIRED_FIELDS],
    ]
    return result[ordered_columns].sort_values(
        ["timestamp", "ticker", "horizon", "target_type", "run_mode", "core_run_id", "scenario_label"]
    ).reset_index(drop=True)
