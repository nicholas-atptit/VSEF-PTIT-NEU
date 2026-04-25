"""Rule-based leakage and governance review for diagnostic feature outputs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ml.feature_engineering import (
    LEGACY_COMPATIBILITY_COLUMNS,
    NON_CANONICAL_ALIAS_COLUMNS,
    PRICE_REFERENCE_COLUMNS,
    FeatureEngineer,
)
from src.ml.features.registry import feature_lookup

FEATURE_GOVERNANCE_REVIEW_COLUMNS = [
    "feature",
    "governance_category",
    "risk_level",
    "reason",
    "source_hint",
    "is_context_feature",
    "is_regime_feature",
    "is_risk_feature",
    "is_alias_feature",
    "is_lagged_or_trailing",
    "appears_in_linear_stability",
    "appears_in_importance_stability",
    "best_linear_stability_level",
    "best_importance_stability_level",
    "best_top_10_ratio",
    "best_sign_consistency_ratio",
    "recommended_action",
]

GOVERNANCE_CATEGORIES = {
    "safe_trailing",
    "requires_review",
    "alias_or_redundant",
    "potential_leakage",
    "target_derived",
    "unknown",
}
RISK_LEVELS = {"low", "medium", "high", "unknown"}
RECOMMENDED_ACTIONS = {
    "keep",
    "keep_but_document",
    "review_timing",
    "review_redundancy",
    "exclude_until_verified",
}

STABILITY_ORDER = {"missing": -1, "low": 0, "medium": 1, "high": 2}
ALIAS_FEATURES = set(LEGACY_COMPATIBILITY_COLUMNS) | set(NON_CANONICAL_ALIAS_COLUMNS) | {
    "m_ret",
    "m_ret_5d",
    "m_ret_20d",
    "s_ret",
    "close_raw",
}
RISK_FEATURE_TOKENS = (
    "risk",
    "var",
    "cvar",
    "covar",
    "drawdown",
    "volatility",
    "atr",
    "amihud",
    "abnormal_gap",
    "corporate_action",
)
CONTEXT_PREFIXES = (
    "m_",
    "s_",
    "market_",
    "sector_",
    "relative_strength_market",
    "relative_strength_sector",
    "rolling_beta_",
    "rolling_corr_",
    "breadth_",
    "advance_decline",
    "new_high",
    "new_low",
    "up_down_volume",
    "foreign_",
    "fx_",
    "interest_",
    "gold_",
    "oil_",
    "macro_",
)
TRAILING_TOKENS = (
    "lag",
    "rolling",
    "return",
    "ret_",
    "momentum",
    "roc",
    "sma",
    "ema",
    "rsi",
    "macd",
    "bb_",
    "stoch",
    "adx",
    "atr",
    "zscore",
    "ratio",
    "volatility",
    "drawdown",
    "turnover",
    "amihud",
    "persistence",
    "transition",
)
TARGET_DERIVED_TOKENS = (
    "target_",
    "target_return",
    "target_direction",
    "target_profit",
    "target_net_return",
    "target_date",
    "forward_return",
)
FUTURE_LEAKAGE_TOKENS = (
    "future_",
    "_future",
    "lookahead",
    "look_ahead",
    "next_",
    "lead_",
    "_lead",
    "ahead",
    "tomorrow",
)


def empty_feature_governance_review_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_GOVERNANCE_REVIEW_COLUMNS)


def _safe_registry_lookup() -> dict[str, dict[str, Any]]:
    try:
        return feature_lookup()
    except Exception:
        return {}


def _best_stability_level(values: pd.Series | Iterable[Any]) -> str:
    series = pd.Series(values).dropna()
    if series.empty:
        return "missing"
    levels = [str(value).strip().lower() for value in series if str(value).strip()]
    if not levels:
        return "missing"
    return max(levels, key=lambda value: STABILITY_ORDER.get(value, -1))


def _category_from_registry(feature: str, registry_entry: dict[str, Any] | None) -> str:
    if registry_entry and registry_entry.get("category"):
        return str(registry_entry["category"])
    return FeatureEngineer._feature_category(feature)


def _is_context_feature(feature: str, category: str) -> bool:
    return category in {"market_context", "macro_cross_asset", "flow_microstructure", "sentiment_news"} or feature.startswith(CONTEXT_PREFIXES)


def _is_risk_feature(feature: str, category: str) -> bool:
    lowered = feature.lower()
    return category == "corporate_action_diagnostics" or any(token in lowered for token in RISK_FEATURE_TOKENS)


def _is_lagged_or_trailing(feature: str, registry_entry: dict[str, Any] | None) -> bool:
    lowered = feature.lower()
    formula = str((registry_entry or {}).get("formula_logic", "")).lower()
    if any(token in lowered for token in TRAILING_TOKENS):
        return True
    return "rolling" in formula or "lagged" in formula or "current and past" in str((registry_entry or {}).get("leakage_note", "")).lower()


def _source_hint(feature: str, registry_entry: dict[str, Any] | None, category: str) -> str:
    if registry_entry:
        input_source = str(registry_entry.get("input_source", "")).strip()
        expected = str(registry_entry.get("expected_availability", "")).strip()
        if input_source and expected:
            return f"{input_source}; {expected}"
        if input_source:
            return input_source
        if expected:
            return expected
    if category == "market_context":
        return "market/sector/breadth context"
    if category == "macro_cross_asset":
        return "macro/commodity context"
    if category == "sentiment_news":
        return "sentiment/news context"
    return "feature name rule"


def classify_feature_governance(
    feature: str,
    *,
    registry_entry: dict[str, Any] | None = None,
    linear_stability_level: str = "missing",
    importance_stability_level: str = "missing",
) -> dict[str, Any]:
    name = str(feature)
    lowered = name.lower()
    category = _category_from_registry(name, registry_entry)
    is_alias = name in ALIAS_FEATURES or lowered.startswith("d_") and lowered[2:] in ALIAS_FEATURES
    is_context = _is_context_feature(lowered, category)
    is_regime = "regime" in lowered
    is_risk = _is_risk_feature(lowered, category)
    is_trailing = _is_lagged_or_trailing(lowered, registry_entry)
    source_hint = _source_hint(name, registry_entry, category)

    target_like = lowered.startswith("target_") or any(token in lowered for token in TARGET_DERIVED_TOKENS)
    future_like = any(token in lowered for token in FUTURE_LEAKAGE_TOKENS) or name in PRICE_REFERENCE_COLUMNS
    unstable = (
        str(linear_stability_level).lower() == "low"
        or str(importance_stability_level).lower() == "low"
    )

    if target_like:
        governance_category = "target_derived"
        risk_level = "high"
        reason = "Feature name matches target or forward-return construction and should not enter model features."
        recommended_action = "exclude_until_verified"
    elif future_like:
        governance_category = "potential_leakage"
        risk_level = "high"
        reason = "Feature name suggests future, lead, lookahead, or price-reference information requiring timing proof."
        recommended_action = "exclude_until_verified"
    elif is_alias:
        governance_category = "alias_or_redundant"
        risk_level = "medium"
        reason = "Feature is a compatibility alias, legacy column, or overlapping definition that can duplicate a canonical signal."
        recommended_action = "review_redundancy"
    elif is_context or category in {"sentiment_news", "macro_cross_asset", "flow_microstructure"}:
        governance_category = "requires_review"
        risk_level = "medium"
        reason = "External or joined context feature is valid only if source timestamps are aligned without forward-looking fills."
        recommended_action = "review_timing"
    elif is_trailing or is_regime or is_risk:
        governance_category = "safe_trailing"
        risk_level = "medium" if unstable else "low"
        reason = "Feature appears to be lagged, trailing, regime/risk, or current-day known data based on transparent rules."
        recommended_action = "keep_but_document" if unstable else "keep"
    elif registry_entry:
        governance_category = "requires_review"
        risk_level = "medium"
        reason = "Feature is registry-known but does not match a simple trailing or context rule; confirm timing before stronger use."
        recommended_action = "review_timing"
    else:
        governance_category = "unknown"
        risk_level = "unknown"
        reason = "Feature is not recognized by registry metadata or conservative name rules."
        recommended_action = "review_timing"

    return {
        "feature": name,
        "governance_category": governance_category,
        "risk_level": risk_level,
        "reason": reason,
        "source_hint": source_hint,
        "is_context_feature": bool(is_context),
        "is_regime_feature": bool(is_regime),
        "is_risk_feature": bool(is_risk),
        "is_alias_feature": bool(is_alias),
        "is_lagged_or_trailing": bool(is_trailing),
        "recommended_action": recommended_action,
    }


def _aggregate_linear(linear_summary: pd.DataFrame | None) -> pd.DataFrame:
    if linear_summary is None or linear_summary.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "appears_in_linear_stability",
                "best_linear_stability_level",
                "best_sign_consistency_ratio",
            ]
        )
    working = linear_summary.copy()
    working["sign_consistency_ratio"] = pd.to_numeric(working.get("sign_consistency_ratio"), errors="coerce")
    grouped = (
        working.groupby("feature", sort=True)
        .agg(
            appears_in_linear_stability=("feature", "size"),
            best_linear_stability_level=("stability_level", _best_stability_level),
            best_sign_consistency_ratio=("sign_consistency_ratio", "max"),
        )
        .reset_index()
    )
    grouped["appears_in_linear_stability"] = grouped["appears_in_linear_stability"] > 0
    return grouped


def _aggregate_importance(importance_summary: pd.DataFrame | None) -> pd.DataFrame:
    if importance_summary is None or importance_summary.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "appears_in_importance_stability",
                "best_importance_stability_level",
                "best_top_10_ratio",
            ]
        )
    working = importance_summary.copy()
    working["top_10_ratio"] = pd.to_numeric(working.get("top_10_ratio"), errors="coerce")
    grouped = (
        working.groupby("feature", sort=True)
        .agg(
            appears_in_importance_stability=("feature", "size"),
            best_importance_stability_level=("importance_stability_level", _best_stability_level),
            best_top_10_ratio=("top_10_ratio", "max"),
        )
        .reset_index()
    )
    grouped["appears_in_importance_stability"] = grouped["appears_in_importance_stability"] > 0
    return grouped


def build_feature_governance_review(
    *,
    linear_summary: pd.DataFrame | None = None,
    importance_summary: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    feature_names: Iterable[str] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    lookup = registry if registry is not None else _safe_registry_lookup()
    features = {str(name) for name in (feature_names or []) if str(name).strip()}
    for frame in (linear_summary, importance_summary, comparison):
        if frame is not None and not frame.empty and "feature" in frame.columns:
            features.update(str(value) for value in frame["feature"].dropna().astype(str) if str(value).strip())
    if not features:
        return empty_feature_governance_review_frame()

    linear_agg = _aggregate_linear(linear_summary)
    importance_agg = _aggregate_importance(importance_summary)
    metrics = pd.DataFrame({"feature": sorted(features)})
    metrics = metrics.merge(linear_agg, on="feature", how="left")
    metrics = metrics.merge(importance_agg, on="feature", how="left")

    defaults = {
        "appears_in_linear_stability": False,
        "appears_in_importance_stability": False,
        "best_linear_stability_level": "missing",
        "best_importance_stability_level": "missing",
        "best_top_10_ratio": np.nan,
        "best_sign_consistency_ratio": np.nan,
    }
    for column, value in defaults.items():
        if column not in metrics.columns:
            metrics[column] = value
        metrics[column] = metrics[column].fillna(value)

    rows: list[dict[str, Any]] = []
    for item in metrics.itertuples(index=False):
        classification = classify_feature_governance(
            str(item.feature),
            registry_entry=lookup.get(str(item.feature)),
            linear_stability_level=str(item.best_linear_stability_level),
            importance_stability_level=str(item.best_importance_stability_level),
        )
        rows.append(
            {
                **classification,
                "appears_in_linear_stability": bool(item.appears_in_linear_stability),
                "appears_in_importance_stability": bool(item.appears_in_importance_stability),
                "best_linear_stability_level": str(item.best_linear_stability_level),
                "best_importance_stability_level": str(item.best_importance_stability_level),
                "best_top_10_ratio": item.best_top_10_ratio,
                "best_sign_consistency_ratio": item.best_sign_consistency_ratio,
            }
        )

    return pd.DataFrame(rows).reindex(columns=FEATURE_GOVERNANCE_REVIEW_COLUMNS)
