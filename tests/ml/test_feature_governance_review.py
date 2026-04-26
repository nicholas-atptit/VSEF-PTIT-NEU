from __future__ import annotations

import pandas as pd

from src.ml.backtest.feature_governance_review import (
    FEATURE_GOVERNANCE_REVIEW_COLUMNS,
    build_feature_governance_review,
    classify_feature_governance,
    empty_feature_governance_review_frame,
)


def test_trailing_or_lagged_features_classify_as_safe_trailing() -> None:
    review = classify_feature_governance("close_return_20d")

    assert review["governance_category"] == "safe_trailing"
    assert review["risk_level"] == "low"
    assert review["recommended_action"] == "keep"
    assert review["is_lagged_or_trailing"] is True


def test_alias_features_classify_as_alias_or_redundant() -> None:
    review = classify_feature_governance("m_ret_20d")

    assert review["governance_category"] == "alias_or_redundant"
    assert review["risk_level"] == "medium"
    assert review["recommended_action"] == "review_redundancy"
    assert review["is_alias_feature"] is True


def test_target_like_and_future_like_names_are_flagged_conservatively() -> None:
    target_review = classify_feature_governance("target_return_short_5d")
    future_review = classify_feature_governance("next_return_5d")

    assert target_review["governance_category"] == "target_derived"
    assert target_review["risk_level"] == "high"
    assert target_review["recommended_action"] == "exclude_until_verified"
    assert future_review["governance_category"] == "potential_leakage"
    assert future_review["risk_level"] == "high"


def test_unknown_features_classify_as_unknown() -> None:
    review = classify_feature_governance("mystery_alpha_feature")

    assert review["governance_category"] == "unknown"
    assert review["risk_level"] == "unknown"
    assert review["recommended_action"] == "review_timing"


def test_context_features_require_timing_review() -> None:
    review = classify_feature_governance("foreign_net_value_ratio")

    assert review["governance_category"] == "requires_review"
    assert review["risk_level"] == "medium"
    assert review["recommended_action"] == "review_timing"
    assert review["is_context_feature"] is True


def test_governance_review_merges_linear_and_importance_stability() -> None:
    linear_summary = pd.DataFrame(
        [
            {
                "model": "linear",
                "horizon": "short_5d",
                "task": "return",
                "feature": "close_return_20d",
                "sign_consistency_ratio": 0.9,
                "stability_level": "high",
            },
            {
                "model": "ridge",
                "horizon": "short_5d",
                "task": "return",
                "feature": "rolling_volatility_20",
                "sign_consistency_ratio": 0.4,
                "stability_level": "low",
            },
        ]
    )
    importance_summary = pd.DataFrame(
        [
            {
                "model": "cart",
                "horizon": "short_5d",
                "task": "return",
                "feature": "close_return_20d",
                "top_10_ratio": 1.0,
                "importance_stability_level": "high",
            },
            {
                "model": "xgboost",
                "horizon": "short_5d",
                "task": "return",
                "feature": "foreign_net_value_ratio",
                "top_10_ratio": 0.6,
                "importance_stability_level": "medium",
            },
        ]
    )

    review = build_feature_governance_review(
        linear_summary=linear_summary,
        importance_summary=importance_summary,
    )
    by_feature = review.set_index("feature")

    assert list(review.columns) == FEATURE_GOVERNANCE_REVIEW_COLUMNS
    assert bool(by_feature.loc["close_return_20d", "appears_in_linear_stability"]) is True
    assert bool(by_feature.loc["close_return_20d", "appears_in_importance_stability"]) is True
    assert by_feature.loc["close_return_20d", "best_linear_stability_level"] == "high"
    assert by_feature.loc["close_return_20d", "best_importance_stability_level"] == "high"
    assert by_feature.loc["close_return_20d", "best_top_10_ratio"] == 1.0
    assert by_feature.loc["close_return_20d", "best_sign_consistency_ratio"] == 0.9
    assert by_feature.loc["foreign_net_value_ratio", "recommended_action"] == "review_timing"
    assert by_feature.loc["rolling_volatility_20", "recommended_action"] == "keep_but_document"


def test_missing_diagnostics_do_not_crash_review_utility() -> None:
    review = build_feature_governance_review(
        feature_names=["close_return_20d", "mystery_alpha_feature"],
        linear_summary=pd.DataFrame(),
        importance_summary=pd.DataFrame(),
    )

    assert set(review["feature"]) == {"close_return_20d", "mystery_alpha_feature"}
    assert list(empty_feature_governance_review_frame().columns) == FEATURE_GOVERNANCE_REVIEW_COLUMNS
