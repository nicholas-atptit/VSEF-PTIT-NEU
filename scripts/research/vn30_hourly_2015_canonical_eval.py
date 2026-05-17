"""Canonical evaluator for VN30 hourly 2015 benchmark.

Version: 1.0.0
Date: 2026-05-17

This module defines a single, unambiguous accuracy computation method
for all VN30 hourly 2015 experiments. All later experiments must use
this evaluator to ensure consistency.

Rules:
1. Row-level accuracy: count correct predictions / total valid predictions.
   - Valid = label is not NaN.
   - Correct = prediction equals label.
2. Pooled/global accuracy: sum of correct across all tickers / sum of valid across all tickers.
   - NOT macro-average (no per-ticker averaging).
   - NOT weighted by ticker market cap or any external factor.
3. Coverage: valid rows after filtering / total valid rows before filtering.
4. Row count: number of valid (non-NaN) label rows in the evaluation set.
5. No silent filtering: all confidence/ticker/regime filters must be explicit.
6. No macro/micro ambiguity: always use pooled (micro) accuracy.
7. Validation/final split separation: compute metrics separately for each split.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

EVALUATOR_VERSION = "canonical_v1.0.0"

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute canonical accuracy metrics.

    Args:
        y_true: Ground truth labels (float, NaN for invalid).
        y_pred: Predicted labels (int/float).

    Returns:
        Dict with accuracy, correct_count, incorrect_count, total_valid.
    """
    valid_mask = ~np.isnan(y_true)
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return {"accuracy": 0.0, "correct_count": 0, "incorrect_count": 0, "total_valid": 0}
    correct = int(np.sum(y_true[valid_mask] == y_pred[valid_mask]))
    incorrect = total_valid - correct
    accuracy = correct / total_valid
    return {"accuracy": accuracy, "correct_count": correct, "incorrect_count": incorrect, "total_valid": total_valid}

def compute_pooled_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pooled (micro) accuracy across multiple ticker/experiment results.

    Args:
        results: List of dicts from compute_accuracy().

    Returns:
        Dict with pooled_accuracy, total_correct, total_valid.
    """
    total_correct = sum(r["correct_count"] for r in results)
    total_valid = sum(r["total_valid"] for r in results)
    if total_valid == 0:
        return {"pooled_accuracy": 0.0, "total_correct": 0, "total_valid": 0}
    return {"pooled_accuracy": total_correct / total_valid, "total_correct": total_correct, "total_valid": total_valid}

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         confidence: np.ndarray | None = None,
                         ticker_mask: np.ndarray | None = None,
                         regime_mask: np.ndarray | None = None,
                         total_valid_before_filter: int | None = None) -> dict[str, Any]:
    """Evaluate predictions with optional filters.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        confidence: Optional confidence scores for threshold filtering.
        ticker_mask: Optional boolean mask for ticker filtering.
        regime_mask: Optional boolean mask for regime filtering.
        total_valid_before_filter: Total valid rows before any filtering (for coverage calc).

    Returns:
        Dict with all canonical metrics.
    """
    valid_mask = ~np.isnan(y_true)
    total_valid = int(valid_mask.sum())

    # Apply filters
    combined_mask = valid_mask.copy()
    if confidence is not None:
        combined_mask = combined_mask & (confidence >= 0.5)  # default threshold, should be overridden
    if ticker_mask is not None:
        combined_mask = combined_mask & ticker_mask
    if regime_mask is not None:
        combined_mask = combined_mask & regime_mask

    filtered_valid = int(combined_mask.sum())
    if filtered_valid == 0:
        coverage = 0.0
        return {"accuracy": 0.0, "correct_count": 0, "incorrect_count": 0,
                "total_valid": 0, "filtered_valid": 0, "coverage": coverage,
                "pass_60": False, "pass_65": False, "claim_level": "failed"}

    correct = int(np.sum(y_true[combined_mask] == y_pred[combined_mask]))
    incorrect = filtered_valid - correct
    accuracy = correct / filtered_valid

    # Coverage: filtered valid / total valid (or total_valid_before_filter if provided)
    denominator = total_valid_before_filter if total_valid_before_filter is not None else total_valid
    coverage = filtered_valid / denominator if denominator > 0 else 0.0

    pass_60 = accuracy >= 0.60 and coverage >= 0.95 and filtered_valid >= 1000
    pass_65 = accuracy >= 0.65 and coverage >= 0.30 and filtered_valid >= 1000

    if pass_65:
        claim_level = "final65_coverage_qualified"
    elif pass_60:
        claim_level = "baseline60_global"
    elif accuracy >= 0.65:
        claim_level = "exploratory"
    else:
        claim_level = "failed"

    return {"accuracy": accuracy, "correct_count": correct, "incorrect_count": incorrect,
            "total_valid": filtered_valid, "filtered_valid": filtered_valid,
            "coverage": coverage, "pass_60": pass_60, "pass_65": pass_65,
            "claim_level": claim_level}

def classify_result(accuracy: float, coverage: float, rows: int,
                    is_full_universe: bool = False) -> str:
    """Classify result into claim level.

    Args:
        accuracy: Final accuracy.
        coverage: Coverage ratio (0-1).
        rows: Number of valid rows.
        is_full_universe: Whether this is a full-universe result.

    Returns:
        Claim level string.
    """
    if accuracy >= 0.65 and coverage >= 0.30 and rows >= 1000:
        return "final65_coverage_qualified"
    if accuracy >= 0.60 and coverage >= 0.95 and rows >= 1000 and is_full_universe:
        return "baseline60_global"
    if accuracy >= 0.65:
        return "exploratory"
    return "failed"
