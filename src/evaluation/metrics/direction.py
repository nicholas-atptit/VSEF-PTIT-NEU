"""Direction classification metrics, including repaired balance metrics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ._common import paired_numeric


def direction_metrics(y_true: object, y_pred: object, y_probability: object | None = None) -> dict[str, float | int]:
    arrays = paired_numeric(y_true, y_pred, *([y_probability] if y_probability is not None else []))
    if not arrays or len(arrays[0]) == 0:
        return {"rows": 0, "raw_accuracy": math.nan, "balanced_accuracy": math.nan, "macro_f1": math.nan, "mcc": math.nan, "precision": math.nan, "recall": math.nan, "auc": math.nan, "brier_score": math.nan, "prediction_balance": math.nan}
    true = arrays[0].astype(int)
    pred = arrays[1].astype(int)
    probability = arrays[2] if y_probability is not None else None
    auc = math.nan
    brier = math.nan
    if probability is not None:
        brier = float(brier_score_loss(true, probability))
        if len(np.unique(true)) > 1:
            auc = float(roc_auc_score(true, probability))
    return {
        "rows": int(len(true)),
        "raw_accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "auc": auc,
        "brier_score": brier,
        "prediction_balance": float(np.mean(pred == 1)),
    }
