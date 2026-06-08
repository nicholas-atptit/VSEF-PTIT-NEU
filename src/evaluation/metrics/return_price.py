"""Return and price forecast metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._common import paired_numeric


def return_price_metrics(actual: object, predicted: object) -> dict[str, float | int]:
    arrays = paired_numeric(actual, predicted)
    if not arrays or len(arrays[0]) == 0:
        return {"rows": 0, "rmse": math.nan, "mae": math.nan, "smape": math.nan, "correlation_pred_actual": math.nan, "sign_accuracy": math.nan, "rank_ic": math.nan}
    true, pred = arrays
    error = pred - true
    denominator = np.abs(true) + np.abs(pred)
    valid_smape = denominator > 0
    smape = float(np.mean(2.0 * np.abs(error[valid_smape]) / denominator[valid_smape])) if valid_smape.any() else math.nan
    correlation = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0 else math.nan
    rank_ic = float(pd.Series(true).corr(pd.Series(pred), method="spearman")) if len(true) > 1 else math.nan
    nonzero = true != 0
    return {
        "rows": int(len(true)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "smape": smape,
        "correlation_pred_actual": correlation,
        "sign_accuracy": float(np.mean(np.sign(true[nonzero]) == np.sign(pred[nonzero]))) if nonzero.any() else math.nan,
        "rank_ic": rank_ic,
    }
