"""Cross-sectional ranking metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from ._common import paired_numeric


def _top_precision(actual: np.ndarray, predicted: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(actual) * fraction)))
    actual_top = set(np.argsort(actual)[-count:])
    predicted_top = set(np.argsort(predicted)[-count:])
    return len(actual_top.intersection(predicted_top)) / count


def ranking_metrics(actual_relevance: object, predicted_score: object) -> dict[str, float | int]:
    arrays = paired_numeric(actual_relevance, predicted_score)
    if not arrays or len(arrays[0]) == 0:
        return {"rows": 0, "spearman_ic": math.nan, "ndcg_at_5": math.nan, "ndcg_at_10": math.nan, "top20_precision": math.nan, "top30_precision": math.nan}
    actual, predicted = arrays
    relevance = actual - np.min(actual)
    spearman = math.nan
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
        spearman = float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))
    return {
        "rows": int(len(actual)),
        "spearman_ic": spearman,
        "ndcg_at_5": float(ndcg_score([relevance], [predicted], k=min(5, len(actual)))),
        "ndcg_at_10": float(ndcg_score([relevance], [predicted], k=min(10, len(actual)))),
        "top20_precision": _top_precision(actual, predicted, 0.20),
        "top30_precision": _top_precision(actual, predicted, 0.30),
    }
