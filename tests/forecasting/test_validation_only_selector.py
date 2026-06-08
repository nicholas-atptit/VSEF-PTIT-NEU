import pandas as pd

from src.forecasting.selectors import select_direction


def test_final_rows_cannot_select_direction_model():
    rows = pd.DataFrame(
        [
            {"model": "validation", "split": "validation", "balanced_accuracy": 0.6, "macro_f1": 0.6, "mcc": 0.2, "prediction_balance": 0.5},
            {"model": "final", "split": "final", "balanced_accuracy": 1.0, "macro_f1": 1.0, "mcc": 1.0, "prediction_balance": 0.5},
        ]
    )
    assert select_direction(rows)["model"] == "validation"
