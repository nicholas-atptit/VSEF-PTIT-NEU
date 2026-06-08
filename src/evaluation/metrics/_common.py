from __future__ import annotations

import numpy as np
import pandas as pd


def paired_numeric(*values: object) -> list[np.ndarray]:
    frame = pd.DataFrame(
        {
            str(index): pd.to_numeric(pd.Series(value), errors="coerce").reset_index(drop=True)
            for index, value in enumerate(values)
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return [frame[column].to_numpy() for column in frame.columns]
