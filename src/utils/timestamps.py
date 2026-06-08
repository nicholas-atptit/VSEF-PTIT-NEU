"""Timestamp parsing helpers."""

from __future__ import annotations

import pandas as pd


def parse_timestamp(value: object, *, required: bool = False) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        if required:
            raise ValueError(f"Invalid timestamp: {value!r}")
        return None
    return pd.Timestamp(parsed)
