"""Stable serialization helpers for research runners."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import relative_to_repo


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_frame(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[columns]
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return relative_to_repo(path)


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan
