"""Future-blind timestamp split discipline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitPolicy:
    train_end: pd.Timestamp = pd.Timestamp("2023-12-31 23:59:59")
    validation_start: pd.Timestamp = pd.Timestamp("2024-01-01 00:00:00")
    validation_end: pd.Timestamp = pd.Timestamp("2024-12-31 23:59:59")
    final_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")

    def classify(self, feature_timestamp: object, target_timestamp: object) -> str | None:
        feature = pd.to_datetime(feature_timestamp, errors="coerce")
        target = pd.to_datetime(target_timestamp, errors="coerce")
        if pd.isna(feature) or pd.isna(target):
            return None
        if feature <= self.train_end and target <= self.train_end:
            return "train"
        if self.validation_start <= feature <= self.validation_end and self.validation_start <= target <= self.validation_end:
            return "validation"
        if feature >= self.final_start and target >= self.final_start:
            return "final"
        return None

    def masks(
        self,
        frame: pd.DataFrame,
        *,
        feature_column: str = "feature_timestamp",
        target_column: str = "target_timestamp",
    ) -> dict[str, pd.Series]:
        feature = pd.to_datetime(frame[feature_column], errors="coerce")
        target = pd.to_datetime(frame[target_column], errors="coerce")
        return {
            "train": (feature <= self.train_end) & (target <= self.train_end),
            "validation": feature.between(self.validation_start, self.validation_end)
            & target.between(self.validation_start, self.validation_end),
            "final": (feature >= self.final_start) & (target >= self.final_start),
        }


DEFAULT_SPLIT_POLICY = SplitPolicy()
TRAIN_END = DEFAULT_SPLIT_POLICY.train_end
VAL_START = DEFAULT_SPLIT_POLICY.validation_start
VAL_END = DEFAULT_SPLIT_POLICY.validation_end
FINAL_START = DEFAULT_SPLIT_POLICY.final_start


def assign_split(
    frame: pd.DataFrame,
    *,
    feature_column: str = "feature_timestamp",
    target_column: str = "target_timestamp",
    split_column: str = "split",
    policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
) -> pd.DataFrame:
    """Return a copy with strict split labels; boundary-crossing rows stay null."""

    result = frame.copy()
    result[split_column] = pd.Series(pd.NA, index=result.index, dtype="string")
    for split, mask in policy.masks(result, feature_column=feature_column, target_column=target_column).items():
        result.loc[mask, split_column] = split
    return result
