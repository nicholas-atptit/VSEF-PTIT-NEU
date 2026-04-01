"""Adapter for integrating the labels package with training scripts.

This module maps user-facing CLI label modes (e.g. ``binary_1d``,
``regression_5d``) to the core label registry and provides configuration
details needed to route them to the correct model type (classifier vs
regressor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ml.labels.base import BaseLabelGenerator
from src.ml.labels import get_generator


@dataclass
class LabelTrainingConfig:
    """Configuration mapping a CLI mode to a label generator and ML task."""
    mode: str                  # e.g., "binary_1d"
    registry_key: str          # e.g., "cls_1d_updown"
    label_column: str          # extracted from generator.label_columns[0]
    task_type: str             # "classification" or "regression"
    n_classes: Optional[int]   # 2, 3, or None for regression
    generator: BaseLabelGenerator


# Mapping of CLI `--label-mode` to details
_LABEL_MODE_MAP = {
    "binary_1d": {
        "registry_key": "cls_1d_updown",
        "task_type": "classification",
        "n_classes": 2,
    },
    "binary_5d": {
        "registry_key": "cls_5d_updown",
        "task_type": "classification",
        "n_classes": 2,
    },
    "ternary_1d": {
        "registry_key": "cls_1d_3class",
        "task_type": "classification",
        "n_classes": 3,
    },
    "ternary_5d": { # Adding 5d ternary as it exists in registry
        "registry_key": "cls_5d_3class",
        "task_type": "classification",
        "n_classes": 3,
    },
    "regression_next_close": { # Adding for completeness
        "registry_key": "reg_next_close_return",
        "task_type": "regression",
        "n_classes": None,
    },
    "regression_5d": {
        "registry_key": "reg_5d_return",
        "task_type": "regression",
        "n_classes": None,
    },
    "volatility_5d": {
        "registry_key": "future_realized_vol_5d",
        "task_type": "regression",
        "n_classes": None,
    },
}

SUPPORTED_LABEL_MODES = list(_LABEL_MODE_MAP.keys())


def resolve_label_config(mode: str, use_settings: bool = True) -> LabelTrainingConfig:
    """Resolve a user-facing label mode into a full training configuration.

    Args:
        mode: The CLI label mode string (e.g., ``binary_1d``).
        use_settings: Pull threshold options from config.settings.

    Returns:
        A ``LabelTrainingConfig`` instance containing the instantiated
        generator and task classification details.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    if mode not in _LABEL_MODE_MAP:
        raise ValueError(
            f"Unknown label mode: {mode}. Supported modes: {SUPPORTED_LABEL_MODES}"
        )

    mapping = _LABEL_MODE_MAP[mode]
    generator = get_generator(mapping["registry_key"], use_settings=use_settings)
    
    # We assume each generator produces exactly one primary label column for training
    label_column = generator.label_columns[0]

    return LabelTrainingConfig(
        mode=mode,
        registry_key=mapping["registry_key"],
        label_column=label_column,
        task_type=mapping["task_type"],
        n_classes=mapping["n_classes"],
        generator=generator,
    )
