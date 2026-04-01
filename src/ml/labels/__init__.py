"""Label engineering package for VN100 daily prediction tasks.

Provides a registry-based system for creating, discovering, and applying
label generators.  Each generator is a sub-class of
:class:`BaseLabelGenerator`.

Quick-start::

    from src.ml.labels import get_generator, apply_all_labels

    # Single label
    df = get_generator("cls_1d_updown").generate(ohlcv_df)

    # All labels at once
    df = apply_all_labels(ohlcv_df)

Registry:
    The ``LABEL_REGISTRY`` dict maps canonical name -> generator class.
    Use :func:`get_generator` to instantiate by name (with optional
    settings-driven threshold overrides).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import pandas as pd

from config.settings import get_settings
from src.ml.labels.base import BaseLabelGenerator
from src.ml.labels.classification import (
    Cls1d3Class,
    Cls1dUpDown,
    Cls20dUpDown,
    Cls5d3Class,
    Cls5dUpDown,
)
from src.ml.labels.regression import Reg5dReturn, RegNextCloseReturn
from src.ml.labels.volatility import FutureRealizedVol5d
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# LABEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

LABEL_REGISTRY: Dict[str, Type[BaseLabelGenerator]] = {
    # Binary classification
    "cls_1d_updown": Cls1dUpDown,
    "cls_5d_updown": Cls5dUpDown,
    "cls_20d_updown": Cls20dUpDown,
    # Ternary classification
    "cls_1d_3class": Cls1d3Class,
    "cls_5d_3class": Cls5d3Class,
    # Regression
    "reg_next_close_return": RegNextCloseReturn,
    "reg_5d_return": Reg5dReturn,
    # Volatility
    "future_realized_vol_5d": FutureRealizedVol5d,
}

# Canonical ordered list for documentation / iteration
LABEL_NAMES: List[str] = list(LABEL_REGISTRY.keys())


def get_generator(
    name: str,
    use_settings: bool = True,
    **kwargs,
) -> BaseLabelGenerator:
    """Instantiate a label generator by its registry name.

    When ``use_settings=True`` (default), threshold parameters are
    sourced from ``config.settings.Settings`` if available:

        - ``cls_1d_3class`` -> ``Settings.label_cls_1d_threshold``
        - ``cls_5d_3class`` -> ``Settings.label_cls_5d_threshold``

    Any explicit ``**kwargs`` override settings values.

    Args:
        name: Registry key (e.g. ``"cls_1d_updown"``).
        use_settings: Pull configurable thresholds from settings.
        **kwargs: Forwarded to the generator constructor.

    Returns:
        An initialised ``BaseLabelGenerator`` sub-class instance.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in LABEL_REGISTRY:
        raise KeyError(
            f"Unknown label '{name}'. Available: {list(LABEL_REGISTRY.keys())}"
        )

    cls = LABEL_REGISTRY[name]

    # ── Inject settings-driven thresholds ────────────────────
    if use_settings:
        settings = get_settings()
        if name == "cls_1d_3class" and "threshold" not in kwargs:
            kwargs["threshold"] = settings.label_cls_1d_threshold
        elif name == "cls_5d_3class" and "threshold" not in kwargs:
            kwargs["threshold"] = settings.label_cls_5d_threshold

    return cls(**kwargs)


def apply_all_labels(
    df: pd.DataFrame,
    names: Optional[List[str]] = None,
    use_settings: bool = True,
) -> pd.DataFrame:
    """Apply multiple label generators to a DataFrame.

    Args:
        df: OHLCV DataFrame sorted ascending by date.
        names: List of registry keys to apply.  Defaults to *all*.
        use_settings: Pull configurable thresholds from settings.

    Returns:
        A copy of ``df`` with all requested label columns appended.
    """
    if names is None:
        names = LABEL_NAMES

    result = df.copy()
    applied: List[str] = []

    for name in names:
        gen = get_generator(name, use_settings=use_settings)
        result = gen.generate(result)
        applied.extend(gen.label_columns)

    logger.info(
        "all_labels_applied",
        count=len(applied),
        columns=applied,
        rows=len(result),
    )
    return result


__all__ = [
    "BaseLabelGenerator",
    "Cls1dUpDown",
    "Cls5dUpDown",
    "Cls20dUpDown",
    "Cls1d3Class",
    "Cls5d3Class",
    "RegNextCloseReturn",
    "Reg5dReturn",
    "FutureRealizedVol5d",
    "LABEL_REGISTRY",
    "LABEL_NAMES",
    "get_generator",
    "apply_all_labels",
]
