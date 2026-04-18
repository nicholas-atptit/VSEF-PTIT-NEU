"""Markov-switching regime model with a deterministic threshold fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.regime.base import RegimeModel
from src.regime.labels import map_state_probabilities, normalize_regime_probabilities, ordered_states_from_means


class MarkovSwitchingRegimeModel(RegimeModel):
    """Infer bull, bear, and sideway states from realized returns per ticker."""

    model_name = "markov_switching"

    def __init__(
        self,
        *,
        return_column: str = "daily_return",
        price_column: str = "close",
        k_regimes: int = 3,
        trend: str = "c",
        switching_variance: bool = True,
        min_train_observations: int = 80,
        lookback: int = 20,
        bull_threshold: float = 0.03,
        bear_threshold: float = -0.03,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(return_column=return_column, price_column=price_column)
        self.k_regimes = int(k_regimes)
        self.trend = trend
        self.switching_variance = bool(switching_variance)
        self.min_train_observations = int(min_train_observations)
        self.lookback = int(lookback)
        self.bull_threshold = float(bull_threshold)
        self.bear_threshold = float(bear_threshold)
        self.fit_kwargs = dict(fit_kwargs or {"disp": False})
        self._fit_params: pd.Series | None = None
        self._state_order: list[int] | None = None
        self._fit_index: pd.Index | None = None
        self._fitted_returns = pd.Series(dtype=float)
        self._fallback_reason: str | None = None

    @staticmethod
    def _markov_regression_class() -> Any:
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MarkovSwitchingRegimeModel requires statsmodels. Install statsmodels in the active interpreter."
            ) from exc
        return MarkovRegression

    def _extract_returns(self, history_frame: pd.DataFrame) -> pd.Series:
        returns = self._coerce_numeric_series(history_frame[self.return_column])
        return returns.dropna().astype(float)

    def _threshold_probabilities(self, history_frame: pd.DataFrame) -> pd.DataFrame:
        self.source_model_ = f"{self.model_name}_threshold_fallback"
        returns = self._coerce_numeric_series(history_frame[self.return_column]).fillna(0.0)
        compounded = (1.0 + returns).rolling(window=self.lookback, min_periods=1).apply(np.prod, raw=True) - 1.0
        labels = pd.Series("sideway", index=history_frame.index, dtype="object")
        labels.loc[compounded > self.bull_threshold] = "bull"
        labels.loc[compounded < self.bear_threshold] = "bear"

        probabilities = pd.DataFrame(
            {
                "regime_prob_bull": (labels == "bull").astype(float),
                "regime_prob_bear": (labels == "bear").astype(float),
                "regime_prob_sideway": (labels == "sideway").astype(float),
            },
            index=history_frame.index,
        )
        probabilities["regime_label"] = labels
        return normalize_regime_probabilities(probabilities)

    def _state_occupancy_ok(self, state_probabilities: pd.DataFrame) -> bool:
        occupancy = state_probabilities.sum(axis=0)
        return bool((occupancy >= 5.0).all())

    def _fit_model(self, history_frame: pd.DataFrame) -> None:
        returns = self._extract_returns(history_frame)
        self._fit_index = returns.index
        self._fitted_returns = returns.copy()
        self._fit_params = None
        self._state_order = None
        self._fallback_reason = None

        if self.k_regimes != 3:
            raise ValueError("Phase 2 regime output expects exactly three latent states")
        if len(returns) < self.min_train_observations:
            self._fallback_reason = "insufficient_train_observations"
            return

        markov_cls = self._markov_regression_class()
        try:
            model = markov_cls(
                returns.reset_index(drop=True),
                k_regimes=self.k_regimes,
                trend=self.trend,
                switching_variance=self.switching_variance,
            )
            results = model.fit(**self.fit_kwargs)
            filtered = results.filtered_marginal_probabilities
            if not self._state_occupancy_ok(filtered):
                self._fallback_reason = "collapsed_markov_state"
                return
            self._fit_params = results.params.copy()
            self._state_order = ordered_states_from_means(returns.reset_index(drop=True), filtered)
        except Exception as exc:
            self._fallback_reason = f"markov_fit_failed:{exc.__class__.__name__}"

    def _predict_probabilities(self, history_frame: pd.DataFrame) -> pd.DataFrame:
        if self._fallback_reason or self._fit_params is None or self._state_order is None:
            return self._threshold_probabilities(history_frame)

        full_returns = self._coerce_numeric_series(history_frame[self.return_column])
        observed_index = full_returns.dropna().index
        full_returns = full_returns.dropna().reset_index(drop=True)
        if full_returns.empty:
            return self._threshold_probabilities(history_frame)

        markov_cls = self._markov_regression_class()
        try:
            filtered = markov_cls(
                full_returns,
                k_regimes=self.k_regimes,
                trend=self.trend,
                switching_variance=self.switching_variance,
            ).filter(self._fit_params)
            self.source_model_ = self.model_name
            mapped = map_state_probabilities(filtered.filtered_marginal_probabilities, self._state_order)
            aligned = pd.DataFrame(
                {
                    "regime_prob_bull": 0.0,
                    "regime_prob_bear": 0.0,
                    "regime_prob_sideway": 1.0,
                },
                index=history_frame.index,
            )
            aligned.loc[observed_index, ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]] = mapped[
                ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideway"]
            ].to_numpy()
            aligned = aligned.ffill()
            aligned["regime_label"] = mapped["regime_label"].reindex(
                observed_index
            ).reindex(history_frame.index).ffill().fillna("sideway")
            return normalize_regime_probabilities(aligned)
        except Exception:
            return self._threshold_probabilities(history_frame)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "k_regimes": self.k_regimes,
                "trend": self.trend,
                "switching_variance": bool(self.switching_variance),
                "min_train_observations": self.min_train_observations,
                "lookback": self.lookback,
                "bull_threshold": self.bull_threshold,
                "bear_threshold": self.bear_threshold,
                "fit_kwargs": dict(self.fit_kwargs),
                "state_order": list(self._state_order or []),
                "fallback_reason": self._fallback_reason,
            }
        )
        return metadata
