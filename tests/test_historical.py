"""Tests for Module 3: Historical Time-Series DB."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from src.historical.price_adjuster import PriceAdjuster
from src.utils.time_utils import VN_TZ


class TestPriceAdjuster:
    """Test corporate action price adjustment calculations."""

    def test_dividend_factor(self):
        """Cash dividend factor should reduce price."""
        factor = PriceAdjuster.compute_dividend_factor(
            price_before_ex=Decimal("30000"),
            dividend_per_share=Decimal("1500"),
        )
        expected = (Decimal("30000") - Decimal("1500")) / Decimal("30000")
        assert factor == expected
        assert factor < 1  # Price should decrease

    def test_dividend_factor_zero_price(self):
        """Zero price should return factor 1.0."""
        factor = PriceAdjuster.compute_dividend_factor(
            price_before_ex=Decimal("0"),
            dividend_per_share=Decimal("1000"),
        )
        assert factor == Decimal("1.0")

    def test_split_factor(self):
        """Stock split 2:1 should halve the factor."""
        factor = PriceAdjuster.compute_split_factor(Decimal("2"))
        assert factor == Decimal("0.5")

    def test_split_factor_3_to_1(self):
        """Stock split 3:1."""
        factor = PriceAdjuster.compute_split_factor(Decimal("3"))
        assert round(factor, 10) == round(Decimal("1") / Decimal("3"), 10)

    def test_bonus_factor(self):
        """50% bonus share should adjust by 2/3."""
        factor = PriceAdjuster.compute_bonus_factor(Decimal("0.5"))
        expected = Decimal("1.0") / Decimal("1.5")
        assert round(factor, 10) == round(expected, 10)

    def test_adjust_volume(self):
        """Volume should be adjusted inversely."""
        volume = PriceAdjuster._adjust_volume(1000, Decimal("0.5"))
        assert volume == 2000

    def test_adjust_volume_no_change(self):
        """Volume should not change with factor 1.0."""
        volume = PriceAdjuster._adjust_volume(1000, Decimal("1.0"))
        assert volume == 1000


class TestFactorTimeline:
    """Test cumulative factor timeline building."""

    def test_empty_actions(self):
        """No actions should return empty timeline."""
        adjuster = PriceAdjuster()
        timeline = adjuster._build_factor_timeline([])
        assert timeline == []

    def test_single_action(self):
        """Single action should produce one-entry timeline."""
        from unittest.mock import MagicMock

        action = MagicMock()
        action.event_date = dt.date(2024, 6, 15)
        action.factor = Decimal("0.95")

        adjuster = PriceAdjuster()
        timeline = adjuster._build_factor_timeline([action])

        assert len(timeline) == 1
        assert timeline[0][0] == dt.date(2024, 6, 15)
        assert timeline[0][1] == Decimal("0.95")

    def test_get_factor_before_all_events(self):
        """Data before all events should get full cumulative factor."""
        adjuster = PriceAdjuster()
        timeline = [
            (dt.date(2024, 3, 1), Decimal("0.9")),
            (dt.date(2024, 6, 1), Decimal("0.95")),
        ]

        # Date before earliest event
        factor = adjuster._get_factor_for_date(timeline, dt.date(2024, 1, 1))
        assert factor == Decimal("0.9")

    def test_get_factor_after_all_events(self):
        """Data after all events should get factor 1.0."""
        adjuster = PriceAdjuster()
        timeline = [
            (dt.date(2024, 3, 1), Decimal("0.9")),
        ]

        factor = adjuster._get_factor_for_date(timeline, dt.date(2024, 12, 1))
        assert factor == Decimal("1.0")

    def test_get_factor_empty_timeline(self):
        """Empty timeline should always return 1.0."""
        adjuster = PriceAdjuster()
        factor = adjuster._get_factor_for_date([], dt.date(2024, 1, 1))
        assert factor == Decimal("1.0")
