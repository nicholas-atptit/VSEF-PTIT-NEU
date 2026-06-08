import pytest

from src.evaluation.metrics import direction_metrics, interval_metrics, ranking_metrics, return_price_metrics


def test_direction_metrics_include_repaired_metrics_and_probability_metrics():
    metrics = direction_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert metrics["raw_accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["prediction_balance"] == pytest.approx(0.75)
    assert metrics["auc"] == pytest.approx(1.0)


def test_return_price_metrics_include_rank_and_sign_accuracy():
    metrics = return_price_metrics([-2.0, -1.0, 1.0, 2.0], [-1.0, -0.5, 0.5, 3.0])
    assert metrics["sign_accuracy"] == pytest.approx(1.0)
    assert metrics["rank_ic"] == pytest.approx(1.0)


def test_interval_metrics_include_coverage_breaches_and_winkler():
    metrics = interval_metrics([10.0, 12.0, 15.0], [9.0, 11.0, 12.0], [11.0, 13.0, 14.0])
    assert metrics["interval_coverage"] == pytest.approx(2 / 3)
    assert metrics["high_breach_rate"] == pytest.approx(1 / 3)
    assert metrics["winkler_score"] > metrics["average_interval_width"]


def test_ranking_metrics_reward_correct_order():
    metrics = ranking_metrics([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert metrics["spearman_ic"] == pytest.approx(1.0)
    assert metrics["ndcg_at_5"] == pytest.approx(1.0)
    assert metrics["top20_precision"] == pytest.approx(1.0)
