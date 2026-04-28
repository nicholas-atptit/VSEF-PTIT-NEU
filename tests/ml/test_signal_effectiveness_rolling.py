from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.ml.backtest.signal_effectiveness import (
    EVALUATION_MODE_ROLLING_HELDOUT_THRESHOLD_SELECTION,
    POLICY_RETURN_THRESHOLD,
    SUCCESS_RAW_POSITIVE,
    SignalEffectivenessConfig,
    SignalEffectivenessRunner,
    parse_rolling_splits,
)


INLINE_SPLITS = (
    "2024-01-01:2024-01-31:2024-02-01:2024-02-29,"
    "2024-03-01:2024-03-31:2024-04-01:2024-04-30,"
    "2024-05-01:2024-05-31:2024-06-01:2024-06-30"
)


def _row(date: str, ticker: str, predicted: float, realized: float, regime: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "prediction_date": date,
        "target_date": str((pd.Timestamp(date) + pd.Timedelta(days=5)).date()),
        "ticker": ticker,
        "horizon": "short_10d",
        "model_name": "stacking_final",
        "predicted_return": predicted,
        "predicted_direction": int(predicted > 0.0),
        "actual_return": realized,
    }
    if regime is not None:
        row["regime"] = regime
    return row


def _fold_rows(
    *,
    selection_month: str,
    test_month: str,
    high_selection_success: bool,
    test_realized: list[float],
    regime: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, realized in enumerate([0.01, 0.01, -0.01, -0.01], start=1):
        rows.append(
            _row(
                f"2024-{selection_month}-{idx:02d}",
                f"L{selection_month}{idx}",
                0.006,
                realized if high_selection_success else 0.02,
                "sideway" if regime else None,
            )
        )
    high_selection_realized = [0.03, 0.02] if high_selection_success else [-0.02, -0.01]
    for idx, realized in enumerate(high_selection_realized, start=1):
        rows.append(
            _row(
                f"2024-{selection_month}-{10 + idx:02d}",
                f"H{selection_month}{idx}",
                0.035,
                realized,
                "bull" if regime else None,
            )
        )
    for idx, realized in enumerate(test_realized, start=1):
        rows.append(
            _row(
                f"2024-{test_month}-{idx:02d}",
                f"T{test_month}{idx}",
                0.035,
                realized,
                "bull" if regime else None,
            )
        )
    rows.append(
        _row(
            f"2024-{test_month}-20",
            f"U{test_month}",
            0.006,
            -0.01,
            "bear" if regime else None,
        )
    )
    return rows


def _rolling_frame(*, regime: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _fold_rows(
            selection_month="01",
            test_month="02",
            high_selection_success=True,
            test_realized=[0.02, 0.01, -0.01],
            regime=regime,
        )
    )
    rows.extend(
        _fold_rows(
            selection_month="03",
            test_month="04",
            high_selection_success=False,
            test_realized=[0.02, 0.02, 0.01],
            regime=regime,
        )
    )
    rows.extend(
        _fold_rows(
            selection_month="05",
            test_month="06",
            high_selection_success=True,
            test_realized=[0.03, 0.02],
            regime=regime,
        )
    )
    return pd.DataFrame(rows)


def _rolling_config(
    tmp_path: Path,
    *,
    minimum_signal_counts: list[int] | None = None,
    enable_regime_diagnostics: bool = False,
) -> SignalEffectivenessConfig:
    return SignalEffectivenessConfig(
        output_dir=str(tmp_path / "rolling"),
        evaluation_mode=EVALUATION_MODE_ROLLING_HELDOUT_THRESHOLD_SELECTION,
        rolling_splits=parse_rolling_splits(INLINE_SPLITS),
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.005, 0.03],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=minimum_signal_counts or [2],
        precision_targets=[0.60, 0.65, 0.70],
        enable_regime_diagnostics=enable_regime_diagnostics,
    )


def test_rolling_split_parser_handles_valid_inline_splits() -> None:
    splits = parse_rolling_splits(INLINE_SPLITS)

    assert [split.fold_id for split in splits] == ["fold_1", "fold_2", "fold_3"]
    assert splits[0].selection_start == "2024-01-01"
    assert splits[2].test_end == "2024-06-30"


def test_rolling_split_parser_rejects_invalid_splits() -> None:
    with pytest.raises(ValueError, match="Invalid --rolling-splits item"):
        parse_rolling_splits("2024-01-01:2024-01-31:2024-02-01")


def test_rolling_mode_selects_thresholds_using_only_each_selection_period(tmp_path) -> None:
    base = _rolling_frame()
    modified = base.copy()
    test_mask = pd.to_datetime(modified["prediction_date"]).dt.month.isin([2, 4, 6])
    modified.loc[test_mask, "actual_return"] = -0.99

    first = SignalEffectivenessRunner(_rolling_config(tmp_path / "first")).run(base)
    second = SignalEffectivenessRunner(_rolling_config(tmp_path / "second")).run(modified)

    first_thresholds = first["rolling_selected_thresholds"].sort_values("fold_id")[
        "selected_predicted_return_threshold"
    ].tolist()
    second_thresholds = second["rolling_selected_thresholds"].sort_values("fold_id")[
        "selected_predicted_return_threshold"
    ].tolist()
    assert first_thresholds == [0.03, 0.005, 0.03]
    assert first_thresholds == second_thresholds


def test_rolling_heldout_metrics_use_only_each_test_period(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path)).run(_rolling_frame())
    heldout_rows = result["rolling_heldout_signal_rows"]

    for fold_id, min_date, max_date in [
        ("fold_1", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-29")),
        ("fold_2", pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-30")),
        ("fold_3", pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30")),
    ]:
        dates = pd.to_datetime(heldout_rows.loc[heldout_rows["fold_id"] == fold_id, "prediction_date"])
        assert dates.min() >= min_date
        assert dates.max() <= max_date


def test_multiple_folds_produce_expected_fold_ids(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path)).run(_rolling_frame())

    assert result["rolling_selected_thresholds"]["fold_id"].tolist() == ["fold_1", "fold_2", "fold_3"]


def test_threshold_stability_summary_computes_pass_rates(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path)).run(_rolling_frame())
    stability = result["threshold_stability_summary"].iloc[0]

    assert stability["fold_count"] == 3
    assert stability["most_common_threshold"] == pytest.approx(0.03)
    assert stability["threshold_stability_level"] == "medium"
    assert stability["total_heldout_buy_count"] == 9
    assert stability["pass_rate_70"] == pytest.approx(2 / 3)


def test_rolling_precision_target_70_pass_fail_is_computed_per_fold(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path)).run(_rolling_frame())
    pass_70 = result["rolling_precision_target_pass_fail"]
    pass_70 = pass_70[pass_70["precision_target"] == 0.70].sort_values("fold_id")

    assert pass_70["pass_fail"].tolist() == [False, True, True]


def test_no_regime_column_records_metadata_skip_without_crashing(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path, enable_regime_diagnostics=True)).run(_rolling_frame())
    metadata = result["run_metadata"]["regime_diagnostics"]

    assert metadata["requested"] is True
    assert metadata["enabled"] is False
    assert "No recognized regime column" in metadata["skipped_reason"]
    assert "regime_buy_precision_summary" not in result["paths"]


def test_regime_column_produces_regime_summaries(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path, enable_regime_diagnostics=True)).run(
        _rolling_frame(regime=True)
    )

    assert result["run_metadata"]["regime_diagnostics"]["enabled"] is True
    assert not result["regime_buy_precision_summary"].empty
    assert not result["regime_precision_stability_summary"].empty
    assert (tmp_path / "rolling" / "regime_buy_precision_summary.csv").exists()
    assert (tmp_path / "rolling" / "regime_precision_stability_summary.csv").exists()


def test_empty_fold_selection_candidates_are_handled_safely(tmp_path) -> None:
    result = SignalEffectivenessRunner(_rolling_config(tmp_path, minimum_signal_counts=[100])).run(_rolling_frame())

    assert result["rolling_selected_thresholds"].empty
    assert result["rolling_heldout_buy_precision"].empty
    assert result["threshold_stability_summary"].empty
    assert not result["rolling_threshold_selection_trace"].empty
    assert result["rolling_threshold_selection_trace"]["selected"].sum() == 0


def test_empty_heldout_buy_cases_are_handled_safely(tmp_path) -> None:
    frame = _rolling_frame()
    test_mask = pd.to_datetime(frame["prediction_date"]).dt.month.isin([2, 4, 6])
    frame.loc[test_mask, "predicted_return"] = 0.001
    frame.loc[test_mask, "predicted_direction"] = 1

    result = SignalEffectivenessRunner(_rolling_config(tmp_path)).run(frame)
    heldout = result["rolling_heldout_buy_precision"]
    pass_fail = result["rolling_precision_target_pass_fail"]

    assert heldout["heldout_buy_count"].tolist() == [0, 0, 0]
    assert heldout["heldout_buy_precision"].isna().all()
    assert pass_fail["pass_fail"].tolist() == [False] * 9


def test_cli_rolling_mode_runs_on_synthetic_csv(tmp_path) -> None:
    input_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / "rolling_cli"
    _rolling_frame().to_csv(input_path, index=False)
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_signal_effectiveness_backtest.py",
            "--predictions-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--evaluation-mode",
            EVALUATION_MODE_ROLLING_HELDOUT_THRESHOLD_SELECTION,
            "--rolling-splits",
            INLINE_SPLITS,
            "--policy",
            POLICY_RETURN_THRESHOLD,
            "--threshold-grid",
            "0.005,0.03",
            "--cost-per-trade",
            "0.0",
            "--slippage",
            "0.0",
            "--success-definition",
            SUCCESS_RAW_POSITIVE,
            "--minimum-signal-count",
            "2",
            "--precision-targets",
            "0.60,0.65,0.70",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "rolling_selected_thresholds.csv").exists()
    assert (output_dir / "threshold_stability_summary.csv").exists()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation_mode"] == EVALUATION_MODE_ROLLING_HELDOUT_THRESHOLD_SELECTION
