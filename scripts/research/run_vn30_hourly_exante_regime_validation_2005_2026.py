"""Validate frozen VN30 hourly regime diagnostics with ex-ante proxy labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    REPORT_ROOT,
    load_hourly_predictions,
    markdown_table,
    rel,
    save_placeholder_figure,
)


DEFAULT_OUTPUT_DIR = REPORT_ROOT / "regime"
REGIME_WINDOW = 20
MIN_PRIOR_OBS = 5
BEAR_THRESHOLD = -0.03
BULL_THRESHOLD = 0.03
SUMMARY_COLUMNS = [
    "regime_source",
    "frequency",
    "model",
    "horizon",
    "regime",
    "ticker",
    "observation_count",
    "accuracy",
    "reliability",
    "passed_60pct",
    "passed_63pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly ex-ante regime validation.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def add_exante_proxy_regimes(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    working = data.sort_values(["model", "horizon", "ticker", "timestamp_sort"]).copy()
    group_cols = ["model", "horizon", "ticker"]
    working["prior_actual_return"] = working.groupby(group_cols)["actual_return"].shift(1)
    working["exante_trailing_return"] = (
        working.groupby(group_cols)["prior_actual_return"]
        .rolling(REGIME_WINDOW, min_periods=MIN_PRIOR_OBS)
        .mean()
        .reset_index(level=group_cols, drop=True)
    )
    working["exante_regime"] = "insufficient_history"
    mask = working["exante_trailing_return"].notna()
    working.loc[mask & (working["exante_trailing_return"] <= BEAR_THRESHOLD), "exante_regime"] = "bear"
    working.loc[mask & (working["exante_trailing_return"] >= BULL_THRESHOLD), "exante_regime"] = "bull"
    working.loc[
        mask
        & (working["exante_trailing_return"] > BEAR_THRESHOLD)
        & (working["exante_trailing_return"] < BULL_THRESHOLD),
        "exante_regime",
    ] = "sideways"
    return working


def summarize(data: pd.DataFrame, *, regime_column: str, regime_source: str) -> list[dict[str, Any]]:
    if data.empty or regime_column not in data.columns:
        return []
    required = {"frequency", "model", "horizon", regime_column, "ticker", "is_correct"}
    if required.difference(data.columns):
        return []
    rows: list[dict[str, Any]] = []
    grouped = data.dropna(subset=["model", "horizon", "ticker", "is_correct"]).groupby(
        ["frequency", "model", "horizon", regime_column, "ticker"],
        sort=True,
    )
    for (frequency, model, horizon, regime, ticker), group in grouped:
        n_obs = int(len(group))
        accuracy = float(pd.to_numeric(group["is_correct"], errors="coerce").mean()) if n_obs else None
        reliable = n_obs >= 50
        rows.append(
            {
                "regime_source": regime_source,
                "frequency": frequency,
                "model": model,
                "horizon": int(horizon),
                "regime": regime,
                "ticker": ticker,
                "observation_count": n_obs,
                "accuracy": accuracy,
                "reliability": bool(reliable),
                "passed_60pct": bool(accuracy is not None and accuracy >= 0.60),
                "passed_63pct": bool(accuracy is not None and accuracy >= 0.63),
            }
        )
    return rows


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    working = data.copy()
    working = working[working["frequency"].astype(str).str.lower().eq("hourly")]
    working = working.dropna(subset=["timestamp_sort", "actual_return", "is_correct"])
    if working.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    with_exante = add_exante_proxy_regimes(working)
    rows = summarize(with_exante, regime_column="exante_regime", regime_source="exante_proxy")
    if "regime" in working.columns:
        rows.extend(summarize(working, regime_column="regime", regime_source="posthoc_artifact"))
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["regime_source", "model", "horizon", "regime", "ticker"],
    )


def aggregate_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    rows: list[dict[str, Any]] = []
    working = summary.copy()
    working["observation_count"] = pd.to_numeric(working["observation_count"], errors="coerce").fillna(0).astype(int)
    working["accuracy"] = pd.to_numeric(working["accuracy"], errors="coerce")
    for (regime_source, model, horizon, regime), group in working.groupby(
        ["regime_source", "model", "horizon", "regime"],
        sort=True,
    ):
        n_obs = int(group["observation_count"].sum())
        accuracy = (
            float((group["accuracy"] * group["observation_count"]).sum() / n_obs)
            if n_obs and group["accuracy"].notna().any()
            else None
        )
        rows.append(
            {
                "regime_source": regime_source,
                "model": model,
                "horizon": int(horizon),
                "regime": regime,
                "observation_count": n_obs,
                "accuracy": accuracy,
                "passed_60pct": bool(accuracy is not None and accuracy >= 0.60),
                "passed_63pct": bool(accuracy is not None and accuracy >= 0.63),
            }
        )
    return sorted(rows, key=lambda row: (str(row["regime_source"]), str(row["model"]), int(row["horizon"]), str(row["regime"])))


def write_report(path: Path, artifact_dir: Path, summary: pd.DataFrame) -> None:
    aggregate = aggregate_rows(summary)
    content = [
        "# VN30 Hourly Ex-Ante Regime Validation Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Ex-ante proxy rule: regime labels use shifted prior actual returns only.",
        f"- Rolling window: {REGIME_WINDOW}; minimum prior observations: {MIN_PRIOR_OBS}.",
        "",
        "## Aggregate Regime Diagnostics",
        "",
        markdown_table(
            [
                "regime_source",
                "model",
                "horizon",
                "regime",
                "observation_count",
                "accuracy",
                "passed_60pct",
                "passed_63pct",
            ],
            aggregate,
            max_rows=40,
        )
        if aggregate
        else "No regime diagnostics are available.",
        "",
        "## Boundary",
        "",
    ]
    if summary.empty:
        content.append("No ex-ante regime claims are available because official hourly VN30 predictions are missing or empty.")
    else:
        pass_63 = int((summary["passed_63pct"].astype(str).str.lower() == "true").sum())
        content.append(f"- Per-ticker regime rows: {len(summary)}.")
        content.append(f"- Per-ticker rows passing 63%: {pass_63}.")
        content.append("- Ex-ante proxy labels avoid current/future return leakage but remain diagnostic approximations.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, summary: pd.DataFrame) -> str:
    if summary.empty:
        return save_placeholder_figure(
            path,
            "VN30 hourly regime accuracy",
            "No official hourly predictions were available because the full VN30 coverage gate failed.",
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        aggregate = pd.DataFrame(aggregate_rows(summary))
        aggregate = aggregate[aggregate["regime_source"].astype(str).eq("exante_proxy")].copy()
        if aggregate.empty:
            return save_placeholder_figure(path, "VN30 hourly regime accuracy", "No ex-ante aggregate rows were available.")
        aggregate = aggregate.sort_values(["accuracy", "observation_count"], ascending=[False, False]).head(12)
        labels = [
            f"{row.model} h={int(row.horizon)} {row.regime}"
            for row in aggregate.itertuples(index=False)
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, aggregate["accuracy"], color="#356f8c")
        ax.axvline(0.60, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.63, color="#7a3b2e", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("VN30 hourly ex-ante regime accuracy")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(path, "VN30 hourly regime accuracy", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    data = load_hourly_predictions(args.artifact_dir)
    summary = build_summary(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_hourly_exante_regime_accuracy_summary.csv"
    report_path = args.output_dir / "vn30_hourly_exante_regime_validation_report.md"
    figure_path = args.output_dir / "vn30_hourly_regime_accuracy.png"
    summary.to_csv(summary_path, index=False)
    write_report(report_path, args.artifact_dir, summary)
    figure_status = write_plot(figure_path, summary)
    print(
        "VN30 hourly ex-ante regime validation complete: "
        f"rows={len(summary)} report={rel(report_path)} figure={rel(figure_path)} status={figure_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
