"""Consolidated final reporting for benchmark, stress, and tuning outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_full_system_report(
    *,
    benchmark_summary: pd.DataFrame | None = None,
    stress_summary: pd.DataFrame | None = None,
    tuning_result: dict[str, Any] | None = None,
    output_path: str | Path = "reports/full_system_report.md",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tuning_lines = ["_No tuning run available._"]
    if tuning_result:
        tuning_lines = [
            f"Best validation score: `{tuning_result.get('best_score', 0.0):.6f}`",
            "",
            "Best parameters:",
            *[f"- `{k}`: `{v}`" for k, v in tuning_result.get("best_params", {}).items()],
        ]

    content = "\n".join(
        [
            "# Full System Report",
            "",
            "## 1. System Architecture Summary",
            "- Forecasting layer: existing CART/XGBoost/LightGBM/SARIMAX/ETS/LSTM/BiLSTM models through the manifest-driven trainer.",
            "- Risk layer: rolling VaR/CVaR/CoVaR/Delta-CoVaR/drawdown features and system-level summaries.",
            "- Regime layer: feature-based NORMAL/HIGH_VOL/CRISIS classification.",
            "- Allocation layer: optional risk-aware exposure scaling using risk and regime state.",
            "",
            "## 2. Benchmark Results",
            _markdown_table(benchmark_summary.round(6) if benchmark_summary is not None and not benchmark_summary.empty else pd.DataFrame()),
            "",
            "## 3. Stress Test Results",
            _markdown_table(stress_summary.round(6) if stress_summary is not None and not stress_summary.empty else pd.DataFrame()),
            "",
            "## 4. Risk Tuning Improvements",
            *tuning_lines,
            "",
            "## 5. Trade-offs",
            "- Performance vs stability: richer risk/regime overlays can reduce exposure and headline return while improving drawdown behavior.",
            "- Complexity vs benefit: benchmark/stress/tuning orchestration adds operational complexity but makes model comparisons and deployment decisions auditable.",
            "",
            "## 6. Limitations",
            "- Stress testing re-evaluates held-out predictions under shocked returns/costs instead of retraining on synthetic crisis histories.",
            "- Regime logic remains rule-based Option A; no latent-state model is introduced.",
            "- Allocation remains a modular overlay, not a mandatory execution engine.",
            "",
            "## 7. Recommendation",
            "- Use the full system in production only with benchmark plus stress plus tuning outputs reviewed together.",
            "- Prefer the tuned full-system configuration when it improves Sharpe/Sortino without materially worsening max drawdown or turnover.",
        ]
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path
