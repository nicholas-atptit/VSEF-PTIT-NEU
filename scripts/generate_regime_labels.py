"""Generate Phase 4 ticker-level regime labels from local vnstock-derived OHLCV."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - incomplete runtime only
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.regime import REGIME_OUTPUT_COLUMNS, RegimeDetector  # noqa: E402


SUMMARY_COLUMNS = [
    "ticker",
    "trend_regime",
    "volatility_regime",
    "combined_regime",
    "observation_count",
    "start_date",
    "end_date",
    "mean_return",
    "mean_realized_volatility",
]

FAILURE_COLUMNS = ["ticker", "source_path", "reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 4 rule-based regime labels.")
    parser.add_argument("--policy", required=True, help="Path to regime policy YAML")
    parser.add_argument("--output", required=True, help="Report CSV path for regime labels")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/EXP-RG-000.yaml",
        help="Optional EXP-RG-000 config with universe and date bounds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_path = resolve_repo_path(args.policy)
    output_path = resolve_repo_path(args.output)
    experiment_config_path = resolve_repo_path(args.experiment_config)

    policy = read_yaml(policy_path)
    experiment_config = read_yaml(experiment_config_path) if experiment_config_path.exists() else {}
    validate_policy(policy)

    data_cfg = experiment_config.get("data", {}) or {}
    universe = [str(ticker).upper().strip() for ticker in data_cfg.get("universe", []) if str(ticker).strip()]
    if not universe:
        universe = ["FPT", "ACB", "HPG", "MWG", "DGC"]
    start_date = str(data_cfg.get("start_date") or "2023-01-01")
    end_date = str(data_cfg.get("end_date") or "2024-12-31")
    csv_dir = resolve_repo_path(str(data_cfg.get("source_cache_dir") or "data/daily_market_split_data"))

    experiment_id = str(experiment_config.get("experiment", {}).get("id") or "EXP-RG-000")
    output_root = resolve_repo_path(str(experiment_config.get("outputs", {}).get("root_dir") or "outputs/experiments"))
    raw_dir = output_root / experiment_id
    prepare_raw_dir(raw_dir)
    run_id = f"{experiment_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    detector = RegimeDetector(policy)
    label_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for ticker in universe:
        source_path = csv_dir / f"{ticker}.csv"
        try:
            ohlcv = load_local_ohlcv(source_path, ticker=ticker, start_date=start_date, end_date=end_date)
            labels = detector.tag(ohlcv)
            label_frames.append(labels)
        except Exception as exc:
            failures.append({"ticker": ticker, "source_path": str(source_path), "reason": str(exc)})

    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame(columns=REGIME_OUTPUT_COLUMNS)
    summary = detector.summarize(labels) if not labels.empty else pd.DataFrame(columns=SUMMARY_COLUMNS)
    failures_frame = pd.DataFrame(failures, columns=FAILURE_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.parent / "regime_summary.csv"
    labels.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    labels.to_csv(raw_dir / "artifacts" / "regime_labels.csv", index=False)
    summary.to_csv(raw_dir / "artifacts" / "regime_summary.csv", index=False)
    failures_frame.to_csv(raw_dir / "artifacts" / "tagging_failures.csv", index=False)
    failures_frame.to_csv(output_path.parent / "regime_label_failures.csv", index=False)

    if policy_path.exists():
        shutil.copyfile(policy_path, raw_dir / "config" / "regime_policy.yaml")
    if experiment_config_path.exists():
        shutil.copyfile(experiment_config_path, raw_dir / "config" / "original_config.yaml")

    manifest = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "status": "completed_with_errors" if failures else "completed",
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": str(policy_path),
        "output": str(output_path),
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "label_rows": int(len(labels)),
        "summary_rows": int(len(summary)),
        "failure_count": int(len(failures)),
        "failures": failures,
        "diagnostic_only": True,
    }
    write_json(raw_dir / "manifests" / "run_manifest.json", manifest)
    write_text(raw_dir / "logs" / "run.log", json.dumps(manifest, indent=2, default=str))
    write_text(raw_dir / "logs" / "errors.log", "\n".join(item["reason"] for item in failures))
    write_text(raw_dir / "reports" / "summary.md", render_summary(manifest, summary, failures_frame))

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "label_rows": manifest["label_rows"],
                "summary_rows": manifest["summary_rows"],
                "failure_count": manifest["failure_count"],
                "output": str(output_path),
                "summary": str(summary_path),
                "raw_dir": str(raw_dir),
            },
            indent=2,
        )
    )
    return 0 if labels is not None else 1


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Phase 4 YAML configs.")
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def validate_policy(policy: dict[str, Any]) -> None:
    data_cfg = policy.get("data", {}) or {}
    if str(data_cfg.get("provider")) != "vnstock_data":
        raise ValueError("Regime policy data.provider must be vnstock_data")
    if str(data_cfg.get("frequency")).lower() != "daily":
        raise ValueError("Regime policy data.frequency must be daily")
    if not bool(policy.get("governance", {}).get("no_future_data", True)):
        raise ValueError("Regime policy must preserve no_future_data governance")
    if not bool(policy.get("governance", {}).get("diagnostic_only", True)):
        raise ValueError("Regime policy must be diagnostic_only")


def prepare_raw_dir(raw_dir: Path) -> None:
    for relative in ("artifacts", "config", "logs", "manifests", "reports", "charts", "metrics", "predictions"):
        (raw_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("artifacts/.gitkeep", "charts/.gitkeep", "metrics/.gitkeep", "predictions/.gitkeep"):
        (raw_dir / relative).write_text("", encoding="utf-8")


def load_local_ohlcv(source_path: Path, *, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"local vnstock-derived OHLCV cache not found: {source_path}")
    frame = pd.read_csv(source_path)
    if "date" not in frame.columns and "time" in frame.columns:
        frame = frame.rename(columns={"time": "date"})
    if "ticker" not in frame.columns:
        frame["ticker"] = ticker

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OHLCV schema missing columns {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "close"]).copy()
    frame = frame[(frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))].copy()
    frame = frame[frame["ticker"].eq(ticker)].copy()
    if frame.empty:
        raise ValueError(f"no OHLCV rows available for {ticker} in {start_date}..{end_date}")
    return frame[["date", "ticker", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_summary(manifest: dict[str, Any], summary: pd.DataFrame, failures: pd.DataFrame) -> str:
    lines = [
        f"# {manifest['experiment_id']} Summary",
        "",
        f"- Status: {manifest['status']}",
        f"- Run ID: {manifest['run_id']}",
        f"- Label rows: {manifest['label_rows']}",
        f"- Summary rows: {manifest['summary_rows']}",
        f"- Failure count: {manifest['failure_count']}",
        "",
        "All outputs are diagnostic research artifacts only.",
        "",
        "## Regime Counts",
        "",
        markdown_table(summary.head(20)),
        "",
        "## Failures",
        "",
        markdown_table(failures),
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    clean = frame.copy().where(pd.notna(frame), "")
    headers = [str(column) for column in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in clean.columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
