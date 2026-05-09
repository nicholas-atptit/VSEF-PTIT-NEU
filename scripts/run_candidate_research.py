"""Run Phase 3 risk-aware diagnostic candidate research experiments."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - incomplete environment only
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments"
REPORT_ROOT = REPO_ROOT / "reports" / "risk_aware"
DATA_ROOT = REPO_ROOT / "data" / "daily_market_split_data"

DISCLAIMER = (
    "All Phase 3 outputs are diagnostic decision-support research artifacts only. "
    "They are not BUY / SELL / HOLD advice, capital allocation guidance, broker "
    "execution instructions, portfolio recommendations, or proof of guaranteed "
    "profitable trading."
)

CANDIDATE_COLUMNS = [
    "experiment_id",
    "policy_id",
    "candidate_type",
    "candidate_date",
    "ticker",
    "horizon",
    "rank",
    "candidate_score",
    "forecast_score",
    "risk_penalty",
    "risk_adjusted_score",
    "expected_return_proxy",
    "directional_confidence",
    "realized_volatility",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "model_name",
    "model_type",
    "source_experiment",
    "diagnostics",
    "diagnostic_only",
    "realized_return",
    "prediction_count",
    "model_consensus_count",
    "consensus_score",
    "missing_prediction_rate",
    "current_close",
    "y_true",
]

BASKET_COLUMNS = [
    "experiment_id",
    "policy_id",
    "candidate_type",
    "basket_date",
    "horizon",
    "top_n",
    "candidate_count",
    "average_realized_return",
    "median_realized_return",
    "hit_ratio",
    "return_volatility_proxy",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "worst_period_return",
    "missing_outcome_rate",
    "diagnostic_only",
    "basket_count",
]


@dataclass
class RunContext:
    config_path: Path
    config: dict[str, Any]
    experiment_id: str
    output_dir: Path
    run_id: str
    started_at: datetime
    errors: list[dict[str, Any]]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 diagnostic candidate research.")
    parser.add_argument("--config", required=True, help="Path to EXP-RK YAML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    try:
        config = read_yaml(config_path)
        experiment_id = str(config.get("experiment", {}).get("id") or "UNKNOWN-EXPERIMENT")
        output_root = resolve_repo_path(str(config.get("outputs", {}).get("root_dir") or "outputs/experiments"))
        context = RunContext(
            config_path=config_path,
            config=config,
            experiment_id=experiment_id,
            output_dir=output_root / experiment_id,
            run_id=f"{experiment_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
            started_at=datetime.now(UTC),
            errors=[],
            warnings=[],
        )
        prepare_output_dir(context)
        save_config_copies(context)
        initialize_logs(context)
        log(context, "run.log", f"{experiment_id} started with run_id={context.run_id}")
        validate_governance_config(context)

        method = str(config.get("evaluation", {}).get("method") or "")
        if method == "candidate_policy_comparison":
            result = run_candidate_policy_comparison(context)
        elif method == "realized_candidate_outcome":
            result = run_basket_evaluation(context)
        else:
            raise ValueError(f"Unsupported Phase 3 evaluation.method: {method}")

        status = "completed_with_errors" if context.errors else "completed"
        write_manifest(context, status=status, result=result)
        log(context, "run.log", f"{experiment_id} completed with status={status}")
        print(json.dumps({"status": status, "run_id": context.run_id, "output_dir": str(context.output_dir)}, indent=2))
        return 0 if status in {"completed", "completed_with_errors"} else 1
    except Exception as exc:
        try:
            if "context" in locals():
                record_error(context, "run", str(exc), traceback_text=None)
                write_manifest(context, status="failed", result={})
                write_summary(context, "failed", pd.DataFrame(), pd.DataFrame())
                log(context, "errors.log", str(exc))
        finally:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1


def run_candidate_policy_comparison(context: RunContext) -> dict[str, Any]:
    validate_common_config(context)
    policies = load_policy_configs(context)
    source_inventory = source_artifact_inventory(context)
    missing_sources = [item for item in source_inventory if not item["exists"] and item["required"]]
    for item in missing_sources:
        context.warnings.append(f"missing_source_artifact:{item['path']}")

    predictions = load_source_predictions(context)
    if predictions.empty:
        candidates = empty_candidates()
        metrics = candidate_metrics(candidates)
    else:
        prices = load_price_frames(context)
        base = build_base_candidate_evidence(context, predictions, prices)
        candidates = build_candidate_rankings(context, base, policies)
        metrics = candidate_metrics(candidates)

    risk_adjusted = candidates.loc[candidates["candidate_type"] == "risk_aware"].copy()
    risk_summary = build_risk_summary(candidates, pd.DataFrame())

    write_csv(context.output_dir / "artifacts" / "candidate_comparison.csv", candidates)
    write_csv(context.output_dir / "artifacts" / "risk_adjusted_ranking.csv", risk_adjusted)
    write_csv(context.output_dir / "artifacts" / "risk_summary.csv", risk_summary)
    write_csv(context.output_dir / "metrics" / "metrics.csv", metrics)
    write_json(context.output_dir / "metrics" / "metrics_summary.json", summarize_metrics(metrics))

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_ROOT / "candidate_comparison.csv", candidates)
    write_csv(REPORT_ROOT / "risk_adjusted_ranking.csv", risk_adjusted)
    write_csv(REPORT_ROOT / "risk_summary.csv", risk_summary)

    report_path = REPORT_ROOT / "EXP-RK-001_CANDIDATE_COMPARISON.md"
    report_text = render_candidate_report(context, candidates, metrics, source_inventory)
    report_path.write_text(report_text, encoding="utf-8")
    (context.output_dir / "reports" / "EXP-RK-001_CANDIDATE_COMPARISON.md").write_text(report_text, encoding="utf-8")
    write_summary(context, "completed_with_errors" if context.errors else "completed", candidates, metrics)
    return {
        "candidate_rows": int(len(candidates)),
        "metric_rows": int(len(metrics)),
        "source_artifacts": source_inventory,
    }


def run_basket_evaluation(context: RunContext) -> dict[str, Any]:
    validate_common_config(context)
    candidates = load_candidate_comparison_for_basket(context)
    if candidates.empty:
        context.warnings.append("candidate_comparison_unavailable")
        basket_metrics = empty_baskets()
        period_returns = pd.DataFrame()
    else:
        basket_metrics, period_returns = build_basket_metrics(context, candidates)

    drawdown = build_drawdown_comparison(basket_metrics)
    hit_ratio = build_hit_ratio_comparison(basket_metrics)
    metrics = basket_metrics_to_metric_rows(context, basket_metrics)

    write_csv(context.output_dir / "artifacts" / "topn_basket_metrics.csv", basket_metrics)
    write_csv(context.output_dir / "artifacts" / "basket_period_returns.csv", period_returns)
    write_csv(context.output_dir / "artifacts" / "drawdown_comparison.csv", drawdown)
    write_csv(context.output_dir / "artifacts" / "hit_ratio_comparison.csv", hit_ratio)
    write_csv(context.output_dir / "metrics" / "metrics.csv", metrics)
    write_json(context.output_dir / "metrics" / "metrics_summary.json", summarize_metrics(metrics))

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_ROOT / "topn_basket_metrics.csv", basket_metrics)
    write_csv(REPORT_ROOT / "drawdown_comparison.csv", drawdown)
    write_csv(REPORT_ROOT / "hit_ratio_comparison.csv", hit_ratio)

    report_text = render_basket_report(context, basket_metrics, drawdown, hit_ratio)
    (REPORT_ROOT / "EXP-RK-002_BASKET_EVALUATION.md").write_text(report_text, encoding="utf-8")
    (context.output_dir / "reports" / "EXP-RK-002_BASKET_EVALUATION.md").write_text(report_text, encoding="utf-8")
    write_summary(context, "completed_with_errors" if context.errors else "completed", basket_metrics, metrics)
    return {
        "basket_metric_rows": int(len(basket_metrics)),
        "period_return_rows": int(len(period_returns)),
    }


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to run Phase 3 YAML configs.")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame()
    clean = frame.copy()
    clean.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def prepare_output_dir(context: RunContext) -> None:
    for relative in ("config", "manifests", "logs", "metrics", "predictions", "artifacts", "charts", "reports"):
        (context.output_dir / relative).mkdir(parents=True, exist_ok=True)
    (context.output_dir / "charts" / ".gitkeep").write_text("", encoding="utf-8")
    (context.output_dir / "artifacts" / ".gitkeep").write_text("", encoding="utf-8")


def save_config_copies(context: RunContext) -> None:
    shutil.copyfile(context.config_path, context.output_dir / "config" / "original_config.yaml")
    with (context.output_dir / "config" / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(context.config, handle, sort_keys=False)


def initialize_logs(context: RunContext) -> None:
    for name in ("run.log", "errors.log"):
        (context.output_dir / "logs" / name).write_text("", encoding="utf-8")


def log(context: RunContext, name: str, message: str) -> None:
    timestamp = datetime.now(UTC).isoformat()
    with (context.output_dir / "logs" / name).open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def record_error(context: RunContext, stage: str, message: str, traceback_text: str | None = None) -> None:
    payload = {"stage": stage, "message": message, "traceback": traceback_text}
    context.errors.append(payload)
    log(context, "errors.log", json.dumps({k: v for k, v in payload.items() if k != "traceback"}, default=json_default))
    if traceback_text:
        log(context, "errors.log", traceback_text)


def validate_governance_config(context: RunContext) -> None:
    config_text = context.config_path.read_text(encoding="utf-8")
    blocked = [
        "recommended " + "stocks",
        "trade " + "recommendation",
        "buy " + "candidate",
        "sell " + "candidate",
        "investment-ready " + "result",
        "profitable trading " + "signal",
    ]
    found = [token for token in blocked if token.lower() in config_text.lower()]
    if found:
        raise ValueError(f"Phase 3 config contains forbidden decision wording: {found}")


def validate_common_config(context: RunContext) -> None:
    provider = str(context.config.get("data", {}).get("provider") or "")
    frequency = str(context.config.get("data", {}).get("frequency") or "").lower()
    if provider != "vnstock_data":
        raise ValueError("data.provider must be vnstock_data")
    if frequency != "daily":
        raise ValueError("data.frequency must be daily")
    universe = context.config.get("data", {}).get("universe") or []
    if not isinstance(universe, list) or not universe:
        raise ValueError("data.universe must be a non-empty list")
    output_root = resolve_repo_path(str(context.config.get("outputs", {}).get("root_dir") or "outputs/experiments"))
    expected = (REPO_ROOT / "outputs" / "experiments").resolve()
    if output_root != expected and expected not in output_root.parents:
        raise ValueError("outputs.root_dir must be under outputs/experiments")


def load_policy_configs(context: RunContext) -> dict[str, dict[str, Any]]:
    policy_paths = context.config.get("candidate_policies", {}) or {}
    policies: dict[str, dict[str, Any]] = {}
    for key, raw_path in policy_paths.items():
        path = resolve_repo_path(str(raw_path))
        if not path.exists():
            raise FileNotFoundError(f"Candidate policy not found: {path}")
        policy = read_yaml(path)
        if not bool(policy.get("policy", {}).get("diagnostic_only")):
            raise ValueError(f"Candidate policy is not diagnostic-only: {path}")
        policies[str(key)] = policy
    required = {"forecast_only", "risk_aware"}
    missing = required - set(policies)
    if missing:
        raise ValueError(f"Missing required candidate policies: {sorted(missing)}")
    return policies


def source_artifact_inventory(context: RunContext) -> list[dict[str, Any]]:
    source_cfg = context.config.get("source_experiments", {}) or {}
    output_root = resolve_repo_path(str(source_cfg.get("source_output_root") or "outputs/experiments"))
    experiments = [str(item) for item in source_cfg.get("forecasting_core", [])]
    inventory: list[dict[str, Any]] = []
    for experiment_id in experiments:
        base = output_root / experiment_id
        for relative, required in (
            ("predictions/predictions.csv", True),
            ("metrics/metrics.csv", False),
            ("manifests/run_manifest.json", False),
            ("logs/run.log", False),
            ("logs/errors.log", False),
        ):
            path = base / relative
            inventory.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(path),
                    "relative_path": relative,
                    "exists": path.exists(),
                    "required": required,
                }
            )
    return inventory


def load_source_predictions(context: RunContext) -> pd.DataFrame:
    source_cfg = context.config.get("source_experiments", {}) or {}
    output_root = resolve_repo_path(str(source_cfg.get("source_output_root") or "outputs/experiments"))
    frames: list[pd.DataFrame] = []
    for priority, experiment_id in enumerate([str(item) for item in source_cfg.get("forecasting_core", [])]):
        path = output_root / experiment_id / "predictions" / "predictions.csv"
        frame = read_csv(path)
        if frame.empty:
            context.warnings.append(f"source_predictions_empty_or_missing:{path}")
            continue
        frame["source_experiment"] = experiment_id
        frame["source_priority"] = priority
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["ticker"] = predictions["ticker"].astype(str).str.upper().str.strip()
    predictions["horizon"] = pd.to_numeric(predictions["horizon"], errors="coerce").astype("Int64")
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="coerce")
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
    predictions = predictions.dropna(subset=["date", "ticker", "horizon", "y_pred", "y_true"]).copy()
    horizons = set(int(value) for value in context.config.get("candidate_generation", {}).get("horizons", []))
    universe = set(str(ticker).upper().strip() for ticker in context.config.get("data", {}).get("universe", []))
    if horizons:
        predictions = predictions[predictions["horizon"].astype(int).isin(horizons)].copy()
    if universe:
        predictions = predictions[predictions["ticker"].isin(universe)].copy()
    model_rows = predictions[predictions["model_type"].astype(str).str.lower() == "model"].copy()
    if not model_rows.empty:
        predictions = model_rows
        context.warnings.append("candidate_evidence_filter:model_type_model")
    if predictions.empty:
        return predictions
    keys = ["date", "ticker", "horizon"]
    predictions["source_priority"] = pd.to_numeric(predictions["source_priority"], errors="coerce").fillna(-1)
    max_priority = predictions.groupby(keys)["source_priority"].transform("max")
    predictions = predictions[predictions["source_priority"] == max_priority].copy()
    return predictions.reset_index(drop=True)


def load_price_frames(context: RunContext) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker in [str(item).upper().strip() for item in context.config.get("data", {}).get("universe", [])]:
        path = DATA_ROOT / f"{ticker}.csv"
        if not path.exists():
            context.warnings.append(f"missing_local_ohlcv:{path}")
            continue
        frame = pd.read_csv(path)
        if "date" not in frame.columns and "time" in frame.columns:
            frame = frame.rename(columns={"time": "date"})
        if "ticker" not in frame.columns:
            frame["ticker"] = ticker
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(frame.columns))
        if missing:
            context.warnings.append(f"local_ohlcv_missing_columns:{ticker}:{missing}")
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        prices[ticker] = frame.reset_index(drop=True)
    return prices


def build_base_candidate_evidence(
    context: RunContext,
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    risk_cfg = context.config.get("risk", {}) or {}
    lookback = int(risk_cfg.get("lookback_window") or 20)
    confidence = float(risk_cfg.get("confidence_level") or 0.95)
    min_prediction_count = int(context.config.get("candidate_generation", {}).get("minimum_candidate_rows") or 1)
    allow_negative = bool(context.config.get("candidate_generation", {}).get("allow_negative_expected_return") or False)

    for (date_value, ticker, horizon), group in predictions.groupby(["date", "ticker", "horizon"], sort=True):
        price_frame = prices.get(str(ticker))
        if price_frame is None or price_frame.empty:
            continue
        current_close = lookup_close(price_frame, pd.Timestamp(date_value))
        if current_close is None or not np.isfinite(current_close) or current_close == 0:
            continue
        group = group.copy()
        group["expected_return_proxy_row"] = (pd.to_numeric(group["y_pred"], errors="coerce") / current_close) - 1.0
        group = group.dropna(subset=["expected_return_proxy_row", "y_true"])
        if group.empty:
            continue
        prediction_count = int(len(group))
        if prediction_count < min_prediction_count:
            continue
        expected_return_proxy = float(group["expected_return_proxy_row"].mean())
        if not allow_negative and expected_return_proxy < 0:
            continue
        y_true = first_finite(group["y_true"])
        realized_return = np.nan if y_true is None else float(y_true / current_close - 1.0)
        direction = 1 if expected_return_proxy >= 0 else -1
        predicted_direction = pd.to_numeric(group.get("predicted_direction"), errors="coerce")
        model_consensus_count = int((np.sign(predicted_direction.fillna(0)) == direction).sum())
        directional_confidence = model_consensus_count / prediction_count if prediction_count else np.nan
        best_idx = group["expected_return_proxy_row"].idxmax()
        best = group.loc[best_idx]
        risk = compute_lookback_risk(price_frame, pd.Timestamp(date_value), lookback, confidence)
        diagnostics = {
            "candidate_evidence": "aggregated_source_model_predictions",
            "source_model_names": sorted(set(group["model_name"].astype(str))),
            "source_model_types": sorted(set(group["model_type"].astype(str))),
            "source_rows": prediction_count,
            "source_selection": "highest_priority_source_experiment_per_date_ticker_horizon",
            "risk_lookback_window": lookback,
            "risk_confidence_level": confidence,
            "realized_outcome_source": "forecast_artifact_y_true_vs_local_candidate_date_close",
            "not_investment_advice": True,
        }
        rows.append(
            {
                "candidate_date": pd.Timestamp(date_value).date().isoformat(),
                "ticker": str(ticker),
                "horizon": int(horizon),
                "expected_return_proxy": expected_return_proxy,
                "directional_confidence": directional_confidence,
                "consensus_score": directional_confidence,
                "prediction_count": prediction_count,
                "model_consensus_count": model_consensus_count,
                "missing_prediction_rate": 0.0,
                "model_name": str(best.get("model_name")),
                "model_type": str(best.get("model_type")),
                "source_experiment": str(best.get("source_experiment")),
                "current_close": float(current_close),
                "y_true": np.nan if y_true is None else float(y_true),
                "realized_return": realized_return,
                "realized_volatility": risk["realized_volatility"],
                "max_drawdown": risk["max_drawdown"],
                "var_95": risk["var_95"],
                "cvar_95": risk["cvar_95"],
                "diagnostics": json.dumps(diagnostics, sort_keys=True, default=json_default),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def lookup_close(price_frame: pd.DataFrame, date_value: pd.Timestamp) -> float | None:
    matches = price_frame.loc[price_frame["date"] == date_value, "close"]
    if matches.empty:
        return None
    value = pd.to_numeric(matches.iloc[-1], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def first_finite(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def compute_lookback_risk(
    price_frame: pd.DataFrame,
    date_value: pd.Timestamp,
    lookback: int,
    confidence: float,
) -> dict[str, float]:
    history = price_frame.loc[price_frame["date"] <= date_value].sort_values("date").tail(max(lookback, 2) + 1)
    if history.empty or len(history) < 3:
        return {"realized_volatility": np.nan, "max_drawdown": np.nan, "var_95": np.nan, "cvar_95": np.nan}
    closes = pd.to_numeric(history["close"], errors="coerce").dropna()
    returns = closes.pct_change().dropna()
    if returns.empty:
        return {"realized_volatility": np.nan, "max_drawdown": np.nan, "var_95": np.nan, "cvar_95": np.nan}
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    var_value = float(returns.quantile(alpha))
    tail = returns[returns <= var_value]
    cvar_value = float(tail.mean()) if not tail.empty else np.nan
    rolling_peak = closes.cummax()
    drawdowns = (closes / rolling_peak) - 1.0
    return {
        "realized_volatility": float(returns.std(ddof=0)) if len(returns) > 1 else np.nan,
        "max_drawdown": float(drawdowns.min()) if not drawdowns.empty else np.nan,
        "var_95": var_value,
        "cvar_95": cvar_value,
    }


def build_candidate_rankings(
    context: RunContext,
    base: pd.DataFrame,
    policies: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if base.empty:
        return empty_candidates()
    top_n = int(context.config.get("candidate_generation", {}).get("top_n") or 5)
    forecast = rank_forecast_only(context, base, top_n, policies["forecast_only"])
    risk_aware = rank_risk_aware(context, base, top_n, policies["risk_aware"])
    candidates = pd.concat([forecast, risk_aware], ignore_index=True) if not risk_aware.empty else forecast
    for column in CANDIDATE_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = np.nan
    return candidates[CANDIDATE_COLUMNS].reset_index(drop=True)


def rank_forecast_only(
    context: RunContext,
    base: pd.DataFrame,
    top_n: int,
    policy: dict[str, Any],
) -> pd.DataFrame:
    frame = base.copy()
    frame["experiment_id"] = context.experiment_id
    frame["policy_id"] = str(policy.get("policy", {}).get("id") or "candidate_policy_forecast_only")
    frame["candidate_type"] = "forecast_only"
    frame["forecast_score"] = pd.to_numeric(frame["expected_return_proxy"], errors="coerce")
    frame["risk_penalty"] = 0.0
    frame["risk_adjusted_score"] = frame["forecast_score"]
    frame["candidate_score"] = frame["forecast_score"]
    frame = frame.sort_values(
        [
            "candidate_date",
            "horizon",
            "candidate_score",
            "directional_confidence",
            "prediction_count",
            "model_consensus_count",
            "ticker",
        ],
        ascending=[True, True, False, False, False, False, True],
    )
    ranked = add_group_rank(frame, top_n)
    ranked["diagnostics"] = ranked["diagnostics"].map(
        lambda value: append_diagnostic(value, {"policy_ranking": "forecast_only", "risk_controls_enabled": False})
    )
    return ranked


def rank_risk_aware(
    context: RunContext,
    base: pd.DataFrame,
    top_n: int,
    policy: dict[str, Any],
) -> pd.DataFrame:
    if base.empty:
        return empty_candidates()
    penalty = float(policy.get("risk_controls", {}).get("missing_metric_penalty") or 0.10)
    frames: list[pd.DataFrame] = []
    for _, group in base.groupby(["candidate_date", "horizon"], sort=True):
        scored = score_risk_aware_group(group.copy(), missing_metric_penalty=penalty)
        frames.append(scored)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        return empty_candidates()
    frame["experiment_id"] = context.experiment_id
    frame["policy_id"] = str(policy.get("policy", {}).get("id") or "candidate_policy_risk_aware")
    frame["candidate_type"] = "risk_aware"
    frame["candidate_score"] = frame["risk_adjusted_score"]
    frame = frame.sort_values(
        [
            "candidate_date",
            "horizon",
            "candidate_score",
            "max_drawdown",
            "realized_volatility",
            "prediction_count",
            "ticker",
        ],
        ascending=[True, True, False, False, True, False, True],
    )
    ranked = add_group_rank(frame, top_n)
    ranked["diagnostics"] = ranked["diagnostics"].map(
        lambda value: append_diagnostic(value, {"policy_ranking": "risk_aware", "risk_controls_enabled": True})
    )
    return ranked


def score_risk_aware_group(group: pd.DataFrame, missing_metric_penalty: float) -> pd.DataFrame:
    expected_rank = percentile_rank(group["expected_return_proxy"], higher_is_better=True)
    confidence = pd.to_numeric(group["directional_confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    consensus = pd.to_numeric(group["consensus_score"], errors="coerce").fillna(confidence).clip(0, 1)
    group["forecast_score"] = (0.50 * expected_rank) + (0.25 * confidence) + (0.25 * consensus)

    risk_columns = ["realized_volatility", "max_drawdown", "var_95", "cvar_95", "missing_prediction_rate"]
    risk_parts = []
    missing_count = pd.Series(0, index=group.index, dtype=float)
    vol = pd.to_numeric(group["realized_volatility"], errors="coerce")
    drawdown = pd.to_numeric(group["max_drawdown"], errors="coerce").abs()
    var_loss = pd.to_numeric(group["var_95"], errors="coerce").map(lambda value: abs(min(value, 0.0)) if pd.notna(value) else np.nan)
    cvar_loss = pd.to_numeric(group["cvar_95"], errors="coerce").map(lambda value: abs(min(value, 0.0)) if pd.notna(value) else np.nan)
    missing_prediction = pd.to_numeric(group["missing_prediction_rate"], errors="coerce")
    for series in (vol, drawdown, var_loss, cvar_loss, missing_prediction):
        missing_count = missing_count + series.isna().astype(float)
        risk_parts.append(percentile_rank(series, higher_is_better=False).fillna(1.0))
    group["risk_penalty"] = pd.concat(risk_parts, axis=1).mean(axis=1) + (missing_count / len(risk_columns)) * missing_metric_penalty
    group["risk_adjusted_score"] = (0.60 * group["forecast_score"]) - (0.40 * group["risk_penalty"])
    return group


def percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    ranked = values.rank(pct=True, method="average", ascending=True)
    return ranked


def add_group_rank(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby(["candidate_date", "horizon"]).cumcount() + 1
    return ranked[ranked["rank"] <= top_n].reset_index(drop=True)


def append_diagnostic(value: Any, additions: dict[str, Any]) -> str:
    payload: dict[str, Any]
    try:
        payload = json.loads(str(value)) if pd.notna(value) else {}
    except Exception:
        payload = {"raw_diagnostics": str(value)}
    payload.update(additions)
    return json.dumps(payload, sort_keys=True, default=json_default)


def candidate_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "policy_id",
        "candidate_type",
        "horizon",
        "metric_name",
        "metric_value",
        "sample_size",
        "notes",
        "diagnostic_only",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    metric_map = {
        "average_candidate_score": "candidate_score",
        "average_expected_return_proxy": "expected_return_proxy",
        "average_realized_volatility": "realized_volatility",
        "average_max_drawdown": "max_drawdown",
        "average_var_95": "var_95",
        "average_cvar_95": "cvar_95",
    }
    for (candidate_type, horizon), group in candidates.groupby(["candidate_type", "horizon"], sort=True):
        policy_id = str(group["policy_id"].iloc[0])
        for metric_name, source_column in metric_map.items():
            values = pd.to_numeric(group[source_column], errors="coerce")
            rows.append(
                {
                    "experiment_id": str(group["experiment_id"].iloc[0]),
                    "policy_id": policy_id,
                    "candidate_type": candidate_type,
                    "horizon": int(horizon),
                    "metric_name": metric_name,
                    "metric_value": float(values.mean()) if values.notna().any() else np.nan,
                    "sample_size": int(values.notna().sum()),
                    "notes": f"mean_of_{source_column}",
                    "diagnostic_only": True,
                }
            )
        risk_missing = group[["realized_volatility", "max_drawdown", "var_95", "cvar_95"]].isna().any(axis=1)
        rows.append(
            {
                "experiment_id": str(group["experiment_id"].iloc[0]),
                "policy_id": policy_id,
                "candidate_type": candidate_type,
                "horizon": int(horizon),
                "metric_name": "missing_risk_metric_rate",
                "metric_value": float(risk_missing.mean()) if len(group) else np.nan,
                "sample_size": int(len(group)),
                "notes": "share_of_candidate_rows_with_any_missing_risk_metric",
                "diagnostic_only": True,
            }
        )
    for horizon, group in candidates.groupby("horizon", sort=True):
        overlaps = []
        for _, date_group in group.groupby("candidate_date", sort=True):
            forecast = set(date_group.loc[date_group["candidate_type"] == "forecast_only", "ticker"].astype(str))
            risk = set(date_group.loc[date_group["candidate_type"] == "risk_aware", "ticker"].astype(str))
            denom = max(len(forecast), len(risk), 1)
            overlaps.append(len(forecast & risk) / denom)
        rows.append(
            {
                "experiment_id": str(group["experiment_id"].iloc[0]),
                "policy_id": "forecast_only_vs_risk_aware",
                "candidate_type": "policy_comparison",
                "horizon": int(horizon),
                "metric_name": "candidate_overlap_rate",
                "metric_value": float(np.mean(overlaps)) if overlaps else np.nan,
                "sample_size": int(len(overlaps)),
                "notes": "average_ticker_overlap_per_candidate_date",
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_candidate_comparison_for_basket(context: RunContext) -> pd.DataFrame:
    source_cfg = context.config.get("source_candidates", {}) or {}
    source_root = resolve_repo_path(str(source_cfg.get("source_output_root") or "outputs/experiments"))
    experiment_id = str(source_cfg.get("candidate_comparison_experiment") or "EXP-RK-001")
    candidates_path = source_root / experiment_id / "artifacts" / "candidate_comparison.csv"
    candidates = read_csv(candidates_path)
    if candidates.empty:
        fallback = resolve_repo_path(str(source_cfg.get("fallback_report_path") or REPORT_ROOT / "candidate_comparison.csv"))
        candidates = read_csv(fallback)
        if candidates.empty:
            context.warnings.append(f"missing_candidate_comparison:{candidates_path}")
            context.warnings.append(f"missing_candidate_comparison_fallback:{fallback}")
    return candidates


def build_basket_metrics(context: RunContext, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["candidate_date"] = pd.to_datetime(candidates["candidate_date"], errors="coerce")
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="coerce")
    candidates["realized_return"] = pd.to_numeric(candidates.get("realized_return"), errors="coerce")
    horizons = [int(value) for value in context.config.get("basket", {}).get("horizons", [])]
    top_ns = [int(value) for value in context.config.get("basket", {}).get("top_n", [])]
    if horizons:
        candidates = candidates[candidates["horizon"].astype(int).isin(horizons)].copy()

    period_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for candidate_type in sorted(candidates["candidate_type"].dropna().astype(str).unique()):
        policy_id = str(candidates.loc[candidates["candidate_type"] == candidate_type, "policy_id"].iloc[0])
        for horizon in sorted(candidates["horizon"].dropna().astype(int).unique()):
            subset = candidates[(candidates["candidate_type"] == candidate_type) & (candidates["horizon"].astype(int) == horizon)].copy()
            if subset.empty:
                continue
            for top_n in top_ns:
                selected_rows = []
                for basket_date, date_group in subset.groupby("candidate_date", sort=True):
                    selected = date_group.sort_values("rank").head(top_n).copy()
                    if selected.empty:
                        continue
                    selected_rows.append(selected)
                    returns = pd.to_numeric(selected["realized_return"], errors="coerce")
                    valid = returns.dropna()
                    period_return = float(valid.mean()) if not valid.empty else np.nan
                    period_rows.append(
                        {
                            "experiment_id": context.experiment_id,
                            "policy_id": policy_id,
                            "candidate_type": candidate_type,
                            "basket_date": pd.Timestamp(basket_date).date().isoformat(),
                            "horizon": int(horizon),
                            "top_n": int(top_n),
                            "candidate_count": int(len(selected)),
                            "period_realized_return": period_return,
                            "missing_outcome_rate": float(returns.isna().mean()) if len(returns) else np.nan,
                            "diagnostic_only": True,
                        }
                    )
                if not selected_rows:
                    continue
                selected_all = pd.concat(selected_rows, ignore_index=True)
                period_frame = pd.DataFrame([row for row in period_rows if row["candidate_type"] == candidate_type and row["horizon"] == horizon and row["top_n"] == top_n])
                period_returns = pd.to_numeric(period_frame["period_realized_return"], errors="coerce").dropna()
                selected_returns = pd.to_numeric(selected_all["realized_return"], errors="coerce")
                metric_rows.append(
                    {
                        "experiment_id": context.experiment_id,
                        "policy_id": policy_id,
                        "candidate_type": candidate_type,
                        "basket_date": "ALL",
                        "horizon": int(horizon),
                        "top_n": int(top_n),
                        "candidate_count": int(len(selected_all)),
                        "average_realized_return": float(period_returns.mean()) if not period_returns.empty else np.nan,
                        "median_realized_return": float(period_returns.median()) if not period_returns.empty else np.nan,
                        "hit_ratio": float((period_returns > 0).mean()) if not period_returns.empty else np.nan,
                        "return_volatility_proxy": return_volatility_proxy(period_returns),
                        "max_drawdown": max_drawdown_from_returns(period_returns),
                        "var_95": var_from_returns(period_returns, 0.95),
                        "cvar_95": cvar_from_returns(period_returns, 0.95),
                        "worst_period_return": float(period_returns.min()) if not period_returns.empty else np.nan,
                        "missing_outcome_rate": float(selected_returns.isna().mean()) if len(selected_returns) else np.nan,
                        "diagnostic_only": True,
                        "basket_count": int(len(period_returns)),
                    }
                )
    basket_metrics = pd.DataFrame(metric_rows)
    for column in BASKET_COLUMNS:
        if column not in basket_metrics.columns:
            basket_metrics[column] = np.nan
    period_returns = pd.DataFrame(period_rows)
    return basket_metrics[BASKET_COLUMNS].reset_index(drop=True), period_returns.reset_index(drop=True)


def return_volatility_proxy(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    volatility = values.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return np.nan
    return float(values.mean() / volatility)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def var_from_returns(returns: pd.Series, confidence: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(1.0 - confidence))


def cvar_from_returns(returns: pd.Series, confidence: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    var_value = var_from_returns(values, confidence)
    tail = values[values <= var_value]
    return float(tail.mean()) if not tail.empty else np.nan


def build_drawdown_comparison(basket_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "top_n",
        "forecast_only_max_drawdown",
        "risk_aware_max_drawdown",
        "drawdown_reduction_vs_forecast_only",
        "diagnostic_only",
    ]
    if basket_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (horizon, top_n), group in basket_metrics.groupby(["horizon", "top_n"], sort=True):
        forecast = first_metric(group, "forecast_only", "max_drawdown")
        risk = first_metric(group, "risk_aware", "max_drawdown")
        reduction = np.nan
        if pd.notna(forecast) and pd.notna(risk):
            reduction = abs(float(forecast)) - abs(float(risk))
        rows.append(
            {
                "horizon": int(horizon),
                "top_n": int(top_n),
                "forecast_only_max_drawdown": forecast,
                "risk_aware_max_drawdown": risk,
                "drawdown_reduction_vs_forecast_only": reduction,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_hit_ratio_comparison(basket_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "top_n",
        "forecast_only_hit_ratio",
        "risk_aware_hit_ratio",
        "hit_ratio_difference_vs_forecast_only",
        "diagnostic_only",
    ]
    if basket_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (horizon, top_n), group in basket_metrics.groupby(["horizon", "top_n"], sort=True):
        forecast = first_metric(group, "forecast_only", "hit_ratio")
        risk = first_metric(group, "risk_aware", "hit_ratio")
        diff = np.nan
        if pd.notna(forecast) and pd.notna(risk):
            diff = float(risk) - float(forecast)
        rows.append(
            {
                "horizon": int(horizon),
                "top_n": int(top_n),
                "forecast_only_hit_ratio": forecast,
                "risk_aware_hit_ratio": risk,
                "hit_ratio_difference_vs_forecast_only": diff,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def first_metric(group: pd.DataFrame, candidate_type: str, column: str) -> float:
    values = group.loc[group["candidate_type"] == candidate_type, column]
    if values.empty:
        return np.nan
    value = pd.to_numeric(values, errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else np.nan


def basket_metrics_to_metric_rows(context: RunContext, basket_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "policy_id",
        "candidate_type",
        "horizon",
        "top_n",
        "metric_name",
        "metric_value",
        "sample_size",
        "notes",
        "diagnostic_only",
    ]
    if basket_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    metric_names = [
        "average_realized_return",
        "median_realized_return",
        "hit_ratio",
        "return_volatility_proxy",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "worst_period_return",
        "candidate_count",
        "missing_outcome_rate",
    ]
    for _, row in basket_metrics.iterrows():
        for metric_name in metric_names:
            rows.append(
                {
                    "experiment_id": context.experiment_id,
                    "policy_id": row.get("policy_id"),
                    "candidate_type": row.get("candidate_type"),
                    "horizon": row.get("horizon"),
                    "top_n": row.get("top_n"),
                    "metric_name": metric_name,
                    "metric_value": row.get(metric_name),
                    "sample_size": row.get("basket_count"),
                    "notes": "aggregated_equal_weight_diagnostic_basket",
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_risk_summary(candidates: pd.DataFrame, baskets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidates is not None and not candidates.empty:
        for (candidate_type, horizon), group in candidates.groupby(["candidate_type", "horizon"], sort=True):
            rows.append(
                {
                    "source": "candidate_comparison",
                    "candidate_type": candidate_type,
                    "horizon": int(horizon),
                    "top_n": "",
                    "average_realized_volatility": numeric_mean(group, "realized_volatility"),
                    "average_max_drawdown": numeric_mean(group, "max_drawdown"),
                    "average_var_95": numeric_mean(group, "var_95"),
                    "average_cvar_95": numeric_mean(group, "cvar_95"),
                    "average_realized_return": numeric_mean(group, "realized_return"),
                    "return_volatility_proxy": "",
                    "hit_ratio": float((pd.to_numeric(group["realized_return"], errors="coerce").dropna() > 0).mean())
                    if pd.to_numeric(group["realized_return"], errors="coerce").notna().any()
                    else np.nan,
                    "diagnostic_only": True,
                }
            )
    if baskets is not None and not baskets.empty:
        for _, row in baskets.iterrows():
            rows.append(
                {
                    "source": "basket_outcome",
                    "candidate_type": row.get("candidate_type"),
                    "horizon": row.get("horizon"),
                    "top_n": row.get("top_n"),
                    "average_realized_volatility": "",
                    "average_max_drawdown": row.get("max_drawdown"),
                    "average_var_95": row.get("var_95"),
                    "average_cvar_95": row.get("cvar_95"),
                    "average_realized_return": row.get("average_realized_return"),
                    "return_volatility_proxy": row.get("return_volatility_proxy"),
                    "hit_ratio": row.get("hit_ratio"),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics is None or metrics.empty:
        return {"metric_rows": 0}
    return {
        "metric_rows": int(len(metrics)),
        "metric_names": sorted(set(metrics["metric_name"].astype(str))) if "metric_name" in metrics.columns else [],
        "diagnostic_only": True,
    }


def write_manifest(context: RunContext, status: str, result: dict[str, Any]) -> None:
    completed_at = datetime.now(UTC)
    artifact_paths = {
        "original_config": str(context.output_dir / "config" / "original_config.yaml"),
        "resolved_config": str(context.output_dir / "config" / "resolved_config.yaml"),
        "run_manifest": str(context.output_dir / "manifests" / "run_manifest.json"),
        "run_log": str(context.output_dir / "logs" / "run.log"),
        "errors_log": str(context.output_dir / "logs" / "errors.log"),
        "metrics": str(context.output_dir / "metrics" / "metrics.csv"),
        "summary": str(context.output_dir / "reports" / "summary.md"),
    }
    for name in (
        "candidate_comparison.csv",
        "risk_adjusted_ranking.csv",
        "risk_summary.csv",
        "topn_basket_metrics.csv",
        "basket_period_returns.csv",
        "drawdown_comparison.csv",
        "hit_ratio_comparison.csv",
    ):
        path = context.output_dir / "artifacts" / name
        if path.exists():
            artifact_paths[Path(name).stem] = str(path)
    payload = {
        "manifest_type": "risk_aware_decision_research_v1_manifest",
        "experiment_id": context.experiment_id,
        "experiment_name": context.config.get("experiment", {}).get("name"),
        "phase": context.config.get("experiment", {}).get("phase"),
        "run_id": context.run_id,
        "started_at": context.started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "status": status,
        "config_path": str(context.config_path),
        "output_dir": str(context.output_dir),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "provider": context.config.get("data", {}).get("provider"),
        "frequency": context.config.get("data", {}).get("frequency"),
        "universe": context.config.get("data", {}).get("universe") or [],
        "diagnostic_only_authority": True,
        "no_buy_sell_hold_advice_authority": True,
        "no_capital_allocation_authority": True,
        "not_investment_advice": True,
        "disclaimer": DISCLAIMER,
        "artifact_paths": artifact_paths,
        "result": result,
        "errors": context.errors,
        "warnings": sorted(set(context.warnings)),
    }
    write_json(context.output_dir / "manifests" / "run_manifest.json", payload)


def git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_summary(context: RunContext, status: str, primary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        f"# {context.experiment_id} Summary",
        "",
        f"- Status: `{status}`",
        f"- Run ID: `{context.run_id}`",
        f"- Experiment name: {context.config.get('experiment', {}).get('name') or ''}",
        "- Candidate outputs: diagnostic decision-support artifacts only",
        "- Not investment advice: `true`",
        "",
        "## Evidence",
        "",
        "- `config/original_config.yaml`",
        "- `config/resolved_config.yaml`",
        "- `manifests/run_manifest.json`",
        "- `logs/run.log`",
        "- `logs/errors.log`",
        "- `metrics/metrics.csv`",
        f"- Primary rows: `{0 if primary is None else len(primary)}`",
        f"- Metric rows: `{0 if metrics is None else len(metrics)}`",
        "",
        "## Disclaimer",
        "",
        DISCLAIMER,
        "",
    ]
    if context.warnings:
        lines.extend(["## Missing Artifacts And Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in sorted(set(context.warnings)))
        lines.append("")
    if context.errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{error.get('stage')}`: {error.get('message')}" for error in context.errors)
        lines.append("")
    (context.output_dir / "reports" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def render_candidate_report(
    context: RunContext,
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    source_inventory: list[dict[str, Any]],
) -> str:
    overlap = metrics.loc[metrics["metric_name"] == "candidate_overlap_rate"].copy() if not metrics.empty else pd.DataFrame()
    preview_columns = [
        "candidate_type",
        "candidate_date",
        "ticker",
        "horizon",
        "rank",
        "candidate_score",
        "expected_return_proxy",
        "realized_volatility",
        "max_drawdown",
        "diagnostic_only",
    ]
    preview = candidates[preview_columns].head(20) if not candidates.empty else pd.DataFrame(columns=preview_columns)
    source_frame = pd.DataFrame(source_inventory)
    lines = [
        "# EXP-RK-001 Candidate Comparison",
        "",
        "## Purpose",
        "",
        "Compare forecast-only ranking against risk-aware ranking using the same Phase 2 forecast evidence where possible.",
        "",
        "## Candidate Rows",
        "",
        f"- Total candidate rows: `{len(candidates)}`",
        f"- Forecast-only rows: `{len(candidates[candidates['candidate_type'] == 'forecast_only']) if not candidates.empty else 0}`",
        f"- Risk-aware rows: `{len(candidates[candidates['candidate_type'] == 'risk_aware']) if not candidates.empty else 0}`",
        "- Every row is diagnostic-only and not investment advice.",
        "",
        "## Source Artifact Evidence",
        "",
        markdown_table(source_frame),
        "",
        "## Candidate Overlap",
        "",
        markdown_table(overlap),
        "",
        "## Candidate Preview",
        "",
        markdown_table(preview),
        "",
        "## Disclaimer",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def render_basket_report(
    context: RunContext,
    basket_metrics: pd.DataFrame,
    drawdown: pd.DataFrame,
    hit_ratio: pd.DataFrame,
) -> str:
    lines = [
        "# EXP-RK-002 Basket Evaluation",
        "",
        "## Purpose",
        "",
        "Evaluate realized outcomes for equal-weight diagnostic candidate baskets formed from forecast-only and risk-aware policies.",
        "",
        "## Top-N Basket Metrics",
        "",
        markdown_table(basket_metrics),
        "",
        "## Drawdown Comparison",
        "",
        markdown_table(drawdown),
        "",
        "## Hit Ratio Comparison",
        "",
        markdown_table(hit_ratio),
        "",
        "## Disclaimer",
        "",
        DISCLAIMER,
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
        values = [format_markdown_value(row[column]) for column in clean.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)


def empty_baskets() -> pd.DataFrame:
    return pd.DataFrame(columns=BASKET_COLUMNS)


if __name__ == "__main__":
    raise SystemExit(main())
