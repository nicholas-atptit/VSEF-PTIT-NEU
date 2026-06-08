"""Run VN30 stock-only hourly expanded model-pool screening."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import (  # noqa: E402
    BASELINE_STOCK_RF_H60_REFERENCE,
    HORIZONS,
    MARKET_CONTEXT_FEATURES,
    OWN_FEATURES,
    read_joint_panel_universe,
    write_csv,
    write_json,
)
from scripts.research.vn30_hourly_common import standardize_hourly_frame  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_expanded_model_pool"
STOCK_CACHE_DIRS = [
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015",
    REPO_ROOT / "data" / "hourly_market_split_data",
]
INDEX_CACHE_DIRS = [
    REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015",
    REPO_ROOT / "archive" / "generated_data_snapshots" / "vn30_hourly_pre_benchmark_20260514_062528" / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly",
]
RANDOM_STATE = 42
LOCKED_BASELINE = BASELINE_STOCK_RF_H60_REFERENCE
ACTIVE_TICKER_COUNT = 30
SCREENING_MODELS = ["random_forest", "extra_trees", "decision_tree_cart", "xgboost", "lightgbm", "logistic_regression"]
BASELINES = ["majority", "always_up", "previous_direction", "moving_average", "random_seeded"]
FEATURE_SETS = {
    "stock_lagged_rolling": OWN_FEATURES,
    "stock_lagged_rolling_plus_index_context": OWN_FEATURES + MARKET_CONTEXT_FEATURES,
}
RESULT_COLUMNS = [
    "model_family",
    "model_name",
    "horizon",
    "feature_set",
    "hyperparams_id",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "validation_stability_score",
    "final_accuracy",
    "final_baseline_accuracy",
    "final_delta_vs_baseline",
    "final_rows",
    "final_coverage",
    "active_ticker_count",
    "selected_on_validation",
    "pass_locked_60_31",
    "pass_65",
    "claim_level",
]
PREDICTION_COLUMNS = [
    "candidate_id",
    "model_family",
    "model_name",
    "horizon",
    "feature_set",
    "hyperparams_id",
    "split",
    "datetime",
    "ticker",
    "target_direction",
    "prediction",
    "probability_up",
    "is_correct",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def load_stock_frame(ticker: str) -> pd.DataFrame:
    best = pd.DataFrame()
    for directory in STOCK_CACHE_DIRS:
        path = directory / f"{ticker}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, low_memory=False)
        frame = standardize_hourly_frame(raw, ticker)
        if len(frame) > len(best):
            best = frame
    return best


def read_index_frame(path: Path, code: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    frame = raw.copy()
    normalized = {str(col).lower().strip(): col for col in frame.columns}
    time_col = next((normalized[key] for key in ("datetime", "timestamp", "time", "date", "trading_date") if key in normalized), None)
    if time_col is None:
        return pd.DataFrame()
    if time_col != "datetime":
        frame = frame.rename(columns={time_col: "datetime"})
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ticker"] = code
    return frame.dropna(subset=["datetime", "close"])[["datetime", "ticker", "open", "high", "low", "close", "volume"]]


def load_index_context_frame(code: str) -> pd.DataFrame:
    best = pd.DataFrame()
    for directory in INDEX_CACHE_DIRS:
        path = directory / f"{code}.csv"
        if not path.exists():
            continue
        frame = read_index_frame(path, code)
        if len(frame) > len(best):
            best = frame
    return best


def build_stock_panel() -> pd.DataFrame:
    tickers, _indices = read_joint_panel_universe()
    frames = []
    missing = []
    for ticker in tickers:
        frame = load_stock_frame(ticker)
        if frame.empty:
            missing.append(ticker)
        else:
            frames.append(frame)
    if missing:
        raise RuntimeError(f"Missing hourly cache for active VN30 tickers: {missing}")
    panel = pd.concat(frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"], errors="coerce")
    return panel.dropna(subset=["datetime"]).sort_values(["ticker", "datetime"]).reset_index(drop=True)


def add_stock_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["ticker", "datetime"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    out["return_1"] = grouped["close"].pct_change(fill_method=None)
    out["volume_change_1"] = grouped["volume"].pct_change(fill_method=None)
    out["high_low_range"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["open_close_spread"] = (out["close"] - out["open"]) / out["open"].replace(0.0, np.nan)
    for lag in (1, 2, 3, 5, 10):
        out[f"return_1_lag_{lag}"] = grouped["return_1"].shift(lag)
    for window in (3, 5, 10, 20):
        min_periods = max(2, window // 2)
        out[f"rolling_return_{window}"] = grouped["close"].pct_change(window, fill_method=None)
        out[f"rolling_volatility_{window}"] = grouped["return_1"].rolling(window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        out[f"momentum_{window}"] = out["close"] / grouped["close"].shift(window) - 1.0
        out[f"range_shock_{window}"] = out["high_low_range"] / grouped["high_low_range"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True) - 1.0
        out[f"volume_shock_{window}"] = out["volume"] / grouped["volume"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True) - 1.0
    out["previous_direction"] = (grouped["close"].diff() > 0).astype(float)
    out["previous_direction"] = grouped["previous_direction"].shift(1)
    return out


def build_market_context() -> pd.DataFrame:
    _tickers, indices = read_joint_panel_universe()
    pieces: list[pd.DataFrame] = []
    for code in indices:
        frame = load_index_context_frame(code)
        if frame.empty:
            continue
        frame = frame.sort_values("datetime")
        returns = frame["close"].pct_change(fill_method=None)
        piece = pd.DataFrame({"datetime": frame["datetime"]})
        piece[f"{code}_lag_return_1"] = returns.shift(1)
        piece[f"{code}_lag_return_3"] = frame["close"].pct_change(3, fill_method=None).shift(1)
        piece[f"{code}_lag_volatility_10"] = returns.rolling(10, min_periods=5).std().shift(1)
        piece[f"{code}_lag_trend_10"] = (frame["close"] / frame["close"].rolling(10, min_periods=5).mean() - 1.0).shift(1)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["datetime"])
    context = pieces[0]
    for piece in pieces[1:]:
        context = pd.merge(context, piece, on="datetime", how="outer")
    return context.sort_values("datetime").reset_index(drop=True)


def build_base_panel() -> pd.DataFrame:
    stock = add_stock_features(build_stock_panel())
    context = build_market_context()
    if not context.empty:
        stock = pd.merge_asof(stock.sort_values("datetime"), context.sort_values("datetime"), on="datetime", direction="backward", allow_exact_matches=False)
    return stock.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def add_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.sort_values(["ticker", "datetime"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    out["target_horizon"] = horizon
    out["future_close"] = grouped["close"].shift(-horizon)
    out["target_direction"] = (out["future_close"] > out["close"]).astype(float)
    out.loc[out["future_close"].isna(), "target_direction"] = np.nan
    return out.dropna(subset=["target_direction"]).reset_index(drop=True)


def add_splits(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["ticker", "datetime"]).copy()
    split_values: list[str] = []
    for _ticker, group in out.groupby("ticker", sort=False):
        n = len(group)
        train_end = max(1, int(n * 0.60))
        validation_end = max(train_end + 1, int(n * 0.80))
        validation_end = min(validation_end, max(train_end + 1, n - 1)) if n >= 3 else n
        labels = ["train"] * n
        for pos in range(train_end, validation_end):
            if pos < n:
                labels[pos] = "validation"
        for pos in range(validation_end, n):
            labels[pos] = "final"
        split_values.extend(labels)
    out["split"] = split_values
    return out.reset_index(drop=True)


def make_model(model_name: str) -> Any | None:
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=220, max_depth=9, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=220, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "decision_tree_cart":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE)
    if model_name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2)
    if model_name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=180, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=2, verbose=-1)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    return None


def model_family(model_name: str) -> str:
    return {
        "random_forest": "Random Forest",
        "extra_trees": "ExtraTrees",
        "decision_tree_cart": "Decision Tree / CART",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "logistic_regression": "Logistic Regression",
    }.get(model_name, model_name)


def predict_baseline(data: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    if name == "always_up":
        pred = pd.Series(1, index=data.index, dtype=float)
    elif name == "majority":
        majority = 1 if data[data["split"].eq("train")]["target_direction"].mean() >= 0.5 else 0
        pred = pd.Series(majority, index=data.index, dtype=float)
    elif name == "previous_direction":
        pred = data["previous_direction"].fillna(1).astype(float)
    elif name == "moving_average":
        pred = (data["rolling_return_5"].fillna(0) > 0).astype(float)
    elif name == "random_seeded":
        rng = np.random.default_rng(RANDOM_STATE + int(data["target_horizon"].iloc[0]))
        pred = pd.Series(rng.integers(0, 2, size=len(data)), index=data.index, dtype=float)
    else:
        raise ValueError(f"Unknown baseline {name}")
    return pred, pred.astype(float)


def metric(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.dropna(subset=["target_direction", "prediction"])
    rows = len(valid)
    correct = int((valid["target_direction"].astype(int) == valid["prediction"].astype(int)).sum()) if rows else 0
    return {"accuracy": correct / rows if rows else math.nan, "rows": rows, "coverage": rows / len(frame) if len(frame) else 0.0}


def simple_baseline_accuracy(data: pd.DataFrame, split: str) -> float:
    scoped = data.copy()
    scoped["prediction"], _prob = predict_baseline(scoped, "majority")
    return metric(scoped[scoped["split"].eq(split)])["accuracy"]


def stability_score(scored: pd.DataFrame) -> float:
    val = scored[scored["split"].eq("validation")].copy()
    if val.empty:
        return math.nan
    val["month"] = pd.to_datetime(val["datetime"]).dt.to_period("M").astype(str)
    accuracies = [metric(group)["accuracy"] for _month, group in val.groupby("month") if metric(group)["rows"]]
    if not accuracies:
        return math.nan
    return float(np.nanmean(accuracies) - np.nanstd(accuracies))


def fit_predict_model(data: pd.DataFrame, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    model = make_model(model_name)
    if model is None:
        raise RuntimeError(f"model_unavailable:{model_name}")
    train = data[data["split"].eq("train")]
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipeline.fit(train[feature_cols], train["target_direction"].astype(int))
    scored = data.copy()
    scored["prediction"] = pipeline.predict(scored[feature_cols]).astype(float)
    if hasattr(pipeline, "predict_proba"):
        scored["probability_up"] = pipeline.predict_proba(scored[feature_cols])[:, 1]
    else:
        scored["probability_up"] = scored["prediction"]
    return scored


def scored_to_predictions(candidate_id: str, row: dict[str, Any], scored: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in scored[scored["split"].isin(["validation", "final"])].itertuples(index=False):
        actual = int(getattr(record, "target_direction"))
        pred = int(getattr(record, "prediction"))
        out.append(
            {
                "candidate_id": candidate_id,
                "model_family": row["model_family"],
                "model_name": row["model_name"],
                "horizon": row["horizon"],
                "feature_set": row["feature_set"],
                "hyperparams_id": row["hyperparams_id"],
                "split": getattr(record, "split"),
                "datetime": pd.Timestamp(getattr(record, "datetime")).strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": getattr(record, "ticker"),
                "target_direction": actual,
                "prediction": pred,
                "probability_up": float(getattr(record, "probability_up")),
                "is_correct": int(actual == pred),
            }
        )
    return out


def result_row(model_name: str, horizon: int, feature_set: str, data: pd.DataFrame, scored: pd.DataFrame) -> dict[str, Any]:
    validation = metric(scored[scored["split"].eq("validation")])
    final = metric(scored[scored["split"].eq("final")])
    validation_baseline = simple_baseline_accuracy(data, "validation")
    final_baseline = simple_baseline_accuracy(data, "final")
    final_accuracy = final["accuracy"]
    return {
        "model_family": model_family(model_name) if model_name not in BASELINES else "Baseline",
        "model_name": model_name,
        "horizon": horizon,
        "feature_set": feature_set,
        "hyperparams_id": "v1_default",
        "validation_accuracy": validation["accuracy"],
        "validation_baseline_accuracy": validation_baseline,
        "validation_delta_vs_baseline": validation["accuracy"] - validation_baseline if math.isfinite(validation["accuracy"]) else math.nan,
        "validation_stability_score": stability_score(scored),
        "final_accuracy": final_accuracy,
        "final_baseline_accuracy": final_baseline,
        "final_delta_vs_baseline": final_accuracy - final_baseline if math.isfinite(final_accuracy) else math.nan,
        "final_rows": final["rows"],
        "final_coverage": final["coverage"],
        "active_ticker_count": ACTIVE_TICKER_COUNT,
        "selected_on_validation": False,
        "pass_locked_60_31": bool(math.isfinite(final_accuracy) and final_accuracy > LOCKED_BASELINE),
        "pass_65": bool(math.isfinite(final_accuracy) and final_accuracy >= 0.65),
        "claim_level": "exploratory",
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("validation_accuracy") or -1),
        float(row.get("validation_delta_vs_baseline") or -1),
        float(row.get("validation_stability_score") or -1),
    )


def write_markdown_log(rows: list[dict[str, Any]], selected: dict[str, Any] | None, skipped: list[str]) -> None:
    lines = [
        "# VN30 Hourly Expanded Model Pool Screening Run Log",
        "",
        "- Status: completed.",
        "- Scope: stock-only hourly, 30/30 VN30 January 2025 tickers.",
        "- Coverage: full coverage; no confidence abstention.",
        "- Selection: validation-only.",
        "- Final window: scoring-only.",
        f"- Candidate rows: {len(rows)}.",
        f"- Skipped models: {', '.join(skipped) if skipped else 'none'}.",
        f"- Selected candidate: `{selected.get('model_name')} h={selected.get('horizon')} {selected.get('feature_set')}`." if selected else "- Selected candidate: none.",
        "",
    ]
    (OUTPUT_DIR / "screening_run_log.md").write_text("\n".join(lines), encoding="utf-8")


def build_error_correlation(predictions: pd.DataFrame) -> pd.DataFrame:
    final = predictions[predictions["split"].eq("final")].copy()
    if final.empty:
        return pd.DataFrame(columns=["candidate_id_a", "candidate_id_b", "error_correlation"])
    final["key"] = final["datetime"].astype(str) + "|" + final["ticker"].astype(str) + "|" + final["horizon"].astype(str)
    wide = final.pivot_table(index="key", columns="candidate_id", values="is_correct", aggfunc="first")
    errors = 1 - wide
    corr = errors.corr()
    rows = []
    for a in corr.columns:
        for b in corr.columns:
            if a < b:
                rows.append({"candidate_id_a": a, "candidate_id_b": b, "error_correlation": corr.loc[a, b]})
    return pd.DataFrame(rows)


def build_disagreement(predictions: pd.DataFrame) -> pd.DataFrame:
    final = predictions[predictions["split"].eq("final")].copy()
    if final.empty:
        return pd.DataFrame(columns=["candidate_id_a", "candidate_id_b", "shared_rows", "disagreement_rate"])
    final["key"] = final["datetime"].astype(str) + "|" + final["ticker"].astype(str) + "|" + final["horizon"].astype(str)
    wide = final.pivot_table(index="key", columns="candidate_id", values="prediction", aggfunc="first")
    rows = []
    columns = list(wide.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            paired = wide[[a, b]].dropna()
            rows.append({"candidate_id_a": a, "candidate_id_b": b, "shared_rows": len(paired), "disagreement_rate": float((paired[a] != paired[b]).mean()) if len(paired) else math.nan})
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_panel = build_base_panel()
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    available_models = [name for name in SCREENING_MODELS if make_model(name) is not None]
    skipped.extend([name for name in SCREENING_MODELS if name not in available_models])

    for horizon in HORIZONS:
        labeled = add_splits(add_targets(base_panel, horizon))
        for feature_set, requested_features in FEATURE_SETS.items():
            feature_cols = [col for col in requested_features if col in labeled.columns]
            data = labeled[["datetime", "ticker", "target_direction", "previous_direction", "target_horizon", "split", *feature_cols]].copy()
            for model_name in available_models:
                candidate_id = f"{model_name}__h{horizon}__{feature_set}"
                try:
                    scored = fit_predict_model(data, feature_cols, model_name)
                    row = result_row(model_name, horizon, feature_set, data, scored)
                except Exception as exc:
                    row = {
                        "model_family": model_family(model_name),
                        "model_name": model_name,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "hyperparams_id": "v1_default",
                        "claim_level": f"failed:{exc}",
                    }
                    rows.append(row)
                    continue
                rows.append(row)
                prediction_rows.extend(scored_to_predictions(candidate_id, row, scored))
            for baseline_name in BASELINES:
                scored = data.copy()
                scored["prediction"], scored["probability_up"] = predict_baseline(scored, baseline_name)
                row = result_row(baseline_name, horizon, feature_set, data, scored)
                rows.append(row)
                candidate_id = f"{baseline_name}__h{horizon}__{feature_set}"
                prediction_rows.extend(scored_to_predictions(candidate_id, row, scored))

    valid_model_rows = [
        row for row in rows
        if row.get("model_family") != "Baseline"
        and str(row.get("claim_level")) == "exploratory"
        and float(row.get("validation_delta_vs_baseline") or -1) > 0
    ]
    selected = max(valid_model_rows, key=selection_key) if valid_model_rows else None
    if selected:
        selected["selected_on_validation"] = True

    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "status": "completed",
            "scope": "VN30 stock-only hourly",
            "active_ticker_count": ACTIVE_TICKER_COUNT,
            "horizons": list(HORIZONS),
            "feature_sets": list(FEATURE_SETS),
            "models_requested": SCREENING_MODELS,
            "models_executed": available_models,
            "baselines": BASELINES,
            "locked_rf_h60_baseline": LOCKED_BASELINE,
            "selection_rule": "validation-only",
            "final_window": "scoring-only",
            "confidence_abstention": False,
            "ticker_subset": False,
            "data_fetch": False,
        },
    )
    write_json(
        OUTPUT_DIR / "screening_manifest.json",
        {
            "status": "completed",
            "selected_on_validation": bool(selected),
            "selected_candidate": selected,
            "final_accuracy_used_for_selection": False,
            "stock_only": True,
            "active_ticker_count": ACTIVE_TICKER_COUNT,
            "full_coverage": True,
        },
    )
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS)
    pred_frame = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    pred_frame.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False)
    build_error_correlation(pred_frame).to_csv(OUTPUT_DIR / "model_error_correlation.csv", index=False)
    build_disagreement(pred_frame).to_csv(OUTPUT_DIR / "model_disagreement_summary.csv", index=False)
    write_markdown_log(rows, selected, skipped)
    print(f"screening_status=completed rows={len(rows)} selected={selected.get('model_name') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
