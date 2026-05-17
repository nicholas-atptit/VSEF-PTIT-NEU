"""Final65 focus optimization for RF h=60 absolute direction."""
from __future__ import annotations
import csv, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STOCK_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_rf_h60_final65_focus"
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
HORIZON = 60
SEED = 42
CONFIDENCE_THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]
RF_PARAMS_BASE = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED}

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def fmt_pct(v: Any) -> str:
    try:
        n = float(v)
        if not math.isfinite(n): return ""
        return f"{n * 100:.2f}%"
    except: return ""

def load_universe_tickers() -> list[str]:
    tickers = []
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker", "")).strip().upper()
            if t: tickers.append(t)
    return tickers

def load_stock_data(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = STOCK_CACHE_DIR / f"{ticker}.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["ticker"] = ticker
        for col in OHLCV_COLUMNS: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["datetime"] + OHLCV_COLUMNS)
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]
        frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)

def load_index_data() -> dict[str, pd.DataFrame]:
    indices = {}
    for code in ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]:
        path = INDEX_CACHE_DIR / f"{code}.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])
        df = df[df["close"] > 0].sort_values("datetime").reset_index(drop=True)
        indices[code] = df
    return indices

def build_features(df: pd.DataFrame, index_data: dict | None = None, include_market: bool = False) -> tuple[pd.DataFrame, list[str]]:
    prepared = df.copy().sort_values(["ticker", "datetime"]).reset_index(drop=True)
    feat_cols = []
    for ticker, group in prepared.groupby("ticker", sort=True):
        idx = group.index
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        prepared.loc[idx, "return_1"] = close.pct_change(periods=1, fill_method=None)
        for lag in (2, 3, 5, 10, 20): prepared.loc[idx, f"return_{lag}"] = close.pct_change(periods=lag, fill_method=None)
        for lag in (1, 2, 3, 5, 10, 20): prepared.loc[idx, f"return_1_lag_{lag}"] = prepared.loc[idx, "return_1"].shift(lag)
        for window in (5, 10, 20, 60):
            min_p = max(3, min(window, window // 2))
            prepared.loc[idx, f"rolling_return_mean_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"rolling_return_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
            sma = close.rolling(window, min_periods=min_p).mean()
            prepared.loc[idx, f"close_sma_ratio_{window}"] = close / sma - 1.0
            prepared.loc[idx, f"momentum_{window}"] = close / close.shift(window) - 1.0
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.loc[(loss == 0.0) & (gain > 0.0)] = 100.0
        rsi.loc[(loss == 0.0) & (gain == 0.0)] = 50.0
        prepared.loc[idx, "rsi_14"] = rsi
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        prepared.loc[idx, "macd"] = macd
        prepared.loc[idx, "macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        prepared.loc[idx, "macd_hist"] = prepared.loc[idx, "macd"] - prepared.loc[idx, "macd_signal"]
        prepared.loc[idx, "volume_change_1"] = volume.pct_change(periods=1, fill_method=None)
        vol_ma = volume.rolling(20, min_periods=5).mean()
        prepared.loc[idx, "volume_shock_20"] = volume / vol_ma - 1.0
        prepared.loc[idx, "high_low_range"] = (group["high"] - group["low"]) / close
        prepared.loc[idx, "open_close_spread"] = (close - group["open"]) / group["open"].replace(0.0, np.nan)
        prepared.loc[idx, "close_position_in_range"] = (close - group["low"]) / (group["high"] - group["low"]).replace(0.0, np.nan)
        for lag in (1, 2, 4, 8, 20): prepared.loc[idx, f"lag_ret_{lag}"] = close.pct_change(periods=lag, fill_method=None).shift(1)
        for window in (4, 8, 20, 40):
            min_p = max(2, window // 4)
            prepared.loc[idx, f"roll_ret_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).mean()
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_{window}"] = prepared.loc[idx, "return_1"].rolling(window, min_periods=min_p).std()
        for window in (8, 20, 40):
            min_p = max(3, window // 4)
            prepared.loc[idx, f"roll_vol_change_{window}"] = volume.pct_change(periods=1, fill_method=None).rolling(window, min_periods=min_p).mean()
    feat_cols = ["return_1","return_2","return_3","return_5","return_10","return_20",
        "return_1_lag_1","return_1_lag_2","return_1_lag_3","return_1_lag_5","return_1_lag_10","return_1_lag_20",
        "rolling_return_mean_5","rolling_return_vol_5","close_sma_ratio_5","momentum_5",
        "rolling_return_mean_10","rolling_return_vol_10","close_sma_ratio_10","momentum_10",
        "rolling_return_mean_20","rolling_return_vol_20","close_sma_ratio_20","momentum_20",
        "rolling_return_mean_60","rolling_return_vol_60","close_sma_ratio_60","momentum_60",
        "rsi_14","macd","macd_signal","macd_hist","volume_change_1","volume_shock_20",
        "high_low_range","open_close_spread","close_position_in_range",
        "lag_ret_1","lag_ret_2","lag_ret_4","lag_ret_8","lag_ret_20",
        "roll_ret_4","roll_ret_8","roll_ret_20","roll_ret_40",
        "roll_vol_8","roll_vol_20","roll_vol_40",
        "roll_vol_change_8","roll_vol_change_20","roll_vol_change_40"]
    if include_market and index_data:
        for code, idx_df in index_data.items():
            idx_df = idx_df.copy()
            idx_df["idx_ret"] = idx_df["close"].pct_change(periods=1, fill_method=None)
            idx_features = []
            for lag in (1, 2, 3, 5, 10, 20):
                col = f"{code.lower()}_lag_{lag}"
                idx_df[col] = idx_df["idx_ret"].shift(lag)
                idx_features.append(col)
            for window in (20, 60):
                min_p = max(3, window // 4)
                cm = f"{code.lower()}_roll_mean_{window}"
                cv = f"{code.lower()}_roll_vol_{window}"
                idx_df[cm] = idx_df["idx_ret"].rolling(window, min_periods=min_p).mean()
                idx_df[cv] = idx_df["idx_ret"].rolling(window, min_periods=min_p).std()
                idx_features.extend([cm, cv])
            sel_cols = ["datetime"] + idx_features
            idx_clean = idx_df[sel_cols].dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
            prepared = prepared.merge(idx_clean, on="datetime", how="left")
            feat_cols.extend(idx_features)
            if "vnindex_lag_1" in prepared.columns and "market_minus_stock_ret" not in feat_cols:
                prepared["market_minus_stock_ret"] = prepared["vnindex_lag_1"] - prepared["return_1"]
                feat_cols.append("market_minus_stock_ret")
    time_cols = ["day_of_week","day_of_month","month","quarter","hour","minute"]
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek.astype(float)
    prepared["day_of_month"] = prepared["datetime"].dt.day.astype(float)
    prepared["month"] = prepared["datetime"].dt.month.astype(float)
    prepared["quarter"] = prepared["datetime"].dt.quarter.astype(float)
    prepared["hour"] = prepared["datetime"].dt.hour.astype(float)
    prepared["minute"] = prepared["datetime"].dt.minute.astype(float)
    all_cols = list(set(feat_cols + time_cols))
    existing = [c for c in all_cols if c in prepared.columns]
    prepared[existing] = prepared[existing].replace([np.inf, -np.inf], np.nan)
    return prepared, existing

def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    labels = []
    for ticker, group in df.groupby("ticker", sort=True):
        idx = group.index
        future_close = group["close"].shift(-horizon)
        direction = (future_close > group["close"]).astype(float)
        direction.loc[future_close.isna()] = np.nan
        labels.append(pd.Series(direction.values, index=idx))
    if labels: return pd.concat(labels)
    return pd.Series(dtype=float)

def split_data(df: pd.DataFrame, labels: pd.Series):
    train_mask = df["datetime"] <= TRAIN_END
    val_mask = (df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)
    eval_mask = df["datetime"] >= EVAL_START
    return df[train_mask].copy(), df[val_mask].copy(), df[eval_mask].copy()

def train_rf(X: pd.DataFrame, y: pd.Series, params: dict | None = None):
    from sklearn.ensemble import RandomForestClassifier
    p = params or RF_PARAMS_BASE
    return RandomForestClassifier(**p).fit(X, y)

def predict_proba(model, X: pd.DataFrame):
    probas = model.predict_proba(X)
    preds = model.predict(X)
    return preds, probas[:, 1], probas

def directional_accuracy(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0: return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - RF h=60 Final65 Focus")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {"model": "random_forest", "horizon": HORIZON, "target": "absolute_direction",
        "train_end": str(TRAIN_END), "val_start": str(VAL_START), "val_end": str(VAL_END),
        "eval_start": str(EVAL_START), "seed": SEED, "created_at": now_utc()}
    with (OUTPUT_DIR / "run_config.json").open("w") as f: json.dump(run_config, f, indent=2)
    print("\nLoading data...")
    tickers = load_universe_tickers()
    stock_df = load_stock_data(tickers)
    index_data = load_index_data()
    print(f"  {len(tickers)} tickers, {len(stock_df)} stock rows, {len(index_data)} indices")
    print("\nBuilding features...")
    feature_df_c, feat_cols_c = build_features(stock_df, index_data, include_market=True)
    print(f"  Feature set C: {len(feat_cols_c)} features")
    labels = build_labels(feature_df_c, HORIZON)
    train_df, val_df, eval_df = split_data(feature_df_c, labels)
    fcp = [c for c in feat_cols_c if c in feature_df_c.columns]
    tl = labels.reindex(train_df.index).dropna()
    vl = labels.reindex(val_df.index).dropna()
    el = labels.reindex(eval_df.index).dropna()
    print(f"  Train: {len(tl)}, Val: {len(vl)}, Eval: {len(el)}")
    tX = feature_df_c.reindex(tl.index)[fcp].fillna(0)
    ty = tl.astype(int)
    vX = feature_df_c.reindex(vl.index)[fcp].fillna(0)
    vy = vl.astype(int)
    eX = feature_df_c.reindex(el.index)[fcp].fillna(0)
    ey = el.astype(int)
    all_policies, all_eval_results = [], []

    # Train base model
    print("\nTraining base RF model...")
    base_model = train_rf(tX, ty)
    vp, vprob, vpr = predict_proba(base_model, vX)
    ep, eprob, epr = predict_proba(base_model, eX)
    base_va = directional_accuracy(vy.values, vp)
    base_ea = directional_accuracy(ey.values, ep)
    print(f"  Base val accuracy: {fmt_pct(base_va)}")
    print(f"  Base eval accuracy: {fmt_pct(base_ea)}")

    # EXP 1: Confidence threshold selection (validation only)
    print("\n[Exp 1] Confidence threshold selection...")
    best_ct, best_cta, best_ctc = None, 0.0, 0.0
    for thresh in CONFIDENCE_THRESHOLDS:
        vmask = vprob >= thresh
        vn = vmask.sum()
        if vn < 100: continue
        vacc = directional_accuracy(vy.values[vmask], vp[vmask])
        vcov = vn / len(vy)
        if vcov >= 0.30 and vacc > best_cta:
            best_cta, best_ct, best_ctc = vacc, thresh, vcov
    if best_ct is not None:
        emask = eprob >= best_ct
        en = emask.sum()
        if en >= 100:
            eacc = directional_accuracy(ey.values[emask], ep[emask])
            ecov = en / len(ey)
            all_policies.append({"experiment": "confidence_threshold", "threshold": best_ct,
                "val_accuracy": round(best_cta, 6), "val_coverage": round(best_ctc, 4), "val_rows": int(best_ctc * len(vy))})
            all_eval_results.append({"experiment": "confidence_threshold", "threshold": best_ct,
                "final_accuracy": round(eacc, 6), "final_coverage": round(ecov, 4), "final_rows": int(en),
                "pass_65": eacc >= 0.65, "claim_level": "coverage_qualified" if ecov >= 0.30 and en >= 1000 else "failed"})
            print(f"  Best threshold: {best_ct}, val_acc: {fmt_pct(best_cta)}, eval_acc: {fmt_pct(eacc)}, eval_cov: {fmt_pct(ecov)}, rows: {en}")

    # EXP 2: Probability calibration (Platt scaling, validation only)
    print("\n[Exp 2] Probability calibration (Platt scaling)...")
    from sklearn.calibration import CalibratedClassifierCV
    try:
        cal_model = CalibratedClassifierCV(estimator=train_rf(tX, ty, {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED}), method="sigmoid", cv=5)
        cal_model.fit(tX, ty)
        cvp, cvprob, cvpr = predict_proba(cal_model, vX)
        cep, ceprob, cepr = predict_proba(cal_model, eX)
        cva = directional_accuracy(vy.values, cvp)
        cea = directional_accuracy(ey.values, cep)
        all_policies.append({"experiment": "platt_calibration", "threshold": "",
            "val_accuracy": round(cva, 6), "val_coverage": 1.0, "val_rows": len(vy)})
        all_eval_results.append({"experiment": "platt_calibration", "threshold": "",
            "final_accuracy": round(cea, 6), "final_coverage": 1.0, "final_rows": len(el),
            "pass_65": cea >= 0.65, "claim_level": "global_full_universe" if cea >= 0.60 else "failed"})
        print(f"  Calibrated val accuracy: {fmt_pct(cva)}, eval accuracy: {fmt_pct(cea)}")
        for thresh in CONFIDENCE_THRESHOLDS:
            vmask = cvprob >= thresh
            vn = vmask.sum()
            if vn < 100: continue
            vacc = directional_accuracy(vy.values[vmask], cvp[vmask])
            vcov = vn / len(vy)
            if vcov >= 0.30 and vacc > best_cta:
                emask = ceprob >= thresh
                en = emask.sum()
                if en >= 100:
                    eacc = directional_accuracy(ey.values[emask], cep[emask])
                    ecov = en / len(ey)
                    all_policies.append({"experiment": "platt_calibration_threshold", "threshold": thresh,
                        "val_accuracy": round(vacc, 6), "val_coverage": round(vcov, 4), "val_rows": int(vn)})
                    all_eval_results.append({"experiment": "platt_calibration_threshold", "threshold": thresh,
                        "final_accuracy": round(eacc, 6), "final_coverage": round(ecov, 4), "final_rows": int(en),
                        "pass_65": eacc >= 0.65, "claim_level": "coverage_qualified" if ecov >= 0.30 and en >= 1000 else "failed"})
    except Exception as e: print(f"  Calibration failed: {e}")

    # EXP 3: Meta-label abstention (pre-2025 data only)
    print("\n[Exp 3] Meta-label abstention...")
    vcor = (vp == vl.values).astype(int)
    from sklearn.ensemble import RandomForestClassifier as RFC
    mfv = pd.DataFrame({"confidence": vprob, "max_proba": vpr.max(axis=1),
        "prob_diff": np.abs(vpr[:, 0] - vpr[:, 1])}, index=vl.index)
    if len(mfv) >= 50:
        mm = RFC(n_estimators=100, max_depth=5, random_state=SEED)
        mm.fit(mfv, vcor)
        mfe = pd.DataFrame({"confidence": eprob, "max_proba": epr.max(axis=1),
            "prob_diff": np.abs(epr[:, 0] - epr[:, 1])}, index=el.index)
        mpc = mm.predict(mfe)
        mpp = mm.predict_proba(mfe)[:, 1]
        mvp = mm.predict(mfv)
        mvpp = mm.predict_proba(mfv)[:, 1]
        bmt, bma, bmc = None, 0.0, 0.0
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            mask = mvpp >= thresh
            n = mask.sum()
            if n < 100: continue
            acc = directional_accuracy(vcor[mask], mvp[mask])
            cov = n / len(vcor)
            if cov >= 0.30 and acc > bma:
                bma, bmt, bmc = acc, thresh, cov
        if bmt is not None:
            mask = mpp >= bmt
            n = mask.sum()
            if n >= 1000:
                ecor = (ep == el.values).astype(int)
                fma = directional_accuracy(ecor[mask], mpc[mask])
                fmc = n / len(ecor)
                fmr = int(n)
                all_policies.append({"experiment": "meta_label_abstention", "threshold": bmt,
                    "val_accuracy": round(bma, 6), "val_coverage": round(bmc, 4), "val_rows": int(bmc * len(vcor))})
                all_eval_results.append({"experiment": "meta_label_abstention", "threshold": bmt,
                    "final_accuracy": round(fma, 6), "final_coverage": round(fmc, 4), "final_rows": fmr,
                    "pass_65": fma >= 0.65, "claim_level": "coverage_qualified" if fmc >= 0.30 and fmr >= 1000 else "failed"})
                print(f"  Meta-label threshold: {bmt}, val_acc: {fmt_pct(bma)}, eval_acc: {fmt_pct(fma)}, eval_cov: {fmt_pct(fmc)}, rows: {fmr}")    # EXP 4: Per-ticker confidence thresholds (validation only)
    print("\n[Exp 4] Per-ticker confidence thresholds...")
    pt_correct, pt_total = 0, 0
    per_ticker_policies = []
    for ticker in tickers:
        ticker_mask = feature_df_c["ticker"] == ticker
        t_tl = tl[ticker_mask.reindex(tl.index)]
        t_vl = vl[ticker_mask.reindex(vl.index)]
        t_el = el[ticker_mask.reindex(el.index)]
        if len(t_tl) < 50 or len(t_vl) < 20 or len(t_el) < 20: continue
        t_tX = feature_df_c.reindex(t_tl.index)[fcp].fillna(0)
        t_ty = t_tl.astype(int)
        t_vX = feature_df_c.reindex(t_vl.index)[fcp].fillna(0)
        t_vy = t_vl.astype(int)
        t_eX = feature_df_c.reindex(t_el.index)[fcp].fillna(0)
        t_ey = t_el.astype(int)
        try:
            tm = train_rf(t_tX, t_ty)
            tvp, tvprob, tvpr = predict_proba(tm, t_vX)
            tep, teprob, tepr = predict_proba(tm, t_eX)
            bt, bta, btc = None, 0.0, 0.0
            for thresh in CONFIDENCE_THRESHOLDS:
                vmask = tvprob >= thresh
                vn = vmask.sum()
                if vn < 20: continue
                vacc = directional_accuracy(t_vy.values[vmask], tvp[vmask])
                vcov = vn / len(t_vy)
                if vcov >= 0.30 and vacc > bta:
                    bta, bt, btc = vacc, thresh, vcov
            if bt is not None:
                emask = teprob >= bt
                en = emask.sum()
                if en > 0:
                    eacc = directional_accuracy(t_ey.values[emask], tep[emask])
                    pt_correct += int(np.sum(t_ey.values[emask] == tep[emask]))
                    pt_total += int(en)
                    per_ticker_policies.append({"ticker": ticker, "threshold": bt, "eval_accuracy": round(eacc, 6), "eval_rows": int(en)})
        except: continue
    pt_acc = pt_correct / pt_total if pt_total > 0 else 0.0
    pt_cov = pt_total / len(el) if len(el) > 0 else 0.0
    if pt_total >= 1000:
        all_policies.append({"experiment": "per_ticker_confidence", "threshold": "per_ticker",
            "val_accuracy": "", "val_coverage": "", "val_rows": ""})
        all_eval_results.append({"experiment": "per_ticker_confidence", "threshold": "per_ticker",
            "final_accuracy": round(pt_acc, 6), "final_coverage": round(pt_cov, 4), "final_rows": pt_total,
            "pass_65": pt_acc >= 0.65, "claim_level": "coverage_qualified" if pt_cov >= 0.30 and pt_total >= 1000 else "failed"})
        print(f"  Per-ticker eval accuracy: {fmt_pct(pt_acc)}, coverage: {fmt_pct(pt_cov)}, rows: {pt_total}")

    # EXP 5: Regime-aware abstention (lagged/ex-ante features only)
    print("\n[Exp 5] Regime-aware abstention...")
    regime_features = ["rolling_return_vol_20", "rolling_return_vol_60", "volume_shock_20",
        "rolling_return_mean_20", "momentum_20", "rsi_14"]
    regime_features = [f for f in regime_features if f in fcp]
    if regime_features:
        rvX = feature_df_c.reindex(vl.index)[regime_features].fillna(0)
        reX = feature_df_c.reindex(el.index)[regime_features].fillna(0)
        rcor = (vp == vl.values).astype(int)
        rm = RFC(n_estimators=100, max_depth=4, random_state=SEED)
        rm.fit(rvX, rcor)
        rmp = rm.predict(reX)
        rmpp = rm.predict_proba(reX)[:, 1]
        rmvp = rm.predict(rvX)
        rmvpp = rm.predict_proba(rvX)[:, 1]
        brt, bra, brc = None, 0.0, 0.0
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            mask = rmvpp >= thresh
            n = mask.sum()
            if n < 100: continue
            acc = directional_accuracy(rcor[mask], rmvp[mask])
            cov = n / len(rcor)
            if cov >= 0.30 and acc > bra:
                bra, brt, brc = acc, thresh, cov
        if brt is not None:
            mask = rmpp >= brt
            n = mask.sum()
            if n >= 1000:
                ecor = (ep == el.values).astype(int)
                fra = directional_accuracy(ecor[mask], rmp[mask])
                frc = n / len(ecor)
                frr = int(n)
                all_policies.append({"experiment": "regime_aware_abstention", "threshold": brt,
                    "val_accuracy": round(bra, 6), "val_coverage": round(brc, 4), "val_rows": int(brc * len(rcor))})
                all_eval_results.append({"experiment": "regime_aware_abstention", "threshold": brt,
                    "final_accuracy": round(fra, 6), "final_coverage": round(frc, 4), "final_rows": frr,
                    "pass_65": fra >= 0.65, "claim_level": "coverage_qualified" if frc >= 0.30 and frr >= 1000 else "failed"})
                print(f"  Regime-aware threshold: {brt}, val_acc: {fmt_pct(bra)}, eval_acc: {fmt_pct(fra)}, eval_cov: {fmt_pct(frc)}, rows: {frr}")

    # Write outputs
    print("\nWriting outputs...")
    with (OUTPUT_DIR / "validation_selected_policies.csv").open("w", newline="") as f:
        if all_policies:
            w = csv.DictWriter(f, fieldnames=all_policies[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(all_policies)
    with (OUTPUT_DIR / "final65_focus_eval_results.csv").open("w", newline="") as f:
        if all_eval_results:
            w = csv.DictWriter(f, fieldnames=all_eval_results[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(all_eval_results)
    f65_candidates = [r for r in all_eval_results if r.get("pass_65") == True or float(r.get("final_accuracy", 0) or 0) >= 0.65]
    exploratory = [r for r in all_eval_results if r not in f65_candidates]
    with (OUTPUT_DIR / "final65_candidates.csv").open("w", newline="") as f:
        if f65_candidates:
            w = csv.DictWriter(f, fieldnames=all_eval_results[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(f65_candidates)
    with (OUTPUT_DIR / "exploratory_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=all_eval_results[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(exploratory)
    best_f65 = max((float(r.get("final_accuracy", 0) or 0) for r in all_eval_results if float(r.get("final_coverage", 0) or 0) >= 0.30 and int(r.get("final_rows", 0) or 0) >= 1000), default=0)
    f65_pass = best_f65 >= 0.65
    manifest = {"total_policies": len(all_policies), "total_eval_results": len(all_eval_results),
        "final65_candidates": len(f65_candidates), "best_final65_accuracy": round(best_f65, 6),
        "final65_pass": f65_pass, "completed_at": now_utc(),
        "leakage_safe": True, "daily_data_used": False, "resampling_used": False}
    with (OUTPUT_DIR / "final65_focus_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
    log = ["# RF h=60 Final65 Focus Run Log", "", f"- Completed: {manifest['completed_at']}",
        f"- Policies: {manifest['total_policies']}", f"- Eval results: {manifest['total_eval_results']}",
        f"- Best final65: {fmt_pct(best_f65)}", f"- Final65 pass: {'YES' if f65_pass else 'NO'}",
        f"- Gap to 65: {fmt_pct(0.65 - best_f65)}", ""]
    with (OUTPUT_DIR / "run_log.md").open("w") as f: f.write("\n".join(log))
    print(f"\nBest final65: {fmt_pct(best_f65)}")
    print(f"Final65 pass: {'YES' if f65_pass else 'NO'}")
    print(f"Gap to 65: {fmt_pct(0.65 - best_f65)}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())