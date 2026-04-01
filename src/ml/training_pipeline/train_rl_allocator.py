import os
import sys
# Add project root to sys.path so 'src' can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import warnings

# Tắt cảnh báo numpy trong SB3
warnings.filterwarnings("ignore")

try:
    from src.ml.data_loader import load_ohlcv_from_db
except ImportError:
    # Fallback mock if data_loader missing
    load_ohlcv_from_db = None

from src.ml.training_pipeline.rl_env_portfolio import PortfolioVn30Env

def main():
    print("=== Khởi tạo Baseline PPO (Stable-Baselines3) cho VN30 ===")
    
    # 1. Load Data từ Database qua src.ml.data_loader
    vn30 = ["ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"]
    viettel = ["VGI", "VTK", "VTP", "CTR"]
    vn100_extra = ["VND", "VCI", "HCM", "DIG", "DXG", "KBC", "NLG", "KDH", "PNJ", "REE", "DGC", "DPM", "DCM", "HSG", "NKG", "VPI", "GEX", "HDG", "VHC", "ANV", "FRT", "DGW", "SBT", "HAH", "PVT", "PVS", "PVD", "BSR", "TCH", "HAG", "ASM", "LCG", "HHV", "VCG", "FCN", "CII"]
    
    # Gom danh sách (Xóa trùng type list(set(...))
    tickers = sorted(list(set(vn30 + viettel + vn100_extra)))
    print(f"Bắt đầu trích xuất dữ liệu lịch sử cho {len(tickers)} mã: {tickers}.")
    
    # Tạo DataFrame tổng
    merged_df = None
    
    if load_ohlcv_from_db is not None:
        for ticker in tickers:
            df_t = load_ohlcv_from_db(ticker)
            if df_t is not None and not df_t.empty:
                df_t = df_t[['close', 'volume']].copy()
                df_t.columns = [f"{ticker}_close", f"{ticker}_volume"]
                # Tính return thực tế
                df_t[f"{ticker}_return"] = df_t[f"{ticker}_close"].pct_change().fillna(0)
                
                # Mock indicators to pad features (Giả lập để đủ 4 features / 1 asset)
                df_t[f"{ticker}_macd"] = np.random.randn(len(df_t))
                df_t[f"{ticker}_rsi"] = np.random.uniform(30, 70, len(df_t))
                
                if merged_df is None:
                    merged_df = df_t
                else:
                    merged_df = merged_df.join(df_t, how='outer')
                    
        if merged_df is not None:
            merged_df = merged_df.ffill().fillna(0)
    
    if merged_df is None or merged_df.empty or len(merged_df) < 50:
        print("[WARNING] Không đủ dữ liệu từ DB. Fallback về Mock Data (POC).")
        days = 1000
        merged_df = pd.DataFrame(index=range(days))
        for ticker in tickers:
            prices = np.random.normal(100, 2, days).cumprod()
            merged_df[f"{ticker}_close"] = prices
            merged_df[f"{ticker}_return"] = pd.Series(prices).pct_change().fillna(0).values
            merged_df[f"{ticker}_volume"] = np.random.randint(10_000, 1_000_000, days)
            merged_df[f"{ticker}_macd"] = np.random.randn(days)
            merged_df[f"{ticker}_rsi"] = np.random.uniform(30, 70, days)
    
    # 2. Khởi tạo Gym Environment
    # Reset index để step bằng số nguyên
    env_df = merged_df.reset_index(drop=True)
    env = PortfolioVn30Env(df=env_df, tickers=tickers, initial_amount=1_000_000_000, lookback=30)
    
    # Kích hoạt hàm kiểm tra Env xem có chuẩn Gymnasium Interface không 
    # (Fix các lỗi numpy dtypes bị lệch)
    try:
        check_env(env, warn=True)
        print("Môi trường Gym hợp lệ (Pass gymnasium check)!")
    except Exception as e:
        print(f"Lỗi khởi tạo Gym Env: {e}")
        return

    # 3. Tạo mô hình (MLP Policy)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    
    # 4. Huấn luyện (Giả lập Train 10,000 timesteps)
    print("Bắt đầu huấn luyện PPO Model (10,000 timesteps)...")
    model.learn(total_timesteps=10000)
    
    # 5. Lưu mô hình (Weight-centric allocator)
    output_dir = "models/"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "ppo_vn30_poc")
    model.save(model_path)
    
    print(f"\n[HOÀN TẤT] Trọng số mô hình đã lưu tại: {model_path}.zip")
    print("Bạn có thể gọi lại mô hình này thông qua DRLAllocator trong benchmark.")

if __name__ == "__main__":
    main()
