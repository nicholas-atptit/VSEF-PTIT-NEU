import os
import sys
# Add project root to sys.path so 'src' can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.training_pipeline.rl_env_portfolio import PortfolioVn30Env

def main():
    print("=== Khởi tạo Baseline PPO (Stable-Baselines3) cho VN30 ===")
    
    # 1. Khởi tạo Data Mock (Bản thật sẽ load từ CSV hoặc DB)
    # 1000 ngày, 3 mã cổ phiếu (VN30 POC: SSI, HPG, FPT)
    days = 1000
    tickers = ['SSI', 'HPG', 'FPT']
    print(f"Bắt đầu thiết lập môi trường cho {len(tickers)} mã: {tickers} với lịch sử {days} ngày.")
    
    mock_df = pd.DataFrame({'fake_col': range(days)}) # Placeholder dataframe
    
    # 2. Khởi tạo Gym Environment
    env = PortfolioVn30Env(df=mock_df, tickers=tickers, initial_amount=1_000_000_000, lookback=30)
    
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
