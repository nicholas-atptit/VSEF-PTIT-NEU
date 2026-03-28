import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List

class PortfolioVn30Env(gym.Env):
    """
    Môi trường RL phân bổ danh mục (Weight-centric).
    Phù hợp cho VN30. 
    Action: Target Weights [% cho từng mã].
    State: [Giá, Volume, MACD, RSI] x N_assets.
    Reward: PnL / Volatility (Tương tự Sortino phi tuyến tính).
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        df: pd.DataFrame, 
        tickers: List[str], 
        initial_amount: float = 1_000_000_000, 
        lookback: int = 30
    ):
        super(PortfolioVn30Env, self).__init__()
        self.df = df
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.initial_amount = initial_amount
        self.lookback = lookback
        
        # State: 4 features per asset (VD: Close, Vol, MACD, RSI)
        self.features_per_asset = 4 
        self.obs_shape = self.num_assets * self.features_per_asset
        
        # Chặn action output bằng Softmax wrapper bên ngoài, hoặc cho phép [0, 1] rồi chuẩn hóa.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation space chuẩn State representation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

        # Trạng thái game
        self.current_step = self.lookback
        self.portfolio_value = self.initial_amount
        self.portfolio_history = []
        
        # Biến giả lập data index
        self._max_step = len(self.df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.portfolio_value = self.initial_amount
        self.portfolio_history = [self.initial_amount]
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Mock Observation Vector: Ở môi trường thật sẽ query data tại self.current_step
        # V1: Mock Random cho Proof of Concept
        return np.random.randn(self.obs_shape).astype(np.float32)

    def step(self, action: np.ndarray):
        # 1. Action là Weight
        weights = action / (np.sum(action) + 1e-8) # Chuẩn hóa sao cho sum = 1
        
        # 2. Lấy Daily Return của N asset (Mock Data)
        # Bản chuẩn: return = df.iloc[self.current_step][['close_A', 'close_B']].pct_change()
        daily_returns = np.random.normal(0.0005, 0.02, size=self.num_assets)
        
        # 3. Tính PnL ngày hôm nay (Bỏ qua phí giao dịch như user yêu cầu)
        portfolio_return = np.sum(weights * daily_returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # 4. Tính Reward = Return ngày hiện tại trừ rủi ro (hoặc dùng hàm tiện ích log)
        # Để cho nhanh hội tụ PnL, chích Sortino/Sharpe Reward
        reward = float(portfolio_return * 100) # Hệ số khuếch đại cho model nhạy cảm
        
        self.current_step += 1
        done = self.current_step >= self._max_step
        
        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return": portfolio_return
        }
        
        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Portfolio Value: {self.portfolio_value:.2f}")

