"""Experimental placeholder module.
Not part of canonical governed runtime.
"""

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
        self.peak_value = self.initial_amount
        self.previous_weights = np.zeros(self.num_assets, dtype=np.float32)
        
        # Penalty constants
        self.trading_fee_rate = 0.001  # 0.1% per trade
        self.drawdown_penalty_weight = 0.5
        self.turnover_penalty_weight = 0.1
        
        # Biến giả lập data index
        self._max_step = len(self.df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.portfolio_value = self.initial_amount
        self.portfolio_history = [self.initial_amount]
        self.peak_value = self.initial_amount
        self.previous_weights = np.zeros(self.num_assets, dtype=np.float32)
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # V2: Thực tế bóc tách dữ liệu từ self.df tại self.current_step
        # Giả định df có các cột dạng {ticker}_close, {ticker}_volume...
        # Hoặc ít nhất df đã là feature space chuẩn (1 hàng = obs_shape items)
        
        # Lấy dòng hiện tại theo lookback
        row_data = self.df.iloc[self.current_step - self.lookback : self.current_step]
        # Nếu truyền vào DF đã được feature extraction (vd 12 columns cho 3 mã),
        # ta flatten nó ra làm mảng 1D State. 
        # Trong bản Poc ta dùng the latest row features hoặc random nếu không đủ cols
        if len(self.df.columns) >= self.obs_shape:
            # Chọn các cột feature
            obs = self.df.iloc[self.current_step].values[:self.obs_shape]
            return obs.astype(np.float32)
        else:
            # Fallback mock features nếu chưa map feature (Tránh crash Env)
            return np.random.randn(self.obs_shape).astype(np.float32)

    def step(self, action: np.ndarray):
        # 1. Action là Weight
        weights = action / (np.sum(action) + 1e-8) # Chuẩn hóa sao cho sum = 1
        
        # 2. Lấy Daily Return của N asset (Real Data lookup)
        # Giả sử self.df có cột 'return_SSI', 'return_HPG', 'return_FPT'
        # Hoặc 'close_SSI' để tự tính. Thống nhất dùng '{ticker}_return'
        daily_returns = np.zeros(self.num_assets)
        for i, ticker in enumerate(self.tickers):
            col_name = f"{ticker}_return"
            if col_name in self.df.columns:
                daily_returns[i] = self.df.iloc[self.current_step][col_name]
            else:
                 daily_returns[i] = np.random.normal(0.0005, 0.02) # Dự phòng
        
        # 3. Tính Turnover (sự thay đổi weights)
        turnover = np.sum(np.abs(weights - self.previous_weights))
        self.previous_weights = weights.copy()
        
        # 4. Tính PnL ngày hôm nay (Fee adjusted)
        gross_return = np.sum(weights * daily_returns)
        fee_cost = turnover * self.trading_fee_rate
        net_return = gross_return - fee_cost
        
        self.portfolio_value *= (1 + net_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Update peak value
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        # 5. Drawdown Penalty
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        drawdown_penalty = drawdown * self.drawdown_penalty_weight
        
        # Turnover Penalty modifier 
        turnover_penalty = turnover * self.turnover_penalty_weight
        
        # 6. Final Reward
        # reward = fee-adjusted return - drawdown penalty - turnover penalty
        reward = float(net_return * 100 - drawdown_penalty - turnover_penalty)
        
        self.current_step += 1
        done = self.current_step >= self._max_step
        
        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return": net_return,
            "turnover": turnover,
            "drawdown": drawdown
        }
        
        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Portfolio Value: {self.portfolio_value:.2f}")

