"""OpenAI Gym Environment for Stock Portfolio Optimization (RL).

Implements the VN stock market simulation with a Sortino Ratio 
reward function and a hard-capped Action Space (0.70).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockPortfolioEnv(gym.Env):
    """
    A custom environment for reinforcement learning based on Sortino Ratio.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, max_risk_cap=0.70):
        super(StockPortfolioEnv, self).__init__()
        self.data = data
        self.max_risk_cap = max_risk_cap
        self.current_step = 0
        
        # State Space: [Balance, Holdings, TFT_Q10, TFT_Q50, TFT_Q90, Sentiment]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action Space: Allocation Percentage [0.0 to 1.0]
        # Wrapper will cap this at 0.70
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.returns_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.returns_history = []
        # Initial state: [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        # 1. Apply Hard Constraint (Action Space Wrapper Logic)
        allocation_pct = action[0] * self.max_risk_cap
        
        # 2. Simulate Market Movement (Simplified)
        # In real case, fetch from self.data[self.current_step]
        daily_return = np.random.normal(0.0005, 0.02) # Mock return
        portfolio_return = allocation_pct * daily_return
        self.returns_history.append(portfolio_return)

        # 3. Calculate Reward (Sortino Ratio)
        reward = self._calculate_sortino_reward()
        
        # 4. Advance Step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # New State Observation
        obs = np.random.randn(6).astype(np.float32)
        
        return obs, reward, done, False, {}

    def _calculate_sortino_reward(self):
        """
        Sortino Ratio = (R_p - R_f) / Downside Deviation
        Only penalizes negative volatility.
        """
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        # Downside returns (returns below 0)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) < 2:
            return mean_return # Minimal penalty if no downside yet
        
        downside_deviation = np.std(downside_returns)
        sortino = mean_return / (downside_deviation + 1e-6)
        
        # Scale reward to stabilize PPO
        return float(sortino * 0.1)

if __name__ == "__main__":
    env = StockPortfolioEnv(data=np.zeros(1000))
    print("🚀 RL Gym Environment Loaded. Action Cap: 0.70. Reward: Sortino Ratio.")
