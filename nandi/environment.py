"""
Nandi Trading Environment — Custom RL environment for forex trading.

Action space: continuous [-1, 1] representing target position
  -1 = full short, 0 = flat, +1 = full long

Reward: Anti-fragile risk-adjusted return with asymmetric loss penalty.
"""

import numpy as np
import logging

from nandi.config import (
    INITIAL_BALANCE, TRANSACTION_COST_BPS, LEVERAGE,
    MAX_POSITION, RISK_LIMITS, LOOKBACK_WINDOW,
)

logger = logging.getLogger(__name__)


class ForexTradingEnv:
    """Forex trading environment for reinforcement learning."""

    def __init__(self, features, prices, lookback=LOOKBACK_WINDOW,
                 initial_balance=INITIAL_BALANCE):
        """
        Args:
            features: (N, n_features) scaled feature array
            prices: (N,) close prices
            lookback: number of past days the agent sees
        """
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.n_features = features.shape[1]

        # State dimensions
        self.market_state_shape = (lookback, self.n_features)
        self.position_info_dim = 4  # position, norm_pnl, drawdown, vol_regime

        self.reset()

    def reset(self, start_idx=None):
        """Reset environment. Random start if not specified."""
        max_start = len(self.features) - self.lookback - 2
        if max_start <= self.lookback:
            max_start = self.lookback + 1
        if start_idx is not None:
            self.start_idx = max(self.lookback, min(start_idx, max_start))
        else:
            self.start_idx = np.random.randint(self.lookback, max_start)

        self.current_idx = self.start_idx
        self.position = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.daily_start_equity = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.done = False

        return self._get_state()

    def step(self, action):
        """Execute one trading day.

        Args:
            action: target position in [-1, 1]

        Returns:
            state, reward, done, info
        """
        action = float(np.clip(action, -MAX_POSITION, MAX_POSITION))

        # Current price
        price_now = self.prices[self.current_idx]
        self.current_idx += 1

        if self.current_idx >= len(self.prices) - 1:
            self.done = True
            return self._get_state(), 0.0, True, self._get_info()

        price_next = self.prices[self.current_idx]

        # Market return
        market_ret = (price_next - price_now) / price_now

        # P&L from current position (before changing it)
        position_pnl = self.position * market_ret * self.equity * LEVERAGE

        # Transaction cost from position change
        position_change = abs(action - self.position)
        cost = position_change * self.equity * TRANSACTION_COST_BPS / 10000

        # Track trade
        if position_change > 0.05:  # meaningful position change
            self.total_trades += 1
            if position_pnl > 0:
                self.winning_trades += 1

        # Update equity
        self.equity += position_pnl - cost
        self.total_pnl += position_pnl - cost

        # Update position
        self.position = action

        # Peak equity & drawdown
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # ── Hard risk limits (always enforced, not learned) ──
        if drawdown > RISK_LIMITS["max_drawdown"]:
            self.position = 0.0  # Force flat
            self.done = True

        if self.equity < self.initial_balance * 0.5:
            self.done = True  # Catastrophic loss

        # Daily loss check
        daily_ret = (self.equity - self.daily_start_equity) / self.daily_start_equity
        if daily_ret < -RISK_LIMITS["max_daily_loss"]:
            self.position = 0.0  # Force flat for "rest of day"

        # Scale down at threshold
        if drawdown > RISK_LIMITS["scale_down_threshold"]:
            self.position *= 0.5

        # Compute reward
        reward = self._compute_reward(position_pnl, cost, drawdown, market_ret)

        return self._get_state(), reward, self.done, self._get_info()

    def _get_state(self):
        """Return (market_state, position_info) tuple."""
        start = self.current_idx - self.lookback
        start = max(0, start)
        market_state = self.features[start:self.current_idx]

        # Pad if needed (at start of data)
        if market_state.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - market_state.shape[0], self.n_features))
            market_state = np.vstack([pad, market_state])

        # Drawdown
        dd = (self.peak_equity - self.equity) / self.peak_equity

        # Recent volatility regime (from features if available, else 1.0)
        vol_regime = 1.0
        if self.current_idx < len(self.features):
            # Use vol_ratio feature if it exists (index varies)
            vol_regime = float(np.std(
                self.features[max(0, self.current_idx - 10):self.current_idx, 0]
            )) if self.current_idx > 10 else 1.0

        position_info = np.array([
            self.position,
            (self.equity / self.initial_balance) - 1.0,  # normalized P&L
            dd,
            vol_regime,
        ], dtype=np.float32)

        return market_state.astype(np.float32), position_info

    def _compute_reward(self, pnl, cost, drawdown, market_return):
        """Anti-fragile reward function.

        Key properties:
        1. Asymmetric: losses penalized 2x more than gains rewarded
        2. Drawdown penalty: exponential above 5%
        3. Anti-fragile bonus: extra reward for profits during high volatility
        4. Patience: small reward for staying flat in noisy markets
        5. Clipped to [-1, 1] for training stability
        """
        # Normalized return (divide by equity AND scale down leverage effect)
        ret = (pnl - cost) / self.initial_balance / LEVERAGE

        # Asymmetric scaling: losses hurt more
        if ret < 0:
            ret *= 2.0

        # Drawdown penalty (gentle)
        dd_penalty = 0.0
        if drawdown > 0.05:
            dd_penalty = (drawdown - 0.05) * 0.1

        # Anti-fragile: bonus for profiting during volatile periods
        recent_vol = abs(market_return)
        af_bonus = 0.0
        if ret > 0 and recent_vol > 0.005:
            af_bonus = ret * 0.5

        # Patience: small reward for being flat in low-signal environments
        patience_bonus = 0.0
        if abs(self.position) < 0.05 and abs(market_return) < 0.002:
            patience_bonus = 0.001

        reward = ret - dd_penalty + af_bonus + patience_bonus

        # Clip for training stability
        return float(np.clip(reward, -1.0, 1.0))

    def _get_info(self):
        """Return episode statistics."""
        return {
            "equity": self.equity,
            "position": self.position,
            "drawdown": (self.peak_equity - self.equity) / self.peak_equity,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "return_pct": (self.equity / self.initial_balance - 1) * 100,
        }


class MultiEpisodeEnv:
    """Wraps ForexTradingEnv to generate diverse training episodes."""

    def __init__(self, features, prices, lookback=LOOKBACK_WINDOW,
                 episode_length=252):
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.episode_length = episode_length
        self.env = ForexTradingEnv(features, prices, lookback)

    def reset(self):
        """Reset with random starting point."""
        max_start = len(self.features) - self.episode_length - self.lookback - 1
        if max_start <= self.lookback:
            start = self.lookback
        else:
            start = np.random.randint(self.lookback, max_start)
        self.steps_in_episode = 0
        return self.env.reset(start_idx=start)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.steps_in_episode += 1
        if self.steps_in_episode >= self.episode_length:
            done = True
        return state, reward, done, info

    @property
    def market_state_shape(self):
        return self.env.market_state_shape

    @property
    def position_info_dim(self):
        return self.env.position_info_dim
