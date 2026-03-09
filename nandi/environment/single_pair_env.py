"""
Single-pair Forex Trading Environment for RL training.

Action space: continuous [-1, 1] representing target position.
Reward: Anti-fragile risk-adjusted return with asymmetric loss penalty.

Supports D1 (daily) and M5 (scalping) timeframes via TIMEFRAME_PROFILES.
"""

import numpy as np
import logging

from nandi.config import (
    INITIAL_BALANCE, TRANSACTION_COST_BPS, LEVERAGE,
    MAX_POSITION, RISK_LIMITS, RISK_LIMITS_M5, LOOKBACK_WINDOW,
    TIMEFRAME_PROFILES,
)
from nandi.environment.rewards import CompositeReward

logger = logging.getLogger(__name__)


class ForexTradingEnv:
    """Forex trading environment for reinforcement learning."""

    # Hard per-bar loss cap as fraction of equity — no single bar can take more than this
    MAX_BAR_LOSS_FRAC = 0.05  # 5% of equity max per bar

    # Margin call: stop trading if equity drops below this fraction of initial balance
    MARGIN_CALL_FRAC = 0.20  # 20% of initial = stop (80% loss)

    # Cooldown bars after session loss trigger — force flat during cooldown
    SESSION_LOSS_COOLDOWN = {"D1": 1, "M5": 12}  # D1: 1 bar, M5: 12 bars (~1 hour)

    def __init__(self, features, prices, lookback=LOOKBACK_WINDOW,
                 initial_balance=INITIAL_BALANCE, pair_name="unknown",
                 market_sim=None, use_composite_reward=True, timeframe="D1"):
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.pair_name = pair_name
        self.market_sim = market_sim
        self.use_composite_reward = use_composite_reward
        self.timeframe = timeframe
        self.n_features = features.shape[1]

        # Load profile
        self.profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["D1"])
        self.leverage = self.profile["leverage"]
        self.max_position = self.profile["max_position"]
        self.bars_per_session = self.profile["bars_per_session"]
        self.max_hold_bars = self.profile.get("max_hold_bars", 0)

        # Timeframe-specific risk limits: M5 gets tighter limits
        self.risk_limits = RISK_LIMITS_M5 if timeframe != "D1" else RISK_LIMITS

        # Session loss cooldown period
        self.cooldown_bars = self.SESSION_LOSS_COOLDOWN.get(timeframe, 1)

        self.composite_reward = CompositeReward(timeframe=timeframe) if use_composite_reward else None

        self.market_state_shape = (lookback, self.n_features)
        self.position_info_dim = 4

        self.reset()

    def reset(self, start_idx=None):
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
        self.session_start_equity = self.initial_balance
        self.session_step = 0  # steps within current session
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.done = False

        # Trade tracking for accurate win rate
        self.entry_price = 0.0
        self.trade_direction = 0.0  # +1 long, -1 short, 0 flat
        self.trade_entry_equity = self.initial_balance
        self.bars_in_trade = 0  # how long current trade has been held

        # Risk: cooldown after session loss trigger
        self.cooldown_remaining = 0

        # Reset reward components
        if self.composite_reward is not None:
            self.composite_reward.reset()

        return self._get_state()

    def step(self, action):
        action = float(np.clip(action, -self.max_position, self.max_position))

        # ── RISK GATE 1: Cooldown forces flat ──
        # After session loss trigger, agent must stay flat for cooldown_bars
        if self.cooldown_remaining > 0:
            action = 0.0
            self.cooldown_remaining -= 1

        # ── RISK GATE 2: Pre-trade DD budget check ──
        # If near DD limit, scale down action so max possible loss stays within budget
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        dd_budget = self.risk_limits["max_drawdown"] - current_dd
        if dd_budget < self.MAX_BAR_LOSS_FRAC and dd_budget > 0:
            # Near limit — scale action proportionally
            scale = dd_budget / self.MAX_BAR_LOSS_FRAC
            action *= scale
        elif dd_budget <= 0:
            # Already at or past limit — force flat and stop
            self.position = 0.0
            self.done = True
            return self._get_state(), -1.0, True, self._get_info()

        price_now = self.prices[self.current_idx]
        self.current_idx += 1
        self.session_step += 1

        if self.current_idx >= len(self.prices) - 1:
            self.done = True
            return self._get_state(), 0.0, True, self._get_info()

        price_next = self.prices[self.current_idx]
        market_ret = (price_next - price_now) / price_now

        # ── PnL from CURRENT position (before update) ──
        position_pnl = self.position * market_ret * self.equity * self.leverage

        # ── RISK GATE 3: Dynamic per-bar loss cap ──
        # Cap = min(fixed 5% of equity, remaining DD budget in $ terms)
        # This guarantees DD can never overshoot the configured limit
        dd_floor_equity = self.peak_equity * (1.0 - self.risk_limits["max_drawdown"])
        remaining_budget_dollars = max(0.0, self.equity - dd_floor_equity)
        fixed_cap = self.equity * self.MAX_BAR_LOSS_FRAC
        max_loss = min(fixed_cap, remaining_budget_dollars)

        if position_pnl < -max_loss:
            position_pnl = -max_loss

        # Transaction cost (capped so cost alone can't push past DD limit)
        position_change = abs(action - self.position)
        if self.market_sim is not None:
            cost = self.market_sim.get_total_cost(
                price_now, self.position, action, abs(market_ret)
            ) * self.equity
        else:
            cost = position_change * self.equity * TRANSACTION_COST_BPS / 10000
        # Cap cost so pnl - cost doesn't exceed remaining budget
        if position_pnl - cost < -remaining_budget_dollars:
            cost = min(cost, max(0.0, remaining_budget_dollars + position_pnl))

        # Accurate trade win/loss tracking
        if position_change > 0.05:
            # Closing the previous trade
            if abs(self.trade_direction) > 0.05:
                self.total_trades += 1
                realized_pnl = self.equity - self.trade_entry_equity
                if realized_pnl > 0:
                    self.winning_trades += 1
            # Opening a new trade
            if abs(action) > 0.05:
                self.entry_price = price_now
                self.trade_direction = 1.0 if action > 0 else -1.0
                self.trade_entry_equity = self.equity
                self.bars_in_trade = 0
            else:
                self.trade_direction = 0.0
                self.bars_in_trade = 0

        # Track bars in trade
        if abs(self.position) > 0.05:
            self.bars_in_trade += 1

        # ── RISK GATE 4: Equity floor at zero — never go negative ──
        new_equity = self.equity + position_pnl - cost
        if new_equity <= 0:
            self.equity = 0.0
            self.total_pnl = -(self.initial_balance)
            self.position = 0.0
            self.done = True
            return self._get_state(), -1.0, True, self._get_info()

        self.equity = new_equity
        self.total_pnl += position_pnl - cost
        self.position = action

        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # ── RISK GATE 5: Hard drawdown stop (should never overshoot now) ──
        if drawdown >= self.risk_limits["max_drawdown"]:
            self.position = 0.0
            self.done = True

        # ── RISK GATE 6: Margin call — stop at 80% loss of initial balance ──
        if self.equity < self.initial_balance * self.MARGIN_CALL_FRAC:
            self.position = 0.0
            self.done = True

        # ── RISK GATE 7: Session loss limit with cooldown ──
        if self.session_start_equity > 0:
            session_ret = (self.equity - self.session_start_equity) / self.session_start_equity
            if session_ret < -self.risk_limits["max_daily_loss"]:
                self.position = 0.0
                self.cooldown_remaining = self.cooldown_bars

        if self.session_step >= self.bars_per_session:
            self.session_start_equity = self.equity
            self.session_step = 0

        # ── RISK GATE 8: Scale down at elevated drawdown ──
        if drawdown > self.risk_limits["scale_down_threshold"]:
            self.position *= 0.5

        if self.composite_reward is not None:
            reward = self.composite_reward.compute(
                position_pnl, cost, self.equity, drawdown, market_ret,
                self.position, bars_in_trade=self.bars_in_trade,
            )
        else:
            reward = self._compute_reward(position_pnl, cost, drawdown, market_ret)
        return self._get_state(), reward, self.done, self._get_info()

    def _get_state(self):
        start = max(0, self.current_idx - self.lookback)
        market_state = self.features[start:self.current_idx]

        if market_state.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - market_state.shape[0], self.n_features))
            market_state = np.vstack([pad, market_state])

        dd = (self.peak_equity - self.equity) / self.peak_equity

        vol_regime = 1.0
        if self.current_idx < len(self.features) and self.current_idx > 10:
            vol_regime = float(np.std(
                self.features[max(0, self.current_idx - 10):self.current_idx, 0]
            ))

        position_info = np.array([
            self.position,
            (self.equity / self.initial_balance) - 1.0,
            dd,
            vol_regime,
        ], dtype=np.float32)

        return market_state.astype(np.float32), position_info

    def _compute_reward(self, pnl, cost, drawdown, market_return):
        ret = (pnl - cost) / self.initial_balance / self.leverage

        if ret < 0:
            ret *= 2.0

        dd_penalty = 0.0
        if drawdown > 0.05:
            dd_penalty = (drawdown - 0.05) * 0.1

        recent_vol = abs(market_return)
        af_bonus = 0.0
        if ret > 0 and recent_vol > 0.005:
            af_bonus = ret * 0.5

        # Inactivity penalty for M5 — agent must trade to learn
        activity_bonus = 0.0
        if self.timeframe != "D1":
            if abs(self.position) > 0.05:
                activity_bonus = 0.0005  # small reward for being in a trade
            else:
                activity_bonus = -0.0002  # small penalty for sitting flat
        else:
            # D1: patience is fine
            if abs(self.position) < 0.05 and abs(market_return) < 0.002:
                activity_bonus = 0.001

        reward = ret - dd_penalty + af_bonus + activity_bonus
        return float(np.clip(reward, -1.0, 1.0))

    def _get_info(self):
        return {
            "pair": self.pair_name,
            "equity": self.equity,
            "position": self.position,
            "price": self.prices[min(self.current_idx, len(self.prices) - 1)],
            "drawdown": (self.peak_equity - self.equity) / self.peak_equity,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "return_pct": (self.equity / self.initial_balance - 1) * 100,
            "bars_in_trade": self.bars_in_trade,
            "timeframe": self.timeframe,
        }


class MultiEpisodeEnv:
    """Wraps ForexTradingEnv to generate diverse training episodes."""

    def __init__(self, features, prices, lookback=LOOKBACK_WINDOW,
                 episode_length=252, pair_name="unknown", market_sim=None,
                 timeframe="D1"):
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.episode_length = episode_length
        self.env = ForexTradingEnv(
            features, prices, lookback, pair_name=pair_name,
            market_sim=market_sim, timeframe=timeframe,
        )

    def reset(self):
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
