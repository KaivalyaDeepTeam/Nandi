"""
SPINTradingEnv — Trading environment with hard-wired risk management.

The "Don't Be the 90%" module: these are ENVIRONMENT CONSTRAINTS, not rewards.
The agent literally cannot make the deadly trading mistakes.

Hard constraints:
  1. ATR-based stop-loss — auto-executed, non-negotiable
  2. Max hold 12 bars (1hr) — force close
  3. Cooldown after SL — 3 bars (6 after 2 consecutive losses)
  4. Min R:R ratio — entry masked if ATR too small
  5. Session loss limit — -1.5%, max 20 trades
  6. Trend filter — only LONG if H1 trend >= 0, SHORT if <= 0
  7. Fixed position size — hard cap
"""

import logging

import numpy as np

from nandi.config import (
    SPIN_RISK_CONFIG, SPIN_CONFIG, TIMEFRAME_PROFILES,
    INITIAL_BALANCE, PAIR_TO_IDX,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
)
from nandi.environment.spin_reward import SPINReward

logger = logging.getLogger(__name__)

HOLD = 0
LONG = 1
SHORT = 2
CLOSE = 3


class SPINTradingEnv:
    """SPIN trading environment with hard risk gates.

    Self-contained: does not wrap ForexTradingEnv.
    Directly manages position, PnL, and market stepping.

    Observation: (market_state, position_info_12d)
    Action space: {HOLD=0, LONG=1, SHORT=2, CLOSE=3} with action masking.
    """

    def __init__(self, features, prices, lookback=120,
                 initial_balance=INITIAL_BALANCE, pair_name="unknown",
                 timeframe="M5", atr_series=None, h1_trend_series=None,
                 risk_config=None, reward_fn=None):
        """
        Args:
            features: (N, n_features) scaled feature array
            prices: (N,) close price array
            lookback: int, lookback window
            initial_balance: float
            pair_name: str
            timeframe: str
            atr_series: (N,) ATR(14) values aligned to features/prices
            h1_trend_series: (N,) H1 trend direction (-1, 0, +1)
            risk_config: override SPIN_RISK_CONFIG
            reward_fn: override SPINReward
        """
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.pair_name = pair_name
        self.pair_idx = PAIR_TO_IDX.get(pair_name, 0)
        self.timeframe = timeframe
        self.n_features = features.shape[1]

        # ATR and trend series (if not provided, use dummy)
        self.atr_series = atr_series if atr_series is not None else np.ones(len(prices)) * 0.001
        self.h1_trend = h1_trend_series if h1_trend_series is not None else np.zeros(len(prices))

        # Risk config
        self.risk = risk_config or SPIN_RISK_CONFIG
        self.reward_fn = reward_fn or SPINReward()

        profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["M5"])
        self.leverage = profile.get("leverage", 15)
        self.position_size = self.risk["max_position"]

        # Transaction cost
        self.cost_return = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0

        # Shape info
        self.market_state_shape = (lookback, self.n_features)
        self.position_info_dim = SPIN_CONFIG["position_dim"]  # 12

        # State variables (set in reset)
        self._reset_state()

    def _reset_state(self):
        """Initialize all mutable state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_state = 0  # -1, 0, +1
        self.entry_price = 0.0
        self.entry_bar = 0
        self.entry_atr = 0.0
        self.stop_price = 0.0

        # Trade tracking
        self.bars_in_trade = 0
        self.trade_mfe = 0.0
        self.trade_mae = 0.0
        self.current_excursion = 0.0

        # Risk state
        self.cooldown_remaining = 0
        self.consecutive_losses = 0
        self.session_trades = 0
        self.session_pnl = 0.0
        self.session_bar = 0
        self.bars_per_session = TIMEFRAME_PROFILES.get(
            self.timeframe, TIMEFRAME_PROFILES["M5"]
        ).get("bars_per_session", 288)

        # Step counter
        self.current_step = 0
        self.start_idx = 0

    def reset(self, start_idx=None):
        """Reset environment and return initial observation.

        Args:
            start_idx: starting index in the data arrays (default: lookback)

        Returns:
            (market_state, position_info): tuple of numpy arrays
        """
        self._reset_state()

        if start_idx is not None:
            self.start_idx = max(start_idx, self.lookback)
        else:
            self.start_idx = self.lookback

        self.current_step = self.start_idx
        return self._get_observation()

    def step(self, action):
        """Execute one step.

        Args:
            action: int in {0, 1, 2, 3}

        Returns:
            observation: (market_state, position_info)
            reward: float
            done: bool
            info: dict
        """
        action = int(action)
        price = self.prices[self.current_step]
        atr = self.atr_series[self.current_step]
        old_position = self.position_state
        trade_closed = False
        stop_loss_hit = False
        net_return = 0.0
        close_mfe = 0.0
        close_excursion = 0.0
        close_bars = 0

        # ── Check stop-loss FIRST (before any action) ──
        if self.position_state != 0 and self.stop_price > 0:
            if self.position_state == 1 and price <= self.stop_price:
                stop_loss_hit = True
            elif self.position_state == -1 and price >= self.stop_price:
                stop_loss_hit = True

            if stop_loss_hit:
                # Close at stop price
                net_return = self._compute_net_return(self.stop_price)
                close_mfe = self.trade_mfe
                close_excursion = self.current_excursion
                close_bars = self.bars_in_trade
                self._apply_trade_close(net_return, is_stop_loss=True)
                trade_closed = True

        # ── Check max hold bars ──
        if (not trade_closed and self.position_state != 0
                and self.bars_in_trade >= self.risk["max_hold_bars"]):
            net_return = self._compute_net_return(price)
            close_mfe = self.trade_mfe
            close_excursion = self.current_excursion
            close_bars = self.bars_in_trade
            self._apply_trade_close(net_return, is_stop_loss=False)
            trade_closed = True

        # ── Execute agent action (if not already closed by risk gate) ──
        if not trade_closed:
            if action == LONG and self.position_state == 0:
                self._open_trade(1, price, atr)
            elif action == SHORT and self.position_state == 0:
                self._open_trade(-1, price, atr)
            elif action == CLOSE and self.position_state != 0:
                net_return = self._compute_net_return(price)
                close_mfe = self.trade_mfe
                close_excursion = self.current_excursion
                close_bars = self.bars_in_trade
                self._apply_trade_close(net_return, is_stop_loss=False)
                trade_closed = True

        # ── Update trade tracking if still in trade ──
        if self.position_state != 0:
            self.bars_in_trade += 1
            exc = self._compute_excursion(price)
            self.current_excursion = exc
            self.trade_mfe = max(self.trade_mfe, exc)
            self.trade_mae = min(self.trade_mae, exc)

        # ── Advance step ──
        self.current_step += 1
        self.session_bar += 1

        # Session reset (new session)
        if self.session_bar >= self.bars_per_session:
            self.session_bar = 0
            self.session_trades = 0
            self.session_pnl = 0.0

        # Cooldown decrement
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # Check done
        done = self.current_step >= len(self.prices) - 1

        # Compute equity
        unrealized = 0.0
        if self.position_state != 0 and self.entry_price > 0:
            current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            exc = self._compute_excursion(current_price)
            unrealized = exc * self.position_size * self.leverage * self.initial_balance

        self.equity = self.balance + unrealized

        # Build info
        info = {
            "position_state": self.position_state,
            "trade_closed": trade_closed,
            "stop_loss_hit": stop_loss_hit,
            "net_return": net_return,
            "atr_at_entry": self.entry_atr if trade_closed else atr,
            "bars_in_trade": close_bars if trade_closed else self.bars_in_trade,
            "mfe": close_mfe if trade_closed else self.trade_mfe,
            "unrealized_pnl": close_excursion if trade_closed else self.current_excursion,
            "balance": self.balance,
            "equity": self.equity,
            "return_pct": (self.equity / self.initial_balance - 1.0) * 100.0,
            "drawdown": max(0.0, 1.0 - self.equity / self.initial_balance),
            "session_trades": self.session_trades,
            "session_pnl": self.session_pnl,
            "price": price,
            "pair_idx": self.pair_idx,
            "discrete_action": action,
            "raw_pnl": net_return * self.position_size * self.leverage if trade_closed else 0.0,
        }

        # Compute reward
        reward = self.reward_fn.compute(info)

        obs = self._get_observation()
        return obs, reward, done, info

    def _open_trade(self, direction, price, atr):
        """Open a new trade with ATR-based stop-loss."""
        self.position_state = direction
        self.entry_price = price
        self.entry_bar = self.current_step
        self.entry_atr = atr
        self.bars_in_trade = 0
        self.trade_mfe = 0.0
        self.trade_mae = 0.0
        self.current_excursion = 0.0

        # Set stop-loss: entry ∓ ATR × mult
        sl_distance = atr * self.risk["stop_loss_atr_mult"]
        if direction == 1:  # long
            self.stop_price = price - sl_distance
        else:  # short
            self.stop_price = price + sl_distance

        self.session_trades += 1

    def _compute_excursion(self, price):
        """Compute current excursion from entry."""
        if self.entry_price <= 0:
            return 0.0
        exc = (price - self.entry_price) / self.entry_price
        if self.position_state == -1:
            exc = -exc
        return exc

    def _compute_net_return(self, exit_price):
        """Compute net return after costs."""
        if self.entry_price <= 0:
            return 0.0
        gross = (exit_price - self.entry_price) / self.entry_price
        if self.position_state == -1:
            gross = -gross
        return gross - self.cost_return * 2  # entry + exit cost

    def _apply_trade_close(self, net_return, is_stop_loss):
        """Close current trade, update balance and risk state."""
        # Update balance
        pnl = net_return * self.position_size * self.leverage * self.balance
        self.balance += pnl
        self.session_pnl += net_return * 100.0  # as percent

        # Update consecutive losses
        if net_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Apply cooldown
        if is_stop_loss:
            self.cooldown_remaining = self.risk["cooldown_bars"]
        if self.consecutive_losses >= 2:
            self.cooldown_remaining = max(
                self.cooldown_remaining,
                self.risk["cooldown_consecutive"],
            )

        # Reset position
        self.position_state = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.bars_in_trade = 0
        self.trade_mfe = 0.0
        self.trade_mae = 0.0
        self.current_excursion = 0.0

    def get_action_mask(self):
        """Return valid action mask with all risk gates applied.

        Returns:
            mask: numpy (4,) bool, True = valid action
        """
        mask = np.zeros(4, dtype=bool)
        mask[HOLD] = True  # HOLD always valid

        if self.position_state == 0:
            # ── Flat: check entry conditions ──
            # Gate 1: Cooldown active?
            if self.cooldown_remaining > 0:
                return mask  # only HOLD

            # Gate 2: Session loss limit
            if self.session_pnl <= -self.risk["max_session_loss_pct"]:
                return mask  # only HOLD

            # Gate 3: Session trade limit
            if self.session_trades >= self.risk["max_session_trades"]:
                return mask  # only HOLD

            # Gate 4: Min R:R — need at least 1 ATR of room
            current_atr = self.atr_series[min(self.current_step, len(self.atr_series) - 1)]
            current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            if current_atr / (current_price + 1e-10) < 1e-5:
                return mask  # ATR too small, only HOLD

            # Gate 5: Trend filter
            h1_trend = self.h1_trend[min(self.current_step, len(self.h1_trend) - 1)]

            if self.risk.get("trend_filter", True):
                # LONG only if H1 trend >= 0
                if h1_trend >= 0:
                    mask[LONG] = True
                # SHORT only if H1 trend <= 0
                if h1_trend <= 0:
                    mask[SHORT] = True
            else:
                mask[LONG] = True
                mask[SHORT] = True

        else:
            # ── In trade: only HOLD or CLOSE ──
            mask[CLOSE] = True

        return mask

    def _get_observation(self):
        """Build (market_state, position_info_12d) observation."""
        t = min(self.current_step, len(self.features) - 1)

        # Market state: (lookback, n_features)
        start = max(0, t - self.lookback)
        ms = self.features[start:t]
        if ms.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - ms.shape[0], self.n_features), dtype=np.float32)
            ms = np.vstack([pad, ms])

        # Position info (12-dim)
        equity_return = (self.equity / self.initial_balance) - 1.0
        drawdown = max(0.0, 1.0 - self.equity / self.initial_balance)
        current_atr = self.atr_series[min(t, len(self.atr_series) - 1)]
        current_price = self.prices[min(t, len(self.prices) - 1)]
        vol_regime = current_atr / (current_price + 1e-10) * 100.0

        bars_norm = self.bars_in_trade / self.risk["max_hold_bars"]
        time_of_day = (self.session_bar % self.bars_per_session) / max(1, self.bars_per_session)

        unrealized_norm = float(np.clip(self.current_excursion * 100.0, -2.0, 2.0))
        mfe_norm = float(np.clip(self.trade_mfe * 100.0, 0.0, 2.0))
        mae_norm = float(np.clip(self.trade_mae * 100.0, -2.0, 0.0))

        # Stop distance: how far from stop-loss (normalized)
        if self.position_state != 0 and self.stop_price > 0 and current_price > 0:
            if self.position_state == 1:
                stop_dist = (current_price - self.stop_price) / current_price
            else:
                stop_dist = (self.stop_price - current_price) / current_price
            stop_dist_norm = float(np.clip(stop_dist * 100.0, 0.0, 3.0))
        else:
            stop_dist_norm = 0.0

        # Current R:R ratio
        if self.position_state != 0 and self.entry_atr > 0:
            rr = self.current_excursion / (self.risk["stop_loss_atr_mult"] * self.entry_atr / (self.entry_price + 1e-10))
        else:
            rr = 0.0
        rr_norm = float(np.clip(rr, -2.0, 3.0))

        # Session loss budget remaining
        session_budget = (self.risk["max_session_loss_pct"] + self.session_pnl) / self.risk["max_session_loss_pct"]
        session_budget = float(np.clip(session_budget, 0.0, 1.0))

        position_info = np.array([
            float(self.position_state),
            float(np.clip(equity_return, -1.0, 1.0)),
            float(np.clip(drawdown, 0.0, 1.0)),
            float(np.clip(vol_regime, 0.0, 5.0)),
            float(np.clip(bars_norm, 0.0, 2.0)),
            time_of_day,
            unrealized_norm,
            mfe_norm,
            mae_norm,
            stop_dist_norm,
            rr_norm,
            session_budget,
        ], dtype=np.float32)

        return ms.astype(np.float32), position_info


class MultiEpisodeSPINEnv:
    """Wraps SPINTradingEnv for diverse training episodes with random starts."""

    def __init__(self, features, prices, lookback=120, episode_length=2016,
                 pair_name="unknown", timeframe="M5",
                 atr_series=None, h1_trend_series=None,
                 risk_config=None, reward_fn=None):
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.episode_length = episode_length

        self.env = SPINTradingEnv(
            features=features, prices=prices, lookback=lookback,
            pair_name=pair_name, timeframe=timeframe,
            atr_series=atr_series, h1_trend_series=h1_trend_series,
            risk_config=risk_config, reward_fn=reward_fn,
        )
        self.steps_in_episode = 0

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

    def get_action_mask(self):
        return self.env.get_action_mask()

    @property
    def market_state_shape(self):
        return self.env.market_state_shape

    @property
    def position_info_dim(self):
        return self.env.position_info_dim

    @property
    def pair_idx(self):
        return self.env.pair_idx
