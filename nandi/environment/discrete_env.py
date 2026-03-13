"""
DiscreteActionEnv — wraps ForexTradingEnv for discrete action DQN.

V4: Adaptive Duration Trading with Binary Exit Masking.

Key changes from V3:
- Binary exit masking: when in trade, only HOLD and CLOSE valid (no reversals)
  Reduces 4-way Q-value competition to binary hold/exit decision
- Raw PnL reward: bypasses base env's loss doubling and activity bonus
  Computes old_position × market_return × leverage directly
- 8-dim position info: adds unrealized_pnl and MFE for exit decisions
- No max_hold_bars: agent learns adaptive exit timing from raw PnL
"""

import logging

import numpy as np

from nandi.config import (
    TIMEFRAME_PROFILES, INITIAL_BALANCE, PAIR_TO_IDX,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
)
from nandi.environment.single_pair_env import ForexTradingEnv

logger = logging.getLogger(__name__)


# Action constants
HOLD = 0
LONG = 1
SHORT = 2
CLOSE = 3


class DiscreteActionEnv:
    """Discrete-action wrapper over ForexTradingEnv.

    Maps:
        HOLD  → maintain current position (no change)
        LONG  → +position_size
        SHORT → -position_size
        CLOSE → 0.0
    Binary exit masking: when in trade, only HOLD and CLOSE are valid.

    Augments position_info to 8 dims:
        [position_state, equity_return, drawdown, vol_regime,
         bars_in_trade_normalized, time_of_day,
         unrealized_pnl_normalized, mfe_normalized]

    Computes raw_pnl directly (bypasses base env's reward shaping).
    """

    def __init__(self, features, prices, lookback=120,
                 initial_balance=INITIAL_BALANCE, pair_name="unknown",
                 market_sim=None, timeframe="M5", reward_fn=None):
        self.base_env = ForexTradingEnv(
            features=features, prices=prices, lookback=lookback,
            initial_balance=initial_balance, pair_name=pair_name,
            market_sim=market_sim, use_composite_reward=False,
            timeframe=timeframe,
        )
        self.timeframe = timeframe
        self.pair_name = pair_name
        self.pair_idx = PAIR_TO_IDX.get(pair_name, 0)
        self.reward_fn = reward_fn

        profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["M5"])
        self.position_size = profile["max_position"]
        self.bars_per_session = profile.get("bars_per_session", 288)
        self.leverage = profile.get("leverage", 5)

        # Transaction cost in return space
        self.cost_return = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0

        # Trailing stop: auto-close when price reverses too far from MFE
        self.trailing_stop_pct = profile.get("trailing_stop_pct", 0.0)

        self.market_state_shape = self.base_env.market_state_shape
        self.position_info_dim = 8  # V4: augmented from 6 to 8
        self.n_features = self.base_env.n_features

        # Trade tracking for MFE/MAE
        self._trade_entry_price = 0.0
        self._trade_direction = 0  # +1 long, -1 short, 0 flat
        self._trade_mfe = 0.0  # max favorable excursion (in return)
        self._trade_mae = 0.0  # max adverse excursion (in return)
        self._trade_bars = 0
        self._current_excursion = 0.0  # current unrealized PnL (return)
        self._position_state = 0  # -1, 0, +1

        # Session tracking for time-of-day feature
        self._session_bar = 0

    def reset(self, start_idx=None):
        base_state = self.base_env.reset(start_idx)
        self._trade_entry_price = 0.0
        self._trade_direction = 0
        self._trade_mfe = 0.0
        self._trade_mae = 0.0
        self._trade_bars = 0
        self._current_excursion = 0.0
        self._position_state = 0
        self._session_bar = 0
        return self._augment_state(base_state)

    def step(self, discrete_action):
        """Take a discrete action, translate to continuous for base env.

        Args:
            discrete_action: int in {0, 1, 2, 3}

        Returns:
            state: (market_state, position_info_8d)
            reward: float
            done: bool
            info: dict (includes mfe, mae, trade_closed, raw_pnl, etc.)
        """
        discrete_action = int(discrete_action)

        # Translate discrete → continuous position
        continuous_action = self._discrete_to_continuous(discrete_action)

        # Save old position for raw PnL computation
        old_position = self.base_env.position

        # Detect trade close/open for MFE/MAE tracking
        was_in_trade = self._position_state != 0
        prev_direction = self._trade_direction
        trade_closed = False
        close_mfe = 0.0
        close_mae = 0.0
        close_bars = 0
        close_excursion = 0.0

        # Step base environment
        base_state, base_reward, done, info = self.base_env.step(continuous_action)
        self._session_bar += 1

        # ── Compute raw PnL (bypass base env's shaped reward) ──
        market_return = info.get("market_return", 0.0)
        raw_pnl = old_position * market_return * self.leverage

        # Subtract transaction cost on position changes
        new_position = self.base_env.position
        position_delta = abs(new_position - old_position)
        if position_delta > 0.01:
            raw_pnl -= self.cost_return * position_delta

        # Update position state after step
        if new_position > 0.01:
            new_state = 1
        elif new_position < -0.01:
            new_state = -1
        else:
            new_state = 0

        # Track trade lifecycle
        current_price = info.get("price", 0.0)

        # Check if trade was closed
        if was_in_trade and (new_state == 0 or
                             (new_state != 0 and new_state != self._position_state)):
            trade_closed = True
            # Compute final excursion at close price
            if self._trade_entry_price > 0 and current_price > 0:
                exc = (current_price - self._trade_entry_price) / self._trade_entry_price
                if prev_direction == -1:
                    exc = -exc
                close_excursion = exc
            close_mfe = self._trade_mfe
            close_mae = self._trade_mae
            close_bars = self._trade_bars

            # Reset for potential new trade
            self._trade_mfe = 0.0
            self._trade_mae = 0.0
            self._trade_bars = 0
            self._current_excursion = 0.0

        # Check if new trade opened
        if new_state != 0 and (not was_in_trade or trade_closed):
            self._trade_entry_price = current_price
            self._trade_direction = new_state
            self._trade_mfe = 0.0
            self._trade_mae = 0.0
            self._trade_bars = 0
            self._current_excursion = 0.0

        # Update MFE/MAE for current trade
        if new_state != 0 and self._trade_entry_price > 0:
            self._trade_bars += 1
            excursion = (current_price - self._trade_entry_price) / self._trade_entry_price
            if self._trade_direction == -1:
                excursion = -excursion  # flip for short
            self._trade_mfe = max(self._trade_mfe, excursion)
            self._trade_mae = min(self._trade_mae, excursion)
            self._current_excursion = excursion

        self._position_state = new_state

        # ── Trailing stop: force close if giveback exceeds threshold ──
        if (self.trailing_stop_pct > 0
                and self._position_state != 0
                and self._trade_mfe > 0.001):
            giveback = self._trade_mfe - self._current_excursion
            if giveback > self.trailing_stop_pct:
                logger.debug(
                    f"Trailing stop triggered: mfe={self._trade_mfe:.4f} "
                    f"current={self._current_excursion:.4f} "
                    f"giveback={giveback:.4f} > {self.trailing_stop_pct}"
                )
                # Force close: re-execute with CLOSE action on base env
                old_pos_for_close = self.base_env.position
                close_state, close_reward, close_done, close_info = \
                    self.base_env.step(0.0)  # flatten position

                # Compute raw PnL for the close
                close_market_return = close_info.get("market_return", 0.0)
                close_raw_pnl = old_pos_for_close * close_market_return * self.leverage
                close_delta = abs(self.base_env.position - old_pos_for_close)
                if close_delta > 0.01:
                    close_raw_pnl -= self.cost_return * close_delta

                # Record the forced close
                trade_closed = True
                close_mfe = self._trade_mfe
                close_mae = self._trade_mae
                close_bars = self._trade_bars
                close_excursion = self._current_excursion

                # Update raw_pnl and state
                raw_pnl += close_raw_pnl
                info = close_info
                info["trailing_stop"] = True
                done = close_done
                base_state = close_state

                # Reset trade state
                self._trade_mfe = 0.0
                self._trade_mae = 0.0
                self._trade_bars = 0
                self._current_excursion = 0.0
                self._position_state = 0

        # Build enriched info
        info["mfe"] = close_mfe if trade_closed else self._trade_mfe
        info["mae"] = close_mae if trade_closed else self._trade_mae
        info["trade_closed"] = trade_closed
        info["close_bars"] = close_bars
        info["discrete_action"] = discrete_action
        info["pair_idx"] = self.pair_idx
        info["raw_pnl"] = raw_pnl

        # Compute reward via custom reward fn if provided
        if self.reward_fn is not None:
            reward = self.reward_fn.compute(
                base_reward=raw_pnl,  # V4: pass raw_pnl instead of base_reward
                info=info,
                position_state=self._position_state,
                trade_closed=trade_closed,
                mfe=close_mfe if trade_closed else self._trade_mfe,
                mae=close_mae if trade_closed else self._trade_mae,
                bars_in_trade=close_bars if trade_closed else self._trade_bars,
                drawdown=info.get("drawdown", 0.0),
                unrealized_pnl=close_excursion if trade_closed else self._current_excursion,
            )
        else:
            reward = raw_pnl

        augmented_state = self._augment_state(base_state)
        return augmented_state, reward, done, info

    def _discrete_to_continuous(self, action):
        """Map discrete action to continuous position target."""
        if action == HOLD:
            return self.base_env.position  # keep current position
        elif action == LONG:
            return self.position_size
        elif action == SHORT:
            return -self.position_size
        elif action == CLOSE:
            return 0.0
        else:
            return self.base_env.position  # fallback: hold

    def _augment_state(self, base_state):
        """Augment position_info from 4 → 8 dimensions.

        Original 4: [position, equity_return, drawdown, vol_regime]
        V4 8-dim:   [position_state, equity_return, drawdown, vol_regime,
                     bars_in_trade_norm, time_of_day,
                     unrealized_pnl_norm, mfe_norm]
        """
        market_state, base_pos_info = base_state

        # position_state: -1 (short), 0 (flat), +1 (long)
        pos_state = float(self._position_state)

        # bars_in_trade: normalized by 100 (no fixed limit — adaptive duration)
        bars_norm = self._trade_bars / 100.0

        # time_of_day: session bar normalized to [0, 1]
        time_of_day = (self._session_bar % self.bars_per_session) / max(
            1, self.bars_per_session
        )

        # unrealized PnL: current trade excursion scaled ×100, clipped [-2, 2]
        unrealized_norm = float(np.clip(self._current_excursion * 100.0, -2.0, 2.0))

        # MFE: max favorable excursion scaled ×100, clipped [0, 2]
        mfe_norm = float(np.clip(self._trade_mfe * 100.0, 0.0, 2.0))

        position_info = np.array([
            pos_state,
            base_pos_info[1],    # equity_return
            base_pos_info[2],    # drawdown
            base_pos_info[3],    # vol_regime
            bars_norm,
            time_of_day,
            unrealized_norm,     # V4: current trade PnL
            mfe_norm,            # V4: max favorable excursion
        ], dtype=np.float32)

        return market_state, position_info

    def get_action_mask(self):
        """Return valid action mask based on current position state.

        V4 Binary Exit Masking:
          Flat  → can HOLD, LONG, SHORT (not CLOSE)
          Long  → can HOLD, CLOSE only (no SHORT reversal — must close first)
          Short → can HOLD, CLOSE only (no LONG reversal — must close first)

        This reduces the in-trade decision to binary hold/exit, making
        Q-value learning much easier than 4-way competition.

        Returns:
            mask: numpy (4,) bool, True = valid action
        """
        mask = np.ones(4, dtype=bool)

        if self._position_state == 0:
            # Flat: can enter LONG or SHORT, can HOLD, cannot CLOSE
            mask[CLOSE] = False
        elif self._position_state == 1:
            # Long: can only HOLD or CLOSE (binary exit decision)
            mask[LONG] = False
            mask[SHORT] = False  # V4: no reversals
        elif self._position_state == -1:
            # Short: can only HOLD or CLOSE (binary exit decision)
            mask[SHORT] = False
            mask[LONG] = False   # V4: no reversals

        return mask


class MultiEpisodeDiscreteEnv:
    """Wraps DiscreteActionEnv for diverse training episodes with random starts."""

    def __init__(self, features, prices, lookback=120, episode_length=2016,
                 pair_name="unknown", market_sim=None, timeframe="M5",
                 reward_fn=None):
        self.features = features
        self.prices = prices
        self.lookback = lookback
        self.episode_length = episode_length
        self.env = DiscreteActionEnv(
            features=features, prices=prices, lookback=lookback,
            pair_name=pair_name, market_sim=market_sim, timeframe=timeframe,
            reward_fn=reward_fn,
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
