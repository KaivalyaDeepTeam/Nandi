"""LiveRiskManager — Mirrors SPINTradingEnv risk gates for live trading.

Tracks per-pair position state, stop-loss, cooldown, session limits,
and produces action masks + position_info identical to training env.
"""

import logging

import numpy as np

from nandi.config import SPIN_RISK_CONFIG, TIMEFRAME_PROFILES

logger = logging.getLogger(__name__)

HOLD = 0
LONG = 1
SHORT = 2
CLOSE = 3


class PairState:
    """Mutable state for one trading pair."""

    def __init__(self, risk_config):
        self.risk = risk_config
        self.bars_per_session = TIMEFRAME_PROFILES["M5"].get("bars_per_session", 288)
        self.reset_all()

    def reset_all(self):
        self.position_state = 0   # -1, 0, +1
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.stop_price = 0.0
        self.bars_in_trade = 0
        self.trade_mfe = 0.0
        self.trade_mae = 0.0
        self.current_excursion = 0.0
        self.cooldown_remaining = 0
        self.consecutive_losses = 0
        self.session_trades = 0
        self.session_pnl = 0.0
        self.session_bar = 0
        self.equity_return = 0.0
        self.drawdown = 0.0

    def new_session(self):
        self.session_trades = 0
        self.session_pnl = 0.0
        self.session_bar = 0


class LiveRiskManager:
    """Per-pair risk management mirroring SPINTradingEnv."""

    def __init__(self, pairs, risk_config=None):
        self.risk = risk_config or SPIN_RISK_CONFIG
        self.states = {pair: PairState(self.risk) for pair in pairs}

    def get_action_mask(self, pair, atr, price, h1_trend):
        """Return valid action mask — mirrors spin_env.py:337-384.

        Args:
            pair: str
            atr: float, current ATR(14)
            price: float, current close price
            h1_trend: int, H1 trend direction (-1, 0, +1)

        Returns:
            numpy (4,) bool, True = valid action
        """
        s = self.states[pair]
        mask = np.zeros(4, dtype=bool)
        mask[HOLD] = True

        if s.position_state == 0:
            # Flat: check entry conditions
            if s.cooldown_remaining > 0:
                return mask
            if s.session_pnl <= -self.risk["max_session_loss_pct"]:
                return mask
            if s.session_trades >= self.risk["max_session_trades"]:
                return mask
            if atr / (price + 1e-10) < 1e-5:
                return mask

            # Trend filter
            if self.risk.get("trend_filter", True):
                if h1_trend >= 0:
                    mask[LONG] = True
                if h1_trend <= 0:
                    mask[SHORT] = True
            else:
                mask[LONG] = True
                mask[SHORT] = True
        else:
            mask[CLOSE] = True

        return mask

    def get_position_info(self, pair, atr, price):
        """Build 12-dim position info — mirrors spin_env.py:386-447.

        Args:
            pair: str
            atr: float, current ATR(14)
            price: float, current close price

        Returns:
            numpy (12,) float32
        """
        s = self.states[pair]
        vol_regime = atr / (price + 1e-10) * 100.0
        bars_norm = s.bars_in_trade / self.risk["max_hold_bars"]
        time_of_day = (s.session_bar % s.bars_per_session) / max(1, s.bars_per_session)

        unrealized_norm = float(np.clip(s.current_excursion * 100.0, -2.0, 2.0))
        mfe_norm = float(np.clip(s.trade_mfe * 100.0, 0.0, 2.0))
        mae_norm = float(np.clip(s.trade_mae * 100.0, -2.0, 0.0))

        # Stop distance
        if s.position_state != 0 and s.stop_price > 0 and price > 0:
            if s.position_state == 1:
                stop_dist = (price - s.stop_price) / price
            else:
                stop_dist = (s.stop_price - price) / price
            stop_dist_norm = float(np.clip(stop_dist * 100.0, 0.0, 3.0))
        else:
            stop_dist_norm = 0.0

        # R:R ratio
        if s.position_state != 0 and s.entry_atr > 0:
            rr = s.current_excursion / (
                self.risk["stop_loss_atr_mult"] * s.entry_atr / (s.entry_price + 1e-10)
            )
        else:
            rr = 0.0
        rr_norm = float(np.clip(rr, -2.0, 3.0))

        # Session budget
        session_budget = (
            (self.risk["max_session_loss_pct"] + s.session_pnl)
            / self.risk["max_session_loss_pct"]
        )
        session_budget = float(np.clip(session_budget, 0.0, 1.0))

        return np.array([
            float(s.position_state),
            float(np.clip(s.equity_return, -1.0, 1.0)),
            float(np.clip(s.drawdown, 0.0, 1.0)),
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

    def on_entry(self, pair, direction, price, atr):
        """Record new trade entry."""
        s = self.states[pair]
        s.position_state = direction
        s.entry_price = price
        s.entry_atr = atr
        s.bars_in_trade = 0
        s.trade_mfe = 0.0
        s.trade_mae = 0.0
        s.current_excursion = 0.0

        # Set stop-loss
        sl_dist = atr * self.risk["stop_loss_atr_mult"]
        if direction == 1:
            s.stop_price = price - sl_dist
        else:
            s.stop_price = price + sl_dist

        s.session_trades += 1
        logger.info(
            f"[{pair}] ENTRY {'LONG' if direction == 1 else 'SHORT'} "
            f"@ {price:.5f}  SL={s.stop_price:.5f}  ATR={atr:.5f}"
        )

    def on_close(self, pair, exit_price, is_stop_loss=False):
        """Record trade close, update risk state."""
        s = self.states[pair]
        if s.entry_price <= 0:
            s.position_state = 0
            s.stop_price = 0.0
            return 0.0

        # Compute net return
        gross = (exit_price - s.entry_price) / s.entry_price
        if s.position_state == -1:
            gross = -gross
        cost_return = (3.0 + 1.0) / 10000.0  # TRANSACTION_COST_BPS + SLIPPAGE_BPS
        net_return = gross - cost_return * 2

        s.session_pnl += net_return * 100.0

        # Consecutive losses
        if net_return < 0:
            s.consecutive_losses += 1
        else:
            s.consecutive_losses = 0

        # Cooldown
        if is_stop_loss:
            s.cooldown_remaining = self.risk["cooldown_bars"]
        if s.consecutive_losses >= 2:
            s.cooldown_remaining = max(
                s.cooldown_remaining,
                self.risk["cooldown_consecutive"],
            )

        direction_str = "LONG" if s.position_state == 1 else "SHORT"
        logger.info(
            f"[{pair}] CLOSE {direction_str} @ {exit_price:.5f}  "
            f"net={net_return*100:.3f}%  SL={'Y' if is_stop_loss else 'N'}  "
            f"bars={s.bars_in_trade}"
        )

        # Reset position
        s.position_state = 0
        s.entry_price = 0.0
        s.stop_price = 0.0
        s.bars_in_trade = 0
        s.trade_mfe = 0.0
        s.trade_mae = 0.0
        s.current_excursion = 0.0

        return net_return

    def check_stop_loss(self, pair, current_price):
        """Check if current price hit stop-loss.

        Returns:
            bool: True if SL hit
        """
        s = self.states[pair]
        if s.position_state == 0 or s.stop_price <= 0:
            return False
        if s.position_state == 1 and current_price <= s.stop_price:
            return True
        if s.position_state == -1 and current_price >= s.stop_price:
            return True
        return False

    def check_max_hold(self, pair):
        """Check if max hold bars exceeded."""
        s = self.states[pair]
        if s.position_state == 0:
            return False
        return s.bars_in_trade >= self.risk["max_hold_bars"]

    def tick_bar(self, pair, current_price):
        """Call on each new M5 bar to update trade tracking."""
        s = self.states[pair]
        s.session_bar += 1

        if s.position_state != 0 and s.entry_price > 0:
            s.bars_in_trade += 1
            exc = (current_price - s.entry_price) / s.entry_price
            if s.position_state == -1:
                exc = -exc
            s.current_excursion = exc
            s.trade_mfe = max(s.trade_mfe, exc)
            s.trade_mae = min(s.trade_mae, exc)

        if s.cooldown_remaining > 0:
            s.cooldown_remaining -= 1

    def new_session(self, pair):
        """Reset session counters."""
        self.states[pair].new_session()

    def get_position_state(self, pair):
        return self.states[pair].position_state

    def get_stop_price(self, pair):
        return self.states[pair].stop_price

    def get_entry_price(self, pair):
        return self.states[pair].entry_price
