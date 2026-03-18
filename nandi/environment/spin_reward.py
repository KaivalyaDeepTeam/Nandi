"""
SPIN Reward: Trade-Outcome Based Reward Function.

Key design principles:
  - While flat: reward = 0 (patience is free, not penalized)
  - While in trade: reward = 0 (no noisy bar-by-bar signal)
    except: time penalty for overly long holds
  - On trade close: the main reward signal
    net_return normalized by ATR, asymmetric loss penalty
  - On stop-loss hit: same as close + extra SL penalty
"""

import numpy as np


class SPINReward:
    """Trade-outcome reward for SPIN environment.

    Rewards are concentrated at trade completion to avoid noisy
    bar-by-bar signals that confuse the agent.
    """

    PNL_SCALE = 50.0         # net_return / atr → reward scale
    LOSS_MULT = 1.5           # losses hurt 1.5x more (asymmetric)
    MFE_CAPTURE_BONUS = 0.3   # bonus for high capture ratio
    QUICK_PROFIT_BONUS = 0.2  # bonus for profitable trades <= 6 bars
    SL_PENALTY = -0.3         # extra penalty for stop-loss hits
    TIME_PENALTY_RATE = 0.001 # per bar after 6 bars in trade
    FLAT_COST = 0.003         # small opportunity cost when flat — prevents HOLD collapse

    def compute(self, info):
        """Compute reward from trade info dict.

        Args:
            info: dict with keys:
                position_state: int (-1, 0, +1) after step
                trade_closed: bool
                stop_loss_hit: bool
                net_return: float (exit - entry / entry * direction - costs)
                atr_at_entry: float
                bars_in_trade: int
                mfe: float (max favorable excursion)
                unrealized_pnl: float (final excursion at close)

        Returns:
            reward: float
        """
        trade_closed = info.get("trade_closed", False)
        position_state = info.get("position_state", 0)
        bars_in_trade = info.get("bars_in_trade", 0)

        # ── While flat: small opportunity cost to prevent HOLD collapse ──
        if position_state == 0 and not trade_closed:
            return -self.FLAT_COST

        # ── While in trade (not closing): small time penalty only ──
        if position_state != 0 and not trade_closed:
            if bars_in_trade > 6:
                return -self.TIME_PENALTY_RATE * (bars_in_trade - 6)
            return 0.0

        # ── Trade closed: the main reward signal ──
        net_return = info.get("net_return", 0.0)
        atr_at_entry = info.get("atr_at_entry", 1e-6)
        stop_loss_hit = info.get("stop_loss_hit", False)
        mfe = info.get("mfe", 0.0)
        unrealized_pnl = info.get("unrealized_pnl", 0.0)

        # Normalize by ATR at entry
        normalized = net_return / max(atr_at_entry, 1e-6)
        reward = normalized * self.PNL_SCALE

        # Asymmetric: losses hurt more
        if net_return < 0:
            reward *= self.LOSS_MULT

        # MFE capture bonus: reward catching most of the peak
        if mfe > 0:
            capture_ratio = unrealized_pnl / mfe
            reward += self.MFE_CAPTURE_BONUS * max(0.0, capture_ratio)

        # Quick profit bonus
        if net_return > 0 and bars_in_trade <= 6:
            reward += self.QUICK_PROFIT_BONUS

        # Stop-loss penalty
        if stop_loss_hit:
            reward += self.SL_PENALTY

        return float(np.clip(reward, -5.0, 5.0))

    def reset(self):
        """Stateless — nothing to reset."""
        pass
