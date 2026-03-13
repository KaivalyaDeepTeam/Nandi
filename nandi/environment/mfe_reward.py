"""
V4: Clean PnL Reward — Adaptive Duration Trading.

Previous versions failed because:
  V1: base_reward × 0.5 — invisible signal
  V2: base_reward × 50, no holding cost — disposition effect (never CLOSE)
  V3: base_reward × 50, underwater cost ×15 — trade avoidance (98% HOLD)

Root cause: base_reward from ForexTradingEnv contains hidden shaping:
  - Loss doubling: `if ret < 0: ret *= 2.0` (losses 2x more painful)
  - Activity bonus: +0.0005/bar in trade (comparable to actual PnL signal)
  - DD penalty and vol bonus baked in
  These get multiplied by 50×, drowning the actual market PnL signal.

V4 fix: Use raw_pnl computed directly in DiscreteActionEnv:
  raw_pnl = old_position × market_return × leverage - transaction_cost
  No loss doubling, no activity bonus, no shaping — pure market signal.

Key design: Let raw PnL alone teach "cut losers, let winners run":
  - Losing bar: negative reward → Q(CLOSE) > Q(HOLD)
  - Winning bar: positive reward → Q(HOLD) > Q(CLOSE)
  - Flat: small fixed cost → prefer entering when edge exists
  - No max_hold_bars limit → agent learns adaptive exit timing
"""

import numpy as np


class MFEMAEReward:
    """Clean PnL reward for discrete-action trading.

    Reward = scaled raw PnL per bar (no synthetic bonuses/penalties).

    Components:
    1. In trade / trade close: raw_pnl × PNL_SCALE
    2. Flat (no trade): -FLAT_COST (opportunity cost, prevents "never trade")
    3. Drawdown: soft penalty beyond threshold (safety net)
    """

    PNL_SCALE = 200.0    # raw_pnl ~0.002/bar (leverage=15) → scaled ~0.40 (learnable)
    FLAT_COST = 0.002    # V6: reduced from 0.01 — old value incentivized overtrading

    def __init__(self, max_hold_bars=36, dd_threshold=0.05, dd_scale=0.5):
        # max_hold_bars unused in V4 (adaptive duration) but kept for API compat
        self.dd_threshold = dd_threshold
        self.dd_scale = dd_scale

    def compute(self, base_reward, info, position_state, trade_closed,
                mfe, mae, bars_in_trade, drawdown, unrealized_pnl=0.0):
        """Compute reward from raw PnL.

        Args:
            base_reward: raw_pnl from DiscreteActionEnv (NOT base env's shaped reward)
            info: dict from env step
            position_state: -1, 0, or +1 (AFTER step)
            trade_closed: bool
            mfe, mae, bars_in_trade: unused in V4 (kept for API compat)
            drawdown: float, current drawdown ratio
            unrealized_pnl: float, current excursion

        Returns:
            reward: float, clipped to [-3, 3]
        """
        # ── Flat: opportunity cost ──
        if position_state == 0 and not trade_closed:
            return -self.FLAT_COST

        # ── In trade or just closed: scaled raw PnL ──
        reward = base_reward * self.PNL_SCALE

        # ── Exit-quality shaping ──

        # 1. On trade close: MFE capture ratio bonus
        #    Reward capturing most of the peak gain (good exit timing)
        if trade_closed and mfe > 0:
            capture_ratio = unrealized_pnl / mfe
            reward += 0.3 * max(0.0, capture_ratio)

        # 2. While in trade: escalating drawdown from MFE
        #    Penalize holding when price has reversed far from peak
        if bars_in_trade > 0 and mfe > 0:
            giveback = (mfe - unrealized_pnl) / max(mfe, 1e-6)
            if giveback > 0.5:
                reward -= 0.1 * giveback

        # 3. Time decay: escalating per-bar cost for overly long holds
        if bars_in_trade > 12:
            reward -= 0.005 * (bars_in_trade - 12)

        # ── Drawdown penalty (safety net) ──
        if drawdown > self.dd_threshold:
            dd_penalty = (drawdown - self.dd_threshold) * self.dd_scale
            reward -= dd_penalty

        return float(np.clip(reward, -3.0, 3.0))

    def reset(self):
        """No state to reset — all computation is stateless."""
        pass
