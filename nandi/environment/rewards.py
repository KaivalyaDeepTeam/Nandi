"""
rewards.py — Reward engineering for Nandi RL trading agent.

Implements reward classes:
    - DifferentialSharpeReward: Moody & Saffell (1998) online Sharpe gradient.
    - CVaRReward: Conditional Value at Risk tail-loss penalty.
    - ScalpingReward: "Many small wins" reward for M5 scalping.
    - CompositeReward: Timeframe-aware wrapper — D1 uses DSR+CVaR, M5 uses ScalpingReward.
"""

import numpy as np

from nandi.config import TIMEFRAME_PROFILES


class DifferentialSharpeReward:
    """Differential Sharpe Ratio reward following Moody & Saffell (1998)."""

    def __init__(self, eta: float = 0.01) -> None:
        self.eta = eta
        self.A: float = 0.0
        self.B: float = 0.0

    def compute(self, R_t: float) -> float:
        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B

        variance = self.B - self.A ** 2
        if variance > 1e-10:
            D_t = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance ** 1.5)
        else:
            D_t = 0.0

        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return D_t

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0


class CVaRReward:
    """Conditional Value at Risk (CVaR) tail-loss penalty."""

    def __init__(self, alpha: float = 0.05, window: int = 20) -> None:
        self.alpha = alpha
        self.window = window
        self.buffer: list = []

    def compute_penalty(self, current_return: float) -> float:
        self.buffer.append(current_return)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window:]

        if len(self.buffer) < 5:
            return 0.0

        sorted_returns = np.sort(self.buffer)
        n_tail = max(1, int(np.floor(self.alpha * len(sorted_returns))))
        tail = sorted_returns[:n_tail]
        cvar = float(np.mean(tail))

        return min(0.0, cvar)

    def reset(self) -> None:
        self.buffer = []


class ScalpingReward:
    """Scalping reward: "take many small high-probability edges, skip everything else."

    Designed for M5 (and other intraday) timeframes. Encourages:
    - Quick profitable round-trips (closed within ~1 hour)
    - Staying flat when uncertain
    - Small positions (capital preservation)

    Penalizes:
    - Holding too long (overnight risk)
    - Greed (oversized positions)
    """

    def __init__(self, timeframe: str = "M5") -> None:
        profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["M5"])
        self.max_hold_bars = profile.get("max_hold_bars", 36)
        self.max_position = profile.get("max_position", 0.3)
        self.dsr = DifferentialSharpeReward(eta=0.02)  # faster adaptation for scalping
        self.cvar = CVaRReward(alpha=0.05, window=50)
        self.prev_position = 0.0

    def compute(
        self,
        pnl: float,
        cost: float,
        equity: float,
        drawdown: float,
        market_return: float,
        position: float,
        bars_in_trade: int = 0,
    ) -> float:
        ret = (pnl - cost) / equity

        # Base: DSR
        dsr = self.dsr.compute(ret)
        cvar_penalty = self.cvar.compute_penalty(ret)

        reward = dsr + 0.2 * cvar_penalty

        # ── Quick profit bonus ──
        # +0.5x reward for profitable trades closed within 12 bars (~1h)
        trade_just_closed = (abs(self.prev_position) > 0.05 and abs(position) < 0.05)
        if trade_just_closed and ret > 0 and bars_in_trade <= 12:
            reward += abs(ret) * 0.5

        # ── Spread recovery bonus ──
        # +0.3x if trade profit > 2x transaction cost
        if trade_just_closed and cost > 0 and pnl > 2 * cost:
            reward += abs(ret) * 0.3

        # ── Holding penalty ──
        # -0.001 per bar after max_hold_bars (default 36 = ~3h)
        if bars_in_trade > self.max_hold_bars and abs(position) > 0.05:
            excess_bars = bars_in_trade - self.max_hold_bars
            reward -= 0.001 * excess_bars

        # ── Greed penalty ──
        # -0.2x if position size exceeds max_position
        if abs(position) > self.max_position:
            overshoot = abs(position) - self.max_position
            reward -= 0.2 * overshoot

        # ── Flat-when-uncertain bonus ──
        # +0.002 for staying flat when market is quiet
        if abs(position) < 0.05 and abs(market_return) < 0.0005:
            reward += 0.002

        # ── Drawdown penalty ──
        dd_penalty = max(0.0, drawdown - 0.03) * 0.8  # tighter threshold for scalping
        reward -= dd_penalty

        self.prev_position = position
        return float(np.clip(reward, -1.0, 1.0))

    def reset(self) -> None:
        self.dsr.reset()
        self.cvar.reset()
        self.prev_position = 0.0


class CompositeReward:
    """Timeframe-aware composite reward.

    - D1: DSR + CVaR + drawdown penalty + patience bonus (original behavior).
    - M5/intraday: ScalpingReward (quick profit, holding penalty, greed penalty).
    """

    def __init__(
        self,
        dsr_eta: float = 0.01,
        cvar_alpha: float = 0.05,
        cvar_window: int = 20,
        dd_threshold: float = 0.05,
        dd_scale: float = 0.5,
        cvar_weight: float = 0.3,
        timeframe: str = "D1",
    ) -> None:
        self.timeframe = timeframe

        if timeframe != "D1":
            # Use ScalpingReward for any intraday timeframe
            self.scalping_reward = ScalpingReward(timeframe=timeframe)
            self.dsr = None
            self.cvar = None
        else:
            # Original D1 reward
            self.scalping_reward = None
            self.dsr = DifferentialSharpeReward(eta=dsr_eta)
            self.cvar = CVaRReward(alpha=cvar_alpha, window=cvar_window)

        self.dd_threshold = dd_threshold
        self.dd_scale = dd_scale
        self.cvar_weight = cvar_weight

    def compute(
        self,
        pnl: float,
        cost: float,
        equity: float,
        drawdown: float,
        market_return: float,
        position: float,
        bars_in_trade: int = 0,
    ) -> float:
        if self.scalping_reward is not None:
            return self.scalping_reward.compute(
                pnl, cost, equity, drawdown, market_return, position,
                bars_in_trade=bars_in_trade,
            )

        # Original D1 reward logic
        ret = (pnl - cost) / equity

        dsr = self.dsr.compute(ret)
        cvar_penalty = self.cvar.compute_penalty(ret)
        dd_penalty = max(0.0, drawdown - self.dd_threshold) * self.dd_scale

        patience = (
            0.001
            if abs(position) < 0.05 and abs(market_return) < 0.002
            else 0.0
        )

        reward = dsr + self.cvar_weight * cvar_penalty - dd_penalty + patience

        return float(np.clip(reward, -1.0, 1.0))

    def reset(self) -> None:
        if self.scalping_reward is not None:
            self.scalping_reward.reset()
        else:
            self.dsr.reset()
            self.cvar.reset()
