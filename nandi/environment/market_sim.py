"""
Realistic market simulation with per-pair spreads, slippage, and market impact.

Replaces the naive fixed-cost model to produce realistic backtest results.
Supports timeframe-aware spread scaling (tighter during London+NY overlap).
"""

import numpy as np
from nandi.config import (
    PAIR_SPREADS, SLIPPAGE_BPS, TRANSACTION_COST_BPS,
    TIMEFRAME_PROFILES, SCALPING_CONFIG,
)


class MarketSimulator:
    """Models realistic execution costs: spread + slippage + market impact."""

    def __init__(self, pair_name="eurusd", timeframe="D1"):
        self.pair_name = pair_name
        self.timeframe = timeframe
        profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["D1"])
        self.spread_multiplier = profile.get("spread_multiplier", 1.0)
        self.spread_pips = PAIR_SPREADS.get(pair_name, 1.5) * self.spread_multiplier
        self.slippage_bps = SLIPPAGE_BPS
        self.base_cost_bps = TRANSACTION_COST_BPS

    def get_session_spread(self, hour_utc=None):
        """Get session-adjusted spread in pips.

        For intraday timeframes, spread varies by session:
        - Asian session: 1.5x wider (low liquidity)
        - London+NY overlap: 0.8x tighter (peak liquidity)
        - D1: always base spread (no session adjustment)
        """
        if self.timeframe == "D1" or hour_utc is None:
            return self.spread_pips

        london_open = SCALPING_CONFIG["london_open_utc"]
        london_close = SCALPING_CONFIG["london_close_utc"]
        ny_open = SCALPING_CONFIG["ny_open_utc"]
        ny_close = SCALPING_CONFIG["ny_close_utc"]

        in_london = london_open <= hour_utc < london_close
        in_ny = ny_open <= hour_utc < ny_close

        if in_london and in_ny:
            return self.spread_pips * 0.8  # overlap — tightest
        elif in_london or in_ny:
            return self.spread_pips  # normal
        else:
            return self.spread_pips * 1.5  # Asian — widest

    def get_execution_price(self, mid_price, direction, size, volatility=0.0,
                            hour_utc=None):
        """Returns fill price accounting for spread and slippage.

        Args:
            mid_price: current mid price
            direction: +1 for buy, -1 for sell
            size: position size as fraction [0, 1]
            volatility: recent absolute return for vol-scaled slippage
            hour_utc: optional hour for session-aware spread (intraday only)
        """
        # Half-spread cost (bid-ask)
        pip_value = 0.0001 if "jpy" not in self.pair_name else 0.01
        spread = self.get_session_spread(hour_utc)
        half_spread = spread * pip_value / 2

        # Volatility-scaled slippage: higher vol = more slippage
        vol_multiplier = 1.0 + min(volatility / 0.01, 2.0)  # caps at 3x in high vol
        slippage = mid_price * self.slippage_bps / 10000 * abs(size) * vol_multiplier

        if direction > 0:  # buying at ask
            return mid_price + half_spread + slippage
        else:  # selling at bid
            return mid_price - half_spread - slippage

    def get_total_cost(self, mid_price, old_position, new_position, volatility=0.0):
        """Total cost of position change as fraction of equity.

        Returns cost as a decimal fraction (e.g., 0.0003 for 3bps).
        """
        position_change = abs(new_position - old_position)
        if position_change < 0.01:
            return 0.0

        # Base transaction cost
        base_cost = position_change * self.base_cost_bps / 10000

        # Spread cost (proportional to position change)
        pip_value = 0.0001 if "jpy" not in self.pair_name else 0.01
        spread_cost = self.spread_pips * pip_value / mid_price * position_change

        # Slippage (vol-scaled)
        vol_multiplier = 1.0 + min(volatility / 0.01, 2.0)
        slippage_cost = self.slippage_bps / 10000 * position_change * vol_multiplier

        return base_cost + spread_cost + slippage_cost
