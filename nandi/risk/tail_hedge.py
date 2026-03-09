"""Tail risk hedging — automatically reduces exposure during stress."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class TailRiskHedger:
    """Auto-reduces exposure when vol or tail risk exceeds historical norms."""

    def __init__(self, lookback=60, vol_threshold=2.0, tail_threshold=3.0):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.tail_threshold = tail_threshold
        self.returns_buffer = []

    def update(self, portfolio_return):
        """Add latest return to buffer."""
        self.returns_buffer.append(portfolio_return)
        if len(self.returns_buffer) > self.lookback * 3:
            self.returns_buffer = self.returns_buffer[-self.lookback * 3:]

    def compute_hedge_ratio(self, current_vol=None):
        """Compute position scaling factor based on tail risk.

        Returns:
            float in (0, 1]: 1.0 = full size, <1.0 = reduce positions
        """
        if len(self.returns_buffer) < self.lookback:
            return 1.0

        recent = np.array(self.returns_buffer[-self.lookback:])
        historical_vol = np.std(recent)

        # Vol-based scaling
        if current_vol is None:
            current_vol = np.std(self.returns_buffer[-5:]) if len(self.returns_buffer) >= 5 else historical_vol

        vol_ratio = current_vol / (historical_vol + 1e-10)

        if vol_ratio > self.vol_threshold * 1.5:
            hedge = 0.3  # severe stress
        elif vol_ratio > self.vol_threshold:
            hedge = 0.5  # elevated vol
        elif vol_ratio > self.vol_threshold * 0.75:
            hedge = 0.75  # slightly elevated
        else:
            hedge = 1.0  # normal

        # Tail-loss check: reduce if recent worst loss exceeds threshold * historical
        worst_recent = abs(min(recent[-10:])) if len(recent) >= 10 else 0
        worst_historical = abs(np.percentile(recent, 1))

        if worst_historical > 0 and worst_recent / worst_historical > self.tail_threshold:
            hedge = min(hedge, 0.5)
            logger.warning(f"Tail risk alert: worst_recent={worst_recent:.4f} "
                           f"vs historical_1pct={worst_historical:.4f}")

        return hedge

    def reset(self):
        self.returns_buffer = []
