"""Execution quality tracking — measures slippage and fill quality."""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionQualityTracker:
    """Tracks execution quality: slippage, fill times, intended vs actual prices."""

    def __init__(self):
        self.fills = []

    def record_fill(self, intended_price, actual_price, size, direction, pair="unknown"):
        """Record a fill for quality analysis."""
        slippage_bps = abs(actual_price - intended_price) / intended_price * 10000
        self.fills.append({
            "timestamp": time.time(),
            "pair": pair,
            "intended": intended_price,
            "actual": actual_price,
            "slippage_bps": slippage_bps,
            "size": size,
            "direction": direction,
        })
        if slippage_bps > 5.0:
            logger.warning(f"High slippage on {pair}: {slippage_bps:.1f}bps "
                          f"(intended={intended_price:.5f}, actual={actual_price:.5f})")

    def get_stats(self):
        """Get aggregate execution quality statistics."""
        if not self.fills:
            return {"n_fills": 0}
        slippages = [f["slippage_bps"] for f in self.fills]
        return {
            "n_fills": len(self.fills),
            "mean_slippage_bps": float(np.mean(slippages)),
            "median_slippage_bps": float(np.median(slippages)),
            "p95_slippage_bps": float(np.percentile(slippages, 95)),
            "max_slippage_bps": float(np.max(slippages)),
            "total_slippage_bps": float(np.sum(slippages)),
        }

    def get_pair_stats(self):
        """Get per-pair execution quality statistics."""
        pair_fills = {}
        for f in self.fills:
            pair = f["pair"]
            if pair not in pair_fills:
                pair_fills[pair] = []
            pair_fills[pair].append(f["slippage_bps"])

        result = {}
        for pair, slippages in pair_fills.items():
            result[pair] = {
                "n_fills": len(slippages),
                "mean_slippage_bps": float(np.mean(slippages)),
                "p95_slippage_bps": float(np.percentile(slippages, 95)),
            }
        return result

    def reset(self):
        """Clear all recorded fills."""
        self.fills = []
