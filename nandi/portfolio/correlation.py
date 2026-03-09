"""
Rolling cross-pair correlation matrix tracking.
"""

import numpy as np
import pandas as pd
import logging

from nandi.config import PORTFOLIO_RISK

logger = logging.getLogger(__name__)


class CorrelationTracker:
    """Tracks rolling correlations between pairs for exposure management."""

    def __init__(self, pairs, window=20):
        self.pairs = pairs
        self.window = window
        self.returns_buffer = {pair: [] for pair in pairs}
        self.correlation_matrix = None

    def update(self, pair_returns):
        """Update with latest daily returns.

        Args:
            pair_returns: {pair: float} latest daily return per pair.
        """
        for pair in self.pairs:
            ret = pair_returns.get(pair, 0.0)
            self.returns_buffer[pair].append(ret)
            # Keep only last window
            if len(self.returns_buffer[pair]) > self.window:
                self.returns_buffer[pair] = self.returns_buffer[pair][-self.window:]

        # Recompute correlation matrix
        if all(len(v) >= self.window for v in self.returns_buffer.values()):
            df = pd.DataFrame({
                p: self.returns_buffer[p][-self.window:]
                for p in self.pairs
            })
            self.correlation_matrix = df.corr()

    def get_correlation(self, pair_a, pair_b):
        """Get rolling correlation between two pairs."""
        if self.correlation_matrix is None:
            return 0.0
        if pair_a in self.correlation_matrix.columns and pair_b in self.correlation_matrix.columns:
            return float(self.correlation_matrix.loc[pair_a, pair_b])
        return 0.0

    def adjust_for_correlation(self, positions):
        """Reduce positions for highly correlated pairs.

        If two pairs have correlation > 0.7 and same-direction positions,
        reduce both to stay within correlated exposure limit.
        """
        if self.correlation_matrix is None:
            return positions

        adjusted = dict(positions)
        max_corr_exposure = PORTFOLIO_RISK.get("max_correlated_exposure", 2.0)

        pairs = list(adjusted.keys())
        for i, pair_a in enumerate(pairs):
            for pair_b in pairs[i + 1:]:
                corr = self.get_correlation(pair_a, pair_b)
                if abs(corr) > 0.7:
                    pos_a = adjusted.get(pair_a, 0)
                    pos_b = adjusted.get(pair_b, 0)

                    # Same direction = correlated risk
                    if pos_a * pos_b > 0:
                        combined = abs(pos_a) + abs(pos_b)
                        if combined > max_corr_exposure:
                            scale = max_corr_exposure / combined
                            adjusted[pair_a] = pos_a * scale
                            adjusted[pair_b] = pos_b * scale

        return adjusted

    def from_dataframe(self, closes_df):
        """Initialize from historical closes DataFrame."""
        returns = closes_df.pct_change().dropna()
        if len(returns) >= self.window:
            self.correlation_matrix = returns.tail(self.window).corr()
            for pair in self.pairs:
                if pair in returns.columns:
                    self.returns_buffer[pair] = returns[pair].tail(self.window).tolist()
