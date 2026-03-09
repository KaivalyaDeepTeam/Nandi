"""
Momentum Alpha — EMA crossover + ADX-weighted confidence.

Phase 2 implementation: pure NumPy, no ML.
"""

from typing import List

import numpy as np
import pandas as pd

from nandi.alpha.base import BaseAlpha, AlphaSignal


class MomentumAlpha(BaseAlpha):
    """EMA(5)/EMA(20) crossover with ADX confidence weighting."""

    def __init__(self, pairs, fast_period=5, slow_period=20, adx_threshold=0.2):
        super().__init__(name="momentum", pairs=pairs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_threshold = adx_threshold

    def generate(self, features, **kwargs) -> List[AlphaSignal]:
        """Generate momentum signals.

        Args:
            features: {pair: pd.DataFrame} with columns including
                      ema_5_dist, ema_20_dist, adx, di_diff.
        """
        signals = []

        for pair in self.pairs:
            feat = features.get(pair)
            if feat is None:
                continue

            if isinstance(feat, pd.DataFrame):
                row = feat.iloc[-1]
            elif isinstance(feat, dict):
                row = feat
            else:
                continue

            ema_fast = row.get("ema_5_dist", 0)
            ema_slow = row.get("ema_20_dist", 0)
            adx = row.get("adx", 0)
            di_diff = row.get("di_diff", 0)

            # EMA crossover direction
            direction = 1.0 if ema_fast > ema_slow else -1.0

            # Confirm with DI direction
            if di_diff * direction < 0:
                direction = 0.0

            # ADX-weighted confidence (higher ADX = stronger trend = higher confidence)
            adx_val = float(adx)
            if adx_val < self.adx_threshold:
                confidence = 0.1
            else:
                confidence = min(1.0, adx_val / 0.5)

            if abs(direction) > 0:
                signals.append(AlphaSignal(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    alpha_name=self.name,
                ))

        return signals
