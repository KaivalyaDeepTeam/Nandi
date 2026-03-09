"""
Mean Reversion Alpha — Bollinger Band fading + RSI, gated by Hurst < 0.45.

Phase 2 implementation: only trades when market is mean-reverting.
"""

from typing import List

import pandas as pd

from nandi.alpha.base import BaseAlpha, AlphaSignal


class MeanReversionAlpha(BaseAlpha):
    """Fade Bollinger Band extremes, only when Hurst exponent < 0.45."""

    def __init__(self, pairs, hurst_threshold=0.45, bb_threshold=0.8, rsi_threshold=0.3):
        super().__init__(name="mean_reversion", pairs=pairs)
        self.hurst_threshold = hurst_threshold
        self.bb_threshold = bb_threshold
        self.rsi_threshold = rsi_threshold

    def generate(self, features, **kwargs) -> List[AlphaSignal]:
        """Generate mean reversion signals.

        Args:
            features: {pair: pd.DataFrame} with bb_position, rsi_14, hurst columns.
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

            hurst = float(row.get("hurst", 0.5))
            bb_pos = float(row.get("bb_position", 0))
            rsi = float(row.get("rsi_14", 0))

            # Only trade in mean-reverting regimes
            if hurst > self.hurst_threshold:
                continue

            direction = 0.0
            confidence = 0.0

            # Overbought: fade short
            if bb_pos > self.bb_threshold and rsi > self.rsi_threshold:
                direction = -1.0
                confidence = min(1.0, abs(bb_pos))

            # Oversold: fade long
            elif bb_pos < -self.bb_threshold and rsi < -self.rsi_threshold:
                direction = 1.0
                confidence = min(1.0, abs(bb_pos))

            if abs(direction) > 0:
                signals.append(AlphaSignal(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    alpha_name=self.name,
                    metadata={"hurst": hurst, "bb_pos": bb_pos, "rsi": rsi},
                ))

        return signals
