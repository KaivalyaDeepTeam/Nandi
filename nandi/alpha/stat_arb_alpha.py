"""
Statistical Arbitrage Alpha — cross-pair spread z-score trading.

Phase 2 implementation: trades when correlated pairs diverge beyond threshold.
"""

from typing import List

import numpy as np

from nandi.alpha.base import BaseAlpha, AlphaSignal
from nandi.config import PAIR_GROUPS


class StatArbAlpha(BaseAlpha):
    """Trade spread z-scores between correlated pair groups."""

    def __init__(self, pairs, z_threshold=2.0, z_exit=0.5):
        super().__init__(name="stat_arb", pairs=pairs)
        self.z_threshold = z_threshold
        self.z_exit = z_exit

    def generate(self, features, spread_zscores=None, **kwargs) -> List[AlphaSignal]:
        """Generate stat arb signals from spread z-scores.

        Args:
            features: not used directly; kept for ABC compat.
            spread_zscores: pd.DataFrame with z-scores per pair group,
                           or dict {group_name: float}.
        """
        signals = []

        if spread_zscores is None:
            return signals

        for group_name, (pair_a, pair_b) in PAIR_GROUPS.items():
            if pair_a not in self.pairs or pair_b not in self.pairs:
                continue

            # Get z-score for this group
            if hasattr(spread_zscores, 'iloc'):
                z = float(spread_zscores[group_name].iloc[-1])
            elif isinstance(spread_zscores, dict):
                z = float(spread_zscores.get(group_name, 0))
            else:
                continue

            if abs(z) < self.z_threshold:
                continue

            # Spread too wide: mean revert
            # If z > threshold: pair_a overpriced relative to pair_b -> short A, long B
            # If z < -threshold: pair_a underpriced -> long A, short B
            confidence = min(1.0, abs(z) / (self.z_threshold * 2))

            if z > self.z_threshold:
                signals.append(AlphaSignal(
                    pair=pair_a, direction=-1.0, confidence=confidence,
                    alpha_name=self.name, metadata={"z_score": z, "group": group_name},
                ))
                signals.append(AlphaSignal(
                    pair=pair_b, direction=1.0, confidence=confidence,
                    alpha_name=self.name, metadata={"z_score": z, "group": group_name},
                ))
            elif z < -self.z_threshold:
                signals.append(AlphaSignal(
                    pair=pair_a, direction=1.0, confidence=confidence,
                    alpha_name=self.name, metadata={"z_score": z, "group": group_name},
                ))
                signals.append(AlphaSignal(
                    pair=pair_b, direction=-1.0, confidence=confidence,
                    alpha_name=self.name, metadata={"z_score": z, "group": group_name},
                ))

        return signals
