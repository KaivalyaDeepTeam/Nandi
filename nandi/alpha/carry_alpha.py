"""Carry trade alpha — interest rate differential signal."""

import logging
from nandi.alpha.base import BaseAlpha, AlphaSignal

logger = logging.getLogger(__name__)


class CarryAlpha(BaseAlpha):
    """Carry trade: buy high-yielding currencies, sell low-yielding ones.

    Uses approximate interest rate differentials.
    Carry is updated periodically based on central bank rate changes.
    """

    # Approximate annual rate differentials (positive = long bias for the pair)
    # Updated periodically; these are rough approximations
    RATE_DIFFERENTIALS = {
        "eurusd": -0.01,    # EUR rate slightly below USD
        "gbpusd": 0.005,    # GBP rate roughly equal
        "usdjpy": 0.04,     # JPY near zero, USD higher
        "audusd": 0.01,     # AUD slightly above USD
        "nzdusd": 0.015,    # NZD slightly above USD
        "usdchf": 0.02,     # CHF near zero, USD higher
        "usdcad": 0.005,    # CAD roughly equal
        "eurjpy": 0.03,     # JPY near zero, EUR higher
    }

    def __init__(self, pairs, min_carry=0.005):
        super().__init__(name="carry", pairs=pairs)
        self.min_carry = min_carry

    def generate(self, features=None, **kwargs):
        """Generate carry signals based on rate differentials.

        Carry trades work best in low-volatility environments,
        so we reduce confidence when vol is high.
        """
        signals = []
        for pair in self.pairs:
            carry = self.RATE_DIFFERENTIALS.get(pair, 0)

            if abs(carry) < self.min_carry:
                continue

            direction = 1.0 if carry > 0 else -1.0
            confidence = min(1.0, abs(carry) / 0.05)  # normalize to [0, 1]

            # Reduce confidence if features suggest high vol
            if features and pair in features:
                feat = features[pair]
                if isinstance(feat, tuple) and len(feat) == 2:
                    market_state = feat[0]
                    last_row = market_state[-1] if market_state.ndim == 2 else market_state
                    vol_regime = float(last_row[7]) if len(last_row) > 7 else 1.0
                    if vol_regime > 1.5:
                        confidence *= 0.3  # reduce in high vol

            signals.append(AlphaSignal(
                pair=pair,
                direction=direction,
                confidence=confidence,
                alpha_name="carry",
                metadata={"rate_differential": carry},
            ))

        return signals
