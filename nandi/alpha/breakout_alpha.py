"""Breakout alpha — Donchian channel breakout with volatility confirmation."""

import logging
import numpy as np
from nandi.alpha.base import BaseAlpha, AlphaSignal

logger = logging.getLogger(__name__)


class BreakoutAlpha(BaseAlpha):
    """Donchian channel breakout with ATR-based volatility filter."""

    def __init__(self, pairs, channel_period=20, atr_threshold=0.3):
        super().__init__(name="breakout", pairs=pairs)
        self.channel_period = channel_period
        self.atr_threshold = atr_threshold

    def generate(self, features, **kwargs):
        """Generate breakout signals from feature dicts.

        Expects features to contain: bb_position, atr_pct, vol_ratio, adx
        """
        signals = []
        for pair in self.pairs:
            feat = features.get(pair)
            if feat is None:
                continue

            try:
                # Use Bollinger band position as breakout proxy
                if isinstance(feat, dict):
                    bb_pos = feat.get("bb_position", 0)
                    atr_pct = feat.get("atr_pct", 0)
                    vol_ratio = feat.get("vol_ratio", 1)
                    adx = feat.get("adx", 0)
                elif isinstance(feat, tuple) and len(feat) == 2:
                    # (market_state, position_info) format
                    market_state = feat[0]
                    last_row = market_state[-1] if market_state.ndim == 2 else market_state
                    bb_pos = float(last_row[23]) if len(last_row) > 23 else 0  # approximate index
                    atr_pct = float(last_row[9]) if len(last_row) > 9 else 0
                    vol_ratio = float(last_row[6]) if len(last_row) > 6 else 1
                    adx = float(last_row[26]) if len(last_row) > 26 else 0
                else:
                    continue

                # Only trigger on strong breakouts
                if abs(bb_pos) < 1.0:
                    continue

                # Volatility expansion confirms breakout
                if vol_ratio < 1.2:
                    continue

                # ADX confirms trend
                if adx < 0.2:
                    continue

                direction = 1.0 if bb_pos > 1.0 else -1.0
                confidence = min(1.0, abs(bb_pos) / 2.0) * min(1.0, adx / 0.5)

                signals.append(AlphaSignal(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    alpha_name="breakout",
                    metadata={"bb_pos": bb_pos, "vol_ratio": vol_ratio, "adx": adx},
                ))
            except (IndexError, TypeError, KeyError):
                continue

        return signals
