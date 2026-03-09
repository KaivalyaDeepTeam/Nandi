"""Sentiment proxy features derived from price action.

No external API needed — uses market microstructure as sentiment proxy.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_sentiment_features(df, window=20):
    """Compute sentiment proxy features from OHLCV data.

    Args:
        df: DataFrame with open, high, low, close columns
        window: rolling window for computations

    Returns:
        DataFrame with sentiment features
    """
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bearish ratio: fraction of down days in recent window
    features["bearish_ratio"] = (close.diff() < 0).astype(float).rolling(window).mean()

    # Fear index: vol expansion relative to longer-term vol
    short_vol = close.pct_change().rolling(window).std()
    long_vol = close.pct_change().rolling(window * 3).std()
    features["fear_index"] = (short_vol / (long_vol + 1e-10)).clip(0, 5) / 5

    # Buying pressure: close position within range (Williams %R style)
    range_high = high.rolling(window).max()
    range_low = low.rolling(window).min()
    features["buying_pressure"] = (close - range_low) / (range_high - range_low + 1e-10)

    # Momentum exhaustion: declining range during trend
    avg_range = (high - low).rolling(window).mean()
    trend_strength = abs(close - close.shift(window)) / (avg_range * window + 1e-10)
    features["momentum_exhaustion"] = (1 - trend_strength).clip(0, 1)

    # Gap indicator: overnight gap as fraction of ATR
    gap = (df["open"] - close.shift(1)).abs()
    atr = (high - low).rolling(14).mean()
    features["gap_ratio"] = (gap / (atr + 1e-10)).clip(0, 3) / 3

    return features.dropna()
