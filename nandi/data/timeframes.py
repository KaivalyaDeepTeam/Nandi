"""
Multi-timeframe data aggregation (H1/H4/D1).

Phase 4 implementation — for now provides proxy derivation from daily data.
"""

import numpy as np
import pandas as pd
import logging

from nandi.data.features import compute_features

logger = logging.getLogger(__name__)


def derive_h4_proxy(daily_df):
    """Derive H4-like features from daily data by using shorter rolling windows.

    This is a proxy until real H4 data is available from MT5.
    Halves rolling windows to approximate higher-frequency patterns.
    """
    proxy = daily_df.copy()
    # Use shorter windows to simulate higher-frequency view
    proxy["h4_vol_3d"] = proxy["close"].pct_change().rolling(3).std()
    proxy["h4_ema_3_dist"] = (
        (proxy["close"] - proxy["close"].ewm(span=3).mean())
        / (proxy["close"].rolling(7).std() + 1e-10)
    )
    proxy["h4_rsi_7"] = _fast_rsi(proxy["close"], 7)
    return proxy[["h4_vol_3d", "h4_ema_3_dist", "h4_rsi_7"]].dropna()


def derive_h1_proxy(daily_df):
    """Derive H1-like features from daily data using very short rolling windows."""
    proxy = daily_df.copy()
    proxy["h1_vol_2d"] = proxy["close"].pct_change().rolling(2).std()
    proxy["h1_momentum_2d"] = proxy["close"].pct_change(2)
    proxy["h1_range_pct"] = (proxy["high"] - proxy["low"]) / proxy["close"]
    return proxy[["h1_vol_2d", "h1_momentum_2d", "h1_range_pct"]].dropna()


def _fast_rsi(close, period):
    """Quick RSI computation."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return (100 - 100 / (1 + rs) - 50) / 50
