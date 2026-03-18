"""
Higher-Timeframe Context Features — what real traders check before entering.

Computes H1, H4, D1 trend/momentum/volatility from M5 data by resampling.
No external data dependency.

Features (8 total):
  H1: trend direction, momentum, volatility (3)
  H4: trend direction, momentum, range position (3)
  D1: trend direction, average daily range (2)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_htf_context(df):
    """Compute higher-timeframe context features from M5 OHLCV.

    Args:
        df: M5 DataFrame with OHLCV and DatetimeIndex.

    Returns:
        DataFrame with 8 HTF context features, indexed to match df.
    """
    result = pd.DataFrame(index=df.index)

    # ── H1 context (12 M5 bars) ──
    h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(h1) > 20:
        h1_ema12 = h1["close"].ewm(span=12).mean()
        h1_ema26 = h1["close"].ewm(span=26).mean()
        h1_atr = _atr(h1, 14)

        # Trend: EMA12 - EMA26 normalized by ATR
        h1_trend = (h1_ema12 - h1_ema26) / (h1_atr + 1e-10)
        # Momentum: 4-bar return normalized by ATR
        h1_mom = h1["close"].pct_change(4) / (h1_atr / h1["close"] + 1e-10)
        # Volatility: ATR percent
        h1_vol = h1_atr / (h1["close"] + 1e-10)

        result["htf_h1_trend"] = h1_trend.reindex(df.index, method="ffill")
        result["htf_h1_momentum"] = h1_mom.reindex(df.index, method="ffill")
        result["htf_h1_volatility"] = h1_vol.reindex(df.index, method="ffill")
    else:
        result["htf_h1_trend"] = 0.0
        result["htf_h1_momentum"] = 0.0
        result["htf_h1_volatility"] = 0.0

    # ── H4 context (48 M5 bars) ──
    h4 = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(h4) > 20:
        h4_ema8 = h4["close"].ewm(span=8).mean()
        h4_ema21 = h4["close"].ewm(span=21).mean()
        h4_atr = _atr(h4, 14)

        # Trend: EMA8 - EMA21 normalized by ATR
        h4_trend = (h4_ema8 - h4_ema21) / (h4_atr + 1e-10)
        # Momentum: 3-bar return normalized
        h4_mom = h4["close"].pct_change(3) / (h4_atr / h4["close"] + 1e-10)
        # Range position: where close is within H4 range
        h4_high_20 = h4["high"].rolling(20).max()
        h4_low_20 = h4["low"].rolling(20).min()
        h4_range_pos = (h4["close"] - h4_low_20) / (h4_high_20 - h4_low_20 + 1e-10)

        result["htf_h4_trend"] = h4_trend.reindex(df.index, method="ffill")
        result["htf_h4_momentum"] = h4_mom.reindex(df.index, method="ffill")
        result["htf_h4_range_pos"] = h4_range_pos.reindex(df.index, method="ffill")
    else:
        result["htf_h4_trend"] = 0.0
        result["htf_h4_momentum"] = 0.0
        result["htf_h4_range_pos"] = 0.5

    # ── D1 context (288 M5 bars) ──
    d1 = df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(d1) > 20:
        d1_ema10 = d1["close"].ewm(span=10).mean()
        d1_ema30 = d1["close"].ewm(span=30).mean()
        d1_atr = _atr(d1, 14)

        # Trend direction
        d1_trend = (d1_ema10 - d1_ema30) / (d1_atr + 1e-10)
        # Average daily range (volatility)
        d1_adr = d1_atr / (d1["close"] + 1e-10)

        result["htf_d1_trend"] = d1_trend.reindex(df.index, method="ffill")
        result["htf_d1_adr"] = d1_adr.reindex(df.index, method="ffill")
    else:
        result["htf_d1_trend"] = 0.0
        result["htf_d1_adr"] = 0.0

    # Clip extreme values
    for col in result.columns:
        result[col] = result[col].clip(-5.0, 5.0)

    result.fillna(0.0, inplace=True)

    logger.info(f"Computed {len(result.columns)} HTF context features")
    return result


def compute_h1_trend_series(df):
    """Compute H1 trend direction series for environment risk gates.

    Returns a Series with values: +1 (uptrend), -1 (downtrend), 0 (neutral).
    Reindexed to M5 bars via forward-fill.
    """
    h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",
    }).dropna()

    if len(h1) < 26:
        return pd.Series(0, index=df.index, dtype=np.float32)

    ema12 = h1["close"].ewm(span=12).mean()
    ema26 = h1["close"].ewm(span=26).mean()
    atr = _atr(h1, 14)

    # Trend signal: normalized EMA cross
    trend_strength = (ema12 - ema26) / (atr + 1e-10)

    # Discretize: >0.3 = uptrend, <-0.3 = downtrend, else neutral
    trend = pd.Series(0, index=h1.index, dtype=np.float32)
    trend[trend_strength > 0.3] = 1.0
    trend[trend_strength < -0.3] = -1.0

    return trend.reindex(df.index, method="ffill").fillna(0.0)


def _atr(df, period=14):
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()
