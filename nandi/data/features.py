"""
Per-pair feature engineering — 45 features from OHLCV data.

Refactored from nandi/data.py to support multi-pair pipeline.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_features(df):
    """Compute 45 features for RL agent from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close] and DatetimeIndex.

    Returns:
        DataFrame of features, NaN rows dropped.
    """
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # Returns at multiple horizons
    for d in [1, 2, 5, 10, 20]:
        f[f"ret_{d}d"] = close.pct_change(d)

    f["log_ret"] = np.log(close / close.shift(1))

    # Volatility
    for w in [5, 10, 20, 60]:
        f[f"vol_{w}d"] = f["ret_1d"].rolling(w).std()

    f["vol_ratio"] = f["vol_5d"] / (f["vol_20d"] + 1e-10)
    f["vol_regime"] = f["vol_5d"] / (f["vol_60d"] + 1e-10)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / close

    # Trend (EMA distances, ATR-normalized)
    for p in [5, 10, 20, 50]:
        ema = close.ewm(span=p).mean()
        f[f"ema_{p}_dist"] = (close - ema) / (f["atr_14"] + 1e-10)

    # RSI
    for period in [14, 28]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        f[f"rsi_{period}"] = (100 - 100 / (1 + rs) - 50) / 50

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    f["macd"] = (ema12 - ema26) / (f["atr_14"] + 1e-10)
    f["macd_signal"] = f["macd"].ewm(span=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    f["bb_position"] = (close - bb_mid) / (2 * bb_std + 1e-10)
    f["bb_width"] = (4 * bb_std) / (close + 1e-10)

    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    f["stoch_k"] = (close - low_14) / (high_14 - low_14 + 1e-10)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # ADX
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    plus_di = plus_dm.rolling(14).mean() / (f["atr_14"] + 1e-10)
    minus_di = minus_dm.rolling(14).mean() / (f["atr_14"] + 1e-10)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    f["adx"] = dx.rolling(14).mean()
    f["di_diff"] = plus_di - minus_di

    # Price Action
    rng = high - low + 1e-10
    f["body_ratio"] = (close - open_).abs() / rng
    f["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / rng
    f["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / rng

    # Hurst Exponent
    f["hurst"] = _rolling_hurst(close.values, window=100)

    # Entropy
    f["entropy"] = _rolling_entropy(f["ret_1d"].values, window=60)

    # Calendar (cyclical)
    dow = df.index.dayofweek
    month = df.index.month
    f["day_sin"] = np.sin(2 * np.pi * dow / 5)
    f["day_cos"] = np.cos(2 * np.pi * dow / 5)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)

    f.dropna(inplace=True)
    logger.info(f"Computed {len(f.columns)} features over {len(f)} days")
    return f


def _rolling_hurst(prices, window=100):
    """Rolling Hurst exponent via R/S analysis."""
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(window, n):
        seg = prices[i - window:i]
        rets = np.diff(seg) / seg[:-1]

        if len(rets) < 20 or np.std(rets) < 1e-10:
            result[i] = 0.5
            continue

        mean_r = np.mean(rets)
        devs = np.cumsum(rets - mean_r)
        R = np.max(devs) - np.min(devs)
        S = np.std(rets, ddof=1)

        if S > 0 and R > 0:
            result[i] = np.clip(np.log(R / S) / np.log(len(rets)), 0, 1)
        else:
            result[i] = 0.5

    return result


def _rolling_entropy(returns, window=60, n_bins=20):
    """Rolling Shannon entropy of returns distribution."""
    n = len(returns)
    result = np.full(n, np.nan)

    for i in range(window, n):
        seg = returns[i - window:i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 10:
            result[i] = 1.0
            continue
        counts, _ = np.histogram(seg, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
        result[i] = ent / np.log2(n_bins) if np.log2(n_bins) > 0 else 1.0

    return result
