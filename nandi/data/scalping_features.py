"""
Scalping feature engine — profile-driven features for M5 (and other intraday) timeframes.

Replaces daily calendar features with intraday-specific signals:
session encoding, spread normalization, micro-momentum, micro-mean-reversion.
"""

import numpy as np
import pandas as pd
import logging

from nandi.config import TIMEFRAME_PROFILES, SCALPING_CONFIG

logger = logging.getLogger(__name__)


def compute_scalping_features(df, profile=None):
    """Compute scalping features from M5 OHLCV DataFrame.

    Uses profile-driven windows (not hardcoded daily windows).

    Args:
        df: DataFrame with columns [open, high, low, close] and DatetimeIndex.
        profile: timeframe profile dict (defaults to TIMEFRAME_PROFILES["M5"]).

    Returns:
        DataFrame of features, NaN rows dropped.
    """
    if profile is None:
        profile = TIMEFRAME_PROFILES["M5"]

    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    ret_windows = profile["feature_windows"]["returns"]
    vol_windows = profile["feature_windows"]["vol"]

    # ── Returns at profile-defined windows ──
    for w in ret_windows:
        f[f"ret_{w}b"] = close.pct_change(w)

    f["log_ret"] = np.log(close / close.shift(1))

    # ── Volatility at profile-defined windows ──
    ret_1 = close.pct_change(1)
    for w in vol_windows:
        f[f"vol_{w}b"] = ret_1.rolling(w).std()

    # Vol ratio: short/long
    if len(vol_windows) >= 2:
        f["vol_ratio"] = f[f"vol_{vol_windows[0]}b"] / (f[f"vol_{vol_windows[-1]}b"] + 1e-10)

    # ── ATR (14-bar) ──
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / (close + 1e-10)

    # ── Trend (EMA distances) ──
    for p in [6, 12, 36, 72]:
        ema = close.ewm(span=p).mean()
        f[f"ema_{p}_dist"] = (close - ema) / (f["atr_14"] + 1e-10)

    # ── RSI (14-bar, 28-bar) ──
    for period in [14, 28]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        f[f"rsi_{period}"] = (100 - 100 / (1 + rs) - 50) / 50

    # ── MACD (adapted for M5: 12/26/9 bars) ──
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    f["macd"] = (ema12 - ema26) / (f["atr_14"] + 1e-10)
    f["macd_signal"] = f["macd"].ewm(span=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # ── Bollinger Bands (20-bar) ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    f["bb_position"] = (close - bb_mid) / (2 * bb_std + 1e-10)
    f["bb_width"] = (4 * bb_std) / (close + 1e-10)

    # ── Stochastic (14-bar) ──
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    f["stoch_k"] = (close - low_14) / (high_14 - low_14 + 1e-10)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # ── Price Action ──
    rng = high - low + 1e-10
    f["body_ratio"] = (close - open_).abs() / rng
    f["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / rng
    f["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / rng

    # ════════════════════════════════════════════
    # INTRADAY-ONLY FEATURES (not in D1)
    # ════════════════════════════════════════════

    # ── Session sin/cos (time-of-day cyclical encoding) ──
    if hasattr(df.index, 'hour'):
        minutes_of_day = df.index.hour * 60 + df.index.minute
        f["session_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
        f["session_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

        # Session type: 0=Asian, 1=London, 2=NY, 3=overlap
        f["session_type"] = _classify_session(df.index.hour)
    else:
        # Fallback for non-datetime index
        f["session_sin"] = 0.0
        f["session_cos"] = 0.0
        f["session_type"] = 0.0

    # ── Spread normalized (current spread vs rolling average) ──
    spread = high - low  # proxy for spread using bar range
    spread_ma = spread.rolling(72).mean()  # 6h rolling average
    f["spread_normalized"] = spread / (spread_ma + 1e-10)

    # ── Bar volume z-score (relative volume) ──
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(72).mean()
        vol_std = vol.rolling(72).std()
        f["bar_volume_zscore"] = (vol - vol_ma) / (vol_std + 1e-10)
    else:
        f["bar_volume_zscore"] = 0.0

    # ── Momentum micro (3-bar momentum for quick entries) ──
    f["momentum_micro"] = close.pct_change(3) / (f["atr_pct"] + 1e-10)

    # ── Mean reversion micro (6-bar z-score) ──
    close_ma6 = close.rolling(6).mean()
    close_std6 = close.rolling(6).std()
    f["mean_rev_micro"] = (close - close_ma6) / (close_std6 + 1e-10)

    f.dropna(inplace=True)
    logger.info(f"Computed {len(f.columns)} scalping features over {len(f)} bars")
    return f


def _classify_session(hours):
    """Classify hour into session type.

    Returns: Series of 0=Asian, 1=London, 2=NY, 3=overlap (London+NY).
    """
    london_open = SCALPING_CONFIG["london_open_utc"]
    london_close = SCALPING_CONFIG["london_close_utc"]
    ny_open = SCALPING_CONFIG["ny_open_utc"]
    ny_close = SCALPING_CONFIG["ny_close_utc"]

    result = np.zeros(len(hours), dtype=np.float32)
    hours_arr = np.array(hours)

    in_london = (hours_arr >= london_open) & (hours_arr < london_close)
    in_ny = (hours_arr >= ny_open) & (hours_arr < ny_close)
    overlap = in_london & in_ny

    result[in_london & ~overlap] = 1.0
    result[in_ny & ~overlap] = 2.0
    result[overlap] = 3.0

    return result
