"""
Path Signature Features — from Rough Path Theory (Terry Lyons, Oxford).

Path signatures are the mathematically proven universal feature map for
sequential data. The level-2 signature captures pairwise lead-lag
relationships, momentum structure, and correlation dynamics.

Computes rolling log-signatures at 3 scales from 4-channel path
(close_ret, range, volume_change, atr_change):
  - 6-bar window  (30 min — micro timing):  7 features
  - 12-bar window (1 hour — trade horizon):  7 features
  - 36-bar window (3 hours — session context): 7 features

Total: 21 features.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Signature windows (in M5 bars)
SIG_WINDOWS = [6, 12, 36]

# 4-channel path: close_ret, range, volume_change, atr_change
N_CHANNELS = 4

# Level-1: 4 features (one per channel increment sum)
# Level-2: 3 features (selected cross-channel areas: 01, 02, 13)
# Total per window: 7
FEATURES_PER_WINDOW = 7


def compute_path_signatures(df, atr_series=None):
    """Compute multi-scale path signature features.

    Args:
        df: DataFrame with OHLCV and DatetimeIndex.
        atr_series: optional pre-computed ATR(14) series aligned to df.

    Returns:
        DataFrame with 21 signature features, indexed to match df.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(df)

    # Channel 0: close returns
    close_ret = np.zeros(n)
    close_ret[1:] = np.diff(close) / (close[:-1] + 1e-10)

    # Channel 1: bar range normalized by close
    bar_range = (high - low) / (close + 1e-10)

    # Channel 2: volume change (if available)
    if "volume" in df.columns:
        vol = df["volume"].values.astype(np.float64)
        vol_ma = pd.Series(vol).rolling(12, min_periods=1).mean().values
        vol_change = (vol - vol_ma) / (vol_ma + 1e-10)
    else:
        vol_change = np.zeros(n)

    # Channel 3: ATR change
    if atr_series is not None:
        atr = atr_series.values.astype(np.float64)
    else:
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_change = np.zeros(n)
    atr_change[1:] = np.diff(atr) / (atr[:-1] + 1e-10)

    # Stack channels: (n, 4)
    path = np.column_stack([close_ret, bar_range, vol_change, atr_change])

    # Compute signatures at each scale
    result = pd.DataFrame(index=df.index)

    for window in SIG_WINDOWS:
        feats = _rolling_signature(path, window)
        prefix = f"sig_{window}"
        # Level-1: sum of increments per channel
        result[f"{prefix}_c0"] = feats[:, 0]
        result[f"{prefix}_c1"] = feats[:, 1]
        result[f"{prefix}_c2"] = feats[:, 2]
        result[f"{prefix}_c3"] = feats[:, 3]
        # Level-2: cross-channel signed areas
        result[f"{prefix}_a01"] = feats[:, 4]
        result[f"{prefix}_a02"] = feats[:, 5]
        result[f"{prefix}_a13"] = feats[:, 6]

    logger.info(f"Computed {len(result.columns)} path signature features")
    return result


def _rolling_signature(path, window):
    """Compute rolling level-2 log-signature over a fixed window.

    For a 4-channel path, level-1 signature = sum of increments per channel.
    Level-2 signature = pairwise iterated integrals (signed areas).
    We select 3 most informative cross-channel areas:
      (0,1) close_ret × range — momentum quality
      (0,2) close_ret × volume — volume confirmation
      (1,3) range × atr_change — volatility dynamics

    Args:
        path: (n, 4) array of channel values.
        window: int, rolling window size.

    Returns:
        (n, 7) array of signature features. NaN for first window-1 bars.
    """
    n = path.shape[0]
    out = np.full((n, FEATURES_PER_WINDOW), np.nan, dtype=np.float64)

    for t in range(window, n):
        segment = path[t - window:t]  # (window, 4)
        increments = np.diff(segment, axis=0)  # (window-1, 4)

        # Level-1: cumulative sum of increments (= total displacement)
        level1 = increments.sum(axis=0)  # (4,)

        # Level-2: signed areas via iterated integrals
        # S^{ij} = sum_{k<l} dx_k^i * dx_l^j
        # Efficient: cumsum trick
        cum = np.cumsum(increments, axis=0)  # (window-1, 4)

        # Area(i,j) = sum_l (cum[l-1, i] * inc[l, j])
        # where cum[l-1, i] = sum of dx_k^i for k < l
        m = len(increments)
        if m < 2:
            continue

        cum_prev = np.zeros_like(cum)
        cum_prev[1:] = cum[:-1]

        area_01 = np.sum(cum_prev[:, 0] * increments[:, 1])
        area_02 = np.sum(cum_prev[:, 0] * increments[:, 2])
        area_13 = np.sum(cum_prev[:, 1] * increments[:, 3])

        out[t, :4] = level1
        out[t, 4] = area_01
        out[t, 5] = area_02
        out[t, 6] = area_13

    # Normalize by window to make features scale-invariant
    out /= max(window, 1)

    return out
