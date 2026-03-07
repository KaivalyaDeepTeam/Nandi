"""
Feature engineering combining classical technical analysis with
advanced mathematical features from fractal theory and information theory.
"""

import numpy as np
import pandas as pd
import ta

from src.advanced_features import (
    rolling_hurst,
    returns_entropy,
    permutation_entropy,
    detect_regime,
    wavelet_energy,
    rolling_fractal_dimension,
    market_quality_index,
)


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features — classical + advanced mathematical."""
    df = df.copy()
    close = df["close"].values

    # ── Time Features ────────────────────────────────────────────
    if hasattr(df.index, 'hour'):
        h = df.index.hour
        dow = df.index.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
        df["session_london"] = ((h >= 7) & (h < 16)).astype(int)
        df["session_ny"] = ((h >= 13) & (h < 22)).astype(int)
        df["session_overlap"] = ((h >= 13) & (h < 16)).astype(int)

    # ── Returns ──────────────────────────────────────────────────
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)

    # ── Volatility ───────────────────────────────────────────────
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_ratio"] = df["vol_5"] / (df["vol_20"] + 1e-10)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_pct"] = abs(df["close"] - df["open"]) / df["close"]

    # ── Trend (ATR-normalized) ───────────────────────────────────
    ema_9 = ta.trend.ema_indicator(df["close"], window=9)
    ema_21 = ta.trend.ema_indicator(df["close"], window=21)
    ema_50 = ta.trend.ema_indicator(df["close"], window=50)

    df["dist_ema9"] = (df["close"] - ema_9) / df["atr"]
    df["dist_ema21"] = (df["close"] - ema_21) / df["atr"]
    df["dist_ema50"] = (df["close"] - ema_50) / df["atr"]
    df["ema_spread"] = (ema_9 - ema_21) / df["atr"]
    df["ema_slope"] = ema_21.pct_change(5) * 1000

    # MACD normalized
    macd_obj = ta.trend.MACD(df["close"])
    df["macd_hist"] = macd_obj.macd_diff() / df["atr"]

    # ADX
    adx_obj = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    adx_raw = adx_obj.adx()
    df["adx"] = adx_raw / 100.0
    df["adx_diff"] = (adx_obj.adx_pos() - adx_obj.adx_neg()) / 100.0

    # ── Oscillators (already bounded) ────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14) / 100.0
    df["rsi_6"] = ta.momentum.rsi(df["close"], window=6) / 100.0

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch() / 100.0
    df["stoch_d"] = stoch.stoch_signal() / 100.0

    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20) / 200.0
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"]) / 100.0

    # ── Mean Reversion ───────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    roll_mean = df["close"].rolling(20).mean()
    roll_std = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"] - roll_mean) / (roll_std + 1e-10)

    # ══════════════════════════════════════════════════════════════
    # ADVANCED MATHEMATICAL FEATURES
    # ══════════════════════════════════════════════════════════════

    # ── Hurst Exponent (Fractal Memory) ──────────────────────────
    # H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random
    df["hurst"] = rolling_hurst(close, window=100, max_lag=15)
    df["hurst_regime"] = np.where(df["hurst"] > 0.55, 1,
                                  np.where(df["hurst"] < 0.45, -1, 0))

    # ── Shannon Entropy (Predictability) ─────────────────────────
    returns = df["ret_1"].values
    df["entropy"] = returns_entropy(returns, window=50, bins=10)

    # ── Permutation Entropy (Temporal Patterns) ──────────────────
    df["perm_entropy"] = permutation_entropy(close, order=3, delay=1, window=50)

    # ── Fractal Dimension (Complexity) ───────────────────────────
    df["fractal_dim"] = rolling_fractal_dimension(close, window=80, k_max=6)

    # ── Wavelet Energy (Dominant Cycles) ─────────────────────────
    wave_e = wavelet_energy(close, window=50)
    df["noise_ratio"] = wave_e["noise_ratio"]

    # ── Market Regime ────────────────────────────────────────────
    regimes = detect_regime(
        close, df["atr"].values, adx_raw.values,
        df["hurst"].values, df["entropy"].values
    )
    df["regime"] = regimes
    # One-hot encode regimes
    for r in range(4):
        df[f"regime_{r}"] = (regimes == r).astype(int)

    # ── Market Quality Index (Tradability) ───────────────────────
    df["market_quality"] = market_quality_index(
        df["hurst"].values, df["entropy"].values,
        df["fractal_dim"].values, adx_raw.values
    )

    # ── Lag Features ─────────────────────────────────────────────
    for lag in [1, 2, 3]:
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi_14"].shift(lag)

    return df


def create_target(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Create target using forward returns normalized by ATR."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = df["future_close"] - df["close"]

    threshold = df["atr"] * 0.3
    df["target"] = np.where(
        df["future_return"] > threshold, 1,
        np.where(df["future_return"] < -threshold, 0, np.nan)
    )
    return df


def prepare_features(df: pd.DataFrame, horizon: int = 12) -> tuple:
    """Full pipeline: add features, create target, clean, return X and y."""
    df = add_all_features(df)
    df = create_target(df, horizon)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    exclude_cols = {"open", "high", "low", "close", "volume",
                    "future_close", "future_return", "target", "atr"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, feature_cols, df
