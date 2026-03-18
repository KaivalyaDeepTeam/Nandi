"""LiveFeatureEngine — Real-time feature computation mirroring training pipeline.

Computes 65 features (36 scalping + 21 path signatures + 8 HTF context),
scales with saved RobustScaler, and provides (120, 65) market state windows.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd

from nandi.config import (
    MODEL_DIR, SPIN_CONFIG, SCALPING_CONFIG, TIMEFRAME_PROFILES,
)
from nandi.data.scalping_features import compute_scalping_features
from nandi.data.path_signatures import compute_path_signatures
from nandi.data.htf_context import compute_htf_context, compute_h1_trend_series
from nandi.data.mt5_data import filter_session_hours

logger = logging.getLogger(__name__)

LOOKBACK = SPIN_CONFIG["lookback_bars"]  # 120
N_FEATURES = SPIN_CONFIG["n_features"]  # 65
M5_PROFILE = TIMEFRAME_PROFILES["M5"]


class LiveFeatureEngine:
    """Compute and cache live features for one pair."""

    def __init__(self, pair, scaler_path=None, feature_names_path=None):
        self.pair = pair
        self.lookback = LOOKBACK

        # Load saved scaler
        pair_dir = os.path.join(MODEL_DIR, pair)
        scaler_path = scaler_path or os.path.join(pair_dir, "scaler_spin.pkl")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"[{pair}] Loaded scaler: {scaler_path}")

        # Load feature names for column ordering verification
        fn_path = feature_names_path or os.path.join(pair_dir, "feature_names_spin.pkl")
        if os.path.exists(fn_path):
            self.expected_feature_names = joblib.load(fn_path)
            logger.info(f"[{pair}] Expected features: {len(self.expected_feature_names)}")
        else:
            self.expected_feature_names = None
            logger.warning(f"[{pair}] No feature_names_spin.pkl — column order unverified")

        # Cached state
        self._features = None       # (N, 65) scaled feature array
        self._atr_series = None      # (N,) ATR(14)
        self._h1_trend_series = None # (N,) H1 trend direction
        self._last_index = None      # last bar timestamp

    def update(self, df_m5):
        """Recompute features from full M5 bar history.

        Mirrors the training pipeline in manager.py:_prepare_pair_spin():
        1. HTF context + H1 trend on FULL (unfiltered) data
        2. Session filter (keep hours 7-21 in data's timezone)
        3. Scalping features + path signatures on FILTERED data
        4. Align, reindex, scale

        Args:
            df_m5: DataFrame with OHLCV and DatetimeIndex.
                   Timestamps should be naive UTC (matching training data).
                   Must have at least lookback+36 rows for path signatures.
        """
        if len(df_m5) < self.lookback + 36:
            logger.warning(f"[{self.pair}] Need {self.lookback + 36} bars, got {len(df_m5)}")
            return

        # 1. HTF context on FULL unfiltered data (matches training)
        htf_features = compute_htf_context(df_m5)

        # 2. H1 trend series on FULL data (for risk gates)
        self._h1_trend_series = compute_h1_trend_series(df_m5)

        # 3. Session filter — keep hours 7-21 (matches training pipeline)
        if SCALPING_CONFIG.get("session_filter", True):
            df_filtered = filter_session_hours(df_m5)
        else:
            df_filtered = df_m5

        if len(df_filtered) < self.lookback:
            logger.warning(
                f"[{self.pair}] Only {len(df_filtered)} bars after session filter"
            )
            return

        # 4. Scalping features on FILTERED data (matches training)
        scalp_features = compute_scalping_features(df_filtered, profile=M5_PROFILE)

        # 5. Path signatures on FILTERED data (matches training)
        sig_features = compute_path_signatures(df_filtered)

        # 6. Align to common index
        common_idx = (
            scalp_features.index
            .intersection(sig_features.index)
            .intersection(htf_features.index)
        )
        if len(common_idx) == 0:
            logger.warning(f"[{self.pair}] No common index after feature alignment")
            return

        combined = (
            scalp_features.loc[common_idx]
            .join(sig_features.loc[common_idx], how="left")
            .join(htf_features.loc[common_idx], how="left")
        )
        combined.fillna(0.0, inplace=True)

        # 7. Verify column ordering matches training
        if self.expected_feature_names is not None:
            combined = combined.reindex(columns=self.expected_feature_names, fill_value=0.0)

        # 8. Scale with saved scaler (transform only, NOT fit)
        feature_vals = combined.values.astype(np.float64)
        scaled = self.scaler.transform(feature_vals).astype(np.float32)

        # 9. Compute ATR(14) on filtered data, aligned to common index
        close = df_filtered["close"]
        high = df_filtered["high"]
        low = df_filtered["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        atr_aligned = atr.reindex(common_idx).ffill().fillna(0.001)

        # Store
        self._features = scaled
        self._atr_series = atr_aligned.values.astype(np.float32)
        self._h1_trend_series = self._h1_trend_series.reindex(
            common_idx, method="ffill"
        ).fillna(0.0).values.astype(np.float32)
        self._last_index = common_idx[-1]

        logger.debug(
            f"[{self.pair}] Features updated: {scaled.shape}, "
            f"last bar: {self._last_index}"
        )

    def get_market_state(self):
        """Get last `lookback` bars of scaled features.

        Returns:
            numpy (lookback, n_features) float32, or None if not ready.
        """
        if self._features is None:
            return None
        n = len(self._features)
        if n < self.lookback:
            # Pad with zeros
            pad = np.zeros((self.lookback - n, self._features.shape[1]), dtype=np.float32)
            return np.vstack([pad, self._features])
        return self._features[-self.lookback:]

    def get_atr(self):
        """Current ATR(14) value."""
        if self._atr_series is None or len(self._atr_series) == 0:
            return 0.001
        return float(self._atr_series[-1])

    def get_h1_trend(self):
        """Current H1 trend direction (-1, 0, +1)."""
        if self._h1_trend_series is None or len(self._h1_trend_series) == 0:
            return 0
        return int(self._h1_trend_series[-1])

    @property
    def ready(self):
        """True if features have been computed at least once."""
        return self._features is not None and len(self._features) >= self.lookback

    @property
    def last_bar_time(self):
        """Timestamp of last processed bar."""
        return self._last_index
