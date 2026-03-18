"""
Multi-Pair Data Manager — Downloads, caches, and prepares data for 8 forex pairs.

Supports D1 (daily via Stooq) and M5 (intraday via MT5/synthetic) timeframes.
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from nandi.config import (
    DATA_DIR, PAIRS, LOOKBACK_YEARS, TEST_MONTHS, LOOKBACK_WINDOW,
    TIMEFRAME_PROFILES, SCALPING_CONFIG,
)
from nandi.data.features import compute_features
from nandi.data.advanced_features import compute_advanced_features
from nandi.data.timeframes import derive_h4_proxy, derive_h1_proxy

logger = logging.getLogger(__name__)


def download_forex_data(symbol="eurusd", years=LOOKBACK_YEARS):
    """Download daily OHLC from Stooq (free, reliable, 20+ years)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{symbol}_daily.csv")

    if os.path.exists(cache):
        logger.info(f"Loading cached data from {cache}")
        df = pd.read_csv(cache, index_col="Date", parse_dates=True)
        return df

    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    d1 = start.strftime("%Y%m%d")
    d2 = end.strftime("%Y%m%d")

    logger.info(f"Downloading {years} years of {symbol.upper()} from Stooq...")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    df = pd.read_csv(url)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]

    if "volume" not in df.columns:
        df["volume"] = 0

    df.dropna(inplace=True)
    df.to_csv(cache)
    logger.info(f"Downloaded {len(df)} daily bars: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


class DataManager:
    """Multi-pair data pipeline: download, feature computation, normalization, splitting.

    Supports timeframe routing:
    - D1: Stooq daily data -> compute_features (original pipeline)
    - M5: MT5 file bridge or synthetic M5 -> compute_scalping_features
    """

    def __init__(self, pairs=None, years=LOOKBACK_YEARS, test_months=TEST_MONTHS,
                 lookback_window=LOOKBACK_WINDOW, timeframe="D1"):
        self.pairs = pairs or PAIRS
        self.years = years
        self.test_months = test_months
        self.timeframe = timeframe

        # Use profile lookback if not explicitly set
        profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["D1"])
        self.lookback_window = profile["lookback_bars"] if lookback_window == LOOKBACK_WINDOW else lookback_window

        self.raw_data = {}       # pair -> raw OHLCV DataFrame
        self.features = {}       # pair -> features DataFrame
        self.scalers = {}        # pair -> RobustScaler
        self.pair_data = {}      # pair -> dict with train/test splits

    def prepare_all(self):
        """Download and prepare data for all pairs. Returns dict of pair -> data dict.

        For M5: uses two-pass approach:
          Pass 1: Download data + compute per-pair features (no split yet)
          Pass 2: Compute cross-pair lead-lag features, inject, THEN split+scale
        This ensures the agent sees what other pairs did before making a trade.
        """
        # Per-pair pipeline (no cross-pair features)
        for pair in self.pairs:
            try:
                self.pair_data[pair] = self.prepare_pair(pair)
            except Exception as e:
                logger.error(f"Failed to prepare {pair}: {e}")
                continue

        # Cross-pair features (D1 — same-bar correlations, DXY proxy)
        # Skip for single-pair training (no cross-pair signal needed)
        if self.timeframe == "D1" and len(self.pairs) > 1:
            try:
                from nandi.data.cross_features import compute_all_cross_features
                closes_df = self.get_all_closes()
                if len(closes_df) > 0:
                    cross = compute_all_cross_features(closes_df)
                    self.cross_features = cross
                    logger.info("Cross-pair features computed (DXY proxy, spread z-scores, correlations)")
            except Exception as e:
                logger.warning(f"Cross-pair features failed: {e}")
                self.cross_features = None

        logger.info(f"Prepared {len(self.pair_data)}/{len(self.pairs)} pairs "
                    f"({self.timeframe}) successfully")
        return self.pair_data

    def _prepare_all_m5_with_cross_features(self):
        """Two-pass M5 pipeline with cross-pair lead-lag features.

        Pass 1: Download + per-pair features for ALL pairs
        Pass 2: Compute lead-lag signals across pairs, inject, then split+scale

        This gives the agent knowledge like "EURUSD moved up 2 bars ago,
        GBPUSD hasn't moved yet — catch-up trade opportunity."
        """
        from nandi.data.mt5_data import MT5DataFetcher, generate_synthetic_m5, filter_session_hours
        from nandi.data.scalping_features import compute_scalping_features
        from nandi.data.cross_pair_scalping import compute_cross_pair_scalping_features
        from nandi.config import SCALPING_CONFIG

        profile = TIMEFRAME_PROFILES.get(self.timeframe, TIMEFRAME_PROFILES["M5"])

        # ── Pass 1: Download data + per-pair features ──
        pair_dfs = {}       # pair -> filtered OHLCV DataFrame
        pair_features = {}  # pair -> per-pair features DataFrame
        pair_closes = {}    # pair -> close price Series (for cross-pair calc)

        for pair in self.pairs:
            try:
                df = None

                # Priority 1: Cached M5 CSV (from HistData/fast_download.py)
                m5_cache = os.path.join(DATA_DIR, "m5", f"{pair}_m5_7y.csv")
                if os.path.exists(m5_cache):
                    try:
                        df = pd.read_csv(m5_cache, index_col=0, parse_dates=True)
                        if len(df) >= profile["lookback_bars"] * 2:
                            logger.info(f"[{pair.upper()}] Loaded cached M5 data ({len(df):,} bars)")
                        else:
                            df = None
                    except Exception as e:
                        logger.warning(f"[{pair.upper()}] Cache load failed: {e}")
                        df = None

                # Priority 2: MT5 file bridge
                if df is None:
                    fetcher = MT5DataFetcher()
                    df = fetcher.fetch(pair)

                # Priority 3: Synthetic fallback (last resort)
                if df is None or len(df) < profile["lookback_bars"] * 2:
                    logger.info(f"[{pair.upper()}] No real M5 data — generating synthetic M5 from daily")
                    daily_df = download_forex_data(symbol=pair, years=self.years)
                    df = generate_synthetic_m5(daily_df, pair_name=pair)

                if df is None or len(df) == 0:
                    logger.error(f"No M5 data available for {pair}")
                    continue

                self.raw_data[pair] = df

                if SCALPING_CONFIG.get("session_filter", True):
                    df = filter_session_hours(df)

                features = compute_scalping_features(df, profile=profile)
                features.dropna(inplace=True)

                if len(features) < profile["lookback_bars"] * 2:
                    logger.warning(f"[{pair.upper()}] Insufficient M5 bars after features: {len(features)}")
                    continue

                pair_dfs[pair] = df
                pair_features[pair] = features
                pair_closes[pair] = df.loc[features.index, "close"]

            except Exception as e:
                logger.error(f"Failed to prepare {pair} (pass 1): {e}")

        if not pair_features:
            logger.error("No pairs survived pass 1")
            return self.pair_data

        # ── Pass 2: Compute cross-pair lead-lag features + inject ──
        # Align all close prices to common timestamps
        all_closes = {}
        common_idx = None
        for pair, closes in pair_closes.items():
            if common_idx is None:
                common_idx = closes.index
            else:
                common_idx = common_idx.intersection(closes.index)

        if common_idx is not None and len(common_idx) > 0:
            for pair in pair_closes:
                all_closes[pair] = pair_closes[pair].reindex(common_idx)

        n_cross_features = 0
        for pair in list(pair_features.keys()):
            try:
                if len(all_closes) >= 2:
                    cross_feats = compute_cross_pair_scalping_features(
                        pair, all_closes
                    )

                    if len(cross_feats) > 0:
                        # Align cross features with per-pair features
                        common = pair_features[pair].index.intersection(cross_feats.index)
                        if len(common) > 0:
                            per_pair = pair_features[pair].loc[common]
                            cross = cross_feats.loc[common].fillna(0)
                            combined = per_pair.join(cross, how='left').fillna(0)
                            pair_features[pair] = combined
                            n_cross_features = len(cross_feats.columns)

                # Now split and scale
                self.features[pair] = pair_features[pair]
                self.pair_data[pair] = self._split_and_scale(
                    pair, pair_dfs[pair], pair_features[pair]
                )

            except Exception as e:
                logger.error(f"Failed to prepare {pair} (pass 2): {e}")

        if n_cross_features > 0:
            logger.info(
                f"Injected {n_cross_features} cross-pair lead-lag features into each pair "
                f"(lagged returns, USD flow, divergence, consensus)"
            )

        logger.info(f"Prepared {len(self.pair_data)}/{len(self.pairs)} pairs "
                    f"({self.timeframe}) successfully")
        return self.pair_data

    def prepare_pair(self, pair):
        """Full pipeline for a single pair: data -> features -> normalize -> split."""
        if self.timeframe == "D1":
            return self._prepare_pair_d1(pair)
        elif self.timeframe == "H1":
            return self._prepare_pair_h1(pair)
        elif self.timeframe == "M5_SPIN":
            return self._prepare_pair_spin(pair)
        else:
            return self._prepare_pair_m5(pair)

    def _prepare_pair_d1(self, pair):
        """Original D1 pipeline: Stooq download -> compute_features -> split."""
        df = download_forex_data(symbol=pair, years=self.years)
        self.raw_data[pair] = df

        features = compute_features(df)

        # Add advanced mathematical features
        try:
            adv_features = compute_advanced_features(df)
            features = features.join(adv_features, how='left')
        except Exception as e:
            logger.warning(f"[{pair}] Advanced features failed: {e}")

        # Add H4/H1 proxy features
        try:
            h4_feats = derive_h4_proxy(df)
            h1_feats = derive_h1_proxy(df)
            features = features.join(h4_feats, how='left').join(h1_feats, how='left')
        except Exception as e:
            logger.warning(f"[{pair}] Timeframe features failed: {e}")

        features.dropna(inplace=True)
        self.features[pair] = features

        return self._split_and_scale(pair, df, features)

    def _prepare_pair_h1(self, pair):
        """H1 pipeline: load cached M5 → resample to H1 → scalping features → split."""
        from nandi.data.scalping_features import compute_scalping_features
        from nandi.config import SCALPING_CONFIG

        profile = TIMEFRAME_PROFILES["H1"]
        df = None

        # Load cached M5 data (same source as M5 pipeline)
        m5_cache = os.path.join(DATA_DIR, "m5", f"{pair}_m5_7y.csv")
        if os.path.exists(m5_cache):
            try:
                df = pd.read_csv(m5_cache, index_col=0, parse_dates=True)
                if len(df) >= profile["lookback_bars"] * 2:
                    logger.info(f"[{pair.upper()}] Loaded cached M5 data for H1 resample ({len(df):,} bars)")
                else:
                    df = None
            except Exception as e:
                logger.warning(f"[{pair.upper()}] M5 cache load failed: {e}")
                df = None

        # Fallback: synthetic M5 from daily
        if df is None:
            from nandi.data.mt5_data import generate_synthetic_m5
            logger.info(f"[{pair.upper()}] No M5 cache — generating synthetic M5 from daily for H1")
            daily_df = download_forex_data(symbol=pair, years=self.years)
            df = generate_synthetic_m5(daily_df, pair_name=pair)

        if df is None or len(df) == 0:
            raise ValueError(f"No M5 data available for {pair} (H1 resample)")

        # Apply session filter before resampling (keeps London+NY only)
        if SCALPING_CONFIG.get("session_filter", True):
            from nandi.data.mt5_data import filter_session_hours
            df = filter_session_hours(df)

        # Resample M5 → H1
        h1 = df.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        logger.info(f"[{pair.upper()}] Resampled M5→H1: {len(h1):,} H1 bars")
        self.raw_data[pair] = h1

        # Compute scalping features using H1 profile windows
        features = compute_scalping_features(h1, profile=profile)
        features.dropna(inplace=True)
        self.features[pair] = features

        if len(features) < profile["lookback_bars"] * 2:
            raise ValueError(
                f"[{pair.upper()}] Insufficient H1 bars after features: {len(features)}"
            )

        return self._split_and_scale(pair, h1, features)

    def _prepare_pair_m5(self, pair):
        """Intraday pipeline: cached CSV -> MT5 bridge -> synthetic fallback -> scalping features -> split."""
        from nandi.data.mt5_data import MT5DataFetcher, generate_synthetic_m5, filter_session_hours
        from nandi.data.scalping_features import compute_scalping_features

        profile = TIMEFRAME_PROFILES.get(self.timeframe, TIMEFRAME_PROFILES["M5"])
        df = None

        # Priority 0: M1 cache (if timeframe is M1)
        if self.timeframe == "M1":
            m1_cache = os.path.join(DATA_DIR, "m1", f"{pair}_m1_7y.csv")
            if os.path.exists(m1_cache):
                try:
                    df = pd.read_csv(m1_cache, index_col=0, parse_dates=True)
                    if len(df) >= profile["lookback_bars"] * 2:
                        logger.info(f"[{pair.upper()}] Loaded cached M1 data ({len(df):,} bars)")
                    else:
                        df = None
                except Exception as e:
                    logger.warning(f"[{pair.upper()}] M1 cache load failed: {e}")
                    df = None

        # Priority 1: Cached M5 CSV (from HistData/fast_download.py)
        m5_cache = os.path.join(DATA_DIR, "m5", f"{pair}_m5_7y.csv")
        if df is None and os.path.exists(m5_cache):
            try:
                df = pd.read_csv(m5_cache, index_col=0, parse_dates=True)
                if len(df) >= profile["lookback_bars"] * 2:
                    logger.info(f"[{pair.upper()}] Loaded cached M5 data ({len(df):,} bars)")
                else:
                    df = None
            except Exception as e:
                logger.warning(f"[{pair.upper()}] Cache load failed: {e}")
                df = None

        # Priority 2: MT5 file bridge
        if df is None:
            fetcher = MT5DataFetcher()
            df = fetcher.fetch(pair)

        # Priority 3: Synthetic fallback (last resort)
        if df is None or len(df) < profile["lookback_bars"] * 2:
            logger.info(f"[{pair.upper()}] No real M5 data — generating synthetic M5 from daily")
            daily_df = download_forex_data(symbol=pair, years=self.years)
            df = generate_synthetic_m5(daily_df, pair_name=pair)

        if df is None or len(df) == 0:
            raise ValueError(f"No M5 data available for {pair}")

        self.raw_data[pair] = df

        # Apply session filter (keep London+NY, drop Asian)
        from nandi.config import SCALPING_CONFIG
        if SCALPING_CONFIG.get("session_filter", True):
            df = filter_session_hours(df)

        # Compute scalping features
        features = compute_scalping_features(df, profile=profile)
        features.dropna(inplace=True)
        self.features[pair] = features

        return self._split_and_scale(pair, df, features)

    def _prepare_pair_spin(self, pair):
        """SPIN pipeline: M5 data -> scalping features + path sigs + HTF context -> split.

        Also produces atr_series and h1_trend_series for the SPIN environment
        risk gates (stop-loss, trend filter).

        Data priority:
          1. MT5 export CSV  -> data/nandi/m5_mt5/{pair}_m5.csv  (UTC, real volume)
          2. HistData cache  -> data/nandi/m5/{pair}_m5_7y.csv   (fallback)
          3. MT5 bridge      -> fx_m5_{PAIR}.csv                 (live 5000 bars)
          4. Synthetic       -> generated from daily             (last resort)
        """
        from nandi.data.mt5_data import MT5DataFetcher, generate_synthetic_m5, filter_session_hours
        from nandi.data.scalping_features import compute_scalping_features
        from nandi.data.path_signatures import compute_path_signatures
        from nandi.data.htf_context import compute_htf_context, compute_h1_trend_series

        profile = TIMEFRAME_PROFILES.get("M5", TIMEFRAME_PROFILES["M5"])
        df = None

        # Priority 1: MT5 export CSV (UTC timestamps, real tick_volume)
        mt5_cache = os.path.join(DATA_DIR, "m5_mt5", f"{pair}_m5.csv")
        if os.path.exists(mt5_cache):
            try:
                df = pd.read_csv(mt5_cache, index_col=0, parse_dates=True)
                if len(df) >= profile["lookback_bars"] * 2:
                    logger.info(f"[{pair.upper()}] Loaded MT5 export data ({len(df):,} bars)")
                else:
                    df = None
            except Exception as e:
                logger.warning(f"[{pair.upper()}] MT5 export load failed: {e}")
                df = None

        # Priority 2: HistData cached M5 CSV
        if df is None:
            m5_cache = os.path.join(DATA_DIR, "m5", f"{pair}_m5_7y.csv")
            if os.path.exists(m5_cache):
                try:
                    df = pd.read_csv(m5_cache, index_col=0, parse_dates=True)
                    if len(df) >= profile["lookback_bars"] * 2:
                        logger.info(f"[{pair.upper()}] Loaded HistData M5 cache ({len(df):,} bars)")
                    else:
                        df = None
                except Exception as e:
                    logger.warning(f"[{pair.upper()}] HistData cache load failed: {e}")
                    df = None

        # Priority 3: MT5 file bridge (live 5000 bars)
        if df is None:
            fetcher = MT5DataFetcher()
            df = fetcher.fetch(pair)

        # Priority 4: Synthetic fallback
        if df is None or len(df) < profile["lookback_bars"] * 2:
            logger.info(f"[{pair.upper()}] No real M5 data — generating synthetic")
            daily_df = download_forex_data(symbol=pair, years=self.years)
            df = generate_synthetic_m5(daily_df, pair_name=pair)

        if df is None or len(df) == 0:
            raise ValueError(f"No M5 data available for {pair}")

        self.raw_data[pair] = df

        # Compute HTF context and H1 trend BEFORE session filter (need full data)
        htf_features = compute_htf_context(df)
        h1_trend_series = compute_h1_trend_series(df)

        # Apply session filter
        if SCALPING_CONFIG.get("session_filter", True):
            df_filtered = filter_session_hours(df)
        else:
            df_filtered = df

        # Compute scalping features (36 features)
        scalp_features = compute_scalping_features(df_filtered, profile=profile)

        # Compute path signatures (21 features) — on filtered data
        sig_features = compute_path_signatures(df_filtered)

        # Align all features to common index
        common_idx = scalp_features.index.intersection(sig_features.index)
        common_idx = common_idx.intersection(htf_features.index)

        if len(common_idx) < profile["lookback_bars"] * 2:
            raise ValueError(f"[{pair.upper()}] Insufficient bars after feature alignment: {len(common_idx)}")

        # Combine: 36 scalping + 21 signatures + 8 HTF = 65 features
        combined = scalp_features.loc[common_idx].join(
            sig_features.loc[common_idx], how="left"
        ).join(
            htf_features.loc[common_idx], how="left"
        )
        combined.dropna(inplace=True)

        if len(combined) < profile["lookback_bars"] * 2:
            raise ValueError(f"[{pair.upper()}] Insufficient bars after dropna: {len(combined)}")

        self.features[pair] = combined

        # Compute ATR series for stop-loss (aligned to combined index)
        tr = pd.concat([
            df_filtered["high"] - df_filtered["low"],
            (df_filtered["high"] - df_filtered["close"].shift()).abs(),
            (df_filtered["low"] - df_filtered["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14, min_periods=1).mean()
        atr_aligned = atr_series.reindex(combined.index).ffill().fillna(0.001)

        # H1 trend aligned
        h1_trend_aligned = h1_trend_series.reindex(combined.index, method="ffill").fillna(0.0)

        # Split and scale
        result = self._split_and_scale(pair, df_filtered, combined)

        # Add extra series for SPIN environment
        train_mask = combined.index < result["test_dates"][0]
        test_mask = combined.index >= result["test_dates"][0]

        result["atr_train"] = atr_aligned[train_mask].values.astype(np.float32)
        result["atr_test"] = atr_aligned[test_mask].values.astype(np.float32)
        result["h1_trend_train"] = h1_trend_aligned[train_mask].values.astype(np.float32)
        result["h1_trend_test"] = h1_trend_aligned[test_mask].values.astype(np.float32)

        logger.info(
            f"[{pair.upper()}] SPIN features: {combined.shape[1]} "
            f"(scalp={len(scalp_features.columns)} + "
            f"sig={len(sig_features.columns)} + "
            f"htf={len(htf_features.columns)})"
        )
        return result

    def _split_and_scale(self, pair, df, features):
        """Common split + normalize logic for both D1 and M5."""
        # Align price data with features
        df_aligned = df.loc[features.index]

        # Train/test split by date
        if self.timeframe == "D1":
            test_start = features.index[-1] - pd.DateOffset(months=self.test_months)
        elif self.timeframe == "H1":
            # For H1: ~22 trading days/month × 24 bars/day
            bars_per_session = TIMEFRAME_PROFILES["H1"]["bars_per_session"]
            test_bars = int(self.test_months * 22 * bars_per_session)
            test_bars = min(test_bars, len(features) // 4)  # max 25% test
            test_start = features.index[-test_bars] if test_bars > 0 else features.index[-1]
        else:
            # For M5: test_months in bars (approximate)
            test_bars = int(self.test_months * 22 * 288)  # ~22 trading days/month * 288 bars/day
            test_bars = min(test_bars, len(features) // 4)  # max 25% test
            test_start = features.index[-test_bars] if test_bars > 0 else features.index[-1]

        train_mask = features.index < test_start
        test_mask = features.index >= test_start

        if train_mask.sum() < 10 or test_mask.sum() < 10:
            raise ValueError(f"Insufficient data for {pair} ({self.timeframe}): "
                           f"train={train_mask.sum()}, test={test_mask.sum()}")

        # Normalize using training statistics only
        scaler = RobustScaler()
        feature_vals = features.values.astype(np.float32)
        train_scaled = scaler.fit_transform(feature_vals[train_mask]).astype(np.float32)
        test_scaled = scaler.transform(feature_vals[test_mask]).astype(np.float32)
        self.scalers[pair] = scaler

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        label = "days" if self.timeframe == "D1" else "bars"
        logger.info(
            f"[{pair.upper()}] {self.timeframe} Train: {n_train} {label} | "
            f"Test: {n_test} {label}"
        )

        return {
            "pair": pair,
            "df": df_aligned,
            "features": features,
            "feature_names": list(features.columns),
            "train_features": train_scaled,
            "test_features": test_scaled,
            "train_prices": df_aligned[train_mask]["close"].values,
            "test_prices": df_aligned[test_mask]["close"].values,
            "train_dates": features.index[train_mask],
            "test_dates": features.index[test_mask],
            "scaler": scaler,
            "n_features": features.shape[1],
            "timeframe": self.timeframe,
        }

    def get_common_dates(self):
        """Get date range common to all pairs (for synchronized backtesting)."""
        if not self.features:
            return None
        date_sets = [set(f.index) for f in self.features.values()]
        common = sorted(set.intersection(*date_sets))
        return pd.DatetimeIndex(common)

    def get_all_closes(self):
        """Returns DataFrame of close prices for all pairs, aligned by date."""
        closes = {}
        for pair, df in self.raw_data.items():
            closes[pair] = df["close"]
        return pd.DataFrame(closes).dropna()
