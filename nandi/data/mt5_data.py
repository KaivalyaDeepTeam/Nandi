"""
MT5 intraday data fetcher and synthetic M5 bar generator.

- MT5DataFetcher: reads M5 bars via file bridge, caches per-pair.
- generate_synthetic_m5: calibrated random walk from daily vol for offline training.
"""

import os
import logging

import numpy as np
import pandas as pd

from nandi.config import (
    MT5_FILES_DIR, DATA_DIR, PAIRS_MT5, SCALPING_CONFIG,
)

logger = logging.getLogger(__name__)

BARS_PER_DAY = 288  # 24h * 60min / 5min


class MT5DataFetcher:
    """Reads M5 bars from MT5 file bridge and caches per-pair."""

    def __init__(self, files_dir=None):
        self.files_dir = files_dir or MT5_FILES_DIR
        self.cache_dir = os.path.join(DATA_DIR, "m5")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch(self, pair, bars=None):
        """Read M5 bars for a pair from the MT5 file bridge.

        Tries multiple file formats:
        1. NandiBridge EA format: fx_m5_PAIR.csv (unix timestamps)
        2. Legacy MT5 export: PAIR_M5.csv (Date + Time columns)
        3. Local cache fallback

        Args:
            pair: lowercase pair name (e.g. 'eurusd')
            bars: optional max bars to return (most recent)

        Returns:
            DataFrame with columns [open, high, low, close, volume] and DatetimeIndex.
        """
        cache_path = os.path.join(self.cache_dir, f"{pair}_M5.csv")
        mt5_sym = PAIRS_MT5.get(pair, pair.upper())

        # Priority 1: NandiBridge EA format (fx_m5_EURUSD.csv)
        nandi_path = os.path.join(self.files_dir, f"fx_m5_{mt5_sym}.csv")
        if os.path.exists(nandi_path):
            df = self._read_nandi_bridge(nandi_path)
            if df is not None and len(df) > 0:
                df.to_csv(cache_path)
                logger.info(f"[{pair.upper()}] Read {len(df)} M5 bars from NandiBridge EA")
                if bars:
                    df = df.tail(bars)
                return df

        # Priority 2: Legacy MT5 export format
        legacy_path = os.path.join(self.files_dir, f"{mt5_sym}_M5.csv")
        if os.path.exists(legacy_path):
            df = self._read_bridge_file(legacy_path)
            if df is not None and len(df) > 0:
                df.to_csv(cache_path)
                logger.info(f"[{pair.upper()}] Read {len(df)} M5 bars from MT5 bridge")
                if bars:
                    df = df.tail(bars)
                return df

        # Priority 3: Cache
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"[{pair.upper()}] Loaded {len(df)} cached M5 bars")
            if bars:
                df = df.tail(bars)
            return df

        return None

    def _read_nandi_bridge(self, path):
        """Parse NandiBridge EA export format (unix timestamps)."""
        try:
            df = pd.read_csv(path)
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
                df.set_index("datetime", inplace=True)
            else:
                return None

            required = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                return None

            if "volume" not in df.columns:
                df["volume"] = 0

            cols = ["open", "high", "low", "close", "volume"]
            if "spread" in df.columns:
                cols.append("spread")
            df = df[cols]
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read NandiBridge file {path}: {e}")
            return None

    def _read_bridge_file(self, path):
        """Parse MT5 CSV export format."""
        try:
            df = pd.read_csv(path)
            # MT5 exports: Date, Time, Open, High, Low, Close, Volume
            if "Date" in df.columns and "Time" in df.columns:
                df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
                df.set_index("datetime", inplace=True)
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
            else:
                df.index = pd.to_datetime(df.iloc[:, 0])

            df.columns = [c.lower() for c in df.columns]
            required = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                logger.warning(f"Missing OHLC columns in {path}")
                return None

            if "volume" not in df.columns:
                df["volume"] = 0

            df = df[["open", "high", "low", "close", "volume"]]
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read bridge file {path}: {e}")
            return None


def generate_synthetic_m5(daily_df, pair_name="unknown"):
    """Generate synthetic M5 bars from daily OHLCV data.

    Each daily bar is expanded into 288 M5 bars using a calibrated random walk
    that preserves daily OHLC boundaries and injects realistic intraday patterns.

    Args:
        daily_df: DataFrame with columns [open, high, low, close] and DatetimeIndex.
        pair_name: pair name for logging.

    Returns:
        DataFrame of synthetic M5 bars with DatetimeIndex.
    """
    london_open = SCALPING_CONFIG["london_open_utc"]
    london_close = SCALPING_CONFIG["london_close_utc"]
    ny_open = SCALPING_CONFIG["ny_open_utc"]
    ny_close = SCALPING_CONFIG["ny_close_utc"]

    all_bars = []

    for date, row in daily_df.iterrows():
        daily_open = row["open"]
        daily_high = row["high"]
        daily_low = row["low"]
        daily_close = row["close"]
        daily_range = daily_high - daily_low

        if daily_range < 1e-10:
            continue

        # Generate intraday volatility profile (higher during London/NY)
        vol_profile = _intraday_vol_profile(london_open, london_close,
                                            ny_open, ny_close)

        # Random walk from open to close, constrained to [low, high]
        m5_bars = _constrained_random_walk(
            daily_open, daily_close, daily_high, daily_low,
            vol_profile, n_bars=BARS_PER_DAY,
        )

        # Create timestamps (M5 intervals starting at 00:00 UTC)
        if hasattr(date, 'date'):
            base_date = date
        else:
            base_date = pd.Timestamp(date)

        timestamps = pd.date_range(
            start=base_date.normalize(),
            periods=BARS_PER_DAY,
            freq="5min",
        )

        day_df = pd.DataFrame(m5_bars, index=timestamps,
                              columns=["open", "high", "low", "close"])
        day_df["volume"] = _synthetic_volume(vol_profile, BARS_PER_DAY)
        all_bars.append(day_df)

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = pd.concat(all_bars)
    logger.info(f"[{pair_name.upper()}] Generated {len(result)} synthetic M5 bars "
                f"from {len(daily_df)} daily bars")
    return result


def filter_session_hours(df):
    """Keep only London + NY session hours, drop low-liquidity Asian session.

    Args:
        df: M5 DataFrame with DatetimeIndex (UTC).

    Returns:
        Filtered DataFrame.
    """
    london_open = SCALPING_CONFIG["london_open_utc"]
    ny_close = SCALPING_CONFIG["ny_close_utc"]
    hours = df.index.hour
    mask = (hours >= london_open) & (hours < ny_close)
    filtered = df[mask]
    logger.info(f"Session filter: {len(df)} -> {len(filtered)} bars "
                f"(kept {london_open}:00-{ny_close}:00 UTC)")
    return filtered


def _intraday_vol_profile(london_open, london_close, ny_open, ny_close):
    """Generate 288-bar intraday volatility multiplier profile.

    Asian session: 0.5x, London: 1.2x, NY: 1.0x, Overlap: 1.5x
    """
    profile = np.ones(BARS_PER_DAY) * 0.5  # base: Asian session

    for bar in range(BARS_PER_DAY):
        hour = (bar * 5) // 60
        in_london = london_open <= hour < london_close
        in_ny = ny_open <= hour < ny_close

        if in_london and in_ny:
            profile[bar] = 1.5  # overlap — highest vol
        elif in_london:
            profile[bar] = 1.2
        elif in_ny:
            profile[bar] = 1.0

    # Normalize so mean = 1.0
    profile /= profile.mean()
    return profile


def _constrained_random_walk(open_price, close_price, high_price, low_price,
                             vol_profile, n_bars=288):
    """Generate OHLC M5 bars via random walk constrained to daily boundaries.

    Returns ndarray of shape (n_bars, 4) with columns [open, high, low, close].
    """
    daily_range = high_price - low_price
    bar_vol = daily_range / np.sqrt(n_bars)

    # Generate random increments scaled by vol profile
    rng = np.random.default_rng()
    increments = rng.normal(0, bar_vol, n_bars) * vol_profile

    # Add drift to ensure we arrive at close
    total_needed = close_price - open_price
    total_random = increments.sum()
    drift = (total_needed - total_random) / n_bars
    increments += drift

    # Build price path
    prices = np.empty(n_bars + 1)
    prices[0] = open_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] + increments[i]

    # Clip to daily high/low range
    prices = np.clip(prices, low_price, high_price)

    # Build OHLC bars
    bars = np.empty((n_bars, 4))
    for i in range(n_bars):
        bar_open = prices[i]
        bar_close = prices[i + 1]
        # Intrabar high/low: small random excursion
        excursion = abs(bar_close - bar_open) * rng.uniform(0.1, 0.5)
        bar_high = max(bar_open, bar_close) + excursion
        bar_low = min(bar_open, bar_close) - excursion
        # Clip to daily range
        bar_high = min(bar_high, high_price)
        bar_low = max(bar_low, low_price)
        bars[i] = [bar_open, bar_high, bar_low, bar_close]

    return bars


def _synthetic_volume(vol_profile, n_bars):
    """Generate synthetic volume proportional to volatility profile."""
    rng = np.random.default_rng()
    base_volume = 1000
    volumes = (vol_profile * base_volume * rng.uniform(0.5, 1.5, n_bars)).astype(int)
    return volumes
