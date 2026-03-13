#!/usr/bin/env python3
"""
Fast M5 data download — HistData.com M1 archives → resample to M5.

~7 ZIP files per pair instead of 43,848 tick requests.
Speed: ~10-15 minutes for all 8 pairs vs ~35 hours with Dukascopy.
"""

import os
import sys
import time
import logging
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fast_download")

PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdchf", "usdcad", "eurjpy"]
OUT_DIR = Path("data/nandi/m5")
M1_DIR = Path("data/nandi/m1")
YEARS = range(2019, 2027)  # 7 years

# Typical spreads for major pairs
SPREADS = {
    "eurusd": 0.00010, "gbpusd": 0.00015, "usdjpy": 0.015,
    "audusd": 0.00015, "nzdusd": 0.00020, "usdchf": 0.00015,
    "usdcad": 0.00015, "eurjpy": 0.020,
}


def parse_histdata_zip(zip_path):
    """Parse a HistData.com ZIP file → M1 DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith('.csv'):
                continue
            with zf.open(name) as f:
                df = pd.read_csv(
                    f, sep=';', header=None,
                    names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'open': float, 'high': float, 'low': float,
                           'close': float, 'volume': float},
                )
                df['datetime'] = pd.to_datetime(
                    df['datetime'].astype(str).str.strip(),
                    format='%Y%m%d %H%M%S',
                    errors='coerce',
                )
                df.dropna(subset=['datetime'], inplace=True)
                df.set_index('datetime', inplace=True)
                df = df[df['open'] > 0]
                return df
    return None


def download_pair(pair):
    """Download M1 data from HistData.com for all years, resample to M5."""
    from histdata import download_hist_data
    from histdata.api import Platform, TimeFrame

    cache_path = OUT_DIR / f"{pair}_m5_7y.csv"
    m1_cache = M1_DIR / f"{pair}_m1_7y.csv"
    if cache_path.exists() and m1_cache.exists():
        try:
            df = pd.read_csv(cache_path, index_col=0, nrows=5)
            if len(df) > 0:
                logger.info(f"[{pair.upper()}] Already cached (M5+M1) ✓")
                return pair, True, "cached"
        except Exception:
            pass

    all_m1 = []
    for year in YEARS:
        try:
            zip_path = download_hist_data(
                year=str(year),
                pair=pair.upper(),
                platform=Platform.GENERIC_ASCII,
                time_frame=TimeFrame.ONE_MINUTE,
            )
            if zip_path and os.path.exists(zip_path):
                df = parse_histdata_zip(zip_path)
                if df is not None and len(df) > 0:
                    all_m1.append(df)
                    logger.info(f"  [{pair.upper()}] {year}: {len(df):,} M1 bars")
                # Clean up zip
                os.remove(zip_path)
            else:
                logger.warning(f"  [{pair.upper()}] {year}: no data")
        except Exception as e:
            logger.warning(f"  [{pair.upper()}] {year}: {e}")

    if not all_m1:
        logger.error(f"[{pair.upper()}] No data downloaded")
        return pair, False, "no data"

    # Combine + deduplicate
    m1_df = pd.concat(all_m1).sort_index()
    m1_df = m1_df[~m1_df.index.duplicated(keep='first')]

    # Save raw M1 data
    M1_DIR.mkdir(parents=True, exist_ok=True)
    m1_cache = M1_DIR / f"{pair}_m1_7y.csv"
    m1_save = m1_df.copy()
    m1_save["spread"] = SPREADS.get(pair, 0.00015)
    m1_save.to_csv(m1_cache)
    logger.info(f"[{pair.upper()}] Saved {len(m1_save):,} M1 bars to {m1_cache}")

    # Resample M1 → M5
    m5_df = m1_df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    m5_df["spread"] = SPREADS.get(pair, 0.00015)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_df.to_csv(cache_path)

    logger.info(f"[{pair.upper()}] ✓ {len(m5_df):,} M5 bars "
                f"({m5_df.index[0].date()} → {m5_df.index[-1].date()})")
    return pair, True, f"{len(m5_df):,} bars"


def main():
    pairs = [p.lower() for p in sys.argv[1:]] if len(sys.argv) > 1 else PAIRS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing (need both M5 and M1 caches)
    pending = []
    for pair in pairs:
        cache_path = OUT_DIR / f"{pair}_m5_7y.csv"
        m1_cache = M1_DIR / f"{pair}_m1_7y.csv"
        if cache_path.exists() and m1_cache.exists():
            logger.info(f"[{pair.upper()}] Already cached (M5+M1) ✓")
        else:
            pending.append(pair)

    if not pending:
        logger.info("All pairs already downloaded!")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"  Fast Download: {len(pending)} pairs × {len(list(YEARS))} years")
    logger.info(f"  Source: HistData.com M1 → resample to M5")
    logger.info(f"{'='*60}\n")

    start = time.time()

    # 2 parallel downloads (HistData rate-limits aggressively)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_pair, p): p for p in pending}
        for future in as_completed(futures):
            pair, success, info = future.result()

    elapsed = time.time() - start
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Summary
    logger.info(f"\nFiles:")
    for f in sorted(OUT_DIR.glob("*_m5_7y.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
