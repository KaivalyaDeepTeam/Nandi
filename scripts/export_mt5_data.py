#!/usr/bin/env python3
"""Export M5 history from MT5 via NandiBridge REFRESH_HISTORY command.

Exports maximum available M5 bars per pair to data/nandi/m5_mt5/{pair}_m5.csv.
The output format matches what the live system sees: UTC timestamps, real tick_volume.

Usage:
    python scripts/export_mt5_data.py
    python scripts/export_mt5_data.py --pairs eurusd gbpusd --bars 500000
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nandi.config import DATA_DIR, PAIRS, PAIRS_MT5, MT5_FILES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# MT5 file encoding
MT5_ENCODING = "utf-16"

# Output directory
MT5_DATA_DIR = os.path.join(DATA_DIR, "m5_mt5")

# Timeouts
REFRESH_TIMEOUT = 60.0   # seconds to wait for REFRESH_HISTORY response
POLL_INTERVAL = 0.5       # seconds between polls


def send_command(files_dir, cmd_str, timeout=REFRESH_TIMEOUT):
    """Write command and poll for response."""
    cmd_path = os.path.join(files_dir, "fx_command.csv")
    resp_path = os.path.join(files_dir, "fx_response.csv")

    # Clean stale files
    for p in [cmd_path, resp_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # Write command (UTF-16 for MQL5)
    with open(cmd_path, "w", encoding=MT5_ENCODING) as f:
        f.write(cmd_str + "\n")
    logger.info(f"CMD -> {cmd_str}")

    # Poll for response
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(resp_path):
            try:
                with open(resp_path, "r", encoding=MT5_ENCODING) as f:
                    response = f.read().strip()
                if response:
                    try:
                        os.remove(resp_path)
                    except OSError:
                        pass
                    try:
                        os.remove(cmd_path)
                    except OSError:
                        pass
                    logger.info(f"RSP <- {response}")
                    return response
            except Exception:
                pass
        time.sleep(POLL_INTERVAL)

    logger.warning(f"Command timed out after {timeout}s: {cmd_str}")
    return "TIMEOUT"


def export_pair(files_dir, pair, bars=500000):
    """Export M5 history for a single pair.

    Flow:
    1. Send REFRESH_HISTORY command
    2. Wait for EA response
    3. Read fx_data.csv (UTF-16, same format as live M5)
    4. Parse timestamps as UTC
    5. Save to standard CSV
    """
    mt5_sym = PAIRS_MT5.get(pair, pair.upper())

    # Send refresh command
    cmd = f"REFRESH_HISTORY,{mt5_sym},{bars},M5"
    resp = send_command(files_dir, cmd)

    if resp == "TIMEOUT":
        logger.error(f"[{pair.upper()}] REFRESH_HISTORY timed out")
        return None

    if not resp.startswith("OK"):
        logger.error(f"[{pair.upper()}] REFRESH_HISTORY failed: {resp}")
        return None

    # Read the exported data from fx_data.csv
    data_path = os.path.join(files_dir, "fx_data.csv")
    if not os.path.exists(data_path):
        logger.error(f"[{pair.upper()}] fx_data.csv not found after REFRESH_HISTORY")
        return None

    try:
        df = pd.read_csv(data_path, encoding=MT5_ENCODING)
        df.columns = [c.strip().lower() for c in df.columns]

        for col in ["time", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                logger.error(f"[{pair.upper()}] Missing column: {col}")
                return None

        # Parse time as UTC (MT5 exports Unix epoch seconds)
        df.index = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df.index = df.index.tz_localize(None)  # naive UTC (matches training)
        df.index.name = "time"
        df.drop(columns=["time"], inplace=True)

        # Keep real tick_volume (rename if needed)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        return df

    except Exception as e:
        logger.error(f"[{pair.upper()}] Failed to parse fx_data.csv: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export M5 history from MT5")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to export (default: all 8)")
    parser.add_argument("--bars", type=int, default=500000,
                        help="Max bars to request (default: 500000)")
    parser.add_argument("--files-dir", type=str, default=None,
                        help="MT5 FILE_COMMON directory")
    args = parser.parse_args()

    pairs = args.pairs or PAIRS
    files_dir = args.files_dir or MT5_FILES_DIR

    os.makedirs(MT5_DATA_DIR, exist_ok=True)

    logger.info(f"Exporting M5 history from MT5")
    logger.info(f"  Pairs:     {', '.join(p.upper() for p in pairs)}")
    logger.info(f"  Max bars:  {args.bars:,}")
    logger.info(f"  Files dir: {files_dir}")
    logger.info(f"  Output:    {MT5_DATA_DIR}")

    # Process one pair at a time (REFRESH_HISTORY writes to single fx_data.csv)
    results = {}
    for pair in pairs:
        logger.info(f"\n--- Exporting {pair.upper()} ---")
        df = export_pair(files_dir, pair, bars=args.bars)

        if df is not None and len(df) > 0:
            out_path = os.path.join(MT5_DATA_DIR, f"{pair}_m5.csv")
            df.to_csv(out_path)
            results[pair] = len(df)
            logger.info(
                f"[{pair.upper()}] Saved {len(df):,} bars "
                f"({df.index[0]} to {df.index[-1]})"
            )
        else:
            logger.error(f"[{pair.upper()}] Export failed")

        # Brief pause between pairs
        time.sleep(1)

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Export complete: {len(results)}/{len(pairs)} pairs")
    for pair, n_bars in results.items():
        logger.info(f"  {pair.upper()}: {n_bars:,} bars")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
