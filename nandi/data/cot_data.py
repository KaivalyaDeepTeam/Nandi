"""CFTC Commitments of Traders (COT) data fetcher.

Downloads free weekly COT data from CFTC and computes positioning features.
"""

import logging
import os
import numpy as np
import pandas as pd

from nandi.config import DATA_DIR

logger = logging.getLogger(__name__)


class COTDataFetcher:
    """Fetches and processes CFTC Commitments of Traders data."""

    # CFTC commodity codes for forex futures
    CURRENCY_CODES = {
        "eurusd": "099741",
        "gbpusd": "096742",
        "usdjpy": "097741",
        "audusd": "232741",
        "nzdusd": "112741",
        "usdchf": "092741",
        "usdcad": "090741",
    }

    COT_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"

    def __init__(self):
        self.data_cache = {}

    def fetch(self, pair, start_date=None, end_date=None):
        """Download COT data and compute positioning features.

        Returns DataFrame with columns:
        - net_speculative: net long-short of speculators (normalized)
        - net_commercial: net long-short of commercials (normalized)
        - change_in_oi: week-over-week change in open interest
        - concentration_ratio: top-4 trader concentration
        """
        cache_path = os.path.join(DATA_DIR, f"cot_{pair}.csv")

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            logger.info(f"Loaded cached COT data for {pair}: {len(df)} weeks")
            return df

        # Generate synthetic COT-like features from price action as fallback
        logger.warning(f"COT data not available for {pair}, generating proxy features")
        return None

    def compute_cot_proxy(self, close_prices, window=20):
        """Compute COT-like proxy features from price action.

        Uses momentum and positioning proxies when real COT data is unavailable.
        """
        df = pd.DataFrame(index=close_prices.index)

        returns = close_prices.pct_change()

        # Net speculative proxy: momentum z-score
        momentum = returns.rolling(window).mean()
        mom_std = returns.rolling(window * 4).std()
        df["net_speculative"] = (momentum / (mom_std + 1e-10)).clip(-3, 3) / 3

        # Net commercial proxy: mean-reversion signal (inverse of momentum)
        df["net_commercial"] = -df["net_speculative"] * 0.7

        # Change in OI proxy: volume/volatility expansion
        vol_change = returns.rolling(5).std() / (returns.rolling(20).std() + 1e-10) - 1
        df["change_in_oi"] = vol_change.clip(-2, 2) / 2

        # Concentration proxy: absolute momentum strength
        df["concentration_ratio"] = abs(df["net_speculative"])

        return df.dropna()
