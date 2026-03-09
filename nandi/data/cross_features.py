"""
Cross-pair features — correlations, DXY proxy, spread z-scores.
"""

import numpy as np
import pandas as pd
import logging

from nandi.config import PAIRS, PAIR_GROUPS, USD_PAIRS, USD_PAIRS_DIRECT

logger = logging.getLogger(__name__)


def compute_cross_pair_correlations(closes_df, window=20):
    """Rolling pairwise correlation matrix.

    Args:
        closes_df: DataFrame with pair names as columns, close prices as values.
        window: rolling window for correlation.

    Returns:
        dict mapping date -> correlation matrix (DataFrame).
    """
    returns = closes_df.pct_change().dropna()
    rolling_corr = returns.rolling(window).corr()
    return rolling_corr


def compute_dxy_proxy(closes_df):
    """Compute DXY (dollar index) proxy from available pairs.

    USD strength = average of USD-denominated pairs (inverted for EUR/GBP/AUD/NZD).
    """
    dxy = pd.Series(0.0, index=closes_df.index)
    count = 0

    for pair in USD_PAIRS:
        if pair in closes_df.columns:
            # Invert: higher EURUSD = weaker USD
            dxy -= closes_df[pair].pct_change()
            count += 1

    for pair in USD_PAIRS_DIRECT:
        if pair in closes_df.columns:
            # Direct: higher USDJPY = stronger USD
            dxy += closes_df[pair].pct_change()
            count += 1

    if count > 0:
        dxy /= count

    return dxy.cumsum().fillna(0)


def compute_spread_zscores(closes_df, window=20):
    """Z-scores of spreads between correlated pair groups.

    Used for stat arb alpha: trade when z-score exceeds threshold.

    Returns:
        DataFrame with spread z-scores per pair group.
    """
    zscores = pd.DataFrame(index=closes_df.index)

    for group_name, (pair_a, pair_b) in PAIR_GROUPS.items():
        if pair_a not in closes_df.columns or pair_b not in closes_df.columns:
            continue

        # Log price ratio (spread)
        spread = np.log(closes_df[pair_a] / closes_df[pair_b])
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        zscores[group_name] = (spread - mean) / (std + 1e-10)

    return zscores.fillna(0)


def compute_all_cross_features(closes_df, window=20):
    """Compute all cross-pair features.

    Returns:
        dict with 'dxy_proxy', 'spread_zscores', 'correlation_matrix'
    """
    return {
        "dxy_proxy": compute_dxy_proxy(closes_df),
        "spread_zscores": compute_spread_zscores(closes_df, window),
        "correlation_matrix": compute_cross_pair_correlations(closes_df, window),
    }
