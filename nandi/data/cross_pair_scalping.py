"""
Cross-pair lead-lag features for M5 scalping.

Key insight: in FX markets, information propagates across correlated pairs
with small delays (1-6 bars at M5 = 5-30 minutes). If EURUSD moved up 2 bars
ago, GBPUSD often follows. This module captures those lead-lag signals.

Features per pair:
- Lagged returns of correlated pairs (1, 3, 6 bars)
- USD flow momentum (composite direction of all USD pairs)
- Pair divergence (this pair vs correlated pair — catch-up signal)
- Cross-pair volatility regime (are all pairs volatile or calm?)
- Lead-lag correlation strength (rolling)
"""

import numpy as np
import pandas as pd
import logging

from nandi.config import PAIR_GROUPS, USD_PAIRS, USD_PAIRS_DIRECT, PAIRS

logger = logging.getLogger(__name__)

# Which pairs are most informative for each target pair (lead-lag relationships)
# Based on fundamental FX correlations
LEAD_LAG_MAP = {
    "eurusd": ["gbpusd", "usdchf", "eurjpy"],    # EUR bloc + inverse CHF
    "gbpusd": ["eurusd", "eurjpy", "usdchf"],     # GBP follows EUR closely
    "usdjpy": ["eurjpy", "audusd", "gbpusd"],     # JPY crosses + risk sentiment
    "audusd": ["nzdusd", "usdjpy", "eurusd"],     # AUD/NZD twins + risk
    "nzdusd": ["audusd", "usdjpy", "eurusd"],     # NZD follows AUD
    "usdchf": ["eurusd", "gbpusd", "usdcad"],     # CHF = inverse EUR + USD bloc
    "usdcad": ["usdchf", "audusd", "eurusd"],     # commodity + USD bloc
    "eurjpy": ["eurusd", "usdjpy", "gbpusd"],     # cross = EUR * JPY
}

# Lead-lag windows in bars (M5: 1 bar = 5 min)
LAG_WINDOWS = [1, 3, 6]  # 5 min, 15 min, 30 min


def compute_cross_pair_scalping_features(target_pair, all_closes, lag_windows=None):
    """Compute lead-lag cross-pair features for a target pair.

    Args:
        target_pair: str, the pair we're computing features FOR (e.g. "gbpusd")
        all_closes: dict of {pair_name: pd.Series of close prices} with aligned index
        lag_windows: list of lag bars (default [1, 3, 6])

    Returns:
        pd.DataFrame of cross-pair features, indexed same as input
    """
    if lag_windows is None:
        lag_windows = LAG_WINDOWS

    if target_pair not in all_closes or len(all_closes) < 2:
        return pd.DataFrame()

    # Get lead pairs for this target
    lead_pairs = LEAD_LAG_MAP.get(target_pair, [])
    available_leads = [p for p in lead_pairs if p in all_closes]

    if not available_leads:
        # Fallback: use any available pairs
        available_leads = [p for p in all_closes if p != target_pair][:3]

    if not available_leads:
        return pd.DataFrame()

    idx = all_closes[target_pair].index
    f = pd.DataFrame(index=idx)

    target_ret = all_closes[target_pair].pct_change()

    # ── 1. Lagged returns of lead pairs ──
    # "What did correlated pairs do 1/3/6 bars ago?"
    for lead_pair in available_leads:
        lead_ret = all_closes[lead_pair].pct_change()
        for lag in lag_windows:
            # Lagged return: what lead_pair did `lag` bars ago
            f[f"lead_{lead_pair}_ret_lag{lag}"] = lead_ret.shift(lag)

    # ── 2. USD flow momentum ──
    # Composite USD strength over recent bars — are all USD pairs moving together?
    usd_momentum = _compute_usd_flow(all_closes, windows=[1, 3, 6])
    for col in usd_momentum.columns:
        f[col] = usd_momentum[col]

    # ── 3. Pair divergence signals ──
    # If this pair and its closest correlated pair diverge, that's a catch-up signal
    for lead_pair in available_leads[:2]:  # top 2 correlated only
        lead_ret = all_closes[lead_pair].pct_change()

        # Cumulative divergence over last 6 bars
        cum_target = target_ret.rolling(6).sum()
        cum_lead = lead_ret.rolling(6).sum()
        f[f"divergence_{lead_pair}_6b"] = cum_target - cum_lead

        # Divergence over last 12 bars (1 hour)
        cum_target_12 = target_ret.rolling(12).sum()
        cum_lead_12 = lead_ret.rolling(12).sum()
        f[f"divergence_{lead_pair}_12b"] = cum_target_12 - cum_lead_12

    # ── 4. Cross-pair volatility regime ──
    # Are all pairs volatile or calm? High cross-vol = risk event, scale down
    cross_vol = _compute_cross_volatility(all_closes, window=12)
    f["cross_vol_12b"] = cross_vol

    # ── 5. Cross-pair momentum consensus ──
    # How many pairs moved in the same direction over last N bars?
    for window in [1, 3]:
        consensus = _compute_momentum_consensus(all_closes, target_pair, window=window)
        f[f"momentum_consensus_{window}b"] = consensus

    # ── 6. Rolling lead-lag correlation ──
    # Strength of the lead-lag relationship (non-stationary — varies with regime)
    for lead_pair in available_leads[:2]:
        lead_ret = all_closes[lead_pair].pct_change()
        # Correlation between lead_pair's lagged return and target's current return
        lagged_lead = lead_ret.shift(1)
        rolling_corr = target_ret.rolling(72).corr(lagged_lead)  # 6-hour window
        f[f"leadlag_corr_{lead_pair}"] = rolling_corr

    f = f.reindex(idx)
    logger.info(
        f"[{target_pair.upper()}] Computed {len(f.columns)} cross-pair features "
        f"from {len(available_leads)} lead pairs"
    )
    return f


def _compute_usd_flow(all_closes, windows=None):
    """Compute USD flow momentum — composite USD strength.

    Positive = USD strengthening across all pairs.
    """
    if windows is None:
        windows = [1, 3, 6]

    # Collect USD-denominated returns (normalize direction)
    usd_returns = []
    for pair in USD_PAIRS:
        if pair in all_closes:
            # EURUSD up = USD weak → invert
            usd_returns.append(-all_closes[pair].pct_change())
    for pair in USD_PAIRS_DIRECT:
        if pair in all_closes:
            # USDJPY up = USD strong → keep
            usd_returns.append(all_closes[pair].pct_change())

    if not usd_returns:
        return pd.DataFrame()

    # Average USD strength
    usd_strength = pd.concat(usd_returns, axis=1).mean(axis=1)

    f = pd.DataFrame(index=usd_strength.index)
    for w in windows:
        f[f"usd_flow_{w}b"] = usd_strength.rolling(w).sum()

    # USD flow acceleration (change in flow)
    f["usd_flow_accel"] = f["usd_flow_1b"] - f["usd_flow_1b"].shift(1)

    return f


def _compute_cross_volatility(all_closes, window=12):
    """Average realized volatility across all available pairs.

    High cross-vol = risk event (news, central bank, etc.).
    """
    vols = []
    for pair, closes in all_closes.items():
        ret = closes.pct_change()
        vol = ret.rolling(window).std()
        vols.append(vol)

    if not vols:
        return pd.Series(0.0)

    # Average vol, then z-score it
    avg_vol = pd.concat(vols, axis=1).mean(axis=1)
    vol_mean = avg_vol.rolling(72).mean()  # 6-hour baseline
    vol_std = avg_vol.rolling(72).std()
    return (avg_vol - vol_mean) / (vol_std + 1e-10)


def _compute_momentum_consensus(all_closes, target_pair, window=1):
    """Fraction of pairs moving in the same direction as the target.

    1.0 = all pairs moved same way (strong trend/flow)
    0.0 = all pairs moved opposite (divergence)
    0.5 = mixed (no consensus)
    """
    target_ret = all_closes[target_pair].pct_change(window)
    target_dir = np.sign(target_ret)

    same_dir_count = pd.Series(0.0, index=target_ret.index)
    total = 0

    for pair, closes in all_closes.items():
        if pair == target_pair:
            continue
        other_ret = closes.pct_change(window)
        other_dir = np.sign(other_ret)
        same_dir_count += (target_dir == other_dir).astype(float)
        total += 1

    if total == 0:
        return pd.Series(0.5, index=target_ret.index)

    return same_dir_count / total
