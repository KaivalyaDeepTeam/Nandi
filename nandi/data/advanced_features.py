"""
Advanced mathematical features for market analysis.

Ported from src/advanced_features.py (V1) into the V2 nandi package.

Provides a single entry point ``compute_advanced_features(df)`` that computes:

1. Permutation entropy        — temporal ordering complexity (order=3, delay=1, window=50)
2. Higuchi fractal dimension  — price-series complexity (window=100, k_max=8)
3. Wavelet energy @ scale 5   — energy in the finest detail band (window=50)
4. Wavelet energy @ scale 10  — energy in the medium detail band (window=50)
5. Wavelet energy @ scale 20  — energy in the coarse detail band (window=50)
6. Wavelet noise ratio        — high-freq / total energy (window=50)
7. Market quality index       — composite tradability score in [0, 1]

Dependencies: numpy, pandas only (no scipy).
"""

from __future__ import annotations

import logging
from math import factorial

import numpy as np
import pandas as pd

from nandi.data.features import _rolling_hurst, _rolling_entropy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _permutation_entropy(series: np.ndarray, order: int = 3, delay: int = 1,
                         window: int = 50) -> np.ndarray:
    """Rolling permutation entropy over *series*.

    Permutation entropy captures the complexity of temporal ordering patterns
    inside a sliding window. A value near 0 indicates a highly predictable
    ordering; a value near 1 indicates maximum randomness.

    Args:
        series: 1-D array of price values (e.g. close prices).
        order:  Embedding dimension — the length of each ordinal pattern.
        delay:  Time delay between elements of each pattern.
        window: Number of observations in each rolling window.

    Returns:
        1-D float array, same length as *series*. Positions before
        ``window`` are left as 0.0.
    """
    n_perms = factorial(order)
    n = len(series)
    result = np.zeros(n, dtype=float)

    for i in range(window, n):
        chunk = series[i - window:i]
        patterns: dict[tuple, int] = {}
        count = 0

        for j in range(len(chunk) - (order - 1) * delay):
            indices = [j + k * delay for k in range(order)]
            pattern = tuple(np.argsort([chunk[idx] for idx in indices]))
            patterns[pattern] = patterns.get(pattern, 0) + 1
            count += 1

        if count == 0:
            result[i] = 1.0
            continue

        probs = np.array(list(patterns.values())) / count
        result[i] = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(n_perms)

    return result


def _higuchi_fractal_dimension(series: np.ndarray, k_max: int = 10) -> float:
    """Higuchi fractal dimension for a single window of price data.

    Values near 1.0 indicate a smooth/trending series; near 1.5 a random
    walk; near 2.0 a very choppy/complex series.

    Args:
        series: 1-D array of price values.
        k_max:  Maximum interval length used in the algorithm.

    Returns:
        Scalar fractal dimension clipped to [1.0, 2.0].
    """
    N = len(series)
    if N < k_max * 2:
        return 1.5

    x = np.array(series, dtype=float)
    L = []

    for k in range(1, k_max + 1):
        Lk = []
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, N, k)
            if len(idxs) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            norm = (N - 1) / (len(idxs) * k * k)
            Lmk = Lmk * norm
            Lk.append(Lmk)

        L.append(np.mean(Lk) if Lk else 1.0)

    ks = np.arange(1, k_max + 1, dtype=float)
    log_k = np.log(1.0 / ks)
    log_L = np.log(np.array(L, dtype=float) + 1e-10)

    coeffs = np.polyfit(log_k, log_L, 1)
    return float(np.clip(coeffs[0], 1.0, 2.0))


def _rolling_fractal_dimension(close: np.ndarray, window: int = 100,
                               k_max: int = 8) -> np.ndarray:
    """Rolling Higuchi fractal dimension.

    Args:
        close:  1-D array of close prices.
        window: Number of bars in each rolling window.
        k_max:  Maximum interval length passed to the Higuchi algorithm.

    Returns:
        1-D float array, same length as *close*. Positions before
        ``window`` default to 1.5 (random-walk baseline).
    """
    n = len(close)
    result = np.full(n, 1.5, dtype=float)
    for i in range(window, n):
        result[i] = _higuchi_fractal_dimension(close[i - window:i], k_max)
    return result


def _wavelet_energy(data: np.ndarray, window: int = 50) -> dict[str, np.ndarray]:
    """Rolling multi-scale wavelet-like energy decomposition.

    Uses successive moving-average smoothing to approximate a Haar-style
    wavelet decomposition.  At each bar we smooth the window with MAs of
    lengths 5, 10, and 20, treating the difference between successive
    smoothed signals as the "detail" (wavelet coefficient) at each scale.

    The energy at each scale is the sum of squared detail coefficients
    inside the window.  The noise ratio is the fraction of total energy
    carried by the finest (scale-5) detail band.

    Args:
        data:   1-D array of price values.
        window: Number of bars in each rolling window.

    Returns:
        Dictionary with keys ``"energy_5"``, ``"energy_10"``,
        ``"energy_20"``, and ``"noise_ratio"`` — each a 1-D float array
        of length ``len(data)``.
    """
    n = len(data)
    scales = [5, 10, 20]
    energies: dict[str, np.ndarray] = {f"energy_{s}": np.zeros(n, dtype=float)
                                        for s in scales}
    energy_ratio = np.zeros(n, dtype=float)

    for i in range(window, n):
        chunk = data[i - window:i]
        prev_smooth = chunk.copy()
        scale_energies = []
        total_energy = 0.0

        for s in scales:
            smooth = np.convolve(chunk, np.ones(s) / s, mode="same")
            detail = prev_smooth - smooth
            e = float(np.sum(detail ** 2))
            energies[f"energy_{s}"][i] = e
            scale_energies.append(e)
            total_energy += e
            prev_smooth = smooth

        if total_energy > 0.0:
            energy_ratio[i] = scale_energies[0] / (total_energy + 1e-10)

    energies["noise_ratio"] = energy_ratio
    return energies


def _market_quality_index(hurst: np.ndarray, entropy: np.ndarray,
                          fractal_dim: np.ndarray,
                          adx: np.ndarray) -> np.ndarray:
    """Composite market-quality (tradability) score in [0, 1].

    Higher values indicate a more tradable market state.  The score is a
    weighted combination of four sub-scores:

    - **h_score** (30 %): |H - 0.5| * 2  — deviation from random walk.
    - **e_score** (30 %): 1 - entropy     — low entropy = more patterned.
    - **f_score** (20 %): |D - 1.5| * 2  — deviation from random-walk dimension.
    - **a_score** (20 %): min(ADX / 50, 1) — trend strength.

    Args:
        hurst:       Rolling Hurst exponent array.
        entropy:     Rolling Shannon / permutation entropy array.
        fractal_dim: Rolling Higuchi fractal dimension array.
        adx:         ADX array (same length).

    Returns:
        1-D float array clipped to [0, 1].
    """
    h_score = np.abs(hurst - 0.5) * 2.0
    e_score = np.clip(1.0 - entropy, 0.0, None)
    f_score = np.abs(fractal_dim - 1.5) * 2.0
    a_score = np.clip(adx / 50.0, 0.0, 1.0)

    quality = h_score * 0.3 + e_score * 0.3 + f_score * 0.2 + a_score * 0.2
    return np.clip(quality, 0.0, 1.0)


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> np.ndarray:
    """Compute ADX (Average Directional Index) inline.

    Uses the same formulation as ``nandi.data.features.compute_features``
    for consistency.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: Smoothing period (default 14).

    Returns:
        1-D float array of ADX values, same length as the input series.
        NaN entries are replaced with 0.0.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    plus_di = plus_dm.rolling(period).mean() / (atr + 1e-10)
    minus_di = minus_dm.rolling(period).mean() / (atr + 1e-10)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx.fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced mathematical features from OHLC price data.

    Ported from V1 ``src/advanced_features.py``.

    Feature set
    -----------
    perm_entropy      Permutation entropy (order=3, delay=1, window=50) of
                      close prices.  Captures temporal ordering complexity.

    fractal_dim       Higuchi fractal dimension (window=100, k_max=8) of
                      close prices.  Measures price-series complexity.

    wavelet_energy_5  Energy of the finest (scale-5) wavelet detail band
                      computed over a 50-bar window.

    wavelet_energy_10 Energy of the medium (scale-10) detail band.

    wavelet_energy_20 Energy of the coarse (scale-20) detail band.

    wavelet_noise_ratio
                      Ratio of finest-band energy to total wavelet energy.
                      High values indicate noisy / random price action.

    market_quality    Composite tradability score in [0, 1].  Combines
                      Hurst exponent (from ``nandi.data.features``),
                      permutation entropy, Higuchi fractal dimension, and ADX.
                      Higher = more tradable market state.

    Args:
        df: DataFrame with columns [open, high, low, close] and a
            DatetimeIndex.  Volume is ignored if present.

    Returns:
        DataFrame with the 7 columns listed above, NaN rows dropped.
    """
    close = df["close"].to_numpy(dtype=float)
    high_s = df["high"]
    low_s = df["low"]
    close_s = df["close"]

    logger.info("Computing advanced features for %d bars …", len(df))

    # -- Permutation entropy (close prices) ----------------------------------
    logger.debug("  permutation entropy …")
    perm_ent = _permutation_entropy(close, order=3, delay=1, window=50)

    # -- Higuchi fractal dimension -------------------------------------------
    logger.debug("  Higuchi fractal dimension …")
    frac_dim = _rolling_fractal_dimension(close, window=100, k_max=8)

    # -- Wavelet energy decomposition ----------------------------------------
    logger.debug("  wavelet energy …")
    wave = _wavelet_energy(close, window=50)

    # -- Hurst exponent and Shannon entropy (from nandi.data.features) -------
    # These are needed for the market quality index.
    logger.debug("  Hurst exponent (for market quality) …")
    hurst = _rolling_hurst(close, window=100)

    # _rolling_entropy expects 1-D returns, not prices.
    rets = np.diff(close, prepend=np.nan) / (close + 1e-10)
    logger.debug("  Shannon entropy (for market quality) …")
    entropy = _rolling_entropy(rets, window=60)

    # -- ADX (computed inline for market quality) ----------------------------
    logger.debug("  ADX …")
    adx = _compute_adx(high_s, low_s, close_s, period=14)

    # -- Market quality index ------------------------------------------------
    logger.debug("  market quality index …")
    mqi = _market_quality_index(hurst, perm_ent, frac_dim, adx)

    # -- Assemble output DataFrame -------------------------------------------
    out = pd.DataFrame(
        {
            "perm_entropy":      perm_ent,
            "fractal_dim":       frac_dim,
            "wavelet_energy_5":  wave["energy_5"],
            "wavelet_energy_10": wave["energy_10"],
            "wavelet_energy_20": wave["energy_20"],
            "wavelet_noise_ratio": wave["noise_ratio"],
            "market_quality":    mqi,
        },
        index=df.index,
    )

    out.dropna(inplace=True)
    logger.info("Advanced features ready: %d rows × %d columns",
                len(out), len(out.columns))
    return out
