"""
Advanced Mathematical Features for Market Analysis.

Novel approach combining:
1. Hurst Exponent - fractal analysis to detect trending vs mean-reverting
2. Shannon Entropy - information theory to detect predictable vs random periods
3. Market Regime Detection - volatility clustering + trend character
4. Wavelet Denoising - separate signal from noise at multiple scales
5. Fractal Dimension - market complexity measure

These features tell us WHEN to trade, not just WHAT to trade.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy


# ═══════════════════════════════════════════════════════════════
# 1. HURST EXPONENT — Fractal Memory of the Market
# ═══════════════════════════════════════════════════════════════
# H > 0.5 → Trending (momentum works)
# H = 0.5 → Random walk (don't trade)
# H < 0.5 → Mean-reverting (fade moves)

def hurst_exponent(series, max_lag=20):
    """Calculate Hurst exponent using R/S analysis."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        subseries = []
        for start in range(0, n - lag, lag):
            chunk = series[start:start + lag]
            if len(chunk) < 2:
                continue
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                subseries.append(R / S)

        if subseries:
            rs_values.append(np.mean(subseries))
        else:
            rs_values.append(1.0)

    # Linear regression of log(R/S) vs log(lag)
    log_lags = np.log(list(lags))
    log_rs = np.log(np.array(rs_values) + 1e-10)

    if len(log_lags) < 2:
        return 0.5

    coeffs = np.polyfit(log_lags, log_rs, 1)
    return np.clip(coeffs[0], 0.0, 1.0)


def rolling_hurst(close, window=100, max_lag=20):
    """Calculate rolling Hurst exponent."""
    result = np.full(len(close), 0.5)
    for i in range(window, len(close)):
        result[i] = hurst_exponent(close[i - window:i], max_lag)
    return result


# ═══════════════════════════════════════════════════════════════
# 2. SHANNON ENTROPY — Market Predictability
# ═══════════════════════════════════════════════════════════════
# Low entropy → Patterns exist → Market is predictable → TRADE
# High entropy → Random noise → Market is chaotic → WAIT

def returns_entropy(returns, window=50, bins=10):
    """Calculate rolling Shannon entropy of returns distribution."""
    result = np.full(len(returns), 0.0)
    for i in range(window, len(returns)):
        chunk = returns[i - window:i]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < bins:
            result[i] = 1.0
            continue
        hist, _ = np.histogram(chunk, bins=bins, density=True)
        hist = hist + 1e-10  # avoid log(0)
        hist = hist / hist.sum()
        result[i] = scipy_entropy(hist)
    return result


def permutation_entropy(series, order=3, delay=1, window=50):
    """Permutation entropy — captures temporal ordering patterns.
    More robust than Shannon entropy for time series."""
    from itertools import permutations
    from math import factorial

    n_perms = factorial(order)
    result = np.full(len(series), 0.0)

    for i in range(window, len(series)):
        chunk = series[i - window:i]
        # Create ordinal patterns
        patterns = {}
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


# ═══════════════════════════════════════════════════════════════
# 3. MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════
# Regime 0: Low volatility ranging (mean reversion)
# Regime 1: Trending with momentum
# Regime 2: High volatility breakout
# Regime 3: Choppy/random (don't trade)

def detect_regime(close, atr, adx, hurst_values, entropy_values):
    """Classify market into 4 regimes based on multiple indicators."""
    n = len(close)
    regimes = np.full(n, 3)  # default: choppy

    for i in range(1, n):
        h = hurst_values[i]
        e = entropy_values[i]
        vol = atr[i] / close[i] if close[i] > 0 else 0
        trend = adx[i]

        if e > 0.85:
            # High entropy = random, don't trade
            regimes[i] = 3
        elif h > 0.55 and trend > 25:
            # Trending regime
            regimes[i] = 1
        elif h < 0.45 and vol < np.median(atr[:i+1] / close[:i+1]):
            # Mean-reverting, low volatility
            regimes[i] = 0
        elif vol > np.percentile(atr[:i+1] / close[:i+1], 80):
            # High volatility breakout
            regimes[i] = 2
        else:
            # Assess tradability
            if e < 0.7:
                regimes[i] = 1 if h > 0.5 else 0
            else:
                regimes[i] = 3

    return regimes


# ═══════════════════════════════════════════════════════════════
# 4. WAVELET DENOISING — Separate Signal from Noise
# ═══════════════════════════════════════════════════════════════

def wavelet_denoise(data, wavelet_scale=10):
    """Simple wavelet-like denoising using multi-scale moving averages.
    Decomposes price into trend + cycles + noise."""
    n = len(data)
    result = {}

    # Multi-scale decomposition using different MA periods
    scales = [5, 10, 20, 50]
    prev_smooth = data.copy()

    for scale in scales:
        if n < scale:
            result[f"detail_{scale}"] = np.zeros(n)
            continue
        smooth = np.convolve(data, np.ones(scale)/scale, mode='same')
        # Detail = difference between scales (the "wavelet coefficients")
        detail = prev_smooth - smooth
        result[f"detail_{scale}"] = detail
        prev_smooth = smooth

    result["trend"] = prev_smooth  # smoothest component
    # Reconstruct denoised signal (without finest detail)
    result["denoised"] = data - result["detail_5"]

    return result


def wavelet_energy(data, window=50):
    """Calculate energy at different wavelet scales — detects dominant cycles."""
    n = len(data)
    scales = [5, 10, 20]
    energies = {f"energy_{s}": np.zeros(n) for s in scales}
    energy_ratio = np.zeros(n)

    for i in range(window, n):
        chunk = data[i-window:i]
        decomp = wavelet_denoise(chunk)
        total_energy = 0
        scale_energies = []
        for s in scales:
            e = np.sum(decomp[f"detail_{s}"]**2)
            energies[f"energy_{s}"][i] = e
            scale_energies.append(e)
            total_energy += e

        if total_energy > 0:
            # Ratio of high-frequency to low-frequency energy
            energy_ratio[i] = scale_energies[0] / (total_energy + 1e-10)

    energies["noise_ratio"] = energy_ratio
    return energies


# ═══════════════════════════════════════════════════════════════
# 5. FRACTAL DIMENSION — Market Complexity
# ═══════════════════════════════════════════════════════════════

def higuchi_fractal_dimension(series, k_max=10):
    """Higuchi's fractal dimension — measures complexity of time series.
    D ≈ 1.0 → smooth/trending
    D ≈ 1.5 → random walk
    D ≈ 2.0 → very complex/choppy"""
    N = len(series)
    if N < k_max * 2:
        return 1.5

    L = []
    x = np.array(series)

    for k in range(1, k_max + 1):
        Lk = []
        for m in range(1, k + 1):
            # Number of segments
            idxs = np.arange(m - 1, N, k)
            if len(idxs) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            norm = (N - 1) / (len(idxs) * k * k)
            Lmk = Lmk * norm
            Lk.append(Lmk)

        if Lk:
            L.append(np.mean(Lk))
        else:
            L.append(1.0)

    # Linear regression of log(L) vs log(1/k)
    ks = np.arange(1, k_max + 1)
    log_k = np.log(1.0 / ks)
    log_L = np.log(np.array(L) + 1e-10)

    coeffs = np.polyfit(log_k, log_L, 1)
    return np.clip(coeffs[0], 1.0, 2.0)


def rolling_fractal_dimension(close, window=100, k_max=8):
    """Rolling Higuchi fractal dimension."""
    result = np.full(len(close), 1.5)
    for i in range(window, len(close)):
        result[i] = higuchi_fractal_dimension(close[i-window:i], k_max)
    return result


# ═══════════════════════════════════════════════════════════════
# 6. MARKET QUALITY INDEX — Tradability Score
# ═══════════════════════════════════════════════════════════════

def market_quality_index(hurst, entropy, fractal_dim, adx):
    """Composite score: how tradable is the market right now?
    Score 0-1. Higher = more tradable.

    Combines:
    - Hurst distance from 0.5 (further = more predictable)
    - Low entropy (more patterned)
    - Fractal dimension away from 1.5 (not random)
    - ADX strength (clear trend or clear range)
    """
    n = len(hurst)
    quality = np.zeros(n)

    for i in range(n):
        h_score = abs(hurst[i] - 0.5) * 2  # 0-1, higher = further from random
        e_score = max(0, 1 - entropy[i])     # lower entropy = higher score
        f_score = abs(fractal_dim[i] - 1.5) * 2  # further from random walk
        a_score = min(adx[i] / 50, 1.0)      # strong trend signal

        quality[i] = (h_score * 0.3 + e_score * 0.3 +
                      f_score * 0.2 + a_score * 0.2)

    return np.clip(quality, 0, 1)
