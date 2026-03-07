"""
Advanced trade filters to increase win rate.
Each filter returns True (allow trade) or False (block trade).
"""

import numpy as np
import pandas as pd


def derive_h1_trend(df_m5: pd.DataFrame) -> dict:
    """Derive H1 trend direction from M5 data.
    Returns the higher-timeframe trend context."""
    close = df_m5["close"].values

    # H1 = 12 M5 candles
    if len(close) < 60:
        return {"h1_trend": 0, "h1_strength": 0}

    # Use last 5 H1 candles (60 M5 bars) to determine trend
    h1_closes = [close[i] for i in range(-60, 0, 12)]
    if len(h1_closes) < 3:
        return {"h1_trend": 0, "h1_strength": 0}

    # Simple trend: compare EMA of H1 closes
    h1_arr = np.array(h1_closes)
    short_avg = np.mean(h1_arr[-2:])
    long_avg = np.mean(h1_arr)

    if short_avg > long_avg * 1.0001:
        h1_trend = 1  # bullish
    elif short_avg < long_avg * 0.9999:
        h1_trend = -1  # bearish
    else:
        h1_trend = 0  # neutral

    # Strength: how consistent is the trend
    changes = np.diff(h1_arr)
    if len(changes) > 0:
        consistency = np.mean(changes > 0) if h1_trend > 0 else np.mean(changes < 0)
    else:
        consistency = 0

    return {"h1_trend": h1_trend, "h1_strength": consistency}


def check_trend_alignment(signal: str, h1_trend: int) -> bool:
    """Trade must align with H1 trend. Neutral H1 allows both."""
    if h1_trend == 0:
        return True
    if signal == "BUY" and h1_trend == 1:
        return True
    if signal == "SELL" and h1_trend == -1:
        return True
    return False


def check_session_filter(hour: int) -> bool:
    """Only trade during high-liquidity sessions.
    London: 07-16 UTC, New York: 13-21 UTC.
    Best: overlap 13-16 UTC."""
    return 7 <= hour <= 21


def detect_candle_pattern(df: pd.DataFrame) -> dict:
    """Detect confirmation candle patterns on the last few candles."""
    if len(df) < 3:
        return {"pattern": "none", "pattern_signal": 0}

    c = df.iloc[-1]
    p = df.iloc[-2]
    pp = df.iloc[-3]

    body = abs(c["close"] - c["open"])
    full_range = c["high"] - c["low"]

    if full_range == 0:
        return {"pattern": "doji", "pattern_signal": 0}

    body_ratio = body / full_range
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]

    pattern = "none"
    pattern_signal = 0

    # Pin bar (rejection candle)
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        pattern = "bullish_pin"
        pattern_signal = 1
    elif upper_wick > body * 2 and lower_wick < body * 0.5:
        pattern = "bearish_pin"
        pattern_signal = -1

    # Engulfing
    p_body = abs(p["close"] - p["open"])
    if body > p_body * 1.2:
        if c["close"] > c["open"] and p["close"] < p["open"]:
            pattern = "bullish_engulfing"
            pattern_signal = 1
        elif c["close"] < c["open"] and p["close"] > p["open"]:
            pattern = "bearish_engulfing"
            pattern_signal = -1

    # Three soldiers / three crows
    bodies = [
        pp["close"] - pp["open"],
        p["close"] - p["open"],
        c["close"] - c["open"],
    ]
    if all(b > 0 for b in bodies):
        pattern = "three_soldiers"
        pattern_signal = 1
    elif all(b < 0 for b in bodies):
        pattern = "three_crows"
        pattern_signal = -1

    return {"pattern": pattern, "pattern_signal": pattern_signal}


def check_pattern_confirmation(signal: str, pattern_signal: int) -> bool:
    """Candle pattern must not contradict the signal.
    No pattern (0) allows trade. Matching pattern boosts it."""
    if pattern_signal == 0:
        return True  # no pattern, allow
    if signal == "BUY" and pattern_signal == 1:
        return True
    if signal == "SELL" and pattern_signal == -1:
        return True
    return False  # pattern contradicts signal


def calculate_adaptive_sl_tp(atr: float, signal: str, entry_price: float,
                             digits: int = 5) -> tuple:
    """ATR-based adaptive SL/TP.
    SL = 1.5 * ATR, TP = 2.5 * ATR (gives ~1.67:1 RR)."""
    sl_distance = atr * 1.5
    tp_distance = atr * 2.5

    # Convert to pips for logging
    sl_pips = sl_distance / 0.0001
    tp_pips = tp_distance / 0.0001

    if signal == "BUY":
        sl_price = round(entry_price - sl_distance, digits)
        tp_price = round(entry_price + tp_distance, digits)
    else:
        sl_price = round(entry_price + sl_distance, digits)
        tp_price = round(entry_price - tp_distance, digits)

    return sl_price, tp_price, sl_pips, tp_pips


def apply_all_filters(signal: str, prediction: dict, df: pd.DataFrame,
                      hour: int = 12) -> dict:
    """Apply all filters and return enriched trade decision."""
    reasons = []

    # 1. Session filter
    session_ok = check_session_filter(hour)
    if not session_ok:
        reasons.append("outside_session")

    # 2. Multi-timeframe trend
    h1_info = derive_h1_trend(df)
    trend_ok = check_trend_alignment(signal, h1_info["h1_trend"])
    if not trend_ok:
        reasons.append("against_h1_trend")

    # 3. Candle pattern
    pattern_info = detect_candle_pattern(df)
    pattern_ok = check_pattern_confirmation(signal, pattern_info["pattern_signal"])
    if not pattern_ok:
        reasons.append("pattern_contradicts")

    # 4. Volatility filter — don't trade in dead markets
    atr_pct = df.iloc[-1].get("atr_pct", 0) if "atr_pct" in df.columns else 0
    vol_ok = atr_pct > 0.0001  # minimum volatility
    if not vol_ok:
        reasons.append("low_volatility")

    all_pass = session_ok and trend_ok and pattern_ok and vol_ok

    # Confidence boost for strong confirmations
    confidence_boost = 0
    if pattern_info["pattern_signal"] != 0 and pattern_ok:
        confidence_boost += 0.1  # pattern confirms
    if h1_info["h1_strength"] > 0.7:
        confidence_boost += 0.05  # strong H1 trend

    return {
        "filters_pass": all_pass,
        "reasons": reasons,
        "h1_trend": h1_info["h1_trend"],
        "h1_strength": h1_info["h1_strength"],
        "candle_pattern": pattern_info["pattern"],
        "confidence_boost": confidence_boost,
    }
