"""
NewsGate — Combines calendar + sentiment + rates into trading signals.

Two functions:
1. HARD GATE: Force-reduce positions before high-impact events (NFP, Fed, etc.)
2. FEATURES: Provide news features to AEGIS as additional input signals

The hard gate overrides AEGIS decisions — no matter how confident the model is,
we reduce exposure before NFP. This is the single biggest edge from news.

Usage:
    gate = NewsGate()
    gate.refresh()

    # Before executing any trade:
    scale = gate.get_position_scale("eurusd")  # 0.0 to 1.0
    final_position = aegis_position * scale

    # Features for AEGIS (12 total):
    features = gate.get_all_features("eurusd")
"""

import logging
from typing import Dict

from nandi.data.news.calendar import EconomicCalendar
from nandi.data.news.sentiment import NewsSentiment
from nandi.data.news.rates import RateDifferentials

logger = logging.getLogger(__name__)

# How much to scale down positions near events
EVENT_SCALE = {
    # (minutes_before_event, scale_factor)
    # 5 min before NFP: reduce to 10% of target
    # 15 min before: 30%
    # 30 min before: 50%
    "thresholds": [
        (5, 0.1),    # 5 min: almost flat
        (15, 0.3),   # 15 min: 30% of target
        (30, 0.5),   # 30 min: half position
        (60, 0.7),   # 1 hour: 70%
        (120, 0.9),  # 2 hours: 90%
    ],
}


class NewsGate:
    """Master news intelligence gate.

    Combines 3 layers:
    1. Economic Calendar → hard position scaling before events
    2. News Sentiment → directional bias features
    3. Rate Differentials → carry trade features

    Total: 12 features per pair
    """

    def __init__(self, finnhub_key=None, alpha_vantage_key=None, fred_key=None):
        self.calendar = EconomicCalendar(finnhub_key=finnhub_key)
        self.sentiment = NewsSentiment(
            alpha_vantage_key=alpha_vantage_key,
            finnhub_key=finnhub_key,
        )
        self.rates = RateDifferentials(fred_key=fred_key)
        self._initialized = False

    def refresh(self):
        """Refresh all data sources. Call once per trading loop iteration."""
        try:
            self.calendar.refresh()
        except Exception as e:
            logger.debug(f"Calendar refresh failed: {e}")

        try:
            self.sentiment.refresh()
        except Exception as e:
            logger.debug(f"Sentiment refresh failed: {e}")

        try:
            self.rates.refresh()
        except Exception as e:
            logger.debug(f"Rates refresh failed: {e}")

        self._initialized = True

    def get_position_scale(self, pair):
        """HARD GATE: Get position scaling factor for a pair.

        Returns float in [0.0, 1.0]:
        - 1.0: no event nearby, trade normally
        - 0.5: 30 min before high-impact event, half position
        - 0.1: 5 min before NFP/Fed, almost flat
        - 0.0: would mean forced flat (currently we use 0.1 minimum)

        This OVERRIDES AEGIS decisions. Safety first.
        """
        if not self._initialized:
            self.refresh()

        in_window, event_name, minutes_until = self.calendar.is_high_impact_window(
            pair, minutes_before=120, minutes_after=15
        )

        if not in_window:
            return 1.0

        # Find the appropriate scale factor
        if minutes_until is None or minutes_until < 0:
            # Event already happened, in the aftermath window
            return 0.3  # reduced for 15 min after event

        scale = 1.0
        for threshold_min, factor in EVENT_SCALE["thresholds"]:
            if minutes_until <= threshold_min:
                scale = factor
                break

        if scale < 1.0:
            logger.info(
                f"[{pair.upper()}] NEWS GATE: scaling to {scale:.0%} — "
                f"\"{event_name}\" in {minutes_until:.0f} min"
            )

        return scale

    def get_all_features(self, pair):
        """Get all news features for AEGIS (12 features total).

        Returns dict with feature names and values:

        Calendar (4):
          - calendar_risk: 0-1 (1 = event imminent)
          - event_countdown_bars: M5 bars until next high event (0-288)
          - n_high_events_24h: count
          - n_medium_events_24h: count

        Sentiment (5):
          - news_sentiment: pair sentiment (-1 to +1)
          - base_sentiment: base currency (-1 to +1)
          - quote_sentiment: quote currency (-1 to +1)
          - risk_sentiment: risk-on/off (-1 to +1)
          - sentiment_strength: |pair_sentiment|

        Rates (3):
          - rate_differential_norm: normalized rate diff (-1 to +1)
          - carry_direction: +1, 0, -1
          - rate_differential: raw rate diff in %
        """
        if not self._initialized:
            self.refresh()

        features = {}

        # Calendar features
        try:
            cal_feats = self.calendar.get_features(pair)
            features.update(cal_feats)
        except Exception:
            features.update({
                "calendar_risk": 0.0,
                "event_countdown_bars": 288.0,
                "n_high_events_24h": 0,
                "n_medium_events_24h": 0,
            })

        # Sentiment features
        try:
            sent_feats = self.sentiment.get_features(pair)
            features.update(sent_feats)
        except Exception:
            features.update({
                "news_sentiment": 0.0,
                "base_sentiment": 0.0,
                "quote_sentiment": 0.0,
                "risk_sentiment": 0.0,
                "sentiment_strength": 0.0,
            })

        # Rate features
        try:
            rate_feats = self.rates.get_features(pair)
            features.update(rate_feats)
        except Exception:
            features.update({
                "rate_differential": 0.0,
                "rate_differential_norm": 0.0,
                "carry_direction": 0.0,
            })

        return features

    def get_feature_names(self):
        """Get list of all feature names (for appending to feature matrix)."""
        return [
            # Calendar
            "calendar_risk",
            "event_countdown_bars",
            "n_high_events_24h",
            "n_medium_events_24h",
            # Sentiment
            "news_sentiment",
            "base_sentiment",
            "quote_sentiment",
            "risk_sentiment",
            "sentiment_strength",
            # Rates
            "rate_differential_norm",
            "carry_direction",
            "rate_differential",
        ]

    def get_feature_values(self, pair):
        """Get features as a list (same order as get_feature_names)."""
        feats = self.get_all_features(pair)
        return [feats.get(name, 0.0) for name in self.get_feature_names()]

    def get_status_display(self):
        """Human-readable status for logging."""
        lines = [
            "┌─ NEWS INTELLIGENCE STATUS ─────────────────",
            "│ UPCOMING EVENTS:",
            self.calendar.get_next_events_display(n=5),
            "│",
            self.rates.get_all_rates_display(),
            "└─────────────────────────────────────────────",
        ]
        return "\n".join(lines)
