"""
Economic Calendar — Fetches high-impact forex events.

Knows when NFP, CPI, Fed decisions, etc. happen so AEGIS can:
1. Reduce position BEFORE high-impact releases (hard gate)
2. Provide countdown features to the agent
3. Skip trading during extreme-volatility windows

Sources (all free):
- Primary: Finnhub economic calendar (60 req/min, free)
- Fallback: Built-in static schedule for major recurring events

IST times included for India-based monitoring.
"""

import os
import json
import logging
import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from nandi.config import DATA_DIR

logger = logging.getLogger(__name__)

# IST offset
IST = timezone(timedelta(hours=5, minutes=30))

# ── Built-in recurring events (fallback when API fails) ────────────
# These are the events that cause 50-100+ pip moves.
# Format: (currency, event_name, impact, recurrence)
MAJOR_RECURRING_EVENTS = [
    # US events (affect ALL pairs)
    ("USD", "Non-Farm Payrolls", "high", "1st_friday_monthly"),
    ("USD", "FOMC Rate Decision", "high", "8x_yearly"),
    ("USD", "CPI m/m", "high", "monthly"),
    ("USD", "Core CPI m/m", "high", "monthly"),
    ("USD", "GDP q/q", "high", "quarterly"),
    ("USD", "Retail Sales m/m", "medium", "monthly"),
    ("USD", "ISM Manufacturing PMI", "medium", "monthly"),
    ("USD", "Unemployment Rate", "high", "1st_friday_monthly"),
    ("USD", "Fed Chair Speech", "high", "irregular"),
    # EUR events
    ("EUR", "ECB Rate Decision", "high", "6x_yearly"),
    ("EUR", "ECB Press Conference", "high", "6x_yearly"),
    ("EUR", "German CPI m/m", "medium", "monthly"),
    ("EUR", "Eurozone CPI y/y", "medium", "monthly"),
    ("EUR", "German Manufacturing PMI", "medium", "monthly"),
    # GBP events
    ("GBP", "BoE Rate Decision", "high", "8x_yearly"),
    ("GBP", "UK CPI y/y", "high", "monthly"),
    ("GBP", "UK GDP m/m", "medium", "monthly"),
    ("GBP", "UK Retail Sales m/m", "medium", "monthly"),
    # JPY events
    ("JPY", "BoJ Rate Decision", "high", "8x_yearly"),
    ("JPY", "Japan CPI y/y", "medium", "monthly"),
    ("JPY", "Tankan Survey", "medium", "quarterly"),
    # AUD events
    ("AUD", "RBA Rate Decision", "high", "8x_yearly"),
    ("AUD", "Australia Employment Change", "medium", "monthly"),
    ("AUD", "Australia CPI q/q", "high", "quarterly"),
    # NZD events
    ("NZD", "RBNZ Rate Decision", "high", "7x_yearly"),
    # CHF events
    ("CHF", "SNB Rate Decision", "high", "4x_yearly"),
    # CAD events
    ("CAD", "BoC Rate Decision", "high", "8x_yearly"),
    ("CAD", "Canada Employment Change", "medium", "monthly"),
    ("CAD", "Canada CPI m/m", "medium", "monthly"),
]

# Which currencies affect which pairs
CURRENCY_TO_PAIRS = {
    "USD": ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdchf", "usdcad", "eurjpy"],
    "EUR": ["eurusd", "eurjpy"],
    "GBP": ["gbpusd"],
    "JPY": ["usdjpy", "eurjpy"],
    "AUD": ["audusd"],
    "NZD": ["nzdusd"],
    "CHF": ["usdchf"],
    "CAD": ["usdcad"],
}


class EconomicCalendar:
    """Fetches and caches economic calendar events.

    Provides:
    - upcoming_events(hours_ahead): events in next N hours
    - is_high_impact_window(pair, minutes_before, minutes_after): should we reduce position?
    - get_features(pair): numeric features for AEGIS [countdown, impact_score, n_events]
    """

    def __init__(self, finnhub_key=None, cache_hours=4):
        self.finnhub_key = finnhub_key or os.environ.get("FINNHUB_API_KEY", "")
        self.cache_hours = cache_hours
        self.cache_dir = os.path.join(DATA_DIR, "news_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._events = []  # list of event dicts
        self._last_fetch = 0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NandiTrader/2.0"})

    def refresh(self):
        """Fetch latest calendar events. Tries Finnhub, falls back to cache/static."""
        now = time_module.time()
        if now - self._last_fetch < self.cache_hours * 3600:
            return  # still fresh

        events = []

        # Try Finnhub
        if self.finnhub_key:
            events = self._fetch_finnhub()

        # Fallback: cached file
        if not events:
            events = self._load_cache()

        # Fallback: static recurring schedule
        if not events:
            events = self._generate_static_events()

        self._events = events
        self._last_fetch = now

        # Cache for next time
        self._save_cache(events)

        n_high = sum(1 for e in events if e.get("impact") == "high")
        logger.info(f"Economic calendar: {len(events)} events loaded ({n_high} high-impact)")

    def _fetch_finnhub(self):
        """Fetch from Finnhub economic calendar API."""
        try:
            today = datetime.utcnow().date()
            from_date = today.isoformat()
            to_date = (today + timedelta(days=7)).isoformat()

            url = "https://finnhub.io/api/v1/calendar/economic"
            params = {
                "from": from_date,
                "to": to_date,
                "token": self.finnhub_key,
            }

            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Finnhub calendar returned {resp.status_code}")
                return []

            data = resp.json()
            raw_events = data.get("economicCalendar", [])

            events = []
            for e in raw_events:
                currency = e.get("country", "").upper()
                # Map country codes to currency
                country_to_currency = {
                    "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY",
                    "AU": "AUD", "NZ": "NZD", "CH": "CHF", "CA": "CAD",
                    "DE": "EUR", "FR": "EUR", "IT": "EUR",
                }
                currency = country_to_currency.get(currency, currency)

                if currency not in CURRENCY_TO_PAIRS:
                    continue

                impact = self._classify_impact(e.get("event", ""), e.get("impact", ""))

                events.append({
                    "currency": currency,
                    "event": e.get("event", "Unknown"),
                    "datetime_utc": e.get("time", ""),
                    "impact": impact,
                    "actual": e.get("actual"),
                    "estimate": e.get("estimate"),
                    "prev": e.get("prev"),
                    "pairs_affected": CURRENCY_TO_PAIRS.get(currency, []),
                })

            logger.info(f"Finnhub: fetched {len(events)} forex calendar events")
            return events

        except Exception as e:
            logger.warning(f"Finnhub calendar fetch failed: {e}")
            return []

    def _classify_impact(self, event_name, raw_impact):
        """Classify event impact as high/medium/low."""
        event_lower = event_name.lower()

        # High-impact keywords
        high_keywords = [
            "nonfarm", "non-farm", "nfp", "fomc", "rate decision", "interest rate",
            "cpi", "gdp", "employment change", "unemployment rate",
            "ecb press", "boe rate", "boj rate", "rba rate", "rbnz rate",
            "snb rate", "boc rate", "fed chair",
        ]
        for kw in high_keywords:
            if kw in event_lower:
                return "high"

        # Medium-impact keywords
        medium_keywords = [
            "retail sales", "pmi", "ism", "trade balance", "housing",
            "consumer confidence", "industrial production", "ppi",
        ]
        for kw in medium_keywords:
            if kw in event_lower:
                return "medium"

        # Use API-provided impact if available
        if raw_impact:
            raw = str(raw_impact).lower()
            if "high" in raw or raw == "3":
                return "high"
            if "medium" in raw or raw == "2":
                return "medium"

        return "low"

    def _generate_static_events(self):
        """Generate upcoming events from known recurring schedule."""
        events = []
        now = datetime.utcnow()

        for currency, event_name, impact, _ in MAJOR_RECURRING_EVENTS:
            if impact != "high":
                continue

            # Generate placeholder events for the next 7 days
            # These won't have exact times but signal "be careful this week"
            events.append({
                "currency": currency,
                "event": event_name,
                "datetime_utc": "",  # unknown exact time
                "impact": impact,
                "actual": None,
                "estimate": None,
                "prev": None,
                "pairs_affected": CURRENCY_TO_PAIRS.get(currency, []),
                "is_static": True,
            })

        return events

    def _save_cache(self, events):
        """Save events to local cache."""
        try:
            path = os.path.join(self.cache_dir, "calendar.json")
            with open(path, "w") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "events": events,
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def _load_cache(self):
        """Load events from local cache."""
        try:
            path = os.path.join(self.cache_dir, "calendar.json")
            if not os.path.exists(path):
                return []

            with open(path) as f:
                data = json.load(f)

            # Check if cache is recent enough (24 hours)
            ts = datetime.fromisoformat(data["timestamp"])
            if (datetime.utcnow() - ts).total_seconds() > 86400:
                return []

            return data.get("events", [])
        except Exception:
            return []

    def upcoming_events(self, hours_ahead=24, impact_filter=None):
        """Get events happening in the next N hours.

        Args:
            hours_ahead: how far ahead to look
            impact_filter: "high", "medium", or None for all

        Returns:
            list of event dicts sorted by time
        """
        self.refresh()

        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []

        for event in self._events:
            dt_str = event.get("datetime_utc", "")
            if not dt_str:
                # Static events with no time — include if impact matches
                if impact_filter and event.get("impact") != impact_filter:
                    continue
                upcoming.append(event)
                continue

            try:
                # Parse various datetime formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                    try:
                        event_dt = datetime.strptime(dt_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                if now <= event_dt <= cutoff:
                    if impact_filter and event.get("impact") != impact_filter:
                        continue
                    event["_parsed_dt"] = event_dt
                    upcoming.append(event)
            except Exception:
                continue

        # Sort by time (static events last)
        upcoming.sort(key=lambda e: e.get("_parsed_dt", datetime.max))
        return upcoming

    def is_high_impact_window(self, pair, minutes_before=30, minutes_after=15):
        """Check if we're within a high-impact event window for this pair.

        Args:
            pair: e.g. "eurusd"
            minutes_before: minutes before event to start reducing
            minutes_after: minutes after event release

        Returns:
            (is_in_window: bool, event_name: str or None, minutes_until: float or None)
        """
        self.refresh()
        now = datetime.utcnow()

        for event in self._events:
            if event.get("impact") != "high":
                continue
            if pair not in event.get("pairs_affected", []):
                continue

            dt_str = event.get("datetime_utc", "")
            if not dt_str:
                continue

            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        event_dt = datetime.strptime(dt_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                diff_minutes = (event_dt - now).total_seconds() / 60

                # In window: [minutes_before before event] to [minutes_after after event]
                if -minutes_after <= diff_minutes <= minutes_before:
                    return True, event.get("event"), diff_minutes

            except Exception:
                continue

        return False, None, None

    def get_features(self, pair):
        """Get numeric features for AEGIS.

        Returns dict with:
        - calendar_risk: 0.0 (no event) to 1.0 (high-impact event imminent)
        - event_countdown_bars: M5 bars until next high-impact event (capped at 288)
        - n_high_events_24h: number of high-impact events in next 24h
        - n_medium_events_24h: number of medium-impact events in next 24h
        """
        self.refresh()

        features = {
            "calendar_risk": 0.0,
            "event_countdown_bars": 288.0,  # 24 hours (max)
            "n_high_events_24h": 0,
            "n_medium_events_24h": 0,
        }

        now = datetime.utcnow()

        nearest_high_minutes = float("inf")

        for event in self._events:
            pairs_affected = event.get("pairs_affected", [])
            if pair not in pairs_affected:
                continue

            impact = event.get("impact", "low")
            dt_str = event.get("datetime_utc", "")

            if not dt_str:
                # Static event — count it conservatively
                if impact == "high":
                    features["n_high_events_24h"] += 1
                elif impact == "medium":
                    features["n_medium_events_24h"] += 1
                continue

            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        event_dt = datetime.strptime(dt_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                diff_minutes = (event_dt - now).total_seconds() / 60

                # Count events in next 24h
                if 0 <= diff_minutes <= 1440:
                    if impact == "high":
                        features["n_high_events_24h"] += 1
                    elif impact == "medium":
                        features["n_medium_events_24h"] += 1

                # Track nearest high-impact event
                if impact == "high" and diff_minutes > 0:
                    nearest_high_minutes = min(nearest_high_minutes, diff_minutes)

            except Exception:
                continue

        # Convert to features
        if nearest_high_minutes < float("inf"):
            # Countdown in M5 bars (1 bar = 5 min)
            features["event_countdown_bars"] = min(nearest_high_minutes / 5, 288.0)

            # Calendar risk: exponential ramp-up as event approaches
            # 0.0 at 2h+ out, 0.5 at 30min, 1.0 at 5min or less
            if nearest_high_minutes <= 5:
                features["calendar_risk"] = 1.0
            elif nearest_high_minutes <= 30:
                features["calendar_risk"] = 0.5 + 0.5 * (1 - nearest_high_minutes / 30)
            elif nearest_high_minutes <= 120:
                features["calendar_risk"] = 0.3 * (1 - (nearest_high_minutes - 30) / 90)
            else:
                features["calendar_risk"] = 0.0

        return features

    def get_next_events_display(self, n=5):
        """Get a human-readable summary of upcoming events (for logging)."""
        events = self.upcoming_events(hours_ahead=48, impact_filter=None)
        lines = []
        for e in events[:n]:
            dt_str = e.get("datetime_utc", "??:??")
            impact = e.get("impact", "?").upper()
            currency = e.get("currency", "?")
            name = e.get("event", "Unknown")

            # Convert to IST for display
            ist_str = ""
            if "_parsed_dt" in e:
                ist_dt = e["_parsed_dt"].replace(tzinfo=timezone.utc).astimezone(IST)
                ist_str = ist_dt.strftime("%I:%M %p IST")

            icon = {"HIGH": "!!!", "MEDIUM": " ! ", "LOW": "   "}.get(impact, "   ")
            lines.append(f"  [{icon}] {currency} {name} — {ist_str or dt_str}")

        return "\n".join(lines) if lines else "  No upcoming events"
