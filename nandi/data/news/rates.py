"""
Central Bank Rate Differentials — The #1 driver of forex trends.

"Money flows to higher yields." The interest rate differential between
two currencies is the most fundamental factor in forex pricing.

Fetches current policy rates from:
- FRED API (Fed Funds Rate, free, 120 req/min)
- ECB Data Portal (ECB deposit rate, free, no key)
- Bank of England (Bank Rate, free, no key)
- Bank of Japan (policy rate, free, no key)

For AUD/NZD/CHF/CAD: uses hardcoded recent rates (updated less frequently).
These update only ~8 times per year per bank anyway.

Computes rate differentials per pair: base_rate - quote_rate
Positive differential = carry trade in pair's direction = bullish signal
"""

import os
import json
import logging
from datetime import datetime, timedelta

import requests

from nandi.config import DATA_DIR

logger = logging.getLogger(__name__)

# Pair to (base_currency, quote_currency) mapping
PAIR_CURRENCIES = {
    "eurusd": ("EUR", "USD"),
    "gbpusd": ("GBP", "USD"),
    "usdjpy": ("USD", "JPY"),
    "audusd": ("AUD", "USD"),
    "nzdusd": ("NZD", "USD"),
    "usdchf": ("USD", "CHF"),
    "usdcad": ("USD", "CAD"),
    "eurjpy": ("EUR", "JPY"),
}

# Fallback rates (update these periodically or when central banks meet)
# Last updated: March 2026
DEFAULT_RATES = {
    "USD": 4.50,    # Fed Funds Rate
    "EUR": 2.75,    # ECB Deposit Facility Rate
    "GBP": 4.50,    # Bank of England Bank Rate
    "JPY": 0.50,    # BOJ Policy Rate
    "AUD": 4.10,    # RBA Cash Rate
    "NZD": 3.75,    # RBNZ OCR
    "CHF": 0.50,    # SNB Policy Rate
    "CAD": 3.00,    # BOC Overnight Rate
}


class RateDifferentials:
    """Fetches central bank rates and computes pair differentials.

    Updates daily (rates change at most 8 times/year per bank).
    """

    def __init__(self, fred_key=None):
        self.fred_key = fred_key or os.environ.get("FRED_API_KEY", "")
        self.cache_dir = os.path.join(DATA_DIR, "news_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NandiTrader/2.0"})

        self._rates = dict(DEFAULT_RATES)  # current policy rates
        self._last_fetch = 0

    def refresh(self):
        """Fetch latest rates from available sources."""
        import time
        now = time.time()
        if now - self._last_fetch < 86400:  # once per day
            return

        updated = False

        # Fetch Fed Funds Rate from FRED
        if self.fred_key:
            fed_rate = self._fetch_fred_rate()
            if fed_rate is not None:
                self._rates["USD"] = fed_rate
                updated = True

        # Fetch ECB rate
        ecb_rate = self._fetch_ecb_rate()
        if ecb_rate is not None:
            self._rates["EUR"] = ecb_rate
            updated = True

        # Fetch BOE rate
        boe_rate = self._fetch_boe_rate()
        if boe_rate is not None:
            self._rates["GBP"] = boe_rate
            updated = True

        # Load cached rates for rest, or use defaults
        cached = self._load_cache()
        if cached:
            for curr in ["JPY", "AUD", "NZD", "CHF", "CAD"]:
                if curr in cached and curr not in ["USD", "EUR", "GBP"]:
                    self._rates[curr] = cached[curr]

        self._last_fetch = now
        self._save_cache()

        if updated:
            logger.info(f"Central bank rates: " +
                       " | ".join(f"{k}={v:.2f}%" for k, v in sorted(self._rates.items())))

    def _fetch_fred_rate(self):
        """Fetch Fed Funds Rate from FRED API."""
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": self.fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            observations = data.get("observations", [])
            if observations:
                value = observations[0].get("value", ".")
                if value != ".":
                    rate = float(value)
                    logger.info(f"FRED: Fed Funds Rate = {rate:.2f}%")
                    return rate
            return None
        except Exception as e:
            logger.warning(f"FRED fetch failed: {e}")
            return None

    def _fetch_ecb_rate(self):
        """Fetch ECB deposit facility rate from ECB Data Portal."""
        try:
            # ECB SDMX RESTful API — deposit facility rate
            url = ("https://data-api.ecb.europa.eu/service/data/"
                   "FM/D.U2.EUR.4F.KR.DFR.LEV")
            params = {
                "format": "jsondata",
                "lastNObservations": "1",
            }
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            # Navigate SDMX JSON structure
            datasets = data.get("dataSets", [{}])
            if datasets:
                series = datasets[0].get("series", {})
                for key, val in series.items():
                    obs = val.get("observations", {})
                    for obs_key, obs_val in obs.items():
                        if obs_val:
                            rate = float(obs_val[0])
                            logger.info(f"ECB: Deposit Rate = {rate:.2f}%")
                            return rate
            return None
        except Exception as e:
            logger.warning(f"ECB fetch failed: {e}")
            return None

    def _fetch_boe_rate(self):
        """Fetch Bank of England Bank Rate."""
        try:
            # BOE Interactive Database — Bank Rate (series IUDBEDR)
            end_date = datetime.utcnow().strftime("%d/%b/%Y")
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%d/%b/%Y")

            url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
            params = {
                "csv.x": "yes",
                "Datefrom": start_date,
                "Dateto": end_date,
                "SeriesCodes": "IUDBEDR",
                "CSVF": "CN",
            }
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return None

            # Parse simple CSV: DATE, IUDBEDR
            lines = resp.text.strip().split("\n")
            for line in reversed(lines):
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        rate = float(parts[-1].strip())
                        logger.info(f"BOE: Bank Rate = {rate:.2f}%")
                        return rate
                    except ValueError:
                        continue
            return None
        except Exception as e:
            logger.warning(f"BOE fetch failed: {e}")
            return None

    def get_differential(self, pair):
        """Get rate differential for a pair (base_rate - quote_rate).

        Positive = carry trade favors long (buy) the pair.
        E.g., AUDUSD: if AUD rate 4.10% > USD rate 4.50% → diff = -0.40 (short bias)
        """
        self.refresh()

        base_curr, quote_curr = PAIR_CURRENCIES.get(pair, ("", ""))
        base_rate = self._rates.get(base_curr, 0)
        quote_rate = self._rates.get(quote_curr, 0)

        return base_rate - quote_rate

    def get_features(self, pair):
        """Get numeric features for AEGIS.

        Returns dict with:
        - rate_differential: base_rate - quote_rate (in %)
        - rate_differential_normalized: differential / 5.0 (scaled to ~[-1, 1])
        - carry_direction: +1 (long carry), -1 (short carry), 0 (negligible)
        - base_rate: base currency rate
        - quote_rate: quote currency rate
        """
        self.refresh()

        base_curr, quote_curr = PAIR_CURRENCIES.get(pair, ("", ""))
        base_rate = self._rates.get(base_curr, 0)
        quote_rate = self._rates.get(quote_curr, 0)
        diff = base_rate - quote_rate

        # Carry direction: significant if |diff| > 0.5%
        if diff > 0.5:
            carry = 1.0
        elif diff < -0.5:
            carry = -1.0
        else:
            carry = 0.0

        return {
            "rate_differential": diff,
            "rate_differential_norm": max(-1, min(1, diff / 5.0)),
            "carry_direction": carry,
            "base_rate": base_rate,
            "quote_rate": quote_rate,
        }

    def get_all_rates_display(self):
        """Human-readable rate summary for logging."""
        self.refresh()
        lines = ["  Central Bank Rates:"]
        for curr in sorted(self._rates.keys()):
            lines.append(f"    {curr}: {self._rates[curr]:.2f}%")
        return "\n".join(lines)

    def _save_cache(self):
        try:
            path = os.path.join(self.cache_dir, "rates.json")
            with open(path, "w") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "rates": self._rates,
                }, f, indent=2)
        except Exception:
            pass

    def _load_cache(self):
        try:
            path = os.path.join(self.cache_dir, "rates.json")
            if not os.path.exists(path):
                return {}
            with open(path) as f:
                data = json.load(f)
            return data.get("rates", {})
        except Exception:
            return {}
