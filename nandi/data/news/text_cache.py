"""
Historical News Text Cache — Aligns headlines with M5 bar timestamps for training.

During RL training, AEGIS needs headlines that were available at each timestamp.
This module:
1. Fetches historical headlines from Finnhub/Alpha Vantage
2. Aligns them to M5 bar timestamps (headlines available BEFORE each bar)
3. Caches everything locally to avoid re-fetching
4. Generates synthetic "headline proxies" for periods without real news

For training on 7 years of data:
- Real headlines: available for ~last 2 years from free APIs
- Synthetic: generated from price action characteristics for older periods
  (e.g., high volatility + USD up → "Dollar rallies on strong data")

This gives AEGIS the ability to learn text-market relationships even
on historical data where we don't have actual news articles.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from nandi.config import DATA_DIR

logger = logging.getLogger(__name__)

# Templates for synthetic headlines based on market conditions
SYNTHETIC_TEMPLATES = {
    # (condition, currency, headlines)
    "usd_strong_vol_high": [
        "Dollar surges amid risk-off sentiment",
        "Fed hawkish signals boost USD across the board",
        "Strong US jobs data sends dollar higher",
        "Treasury yields spike, dollar rallies",
        "Safe haven flows drive dollar demand",
    ],
    "usd_weak_vol_high": [
        "Dollar sells off on dovish Fed commentary",
        "Weak US data weighs on greenback",
        "Fed rate cut expectations grow, dollar slides",
        "Risk appetite returns, safe haven dollar drops",
        "US economic concerns send dollar lower",
    ],
    "usd_strong_vol_low": [
        "Dollar steady, markets await key data",
        "Greenback inches higher in quiet trade",
        "USD supported by yield advantage",
        "Dollar holds gains ahead of Fed speakers",
    ],
    "usd_weak_vol_low": [
        "Dollar drifts lower in thin trading",
        "Greenback slips on positioning adjustments",
        "Dollar edges down, traders eye upcoming data",
    ],
    "eur_strong": [
        "Euro climbs on strong Eurozone PMI data",
        "ECB hawkish stance supports EUR",
        "Euro rallies as German data beats expectations",
        "Eurozone growth surprises to the upside",
    ],
    "eur_weak": [
        "Euro under pressure as ECB signals caution",
        "Weak Eurozone data drags EUR lower",
        "Political uncertainty weighs on euro",
        "ECB dovish shift sends EUR down",
    ],
    "gbp_strong": [
        "Pound jumps on BoE rate decision",
        "UK inflation data supports sterling",
        "GBP rallies on strong labor market",
    ],
    "gbp_weak": [
        "Pound falls on UK recession fears",
        "BoE dovish pivot weighs on sterling",
        "Brexit concerns resurface, GBP slides",
    ],
    "jpy_strong": [
        "Yen strengthens on BoJ policy shift",
        "Safe haven yen rallies amid market turmoil",
        "BoJ intervention fears support JPY",
    ],
    "jpy_weak": [
        "Yen hits new lows as BoJ maintains easing",
        "Widening yield gap pressures JPY",
        "Yen continues slide amid carry trade demand",
    ],
    "risk_on": [
        "Global stocks rally, risk appetite improves",
        "Commodity currencies benefit from risk-on mood",
        "Market optimism lifts AUD and NZD",
        "Trade deal hopes boost market sentiment",
    ],
    "risk_off": [
        "Markets plunge on geopolitical tensions",
        "Risk-off sentiment grips forex markets",
        "Flight to safety benefits USD, JPY, CHF",
        "Global recession fears trigger broad selloff",
    ],
    "neutral": [
        "Forex markets range-bound ahead of key events",
        "Currencies trade sideways in quiet session",
        "Markets await central bank decisions this week",
        "Thin liquidity keeps major pairs in tight ranges",
    ],
}


class NewsTextCache:
    """Fetches, caches, and aligns news headlines with market data timestamps.

    For training AEGIS with text understanding.
    """

    def __init__(self, finnhub_key=None, alpha_vantage_key=None):
        self.finnhub_key = finnhub_key or os.environ.get("FINNHUB_API_KEY", "")
        self.av_key = alpha_vantage_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.cache_dir = os.path.join(DATA_DIR, "news_text_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NandiTrader/2.0"})
        self._rng = np.random.default_rng(42)

    def get_headlines_for_bars(self, pair, timestamps, prices, features_df=None):
        """Get headlines aligned to M5 bar timestamps.

        Args:
            pair: e.g. "eurusd"
            timestamps: DatetimeIndex of M5 bars
            prices: close prices array (same length)
            features_df: optional features DataFrame for market context

        Returns:
            list of list of strings: headlines[bar_idx] = ["headline1", "headline2", ...]
        """
        # Try loading real cached headlines first
        real_headlines = self._load_real_headlines(pair)

        all_headlines = []

        for i, ts in enumerate(timestamps):
            # Check if we have real headlines near this timestamp
            bar_headlines = self._find_nearby_headlines(ts, real_headlines)

            if not bar_headlines:
                # Generate synthetic headlines from market context
                bar_headlines = self._generate_synthetic(
                    pair, ts, prices, i, features_df
                )

            all_headlines.append(bar_headlines)

        n_real = sum(1 for h in all_headlines if any("synthetic" not in str(h) for _ in h))
        logger.info(f"[{pair.upper()}] Aligned headlines: {len(all_headlines)} bars, "
                   f"~{n_real} with real news context")

        return all_headlines

    def fetch_and_cache_real_news(self, pair, start_date, end_date):
        """Fetch real historical headlines from APIs and cache locally.

        Call this once to build the news archive for a pair.
        Rate-limited: ~25 requests/day for Alpha Vantage.
        """
        headlines = []

        # Finnhub market news (forex category, recent)
        if self.finnhub_key:
            try:
                url = "https://finnhub.io/api/v1/news"
                params = {"category": "forex", "token": self.finnhub_key}
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    articles = resp.json()
                    for a in articles:
                        headlines.append({
                            "timestamp": datetime.fromtimestamp(a.get("datetime", 0)).isoformat(),
                            "headline": a.get("headline", ""),
                            "source": a.get("source", "finnhub"),
                            "url": a.get("url", ""),
                        })
                    logger.info(f"Fetched {len(articles)} Finnhub headlines")
            except Exception as e:
                logger.warning(f"Finnhub news fetch failed: {e}")

        # Alpha Vantage news (with sentiment, 25/day limit)
        if self.av_key:
            currency = pair[:3].upper()
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": f"FOREX:{currency}",
                    "apikey": self.av_key,
                    "limit": 200,
                    "sort": "LATEST",
                }
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for a in data.get("feed", []):
                        headlines.append({
                            "timestamp": a.get("time_published", ""),
                            "headline": a.get("title", ""),
                            "source": a.get("source", "alpha_vantage"),
                            "sentiment": a.get("overall_sentiment_score", 0),
                        })
                    logger.info(f"Fetched {len(data.get('feed', []))} Alpha Vantage headlines")
            except Exception as e:
                logger.warning(f"Alpha Vantage news fetch failed: {e}")

        if headlines:
            self._save_headlines(pair, headlines)
            logger.info(f"Cached {len(headlines)} real headlines for {pair.upper()}")

        return headlines

    def _find_nearby_headlines(self, timestamp, headlines_db, window_hours=4):
        """Find headlines published within window_hours before timestamp."""
        if not headlines_db:
            return []

        ts = pd.Timestamp(timestamp)
        window_start = ts - pd.Timedelta(hours=window_hours)

        nearby = []
        for h in headlines_db:
            try:
                h_ts = pd.Timestamp(h["timestamp"])
                if window_start <= h_ts <= ts:
                    nearby.append(h["headline"])
            except Exception:
                continue

        return nearby[:10]  # max 10 headlines

    def _generate_synthetic(self, pair, timestamp, prices, idx, features_df):
        """Generate synthetic headlines from market context.

        Looks at recent price action and generates plausible headlines.
        This teaches AEGIS the relationship between market conditions and news tone.
        """
        if idx < 12:
            return self._pick_templates("neutral", 2)

        # Compute market context from recent bars
        recent_return = (prices[idx] - prices[max(0, idx - 12)]) / prices[max(0, idx - 12)]
        short_return = (prices[idx] - prices[max(0, idx - 3)]) / prices[max(0, idx - 3)]

        # Volatility (std of recent returns)
        if idx >= 36:
            returns = np.diff(prices[max(0, idx - 36):idx + 1]) / prices[max(0, idx - 36):idx]
            vol = np.std(returns) if len(returns) > 1 else 0
        else:
            vol = 0.001

        high_vol = vol > 0.002  # ~20 pips/bar for major pairs
        base_curr = pair[:3].upper()
        quote_curr = pair[3:].upper()

        headlines = []

        # Determine market condition and pick appropriate templates
        if base_curr == "EUR" or quote_curr == "EUR":
            if recent_return > 0.002:
                headlines.extend(self._pick_templates("eur_strong", 1))
            elif recent_return < -0.002:
                headlines.extend(self._pick_templates("eur_weak", 1))

        if base_curr == "GBP" or quote_curr == "GBP":
            if recent_return > 0.002:
                headlines.extend(self._pick_templates("gbp_strong", 1))
            elif recent_return < -0.002:
                headlines.extend(self._pick_templates("gbp_weak", 1))

        if base_curr == "JPY" or quote_curr == "JPY":
            if "usd" in pair and recent_return > 0.002:
                headlines.extend(self._pick_templates("jpy_weak", 1))
            elif "usd" in pair and recent_return < -0.002:
                headlines.extend(self._pick_templates("jpy_strong", 1))

        # USD context
        if "usd" in pair:
            # For EURUSD: positive return = EUR strong = USD weak
            usd_direction = -recent_return if pair.startswith("eur") or pair.startswith("gbp") else recent_return
            if usd_direction > 0.001:
                key = "usd_strong_vol_high" if high_vol else "usd_strong_vol_low"
            elif usd_direction < -0.001:
                key = "usd_weak_vol_high" if high_vol else "usd_weak_vol_low"
            else:
                key = "neutral"
            headlines.extend(self._pick_templates(key, 1))

        # Risk sentiment
        if high_vol and abs(recent_return) > 0.003:
            if pair in ["audusd", "nzdusd"] and recent_return > 0:
                headlines.extend(self._pick_templates("risk_on", 1))
            elif pair in ["audusd", "nzdusd"] and recent_return < 0:
                headlines.extend(self._pick_templates("risk_off", 1))

        # Fill up to at least 2 headlines
        if len(headlines) < 2:
            headlines.extend(self._pick_templates("neutral", 2 - len(headlines)))

        return headlines[:5]  # max 5

    def _pick_templates(self, category, n):
        """Randomly pick n templates from a category."""
        templates = SYNTHETIC_TEMPLATES.get(category, SYNTHETIC_TEMPLATES["neutral"])
        indices = self._rng.choice(len(templates), size=min(n, len(templates)), replace=False)
        return [templates[i] for i in indices]

    def _save_headlines(self, pair, headlines):
        """Save headlines to cache file."""
        path = os.path.join(self.cache_dir, f"{pair}_headlines.json")
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Merge and deduplicate
        seen = set()
        merged = []
        for h in existing + headlines:
            key = h.get("headline", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(h)

        with open(path, "w") as f:
            json.dump(merged, f, indent=2, default=str)

    def _load_real_headlines(self, pair):
        """Load cached real headlines for a pair."""
        path = os.path.join(self.cache_dir, f"{pair}_headlines.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return []


def precompute_text_embeddings(text_encoder, headlines_per_bar, device=None):
    """Pre-compute text embeddings for all bars in training data.

    Args:
        text_encoder: FinancialTextEncoder instance
        headlines_per_bar: list of list of strings [n_bars][n_headlines]
        device: torch device

    Returns:
        embeddings: tensor [n_bars, output_dim] — one embedding per bar
    """
    import torch

    embeddings = []
    batch_size = 50  # process 50 bars at a time

    for i in range(0, len(headlines_per_bar), batch_size):
        batch = headlines_per_bar[i:i + batch_size]
        batch_embs = []

        for headlines in batch:
            if headlines:
                emb, _ = text_encoder(headlines)
                batch_embs.append(emb.squeeze(0).detach())
            else:
                batch_embs.append(torch.zeros(text_encoder.output_dim))

        embeddings.extend(batch_embs)

    result = torch.stack(embeddings)  # [n_bars, output_dim]
    if device:
        result = result.to(device)

    logger.info(f"Pre-computed text embeddings: {result.shape}")
    return result
