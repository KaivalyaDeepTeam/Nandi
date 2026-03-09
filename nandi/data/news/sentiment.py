"""
News Sentiment — Aggregates forex news sentiment from free sources.

Provides per-currency sentiment scores (-1 bearish to +1 bullish)
that AEGIS can use as features for trade decisions.

Sources (all free):
- Primary: Alpha Vantage News Sentiment (25 req/day, pre-computed scores)
- Secondary: Finnhub Market News (60 req/min, needs local scoring)
- Fallback: RSS feeds from FXStreet/DailyFX (unlimited, needs local scoring)

For RSS fallback, uses simple keyword-based sentiment (no ML dependency).
"""

import os
import json
import logging
import time as time_module
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

from nandi.config import DATA_DIR

logger = logging.getLogger(__name__)

# Currency keywords for headline scoring
BULLISH_KEYWORDS = {
    "USD": ["dollar strength", "fed hawkish", "rate hike", "strong jobs", "beat expectations",
            "dollar rally", "us economy strong", "inflation hot", "yields rise"],
    "EUR": ["ecb hawkish", "euro strength", "eurozone growth", "german data strong"],
    "GBP": ["boe hawkish", "pound strength", "uk growth", "uk inflation"],
    "JPY": ["boj tighten", "yen strength", "japan growth", "yield curve control end"],
    "AUD": ["rba hawkish", "aussie strength", "china stimulus", "commodities rally"],
    "NZD": ["rbnz hawkish", "kiwi strength"],
    "CHF": ["snb hawkish", "franc strength", "safe haven demand"],
    "CAD": ["boc hawkish", "loonie strength", "oil rally", "canada jobs strong"],
}

BEARISH_KEYWORDS = {
    "USD": ["dollar weakness", "fed dovish", "rate cut", "weak jobs", "miss expectations",
            "dollar selloff", "us recession", "yields fall", "fed pause"],
    "EUR": ["ecb dovish", "euro weakness", "eurozone recession", "german data weak"],
    "GBP": ["boe dovish", "pound weakness", "uk recession", "brexit"],
    "JPY": ["boj dovish", "yen weakness", "japan deflation", "intervention"],
    "AUD": ["rba dovish", "aussie weakness", "china slowdown", "commodities crash"],
    "NZD": ["rbnz dovish", "kiwi weakness"],
    "CHF": ["snb dovish", "franc weakness"],
    "CAD": ["boc dovish", "loonie weakness", "oil crash", "canada jobs weak"],
}

# General market sentiment keywords
RISK_ON_KEYWORDS = ["risk on", "stocks rally", "equities rise", "optimism", "bull market"]
RISK_OFF_KEYWORDS = ["risk off", "stocks crash", "recession fears", "panic", "bear market",
                     "geopolitical", "war", "crisis", "contagion"]

# Pair to base/quote currency mapping
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


class NewsSentiment:
    """Aggregates forex news sentiment from multiple free sources.

    Returns per-currency and per-pair sentiment scores.
    Caches aggressively to stay within free API rate limits.
    """

    def __init__(self, alpha_vantage_key=None, finnhub_key=None):
        self.av_key = alpha_vantage_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.finnhub_key = finnhub_key or os.environ.get("FINNHUB_API_KEY", "")
        self.cache_dir = os.path.join(DATA_DIR, "news_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NandiTrader/2.0"})

        # Per-currency sentiment scores (-1 to +1)
        self._currency_sentiment: Dict[str, float] = {}
        self._risk_sentiment: float = 0.0  # risk-on (+1) to risk-off (-1)
        self._last_fetch = 0
        self._fetch_interval = 3600  # 1 hour between fetches

    def refresh(self):
        """Fetch latest sentiment from available sources."""
        now = time_module.time()
        if now - self._last_fetch < self._fetch_interval:
            return

        sentiment = {}
        source = "none"

        # Priority 1: Alpha Vantage (best quality, 25 req/day)
        if self.av_key:
            sentiment = self._fetch_alpha_vantage()
            if sentiment:
                source = "alpha_vantage"

        # Priority 2: Finnhub market news + keyword scoring
        if not sentiment and self.finnhub_key:
            sentiment = self._fetch_finnhub_news()
            if sentiment:
                source = "finnhub"

        # Priority 3: RSS feeds + keyword scoring
        if not sentiment:
            sentiment = self._fetch_rss_sentiment()
            if sentiment:
                source = "rss"

        # Fallback: load cache
        if not sentiment:
            sentiment = self._load_cache()
            source = "cache" if sentiment else "none"

        if sentiment:
            self._currency_sentiment = sentiment.get("currencies", {})
            self._risk_sentiment = sentiment.get("risk", 0.0)
            self._last_fetch = now
            self._save_cache(sentiment)
            logger.info(f"News sentiment updated ({source}): "
                       f"USD={self._currency_sentiment.get('USD', 0):.2f} "
                       f"EUR={self._currency_sentiment.get('EUR', 0):.2f} "
                       f"risk={self._risk_sentiment:.2f}")

    def _fetch_alpha_vantage(self):
        """Fetch pre-computed sentiment from Alpha Vantage."""
        try:
            currencies_to_fetch = ["USD", "EUR", "GBP", "JPY"]
            all_sentiments = {}
            risk_scores = []

            for currency in currencies_to_fetch:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": f"FOREX:{currency}",
                    "apikey": self.av_key,
                    "limit": 50,
                    "sort": "LATEST",
                }

                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                articles = data.get("feed", [])

                if not articles:
                    continue

                # Aggregate sentiment from articles
                scores = []
                for article in articles:
                    # Get overall sentiment
                    score = float(article.get("overall_sentiment_score", 0))

                    # Get ticker-specific sentiment
                    for ticker_info in article.get("ticker_sentiment", []):
                        if currency in ticker_info.get("ticker", ""):
                            relevance = float(ticker_info.get("relevance_score", 0.5))
                            ticker_score = float(ticker_info.get("ticker_sentiment_score", 0))
                            scores.append(ticker_score * relevance)
                            break
                    else:
                        scores.append(score * 0.3)  # lower weight for non-specific articles

                    # Risk sentiment from overall tone
                    risk_scores.append(score)

                if scores:
                    all_sentiments[currency] = max(-1, min(1, sum(scores) / len(scores)))

            # Fill in missing currencies using risk sentiment
            risk = sum(risk_scores) / max(len(risk_scores), 1) if risk_scores else 0
            for curr in ["AUD", "NZD", "CHF", "CAD"]:
                if curr not in all_sentiments:
                    # AUD/NZD track risk sentiment, CHF inverse
                    if curr in ["AUD", "NZD"]:
                        all_sentiments[curr] = risk * 0.5
                    elif curr == "CHF":
                        all_sentiments[curr] = -risk * 0.3
                    else:
                        all_sentiments[curr] = 0.0

            return {"currencies": all_sentiments, "risk": risk} if all_sentiments else {}

        except Exception as e:
            logger.warning(f"Alpha Vantage sentiment failed: {e}")
            return {}

    def _fetch_finnhub_news(self):
        """Fetch forex news from Finnhub and score with keywords."""
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": "forex",
                "token": self.finnhub_key,
            }
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return {}

            articles = resp.json()
            if not articles:
                return {}

            return self._score_headlines(articles, title_key="headline", summary_key="summary")

        except Exception as e:
            logger.warning(f"Finnhub news failed: {e}")
            return {}

    def _fetch_rss_sentiment(self):
        """Fetch from RSS feeds and score headlines."""
        try:
            import feedparser
        except ImportError:
            # feedparser not installed — use basic HTTP + regex
            return self._fetch_rss_basic()

        feeds = [
            "https://www.fxstreet.com/rss",
            "https://www.dailyfx.com/feeds/market-news",
        ]

        all_articles = []
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:50]:
                    all_articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                    })
            except Exception:
                continue

        if not all_articles:
            return {}

        return self._score_headlines(all_articles, title_key="title", summary_key="summary")

    def _fetch_rss_basic(self):
        """Minimal RSS fetch without feedparser dependency."""
        feeds = [
            "https://www.fxstreet.com/rss",
        ]

        all_articles = []
        for url in feeds:
            try:
                resp = self._session.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                # Extract titles with regex
                titles = re.findall(r"<title[^>]*>(.*?)</title>", resp.text, re.DOTALL)
                for title in titles:
                    title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title)
                    all_articles.append({"title": title.strip(), "summary": ""})
            except Exception:
                continue

        if not all_articles:
            return {}

        return self._score_headlines(all_articles, title_key="title", summary_key="summary")

    def _score_headlines(self, articles, title_key="title", summary_key="summary"):
        """Score articles using keyword matching for each currency."""
        currency_scores = {curr: [] for curr in BULLISH_KEYWORDS}
        risk_scores = []

        for article in articles:
            text = (article.get(title_key, "") + " " + article.get(summary_key, "")).lower()

            if not text.strip():
                continue

            # Score each currency
            for currency in BULLISH_KEYWORDS:
                score = 0
                for kw in BULLISH_KEYWORDS[currency]:
                    if kw in text:
                        score += 1
                for kw in BEARISH_KEYWORDS[currency]:
                    if kw in text:
                        score -= 1

                if score != 0:
                    currency_scores[currency].append(max(-1, min(1, score)))

            # Risk sentiment
            for kw in RISK_ON_KEYWORDS:
                if kw in text:
                    risk_scores.append(0.5)
            for kw in RISK_OFF_KEYWORDS:
                if kw in text:
                    risk_scores.append(-0.5)

        # Average scores per currency
        result = {}
        for currency, scores in currency_scores.items():
            if scores:
                result[currency] = max(-1, min(1, sum(scores) / len(scores)))
            else:
                result[currency] = 0.0

        risk = sum(risk_scores) / max(len(risk_scores), 1) if risk_scores else 0
        return {"currencies": result, "risk": risk}

    def get_pair_sentiment(self, pair):
        """Get sentiment score for a specific pair.

        For EURUSD: EUR_sentiment - USD_sentiment
        Positive = bullish for pair (buy), Negative = bearish (sell)

        Returns float in [-1, 1]
        """
        self.refresh()

        base_curr, quote_curr = PAIR_CURRENCIES.get(pair, ("", ""))
        base_sent = self._currency_sentiment.get(base_curr, 0.0)
        quote_sent = self._currency_sentiment.get(quote_curr, 0.0)

        # Pair sentiment = base - quote
        # EURUSD: EUR bullish + USD bearish → positive (buy EURUSD)
        return max(-1, min(1, base_sent - quote_sent))

    def get_features(self, pair):
        """Get numeric features for AEGIS.

        Returns dict with:
        - news_sentiment: pair sentiment (-1 to +1)
        - base_sentiment: base currency sentiment
        - quote_sentiment: quote currency sentiment
        - risk_sentiment: risk-on (+1) to risk-off (-1)
        - sentiment_strength: absolute magnitude of pair sentiment
        """
        self.refresh()

        base_curr, quote_curr = PAIR_CURRENCIES.get(pair, ("", ""))
        base_sent = self._currency_sentiment.get(base_curr, 0.0)
        quote_sent = self._currency_sentiment.get(quote_curr, 0.0)
        pair_sent = max(-1, min(1, base_sent - quote_sent))

        return {
            "news_sentiment": pair_sent,
            "base_sentiment": base_sent,
            "quote_sentiment": quote_sent,
            "risk_sentiment": self._risk_sentiment,
            "sentiment_strength": abs(pair_sent),
        }

    def _save_cache(self, data):
        try:
            path = os.path.join(self.cache_dir, "sentiment.json")
            with open(path, "w") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data,
                }, f, indent=2)
        except Exception:
            pass

    def _load_cache(self):
        try:
            path = os.path.join(self.cache_dir, "sentiment.json")
            if not os.path.exists(path):
                return {}
            with open(path) as f:
                cached = json.load(f)
            # Accept cache up to 6 hours old
            ts = datetime.fromisoformat(cached["timestamp"])
            if (datetime.utcnow() - ts).total_seconds() > 21600:
                return {}
            return cached.get("data", {})
        except Exception:
            return {}
