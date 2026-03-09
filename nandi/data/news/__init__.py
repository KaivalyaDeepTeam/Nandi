"""News intelligence module — calendar, sentiment, rate differentials, text."""

from nandi.data.news.calendar import EconomicCalendar
from nandi.data.news.sentiment import NewsSentiment
from nandi.data.news.rates import RateDifferentials
from nandi.data.news.gate import NewsGate
from nandi.data.news.text_cache import NewsTextCache

__all__ = [
    "EconomicCalendar", "NewsSentiment", "RateDifferentials",
    "NewsGate", "NewsTextCache",
]
