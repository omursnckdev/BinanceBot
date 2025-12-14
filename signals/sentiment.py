"""
News sentiment analysis module.

Features:
- Multiple news provider support (CryptoPanic, NewsAPI, etc.)
- Response caching with TTL
- VADER sentiment analysis (with optional transformer model)
- Keyword-based coin mapping
- Rate limit handling
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SentimentConfig, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """Parsed news article."""

    title: str
    source: str
    published_at: datetime
    url: str
    summary: str | None = None
    sentiment_score: float | None = None
    matched_coins: list[str] = field(default_factory=list)

    def age_minutes(self) -> float:
        """Get article age in minutes."""
        now = datetime.now(timezone.utc)
        return (now - self.published_at).total_seconds() / 60


@dataclass
class SentimentSignal:
    """Sentiment analysis signal output."""

    sentiment_score: float  # -1 to +1 per symbol
    sentiment_confidence: float  # 0 to 1
    global_sentiment: float  # Overall crypto market sentiment
    article_count: int
    recent_articles: list[str]  # Titles of most impactful articles
    is_valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_confidence": round(self.sentiment_confidence, 4),
            "global_sentiment": round(self.global_sentiment, 4),
            "article_count": self.article_count,
            "is_valid": self.is_valid,
        }


@dataclass
class CachedResult:
    """Cached API response."""

    data: list[NewsArticle]
    timestamp: float
    symbol: str | None = None


class NewsProvider:
    """Base class for news providers."""

    def fetch_articles(self, symbol: str | None = None) -> list[NewsArticle]:
        """Fetch articles from provider."""
        raise NotImplementedError


class CryptoPanicProvider(NewsProvider):
    """CryptoPanic news provider."""

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str, config: SentimentConfig) -> None:
        self.api_key = api_key
        self.config = config

    def fetch_articles(self, symbol: str | None = None) -> list[NewsArticle]:
        """Fetch articles from CryptoPanic."""
        try:
            params: dict[str, Any] = {
                "auth_token": self.api_key,
                "public": "true",
                "kind": "news",
            }

            if symbol:
                # Map symbol to CryptoPanic currency filter
                currency = symbol.replace("USDT", "").upper()
                params["currencies"] = currency

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("results", [])[:self.config.max_articles_per_fetch]:
                try:
                    published = datetime.fromisoformat(
                        item["published_at"].replace("Z", "+00:00")
                    )
                    articles.append(NewsArticle(
                        title=item.get("title", ""),
                        source=item.get("source", {}).get("title", "Unknown"),
                        published_at=published,
                        url=item.get("url", ""),
                        summary=None,
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")

            return articles

        except Exception as e:
            logger.error(f"CryptoPanic API error: {e}")
            return []


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, config: SentimentConfig) -> None:
        self.api_key = api_key
        self.config = config

    def fetch_articles(self, symbol: str | None = None) -> list[NewsArticle]:
        """Fetch articles from NewsAPI."""
        try:
            # Build search query
            if symbol:
                currency = symbol.replace("USDT", "")
                keywords = self.config.keyword_map.get(currency, [currency.lower()])
                query = " OR ".join(keywords)
            else:
                query = "cryptocurrency OR bitcoin OR ethereum"

            params = {
                "apiKey": self.api_key,
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(100, self.config.max_articles_per_fetch),
            }

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    )
                    articles.append(NewsArticle(
                        title=item.get("title", ""),
                        source=item.get("source", {}).get("name", "Unknown"),
                        published_at=published,
                        url=item.get("url", ""),
                        summary=item.get("description"),
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")

            return articles

        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []


class SentimentAnalyzer:
    """
    Analyzes news sentiment for trading signals.

    Features:
    - Multiple news provider support
    - VADER sentiment analysis
    - Coin-specific sentiment mapping
    - Response caching
    """

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or get_settings().sentiment
        self._cache: dict[str, CachedResult] = {}
        self._vader = SentimentIntensityAnalyzer()
        self._provider = self._create_provider()
        self._last_fetch: dict[str, float] = {}

    def _create_provider(self) -> NewsProvider | None:
        """Create news provider based on config."""
        api_key = os.getenv(self.config.api_key_env_var, "")

        if not api_key:
            logger.warning(
                f"News API key not found in {self.config.api_key_env_var}. "
                "Sentiment analysis will be disabled."
            )
            return None

        if self.config.provider == "cryptopanic":
            return CryptoPanicProvider(api_key, self.config)
        elif self.config.provider == "newsapi":
            return NewsAPIProvider(api_key, self.config)
        else:
            logger.warning(f"Unknown news provider: {self.config.provider}")
            return None

    def _get_cached(self, cache_key: str) -> list[NewsArticle] | None:
        """Get cached result if still valid."""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.timestamp < self.config.cache_ttl_seconds:
                return cached.data
        return None

    def _set_cached(
        self,
        cache_key: str,
        data: list[NewsArticle],
        symbol: str | None = None,
    ) -> None:
        """Cache result."""
        self._cache[cache_key] = CachedResult(
            data=data,
            timestamp=time.time(),
            symbol=symbol,
        )

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER.

        Returns: -1 to +1 sentiment score
        """
        if not text:
            return 0.0

        scores = self._vader.polarity_scores(text)
        return scores["compound"]

    def map_article_to_coins(self, article: NewsArticle) -> list[str]:
        """Map article to relevant coins based on keywords."""
        text = (article.title + " " + (article.summary or "")).lower()
        matched = []

        for coin, keywords in self.config.keyword_map.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    matched.append(coin)
                    break

        return matched

    def fetch_and_analyze(self, symbol: str | None = None) -> list[NewsArticle]:
        """
        Fetch articles and analyze sentiment.

        Returns analyzed articles with sentiment scores.
        """
        if not self._provider:
            return []

        cache_key = f"articles_{symbol or 'global'}"

        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Rate limit check
        last_fetch = self._last_fetch.get(cache_key, 0)
        if time.time() - last_fetch < 60:  # Min 60 seconds between fetches
            return self._cache.get(cache_key, CachedResult([], 0)).data

        # Fetch articles
        articles = self._provider.fetch_articles(symbol)
        self._last_fetch[cache_key] = time.time()

        # Analyze each article
        for article in articles:
            # Analyze title + summary
            text = article.title
            if article.summary:
                text += " " + article.summary

            article.sentiment_score = self.analyze_text(text)
            article.matched_coins = self.map_article_to_coins(article)

        # Cache results
        self._set_cached(cache_key, articles, symbol)

        logger.debug(
            f"Fetched {len(articles)} articles for {symbol or 'global'}",
            extra={"symbol": symbol, "count": len(articles)},
        )

        return articles

    def generate_signal(self, symbol: str) -> SentimentSignal:
        """
        Generate sentiment signal for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            SentimentSignal with score and confidence
        """
        if not self.config.enabled or not self._provider:
            return SentimentSignal(
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                global_sentiment=0.0,
                article_count=0,
                recent_articles=[],
                is_valid=False,
                error="Sentiment analysis disabled or no API key",
            )

        try:
            # Get coin-specific articles
            coin = symbol.replace("USDT", "")
            coin_articles = self.fetch_and_analyze(symbol)

            # Get global crypto articles
            global_articles = self.fetch_and_analyze(None)

            # Filter coin-specific from global
            relevant_articles = []
            for article in coin_articles:
                if coin in article.matched_coins or not article.matched_coins:
                    relevant_articles.append(article)

            # Also check global articles for coin mentions
            for article in global_articles:
                if coin in article.matched_coins and article not in relevant_articles:
                    relevant_articles.append(article)

            if not relevant_articles:
                return SentimentSignal(
                    sentiment_score=0.0,
                    sentiment_confidence=0.0,
                    global_sentiment=self._calculate_global_sentiment(global_articles),
                    article_count=0,
                    recent_articles=[],
                    is_valid=True,
                )

            # Weight by recency (newer articles matter more)
            weighted_scores = []
            weights = []

            for article in relevant_articles:
                if article.sentiment_score is None:
                    continue

                age_hours = article.age_minutes() / 60
                # Exponential decay: half weight every 4 hours
                weight = 2 ** (-age_hours / 4)
                weighted_scores.append(article.sentiment_score * weight)
                weights.append(weight)

            if not weights:
                sentiment_score = 0.0
            else:
                sentiment_score = sum(weighted_scores) / sum(weights)

            # Calculate confidence based on:
            # 1. Number of articles
            # 2. Agreement between articles
            # 3. Recency of articles

            article_count = len(relevant_articles)
            count_factor = min(article_count / 10, 1.0)  # 10 articles = full confidence

            # Agreement: standard deviation of sentiment
            if len(weighted_scores) > 1:
                scores_only = [s / w for s, w in zip(weighted_scores, weights)]
                std_dev = float(np.std(scores_only))
                agreement_factor = max(0, 1 - std_dev)  # Lower std = more agreement
            else:
                agreement_factor = 0.5

            # Recency: average article age
            avg_age_hours = sum(a.age_minutes() for a in relevant_articles) / (60 * len(relevant_articles))
            recency_factor = max(0, 1 - avg_age_hours / 24)  # Full confidence if < 24h old

            confidence = (count_factor * 0.4 + agreement_factor * 0.4 + recency_factor * 0.2)

            # Get most impactful recent articles
            sorted_articles = sorted(
                relevant_articles,
                key=lambda a: abs(a.sentiment_score or 0) * (1 / max(1, a.age_minutes())),
                reverse=True,
            )
            recent_titles = [a.title for a in sorted_articles[:3]]

            return SentimentSignal(
                sentiment_score=max(-1, min(1, sentiment_score)),
                sentiment_confidence=confidence,
                global_sentiment=self._calculate_global_sentiment(global_articles),
                article_count=article_count,
                recent_articles=recent_titles,
                is_valid=True,
            )

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentSignal(
                sentiment_score=0.0,
                sentiment_confidence=0.0,
                global_sentiment=0.0,
                article_count=0,
                recent_articles=[],
                is_valid=False,
                error=str(e),
            )

    def _calculate_global_sentiment(self, articles: list[NewsArticle]) -> float:
        """Calculate overall crypto market sentiment."""
        if not articles:
            return 0.0

        scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._last_fetch.clear()


# Import numpy for std calculation
import numpy as np
