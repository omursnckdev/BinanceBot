#!/usr/bin/env python3
"""
Test script for sentiment analysis with CryptoPanic API.

Usage:
    python test_sentiment.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Get API key from environment or use default for testing
API_KEY = os.environ.get("NEWS_API_KEY", os.environ.get("CRYPTOPANIC_API_KEY", ""))


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


class CryptoPanicProvider:
    """CryptoPanic news provider for testing."""

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _extract_source_from_description(self, description: str) -> str:
        """Try to extract source name from description."""
        import re
        if not description:
            return "CryptoPanic"

        patterns = [
            r"^According to ([A-Za-z0-9\s]+),",
            r"^([A-Za-z0-9\s]+) reports? that",
            r"^([A-Za-z0-9\s]+) announced",
        ]
        for pattern in patterns:
            match = re.match(pattern, description)
            if match:
                return match.group(1).strip()

        return "CryptoPanic"

    def fetch_articles(self, symbol: str | None = None) -> list[NewsArticle]:
        """Fetch articles from CryptoPanic."""
        params: dict[str, Any] = {
            "auth_token": self.api_key,
            "public": "true",
        }

        if symbol:
            currency = symbol.replace("USDT", "").upper()
            params["currencies"] = currency

        response = requests.get(self.BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("results", [])[:50]:
            published_str = item.get("published_at", "")
            if published_str:
                published = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            else:
                published = datetime.now(timezone.utc)

            description = item.get("description", "")
            slug = item.get("slug", "")
            url = f"https://cryptopanic.com/news/{item.get('id', '')}/{slug}" if slug else ""
            source = self._extract_source_from_description(description)

            articles.append(NewsArticle(
                title=item.get("title", ""),
                source=source,
                published_at=published,
                url=url,
                summary=description,
            ))

        return articles


def test_cryptopanic_direct():
    """Test CryptoPanic provider directly."""
    print("=" * 60)
    print("Testing CryptoPanic Provider Directly")
    print("=" * 60)

    provider = CryptoPanicProvider(api_key=API_KEY)

    # Fetch general articles
    print("\n--- Fetching general crypto news ---")
    articles = provider.fetch_articles()
    print(f"Fetched {len(articles)} articles")

    for i, article in enumerate(articles[:5]):
        print(f"\n[{i+1}] {article.title[:70]}...")
        print(f"    Source: {article.source}")
        print(f"    Published: {article.published_at}")
        if article.summary:
            print(f"    Summary: {article.summary[:100]}...")

    # Fetch BTC-specific articles
    print("\n--- Fetching BTC-specific news ---")
    btc_articles = provider.fetch_articles("BTCUSDT")
    print(f"Fetched {len(btc_articles)} BTC articles")

    for i, article in enumerate(btc_articles[:3]):
        print(f"\n[{i+1}] {article.title[:70]}...")


def test_sentiment_analyzer():
    """Test sentiment analysis on fetched articles."""
    print("\n" + "=" * 60)
    print("Testing Sentiment Analysis on Articles")
    print("=" * 60)

    provider = CryptoPanicProvider(api_key=API_KEY)
    vader = SentimentIntensityAnalyzer()

    # Test for each symbol
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    for symbol in symbols:
        print(f"\n--- Sentiment for {symbol} ---")
        articles = provider.fetch_articles(symbol)
        print(f"Fetched {len(articles)} articles")

        if not articles:
            print("No articles found")
            continue

        # Analyze sentiment
        scores = []
        for article in articles[:10]:
            text = article.title
            if article.summary:
                text += " " + article.summary
            sentiment = vader.polarity_scores(text)
            article.sentiment_score = sentiment["compound"]
            scores.append(sentiment["compound"])

        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"Average Sentiment: {avg_score:.4f}")
        print(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}")

        # Show top articles by sentiment impact
        sorted_articles = sorted(articles[:10], key=lambda a: abs(a.sentiment_score or 0), reverse=True)
        print("\nMost impactful headlines:")
        for article in sorted_articles[:3]:
            print(f"  [{article.sentiment_score:+.3f}] {article.title[:60]}...")


def test_vader_sentiment():
    """Test VADER sentiment on sample headlines."""
    print("\n" + "=" * 60)
    print("Testing VADER Sentiment Analysis")
    print("=" * 60)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    headlines = [
        "Bitcoin surges to new all-time high as institutional adoption grows",
        "Crypto market crashes as major exchange files for bankruptcy",
        "Ethereum upgrade successfully deployed, network performance improves",
        "SEC announces new regulations that could impact cryptocurrency trading",
        "Bitcoin mining company reports increase in holdings",
    ]

    for headline in headlines:
        scores = analyzer.polarity_scores(headline)
        print(f"\nHeadline: {headline}")
        print(f"  Compound: {scores['compound']:.4f}")
        print(f"  Positive: {scores['pos']:.4f}")
        print(f"  Negative: {scores['neg']:.4f}")
        print(f"  Neutral: {scores['neu']:.4f}")


if __name__ == "__main__":
    print("CryptoPanic Sentiment Analysis Test")

    if not API_KEY:
        print("ERROR: No API key found!")
        print("Set NEWS_API_KEY or CRYPTOPANIC_API_KEY environment variable")
        print("Example: NEWS_API_KEY=your_api_key python test_sentiment.py")
        sys.exit(1)

    print("API Key: " + API_KEY[:8] + "...")

    try:
        test_cryptopanic_direct()
        test_sentiment_analyzer()
        test_vader_sentiment()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
