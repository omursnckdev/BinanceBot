"""Signals module for technical analysis and market signals."""

from signals.indicators import IndicatorCalculator, TechnicalSignal
from signals.sentiment import SentimentAnalyzer, SentimentSignal
from signals.whales import WhaleDetector, WhaleSignal

__all__ = [
    "IndicatorCalculator",
    "TechnicalSignal",
    "WhaleDetector",
    "WhaleSignal",
    "SentimentAnalyzer",
    "SentimentSignal",
]
