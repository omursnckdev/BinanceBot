"""
Technical indicators module.

Computes:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- EMA (Exponential Moving Average) trend filter

Produces tech_score ∈ [-1, +1] and tech_confidence ∈ [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from config import IndicatorConfig, Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TrendDirection(str, Enum):
    """Trend direction based on EMA filter."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IndicatorValues:
    """Container for all indicator values."""

    # RSI
    rsi: float
    rsi_signal: float  # -1 (oversold), 0 (neutral), +1 (overbought)

    # MACD
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_signal_value: float  # -1, 0, +1 based on histogram

    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float  # 0-1 where price is in bands
    bb_signal: float  # -1 (oversold), 0 (neutral), +1 (overbought)

    # ATR
    atr: float
    atr_pct: float  # ATR as percentage of price

    # EMA Trend
    ema_fast: float
    ema_slow: float
    trend: TrendDirection
    trend_strength: float  # 0-1


@dataclass
class TechnicalSignal:
    """Technical analysis signal output."""

    tech_score: float  # -1 to +1
    tech_confidence: float  # 0 to 1
    trend: TrendDirection
    atr: float
    atr_pct: float
    indicators: IndicatorValues
    is_valid: bool
    invalidation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tech_score": round(self.tech_score, 4),
            "tech_confidence": round(self.tech_confidence, 4),
            "trend": self.trend.value,
            "atr": round(self.atr, 4),
            "atr_pct": round(self.atr_pct, 4),
            "is_valid": self.is_valid,
            "rsi": round(self.indicators.rsi, 2),
            "macd_histogram": round(self.indicators.macd_histogram, 6),
            "bb_position": round(self.indicators.bb_position, 4),
        }


class IndicatorCalculator:
    """
    Calculates technical indicators and generates signals.

    All calculations use pandas for efficiency and correctness.
    """

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.config = config or get_settings().indicators

    def calculate_rsi(self, df: pd.DataFrame, period: int | None = None) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        where RS = avg gain / avg loss over period
        """
        period = period or self.config.rsi_period
        delta = df["close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use exponential moving average for smoother results
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns: (macd_line, signal_line, histogram)
        """
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int | None = None,
        std_dev: float | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns: (upper_band, middle_band, lower_band)
        """
        period = period or self.config.bb_period
        std_dev = std_dev or self.config.bb_std_dev

        middle = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def calculate_atr(self, df: pd.DataFrame, period: int | None = None) -> pd.Series:
        """
        Calculate ATR (Average True Range).

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        period = period or self.config.atr_period

        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate EMA (Exponential Moving Average)."""
        return df["close"].ewm(span=period, adjust=False).mean()

    def get_trend_direction(
        self,
        ema_fast: float,
        ema_slow: float,
        current_price: float,
    ) -> tuple[TrendDirection, float]:
        """
        Determine trend direction and strength based on EMA crossover.

        Returns: (direction, strength)
        """
        # Calculate percentage difference between EMAs
        if ema_slow == 0:
            return TrendDirection.NEUTRAL, 0.0

        ema_diff_pct = ((ema_fast - ema_slow) / ema_slow) * 100

        # Price position relative to EMAs
        price_above_fast = current_price > ema_fast
        price_above_slow = current_price > ema_slow

        # Determine trend
        if ema_fast > ema_slow and price_above_fast:
            direction = TrendDirection.BULLISH
        elif ema_fast < ema_slow and not price_above_fast:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        # Trend strength (0-1 based on EMA separation)
        strength = min(abs(ema_diff_pct) / 2.0, 1.0)  # Cap at 2% = full strength

        return direction, strength

    def calculate_all(self, df: pd.DataFrame) -> IndicatorValues | None:
        """
        Calculate all indicators from OHLCV DataFrame.

        Returns None if insufficient data.
        """
        if df is None or len(df) < max(
            self.config.ema_slow,
            self.config.bb_period,
            self.config.macd_slow,
        ):
            return None

        try:
            current_price = df["close"].iloc[-1]

            # RSI
            rsi_series = self.calculate_rsi(df)
            rsi = rsi_series.iloc[-1]

            # RSI signal
            if rsi <= self.config.rsi_oversold:
                rsi_signal = -1.0  # Oversold = bullish signal (reversal expected)
            elif rsi >= self.config.rsi_overbought:
                rsi_signal = 1.0  # Overbought = bearish signal
            else:
                # Normalize to -1 to +1 based on position in range
                mid = (self.config.rsi_overbought + self.config.rsi_oversold) / 2
                rsi_signal = (rsi - mid) / (self.config.rsi_overbought - mid)

            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(df)
            macd_hist_val = histogram.iloc[-1]

            # MACD signal based on histogram momentum
            if macd_hist_val > 0 and histogram.iloc[-2] < histogram.iloc[-1]:
                macd_signal_val = 1.0  # Bullish momentum
            elif macd_hist_val < 0 and histogram.iloc[-2] > histogram.iloc[-1]:
                macd_signal_val = -1.0  # Bearish momentum
            else:
                macd_signal_val = np.sign(macd_hist_val) * 0.5

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
            bb_u = bb_upper.iloc[-1]
            bb_m = bb_middle.iloc[-1]
            bb_l = bb_lower.iloc[-1]
            bb_width = (bb_u - bb_l) / bb_m if bb_m > 0 else 0

            # Position in bands (0 = at lower, 1 = at upper)
            bb_range = bb_u - bb_l
            if bb_range > 0:
                bb_position = (current_price - bb_l) / bb_range
                bb_position = max(0, min(1, bb_position))
            else:
                bb_position = 0.5

            # BB signal
            if bb_position < 0.2:
                bb_signal = -1.0  # Near lower band = bullish (mean reversion)
            elif bb_position > 0.8:
                bb_signal = 1.0  # Near upper band = bearish
            else:
                bb_signal = (bb_position - 0.5) * 2

            # ATR
            atr_series = self.calculate_atr(df)
            atr = atr_series.iloc[-1]
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

            # EMAs
            ema_fast = self.calculate_ema(df, self.config.ema_fast).iloc[-1]
            ema_slow = self.calculate_ema(df, self.config.ema_slow).iloc[-1]
            trend, trend_strength = self.get_trend_direction(
                ema_fast, ema_slow, current_price
            )

            return IndicatorValues(
                rsi=rsi,
                rsi_signal=rsi_signal,
                macd_line=macd_line.iloc[-1],
                macd_signal=signal_line.iloc[-1],
                macd_histogram=macd_hist_val,
                macd_signal_value=macd_signal_val,
                bb_upper=bb_u,
                bb_middle=bb_m,
                bb_lower=bb_l,
                bb_width=bb_width,
                bb_position=bb_position,
                bb_signal=bb_signal,
                atr=atr,
                atr_pct=atr_pct,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                trend=trend,
                trend_strength=trend_strength,
            )

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def generate_signal(
        self,
        df: pd.DataFrame,
        confirmation_dfs: list[pd.DataFrame] | None = None,
    ) -> TechnicalSignal:
        """
        Generate technical signal from indicators.

        Args:
            df: Primary timeframe OHLCV DataFrame
            confirmation_dfs: List of confirmation timeframe DataFrames

        Returns:
            TechnicalSignal with score, confidence, and indicator values
        """
        indicators = self.calculate_all(df)

        if indicators is None:
            return TechnicalSignal(
                tech_score=0.0,
                tech_confidence=0.0,
                trend=TrendDirection.NEUTRAL,
                atr=0.0,
                atr_pct=0.0,
                indicators=IndicatorValues(
                    rsi=50, rsi_signal=0,
                    macd_line=0, macd_signal=0, macd_histogram=0, macd_signal_value=0,
                    bb_upper=0, bb_middle=0, bb_lower=0, bb_width=0, bb_position=0.5, bb_signal=0,
                    atr=0, atr_pct=0,
                    ema_fast=0, ema_slow=0, trend=TrendDirection.NEUTRAL, trend_strength=0,
                ),
                is_valid=False,
                invalidation_reason="Insufficient data for indicator calculation",
            )

        # Calculate component scores
        # RSI: Inverted because oversold is bullish
        rsi_component = -indicators.rsi_signal * 0.25

        # MACD: Direct signal
        macd_component = indicators.macd_signal_value * 0.35

        # Bollinger: Inverted for mean reversion
        bb_component = -indicators.bb_signal * 0.20

        # Trend alignment bonus
        trend_component = 0.0
        if indicators.trend == TrendDirection.BULLISH:
            trend_component = indicators.trend_strength * 0.20
        elif indicators.trend == TrendDirection.BEARISH:
            trend_component = -indicators.trend_strength * 0.20

        # Combine into tech_score
        tech_score = rsi_component + macd_component + bb_component + trend_component

        # Clamp to [-1, 1]
        tech_score = max(-1.0, min(1.0, tech_score))

        # Calculate confidence based on:
        # 1. Agreement between indicators
        # 2. Trend strength
        # 3. Signal clarity

        signals = [
            -indicators.rsi_signal,  # Inverted
            indicators.macd_signal_value,
            -indicators.bb_signal,  # Inverted
        ]

        # Agreement: how many signals agree on direction
        positive_count = sum(1 for s in signals if s > 0)
        negative_count = sum(1 for s in signals if s < 0)
        agreement = max(positive_count, negative_count) / len(signals)

        # Signal clarity: how strong are the individual signals
        signal_strength = sum(abs(s) for s in signals) / len(signals)

        # Combine into confidence
        confidence = (agreement * 0.5 + signal_strength * 0.3 + indicators.trend_strength * 0.2)
        confidence = max(0.0, min(1.0, confidence))

        # Apply confirmation timeframe adjustment
        if confirmation_dfs:
            confirmation_scores = []
            for conf_df in confirmation_dfs:
                conf_indicators = self.calculate_all(conf_df)
                if conf_indicators:
                    # Get trend alignment
                    if conf_indicators.trend == indicators.trend:
                        confirmation_scores.append(1.0)
                    elif conf_indicators.trend == TrendDirection.NEUTRAL:
                        confirmation_scores.append(0.5)
                    else:
                        confirmation_scores.append(0.0)

            if confirmation_scores:
                avg_confirmation = sum(confirmation_scores) / len(confirmation_scores)
                # Reduce confidence if higher timeframes disagree
                confidence *= (0.5 + 0.5 * avg_confirmation)

        return TechnicalSignal(
            tech_score=tech_score,
            tech_confidence=confidence,
            trend=indicators.trend,
            atr=indicators.atr,
            atr_pct=indicators.atr_pct,
            indicators=indicators,
            is_valid=True,
        )
