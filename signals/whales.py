"""
Whale activity detection module.

Detects large trades and order book imbalances using Binance public market data.

Features:
- AggTrades analysis: flags unusually large notional trades
- Order book imbalance detection
- Rolling percentile thresholds
- Produces whale_score and whale_alert_opposite signals
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from config import WhaleConfig, WhaleDefensiveAction, get_settings
from data.market_data import MarketDataManager, OrderBookSnapshot, Trade
from utils.logger import get_logger

logger = get_logger(__name__)


class WhalePressure(str, Enum):
    """Direction of whale pressure."""

    BULLISH = "bullish"  # Whales buying
    BEARISH = "bearish"  # Whales selling
    NEUTRAL = "neutral"


@dataclass
class WhaleSignal:
    """Whale detection signal output."""

    whale_score: float  # -1 to +1 (negative = bearish whales, positive = bullish)
    whale_confidence: float  # 0 to 1
    whale_alert_opposite: bool  # True if whale activity strongly opposes position
    pressure: WhalePressure
    large_trades_detected: int
    order_book_imbalance: float
    recommended_action: WhaleDefensiveAction | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "whale_score": round(self.whale_score, 4),
            "whale_confidence": round(self.whale_confidence, 4),
            "whale_alert_opposite": self.whale_alert_opposite,
            "pressure": self.pressure.value,
            "large_trades_detected": self.large_trades_detected,
            "order_book_imbalance": round(self.order_book_imbalance, 4),
            "recommended_action": self.recommended_action.value if self.recommended_action else None,
        }


@dataclass
class TradeStats:
    """Rolling statistics for trade analysis."""

    notionals: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    timestamps: deque[int] = field(default_factory=lambda: deque(maxlen=10000))
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    last_update: float = 0.0

    def add_trade(self, notional: float, timestamp: int) -> None:
        """Add a trade to statistics."""
        self.notionals.append(notional)
        self.timestamps.append(timestamp)
        self.last_update = time.time()

    def update_percentiles(self) -> None:
        """Recalculate percentile thresholds."""
        if len(self.notionals) >= 100:
            arr = np.array(list(self.notionals))
            self.percentile_95 = float(np.percentile(arr, 95))
            self.percentile_99 = float(np.percentile(arr, 99))

    def get_recent_notionals(self, window_minutes: int) -> list[float]:
        """Get notionals within time window."""
        cutoff = int((time.time() - window_minutes * 60) * 1000)
        return [
            n for n, t in zip(self.notionals, self.timestamps)
            if t >= cutoff
        ]


class WhaleDetector:
    """
    Detects whale activity using market data.

    Detection methods:
    1. Large trade detection via aggregated trades
    2. Order book imbalance analysis
    3. Rolling percentile thresholds for adaptive detection
    """

    def __init__(
        self,
        market_data: MarketDataManager,
        config: WhaleConfig | None = None,
    ) -> None:
        self.market_data = market_data
        self.config = config or get_settings().whale
        self._trade_stats: dict[str, TradeStats] = {}
        self._last_imbalance: dict[str, float] = {}

    def get_trade_stats(self, symbol: str) -> TradeStats:
        """Get or create trade statistics for a symbol."""
        if symbol not in self._trade_stats:
            self._trade_stats[symbol] = TradeStats()
        return self._trade_stats[symbol]

    def analyze_trades(self, symbol: str) -> tuple[float, int, float]:
        """
        Analyze recent trades for whale activity.

        Returns: (whale_pressure, large_trade_count, confidence)
            whale_pressure: -1 (selling) to +1 (buying)
            large_trade_count: number of whale trades detected
            confidence: 0 to 1
        """
        symbol_data = self.market_data.get_symbol_data(symbol)
        stats = self.get_trade_stats(symbol)

        if not symbol_data.trades:
            return 0.0, 0, 0.0

        # Update trade statistics
        for trade in symbol_data.trades:
            if trade.timestamp > stats.last_update * 1000:
                stats.add_trade(trade.notional, trade.timestamp)

        stats.update_percentiles()

        # Get recent trades within window
        window_notionals = stats.get_recent_notionals(self.config.rolling_window_minutes)
        if not window_notionals:
            return 0.0, 0, 0.0

        # Determine whale threshold
        threshold = max(
            stats.percentile_95,
            self.config.min_whale_notional_usd,
        )

        # Find whale trades
        whale_buys = 0.0
        whale_sells = 0.0
        large_trade_count = 0

        for trade in symbol_data.trades:
            # Check if trade is recent enough
            age_ms = int(time.time() * 1000) - trade.timestamp
            if age_ms > self.config.rolling_window_minutes * 60 * 1000:
                continue

            if trade.notional >= threshold:
                large_trade_count += 1
                if trade.is_buy:
                    whale_buys += trade.notional
                else:
                    whale_sells += trade.notional

        # Calculate whale pressure
        total_whale_volume = whale_buys + whale_sells
        if total_whale_volume == 0:
            return 0.0, 0, 0.0

        whale_pressure = (whale_buys - whale_sells) / total_whale_volume

        # Confidence based on volume significance
        total_recent_volume = sum(t.notional for t in symbol_data.trades)
        if total_recent_volume > 0:
            whale_ratio = total_whale_volume / total_recent_volume
            confidence = min(whale_ratio * 5, 1.0)  # 20% whale volume = full confidence
        else:
            confidence = 0.0

        return whale_pressure, large_trade_count, confidence

    def analyze_order_book(self, symbol: str) -> tuple[float, float]:
        """
        Analyze order book for whale imbalance.

        Returns: (imbalance, confidence)
            imbalance: -1 (more asks) to +1 (more bids)
            confidence: 0 to 1
        """
        symbol_data = self.market_data.get_symbol_data(symbol)
        order_book = symbol_data.order_book

        if not order_book:
            return 0.0, 0.0

        # Calculate imbalance at configured depth
        imbalance = order_book.imbalance(self.config.depth_levels)

        # Calculate imbalance change (spike detection)
        prev_imbalance = self._last_imbalance.get(symbol, 0.0)
        imbalance_change = abs(imbalance - prev_imbalance)
        self._last_imbalance[symbol] = imbalance

        # Confidence based on imbalance magnitude and change
        confidence = 0.0

        # Strong imbalance
        if abs(imbalance) >= self.config.imbalance_threshold:
            confidence += 0.5

        # Sudden change in imbalance (spike)
        if imbalance_change >= 0.3:
            confidence += 0.3

        # Volume significance
        total_depth = order_book.bid_depth(self.config.depth_levels) + \
                     order_book.ask_depth(self.config.depth_levels)
        if total_depth > 1_000_000:  # $1M in depth
            confidence += 0.2

        confidence = min(confidence, 1.0)

        return imbalance, confidence

    def generate_signal(
        self,
        symbol: str,
        current_position_side: str | None = None,
    ) -> WhaleSignal:
        """
        Generate whale detection signal.

        Args:
            symbol: Trading pair
            current_position_side: "LONG", "SHORT", or None

        Returns:
            WhaleSignal with score and alert status
        """
        if not self.config.enabled:
            return WhaleSignal(
                whale_score=0.0,
                whale_confidence=0.0,
                whale_alert_opposite=False,
                pressure=WhalePressure.NEUTRAL,
                large_trades_detected=0,
                order_book_imbalance=0.0,
            )

        # Analyze trades
        trade_pressure, large_trades, trade_confidence = self.analyze_trades(symbol)

        # Analyze order book
        ob_imbalance, ob_confidence = self.analyze_order_book(symbol)

        # Combine signals
        # Trade analysis weighted more heavily as it shows actual execution
        combined_score = trade_pressure * 0.6 + ob_imbalance * 0.4
        combined_confidence = trade_confidence * 0.6 + ob_confidence * 0.4

        # Clamp to [-1, 1]
        whale_score = max(-1.0, min(1.0, combined_score))

        # Determine pressure direction
        if whale_score > 0.2:
            pressure = WhalePressure.BULLISH
        elif whale_score < -0.2:
            pressure = WhalePressure.BEARISH
        else:
            pressure = WhalePressure.NEUTRAL

        # Check for opposing whale alert
        whale_alert_opposite = False
        recommended_action = None

        if current_position_side and abs(whale_score) >= self.config.whale_alert_threshold:
            # Check if whale activity opposes our position
            if current_position_side == "LONG" and whale_score < -self.config.whale_alert_threshold:
                whale_alert_opposite = True
                recommended_action = self.config.defensive_action
                logger.warning(
                    f"Whale alert: Strong bearish whale activity opposing LONG position on {symbol}",
                    extra={
                        "symbol": symbol,
                        "whale_score": whale_score,
                        "position_side": current_position_side,
                    },
                )
            elif current_position_side == "SHORT" and whale_score > self.config.whale_alert_threshold:
                whale_alert_opposite = True
                recommended_action = self.config.defensive_action
                logger.warning(
                    f"Whale alert: Strong bullish whale activity opposing SHORT position on {symbol}",
                    extra={
                        "symbol": symbol,
                        "whale_score": whale_score,
                        "position_side": current_position_side,
                    },
                )

        return WhaleSignal(
            whale_score=whale_score,
            whale_confidence=combined_confidence,
            whale_alert_opposite=whale_alert_opposite,
            pressure=pressure,
            large_trades_detected=large_trades,
            order_book_imbalance=ob_imbalance,
            recommended_action=recommended_action,
        )

    def should_take_defensive_action(
        self,
        signal: WhaleSignal,
    ) -> tuple[bool, WhaleDefensiveAction | None]:
        """
        Determine if defensive action should be taken.

        Returns: (should_act, action)
        """
        if signal.whale_alert_opposite:
            return True, signal.recommended_action
        return False, None

    def clear_stats(self, symbol: str | None = None) -> None:
        """Clear trade statistics."""
        if symbol:
            if symbol in self._trade_stats:
                del self._trade_stats[symbol]
            if symbol in self._last_imbalance:
                del self._last_imbalance[symbol]
        else:
            self._trade_stats.clear()
            self._last_imbalance.clear()
