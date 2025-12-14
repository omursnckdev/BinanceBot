"""
Market data management module.

Handles:
- Candle/OHLCV data
- Aggregated trades
- Order book snapshots
- Data freshness validation
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import Settings, get_settings
from exchange.gate_client import GateClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Candle:
    """OHLCV candlestick data."""

    timestamp: int  # Open time in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_volume: float
    taker_buy_quote_volume: float

    @classmethod
    def from_gate(cls, data: list) -> "Candle":
        """Create from Gate.io kline data."""
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            close_time=int(data[6]),
            quote_volume=float(data[7]),
            trades=int(data[8]),
            taker_buy_volume=float(data[9]),
            taker_buy_quote_volume=float(data[10]),
        )

    @property
    def datetime(self) -> datetime:
        """Get candle open time as datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)

    @property
    def age_seconds(self) -> float:
        """Get age of candle in seconds."""
        return time.time() - (self.close_time / 1000)

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    @property
    def body_size(self) -> float:
        """Get candle body size."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Get candle range (high - low)."""
        return self.high - self.low


@dataclass
class Trade:
    """Aggregated trade data."""

    trade_id: int
    price: float
    quantity: float
    timestamp: int
    is_buyer_maker: bool

    @classmethod
    def from_gate(cls, data: dict[str, Any]) -> "Trade":
        """Create from Gate.io trade data."""
        return cls(
            trade_id=data["a"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp=data["T"],
            is_buyer_maker=data["m"],
        )

    @property
    def notional(self) -> float:
        """Get trade notional value."""
        return self.price * self.quantity

    @property
    def is_buy(self) -> bool:
        """Check if trade is a buy (taker is buyer)."""
        return not self.is_buyer_maker


@dataclass
class OrderBookSnapshot:
    """Order book depth snapshot."""

    timestamp: int
    bids: list[tuple[float, float]]  # [(price, quantity), ...]
    asks: list[tuple[float, float]]  # [(price, quantity), ...]
    last_update_id: int

    @classmethod
    def from_gate(cls, data: dict[str, Any]) -> "OrderBookSnapshot":
        """Create from Gate.io depth data."""
        return cls(
            timestamp=int(time.time() * 1000),
            bids=[(float(b[0]), float(b[1])) for b in data.get("bids", [])],
            asks=[(float(a[0]), float(a[1])) for a in data.get("asks", [])],
            last_update_id=data.get("lastUpdateId", 0),
        )

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Get spread as percentage of mid price."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0

    def bid_depth(self, levels: int | None = None) -> float:
        """Get total bid depth in quote currency."""
        bids = self.bids[:levels] if levels else self.bids
        return sum(price * qty for price, qty in bids)

    def ask_depth(self, levels: int | None = None) -> float:
        """Get total ask depth in quote currency."""
        asks = self.asks[:levels] if levels else self.asks
        return sum(price * qty for price, qty in asks)

    def imbalance(self, levels: int | None = None) -> float:
        """
        Calculate order book imbalance.

        Returns value between -1 (all asks) and +1 (all bids).
        """
        bid_total = self.bid_depth(levels)
        ask_total = self.ask_depth(levels)
        total = bid_total + ask_total

        if total == 0:
            return 0.0

        return (bid_total - ask_total) / total


@dataclass
class SymbolData:
    """Container for all market data for a symbol."""

    symbol: str
    candles: dict[str, deque[Candle]] = field(default_factory=dict)  # timeframe -> candles
    trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=10000))
    order_book: OrderBookSnapshot | None = None
    last_price: float = 0.0
    last_update: float = 0.0

    def add_candles(self, timeframe: str, candles: list[Candle]) -> None:
        """Add candles for a timeframe."""
        if timeframe not in self.candles:
            self.candles[timeframe] = deque(maxlen=500)

        for candle in candles:
            self.candles[timeframe].append(candle)

        self.last_update = time.time()

    def get_candles_df(self, timeframe: str) -> pd.DataFrame | None:
        """Get candles as pandas DataFrame."""
        if timeframe not in self.candles or not self.candles[timeframe]:
            return None

        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in self.candles[timeframe]
        ]

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def add_trades(self, trades: list[Trade]) -> None:
        """Add trades."""
        for trade in trades:
            self.trades.append(trade)
            self.last_price = trade.price

        self.last_update = time.time()

    def get_trade_notionals(self, window_minutes: int = 60) -> list[float]:
        """Get trade notionals within time window."""
        cutoff = int((time.time() - window_minutes * 60) * 1000)
        return [t.notional for t in self.trades if t.timestamp >= cutoff]

    def is_data_fresh(self, max_age_seconds: int) -> bool:
        """Check if data is fresh enough for trading."""
        return time.time() - self.last_update <= max_age_seconds


class MarketDataManager:
    """
    Manages market data for all trading symbols.

    Features:
    - Fetches and caches candles, trades, order books
    - Validates data freshness
    - Provides DataFrames for indicator calculation
    """

    def __init__(
        self,
        client: GateClient,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self._data: dict[str, SymbolData] = {}

    def get_symbol_data(self, symbol: str) -> SymbolData:
        """Get or create symbol data container."""
        if symbol not in self._data:
            self._data[symbol] = SymbolData(symbol=symbol)
        return self._data[symbol]

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[Candle]:
        """Fetch and store candles for a symbol."""
        try:
            raw_klines = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
            )

            candles = [Candle.from_gate(k) for k in raw_klines]
            self.get_symbol_data(symbol).add_candles(timeframe, candles)

            logger.debug(
                f"Fetched {len(candles)} {timeframe} candles for {symbol}",
                extra={"symbol": symbol, "timeframe": timeframe, "count": len(candles)},
            )

            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def fetch_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> list[Trade]:
        """Fetch and store aggregated trades."""
        try:
            raw_trades = self.client.get_agg_trades(symbol=symbol, limit=limit)
            trades = [Trade.from_gate(t) for t in raw_trades]
            self.get_symbol_data(symbol).add_trades(trades)

            logger.debug(
                f"Fetched {len(trades)} trades for {symbol}",
                extra={"symbol": symbol, "count": len(trades)},
            )

            return trades

        except Exception as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return []

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> OrderBookSnapshot | None:
        """Fetch and store order book snapshot."""
        try:
            raw_depth = self.client.get_order_book(symbol=symbol, limit=limit)
            snapshot = OrderBookSnapshot.from_gate(raw_depth)
            self.get_symbol_data(symbol).order_book = snapshot
            self.get_symbol_data(symbol).last_update = time.time()

            logger.debug(
                f"Fetched order book for {symbol}: spread={snapshot.spread_pct:.4f}%",
                extra={"symbol": symbol, "spread_pct": snapshot.spread_pct},
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return None

    def fetch_all(
        self,
        symbol: str,
        timeframes: list[str] | None = None,
    ) -> bool:
        """
        Fetch all market data for a symbol.

        Returns True if all data was fetched successfully.
        """
        if timeframes is None:
            timeframes = [
                self.settings.indicators.primary_timeframe,
                *self.settings.indicators.confirmation_timeframes,
            ]

        success = True

        # Fetch candles for each timeframe
        for tf in timeframes:
            if not self.fetch_candles(symbol, tf):
                success = False

        # Fetch trades and order book
        if not self.fetch_trades(symbol):
            success = False
        if not self.fetch_order_book(symbol):
            success = False

        return success

    def get_candles_df(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Get candles DataFrame for a symbol and timeframe."""
        return self.get_symbol_data(symbol).get_candles_df(timeframe)

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        data = self.get_symbol_data(symbol)

        # Try order book first
        if data.order_book:
            return data.order_book.mid_price

        # Fall back to last trade price
        if data.last_price > 0:
            return data.last_price

        # Fetch from API
        return self.client.get_ticker_price(symbol)

    def is_data_valid(self, symbol: str) -> tuple[bool, str]:
        """
        Check if data is valid for trading.

        Returns (is_valid, reason)
        """
        data = self.get_symbol_data(symbol)

        # Check data freshness
        if not data.is_data_fresh(self.settings.max_candle_age_seconds):
            return False, f"Data stale: last update {time.time() - data.last_update:.1f}s ago"

        # Check we have candles
        primary_tf = self.settings.indicators.primary_timeframe
        if primary_tf not in data.candles or len(data.candles[primary_tf]) < 50:
            return False, f"Insufficient candles for {primary_tf}"

        # Check order book
        if not data.order_book:
            return False, "No order book data"

        # Check spread
        if data.order_book.spread_pct > self.settings.trading.max_spread_pct:
            return False, f"Spread too high: {data.order_book.spread_pct:.4f}%"

        return True, "OK"

    def get_market_conditions(self, symbol: str) -> dict[str, Any]:
        """Get current market conditions for a symbol."""
        data = self.get_symbol_data(symbol)

        conditions: dict[str, Any] = {
            "symbol": symbol,
            "price": self.get_latest_price(symbol),
            "data_age_seconds": time.time() - data.last_update,
        }

        if data.order_book:
            conditions.update({
                "spread_pct": data.order_book.spread_pct,
                "bid_depth": data.order_book.bid_depth(20),
                "ask_depth": data.order_book.ask_depth(20),
                "imbalance": data.order_book.imbalance(20),
            })

        return conditions

    def clear_symbol(self, symbol: str) -> None:
        """Clear cached data for a symbol."""
        if symbol in self._data:
            del self._data[symbol]

    def clear_all(self) -> None:
        """Clear all cached data."""
        self._data.clear()
