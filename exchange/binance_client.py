"""
Binance USDT-M Futures client wrapper.

Supports both testnet and mainnet with proper endpoint switching.
Handles symbol filters, precision, leverage, and margin type.
Uses binance-futures-connector library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from binance.um_futures import UMFutures

from config import Environment, MarginType, Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


# Endpoint URLs
MAINNET_BASE_URL = "https://fapi.binance.com"
TESTNET_BASE_URL = "https://testnet.binancefuture.com"


@dataclass
class SymbolInfo:
    """Symbol trading rules and filters."""

    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    max_leverage: int

    def round_price(self, price: float | Decimal) -> Decimal:
        """Round price to valid tick size."""
        price_dec = Decimal(str(price))
        return (price_dec / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def round_quantity(self, qty: float | Decimal) -> Decimal:
        """Round quantity to valid step size."""
        qty_dec = Decimal(str(qty))
        rounded = (qty_dec / self.step_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.step_size
        return max(self.min_qty, min(rounded, self.max_qty))

    def validate_order(self, price: float, qty: float) -> tuple[bool, str]:
        """Validate order parameters against symbol filters."""
        notional = Decimal(str(price)) * Decimal(str(qty))

        if Decimal(str(qty)) < self.min_qty:
            return False, f"Quantity {qty} below minimum {self.min_qty}"

        if Decimal(str(qty)) > self.max_qty:
            return False, f"Quantity {qty} above maximum {self.max_qty}"

        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"

        return True, ""


class BinanceClient:
    """
    Binance Futures client with testnet/mainnet support.

    Features:
    - Automatic endpoint switching based on environment
    - Time sync and drift detection
    - Symbol info caching
    - Precision handling for orders
    - Leverage and margin type management
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: UMFutures | None = None
        self._symbol_info_cache: dict[str, SymbolInfo] = {}
        self._last_time_sync: float = 0
        self._server_time_offset: int = 0

    @property
    def client(self) -> UMFutures:
        """Get or create the Binance client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> UMFutures:
        """Create the Binance UMFutures client with correct endpoints."""
        base_url = (
            TESTNET_BASE_URL
            if self.settings.env == Environment.TESTNET
            else MAINNET_BASE_URL
        )

        logger.info(
            f"Creating Binance client for {self.settings.env.value}",
            extra={"base_url": base_url},
        )

        return UMFutures(
            key=self.settings.binance_api_key,
            secret=self.settings.binance_api_secret,
            base_url=base_url,
        )

    def sync_time(self) -> int:
        """
        Sync local time with server time.

        Returns:
            Time offset in milliseconds (server - local)
        """
        try:
            local_time = int(time.time() * 1000)
            server_time = self.client.time()["serverTime"]
            self._server_time_offset = server_time - local_time
            self._last_time_sync = time.time()

            if abs(self._server_time_offset) > self.settings.max_time_sync_drift_ms:
                logger.warning(
                    f"Large time drift detected: {self._server_time_offset}ms",
                    extra={"drift_ms": self._server_time_offset},
                )

            return self._server_time_offset
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")
            raise

    def check_time_sync(self) -> bool:
        """Check if time sync is within acceptable drift."""
        if time.time() - self._last_time_sync > 60:  # Re-sync every minute
            self.sync_time()

        return abs(self._server_time_offset) <= self.settings.max_time_sync_drift_ms

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get symbol trading rules and filters.

        Caches results to avoid repeated API calls.
        """
        symbol = symbol.upper()

        if symbol not in self._symbol_info_cache:
            self._load_exchange_info()

        if symbol not in self._symbol_info_cache:
            raise ValueError(f"Unknown symbol: {symbol}")

        return self._symbol_info_cache[symbol]

    def _load_exchange_info(self) -> None:
        """Load and cache exchange info for all symbols."""
        try:
            exchange_info = self.client.exchange_info()

            for sym_info in exchange_info.get("symbols", []):
                if sym_info.get("contractType") != "PERPETUAL":
                    continue

                symbol = sym_info["symbol"]

                # Extract filters
                filters = {f["filterType"]: f for f in sym_info.get("filters", [])}

                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", {})
                min_notional_filter = filters.get("MIN_NOTIONAL", {})

                self._symbol_info_cache[symbol] = SymbolInfo(
                    symbol=symbol,
                    base_asset=sym_info.get("baseAsset", ""),
                    quote_asset=sym_info.get("quoteAsset", ""),
                    price_precision=sym_info.get("pricePrecision", 8),
                    quantity_precision=sym_info.get("quantityPrecision", 8),
                    tick_size=Decimal(str(price_filter.get("tickSize", "0.01"))),
                    step_size=Decimal(str(lot_filter.get("stepSize", "0.001"))),
                    min_qty=Decimal(str(lot_filter.get("minQty", "0.001"))),
                    max_qty=Decimal(str(lot_filter.get("maxQty", "1000000"))),
                    min_notional=Decimal(str(min_notional_filter.get("notional", "5"))),
                    max_leverage=125,  # Will be updated by leverage brackets
                )

            logger.info(
                f"Loaded exchange info for {len(self._symbol_info_cache)} symbols"
            )

        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")
            raise

    def get_account_balance(self) -> dict[str, Any]:
        """Get account balance information."""
        try:
            return self.client.account()
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise

    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        try:
            account = self.get_account_balance()
            for asset in account.get("assets", []):
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableBalance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get USDT balance: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        """Set leverage for a symbol."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            leverage = min(leverage, symbol_info.max_leverage)

            result = self.client.change_leverage(symbol=symbol, leverage=leverage)
            logger.info(
                f"Set leverage for {symbol} to {leverage}x",
                extra={"symbol": symbol, "leverage": leverage},
            )
            return result
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            raise

    def set_margin_type(self, symbol: str, margin_type: MarginType) -> dict[str, Any] | None:
        """Set margin type for a symbol."""
        try:
            result = self.client.change_margin_type(
                symbol=symbol, marginType=margin_type.value
            )
            logger.info(
                f"Set margin type for {symbol} to {margin_type.value}",
                extra={"symbol": symbol, "margin_type": margin_type.value},
            )
            return result
        except Exception as e:
            # Ignore error if margin type is already set
            if "No need to change margin type" in str(e):
                return None
            logger.error(f"Failed to set margin type for {symbol}: {e}")
            raise

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Get current position for a symbol."""
        try:
            positions = self.client.get_position_risk(symbol=symbol)
            for pos in positions:
                if pos.get("symbol") == symbol and float(pos.get("positionAmt", 0)) != 0:
                    return pos
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        try:
            positions = self.client.get_position_risk()
            return [p for p in positions if float(p.get("positionAmt", 0)) != 0]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open orders for a symbol or all symbols."""
        try:
            if symbol:
                return self.client.get_orders(symbol=symbol)
            return self.client.get_orders()
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading pair symbol
            side: BUY or SELL
            quantity: Order quantity
            reduce_only: If True, only reduces position
        """
        symbol_info = self.get_symbol_info(symbol)
        rounded_qty = float(symbol_info.round_quantity(quantity))

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": rounded_qty,
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(
            f"Placing market order: {symbol} {side} {rounded_qty}",
            extra=params,
        )

        try:
            return self.client.new_order(**params)
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """Place a limit order."""
        symbol_info = self.get_symbol_info(symbol)
        rounded_qty = float(symbol_info.round_quantity(quantity))
        rounded_price = float(symbol_info.round_price(price))

        # Validate order
        valid, error_msg = symbol_info.validate_order(rounded_price, rounded_qty)
        if not valid:
            raise ValueError(error_msg)

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": rounded_qty,
            "price": rounded_price,
            "timeInForce": time_in_force,
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(
            f"Placing limit order: {symbol} {side} {rounded_qty} @ {rounded_price}",
            extra=params,
        )

        try:
            return self.client.new_order(**params)
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise

    def place_stop_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> dict[str, Any]:
        """Place a stop market order (for stop-loss)."""
        symbol_info = self.get_symbol_info(symbol)
        rounded_qty = float(symbol_info.round_quantity(quantity))
        rounded_stop = float(symbol_info.round_price(stop_price))

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "quantity": rounded_qty,
            "stopPrice": rounded_stop,
            "reduceOnly": "true" if reduce_only else "false",
        }

        logger.info(
            f"Placing stop market order: {symbol} {side} {rounded_qty} @ stop {rounded_stop}",
            extra=params,
        )

        try:
            return self.client.new_order(**params)
        except Exception as e:
            logger.error(f"Failed to place stop market order: {e}")
            raise

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> dict[str, Any]:
        """Place a take profit market order."""
        symbol_info = self.get_symbol_info(symbol)
        rounded_qty = float(symbol_info.round_quantity(quantity))
        rounded_stop = float(symbol_info.round_price(stop_price))

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "quantity": rounded_qty,
            "stopPrice": rounded_stop,
            "reduceOnly": "true" if reduce_only else "false",
        }

        logger.info(
            f"Placing take profit order: {symbol} {side} {rounded_qty} @ stop {rounded_stop}",
            extra=params,
        )

        try:
            return self.client.new_order(**params)
        except Exception as e:
            logger.error(f"Failed to place take profit order: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Cancel a specific order."""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def cancel_all_orders(self, symbol: str) -> dict[str, Any]:
        """Cancel all open orders for a symbol."""
        try:
            return self.client.cancel_open_orders(symbol=symbol)
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            raise

    def close_position(self, symbol: str) -> dict[str, Any] | None:
        """Close an open position at market price."""
        position = self.get_position(symbol)
        if not position:
            logger.info(f"No open position to close for {symbol}")
            return None

        position_amt = float(position.get("positionAmt", 0))
        if position_amt == 0:
            return None

        # Determine side to close (opposite of position)
        side = "SELL" if position_amt > 0 else "BUY"
        quantity = abs(position_amt)

        logger.info(
            f"Closing position for {symbol}: {side} {quantity}",
            extra={"symbol": symbol, "position_amt": position_amt},
        )

        return self.place_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=True,
        )

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list]:
        """
        Get kline/candlestick data.

        Returns list of [open_time, open, high, low, close, volume, close_time, ...]
        """
        try:
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time

            return self.client.klines(**params)
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise

    def get_agg_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated trades."""
        try:
            params: dict[str, Any] = {
                "symbol": symbol,
                "limit": limit,
            }
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time

            return self.client.agg_trades(**params)
        except Exception as e:
            logger.error(f"Failed to get agg trades for {symbol}: {e}")
            raise

    def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book depth."""
        try:
            return self.client.depth(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise

    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            result = self.client.ticker_price(symbol=symbol)
            return float(result.get("price", 0))
        except Exception as e:
            logger.error(f"Failed to get ticker price for {symbol}: {e}")
            raise

    def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Get funding rate for a symbol."""
        try:
            result = self.client.funding_rate(symbol=symbol, limit=1)
            if result:
                return result[0]
            return {}
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            return {}

    def get_open_interest(self, symbol: str) -> float:
        """Get open interest for a symbol."""
        try:
            result = self.client.open_interest(symbol=symbol)
            return float(result.get("openInterest", 0))
        except Exception as e:
            logger.error(f"Failed to get open interest for {symbol}: {e}")
            return 0.0

    def initialize_symbol(self, symbol: str, leverage: int) -> None:
        """
        Initialize a symbol for trading.

        Sets leverage and margin type.
        """
        logger.info(
            f"Initializing symbol {symbol} with leverage {leverage}x",
            extra={"symbol": symbol, "leverage": leverage},
        )

        # Set margin type first
        try:
            self.set_margin_type(symbol, self.settings.margin_type)
        except Exception as e:
            logger.warning(f"Could not set margin type for {symbol}: {e}")

        # Set leverage
        self.set_leverage(symbol, leverage)
