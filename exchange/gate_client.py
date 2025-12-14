"""
Gate.io USDT-margined futures client wrapper.

Provides a Binance-like interface for the rest of the bot so that only the
exchange layer needs to change when switching providers. The implementation is
intentionally conservative and focuses on the methods the bot actually uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from gate_api import ApiClient, Configuration, FuturesApi, WalletApi

from config import MarginType, Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


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
        price_dec = Decimal(str(price))
        return (price_dec / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def round_quantity(self, qty: float | Decimal) -> Decimal:
        qty_dec = Decimal(str(qty))
        rounded = (qty_dec / self.step_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.step_size
        return max(self.min_qty, min(rounded, self.max_qty))

    def validate_order(self, price: float, qty: float) -> tuple[bool, str]:
        notional = Decimal(str(price)) * Decimal(str(qty))

        if Decimal(str(qty)) < self.min_qty:
            return False, f"Quantity {qty} below minimum {self.min_qty}"

        if Decimal(str(qty)) > self.max_qty:
            return False, f"Quantity {qty} above maximum {self.max_qty}"

        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"

        return True, ""


class GateClient:
    """Gate.io Futures client with a Binance-compatible surface."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._symbol_info_cache: dict[str, SymbolInfo] = {}
        self._last_time_sync: float = 0
        self._server_time_offset: int = 0
        self._api_client = self._create_client()
        self._futures_api = FuturesApi(self._api_client)
        self._wallet_api = WalletApi(self._api_client)

    def _create_client(self) -> ApiClient:
        cfg = Configuration()
        cfg.key = self.settings.gate_api_key
        cfg.secret = self.settings.gate_api_secret
        cfg.host = "https://api.gateio.ws/api/v4"

        logger.info(
            "Creating Gate.io client",
            extra={"environment": self.settings.env.value},
        )

        return ApiClient(cfg)

    def sync_time(self) -> int:
        """Gate.io does not expose a time-sync endpoint; assume local clock."""
        self._last_time_sync = time.time()
        self._server_time_offset = 0
        return self._server_time_offset

    def check_time_sync(self) -> bool:
        if time.time() - self._last_time_sync > 60:
            self.sync_time()
        return True

    # === Market data ===
    def get_klines(self, symbol: str, interval: str, limit: int) -> list[list[Any]]:
        candles = self._futures_api.list_futures_candlesticks(
            settle="usdt", contract=symbol.upper(), interval=interval, limit=limit
        )
        return [
            [
                int(candle.t * 1000),
                float(candle.o),
                float(candle.h),
                float(candle.l),
                float(candle.c),
                float(candle.v),
                int(candle.t * 1000),
                float(candle.v),
                0,
                float(candle.v),
                float(candle.v),
            ]
            for candle in candles
        ]

    def get_agg_trades(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        trades = self._futures_api.list_futures_trades(
            settle="usdt", contract=symbol.upper(), limit=limit
        )
        return [
            {
                "a": int(trade.id),
                "p": float(trade.price),
                "q": float(trade.size),
                "T": int(trade.create_time_ms),
                "m": trade.side.lower() == "sell",
            }
            for trade in trades
        ]

    def get_order_book(self, symbol: str, limit: int) -> dict[str, Any]:
        depth = self._futures_api.list_futures_order_book(
            settle="usdt", contract=symbol.upper(), limit=limit
        )
        return {
            "bids": depth.bids,
            "asks": depth.asks,
            "lastUpdateId": int(time.time() * 1000),
        }

    def get_ticker_price(self, symbol: str) -> float:
        tickers = self._futures_api.list_futures_tickers(
            settle="usdt", contract=symbol.upper()
        )
        if not tickers:
            raise ValueError(f"No ticker data for {symbol}")
        return float(tickers[0].last)

    # === Account ===
    def get_usdt_balance(self) -> float:
        accounts = self._wallet_api.list_futures_accounts(settle="usdt")
        if not accounts:
            return 0.0
        return float(accounts[0].available)

    def get_all_positions(self) -> list[dict[str, Any]]:
        positions = self._futures_api.list_positions(settle="usdt")
        return [
            {
                "symbol": pos.contract,
                "positionAmt": float(pos.size),
                "entryPrice": float(pos.entry_price or 0),
                "unrealizedProfit": float(pos.unrealised_pnl or 0),
            }
            for pos in positions
        ]

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        position = self._futures_api.get_position(settle="usdt", contract=symbol.upper())
        if not position:
            return None
        return {
            "symbol": position.contract,
            "positionAmt": float(position.size),
            "entryPrice": float(position.entry_price or 0),
            "unrealizedProfit": float(position.unrealised_pnl or 0),
        }

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._futures_api.update_position_leverage(
            settle="usdt", contract=symbol.upper(), leverage=leverage
        )

    def set_margin_type(self, symbol: str, margin_type: MarginType) -> None:
        cross = margin_type == MarginType.CROSS
        self._futures_api.update_position_margin_type(
            settle="usdt", contract=symbol.upper(), cross_leverage_limit=0 if cross else None
        )

    # === Trading ===
    def place_market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> dict[str, Any]:
        order = self._futures_api.create_futures_order(
            settle="usdt",
            contract=symbol.upper(),
            side=side.lower(),
            size=quantity,
            type="market",
            reduce_only=reduce_only,
        )
        return {"orderId": order.id, "avg_fill_price": float(order.fill_price or 0)}

    def place_stop_market_order(
        self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = False
    ) -> dict[str, Any]:
        order = self._futures_api.create_futures_order(
            settle="usdt",
            contract=symbol.upper(),
            side=side.lower(),
            size=quantity,
            type="market",
            reduce_only=reduce_only,
            trigger=stop_price,
        )
        return {"orderId": order.id, "avg_fill_price": float(order.fill_price or 0)}

    def place_take_profit_market_order(
        self, symbol: str, side: str, quantity: float, stop_price: float, reduce_only: bool = False
    ) -> dict[str, Any]:
        return self.place_stop_market_order(symbol, side, quantity, stop_price, reduce_only)

    def cancel_all_orders(self, symbol: str) -> None:
        self._futures_api.cancel_futures_orders(settle="usdt", contract=symbol.upper())

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        orders = self._futures_api.list_futures_orders(
            settle="usdt", contract=symbol.upper(), status="open"
        )
        return [
            {
                "orderId": order.id,
                "type": order.type,
                "stopPrice": float(order.trigger or 0),
                "origQty": float(order.size),
                "side": order.side,
            }
            for order in orders
        ]

    # === Metadata ===
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        symbol = symbol.upper()
        if symbol not in self._symbol_info_cache:
            self._load_exchange_info()

        if symbol not in self._symbol_info_cache:
            raise ValueError(f"Unknown symbol: {symbol}")

        return self._symbol_info_cache[symbol]

    def _load_exchange_info(self) -> None:
        contracts = self._futures_api.list_futures_contracts(settle="usdt")
        for contract in contracts:
            symbol = contract.name.upper()
            tick_size = Decimal(str(contract.order_price_round or "0.01"))
            step_size = Decimal(str(contract.order_size_round or "0.001"))
            min_qty = Decimal(str(contract.order_size_min or "0.001"))
            max_qty = Decimal(str(contract.order_size_max or "1000"))
            min_notional = min_qty * Decimal(str(contract.mark_price or 1))

            self._symbol_info_cache[symbol] = SymbolInfo(
                symbol=symbol,
                base_asset=contract.base or "",
                quote_asset=contract.quote or "USDT",
                price_precision=int(contract.mark_price_round or 4),
                quantity_precision=int(contract.order_size_round or 3),
                tick_size=tick_size,
                step_size=step_size,
                min_qty=min_qty,
                max_qty=max_qty,
                min_notional=min_notional,
                max_leverage=int(contract.leverage_max or 50),
            )

        logger.info(
            f"Loaded exchange info for {len(self._symbol_info_cache)} Gate.io futures contracts"
        )


__all__ = ["GateClient", "SymbolInfo"]
