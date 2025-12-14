"""
Order execution module.

Features:
- Dry-run mode (simulated execution)
- Bracket orders (entry + SL + TP)
- Idempotency protection
- Reduce-only orders for closes
- Order tracking and status updates
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import Environment, Settings, get_settings
from exchange.binance_client import BinanceClient
from risk.risk_manager import RiskManager
from strategy.fusion import TradeAction, TradeDecision
from utils.logger import TradingLogger, get_logger

logger: TradingLogger = get_logger(__name__)  # type: ignore


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    SIMULATED = "SIMULATED"  # Dry-run only


class OrderType(str, Enum):
    """Order type."""

    ENTRY = "ENTRY"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    CLOSE = "CLOSE"


@dataclass
class Order:
    """Order record."""

    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    quantity: float
    price: float | None
    stop_price: float | None
    status: OrderStatus
    exchange_order_id: int | None = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "exchange_order_id": self.exchange_order_id,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "is_dry_run": self.is_dry_run,
        }


@dataclass
class BracketOrder:
    """Bracket order (entry + SL + TP)."""

    entry_order: Order
    stop_loss_order: Order | None = None
    take_profit_order: Order | None = None

    @property
    def symbol(self) -> str:
        return self.entry_order.symbol

    @property
    def is_complete(self) -> bool:
        """Check if all orders are placed."""
        if self.entry_order.status not in (OrderStatus.FILLED, OrderStatus.SIMULATED):
            return False
        if self.stop_loss_order and self.stop_loss_order.status not in (
            OrderStatus.SUBMITTED, OrderStatus.SIMULATED
        ):
            return False
        return True


class OrderExecutor:
    """
    Executes trading orders.

    Features:
    - Dry-run mode (logs orders without executing)
    - Safety checks before execution
    - Bracket order management
    - Idempotency via order tracking
    """

    def __init__(
        self,
        client: BinanceClient,
        risk_manager: RiskManager,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.risk = risk_manager
        self.settings = settings or get_settings()
        self._pending_orders: dict[str, Order] = {}
        self._active_brackets: dict[str, BracketOrder] = {}
        self._order_history: list[Order] = []

    @property
    def is_dry_run(self) -> bool:
        """Check if running in dry-run mode."""
        return self.settings.dry_run

    @property
    def is_live_enabled(self) -> bool:
        """Check if live trading is enabled."""
        return self.settings.is_live_trading_enabled

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"bot_{uuid.uuid4().hex[:12]}"

    def _log_order(self, order: Order, event: str) -> None:
        """Log order event."""
        prefix = "[DRY-RUN] " if order.is_dry_run else ""
        logger.order_event(
            event_type=f"{prefix}{event}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price or order.stop_price,
            order_id=order.order_id,
            exchange_order_id=order.exchange_order_id,
            status=order.status.value,
            order_type=order.order_type.value,
        )

    def _simulate_fill(self, order: Order, current_price: float) -> Order:
        """Simulate order fill for dry-run mode."""
        order.status = OrderStatus.SIMULATED
        order.filled_qty = order.quantity
        order.avg_fill_price = current_price
        order.updated_at = time.time()
        order.is_dry_run = True
        return order

    def _place_real_order(self, order: Order) -> Order:
        """Place actual order on exchange."""
        try:
            if order.order_type == OrderType.ENTRY:
                result = self.client.place_market_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                )
            elif order.order_type == OrderType.STOP_LOSS:
                result = self.client.place_stop_market_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    stop_price=order.stop_price,
                    reduce_only=True,
                )
            elif order.order_type == OrderType.TAKE_PROFIT:
                result = self.client.place_take_profit_market_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    stop_price=order.stop_price,
                    reduce_only=True,
                )
            elif order.order_type == OrderType.CLOSE:
                result = self.client.place_market_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    reduce_only=True,
                )
            else:
                raise ValueError(f"Unknown order type: {order.order_type}")

            # Update order with exchange response
            order.exchange_order_id = result.get("orderId")
            order.status = OrderStatus.SUBMITTED

            if result.get("status") == "FILLED":
                order.status = OrderStatus.FILLED
                order.filled_qty = float(result.get("executedQty", 0))
                order.avg_fill_price = float(result.get("avgPrice", 0))

            order.updated_at = time.time()
            return order

        except Exception as e:
            logger.error(f"Order execution failed: {e}", extra={"order": order.to_dict()})
            order.status = OrderStatus.REJECTED
            order.updated_at = time.time()
            raise

    def execute_decision(self, decision: TradeDecision) -> BracketOrder | None:
        """
        Execute a trading decision.

        Args:
            decision: Trade decision from fusion strategy

        Returns:
            BracketOrder if executed, None if skipped
        """
        symbol = decision.symbol

        # Check if decision is valid
        if not decision.is_valid:
            logger.info(
                f"Skipping invalid decision for {symbol}: {decision.rejection_reason}",
                extra={"symbol": symbol, "reason": decision.rejection_reason},
            )
            return None

        # Skip HOLD and FLAT actions for new positions
        if decision.action in (TradeAction.HOLD, TradeAction.FLAT):
            return None

        # Check risk limits
        risk_check = self.risk.can_open_position(symbol)
        if not risk_check.is_allowed:
            logger.info(
                f"Risk check failed for {symbol}: {risk_check.reason}",
                extra={"symbol": symbol, "reason": risk_check.reason},
            )
            return None

        # Get current price
        current_price = self.client.get_ticker_price(symbol)

        # Determine side
        side = "BUY" if decision.action == TradeAction.LONG else "SELL"
        opposite_side = "SELL" if side == "BUY" else "BUY"

        # Calculate stop-loss and take-profit prices
        sl_price = self.risk.calculate_stop_loss_price(
            side="LONG" if side == "BUY" else "SHORT",
            entry_price=current_price,
            stop_loss_pct=decision.stop_loss_pct,
        )

        tp_price = None
        if decision.take_profit_pct:
            tp_price = self.risk.calculate_take_profit_price(
                side="LONG" if side == "BUY" else "SHORT",
                entry_price=current_price,
                take_profit_pct=decision.take_profit_pct,
            )

        # Calculate position size
        quantity, notional = self.risk.calculate_position_size(
            symbol=symbol,
            entry_price=current_price,
            stop_loss_price=sl_price,
            leverage=decision.suggested_leverage,
        )

        # Validate order
        validation = self.risk.validate_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=current_price,
            stop_loss_price=sl_price,
            leverage=decision.suggested_leverage,
        )

        if not validation.is_allowed:
            logger.warning(
                f"Order validation failed: {validation.reason}",
                extra={"symbol": symbol, "reason": validation.reason},
            )
            return None

        # Initialize symbol (set leverage and margin type)
        if not self.is_dry_run:
            self.client.initialize_symbol(symbol, decision.suggested_leverage)

        # Create entry order
        entry_order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.ENTRY,
            quantity=quantity,
            price=None,  # Market order
            stop_price=None,
            status=OrderStatus.PENDING,
        )

        # Execute or simulate entry
        if self.is_dry_run:
            entry_order = self._simulate_fill(entry_order, current_price)
            self._log_order(entry_order, "SIMULATED_ENTRY")
        else:
            entry_order = self._place_real_order(entry_order)
            self._log_order(entry_order, "ENTRY_PLACED")

        # Get actual fill price
        fill_price = entry_order.avg_fill_price or current_price

        # Create stop-loss order
        sl_order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=opposite_side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            price=None,
            stop_price=sl_price,
            status=OrderStatus.PENDING,
        )

        # Execute or simulate stop-loss
        if self.is_dry_run:
            sl_order.status = OrderStatus.SIMULATED
            sl_order.is_dry_run = True
            self._log_order(sl_order, "SIMULATED_SL")
        else:
            sl_order = self._place_real_order(sl_order)
            self._log_order(sl_order, "SL_PLACED")

        # Create take-profit order if enabled
        tp_order = None
        if tp_price:
            tp_order = Order(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=opposite_side,
                order_type=OrderType.TAKE_PROFIT,
                quantity=quantity,
                price=None,
                stop_price=tp_price,
                status=OrderStatus.PENDING,
            )

            if self.is_dry_run:
                tp_order.status = OrderStatus.SIMULATED
                tp_order.is_dry_run = True
                self._log_order(tp_order, "SIMULATED_TP")
            else:
                tp_order = self._place_real_order(tp_order)
                self._log_order(tp_order, "TP_PLACED")

        # Create bracket order
        bracket = BracketOrder(
            entry_order=entry_order,
            stop_loss_order=sl_order,
            take_profit_order=tp_order,
        )

        # Track bracket
        self._active_brackets[symbol] = bracket

        # Log summary
        mode = "DRY-RUN" if self.is_dry_run else "LIVE"
        logger.info(
            f"[{mode}] Bracket order placed for {symbol}",
            extra={
                "mode": mode,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": fill_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "leverage": decision.suggested_leverage,
            },
        )

        return bracket

    def close_position(self, symbol: str, reason: str = "Signal") -> Order | None:
        """
        Close an existing position.

        Args:
            symbol: Trading pair
            reason: Reason for closing

        Returns:
            Close order if executed, None if no position
        """
        # Get current position
        position = self.client.get_position(symbol)
        if not position:
            return None

        position_amt = float(position.get("positionAmt", 0))
        if position_amt == 0:
            return None

        # Determine close side
        side = "SELL" if position_amt > 0 else "BUY"
        quantity = abs(position_amt)

        # Cancel existing SL/TP orders
        if symbol in self._active_brackets:
            bracket = self._active_brackets[symbol]
            if not self.is_dry_run:
                try:
                    self.client.cancel_all_orders(symbol)
                except Exception as e:
                    logger.warning(f"Failed to cancel orders: {e}")
            del self._active_brackets[symbol]

        # Create close order
        close_order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.CLOSE,
            quantity=quantity,
            price=None,
            stop_price=None,
            status=OrderStatus.PENDING,
        )

        # Execute or simulate
        if self.is_dry_run:
            current_price = self.client.get_ticker_price(symbol)
            close_order = self._simulate_fill(close_order, current_price)
            self._log_order(close_order, f"SIMULATED_CLOSE ({reason})")
        else:
            close_order = self._place_real_order(close_order)
            self._log_order(close_order, f"CLOSE_PLACED ({reason})")

        # Calculate PnL
        entry_price = float(position.get("entryPrice", 0))
        exit_price = close_order.avg_fill_price or self.client.get_ticker_price(symbol)

        if position_amt > 0:  # Long position
            pnl = (exit_price - entry_price) * quantity
        else:  # Short position
            pnl = (entry_price - exit_price) * quantity

        is_win = pnl > 0

        # Record trade result
        self.risk.record_trade_result(pnl, is_win)

        # Set cooldown if loss
        if not is_win:
            self.risk.set_cooldown(symbol)

        logger.info(
            f"Position closed for {symbol}: PnL=${pnl:,.2f} ({'WIN' if is_win else 'LOSS'})",
            extra={
                "symbol": symbol,
                "pnl": pnl,
                "is_win": is_win,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "reason": reason,
            },
        )

        return close_order

    def close_all_positions(self, reason: str = "Manual") -> list[Order]:
        """Close all open positions."""
        positions = self.client.get_all_positions()
        orders = []

        for position in positions:
            symbol = position.get("symbol")
            if symbol:
                order = self.close_position(symbol, reason)
                if order:
                    orders.append(order)

        return orders

    def get_active_brackets(self) -> dict[str, BracketOrder]:
        """Get all active bracket orders."""
        return self._active_brackets.copy()

    def has_active_bracket(self, symbol: str) -> bool:
        """Check if symbol has an active bracket order."""
        return symbol in self._active_brackets

    def cancel_bracket(self, symbol: str) -> bool:
        """Cancel bracket orders for a symbol."""
        if symbol not in self._active_brackets:
            return False

        if not self.is_dry_run:
            try:
                self.client.cancel_all_orders(symbol)
            except Exception as e:
                logger.error(f"Failed to cancel orders for {symbol}: {e}")
                return False

        del self._active_brackets[symbol]
        logger.info(f"Bracket orders cancelled for {symbol}")
        return True
