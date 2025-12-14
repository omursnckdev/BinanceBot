"""
Risk management module.

Features:
- Position sizing based on risk per trade
- Kill-switches (daily loss, consecutive losses, exposure)
- Stop-loss and take-profit calculation
- Cooldown management after losses
- Maximum position limits
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from config import RiskConfig, Settings, get_settings
from exchange.gate_client import GateClient
from utils.logger import TradingLogger, get_logger

logger: TradingLogger = get_logger(__name__)  # type: ignore


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str  # YYYY-MM-DD
    starting_balance: float
    current_balance: float
    realized_pnl: float
    trades_count: int
    wins: int
    losses: int
    consecutive_losses: int
    max_drawdown_pct: float

    @property
    def daily_pnl_pct(self) -> float:
        """Daily PnL as percentage."""
        if self.starting_balance == 0:
            return 0.0
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return (self.wins / total) * 100


@dataclass
class PositionRisk:
    """Risk parameters for a position."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    leverage: int
    stop_loss_price: float
    take_profit_price: float | None
    notional_value: float
    risk_amount: float  # Potential loss at stop-loss
    risk_pct: float  # Risk as percentage of account


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    is_allowed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Manages trading risk.

    Features:
    - Position sizing based on account size and risk per trade
    - Daily loss limit kill-switch
    - Consecutive loss kill-switch
    - Maximum exposure limits
    - Cooldown after stop-outs
    """

    def __init__(
        self,
        client: GateClient,
        config: RiskConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self.config = config or self.settings.risk
        self._daily_stats: DailyStats | None = None
        self._cooldown_until: dict[str, float] = {}
        self._open_positions: dict[str, PositionRisk] = {}

    def _get_today_str(self) -> str:
        """Get today's date string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_daily_stats(self) -> DailyStats:
        """Get or create today's daily statistics."""
        today = self._get_today_str()

        if self._daily_stats is None or self._daily_stats.date != today:
            # Reset for new day
            balance = self.client.get_usdt_balance()
            self._daily_stats = DailyStats(
                date=today,
                starting_balance=balance,
                current_balance=balance,
                realized_pnl=0.0,
                trades_count=0,
                wins=0,
                losses=0,
                consecutive_losses=0,
                max_drawdown_pct=0.0,
            )
            logger.info(
                f"New trading day started with balance: ${balance:,.2f}",
                extra={"balance": balance, "date": today},
            )

        return self._daily_stats

    def update_balance(self, new_balance: float) -> None:
        """Update current balance and calculate drawdown."""
        stats = self.get_daily_stats()
        stats.current_balance = new_balance

        # Update max drawdown
        if stats.starting_balance > 0:
            current_dd = ((stats.starting_balance - new_balance) / stats.starting_balance) * 100
            stats.max_drawdown_pct = max(stats.max_drawdown_pct, current_dd)

    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """Record a trade result."""
        stats = self.get_daily_stats()

        stats.trades_count += 1
        stats.realized_pnl += pnl

        if is_win:
            stats.wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losses += 1
            stats.consecutive_losses += 1

        logger.info(
            f"Trade recorded: PnL=${pnl:,.2f}, {'WIN' if is_win else 'LOSS'}",
            extra={
                "pnl": pnl,
                "is_win": is_win,
                "consecutive_losses": stats.consecutive_losses,
                "daily_pnl": stats.realized_pnl,
            },
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        leverage: int,
        risk_pct: float | None = None,
    ) -> tuple[float, float]:
        """
        Calculate position size based on risk per trade.

        Args:
            symbol: Trading pair
            entry_price: Expected entry price
            stop_loss_price: Stop-loss price
            leverage: Position leverage
            risk_pct: Risk percentage of account (uses config default if None)

        Returns:
            (quantity, notional_value)
        """
        # Get account balance
        balance = self.client.get_usdt_balance()

        # Use provided risk_pct or default
        if risk_pct is None:
            symbol_config = self.settings.get_symbol_config(symbol)
            risk_pct = symbol_config.risk_per_trade_pct

        # Risk amount in USDT
        risk_amount = balance * (risk_pct / 100)

        # Calculate stop distance percentage
        stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price

        if stop_distance_pct == 0:
            logger.warning("Stop distance is zero, using default 2%")
            stop_distance_pct = 0.02

        # Position size calculation
        # risk_amount = position_size * stop_distance_pct
        # With leverage: notional = position_size * entry_price * leverage
        position_size = risk_amount / stop_distance_pct

        # Cap by maximum position size
        max_position = min(
            self.config.max_position_size_usd,
            balance * (self.config.max_total_exposure_pct / 100),
        )
        position_size = min(position_size, max_position)

        # Calculate quantity
        quantity = position_size / entry_price

        # Round to symbol precision
        symbol_info = self.client.get_symbol_info(symbol)
        quantity = float(symbol_info.round_quantity(quantity))
        notional = quantity * entry_price

        logger.debug(
            f"Position size calculated for {symbol}: qty={quantity}, notional=${notional:,.2f}",
            extra={
                "symbol": symbol,
                "quantity": quantity,
                "notional": notional,
                "leverage": leverage,
                "risk_pct": risk_pct,
                "stop_distance_pct": stop_distance_pct * 100,
            },
        )

        return quantity, notional

    def calculate_stop_loss_price(
        self,
        side: str,
        entry_price: float,
        stop_loss_pct: float,
    ) -> float:
        """Calculate stop-loss price."""
        if side.upper() == "LONG":
            return entry_price * (1 - stop_loss_pct / 100)
        else:
            return entry_price * (1 + stop_loss_pct / 100)

    def calculate_take_profit_price(
        self,
        side: str,
        entry_price: float,
        take_profit_pct: float,
    ) -> float:
        """Calculate take-profit price."""
        if side.upper() == "LONG":
            return entry_price * (1 + take_profit_pct / 100)
        else:
            return entry_price * (1 - take_profit_pct / 100)

    def check_daily_loss_limit(self) -> RiskCheckResult:
        """Check if daily loss limit has been reached."""
        stats = self.get_daily_stats()
        max_loss = self.config.max_daily_loss_pct

        if abs(stats.daily_pnl_pct) >= max_loss and stats.daily_pnl_pct < 0:
            logger.risk_event(
                "DAILY_LOSS_KILL_SWITCH",
                f"Daily loss limit reached: {stats.daily_pnl_pct:.2f}%",
                daily_pnl_pct=stats.daily_pnl_pct,
                max_daily_loss_pct=max_loss,
            )
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Daily loss limit reached: {stats.daily_pnl_pct:.2f}% (max: {max_loss}%)",
                details={"daily_pnl_pct": stats.daily_pnl_pct, "limit": max_loss},
            )

        return RiskCheckResult(is_allowed=True, reason="OK")

    def check_consecutive_losses(self) -> RiskCheckResult:
        """Check if consecutive loss limit has been reached."""
        stats = self.get_daily_stats()
        max_consecutive = self.config.max_consecutive_losses

        if stats.consecutive_losses >= max_consecutive:
            logger.risk_event(
                "CONSECUTIVE_LOSS_KILL_SWITCH",
                f"Consecutive loss limit reached: {stats.consecutive_losses}",
                consecutive_losses=stats.consecutive_losses,
                max_consecutive=max_consecutive,
            )
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Consecutive losses: {stats.consecutive_losses} (max: {max_consecutive})",
                details={"consecutive_losses": stats.consecutive_losses, "limit": max_consecutive},
            )

        return RiskCheckResult(is_allowed=True, reason="OK")

    def check_cooldown(self, symbol: str) -> RiskCheckResult:
        """Check if symbol is in cooldown after a loss."""
        cooldown_end = self._cooldown_until.get(symbol, 0)

        if time.time() < cooldown_end:
            remaining = cooldown_end - time.time()
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Cooldown active: {remaining:.0f}s remaining",
                details={"remaining_seconds": remaining},
            )

        return RiskCheckResult(is_allowed=True, reason="OK")

    def set_cooldown(self, symbol: str) -> None:
        """Set cooldown for a symbol after a loss."""
        cooldown_seconds = self.config.cooldown_after_loss_seconds
        self._cooldown_until[symbol] = time.time() + cooldown_seconds

        logger.info(
            f"Cooldown set for {symbol}: {cooldown_seconds}s",
            extra={"symbol": symbol, "cooldown_seconds": cooldown_seconds},
        )

    def check_max_positions(self) -> RiskCheckResult:
        """Check if maximum open positions limit is reached."""
        positions = self.client.get_all_positions()
        current_count = len(positions)
        max_positions = self.config.max_open_positions

        if current_count >= max_positions:
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Max positions reached: {current_count}/{max_positions}",
                details={"current": current_count, "max": max_positions},
            )

        return RiskCheckResult(is_allowed=True, reason="OK")

    def check_total_exposure(self) -> RiskCheckResult:
        """Check if total exposure limit is reached."""
        positions = self.client.get_all_positions()
        balance = self.client.get_usdt_balance()

        total_notional = sum(
            abs(float(p.get("notional", 0))) for p in positions
        )
        max_notional = balance * (self.config.max_total_exposure_pct / 100)

        exposure_pct = (total_notional / balance * 100) if balance > 0 else 0

        if total_notional >= max_notional:
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Max exposure reached: {exposure_pct:.1f}% (max: {self.config.max_total_exposure_pct}%)",
                details={
                    "exposure_pct": exposure_pct,
                    "max_exposure_pct": self.config.max_total_exposure_pct,
                    "notional": total_notional,
                },
            )

        return RiskCheckResult(is_allowed=True, reason="OK")

    def can_open_position(self, symbol: str) -> RiskCheckResult:
        """
        Run all risk checks to determine if a new position can be opened.

        Returns combined result of all checks.
        """
        checks = [
            ("daily_loss", self.check_daily_loss_limit()),
            ("consecutive_losses", self.check_consecutive_losses()),
            ("cooldown", self.check_cooldown(symbol)),
            ("max_positions", self.check_max_positions()),
            ("total_exposure", self.check_total_exposure()),
        ]

        for check_name, result in checks:
            if not result.is_allowed:
                return RiskCheckResult(
                    is_allowed=False,
                    reason=f"[{check_name}] {result.reason}",
                    details=result.details,
                )

        return RiskCheckResult(is_allowed=True, reason="All risk checks passed")

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int,
    ) -> RiskCheckResult:
        """
        Validate a proposed order against risk parameters.

        Returns validation result with any issues.
        """
        # Calculate position notional
        notional = quantity * entry_price

        # Check minimum notional
        symbol_info = self.client.get_symbol_info(symbol)
        if Decimal(str(notional)) < symbol_info.min_notional:
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Notional below minimum: ${notional:.2f} < ${symbol_info.min_notional}",
            )

        # Check position size limit
        if notional > self.config.max_position_size_usd:
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Notional exceeds max: ${notional:.2f} > ${self.config.max_position_size_usd:.2f}",
            )

        # Validate stop-loss distance
        stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price * 100
        if stop_distance_pct > self.config.max_sl_pct:
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Stop distance too large: {stop_distance_pct:.2f}% > {self.config.max_sl_pct}%",
            )

        # Calculate risk amount
        risk_amount = quantity * abs(entry_price - stop_loss_price)
        balance = self.client.get_usdt_balance()
        risk_pct = (risk_amount / balance * 100) if balance > 0 else 100

        symbol_config = self.settings.get_symbol_config(symbol)
        if risk_pct > symbol_config.risk_per_trade_pct * 1.5:  # Allow 50% buffer
            return RiskCheckResult(
                is_allowed=False,
                reason=f"Risk too high: {risk_pct:.2f}% > {symbol_config.risk_per_trade_pct * 1.5:.2f}%",
            )

        return RiskCheckResult(is_allowed=True, reason="Order validated")

    def get_position_summary(self) -> dict[str, Any]:
        """Get summary of current positions and risk metrics."""
        positions = self.client.get_all_positions()
        balance = self.client.get_usdt_balance()
        stats = self.get_daily_stats()

        total_notional = sum(abs(float(p.get("notional", 0))) for p in positions)
        total_pnl = sum(float(p.get("unRealizedProfit", 0)) for p in positions)

        return {
            "balance": balance,
            "open_positions": len(positions),
            "total_notional": total_notional,
            "exposure_pct": (total_notional / balance * 100) if balance > 0 else 0,
            "unrealized_pnl": total_pnl,
            "daily_pnl": stats.realized_pnl,
            "daily_pnl_pct": stats.daily_pnl_pct,
            "consecutive_losses": stats.consecutive_losses,
            "trades_today": stats.trades_count,
            "win_rate": stats.win_rate,
        }

    def reset_daily_stats(self) -> None:
        """Force reset of daily statistics."""
        self._daily_stats = None
        logger.info("Daily stats reset")

    def clear_cooldowns(self) -> None:
        """Clear all cooldowns."""
        self._cooldown_until.clear()
        logger.info("All cooldowns cleared")
