"""
State management module.

Tracks:
- Open positions and their state
- Trading history
- Cooldowns and last actions
- Daily performance metrics
- System health status
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionState:
    """State of an open position."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    leverage: int
    entry_time: float
    stop_loss_price: float
    take_profit_price: float | None
    unrealized_pnl: float = 0.0
    last_update: float = field(default_factory=time.time)

    @property
    def age_hours(self) -> float:
        """Get position age in hours."""
        return (time.time() - self.entry_time) / 3600

    @property
    def notional(self) -> float:
        """Get position notional value."""
        return self.quantity * self.entry_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "unrealized_pnl": self.unrealized_pnl,
            "age_hours": self.age_hours,
        }


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    leverage: int
    entry_time: float
    exit_time: float
    exit_reason: str
    duration_hours: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_hours == 0.0:
            self.duration_hours = (self.exit_time - self.entry_time) / 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "leverage": self.leverage,
            "entry_time": datetime.fromtimestamp(self.entry_time, tz=timezone.utc).isoformat(),
            "exit_time": datetime.fromtimestamp(self.exit_time, tz=timezone.utc).isoformat(),
            "exit_reason": self.exit_reason,
            "duration_hours": self.duration_hours,
        }


@dataclass
class SystemHealth:
    """System health status."""

    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    last_data_update: float = field(default_factory=time.time)
    last_trade: float = 0.0
    api_errors_count: int = 0
    data_stale: bool = False
    time_sync_ok: bool = True
    error_message: str | None = None

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def record_api_error(self, message: str) -> None:
        """Record an API error."""
        self.api_errors_count += 1
        self.error_message = message
        if self.api_errors_count >= 5:
            self.is_healthy = False

    def reset_errors(self) -> None:
        """Reset error count."""
        self.api_errors_count = 0
        self.error_message = None
        self.is_healthy = True


class StateStore:
    """
    Centralized state management.

    Features:
    - Position tracking
    - Trade history
    - Performance metrics
    - State persistence (optional)
    - System health monitoring
    """

    def __init__(
        self,
        settings: Settings | None = None,
        persist_path: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.persist_path = Path(persist_path) if persist_path else None

        # State containers
        self._positions: dict[str, PositionState] = {}
        self._trade_history: list[TradeRecord] = []
        self._cooldowns: dict[str, float] = {}
        self._last_actions: dict[str, dict[str, Any]] = {}
        self._health = SystemHealth()

        # Load persisted state if available
        if self.persist_path and self.persist_path.exists():
            self._load_state()

    # ===== Position Management =====

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss_price: float,
        take_profit_price: float | None = None,
    ) -> PositionState:
        """Add a new position."""
        position = PositionState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            entry_time=time.time(),
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        self._positions[symbol] = position
        self._health.last_trade = time.time()
        self._save_state()

        logger.info(
            f"Position added: {symbol} {side} {quantity} @ {entry_price}",
            extra=position.to_dict(),
        )

        return position

    def get_position(self, symbol: str) -> PositionState | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, PositionState]:
        """Get all open positions."""
        return self._positions.copy()

    def update_position(
        self,
        symbol: str,
        current_price: float,
        unrealized_pnl: float,
    ) -> PositionState | None:
        """Update position with current market data."""
        position = self._positions.get(symbol)
        if not position:
            return None

        position.unrealized_pnl = unrealized_pnl
        position.last_update = time.time()
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord | None:
        """Close a position and record the trade."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Calculate PnL
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100

        # Create trade record
        record = TradeRecord(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=time.time(),
            exit_reason=exit_reason,
        )

        # Update state
        del self._positions[symbol]
        self._trade_history.append(record)
        self._save_state()

        logger.info(
            f"Position closed: {symbol} PnL=${pnl:.2f} ({pnl_pct:.2f}%)",
            extra=record.to_dict(),
        )

        return record

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self._positions

    def get_position_side(self, symbol: str) -> str | None:
        """Get position side for a symbol."""
        position = self._positions.get(symbol)
        return position.side if position else None

    # ===== Cooldown Management =====

    def set_cooldown(self, symbol: str, duration_seconds: int) -> None:
        """Set cooldown for a symbol."""
        self._cooldowns[symbol] = time.time() + duration_seconds
        self._save_state()

    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown."""
        cooldown_end = self._cooldowns.get(symbol, 0)
        return time.time() < cooldown_end

    def get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown seconds."""
        cooldown_end = self._cooldowns.get(symbol, 0)
        return max(0, cooldown_end - time.time())

    def clear_cooldown(self, symbol: str) -> None:
        """Clear cooldown for a symbol."""
        if symbol in self._cooldowns:
            del self._cooldowns[symbol]
            self._save_state()

    # ===== Action Tracking =====

    def record_action(
        self,
        symbol: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record last action for a symbol."""
        self._last_actions[symbol] = {
            "action": action,
            "timestamp": time.time(),
            "details": details or {},
        }

    def get_last_action(self, symbol: str) -> dict[str, Any] | None:
        """Get last action for a symbol."""
        return self._last_actions.get(symbol)

    def time_since_last_action(self, symbol: str) -> float:
        """Get seconds since last action."""
        last = self._last_actions.get(symbol)
        if not last:
            return float("inf")
        return time.time() - last["timestamp"]

    # ===== Trade History =====

    def get_trade_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """Get trade history, optionally filtered by symbol."""
        history = self._trade_history

        if symbol:
            history = [t for t in history if t.symbol == symbol]

        return history[-limit:]

    def get_daily_trades(self) -> list[TradeRecord]:
        """Get today's trades."""
        today = datetime.now(timezone.utc).date()
        return [
            t for t in self._trade_history
            if datetime.fromtimestamp(t.entry_time, tz=timezone.utc).date() == today
        ]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get overall performance statistics."""
        if not self._trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        wins = [t for t in self._trade_history if t.pnl > 0]
        losses = [t for t in self._trade_history if t.pnl <= 0]
        pnls = [t.pnl for t in self._trade_history]

        return {
            "total_trades": len(self._trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self._trade_history) * 100,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
        }

    # ===== Health Monitoring =====

    def get_health(self) -> SystemHealth:
        """Get system health status."""
        return self._health

    def update_health(
        self,
        data_update: bool = False,
        api_error: str | None = None,
        time_sync_ok: bool | None = None,
        data_stale: bool | None = None,
    ) -> None:
        """Update system health status."""
        self._health.update_heartbeat()

        if data_update:
            self._health.last_data_update = time.time()

        if api_error:
            self._health.record_api_error(api_error)

        if time_sync_ok is not None:
            self._health.time_sync_ok = time_sync_ok

        if data_stale is not None:
            self._health.data_stale = data_stale
            if data_stale:
                self._health.is_healthy = False

    def is_healthy(self) -> bool:
        """Check if system is healthy for trading."""
        health = self._health

        # Check heartbeat
        if time.time() - health.last_heartbeat > 120:  # 2 minutes
            return False

        # Check data freshness
        if health.data_stale:
            return False

        # Check time sync
        if not health.time_sync_ok:
            return False

        # Check error count
        if health.api_errors_count >= 5:
            return False

        return health.is_healthy

    # ===== State Persistence =====

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self.persist_path:
            return

        try:
            state = {
                "positions": {k: v.to_dict() for k, v in self._positions.items()},
                "trade_history": [t.to_dict() for t in self._trade_history[-1000:]],
                "cooldowns": self._cooldowns,
                "last_actions": self._last_actions,
                "saved_at": time.time(),
            }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                state = json.load(f)

            # Load cooldowns (only if not expired)
            self._cooldowns = {
                k: v for k, v in state.get("cooldowns", {}).items()
                if v > time.time()
            }

            # Load last actions
            self._last_actions = state.get("last_actions", {})

            logger.info(f"State loaded from {self.persist_path}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def clear_all(self) -> None:
        """Clear all state."""
        self._positions.clear()
        self._trade_history.clear()
        self._cooldowns.clear()
        self._last_actions.clear()
        self._health = SystemHealth()
        self._save_state()
        logger.info("All state cleared")
