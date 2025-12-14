"""Tests for risk management module."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from config import RiskConfig, Settings
from exchange.gate_client import SymbolInfo
from risk.risk_manager import RiskManager, RiskCheckResult


@pytest.fixture
def mock_client() -> MagicMock:
    """Create mock Gate.io client."""
    client = MagicMock()
    client.get_usdt_balance.return_value = 10000.0
    client.get_all_positions.return_value = []
    client.get_symbol_info.return_value = SymbolInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=3,
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("1000"),
        min_notional=Decimal("5"),
        max_leverage=125,
    )
    return client


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.risk = RiskConfig()

    def get_symbol_config(symbol: str) -> MagicMock:
        config = MagicMock()
        config.risk_per_trade_pct = 1.0
        return config

    settings.get_symbol_config = get_symbol_config
    return settings


@pytest.fixture
def risk_manager(mock_client: MagicMock, mock_settings: MagicMock) -> RiskManager:
    """Create risk manager for testing."""
    return RiskManager(mock_client, mock_settings.risk, mock_settings)


class TestPositionSizing:
    """Tests for position size calculation."""

    def test_position_size_based_on_risk(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Position size should be calculated based on risk per trade."""
        # With $10000 balance, 1% risk = $100 risk amount
        # If stop is 2% away, position size = $100 / 0.02 = $5000
        entry_price = 50000.0
        stop_loss_price = 49000.0  # 2% away
        leverage = 5

        quantity, notional = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            leverage=leverage,
            risk_pct=1.0,
        )

        # Position size should be around $5000 notional
        # quantity = 5000 / 50000 = 0.1 BTC
        assert quantity > 0
        assert notional > 0
        assert abs(notional - 5000) < 100  # Within 2% tolerance

    def test_position_size_capped(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Position size should be capped at max_position_size_usd."""
        # Set a low max position size
        risk_manager.config.max_position_size_usd = 500.0

        entry_price = 50000.0
        stop_loss_price = 49000.0
        leverage = 5

        quantity, notional = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            leverage=leverage,
        )

        assert notional <= 500.0

    def test_stop_loss_price_calculation(self, risk_manager: RiskManager) -> None:
        """Stop-loss price should be calculated correctly for both sides."""
        entry_price = 50000.0
        stop_loss_pct = 2.0

        # Long position: SL below entry
        long_sl = risk_manager.calculate_stop_loss_price("LONG", entry_price, stop_loss_pct)
        assert long_sl < entry_price
        assert abs(long_sl - 49000.0) < 1

        # Short position: SL above entry
        short_sl = risk_manager.calculate_stop_loss_price("SHORT", entry_price, stop_loss_pct)
        assert short_sl > entry_price
        assert abs(short_sl - 51000.0) < 1

    def test_take_profit_price_calculation(self, risk_manager: RiskManager) -> None:
        """Take-profit price should be calculated correctly for both sides."""
        entry_price = 50000.0
        take_profit_pct = 4.0

        # Long position: TP above entry
        long_tp = risk_manager.calculate_take_profit_price("LONG", entry_price, take_profit_pct)
        assert long_tp > entry_price
        assert abs(long_tp - 52000.0) < 1

        # Short position: TP below entry
        short_tp = risk_manager.calculate_take_profit_price("SHORT", entry_price, take_profit_pct)
        assert short_tp < entry_price
        assert abs(short_tp - 48000.0) < 1


class TestKillSwitches:
    """Tests for kill-switch functionality."""

    def test_daily_loss_limit(self, risk_manager: RiskManager) -> None:
        """Trading should be blocked when daily loss limit is reached."""
        # Set starting balance
        stats = risk_manager.get_daily_stats()
        stats.starting_balance = 10000.0
        stats.current_balance = 9400.0  # 6% loss (exceeds 5% default)

        result = risk_manager.check_daily_loss_limit()

        assert not result.is_allowed
        assert "Daily loss limit" in result.reason

    def test_consecutive_losses_limit(self, risk_manager: RiskManager) -> None:
        """Trading should be blocked after too many consecutive losses."""
        stats = risk_manager.get_daily_stats()
        stats.consecutive_losses = 5  # Exceeds 3 default

        result = risk_manager.check_consecutive_losses()

        assert not result.is_allowed
        assert "Consecutive losses" in result.reason

    def test_max_positions_limit(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Trading should be blocked when max positions reached."""
        # Simulate 3 open positions (default max is 3)
        mock_client.get_all_positions.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.1"},
            {"symbol": "ETHUSDT", "positionAmt": "1.0"},
            {"symbol": "BNBUSDT", "positionAmt": "10.0"},
        ]

        result = risk_manager.check_max_positions()

        assert not result.is_allowed
        assert "Max positions" in result.reason

    def test_total_exposure_limit(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Trading should be blocked when total exposure is too high."""
        # With $10000 balance and 50% max exposure, max notional is $5000
        mock_client.get_all_positions.return_value = [
            {"symbol": "BTCUSDT", "notional": 6000},  # Exceeds 50%
        ]

        result = risk_manager.check_total_exposure()

        assert not result.is_allowed
        assert "exposure" in result.reason.lower()


class TestCooldown:
    """Tests for cooldown functionality."""

    def test_cooldown_blocks_trading(self, risk_manager: RiskManager) -> None:
        """Symbol should be blocked during cooldown."""
        risk_manager.set_cooldown("BTCUSDT")

        result = risk_manager.check_cooldown("BTCUSDT")

        assert not result.is_allowed
        assert "Cooldown" in result.reason

    def test_cooldown_expires(self, risk_manager: RiskManager) -> None:
        """Cooldown should expire after duration."""
        # Set a very short cooldown
        risk_manager.config.cooldown_after_loss_seconds = 0
        risk_manager.set_cooldown("BTCUSDT")

        result = risk_manager.check_cooldown("BTCUSDT")

        assert result.is_allowed

    def test_different_symbols_independent_cooldown(self, risk_manager: RiskManager) -> None:
        """Cooldown on one symbol shouldn't affect others."""
        risk_manager.set_cooldown("BTCUSDT")

        result = risk_manager.check_cooldown("ETHUSDT")

        assert result.is_allowed


class TestOrderValidation:
    """Tests for order validation."""

    def test_valid_order(self, risk_manager: RiskManager) -> None:
        """Valid order should pass validation."""
        result = risk_manager.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            entry_price=50000,
            stop_loss_price=49000,
            leverage=5,
        )

        assert result.is_allowed

    def test_notional_below_minimum(self, risk_manager: RiskManager) -> None:
        """Order with notional below minimum should fail."""
        result = risk_manager.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.0001,  # Very small
            entry_price=50000,
            stop_loss_price=49000,
            leverage=5,
        )

        assert not result.is_allowed
        assert "minimum" in result.reason.lower()

    def test_notional_above_maximum(self, risk_manager: RiskManager) -> None:
        """Order with notional above maximum should fail."""
        result = risk_manager.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=100,  # $5M notional
            entry_price=50000,
            stop_loss_price=49000,
            leverage=5,
        )

        assert not result.is_allowed
        assert "max" in result.reason.lower()


class TestCanOpenPosition:
    """Tests for combined risk check."""

    def test_all_checks_pass(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Should allow opening position when all checks pass."""
        result = risk_manager.can_open_position("BTCUSDT")

        assert result.is_allowed
        assert "passed" in result.reason.lower()

    def test_blocked_by_any_check(
        self,
        risk_manager: RiskManager,
        mock_client: MagicMock,
    ) -> None:
        """Should block if any check fails."""
        # Set consecutive losses
        stats = risk_manager.get_daily_stats()
        stats.consecutive_losses = 10

        result = risk_manager.can_open_position("BTCUSDT")

        assert not result.is_allowed


class TestTradeRecording:
    """Tests for trade result recording."""

    def test_record_win(self, risk_manager: RiskManager) -> None:
        """Recording a win should update stats correctly."""
        risk_manager.record_trade_result(pnl=100.0, is_win=True)

        stats = risk_manager.get_daily_stats()
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.consecutive_losses == 0
        assert stats.realized_pnl == 100.0

    def test_record_loss(self, risk_manager: RiskManager) -> None:
        """Recording a loss should update stats correctly."""
        risk_manager.record_trade_result(pnl=-50.0, is_win=False)

        stats = risk_manager.get_daily_stats()
        assert stats.wins == 0
        assert stats.losses == 1
        assert stats.consecutive_losses == 1
        assert stats.realized_pnl == -50.0

    def test_consecutive_losses_reset_on_win(self, risk_manager: RiskManager) -> None:
        """Winning should reset consecutive loss counter."""
        # Record some losses
        risk_manager.record_trade_result(pnl=-50.0, is_win=False)
        risk_manager.record_trade_result(pnl=-30.0, is_win=False)

        stats = risk_manager.get_daily_stats()
        assert stats.consecutive_losses == 2

        # Record a win
        risk_manager.record_trade_result(pnl=100.0, is_win=True)

        assert stats.consecutive_losses == 0
