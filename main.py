#!/usr/bin/env python3
"""
Binance Futures Trading Bot - Main Entry Point

This bot trades USDT-M Futures using a fusion strategy that combines:
- Technical indicators (RSI, MACD, Bollinger, ATR, EMA)
- Whale activity detection
- News sentiment analysis

SAFETY DEFAULTS:
- ENV=testnet
- DRY_RUN=true
- ALLOW_LIVE_TRADING=false

To enable live trading, ALL of these must be set:
- ENV=mainnet
- DRY_RUN=false
- ALLOW_LIVE_TRADING=true
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import Any

from config import Settings, get_settings
from data.market_data import MarketDataManager
from exchange.binance_client import BinanceClient
from execution.executor import OrderExecutor
from risk.risk_manager import RiskManager
from signals.indicators import IndicatorCalculator
from signals.sentiment import SentimentAnalyzer
from signals.whales import WhaleDetector
from state.store import StateStore
from strategy.fusion import FusionStrategy, TradeAction
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.

    Manages the event loop and coordinates all components.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all bot components."""
        logger.info("Initializing trading bot components...")

        # Core components
        self.client = BinanceClient(self.settings)
        self.market_data = MarketDataManager(self.client, self.settings)
        self.state = StateStore(self.settings, persist_path="data/state.json")

        # Signal generators
        self.indicators = IndicatorCalculator(self.settings.indicators)
        self.whales = WhaleDetector(self.market_data, self.settings.whale)
        self.sentiment = SentimentAnalyzer(self.settings.sentiment)

        # Strategy and execution
        self.strategy = FusionStrategy(
            self.market_data,
            self.indicators,
            self.whales,
            self.sentiment,
            self.settings,
        )
        self.risk = RiskManager(self.client, self.settings.risk, self.settings)
        self.executor = OrderExecutor(self.client, self.risk, self.settings)

        logger.info("All components initialized")

    def _log_startup_info(self) -> None:
        """Log startup information and safety checks."""
        env = self.settings.env.value.upper()
        dry_run = self.settings.dry_run
        live_enabled = self.settings.allow_live_trading

        logger.info("=" * 60)
        logger.info("BINANCE FUTURES TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Environment: {env}")
        logger.info(f"Dry Run: {dry_run}")
        logger.info(f"Live Trading Allowed: {live_enabled}")
        logger.info(f"Symbols: {', '.join(self.settings.symbols)}")
        logger.info("=" * 60)

        if self.settings.is_live_trading_enabled:
            logger.warning("!!! LIVE TRADING ENABLED - REAL MONEY AT RISK !!!")
            logger.warning("Make sure you understand the risks before proceeding.")
        elif self.settings.env.value == "mainnet" and dry_run:
            logger.info("Running on MAINNET in DRY-RUN mode (no real orders)")
        else:
            logger.info("Running on TESTNET - safe for experimentation")

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def shutdown_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._running = False
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    async def _check_prerequisites(self) -> bool:
        """Check all prerequisites before starting."""
        logger.info("Checking prerequisites...")

        # Check API connectivity
        try:
            self.client.sync_time()
            logger.info("✓ API connection OK")
        except Exception as e:
            logger.error(f"✗ API connection failed: {e}")
            return False

        # Check time sync
        if not self.client.check_time_sync():
            logger.error("✗ Time sync drift too large")
            return False
        logger.info("✓ Time sync OK")

        # Check API credentials
        try:
            balance = self.client.get_usdt_balance()
            logger.info(f"✓ Account balance: ${balance:,.2f} USDT")
        except Exception as e:
            logger.error(f"✗ Failed to get account balance: {e}")
            return False

        # Load exchange info
        try:
            for symbol in self.settings.symbols:
                self.client.get_symbol_info(symbol)
            logger.info("✓ Exchange info loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load exchange info: {e}")
            return False

        logger.info("All prerequisites satisfied")
        return True

    async def _fetch_market_data(self, symbol: str) -> bool:
        """Fetch all market data for a symbol."""
        try:
            success = self.market_data.fetch_all(symbol)
            if success:
                self.state.update_health(data_update=True)
            return success
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            self.state.update_health(api_error=str(e))
            return False

    async def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol - fetch data and make decision."""
        # Fetch latest market data
        if not await self._fetch_market_data(symbol):
            logger.warning(f"Skipping {symbol} due to data fetch failure")
            return

        # Validate data
        is_valid, reason = self.market_data.is_data_valid(symbol)
        if not is_valid:
            logger.warning(f"Data validation failed for {symbol}: {reason}")
            self.state.update_health(data_stale=True)
            return

        self.state.update_health(data_stale=False)

        # Get current position
        current_position = self.client.get_position(symbol)
        position_side = None
        if current_position:
            amt = float(current_position.get("positionAmt", 0))
            if amt > 0:
                position_side = "LONG"
            elif amt < 0:
                position_side = "SHORT"

        # Generate trading decision
        decision = self.strategy.generate_decision(symbol, position_side)

        # Handle existing position
        if position_side:
            # Check if we should close
            if decision.action == TradeAction.FLAT:
                self.executor.close_position(symbol, "Whale alert")
                return

            # Check for opposite signal
            should_close, reason = self.strategy.should_close_position(
                symbol,
                position_side,
                float(current_position.get("entryPrice", 0)),
                self.market_data.get_latest_price(symbol),
            )
            if should_close:
                self.executor.close_position(symbol, reason)

        else:
            # No position - check if we should open
            if decision.action in (TradeAction.LONG, TradeAction.SHORT):
                if decision.is_valid:
                    bracket = self.executor.execute_decision(decision)
                    if bracket:
                        # Record in state
                        self.state.add_position(
                            symbol=symbol,
                            side="LONG" if decision.action == TradeAction.LONG else "SHORT",
                            entry_price=bracket.entry_order.avg_fill_price or self.market_data.get_latest_price(symbol),
                            quantity=bracket.entry_order.quantity,
                            leverage=decision.suggested_leverage,
                            stop_loss_price=bracket.stop_loss_order.stop_price if bracket.stop_loss_order else 0,
                            take_profit_price=bracket.take_profit_order.stop_price if bracket.take_profit_order else None,
                        )
                        self.state.record_action(symbol, decision.action.value, decision.to_dict())

    async def _run_loop(self) -> None:
        """Main trading loop."""
        loop_interval = 30  # seconds

        while self._running:
            loop_start = time.time()

            try:
                # Check system health
                if not self.state.is_healthy():
                    logger.warning("System health check failed, waiting...")
                    await asyncio.sleep(10)
                    continue

                # Check time sync periodically
                if not self.client.check_time_sync():
                    logger.warning("Time sync check failed")
                    self.state.update_health(time_sync_ok=False)
                    await asyncio.sleep(10)
                    continue
                self.state.update_health(time_sync_ok=True)

                # Process each symbol
                for symbol in self.settings.symbols:
                    if not self._running:
                        break

                    await self._process_symbol(symbol)

                    # Small delay between symbols
                    await asyncio.sleep(1)

                # Update risk manager balance
                balance = self.client.get_usdt_balance()
                self.risk.update_balance(balance)

                # Log summary
                summary = self.risk.get_position_summary()
                logger.debug(
                    f"Loop complete: {summary['open_positions']} positions, "
                    f"exposure: {summary['exposure_pct']:.1f}%, "
                    f"daily PnL: ${summary['daily_pnl']:.2f}",
                    extra=summary,
                )

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.state.update_health(api_error=str(e))

            # Wait for next iteration
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - elapsed)

            if sleep_time > 0:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time,
                    )
                except asyncio.TimeoutError:
                    pass

    async def run(self) -> None:
        """Run the trading bot."""
        self._log_startup_info()
        self._setup_signal_handlers()

        # Check prerequisites
        if not await self._check_prerequisites():
            logger.error("Prerequisites check failed, exiting")
            sys.exit(1)

        # Start main loop
        self._running = True
        logger.info("Starting main trading loop...")

        try:
            await self._run_loop()
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down trading bot...")
        self._running = False

        # Close all positions if configured
        # (By default, we don't close on shutdown to preserve positions)

        # Cancel any pending orders
        for symbol in self.settings.symbols:
            if self.executor.has_active_bracket(symbol):
                # Don't cancel SL/TP orders - they protect our positions
                pass

        logger.info("Trading bot shutdown complete")


def main() -> None:
    """Main entry point."""
    # Load settings
    settings = get_settings()

    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file,
        use_json=settings.log_json,
    )

    # Create and run bot
    bot = TradingBot(settings)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
