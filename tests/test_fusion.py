"""Tests for fusion strategy module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import Settings, TradingConfig
from signals.indicators import TechnicalSignal, TrendDirection, IndicatorValues
from signals.whales import WhaleSignal, WhalePressure
from signals.sentiment import SentimentSignal
from strategy.fusion import FusionStrategy, TradeAction, TradeDecision


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.trading = TradingConfig()
    settings.indicators = MagicMock()
    settings.indicators.primary_timeframe = "1m"
    settings.indicators.confirmation_timeframes = ["5m", "15m"]

    def get_symbol_config(symbol: str) -> MagicMock:
        config = MagicMock()
        config.tech_weight = 0.5
        config.whale_weight = 0.3
        config.sentiment_weight = 0.2
        return config

    settings.get_symbol_config = get_symbol_config
    return settings


@pytest.fixture
def mock_market_data() -> MagicMock:
    """Create mock market data manager."""
    mm = MagicMock()
    mm.get_candles_df.return_value = pd.DataFrame({
        "open": [100] * 250,
        "high": [101] * 250,
        "low": [99] * 250,
        "close": [100.5] * 250,
        "volume": [1000] * 250,
    })
    mm.get_market_conditions.return_value = {
        "spread_pct": 0.01,
        "bid_depth": 500000,
        "ask_depth": 500000,
    }
    mm.get_latest_price.return_value = 100.0
    return mm


@pytest.fixture
def mock_indicator_values() -> IndicatorValues:
    """Create mock indicator values."""
    return IndicatorValues(
        rsi=50, rsi_signal=0,
        macd_line=0.1, macd_signal=0.05, macd_histogram=0.05, macd_signal_value=0.5,
        bb_upper=105, bb_middle=100, bb_lower=95, bb_width=0.1, bb_position=0.5, bb_signal=0,
        atr=2.0, atr_pct=2.0,
        ema_fast=100, ema_slow=98, trend=TrendDirection.BULLISH, trend_strength=0.5,
    )


@pytest.fixture
def mock_tech_signal(mock_indicator_values: IndicatorValues) -> TechnicalSignal:
    """Create mock technical signal."""
    return TechnicalSignal(
        tech_score=0.5,
        tech_confidence=0.7,
        trend=TrendDirection.BULLISH,
        atr=2.0,
        atr_pct=2.0,
        indicators=mock_indicator_values,
        is_valid=True,
    )


@pytest.fixture
def mock_whale_signal() -> WhaleSignal:
    """Create mock whale signal."""
    return WhaleSignal(
        whale_score=0.3,
        whale_confidence=0.5,
        whale_alert_opposite=False,
        pressure=WhalePressure.BULLISH,
        large_trades_detected=3,
        order_book_imbalance=0.2,
    )


@pytest.fixture
def mock_sentiment_signal() -> SentimentSignal:
    """Create mock sentiment signal."""
    return SentimentSignal(
        sentiment_score=0.4,
        sentiment_confidence=0.6,
        global_sentiment=0.3,
        article_count=10,
        recent_articles=["Bitcoin hits new high"],
        is_valid=True,
    )


class TestFinalScoreCalculation:
    """Tests for final score calculation."""

    def test_weighted_score(self) -> None:
        """Final score should be weighted combination of signals."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()

        score = strategy._calculate_final_score(
            tech_score=0.5,
            whale_score=0.3,
            sentiment_score=0.2,
            weights=(0.5, 0.3, 0.2),  # Normalized weights
        )

        # Expected: 0.5*0.5 + 0.3*0.3 + 0.2*0.2 = 0.25 + 0.09 + 0.04 = 0.38
        assert abs(score - 0.38) < 0.01

    def test_score_clamped_to_range(self) -> None:
        """Final score should be clamped to [-1, 1]."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()

        # Very high positive signals
        score_high = strategy._calculate_final_score(
            tech_score=1.0,
            whale_score=1.0,
            sentiment_score=1.0,
            weights=(1.0, 1.0, 1.0),
        )
        assert score_high <= 1.0

        # Very negative signals
        score_low = strategy._calculate_final_score(
            tech_score=-1.0,
            whale_score=-1.0,
            sentiment_score=-1.0,
            weights=(1.0, 1.0, 1.0),
        )
        assert score_low >= -1.0


class TestTradeDecision:
    """Tests for trade decision logic."""

    def test_long_decision_on_high_score(
        self,
        mock_market_data: MagicMock,
        mock_settings: Settings,
        mock_tech_signal: TechnicalSignal,
        mock_whale_signal: WhaleSignal,
        mock_sentiment_signal: SentimentSignal,
    ) -> None:
        """Should generate LONG decision when score is above threshold."""
        with patch("strategy.fusion.IndicatorCalculator") as MockIndicator, \
             patch("strategy.fusion.WhaleDetector") as MockWhale, \
             patch("strategy.fusion.SentimentAnalyzer") as MockSentiment:

            mock_indicator = MockIndicator.return_value
            mock_indicator.generate_signal.return_value = mock_tech_signal

            mock_whale = MockWhale.return_value
            mock_whale.generate_signal.return_value = mock_whale_signal

            mock_sentiment = MockSentiment.return_value
            mock_sentiment.generate_signal.return_value = mock_sentiment_signal

            strategy = FusionStrategy(
                mock_market_data,
                mock_indicator,
                mock_whale,
                mock_sentiment,
                mock_settings,
            )

            decision = strategy.generate_decision("BTCUSDT")

            assert decision.action == TradeAction.LONG
            assert decision.confidence > 0

    def test_flat_decision_in_no_trade_zone(
        self,
        mock_market_data: MagicMock,
        mock_settings: Settings,
        mock_indicator_values: IndicatorValues,
    ) -> None:
        """Should generate FLAT decision when score is near zero."""
        # Create weak signals
        weak_tech_signal = TechnicalSignal(
            tech_score=0.1,  # Below entry threshold
            tech_confidence=0.3,
            trend=TrendDirection.NEUTRAL,
            atr=2.0,
            atr_pct=2.0,
            indicators=mock_indicator_values,
            is_valid=True,
        )

        weak_whale_signal = WhaleSignal(
            whale_score=0.05,
            whale_confidence=0.2,
            whale_alert_opposite=False,
            pressure=WhalePressure.NEUTRAL,
            large_trades_detected=0,
            order_book_imbalance=0.0,
        )

        weak_sentiment_signal = SentimentSignal(
            sentiment_score=0.0,
            sentiment_confidence=0.1,
            global_sentiment=0.0,
            article_count=2,
            recent_articles=[],
            is_valid=True,
        )

        with patch("strategy.fusion.IndicatorCalculator") as MockIndicator, \
             patch("strategy.fusion.WhaleDetector") as MockWhale, \
             patch("strategy.fusion.SentimentAnalyzer") as MockSentiment:

            mock_indicator = MockIndicator.return_value
            mock_indicator.generate_signal.return_value = weak_tech_signal

            mock_whale = MockWhale.return_value
            mock_whale.generate_signal.return_value = weak_whale_signal

            mock_sentiment = MockSentiment.return_value
            mock_sentiment.generate_signal.return_value = weak_sentiment_signal

            strategy = FusionStrategy(
                mock_market_data,
                mock_indicator,
                mock_whale,
                mock_sentiment,
                mock_settings,
            )

            decision = strategy.generate_decision("BTCUSDT")

            # Score should be low, resulting in FLAT
            assert decision.action == TradeAction.FLAT

    def test_whale_alert_triggers_close(
        self,
        mock_market_data: MagicMock,
        mock_settings: Settings,
        mock_tech_signal: TechnicalSignal,
        mock_sentiment_signal: SentimentSignal,
    ) -> None:
        """Whale alert should trigger position close."""
        whale_alert_signal = WhaleSignal(
            whale_score=-0.8,
            whale_confidence=0.9,
            whale_alert_opposite=True,  # Alert!
            pressure=WhalePressure.BEARISH,
            large_trades_detected=10,
            order_book_imbalance=-0.5,
        )

        with patch("strategy.fusion.IndicatorCalculator") as MockIndicator, \
             patch("strategy.fusion.WhaleDetector") as MockWhale, \
             patch("strategy.fusion.SentimentAnalyzer") as MockSentiment:

            mock_indicator = MockIndicator.return_value
            mock_indicator.generate_signal.return_value = mock_tech_signal

            mock_whale = MockWhale.return_value
            mock_whale.generate_signal.return_value = whale_alert_signal

            mock_sentiment = MockSentiment.return_value
            mock_sentiment.generate_signal.return_value = mock_sentiment_signal

            strategy = FusionStrategy(
                mock_market_data,
                mock_indicator,
                mock_whale,
                mock_sentiment,
                mock_settings,
            )

            # Simulate having a LONG position
            decision = strategy.generate_decision("BTCUSDT", current_position_side="LONG")

            assert decision.action == TradeAction.FLAT


class TestLeverageCalculation:
    """Tests for leverage calculation."""

    def test_leverage_scales_with_confidence(self) -> None:
        """Leverage should increase with confidence."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()
        strategy.settings.trading = TradingConfig(min_leverage=2, max_leverage=10)

        low_confidence_leverage = strategy._calculate_leverage(0.3, 2.0)
        high_confidence_leverage = strategy._calculate_leverage(0.9, 2.0)

        assert high_confidence_leverage > low_confidence_leverage

    def test_leverage_reduced_for_high_volatility(self) -> None:
        """Leverage should be reduced for high ATR."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()
        strategy.settings.trading = TradingConfig(min_leverage=2, max_leverage=10)

        low_vol_leverage = strategy._calculate_leverage(0.7, 1.0)
        high_vol_leverage = strategy._calculate_leverage(0.7, 5.0)

        assert high_vol_leverage <= low_vol_leverage


class TestStopLossCalculation:
    """Tests for stop-loss calculation."""

    def test_stop_loss_based_on_atr(self) -> None:
        """Stop-loss should be proportional to ATR."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()
        strategy.settings.risk = MagicMock()
        strategy.settings.risk.atr_sl_multiplier = 2.0
        strategy.settings.risk.max_sl_pct = 5.0

        sl_low_atr = strategy._calculate_stop_loss(TradeAction.LONG, 1.0)
        sl_high_atr = strategy._calculate_stop_loss(TradeAction.LONG, 3.0)

        assert sl_high_atr > sl_low_atr

    def test_stop_loss_capped(self) -> None:
        """Stop-loss should be capped at maximum."""
        strategy = FusionStrategy.__new__(FusionStrategy)
        strategy.settings = MagicMock()
        strategy.settings.risk = MagicMock()
        strategy.settings.risk.atr_sl_multiplier = 2.0
        strategy.settings.risk.max_sl_pct = 3.0  # Low cap

        sl = strategy._calculate_stop_loss(TradeAction.LONG, 10.0)  # Very high ATR

        assert sl <= 3.0
