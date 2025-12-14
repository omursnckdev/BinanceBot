"""Tests for technical indicators module."""

import numpy as np
import pandas as pd
import pytest

from signals.indicators import IndicatorCalculator, TrendDirection


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 250  # Enough for EMA 200

    # Generate realistic price data
    base_price = 50000
    returns = np.random.normal(0, 0.02, n)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some trend
    trend = np.linspace(0, 0.1, n)
    prices = prices * (1 + trend)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": prices * (1 + np.random.uniform(0, 0.01, n)),
        "low": prices * (1 - np.random.uniform(0, 0.01, n)),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    })
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def calculator() -> IndicatorCalculator:
    """Create indicator calculator with default config."""
    return IndicatorCalculator()


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_bounds(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        rsi = calculator.calculate_rsi(sample_ohlcv_data)
        valid_rsi = rsi.dropna()

        assert all(valid_rsi >= 0), "RSI should not be negative"
        assert all(valid_rsi <= 100), "RSI should not exceed 100"

    def test_rsi_overbought(self, calculator: IndicatorCalculator) -> None:
        """RSI should be high after consistent gains."""
        # Create data with only gains
        df = pd.DataFrame({
            "close": [100 + i for i in range(50)],  # Consistent uptrend
        })

        rsi = calculator.calculate_rsi(df)
        final_rsi = rsi.iloc[-1]

        assert final_rsi > 70, "RSI should be high (overbought) after consistent gains"

    def test_rsi_oversold(self, calculator: IndicatorCalculator) -> None:
        """RSI should be low after consistent losses."""
        # Create data with only losses
        df = pd.DataFrame({
            "close": [100 - i * 0.5 for i in range(50)],  # Consistent downtrend
        })

        rsi = calculator.calculate_rsi(df)
        final_rsi = rsi.iloc[-1]

        assert final_rsi < 30, "RSI should be low (oversold) after consistent losses"


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_components(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """MACD should return three components."""
        macd_line, signal_line, histogram = calculator.calculate_macd(sample_ohlcv_data)

        assert len(macd_line) == len(sample_ohlcv_data)
        assert len(signal_line) == len(sample_ohlcv_data)
        assert len(histogram) == len(sample_ohlcv_data)

    def test_histogram_is_difference(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """Histogram should be MACD line minus signal line."""
        macd_line, signal_line, histogram = calculator.calculate_macd(sample_ohlcv_data)

        # Check calculation (allowing for small floating point differences)
        expected_histogram = macd_line - signal_line
        np.testing.assert_array_almost_equal(histogram.values, expected_histogram.values, decimal=10)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_band_order(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """Upper band should be above middle, which should be above lower."""
        upper, middle, lower = calculator.calculate_bollinger_bands(sample_ohlcv_data)

        # Skip NaN values
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())

        assert all(upper[valid_idx] >= middle[valid_idx]), "Upper band should be >= middle"
        assert all(middle[valid_idx] >= lower[valid_idx]), "Middle band should be >= lower"

    def test_middle_is_sma(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """Middle band should be the SMA."""
        upper, middle, lower = calculator.calculate_bollinger_bands(sample_ohlcv_data, period=20)

        expected_sma = sample_ohlcv_data["close"].rolling(window=20).mean()

        np.testing.assert_array_almost_equal(
            middle.dropna().values,
            expected_sma.dropna().values,
            decimal=10,
        )


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_positive(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """ATR should always be positive."""
        atr = calculator.calculate_atr(sample_ohlcv_data)
        valid_atr = atr.dropna()

        assert all(valid_atr > 0), "ATR should always be positive"

    def test_atr_increases_with_volatility(self, calculator: IndicatorCalculator) -> None:
        """ATR should be higher for more volatile data."""
        # Low volatility data
        low_vol = pd.DataFrame({
            "high": [100 + i * 0.1 for i in range(30)],
            "low": [99 + i * 0.1 for i in range(30)],
            "close": [99.5 + i * 0.1 for i in range(30)],
        })

        # High volatility data
        high_vol = pd.DataFrame({
            "high": [100 + i * 0.1 + 5 for i in range(30)],
            "low": [100 + i * 0.1 - 5 for i in range(30)],
            "close": [100 + i * 0.1 for i in range(30)],
        })

        atr_low = calculator.calculate_atr(low_vol).iloc[-1]
        atr_high = calculator.calculate_atr(high_vol).iloc[-1]

        assert atr_high > atr_low, "ATR should be higher for more volatile data"


class TestTrendDirection:
    """Tests for trend direction detection."""

    def test_bullish_trend(self, calculator: IndicatorCalculator) -> None:
        """Should detect bullish trend when fast EMA > slow EMA."""
        direction, strength = calculator.get_trend_direction(
            ema_fast=100,
            ema_slow=95,
            current_price=102,
        )

        assert direction == TrendDirection.BULLISH
        assert strength > 0

    def test_bearish_trend(self, calculator: IndicatorCalculator) -> None:
        """Should detect bearish trend when fast EMA < slow EMA."""
        direction, strength = calculator.get_trend_direction(
            ema_fast=95,
            ema_slow=100,
            current_price=93,
        )

        assert direction == TrendDirection.BEARISH
        assert strength > 0


class TestGenerateSignal:
    """Tests for signal generation."""

    def test_signal_output_format(self, calculator: IndicatorCalculator, sample_ohlcv_data: pd.DataFrame) -> None:
        """Signal should have correct format."""
        signal = calculator.generate_signal(sample_ohlcv_data)

        assert signal.is_valid
        assert -1 <= signal.tech_score <= 1
        assert 0 <= signal.tech_confidence <= 1
        assert signal.atr > 0
        assert signal.trend in TrendDirection

    def test_invalid_with_insufficient_data(self, calculator: IndicatorCalculator) -> None:
        """Should return invalid signal with insufficient data."""
        small_df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
        })

        signal = calculator.generate_signal(small_df)

        assert not signal.is_valid
        assert signal.invalidation_reason is not None
