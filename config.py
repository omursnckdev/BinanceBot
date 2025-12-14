"""
Configuration module using Pydantic for type-safe settings.
All settings have safe defaults for testnet + dry-run mode.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Trading environment - testnet is the safe default."""
    TESTNET = "testnet"
    MAINNET = "mainnet"


class MarginType(str, Enum):
    """Margin type for futures positions."""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class WhaleDefensiveAction(str, Enum):
    """Action to take when whale activity opposes position."""
    CLOSE_MARKET = "CLOSE_MARKET"
    REDUCE_50_PERCENT = "REDUCE_50_PERCENT"
    TIGHTEN_STOP = "TIGHTEN_STOP"


class SymbolConfig(BaseModel):
    """Per-symbol configuration."""
    symbol: str
    enabled: bool = True
    max_leverage: int = Field(default=5, ge=1, le=125)
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0)

    # Signal weights for fusion
    tech_weight: float = Field(default=0.5, ge=0, le=1)
    whale_weight: float = Field(default=0.3, ge=0, le=1)
    sentiment_weight: float = Field(default=0.2, ge=0, le=1)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper()


class IndicatorConfig(BaseModel):
    """Technical indicator parameters."""
    # RSI
    rsi_period: int = Field(default=14, ge=2, le=50)
    rsi_overbought: float = Field(default=70.0, ge=50, le=100)
    rsi_oversold: float = Field(default=30.0, ge=0, le=50)

    # MACD
    macd_fast: int = Field(default=12, ge=2, le=50)
    macd_slow: int = Field(default=26, ge=5, le=100)
    macd_signal: int = Field(default=9, ge=2, le=50)

    # Bollinger Bands
    bb_period: int = Field(default=20, ge=5, le=50)
    bb_std_dev: float = Field(default=2.0, ge=0.5, le=4.0)

    # ATR
    atr_period: int = Field(default=14, ge=2, le=50)

    # EMA trend filter
    ema_fast: int = Field(default=50, ge=5, le=100)
    ema_slow: int = Field(default=200, ge=50, le=500)

    # Timeframes
    primary_timeframe: str = Field(default="1m")
    confirmation_timeframes: list[str] = Field(default=["5m", "15m"])


class RiskConfig(BaseModel):
    """Risk management configuration."""
    # Position sizing
    max_position_size_usd: float = Field(default=1000.0, ge=10)
    max_open_positions: int = Field(default=3, ge=1, le=10)
    max_total_exposure_pct: float = Field(default=50.0, ge=1, le=100)

    # Stop-loss
    atr_sl_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    max_sl_pct: float = Field(default=5.0, ge=0.5, le=20.0)

    # Take-profit
    enable_take_profit: bool = True
    tp_r_multiple: float = Field(default=2.0, ge=0.5, le=10.0)

    # Kill-switches
    max_daily_loss_pct: float = Field(default=5.0, ge=1, le=50)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)

    # Cooldown
    cooldown_after_loss_seconds: int = Field(default=300, ge=0, le=3600)

    # Time stop
    max_position_duration_hours: int = Field(default=24, ge=1, le=168)


class WhaleConfig(BaseModel):
    """Whale detection configuration."""
    enabled: bool = True

    # Trade size detection
    large_trade_percentile: float = Field(default=95.0, ge=80, le=99.9)
    rolling_window_minutes: int = Field(default=60, ge=5, le=1440)
    min_whale_notional_usd: float = Field(default=50000.0, ge=1000)

    # Order book imbalance
    imbalance_threshold: float = Field(default=0.7, ge=0.5, le=0.95)
    depth_levels: int = Field(default=20, ge=5, le=100)

    # Defensive action
    defensive_action: WhaleDefensiveAction = WhaleDefensiveAction.CLOSE_MARKET
    whale_alert_threshold: float = Field(default=0.7, ge=0.3, le=1.0)


class SentimentConfig(BaseModel):
    """News sentiment configuration."""
    enabled: bool = True

    # API provider
    provider: str = Field(default="cryptopanic")  # cryptopanic, newsapi, gdelt
    api_key_env_var: str = Field(default="NEWS_API_KEY")

    # Caching
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    max_articles_per_fetch: int = Field(default=50, ge=10, le=200)

    # Analysis
    use_transformer_model: bool = False  # Start with VADER for simplicity

    # Coin keyword mapping (defaults)
    keyword_map: dict[str, list[str]] = Field(default_factory=lambda: {
        "BTC": ["bitcoin", "btc", "satoshi"],
        "ETH": ["ethereum", "eth", "vitalik"],
        "BNB": ["binance", "bnb"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
    })


class TradingConfig(BaseModel):
    """Trading strategy configuration."""
    # Entry thresholds
    entry_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    high_confidence_threshold: float = Field(default=0.7, ge=0.5, le=1.0)

    # Leverage scaling
    min_leverage: int = Field(default=2, ge=1, le=10)
    max_leverage: int = Field(default=5, ge=1, le=20)

    # Market conditions
    max_atr_pct: float = Field(default=5.0, ge=0.5, le=20.0)
    max_spread_pct: float = Field(default=0.1, ge=0.01, le=1.0)
    min_liquidity_usd: float = Field(default=100000.0, ge=1000)

    # Anti-flip-flop
    min_time_between_trades_seconds: int = Field(default=60, ge=0, le=3600)
    require_trend_confirmation: bool = True


class Settings(BaseSettings):
    """
    Main settings class with environment variable loading.

    CRITICAL SAFETY DEFAULTS:
    - ENV=testnet
    - DRY_RUN=true
    - ALLOW_LIVE_TRADING=false

    To enable live trading on mainnet, ALL of these must be explicitly set:
    - ENV=mainnet
    - DRY_RUN=false
    - ALLOW_LIVE_TRADING=true
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== SAFETY SETTINGS (DEFAULTS ARE CONSERVATIVE) =====
    env: Environment = Field(default=Environment.TESTNET)
    dry_run: bool = Field(default=True)
    allow_live_trading: bool = Field(default=False)

    # ===== API CREDENTIALS (loaded from env vars) =====
    gate_api_key: str = Field(default="", validation_alias="binance_api_key")
    gate_api_secret: str = Field(default="", validation_alias="binance_api_secret")

    # ===== TRADING SYMBOLS =====
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )

    # ===== MARGIN SETTINGS =====
    margin_type: MarginType = Field(default=MarginType.ISOLATED)

    # ===== SUB-CONFIGURATIONS =====
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    whale: WhaleConfig = Field(default_factory=WhaleConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)

    # ===== LOGGING =====
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trading.log")
    log_json: bool = Field(default=True)

    # ===== DATA INTEGRITY =====
    max_candle_age_seconds: int = Field(default=120, ge=10, le=600)
    max_time_sync_drift_ms: int = Field(default=1000, ge=100, le=5000)

    @model_validator(mode="after")
    def validate_live_trading_requirements(self) -> "Settings":
        """
        Enforce that live trading on mainnet requires explicit opt-in.
        This is a CRITICAL safety check.
        """
        if self.env == Environment.MAINNET:
            if not self.allow_live_trading:
                # Force dry-run on mainnet unless explicitly allowed
                object.__setattr__(self, "dry_run", True)

            if not self.dry_run and not self.allow_live_trading:
                raise ValueError(
                    "SAFETY ERROR: Cannot disable dry_run on mainnet "
                    "without setting ALLOW_LIVE_TRADING=true"
                )

        return self

    @property
    def is_live_trading_enabled(self) -> bool:
        """Check if actual live trading is enabled (not dry-run)."""
        return (
            self.env == Environment.MAINNET
            and not self.dry_run
            and self.allow_live_trading
        )

    @property
    def is_testnet(self) -> bool:
        """Check if running on testnet."""
        return self.env == Environment.TESTNET

    def get_symbol_config(self, symbol: str) -> SymbolConfig:
        """Get configuration for a specific symbol."""
        return SymbolConfig(symbol=symbol.upper())

    def mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        """Mask a secret for safe logging."""
        if len(secret) <= visible_chars:
            return "*" * len(secret)
        return secret[:visible_chars] + "*" * (len(secret) - visible_chars)

    def get_safe_dict(self) -> dict[str, Any]:
        """Get settings dict with secrets masked for logging."""
        data = self.model_dump()
        if data.get("gate_api_key"):
            data["gate_api_key"] = self.mask_secret(data["gate_api_key"])
        if data.get("gate_api_secret"):
            data["gate_api_secret"] = self.mask_secret(data["gate_api_secret"])
        return data


# Global settings instance (lazy-loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload of settings (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
