"""
Fusion strategy module.

Combines technical indicators, whale activity, and news sentiment
into a single decision engine for trading actions.

Features:
- Weighted signal fusion
- Trend confirmation requirements
- Market condition filters
- Leverage and position sizing
- Anti-flip-flop protection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config import Settings, SymbolConfig, TradingConfig, WhaleDefensiveAction, get_settings
from data.market_data import MarketDataManager
from signals.indicators import IndicatorCalculator, TechnicalSignal, TrendDirection
from signals.sentiment import SentimentAnalyzer, SentimentSignal
from signals.whales import WhaleDetector, WhaleSignal
from utils.logger import TradingLogger, get_logger

logger: TradingLogger = get_logger(__name__)  # type: ignore


class TradeAction(str, Enum):
    """
    Possible trading actions.

    IMPORTANT: FLAT means "no new entry" - it does NOT mean "close existing position".
    Exits must be handled via explicit exit_reason or defensive_action.
    """

    LONG = "LONG"      # Open/maintain long position
    SHORT = "SHORT"    # Open/maintain short position
    FLAT = "FLAT"      # No entry signal (do NOT close existing positions!)
    HOLD = "HOLD"      # Explicitly hold current state


@dataclass
class TradeDecision:
    """
    Complete trading decision output.

    IMPORTANT distinction:
    - action=FLAT means "don't open new position" (NOT "close existing")
    - exit_reason (if set) means "close existing position with this reason"
    - defensive_action (if set) means "execute this whale defensive action"
    """

    symbol: str
    action: TradeAction
    confidence: float  # 0 to 1

    # Signal components
    tech_score: float
    whale_score: float
    sentiment_score: float
    final_score: float

    # Position parameters
    suggested_leverage: int
    suggested_position_pct: float  # Percentage of risk budget
    stop_loss_pct: float  # Percentage from entry
    take_profit_pct: float | None  # Percentage from entry

    # Metadata
    trend: TrendDirection
    atr_pct: float
    is_valid: bool
    rejection_reason: str | None = None
    timestamp: float = 0.0

    # EXPLICIT exit signals (new fields)
    exit_reason: str | None = None  # If set, close position with this reason
    defensive_action: WhaleDefensiveAction | None = None  # Whale defensive action

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "tech_score": round(self.tech_score, 4),
            "whale_score": round(self.whale_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "final_score": round(self.final_score, 4),
            "suggested_leverage": self.suggested_leverage,
            "suggested_position_pct": round(self.suggested_position_pct, 2),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4) if self.take_profit_pct else None,
            "trend": self.trend.value,
            "atr_pct": round(self.atr_pct, 4),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "exit_reason": self.exit_reason,
            "defensive_action": self.defensive_action.value if self.defensive_action else None,
        }


class FusionStrategy:
    """
    Fuses multiple signals into trading decisions.

    Decision process:
    1. Compute weighted final_score from tech/whale/sentiment
    2. Apply entry threshold and no-trade zone
    3. Check trend confirmation requirements
    4. Filter by market conditions (ATR, spread, liquidity)
    5. Calculate leverage and position size
    6. Set stop-loss and take-profit levels
    """

    def __init__(
        self,
        market_data: MarketDataManager,
        indicator_calc: IndicatorCalculator,
        whale_detector: WhaleDetector,
        sentiment_analyzer: SentimentAnalyzer,
        settings: Settings | None = None,
    ) -> None:
        self.market_data = market_data
        self.indicators = indicator_calc
        self.whales = whale_detector
        self.sentiment = sentiment_analyzer
        self.settings = settings or get_settings()
        self._last_decisions: dict[str, TradeDecision] = {}
        self._last_action_time: dict[str, float] = {}

    def _get_symbol_weights(self, symbol: str) -> tuple[float, float, float]:
        """Get signal weights for a symbol."""
        config = self.settings.get_symbol_config(symbol)
        return (
            config.tech_weight,
            config.whale_weight,
            config.sentiment_weight,
        )

    def _calculate_final_score(
        self,
        tech_score: float,
        whale_score: float,
        sentiment_score: float,
        weights: tuple[float, float, float],
    ) -> float:
        """Calculate weighted final score."""
        w_tech, w_whale, w_sentiment = weights
        total_weight = w_tech + w_whale + w_sentiment

        if total_weight == 0:
            return 0.0

        # Normalize weights
        w_tech /= total_weight
        w_whale /= total_weight
        w_sentiment /= total_weight

        final_score = (
            tech_score * w_tech +
            whale_score * w_whale +
            sentiment_score * w_sentiment
        )

        return max(-1.0, min(1.0, final_score))

    def _calculate_confidence(
        self,
        tech_confidence: float,
        whale_confidence: float,
        sentiment_confidence: float,
        final_score: float,
        weights: tuple[float, float, float],
    ) -> float:
        """Calculate overall confidence in the decision."""
        w_tech, w_whale, w_sentiment = weights
        total_weight = w_tech + w_whale + w_sentiment

        if total_weight == 0:
            return 0.0

        # Weighted confidence
        weighted_conf = (
            tech_confidence * w_tech +
            whale_confidence * w_whale +
            sentiment_confidence * w_sentiment
        ) / total_weight

        # Bonus for strong signal
        signal_bonus = abs(final_score) * 0.2

        return min(1.0, weighted_conf + signal_bonus)

    def _check_trend_confirmation(
        self,
        action: TradeAction,
        trend: TrendDirection,
        confidence: float,
    ) -> tuple[bool, str]:
        """
        Check if trend confirms the action.

        Returns: (is_confirmed, reason)
        """
        if not self.settings.trading.require_trend_confirmation:
            return True, ""

        # High confidence overrides trend requirement
        if confidence >= self.settings.trading.high_confidence_threshold:
            return True, "High confidence override"

        # Check trend alignment
        if action == TradeAction.LONG:
            if trend == TrendDirection.BULLISH:
                return True, "Bullish trend confirmed"
            elif trend == TrendDirection.BEARISH:
                return False, "Trend is bearish, LONG rejected"

        elif action == TradeAction.SHORT:
            if trend == TrendDirection.BEARISH:
                return True, "Bearish trend confirmed"
            elif trend == TrendDirection.BULLISH:
                return False, "Trend is bullish, SHORT rejected"

        return True, "Neutral trend, proceeding"

    def _check_market_conditions(
        self,
        symbol: str,
        atr_pct: float,
    ) -> tuple[bool, str]:
        """
        Check if market conditions allow trading.

        Includes spread/fee awareness for scalping:
        - Basic spread check against max_spread_pct
        - Tighter spread check for entries (spread_entry_threshold_pct)
        - Liquidity check

        Returns: (is_ok, reason)
        """
        trading_config = self.settings.trading

        # Check ATR volatility
        if atr_pct > trading_config.max_atr_pct:
            return False, f"ATR too high: {atr_pct:.2f}% > {trading_config.max_atr_pct}%"

        # Check spread - basic limit
        conditions = self.market_data.get_market_conditions(symbol)
        spread_pct = conditions.get("spread_pct", 0)

        if spread_pct > trading_config.max_spread_pct:
            return False, f"Spread too high: {spread_pct:.4f}% > {trading_config.max_spread_pct}%"

        # SCALPING: Tighter spread check for entries
        # For small target profits (1-2%), spread + fees must not eat too much
        # Typical fee: 0.04% maker/taker = ~0.08% round trip
        # If spread_pct > spread_entry_threshold_pct, skip entry
        if trading_config.skip_entry_on_high_spread:
            if spread_pct > trading_config.spread_entry_threshold_pct:
                return False, (
                    f"Spread too high for scalping entry: {spread_pct:.4f}% > "
                    f"{trading_config.spread_entry_threshold_pct:.4f}%"
                )

        # Check liquidity
        bid_depth = conditions.get("bid_depth", 0)
        ask_depth = conditions.get("ask_depth", 0)
        total_liquidity = bid_depth + ask_depth

        if total_liquidity < trading_config.min_liquidity_usd:
            return False, f"Liquidity too low: ${total_liquidity:,.0f} < ${trading_config.min_liquidity_usd:,.0f}"

        return True, ""

    def _check_anti_flip_flop(
        self,
        symbol: str,
        proposed_action: TradeAction,
    ) -> tuple[bool, str]:
        """
        Prevent rapid position changes.

        Returns: (is_ok, reason)
        """
        last_time = self._last_action_time.get(symbol, 0)
        min_interval = self.settings.trading.min_time_between_trades_seconds

        elapsed = time.time() - last_time
        if elapsed < min_interval:
            remaining = min_interval - elapsed
            return False, f"Cooldown active: {remaining:.0f}s remaining"

        # Check for flip-flop (going opposite direction immediately)
        last_decision = self._last_decisions.get(symbol)
        if last_decision and elapsed < min_interval * 2:
            if (
                (last_decision.action == TradeAction.LONG and proposed_action == TradeAction.SHORT) or
                (last_decision.action == TradeAction.SHORT and proposed_action == TradeAction.LONG)
            ):
                return False, "Flip-flop protection: opposite signal too soon"

        return True, ""

    def _calculate_leverage(
        self,
        confidence: float,
        atr_pct: float,
    ) -> int:
        """Calculate appropriate leverage based on confidence and volatility."""
        trading = self.settings.trading

        # Base leverage on confidence
        base_leverage = trading.min_leverage + (
            (trading.max_leverage - trading.min_leverage) * confidence
        )

        # Reduce leverage for high volatility
        if atr_pct > 2.0:
            volatility_factor = max(0.5, 1 - (atr_pct - 2.0) / 10)
            base_leverage *= volatility_factor

        return max(trading.min_leverage, min(trading.max_leverage, int(base_leverage)))

    def _calculate_stop_loss(
        self,
        action: TradeAction,
        atr_pct: float,
    ) -> float:
        """Calculate stop-loss percentage."""
        risk = self.settings.risk

        # ATR-based stop-loss
        sl_pct = atr_pct * risk.atr_sl_multiplier

        # Cap at maximum
        sl_pct = min(sl_pct, risk.max_sl_pct)

        # Minimum stop-loss
        sl_pct = max(sl_pct, 0.5)

        return sl_pct

    def _calculate_take_profit(
        self,
        stop_loss_pct: float,
    ) -> float | None:
        """Calculate take-profit percentage based on risk-reward."""
        risk = self.settings.risk

        if not risk.enable_take_profit:
            return None

        return stop_loss_pct * risk.tp_r_multiple

    def _calculate_position_size_pct(
        self,
        confidence: float,
        leverage: int,
    ) -> float:
        """
        Calculate position size as percentage of risk budget.

        Higher confidence = larger position (within limits).
        """
        # Base size on confidence
        base_size = 50 + (confidence * 50)  # 50-100% of budget

        # Reduce for higher leverage
        leverage_factor = max(0.5, 1 - (leverage - 2) * 0.1)
        adjusted_size = base_size * leverage_factor

        return max(25, min(100, adjusted_size))

    def generate_decision(
        self,
        symbol: str,
        current_position_side: str | None = None,
    ) -> TradeDecision:
        """
        Generate trading decision for a symbol.

        Args:
            symbol: Trading pair
            current_position_side: Current position ("LONG", "SHORT", or None)

        Returns:
            TradeDecision with action and parameters
        """
        # Get signal weights
        weights = self._get_symbol_weights(symbol)

        # Generate technical signal
        primary_df = self.market_data.get_candles_df(
            symbol, self.settings.indicators.primary_timeframe
        )

        confirmation_dfs = []
        for tf in self.settings.indicators.confirmation_timeframes:
            df = self.market_data.get_candles_df(symbol, tf)
            if df is not None:
                confirmation_dfs.append(df)

        tech_signal = self.indicators.generate_signal(primary_df, confirmation_dfs)

        # Generate whale signal
        whale_signal = self.whales.generate_signal(symbol, current_position_side)

        # Generate sentiment signal
        sentiment_signal = self.sentiment.generate_signal(symbol)

        # Check for whale alert requiring defensive action
        # IMPORTANT: Return defensive_action, NOT FLAT - main loop handles the action
        if whale_signal.whale_alert_opposite and current_position_side:
            defensive_action = whale_signal.recommended_action
            logger.whale_alert(
                symbol=symbol,
                whale_score=whale_signal.whale_score,
                is_opposing=True,
                action_taken=defensive_action.value if defensive_action else "NONE",
            )
            return TradeDecision(
                symbol=symbol,
                action=TradeAction.HOLD,  # HOLD, not FLAT - let main loop handle defensive action
                confidence=whale_signal.whale_confidence,
                tech_score=tech_signal.tech_score,
                whale_score=whale_signal.whale_score,
                sentiment_score=sentiment_signal.sentiment_score,
                final_score=0.0,
                suggested_leverage=1,
                suggested_position_pct=0,
                stop_loss_pct=0,
                take_profit_pct=None,
                trend=tech_signal.trend,
                atr_pct=tech_signal.atr_pct,
                is_valid=True,
                rejection_reason=None,
                defensive_action=defensive_action,  # Explicit defensive action
            )

        # Calculate final score
        final_score = self._calculate_final_score(
            tech_signal.tech_score,
            whale_signal.whale_score,
            sentiment_signal.sentiment_score,
            weights,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            tech_signal.tech_confidence,
            whale_signal.whale_confidence,
            sentiment_signal.sentiment_confidence,
            final_score,
            weights,
        )

        # Determine action from final score
        entry_threshold = self.settings.trading.entry_threshold

        if final_score >= entry_threshold:
            proposed_action = TradeAction.LONG
        elif final_score <= -entry_threshold:
            proposed_action = TradeAction.SHORT
        else:
            proposed_action = TradeAction.FLAT

        # Validation checks
        rejection_reason = None

        # Check data validity
        if not tech_signal.is_valid:
            proposed_action = TradeAction.HOLD
            rejection_reason = tech_signal.invalidation_reason

        # Check market conditions
        if rejection_reason is None:
            ok, reason = self._check_market_conditions(symbol, tech_signal.atr_pct)
            if not ok:
                proposed_action = TradeAction.HOLD
                rejection_reason = reason

        # Check trend confirmation
        if rejection_reason is None and proposed_action in (TradeAction.LONG, TradeAction.SHORT):
            ok, reason = self._check_trend_confirmation(
                proposed_action, tech_signal.trend, confidence
            )
            if not ok:
                proposed_action = TradeAction.HOLD
                rejection_reason = reason

        # Check anti-flip-flop
        if rejection_reason is None and proposed_action in (TradeAction.LONG, TradeAction.SHORT):
            ok, reason = self._check_anti_flip_flop(symbol, proposed_action)
            if not ok:
                proposed_action = TradeAction.HOLD
                rejection_reason = reason

        # Calculate position parameters
        leverage = self._calculate_leverage(confidence, tech_signal.atr_pct)
        stop_loss_pct = self._calculate_stop_loss(proposed_action, tech_signal.atr_pct)
        take_profit_pct = self._calculate_take_profit(stop_loss_pct)
        position_pct = self._calculate_position_size_pct(confidence, leverage)

        decision = TradeDecision(
            symbol=symbol,
            action=proposed_action,
            confidence=confidence,
            tech_score=tech_signal.tech_score,
            whale_score=whale_signal.whale_score,
            sentiment_score=sentiment_signal.sentiment_score,
            final_score=final_score,
            suggested_leverage=leverage,
            suggested_position_pct=position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trend=tech_signal.trend,
            atr_pct=tech_signal.atr_pct,
            is_valid=rejection_reason is None,
            rejection_reason=rejection_reason,
        )

        # Log decision
        logger.trade_decision(
            symbol=symbol,
            action=decision.action.value,
            tech_score=decision.tech_score,
            whale_score=decision.whale_score,
            sentiment_score=decision.sentiment_score,
            final_score=decision.final_score,
            confidence=decision.confidence,
            reason=rejection_reason or "OK",
        )

        # Update tracking
        if proposed_action in (TradeAction.LONG, TradeAction.SHORT):
            self._last_decisions[symbol] = decision
            if decision.is_valid:
                self._last_action_time[symbol] = time.time()

        return decision

    def should_close_position(
        self,
        symbol: str,
        current_position_side: str,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        Check if existing position should be closed due to opposing strong signal.

        IMPORTANT: FLAT does NOT trigger close - only explicit opposite signals do.
        Defensive actions (whale alerts) are handled separately via defensive_action field.

        Returns: (should_close, reason)
        """
        # Generate fresh decision
        decision = self.generate_decision(symbol, current_position_side)

        # Check for strong opposite signal (reversal)
        # Only close if there's a valid strong signal in the opposite direction
        if decision.is_valid and decision.confidence >= self.settings.trading.entry_threshold:
            if current_position_side == "LONG" and decision.action == TradeAction.SHORT:
                return True, "Strong opposite signal: SHORT"
            elif current_position_side == "SHORT" and decision.action == TradeAction.LONG:
                return True, "Strong opposite signal: LONG"

        # FLAT does NOT close positions - it just means "no new entry"
        # Defensive actions are handled via decision.defensive_action in main loop

        return False, ""

    def clear_tracking(self, symbol: str | None = None) -> None:
        """Clear tracking data."""
        if symbol:
            self._last_decisions.pop(symbol, None)
            self._last_action_time.pop(symbol, None)
        else:
            self._last_decisions.clear()
            self._last_action_time.clear()
