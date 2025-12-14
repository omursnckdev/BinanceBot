"""
Structured JSON logging with rotation support.

Features:
- JSON formatted logs for production
- Console + file output
- Rotating file handler
- Context enrichment for trading decisions
- Secret masking
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class SecretMasker:
    """Masks sensitive data in log messages."""

    SENSITIVE_KEYS = {
        "api_key",
        "api_secret",
        "password",
        "secret",
        "token",
        "binance_api_key",
        "binance_api_secret",
        "news_api_key",
    }

    @classmethod
    def mask_dict(cls, data: dict[str, Any], visible_chars: int = 4) -> dict[str, Any]:
        """Recursively mask sensitive values in a dictionary."""
        result = {}
        for key, value in data.items():
            lower_key = key.lower()
            if any(sensitive in lower_key for sensitive in cls.SENSITIVE_KEYS):
                if isinstance(value, str) and len(value) > visible_chars:
                    result[key] = value[:visible_chars] + "*" * (len(value) - visible_chars)
                elif isinstance(value, str):
                    result[key] = "*" * len(value)
                else:
                    result[key] = "***MASKED***"
            elif isinstance(value, dict):
                result[key] = cls.mask_dict(value, visible_chars)
            elif isinstance(value, list):
                result[key] = [
                    cls.mask_dict(item, visible_chars) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter with trading context support.
    """

    def __init__(self, include_timestamp: bool = True) -> None:
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add location info for errors
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra context (trading decisions, scores, etc.)
        extra_keys = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        for key in extra_keys:
            value = getattr(record, key)
            if isinstance(value, dict):
                value = SecretMasker.mask_dict(value)
            log_data[key] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter with colors.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Build the base message
        base_msg = f"{color}[{timestamp}] [{record.levelname:8}] {record.name}: {record.getMessage()}{self.RESET}"

        # Add extra context for trading decisions
        extra_parts = []
        extra_keys = {"symbol", "action", "tech_score", "whale_score", "sentiment_score", "final_score"}

        for key in extra_keys:
            if hasattr(record, key):
                value = getattr(record, key)
                if isinstance(value, float):
                    extra_parts.append(f"{key}={value:.3f}")
                else:
                    extra_parts.append(f"{key}={value}")

        if extra_parts:
            base_msg += f" | {' '.join(extra_parts)}"

        # Add exception info
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"

        return base_msg


class TradingLogger(logging.Logger):
    """
    Extended logger with trading-specific methods.
    """

    def trade_decision(
        self,
        symbol: str,
        action: str,
        tech_score: float,
        whale_score: float,
        sentiment_score: float,
        final_score: float,
        confidence: float,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Log a trading decision with all signal components."""
        self.info(
            f"Trade decision for {symbol}: {action}",
            extra={
                "symbol": symbol,
                "action": action,
                "tech_score": tech_score,
                "whale_score": whale_score,
                "sentiment_score": sentiment_score,
                "final_score": final_score,
                "confidence": confidence,
                "reason": reason,
                **kwargs,
            },
        )

    def order_event(
        self,
        event_type: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an order event."""
        self.info(
            f"Order {event_type}: {symbol} {side} {quantity}",
            extra={
                "event_type": event_type,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                **kwargs,
            },
        )

    def risk_event(
        self,
        event_type: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Log a risk management event."""
        level = logging.WARNING if "kill" in event_type.lower() else logging.INFO
        self.log(
            level,
            f"Risk event: {event_type} - {reason}",
            extra={
                "event_type": event_type,
                "reason": reason,
                **kwargs,
            },
        )

    def whale_alert(
        self,
        symbol: str,
        whale_score: float,
        is_opposing: bool,
        action_taken: str,
        **kwargs: Any,
    ) -> None:
        """Log a whale activity alert."""
        self.warning(
            f"Whale alert on {symbol}: score={whale_score:.3f}, opposing={is_opposing}",
            extra={
                "symbol": symbol,
                "whale_score": whale_score,
                "is_opposing": is_opposing,
                "action_taken": action_taken,
                **kwargs,
            },
        )


# Register custom logger class
logging.setLoggerClass(TradingLogger)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/trading.log",
    use_json: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Set up logging with console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        use_json: Whether to use JSON formatting for file output
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (JSON or plain text)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    if use_json:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def get_logger(name: str) -> TradingLogger:
    """Get a TradingLogger instance."""
    return logging.getLogger(name)  # type: ignore


# Context manager for temporary log level changes
class LogLevelContext:
    """Temporarily change log level."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self) -> None:
        self.logger.setLevel(self.new_level)

    def __exit__(self, *args: Any) -> None:
        self.logger.setLevel(self.old_level)
