"""
Structured logging configuration.

Provides:
    • JSON-formatted logs for production (machine-parseable)
    • Pretty console logs for development (human-readable)
    • Request-scoped context (request_id, user_ip, endpoint)
    • Performance timing middleware
    • Correlation ID propagation

Usage:
    from backend.app.core.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"lat": 13.08, "lon": 80.27})
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from backend.app.core.config import settings

# ── Context variable for request-scoped data ──
_request_context: ContextVar[Dict[str, Any]] = ContextVar(
    "request_context", default={}
)


def set_request_context(**kwargs: Any) -> None:
    """Set request-scoped log context (call from middleware)."""
    _request_context.set(kwargs)


def get_request_context() -> Dict[str, Any]:
    """Get current request context."""
    return _request_context.get()


# ── JSON Formatter (Production) ──

class JSONFormatter(logging.Formatter):
    """Machine-parseable JSON log output for log aggregation (ELK, Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach request context
        ctx = get_request_context()
        if ctx:
            log_entry["context"] = ctx

        # Attach any extra fields
        for key in ("lat", "lon", "risk_score", "alert_id", "recipient_count",
                     "channel", "duration_ms", "status_code", "endpoint"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


# ── Pretty Formatter (Development) ──

class PrettyFormatter(logging.Formatter):
    """Coloured human-readable format for local development."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()

        ctx = get_request_context()
        ctx_str = ""
        if ctx.get("request_id"):
            ctx_str = f" [{ctx['request_id'][:8]}]"

        formatted = (
            f"{color}{ts} {record.levelname:8s}{self.RESET}"
            f"{ctx_str} {record.name}: {msg}"
        )

        if record.exc_info and record.exc_info[1]:
            formatted += f"\n  {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"

        return formatted


# ── Setup ──

def setup_logging() -> None:
    """Configure logging based on environment."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if settings.is_production:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PrettyFormatter())

    root.addHandler(handler)

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger — call once per module."""
    return logging.getLogger(name)
