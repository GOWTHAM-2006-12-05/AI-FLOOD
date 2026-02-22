"""
Centralised error handling — exception hierarchy + FastAPI handlers.

Provides:
    • Domain-specific exception classes
    • Consistent JSON error response format
    • Automatic logging of unhandled errors
    • Request context in error responses (non-production)

Usage:
    from backend.app.core.errors import (
        DisasterAPIError,
        NotFoundError,
        ValidationError,
        ExternalServiceError,
        ModelPredictionError,
        register_error_handlers,
    )

    raise NotFoundError("Earthquake", id="USP000ABC")
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Exception Hierarchy
# ═══════════════════════════════════════════════════════════════════════════

class DisasterAPIError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class NotFoundError(DisasterAPIError):
    """Resource not found (404)."""

    def __init__(self, resource: str, **identifiers: Any):
        details = {"resource": resource, **identifiers}
        super().__init__(
            message=f"{resource} not found",
            status_code=404,
            error_code="NOT_FOUND",
            details=details,
        )


class ValidationError(DisasterAPIError):
    """Input validation failed (422)."""

    def __init__(self, message: str, *, field: Optional[str] = None, **details: Any):
        d = {**details}
        if field:
            d["field"] = field
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=d,
        )


class ExternalServiceError(DisasterAPIError):
    """External API call failed (502)."""

    def __init__(self, service: str, message: str = "", **details: Any):
        super().__init__(
            message=f"External service '{service}' failed: {message}",
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, **details},
        )


class ModelPredictionError(DisasterAPIError):
    """ML model prediction failed (500)."""

    def __init__(self, model: str, message: str = "", **details: Any):
        super().__init__(
            message=f"Model '{model}' prediction failed: {message}",
            status_code=500,
            error_code="MODEL_ERROR",
            details={"model": model, **details},
        )


class RateLimitError(DisasterAPIError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after_seconds": retry_after},
        )


class AlertDeliveryError(DisasterAPIError):
    """Alert could not be delivered (500)."""

    def __init__(self, alert_id: str, channel: str, message: str = ""):
        super().__init__(
            message=f"Alert {alert_id} delivery failed on {channel}: {message}",
            status_code=500,
            error_code="ALERT_DELIVERY_ERROR",
            details={"alert_id": alert_id, "channel": channel},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Error Response Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""
    body: Dict[str, Any] = {
        "error": {
            "code": error_code,
            "message": message,
            "status": status_code,
        }
    }

    if details:
        body["error"]["details"] = details

    # Include request path in non-production
    if request and not settings.is_production:
        body["error"]["path"] = str(request.url.path)
        body["error"]["method"] = request.method

    return JSONResponse(status_code=status_code, content=body)


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Exception Handlers
# ═══════════════════════════════════════════════════════════════════════════

def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app."""

    @app.exception_handler(DisasterAPIError)
    async def handle_disaster_error(request: Request, exc: DisasterAPIError):
        logger.error(
            "API Error [%s]: %s | details=%s",
            exc.error_code, exc.message, exc.details,
        )
        return _build_error_response(
            exc.status_code, exc.error_code, exc.message,
            exc.details, request,
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError):
        logger.warning("ValueError: %s", exc)
        return _build_error_response(
            422, "VALIDATION_ERROR", str(exc), request=request,
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception):
        logger.critical(
            "Unhandled exception: %s\n%s",
            exc, traceback.format_exc(),
        )
        message = str(exc) if settings.DEBUG else "Internal server error"
        details = (
            {"traceback": traceback.format_exc().split("\n")}
            if settings.DEBUG else None
        )
        return _build_error_response(
            500, "INTERNAL_ERROR", message, details, request,
        )
