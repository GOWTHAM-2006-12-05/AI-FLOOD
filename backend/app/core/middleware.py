"""
Request middleware — logging, timing, correlation IDs.

Provides:
    • X-Request-ID header injection (correlation ID)
    • Request/response timing (X-Process-Time header)
    • Structured log entry per request
    • Request context for downstream log enrichment
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.logging_config import set_request_context

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request with timing, inject correlation ID.

    Request log entry includes:
        - method, path, status_code
        - duration_ms
        - client IP
        - request_id (also returned in X-Request-ID response header)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Set context for downstream loggers
        set_request_context(
            request_id=request_id,
            client_ip=client_ip,
            endpoint=path,
            method=request.method,
        )

        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "%s %s → 500 (%.1fms) [%s]",
                request.method, path, duration_ms, client_ip,
                extra={"duration_ms": duration_ms, "status_code": 500},
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"

        # Log (skip health/docs/favicon spam)
        if not any(path.startswith(p) for p in ("/docs", "/redoc", "/openapi", "/favicon")):
            log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            logger.log(
                log_level,
                "%s %s → %d (%.1fms) [%s]",
                request.method, path, response.status_code,
                duration_ms, client_ip,
                extra={
                    "duration_ms": duration_ms,
                    "status_code": response.status_code,
                    "endpoint": path,
                },
            )

        # Clear context
        set_request_context()

        return response
