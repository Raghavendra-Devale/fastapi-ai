import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger("request_logger")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware for assigning a tracking Correlation ID to each request."""

    async def dispatch(self, request: Request, call_next):
        # Retrieve incoming correlation ID or generate a new UUID
        correlation_id = request.headers.get("X-Correlation-ID") or str(
            uuid.uuid4()
        )
        request.state.correlation_id = correlation_id

        # Bind the Correlation ID to structlog context vars for log tracking
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        response = await call_next(request)

        # Set the correlation ID header in the response
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for profiling and logging incoming HTTP request execution."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            raise exc
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            correlation_id = getattr(request.state, "correlation_id", "unknown")

            # Log request execution metadata using structured event naming
            logger.info(
                event="request_completed",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
            )

        return response
