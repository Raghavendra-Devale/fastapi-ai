import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger("request_logger")


class CorrelationAndLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for assigning Correlation IDs and logging HTTP requests.

    Ensures every request has a tracking correlation ID injected into log
    contextvars, logs request metadata on completion, and adds the
    correlation ID to response headers.
    """

    async def dispatch(self, request: Request, call_next):
        # Retrieve or generate Correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(
            uuid.uuid4()
        )
        request.state.correlation_id = correlation_id

        # Bind the Correlation ID to structlog context vars for log tracking
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            raise exc
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Log request execution metadata.
            # Using request.url.path avoids logging query strings that may contain secrets.
            logger.info(
                "HTTP Request",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
            )

        # Set the correlation ID header in the response
        response.headers["X-Correlation-ID"] = correlation_id
        return response
