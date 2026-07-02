from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger("exception_handlers")


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for the FastAPI application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handle custom application exceptions and return standard JSON error."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        logger.error(
            event="application_exception",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            correlation_id=correlation_id,
            exc_info=exc,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "correlation_id": correlation_id,
                },
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle request validation errors and return standard 400 JSON error."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Format Pydantic validation errors nicely
        errors = exc.errors()
        message = "; ".join(
            [
                f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}"
                for err in errors
            ]
        )

        logger.error(
            event="validation_error",
            error_code="VALIDATION_ERROR",
            message=message,
            correlation_id=correlation_id,
            exc_info=exc,
        )

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": message,
                    "correlation_id": correlation_id,
                },
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ):
        """Handle standard HTTP exceptions (e.g. 404 Not Found) and return JSON error."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Determine a suitable error code based on status code
        code = "HTTP_ERROR"
        if exc.status_code == 404:
            code = "RESOURCE_NOT_FOUND"
        elif exc.status_code == 401:
            code = "UNAUTHORIZED"
        elif exc.status_code == 403:
            code = "FORBIDDEN"

        logger.warn(
            event="http_exception",
            status_code=exc.status_code,
            error_code=code,
            message=str(exc.detail),
            correlation_id=correlation_id,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "correlation_id": correlation_id,
                },
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all handler for unhandled exceptions to prevent stack trace leaks."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        logger.exception(
            event="unhandled_server_error",
            correlation_id=correlation_id,
            exc_info=exc,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                    "correlation_id": correlation_id,
                },
            },
            headers={"X-Correlation-ID": correlation_id},
        )
