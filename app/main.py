from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.core.logging import get_logger, setup_logging
from app.core.middleware import CorrelationAndLoggingMiddleware

# Initialize logging configuration immediately on module load
setup_logging()

logger = get_logger("app_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to log application startup and shutdown events."""
    settings = get_settings()
    logger.info(
        "Application starting up",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        provider=settings.provider,
    )
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="JobRadar AI Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Register Middleware
app.add_middleware(CorrelationAndLoggingMiddleware)


# Exception Handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions and return standard JSON error."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.error(
        "Application exception occurred",
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
        [f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}" for err in errors]
    )

    logger.error(
        "Request validation failed",
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
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
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
        "HTTP exception occurred",
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
        "Unhandled server error occurred",
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


# Root Endpoint
@app.get("/")
def root(settings: Settings = Depends(get_settings)):
    """Return the health and basic metadata of the service."""
    logger.info("Health check endpoint accessed")
    return {
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "status": "UP",
    }