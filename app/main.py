from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import (
    CorrelationIdMiddleware,
    RequestLoggingMiddleware,
)

from app.domain.ai.providers.implementations.ollama_provider import OllamaProvider
from app.domain.ai.providers.implementations.sentence_transformer_provider import SentenceTransformerProvider

# Fetch settings and initialize logging configuration on module load
settings = get_settings()
setup_logging(settings)

logger = get_logger("app_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to log application startup and shutdown events."""
    logger.info(
        event="application_startup",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        provider=settings.provider,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )

    # Perform startup connection health validation on the configured provider
    try:
        # Instantiate providers for startup check
        llm_provider = OllamaProvider(settings)
        emb_provider = SentenceTransformerProvider(settings)
        
        llm_health = await llm_provider.health()
        emb_health = await emb_provider.health()
        
        if llm_health.healthy and emb_health.healthy:
            logger.info(
                event="provider_health_check_success",
                provider=settings.provider,
                message=f"LLM: {llm_health.message} | Embedding: {emb_health.message}",
            )
        else:
            logger.warn(
                event="provider_health_check_warning",
                provider=settings.provider,
                message=f"LLM: {llm_health.message} | Embedding: {emb_health.message}",
            )
    except Exception as exc:
        logger.warn(
            event="provider_health_check_warning",
            provider=settings.provider,
            message=f"Startup health check failed: {str(exc)}",
        )

    yield
    logger.info(event="application_shutdown")


app = FastAPI(
    title="JobRadar AI Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Register Middlewares (CorrelationIdMiddleware wraps RequestLoggingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIdMiddleware)

# Register Global Exception Handlers
register_exception_handlers(app)

# Register API Router
app.include_router(api_router, prefix="/api/v1")


# Root Endpoint
@app.get("/")
def root(settings: Settings = Depends(get_settings)):
    """Return the health and basic metadata of the service."""
    logger.info(event="health_check_accessed")
    return {
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "status": "UP",
    }