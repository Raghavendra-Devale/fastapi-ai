import time

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.ai_health_service import AIHealthService

router = APIRouter()
logger = get_logger("health_endpoint")


@router.get("/health", summary="Get detailed health check status")
async def get_health(
    settings: Settings = Depends(get_settings),
    health_service: AIHealthService = Depends(),
):
    """Retrieve application health metrics and integrated provider status."""
    start_time = time.perf_counter()

    provider_health = await health_service.check_provider_health()

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Log health check operation details
    logger.info(
        event="health_check_completed",
        provider_healthy=provider_health.healthy,
        duration_ms=round(duration_ms, 2),
    )

    return {
        "status": "UP",
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "dependencies": {
            "provider": {
                "healthy": provider_health.healthy,
                "provider": provider_health.provider,
                "message": provider_health.message,
            }
        },
    }
