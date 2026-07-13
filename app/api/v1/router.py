from fastapi import APIRouter

from app.api.v1.endpoints import resume, job
from app.api.recommendations import recommendation_controller
from app.api.health import health_controller

api_router = APIRouter()

# Register health check endpoints
api_router.include_router(health_controller.router, prefix="/health", tags=["health"])


# Register resume endpoints
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])

# Register job endpoints
api_router.include_router(job.router, prefix="/job", tags=["job"])

# Register recommendations endpoints
api_router.include_router(recommendation_controller.router, prefix="/recommendations", tags=["recommendations"])


