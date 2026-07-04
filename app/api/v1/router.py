from fastapi import APIRouter

from app.api.v1.endpoints import health, resume
from app.api.recommendations import recommendation_controller

api_router = APIRouter()

# Register health check endpoints
api_router.include_router(health.router, tags=["health"])

# Register resume endpoints
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])

# Register recommendations endpoints
api_router.include_router(recommendation_controller.router, prefix="/recommendations", tags=["recommendations"])


