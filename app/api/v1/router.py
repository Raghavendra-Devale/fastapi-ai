from fastapi import APIRouter

from app.api.v1.endpoints import health, resume

api_router = APIRouter()

# Register health check endpoints
api_router.include_router(health.router, tags=["health"])

# Register resume endpoints
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])

