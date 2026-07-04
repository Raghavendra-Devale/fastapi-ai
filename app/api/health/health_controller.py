from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_embedding_provider, get_llm_provider

router = APIRouter()


@router.get(
    "",
    summary="Get operational health check status",
    description="Check the operational status of the AI Engine and its configured providers.",
)
async def get_health(
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    llm_provider: LLMProvider = Depends(get_llm_provider),
):
    """Retrieve operational status, active embedding provider name, active LLM provider name, and version."""
    return {
        "status": "UP",
        "embeddingProvider": embedding_provider.__class__.__name__,
        "llmProvider": llm_provider.__class__.__name__,
        "version": settings.version,
    }
