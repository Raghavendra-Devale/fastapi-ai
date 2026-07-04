from fastapi import Depends

from app.core.config import Settings, get_settings
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.implementations.sentence_transformer_provider import SentenceTransformerProvider

from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.implementations.ollama_provider import OllamaProvider

_embedding_provider_instance = None
_llm_provider_instance = None


def get_embedding_provider(
    settings: Settings = Depends(get_settings),
) -> EmbeddingProvider:
    """FastAPI dependency getter for resolving the concrete EmbeddingProvider implementation.

    Decouples the downstream consumers from concrete sentence transformer libraries.
    """
    global _embedding_provider_instance
    if _embedding_provider_instance is None:
        _embedding_provider_instance = SentenceTransformerProvider(settings)
    return _embedding_provider_instance


def get_llm_provider(
    settings: Settings = Depends(get_settings),
) -> LLMProvider:
    """FastAPI dependency getter for resolving the concrete LLMProvider implementation.

    Decouples the downstream consumers from concrete Large Language Model APIs.
    """
    global _llm_provider_instance
    if _llm_provider_instance is None:
        _llm_provider_instance = OllamaProvider(settings)
    return _llm_provider_instance

