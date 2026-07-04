import time
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.providers.base import AIProvider
from app.providers.factory import get_ai_provider
from app.providers.models import EmbeddingResponse

logger = get_logger("embedding_service")


class EmbeddingService:
    """Application service to handle text embedding generation.

    Serves as the unified orchestration layer for generating vector embeddings
    across all domains in the AI Engine, shielding domain logic from raw AI Providers.

    Designed to be future-ready to support:
    - Caching of generated embeddings
    - Retry mechanisms for robust error handling
    - Batching of multiple text inputs
    - Fallback provider strategies
    - Performance and usage metrics collection
    - Rate limiting controls
    """

    def __init__(
        self,
        ai_provider: AIProvider = Depends(get_ai_provider),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the embedding service with the active provider and settings."""
        self._ai_provider = ai_provider
        self._settings = settings

    async def generate_embedding(self, text: str) -> EmbeddingResponse:
        """Validate, normalize, and generate embedding for a single text input.

        Args:
            text (str): Raw input text.

        Returns:
            EmbeddingResponse: Strongly typed response including embedding and model details.

        Raises:
            ValidationException: If input is empty, blank, or exceeds configured length.
        """
        # 1. Input Validation
        if not text:
            raise ValidationException("Embedding input cannot be empty.")

        if not text.strip():
            raise ValidationException("Embedding input cannot be blank.")

        if len(text) > self._settings.max_embedding_input_length:
            raise ValidationException(
                f"Embedding input length ({len(text)}) exceeds maximum allowed limit of "
                f"{self._settings.max_embedding_input_length} characters."
            )

        # 2. Input Normalization
        normalized = text.strip()

        # 3. Call AIProvider
        start_time = time.perf_counter()
        embedding_response = await self._ai_provider.generate_embedding(normalized)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Logging (model, dimensions, duration. Never log text, prompts, embeddings)
        logger.info(
            event="embedding_generated",
            model=embedding_response.model,
            dimensions=embedding_response.dimensions,
            duration=round(duration_ms, 2),
        )

        return embedding_response

    async def generate_embeddings_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Generate vector embeddings for a list of text inputs.

        Designed for future batch processing orchestration. Not implemented yet.
        """
        raise NotImplementedError("Batch embedding generation not implemented yet.")
