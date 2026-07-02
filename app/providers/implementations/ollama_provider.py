import time

from ollama import AsyncClient

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.providers.base import AIProvider
from app.providers.models import ChatResponse, EmbeddingResponse, HealthResponse

logger = get_logger("ollama_provider")


class OllamaProvider(AIProvider):
    """Concrete implementation of AIProvider utilizing the official Ollama SDK."""

    def __init__(self, settings: Settings):
        """Initialize the Ollama AsyncClient using configured base URL."""
        self._settings = settings
        self._client = AsyncClient(host=settings.ollama_base_url)

    async def health(self) -> HealthResponse:
        """Verify the Ollama provider connection health status."""
        try:
            # Query downloaded models to verify server responsiveness
            await self._client.list()
            return HealthResponse(
                provider="ollama",
                healthy=True,
                message="Ollama service is reachable and healthy.",
            )
        except Exception as exc:
            logger.error(
                event="provider_health_check_failed",
                provider="ollama",
                exc_info=exc,
            )
            return HealthResponse(
                provider="ollama",
                healthy=False,
                message=f"Ollama service is unreachable: {str(exc)}",
            )

    async def generate_embedding(self, text: str) -> EmbeddingResponse:
        """Generate vector embedding for a single text input."""
        if not text or not text.strip():
            raise ValidationException(
                error_code="INVALID_INPUT",
                message="Embedding text input cannot be blank.",
            )

        start_time = time.perf_counter()
        try:
            response = await self._client.embed(
                model=self._settings.embedding_model, input=text
            )

            if (
                not response
                or "embeddings" not in response
                or not response["embeddings"]
            ):
                raise ExternalServiceException(
                    error_code="EMBEDDING_FAILED",
                    message="Ollama returned an empty embedding response.",
                )

            embedding_vector = response["embeddings"][0]
            dimensions = len(embedding_vector)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                event="embedding_generated",
                provider="ollama",
                model=self._settings.embedding_model,
                duration_ms=round(duration_ms, 2),
                success=True,
            )

            return EmbeddingResponse(
                embedding=embedding_vector,
                model=self._settings.embedding_model,
                dimensions=dimensions,
            )
        except ValidationException:
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                event="embedding_generation_failed",
                provider="ollama",
                model=self._settings.embedding_model,
                duration_ms=round(duration_ms, 2),
                success=False,
                exc_info=exc,
            )

            error_code = "EMBEDDING_FAILED"
            # Map network/connection errors to service unavailable
            exc_str = str(exc).lower()
            if "connection" in exc_str or "connect" in exc_str or "dns" in exc_str:
                error_code = "OLLAMA_UNAVAILABLE"

            raise ExternalServiceException(
                error_code=error_code,
                message=f"Failed to generate embedding: {str(exc)}",
            )

    async def generate_embeddings(
        self, texts: list[str]
    ) -> list[EmbeddingResponse]:
        """Generate vector embeddings for a list of text inputs sequentially."""
        results: list[EmbeddingResponse] = []
        for text in texts:
            # Skip empty or blank strings
            if not text or not text.strip():
                continue
            res = await self.generate_embedding(text)
            results.append(res)
        return results

    async def chat(self, prompt: str) -> ChatResponse:
        """Submit a prompt context to the chat LLM model."""
        if not prompt or not prompt.strip():
            raise ValidationException(
                error_code="INVALID_INPUT",
                message="Chat prompt cannot be blank.",
            )

        start_time = time.perf_counter()
        try:
            response = await self._client.chat(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.get("message", {}).get("content", "")
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                event="chat_completed",
                provider="ollama",
                model=self._settings.llm_model,
                duration_ms=round(duration_ms, 2),
                success=True,
            )

            return ChatResponse(content=content, model=self._settings.llm_model)
        except ValidationException:
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                event="chat_completion_failed",
                provider="ollama",
                model=self._settings.llm_model,
                duration_ms=round(duration_ms, 2),
                success=False,
                exc_info=exc,
            )

            error_code = "CHAT_COMPLETION_FAILED"
            exc_str = str(exc).lower()
            if "connection" in exc_str or "connect" in exc_str or "dns" in exc_str:
                error_code = "OLLAMA_UNAVAILABLE"

            raise ExternalServiceException(
                error_code=error_code,
                message=f"Failed to generate chat response: {str(exc)}",
            )
