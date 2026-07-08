from sentence_transformers import SentenceTransformer
from app.core.config import Settings
from app.core.exceptions import ExternalServiceException
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Concrete embedding provider implementing SentenceTransformers."""

    def __init__(self, settings: Settings):
        """Initialize SentenceTransformer model based on settings.

        Args:
            settings (Settings): Active application configuration.
        """
        try:
            self._model = SentenceTransformer(settings.embedding_model)
        except Exception as exc:
            raise ExternalServiceException(
                message=f"Failed to initialize SentenceTransformer model: {str(exc)}",
                error_code="EMBEDDING_MODEL_INIT_FAILED",
            ) from exc

    def embed(self, text: str) -> list[float]:
        """Generate vector embedding for a single text input using local SentenceTransformer.

        Args:
            text (str): The text input to embed.

        Returns:
            list[float]: The generated embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        try:
            vector = self._model.encode(text.strip())
            # Convert numpy array to list of floats
            return [float(x) for x in vector]
        except Exception as exc:
            raise ExternalServiceException(
                message=f"SentenceTransformer embedding generation failed: {str(exc)}",
                error_code="EMBEDDING_GENERATION_FAILED",
            ) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text inputs using local SentenceTransformer.

        Args:
            texts (list[str]): List of text inputs to embed.

        Returns:
            list[list[float]]: List of generated embedding vectors.
        """
        if not texts:
            raise ValueError("Input list cannot be empty.")
        try:
            vectors = self._model.encode([t.strip() for t in texts])
            # Convert numpy arrays to list of lists of floats
            return [[float(x) for x in vec] for vec in vectors]
        except Exception as exc:
            raise ExternalServiceException(
                message=f"SentenceTransformer batch embedding generation failed: {str(exc)}",
                error_code="EMBEDDING_BATCH_GENERATION_FAILED",
            ) from exc

    async def health(self):
        """Perform a connection health check to the AI provider."""
        from app.domain.ai.providers.models import HealthResponse
        return HealthResponse(
            provider="sentence-transformers",
            healthy=True,
            message="Local SentenceTransformer loaded and ready.",
        )
