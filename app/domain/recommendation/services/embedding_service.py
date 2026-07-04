import time
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.dependencies import get_embedding_provider
from app.domain.recommendation.models.recommendation_request import JobDocument

logger = get_logger(__name__)


class EmbeddingService:
    """Domain service to generate vector embeddings for resumes and job documents.

    Acts as the service layer coordinator, delegating the physical model operations
    to the decoupled EmbeddingProvider interface.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the embedding service with the EmbeddingProvider interface and settings."""
        self._embedding_provider = embedding_provider
        self._settings = settings

    async def embed_resume(self, resume_text: str) -> list[float]:
        """Generate a vector embedding for normalized resume text.

        Args:
            resume_text (str): The candidate's resume text.

        Returns:
            list[float]: The generated embedding vector.

        Raises:
            ValidationException: If input is empty.
            ExternalServiceException: If the embedding generation fails.
        """
        if not resume_text or not resume_text.strip():
            raise ValidationException("Resume text cannot be empty or blank.")

        try:
            # Delegate to decoupled embedding provider
            return self._embedding_provider.embed(resume_text.strip())
        except ExternalServiceException:
            raise
        except Exception as exc:
            raise ExternalServiceException(
                message=f"Failed to generate resume embedding: {str(exc)}",
                error_code="RESUME_EMBEDDING_FAILED",
            ) from exc

    async def embed_jobs(self, jobs: list[JobDocument]) -> list[list[float]]:
        """Generate vector embeddings for a list of job documents.

        Args:
            jobs (list[JobDocument]): The list of JobDocument objects.

        Returns:
            list[list[float]]: The list of generated embedding vectors.

        Raises:
            ValidationException: If input list is empty.
            ExternalServiceException: If the embedding generation fails.
        """
        if not jobs:
            raise ValidationException("Job list cannot be empty.")

        texts = []
        for job in jobs:
            if not job.title or not job.description:
                raise ValidationException("Job title and description are required for embedding.")
            
            # Format job fields into a clean text representation
            text = (
                f"Title: {job.title.strip()}\n"
                f"Company: {job.company.strip()}\n"
                f"Location: {job.location.strip() if job.location else ''}\n"
                f"Employment Type: {job.employment_type.strip() if job.employment_type else ''}\n"
                f"Description: {job.description.strip()}"
            )
            texts.append(text)

        try:
            # Delegate to decoupled embedding provider
            vectors = self._embedding_provider.embed_batch(texts)
            # Ensure the number of returned embeddings matches the input list
            if len(vectors) != len(jobs):
                raise ExternalServiceException(
                    message="Embedding provider returned an incomplete batch of vectors.",
                    error_code="JOB_EMBEDDING_BATCH_MISMATCH",
                )
            return vectors
        except ExternalServiceException:
            raise
        except Exception as exc:
            raise ExternalServiceException(
                message=f"Failed to generate job embeddings: {str(exc)}",
                error_code="JOB_EMBEDDING_FAILED",
            ) from exc
