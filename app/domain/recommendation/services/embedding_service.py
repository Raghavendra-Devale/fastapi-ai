import time
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.providers.base import AIProvider
from app.providers.factory import get_ai_provider
from app.domain.recommendation.models.recommendation_request import JobDocument

logger = get_logger(__name__)


class EmbeddingService:
    """Domain service to generate vector embeddings for resumes and job documents."""

    def __init__(
        self,
        ai_provider: AIProvider = Depends(get_ai_provider),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the embedding service with configured AIProvider and application settings."""
        self._ai_provider = ai_provider
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
            response = await self._ai_provider.generate_embedding(resume_text.strip())
            return response.embedding
        except ExternalServiceException:
            # Re-raise directly to preserve error details and codes
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
            responses = await self._ai_provider.generate_embeddings(texts)
            # Ensure the number of returned embeddings matches the input list
            if len(responses) != len(jobs):
                raise ExternalServiceException(
                    message="AI provider returned an incomplete batch of embeddings.",
                    error_code="JOB_EMBEDDING_BATCH_MISMATCH",
                )
            return [res.embedding for res in responses]
        except ExternalServiceException:
            # Re-raise directly to preserve error details and codes
            raise
        except Exception as exc:
            raise ExternalServiceException(
                message=f"Failed to generate job embeddings: {str(exc)}",
                error_code="JOB_EMBEDDING_FAILED",
            ) from exc
