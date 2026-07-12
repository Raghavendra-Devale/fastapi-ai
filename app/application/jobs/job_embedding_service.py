from fastapi import Depends
from app.domain.jobs.job_profile import JobProfile
from app.application.ai.embedding_service import EmbeddingService


class JobEmbeddingService:
    """Service responsible for generating vector embeddings for jobs from JobProfile objects."""

    def __init__(self, embedding_service: EmbeddingService = Depends(EmbeddingService)):
        self._embedding_service = embedding_service

    async def generate_embedding(self, job_profile: JobProfile) -> list[float]:
        """Generate a vector embedding from a canonical JobProfile.

        Args:
            job_profile (JobProfile): Standardized canonical job profile.

        Returns:
            list[float]: The generated embedding vector.
        """
        # Formulate a clean text representation from the JobProfile fields to embed
        # (This ensures we never embed raw provider JSON, complying with Section 7)
        text_to_embed = (
            f"Job Title: {job_profile.title}\n"
            f"Company: {job_profile.company}\n"
            f"Summary: {job_profile.summary or ''}\n"
            f"Required Skills: {', '.join(job_profile.required_skills)}\n"
            f"Preferred Skills: {', '.join(job_profile.preferred_skills)}\n"
            f"Responsibilities: {'; '.join(job_profile.responsibilities)}\n"
            f"Requirements: {'; '.join(job_profile.requirements)}"
        )
        response = await self._embedding_service.generate_embedding(text_to_embed)
        return response.embedding
