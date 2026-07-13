from fastapi import Depends
from app.infrastructure.repositories.job_profile_repository import JobProfileRepository
from app.application.jobs.job_profile_retrieval_service import JobProfileRetrievalService
from app.domain.jobs.job_profile import JobProfile


class VectorRetrievalService:
    """Service responsible for performing vector similarity retrieval of JobProfiles."""

    def __init__(
        self,
        repository: JobProfileRepository = Depends(JobProfileRepository),
        retrieval_service: JobProfileRetrievalService = Depends(JobProfileRetrievalService),
    ):
        """Initialize the vector retrieval service with dependencies."""
        self._repository = repository
        self._retrieval_service = retrieval_service

    async def find_similar_jobs(
        self,
        embedding: list[float],
        limit: int = 200,
    ) -> list[JobProfile]:
        """Find the top similar JobProfiles based on vector embedding cosine similarity.

        Args:
            embedding (list[float]): The candidate query embedding.
            limit (int): The maximum number of jobs to return.

        Returns:
            list[JobProfile]: Ranked list of domain job profiles matching the embedding.
        """
        if embedding is None or len(embedding) == 0:
            return []

        models = self._repository.find_similar_jobs(embedding, limit=limit)
        return [self._retrieval_service._map_to_domain(m) for m in models]
