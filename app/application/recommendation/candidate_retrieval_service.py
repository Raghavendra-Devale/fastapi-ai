from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.application.jobs.job_profile_retrieval_service import JobProfileRetrievalService
from app.application.jobs.vector_retrieval_service import VectorRetrievalService


class CandidateRetrievalService:
    """Service responsible for retrieving matching jobs for a CandidateProfile."""

    def __init__(
        self,
        vector_service: VectorRetrievalService = Depends(VectorRetrievalService),
        retrieval_service: JobProfileRetrievalService = Depends(JobProfileRetrievalService),
    ):
        """Initialize retrieval service with database retrieval and vector clients."""
        self._vector_service = vector_service
        self._retrieval_service = retrieval_service

    async def retrieve_jobs(self, candidate_profile: CandidateProfile) -> list[JobProfile]:
        """Retrieve active JobProfiles from database for matchmaking.

        Uses vector similarity retrieval (top 200) if candidate embedding is available,
        otherwise falls back to retrieving all active profiles.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.

        Returns:
            list[JobProfile]: List of database job profiles.
        """
        if candidate_profile.embedding is not None and len(candidate_profile.embedding) > 0:
            return await self._vector_service.find_similar_jobs(
                embedding=candidate_profile.embedding,
                limit=200,
            )

        # Fallback to fetching all active jobs
        return self._retrieval_service.find_active()
