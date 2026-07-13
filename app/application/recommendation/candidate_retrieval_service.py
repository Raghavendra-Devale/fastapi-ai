from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.application.jobs.job_profile_retrieval_service import JobProfileRetrievalService


class CandidateRetrievalService:
    """Service responsible for retrieving matching jobs for a CandidateProfile."""

    def __init__(
        self,
        retrieval_service: JobProfileRetrievalService = Depends(JobProfileRetrievalService),
    ):
        """Initialize retrieval service with database retrieval client."""
        self._retrieval_service = retrieval_service

    async def retrieve_jobs(self, candidate_profile: CandidateProfile) -> list[JobProfile]:
        """Retrieve active JobProfiles from database for matchmaking.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.

        Returns:
            list[JobProfile]: List of database job profiles.
        """
        return self._retrieval_service.find_active()
