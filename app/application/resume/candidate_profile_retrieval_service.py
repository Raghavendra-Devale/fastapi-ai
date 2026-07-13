from fastapi import Depends, HTTPException
from app.infrastructure.repositories.candidate_profile_repository import CandidateProfileRepository
from app.domain.resume.candidate_profile import CandidateProfile

class CandidateProfileRetrievalService:
    """Service responsible for retrieving stored CandidateProfiles from database."""

    def __init__(
        self,
        repository: CandidateProfileRepository = Depends(CandidateProfileRepository),
    ):
        """Initialize the retrieval service with dependency-injected repository."""
        self._repository = repository

    def get_candidate_profile(self, candidate_profile_id: str) -> CandidateProfile:
        """Retrieve candidate profile by database UUID or Spring Boot resume ID.

        Args:
            candidate_profile_id (str): UUID or numeric resume ID.

        Returns:
            CandidateProfile: Domain model representation of the profile.
        """
        model = None
        # If candidate_profile_id is numeric, lookup by resume_id
        if isinstance(candidate_profile_id, str) and candidate_profile_id.isdigit():
            resume_id = int(candidate_profile_id)
            model = self._repository.find_by_resume_id(resume_id)

        # Fallback to lookup by UUID primary key
        if not model:
            model = self._repository.find_by_id(candidate_profile_id)

        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Candidate profile with ID or resume_id '{candidate_profile_id}' not found."
            )
        
        # Map database profile JSON to Pydantic domain model
        profile = CandidateProfile.model_validate(model.profile_json)
        profile.id = str(model.id)
        # Store vector embedding in the domain model
        profile.embedding = model.embedding
        return profile
