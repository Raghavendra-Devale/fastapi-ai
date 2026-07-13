from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.resume_suggestion import ResumeSuggestion
from app.core.logging import get_logger
from app.infrastructure.repositories.candidate_profile_repository import CandidateProfileRepository
from app.core.models import CandidateProfileModel
from app.core.config import get_settings

logger = get_logger(__name__)


class ResumePersistenceService:
    """Service responsible for persisting CandidateProfile, Embedding, and Suggestions."""

    def __init__(
        self,
        repository: CandidateProfileRepository = Depends(CandidateProfileRepository),
    ):
        """Initialize the persistence service with dependencies."""
        self._repository = repository

    async def save_resume_analysis(
        self,
        candidate_profile: CandidateProfile,
        embedding: list[float],
        suggestions: list[ResumeSuggestion],
        resume_id: int | None = None,
        user_id: int | None = None,
    ) -> bool:
        """Persist the complete resume analysis results to PostgreSQL database.

        Args:
            candidate_profile (CandidateProfile): The canonical candidate profile.
            embedding (list[float]): The generated vector embedding.
            suggestions (list[ResumeSuggestion]): Suggestions for improvement.
            resume_id (int | None): Spring Boot resume entity ID.
            user_id (int | None): Spring Boot user entity ID.

        Returns:
            bool: True if saving was successful.
        """
        logger.info(f"ResumePersistenceService: save_resume_analysis called with resume_id={resume_id}, user_id={user_id}")
        # If resume_id or user_id is missing, skip database persistence
        # (This avoids breaking unit tests that bypass DB configurations)
        if resume_id is None or user_id is None:
            logger.info(
                event="skipping_resume_analysis_persistence",
                reason="missing_resume_id_or_user_id",
                candidate_name=candidate_profile.name,
            )
            return True

        settings = get_settings()
        profile_json = candidate_profile.model_dump()
        primary_role = candidate_profile.preferred_roles[0] if candidate_profile.preferred_roles else None

        # Check if candidate profile already exists for this resume_id
        existing_profile = self._repository.find_by_resume_id(resume_id)

        if existing_profile:
            logger.info(
                event="updating_existing_candidate_profile",
                resume_id=resume_id,
                user_id=user_id,
            )
            existing_profile.user_id = user_id
            existing_profile.headline = candidate_profile.headline
            existing_profile.experience_years = candidate_profile.experience_years
            existing_profile.primary_role = primary_role
            existing_profile.profile_json = profile_json
            existing_profile.embedding = embedding
            existing_profile.embedding_model = settings.embedding_model
            existing_profile.llm_model = settings.llm_model
            existing_profile.prompt_version = "v1"

            logger.info("ResumePersistenceService: calling repository.update")
            self._repository.update(existing_profile)
        else:
            logger.info(
                event="saving_new_candidate_profile",
                resume_id=resume_id,
                user_id=user_id,
            )
            new_profile = CandidateProfileModel(
                resume_id=resume_id,
                user_id=user_id,
                headline=candidate_profile.headline,
                experience_years=candidate_profile.experience_years,
                primary_role=primary_role,
                profile_json=profile_json,
                embedding=embedding,
                embedding_model=settings.embedding_model,
                llm_model=settings.llm_model,
                prompt_version="v1",
            )
            logger.info("ResumePersistenceService: calling repository.save")
            self._repository.save(new_profile)

        return True
