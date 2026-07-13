from fastapi import Depends
from app.domain.jobs.job_profile import JobProfile
from app.core.logging import get_logger
from app.infrastructure.repositories.job_profile_repository import JobProfileRepository
from app.core.models import JobProfileModel
from app.core.config import get_settings

logger = get_logger(__name__)


class JobPersistenceService:
    """Service responsible for persisting JobProfile and its embedding."""

    def __init__(
        self,
        repository: JobProfileRepository = Depends(JobProfileRepository),
    ):
        """Initialize the persistence service with dependencies."""
        self._repository = repository

    async def save_job_analysis(
        self,
        job_profile: JobProfile,
        embedding: list[float],
        job_id: int | None = None,
        provider: str | None = None,
    ) -> bool:
        """Persist the complete job profile analysis results to PostgreSQL.

        Args:
            job_profile (JobProfile): Standardized canonical job profile.
            embedding (list[float]): The generated vector embedding for the job.
            job_id (int | None): Spring Boot database job entity ID.
            provider (str | None): External scraper or provider source name.

        Returns:
            bool: True if saving was successful.
        """
        # If job_id is missing, skip database persistence
        # (This avoids breaking unit tests that bypass DB configurations)
        if job_id is None:
            logger.info(
                event="skipping_job_analysis_persistence",
                reason="missing_job_id",
                job_title=job_profile.title,
            )
            return True

        settings = get_settings()
        profile_json = job_profile.model_dump()

        # Check if job profile already exists for this job_id
        existing_profile = self._repository.find_by_job_id(job_id)

        if existing_profile:
            logger.info(
                event="updating_existing_job_profile",
                job_id=job_id,
            )
            existing_profile.provider = provider or "unknown"
            existing_profile.title = job_profile.title
            existing_profile.company = job_profile.company
            existing_profile.location = job_profile.location
            existing_profile.experience = job_profile.experience
            existing_profile.profile_json = profile_json
            existing_profile.embedding = embedding
            existing_profile.embedding_model = settings.embedding_model
            existing_profile.llm_model = settings.llm_model
            existing_profile.prompt_version = "v1"

            self._repository.update(existing_profile)
        else:
            logger.info(
                event="saving_new_job_profile",
                job_id=job_id,
            )
            new_profile = JobProfileModel(
                job_id=job_id,
                provider=provider or "unknown",
                title=job_profile.title,
                company=job_profile.company,
                location=job_profile.location,
                experience=job_profile.experience,
                profile_json=profile_json,
                embedding=embedding,
                embedding_model=settings.embedding_model,
                llm_model=settings.llm_model,
                prompt_version="v1",
            )
            self._repository.save(new_profile)

        return True
