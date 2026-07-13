from fastapi import Depends
from app.infrastructure.repositories.job_profile_repository import JobProfileRepository
from app.domain.jobs.job_profile import JobProfile
from app.core.models import JobProfileModel


class JobProfileRetrievalService:
    """Service responsible for retrieving stored JobProfiles from database."""

    def __init__(
        self,
        repository: JobProfileRepository = Depends(JobProfileRepository),
    ):
        """Initialize the retrieval service with dependencies."""
        self._repository = repository

    def find_by_job_id(self, job_id: int) -> JobProfile | None:
        """Retrieve JobProfile domain model by Spring Boot job ID.

        Args:
            job_id (int): Spring Boot job entity ID.

        Returns:
            JobProfile | None: Domain job profile or None.
        """
        model = self._repository.find_by_job_id(job_id)
        return self._map_to_domain(model) if model else None

    def find_all(self) -> list[JobProfile]:
        """Retrieve all JobProfiles from the database.

        Returns:
            list[JobProfile]: List of domain job profiles.
        """
        models = self._repository.find_all()
        return [self._map_to_domain(m) for m in models]

    def find_active(self) -> list[JobProfile]:
        """Retrieve all active/processed JobProfiles from the database.

        Returns:
            list[JobProfile]: List of active domain job profiles.
        """
        models = self._repository.find_active()
        return [self._map_to_domain(m) for m in models]

    def find_by_provider(self, provider: str) -> list[JobProfile]:
        """Retrieve JobProfiles by provider.

        Args:
            provider (str): Scraper or source name.

        Returns:
            list[JobProfile]: List of domain job profiles.
        """
        models = self._repository.find_by_provider(provider)
        return [self._map_to_domain(m) for m in models]

    def _map_to_domain(self, model: JobProfileModel) -> JobProfile:
        """Map database SQLAlchemy model to domain Pydantic model.

        Args:
            model (JobProfileModel): SQLAlchemy model.

        Returns:
            JobProfile: Pydantic domain model.
        """
        profile = JobProfile.model_validate(model.profile_json)
        # Overwrite values from database columns to be consistent
        profile.id = str(model.id)
        profile.title = model.title
        profile.company = model.company
        profile.location = model.location
        profile.experience = model.experience
        return profile
