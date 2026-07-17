from fastapi import Depends
from app.infrastructure.repositories.job_profile_repository import JobProfileRepository
from app.domain.jobs.job_profile import JobProfile
from app.core.models import JobProfileModel


from app.application.pipelines.job_pipeline import JobPipeline


class JobProfileRetrievalService:
    """Service responsible for retrieving stored JobProfiles from database."""

    def __init__(
        self,
        repository: JobProfileRepository = Depends(JobProfileRepository),
        job_pipeline: JobPipeline = Depends(JobPipeline),
    ):
        """Initialize the retrieval service with dependencies."""
        self._repository = repository
        self._job_pipeline = job_pipeline

    async def find_by_job_id(self, job_id: int) -> JobProfile | None:
        """Retrieve JobProfile domain model by Spring Boot job ID.

        Args:
            job_id (int): Spring Boot job entity ID.

        Returns:
            JobProfile | None: Domain job profile or None.
        """
        model = self._repository.find_by_job_id(job_id)
        if model:
            return self._map_to_domain(model)

        # Self-healing fallback: try to find raw job and process it dynamically
        from app.core.models import RawJobModel
        from app.domain.jobs.models.raw_job import RawJob
        
        db = self._repository.db
        raw_job_db = db.query(RawJobModel).filter(RawJobModel.id == job_id).first()
        if raw_job_db:
            try:
                raw_job = RawJob(
                    title=raw_job_db.title,
                    company=raw_job_db.company,
                    location=raw_job_db.location,
                    description=raw_job_db.description,
                    apply_url=raw_job_db.apply_url,
                    salary=raw_job_db.salary,
                    employment_type=raw_job_db.job_type,
                    source=raw_job_db.source
                )
                await self._job_pipeline.run(raw_job, job_id=job_id)
                model = self._repository.find_by_job_id(job_id)
                if model:
                    return self._map_to_domain(model)
            except Exception as e:
                import logging
                logging.getLogger("job_profile_retrieval_service").exception(
                    f"Failed to dynamically process raw job {job_id}"
                )
        return None

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
        profile.job_id = model.job_id
        profile.title = model.title
        profile.company = model.company
        profile.location = model.location
        profile.experience = model.experience
        profile.embedding = model.embedding
        return profile
