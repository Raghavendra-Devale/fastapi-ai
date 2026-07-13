from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.models import JobProfileModel


class JobProfileRepository:
    """Repository handling database operations for JobProfileModel."""

    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def save(self, profile: JobProfileModel) -> JobProfileModel:
        """Create a new JobProfile in the database.

        Args:
            profile (JobProfileModel): Database entity model.

        Returns:
            JobProfileModel: Refreshed database entity model.
        """
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def find_by_job_id(self, job_id: int) -> JobProfileModel | None:
        """Find a JobProfile by Spring Boot job ID.

        Args:
            job_id (int): Spring Boot job entity ID.

        Returns:
            JobProfileModel | None: Matching profile or None.
        """
        return self.db.query(JobProfileModel).filter(
            JobProfileModel.job_id == job_id
        ).first()

    def update(self, profile: JobProfileModel) -> JobProfileModel:
        """Update an existing JobProfile in the database.

        Args:
            profile (JobProfileModel): Database entity model to update.

        Returns:
            JobProfileModel: Refreshed database entity model.
        """
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def delete(self, profile: JobProfileModel) -> None:
        """Delete a JobProfile from the database.

        Args:
            profile (JobProfileModel): Database entity model to delete.
        """
        self.db.delete(profile)
        self.db.commit()

    def find_all(self) -> list[JobProfileModel]:
        """Find all JobProfiles in the database.

        Returns:
            list[JobProfileModel]: List of all job profiles.
        """
        return self.db.query(JobProfileModel).all()

    def find_active(self) -> list[JobProfileModel]:
        """Find all active/processed JobProfiles in the database (i.e. having embeddings).

        Returns:
            list[JobProfileModel]: List of active job profiles.
        """
        return self.db.query(JobProfileModel).filter(
            JobProfileModel.embedding.isnot(None)
        ).all()

    def find_by_provider(self, provider: str) -> list[JobProfileModel]:
        """Find JobProfiles by external scraper/source provider.

        Args:
            provider (str): Scraper or source name.

        Returns:
            list[JobProfileModel]: List of matching job profiles.
        """
        return self.db.query(JobProfileModel).filter(
            JobProfileModel.provider == provider
        ).all()

    def find_similar_jobs(
        self,
        embedding: list[float],
        limit: int = 200,
    ) -> list[JobProfileModel]:
        """Find top N similar job profiles using pgvector cosine distance.

        Args:
            embedding (list[float]): The candidate query embedding.
            limit (int): The maximum number of jobs to return.

        Returns:
            list[JobProfileModel]: List of similar job profiles.
        """
        # cosine_distance is 1 - cosine_similarity. Lower distance means higher similarity.
        return (
            self.db.query(JobProfileModel)
            .filter(JobProfileModel.embedding.isnot(None))
            .order_by(JobProfileModel.embedding.cosine_distance(embedding))
            .limit(limit)
            .all()
        )
