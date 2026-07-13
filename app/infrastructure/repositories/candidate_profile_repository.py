from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.models import CandidateProfileModel
from app.core.logging import get_logger

logger = get_logger("candidate_profile_repository")


class CandidateProfileRepository:
    """Repository handling database operations for CandidateProfileModel."""

    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def save(self, profile: CandidateProfileModel) -> CandidateProfileModel:
        """Create a new CandidateProfile in the database.

        Args:
            profile (CandidateProfileModel): Database entity model.

        Returns:
            CandidateProfileModel: Refreshed database entity model.
        """
        logger.info("CandidateProfileRepository: db.add and committing profile")
        self.db.add(profile)
        self.db.commit()
        logger.info("CandidateProfileRepository: db.commit completed, refreshing profile")
        self.db.refresh(profile)
        logger.info("CandidateProfileRepository: db.refresh completed successfully")
        return profile

    def find_by_resume_id(self, resume_id: int) -> CandidateProfileModel | None:
        """Find a CandidateProfile by Spring Boot resume ID.

        Args:
            resume_id (int): Spring Boot resume entity ID.

        Returns:
            CandidateProfileModel | None: Matching profile or None.
        """
        return self.db.query(CandidateProfileModel).filter(
            CandidateProfileModel.resume_id == resume_id
        ).first()

    def find_by_user_id(self, user_id: int) -> list[CandidateProfileModel]:
        """Find all CandidateProfiles by Spring Boot user ID.

        Args:
            user_id (int): Spring Boot user entity ID.

        Returns:
            list[CandidateProfileModel]: List of profiles matching the user.
        """
        return self.db.query(CandidateProfileModel).filter(
            CandidateProfileModel.user_id == user_id
        ).all()

    def update(self, profile: CandidateProfileModel) -> CandidateProfileModel:
        """Update an existing CandidateProfile in the database.

        Args:
            profile (CandidateProfileModel): Database entity model to update.

        Returns:
            CandidateProfileModel: Refreshed database entity model.
        """
        logger.info("CandidateProfileRepository: committing updated profile")
        self.db.commit()
        logger.info("CandidateProfileRepository: update commit completed, refreshing profile")
        self.db.refresh(profile)
        logger.info("CandidateProfileRepository: update refresh completed successfully")
        return profile

    def find_by_id(self, id: str) -> CandidateProfileModel | None:
        """Find a CandidateProfile by its UUID primary key.

        Args:
            id (str): UUID primary key.

        Returns:
            CandidateProfileModel | None: Matching profile or None.
        """
        import uuid
        if isinstance(id, str):
            try:
                id_uuid = uuid.UUID(id)
            except ValueError:
                return None
        else:
            id_uuid = id
        return self.db.query(CandidateProfileModel).filter(
            CandidateProfileModel.id == id_uuid
        ).first()
