import uuid
from sqlalchemy import Column, String, Numeric, DateTime, ForeignKey, Text, BigInteger, func
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.core.database import Base


class AIModelModel(Base):
    """Represents prompt and model versions logs."""
    __tablename__ = "ai_models"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider = Column(String(50), nullable=False)
    llm_model = Column(String(100), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    prompt_version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CandidateProfileModel(Base):
    """Represents a candidate resume profile structured understanding and its vector embedding."""
    __tablename__ = "candidate_profiles"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_id = Column(BigInteger, nullable=False, unique=True)
    user_id = Column(BigInteger, nullable=False)
    headline = Column(String(255))
    experience_years = Column(Numeric(4, 2))
    primary_role = Column(String(100))
    profile_json = Column(JSONB)
    embedding = Column(Vector(384))  # Dimension matches all-MiniLM-L6-v2 vector size
    embedding_model = Column(String(100), nullable=False)
    llm_model = Column(String(100), nullable=False)
    prompt_version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    recommendations = relationship(
        "RecommendationResultModel",
        back_populates="candidate_profile",
        cascade="all, delete-orphan",
    )


class JobProfileModel(Base):
    """Represents a normalized job profile and its vector embedding."""
    __tablename__ = "job_profiles"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(BigInteger, nullable=False, unique=True)
    provider = Column(String(100))
    title = Column(String(255))
    company = Column(String(255))
    location = Column(String(255))
    experience = Column(String(100))
    profile_json = Column(JSONB)
    embedding = Column(Vector(384))
    embedding_model = Column(String(100), nullable=False)
    llm_model = Column(String(100), nullable=False)
    prompt_version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    recommendations = relationship(
        "RecommendationResultModel",
        back_populates="job_profile",
        cascade="all, delete-orphan",
    )


class RecommendationResultModel(Base):
    """Represents multi-dimensional recommendations generated for job profiles and candidates."""
    __tablename__ = "recommendation_results"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai.candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_profile_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai.job_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    semantic_score = Column(Numeric(5, 4), nullable=False)
    skill_score = Column(Numeric(5, 4), nullable=False)
    experience_score = Column(Numeric(5, 4), nullable=False)
    location_score = Column(Numeric(5, 4), nullable=False)
    education_score = Column(Numeric(5, 4), nullable=False)
    final_score = Column(Numeric(5, 4), nullable=False)
    reason = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    candidate_profile = relationship("CandidateProfileModel", back_populates="recommendations")
    job_profile = relationship("JobProfileModel", back_populates="recommendations")
    skill_gap = relationship(
        "SkillGapModel",
        back_populates="recommendation",
        uselist=False,
        cascade="all, delete-orphan",
    )


class SkillGapModel(Base):
    """Represents matched and missing skills breakdown of a generated recommendation score."""
    __tablename__ = "skill_gap"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai.recommendation_results.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    matched_required = Column(ARRAY(Text))
    matched_preferred = Column(ARRAY(Text))
    missing_required = Column(ARRAY(Text))
    missing_preferred = Column(ARRAY(Text))
    coverage = Column(Numeric(5, 4), nullable=False)

    recommendation = relationship("RecommendationResultModel", back_populates="skill_gap")


class ProcessingStatusModel(Base):
    """Logs the asynchronous processing status of entities (Resume / Jobs)."""
    __tablename__ = "processing_status"
    __table_args__ = {"schema": "ai"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(50), nullable=False)  # 'RESUME' or 'JOB'
    entity_id = Column(BigInteger, nullable=False)
    status = Column(String(50), nullable=False)  # 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'STALE'
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
