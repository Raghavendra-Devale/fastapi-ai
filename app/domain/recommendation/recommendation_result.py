from pydantic import BaseModel, Field
from app.domain.jobs.job_profile import JobProfile


class RecommendationResult(BaseModel):
    """Domain model representing a recommended job and the matching details."""

    job_profile: JobProfile = Field(..., description="The recommended job profile.")
    semantic_score: float = Field(..., description="The semantic similarity score (cosine similarity).")
    skill_score: float = Field(..., description="The skill match score (0.0 to 1.0).")
    experience_score: float = Field(..., description="The experience match score (0.0 to 1.0).")
    location_score: float = Field(..., description="The location match score (0.0 to 1.0).")
    education_score: float = Field(..., description="The education match score (0.0 to 1.0).")
    final_score: float = Field(..., description="The final weighted recommendation score (0.0 to 1.0).")
    matched_skills: list[str] = Field(default_factory=list, description="Skills present in both candidate and job.")
    missing_skills: list[str] = Field(default_factory=list, description="Skills required by the job but missing in the candidate.")
    recommendation_reason: str | None = Field(None, description="Actionable explanation of why this job was recommended.")
