from pydantic import BaseModel, Field
from app.domain.jobs.job_profile import JobProfile


class RecommendationResult(BaseModel):
    """Domain model representing a recommended job and the matching details."""

    job_profile: JobProfile = Field(..., description="The recommended job profile.")
    similarity_score: float = Field(..., description="The semantic similarity score (cosine similarity).")
    final_score: float = Field(..., description="The final recommendation score (incorporating other factors).")
    matched_skills: list[str] = Field(default_factory=list, description="Skills present in both candidate and job.")
    missing_skills: list[str] = Field(default_factory=list, description="Skills required by the job but missing in the candidate.")
    recommendation_reason: str | None = Field(None, description="Actionable explanation of why this job was recommended.")
