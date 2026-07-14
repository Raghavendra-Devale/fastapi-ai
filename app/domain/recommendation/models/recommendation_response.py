from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Represents a single ranked job recommendation match returned by the AI Engine."""

    job_id: int = Field(..., description="The unique Spring Boot job ID.")
    similarity_score: float = Field(..., description="The calculated match score between the resume and job description.")
    matching_skills: list[str] = Field(default_factory=list, description="Skills present in both candidate and job.")
    missing_skills: list[str] = Field(default_factory=list, description="Skills required by the job but missing in the candidate.")
    recommendation_reason: str | None = Field(None, description="An explanation of why this job was recommended.")


class RecommendationResponse(BaseModel):
    """Represents the complete list of ranked job recommendations returned by the AI Engine."""

    recommendations: list[RecommendationItem] = Field(..., description="The list of ranked job recommendation items.")
