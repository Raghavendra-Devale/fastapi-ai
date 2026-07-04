from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Represents a single ranked job recommendation item returned by the AI Engine."""

    title: str = Field(..., description="The job title.")
    company: str = Field(..., description="The hiring company name.")
    location: str | None = Field(None, description="The job location.")
    description: str = Field(..., description="The job description content.")
    employment_type: str | None = Field(None, description="The employment type (e.g. Full-time, Part-time).")
    apply_url: str = Field(..., description="The URL to apply for the job.")
    similarity_score: float = Field(..., description="The calculated cosine similarity score between the resume and job description.")
    recommendation_reason: str | None = Field(None, description="An optional explanation or reason for recommending this job.")


class RecommendationResponse(BaseModel):
    """Represents the complete list of ranked job recommendations returned by the AI Engine."""

    recommendations: list[RecommendationItem] = Field(..., description="The list of ranked job recommendation items.")
