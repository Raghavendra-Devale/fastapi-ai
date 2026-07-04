from datetime import datetime
from pydantic import BaseModel, Field


class JobDocument(BaseModel):
    """Represents normalized job posting data received from Spring Boot for recommendations."""

    title: str = Field(..., description="The job title.")
    company: str = Field(..., description="The hiring company name.")
    location: str | None = Field(None, description="The job location.")
    description: str = Field(..., description="The job description content.")
    employment_type: str | None = Field(None, description="The employment type (e.g. Full-time, Part-time).")
    apply_url: str = Field(..., description="The URL to apply for the job.")
    published_at: datetime | None = Field(None, description="The date and time when the job was published.")


class RecommendationRequest(BaseModel):
    """Represents the request payload sent by Spring Boot to get job recommendations."""

    resume_text: str = Field(..., description="The normalized text content of the candidate's resume.")
    jobs: list[JobDocument] = Field(..., description="The list of normalized jobs to run recommendations against.")
