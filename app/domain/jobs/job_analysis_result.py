from pydantic import BaseModel, Field
from app.domain.jobs.job_profile import JobProfile


class JobAnalysisResult(BaseModel):
    """Result wrapper containing all output elements of the job intelligence pipeline."""

    job_profile: JobProfile = Field(..., description="The parsed/analyzed job profile details.")
    embedding: list[float] = Field(default_factory=list, description="The generated vector embedding for the job description.")
    normalized_job: str = Field(..., description="The fully formatted, cleaned, and normalized job description text.")
