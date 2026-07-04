from datetime import datetime
from pydantic import BaseModel, Field


class RawJob(BaseModel):
    """Canonical, provider-independent representation of a raw job posting fetched from an external source."""

    title: str = Field(..., description="The title of the job posting.")
    company: str = Field(..., description="The name of the hiring organization.")
    location: str | None = Field(None, description="The job location (city, state, country).")
    description: str = Field(..., description="The raw, unprocessed text description of the job posting.")
    employment_type: str | None = Field(None, description="The employment type (e.g. Full-time, Contract).")
    apply_url: str = Field(..., description="The direct URL to apply for the job posting.")
    source: str = Field(..., description="The name of the external job feed or provider.")
    published_at: datetime | None = Field(None, description="The timestamp when the job posting was published.")
