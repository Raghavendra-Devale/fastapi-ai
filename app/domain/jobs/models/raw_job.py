from pydantic import BaseModel, Field


class RawJob(BaseModel):
    """Represents raw job posting data fetched from external job sources."""

    title: str = Field(..., description="Raw title of the job.")
    company: str = Field(..., description="Raw company name.")
    location: str | None = Field(None, description="Raw location string.")
    description: str | None = Field(None, description="Raw HTML or text description of the job.")
    apply_url: str | None = Field(None, description="URL to apply for the job.")
    salary: str | None = Field(None, description="Raw salary information.")
    employment_type: str | None = Field(None, description="Raw employment type string.")
    source: str | None = Field(None, description="The name of the external job source provider.")
