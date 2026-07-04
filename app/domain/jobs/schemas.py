from pydantic import BaseModel, Field
from app.domain.jobs.models import JobIntelligence


class SkillSchema(BaseModel):
    """Pydantic schema representing skill input/output payloads."""

    name: str = Field(..., description="The name of the skill.")
    required: bool = Field(True, description="Whether this skill is required (True) or preferred (False).")
    confidence: float = Field(
        1.0,
        description="The confidence score of the extracted skill (between 0.0 and 1.0).",
    )


class JobNormalizeRequest(BaseModel):
    """Pydantic schema representing raw job data submitted for normalization."""

    title: str = Field(..., description="The raw job title.")
    company: str = Field(..., description="The raw hiring company name.")
    location: str | None = Field(None, description="The raw job location.")
    employment_type: str | None = Field(None, description="The raw employment type.")
    experience_level: str | None = Field(None, description="The raw experience level requirements.")
    description: str | None = Field(None, description="The raw job description body.")
    responsibilities: list[str] = Field(default_factory=list, description="The raw responsibilities list.")
    required_skills: list[SkillSchema] = Field(default_factory=list, description="The raw required skills list.")
    preferred_skills: list[SkillSchema] = Field(default_factory=list, description="The raw preferred skills list.")
    salary: str | None = Field(None, description="The raw salary information.")
    remote: bool | None = Field(None, description="The raw remote indicator.")
    source: str | None = Field(None, description="Origin source or site of the job.")
    apply_url: str | None = Field(None, description="The application link.")


class JobNormalizeResponse(BaseModel):
    """Pydantic schema representing the output of the job normalization process."""

    success: bool = Field(True, description="Indicates if the normalization was successful.")
    job_intelligence: JobIntelligence = Field(..., description="The normalized job intelligence output.")
