from pydantic import BaseModel, Field


class Skill(BaseModel):
    """Represents a skill associated with a job description."""

    name: str = Field(..., description="The name of the skill.")
    required: bool = Field(True, description="Whether this skill is required (True) or preferred (False).")
    confidence: float = Field(
        1.0,
        description="The confidence score of the extracted skill (between 0.0 and 1.0).",
    )


class JobIntelligence(BaseModel):
    """Canonical Job Intelligence model representing structured job parsing results.

    Contains extracted and normalized metadata fields of a job posting.
    """

    title: str = Field(..., description="The job title.")
    company: str = Field(..., description="The hiring company name.")
    location: str | None = Field(None, description="The job location (city, state, region).")
    employment_type: str | None = Field(
        None, description="Standardized employment type (e.g. Full-time, Part-time, Contract, Internship, Co-op)."
    )
    experience_level: str | None = Field(None, description="Required experience level (e.g. Junior, Mid, Senior, Lead).")
    description: str | None = Field(None, description="Raw or normalized description of the job posting.")
    responsibilities: list[str] = Field(
        default_factory=list,
        description="List of responsibilities, tasks, and duties associated with the role.",
    )
    required_skills: list[Skill] = Field(
        default_factory=list,
        description="The list of required skills for this job role.",
    )
    preferred_skills: list[Skill] = Field(
        default_factory=list,
        description="The list of preferred (optional/bonus) skills for this job role.",
    )
    salary: str | None = Field(None, description="Standardized or raw salary information.")
    remote: bool | None = Field(None, description="Indicates if the job is remote.")
    source: str | None = Field(None, description="Origin or platform of the job posting.")
    apply_url: str | None = Field(None, description="The link to apply for the job.")
