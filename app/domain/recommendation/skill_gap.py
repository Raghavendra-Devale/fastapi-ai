from pydantic import BaseModel, Field


class SkillGap(BaseModel):
    """Domain model representing the deterministic skill gap analysis between a candidate and a job."""

    matched_skills: list[str] = Field(
        default_factory=list,
        description="Skills present in both candidate and job requirements.",
    )
    missing_required_skills: list[str] = Field(
        default_factory=list,
        description="Required job skills missing in the candidate profile.",
    )
    missing_preferred_skills: list[str] = Field(
        default_factory=list,
        description="Preferred job skills missing in the candidate profile.",
    )
    coverage_percentage: float = Field(
        ...,
        description="Percentage of required skills met by the candidate (0.0 to 100.0).",
    )
