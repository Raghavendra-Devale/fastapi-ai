from pydantic import BaseModel, Field
from app.domain.resume.models import Skill, Education, Experience, Project, Certification, Language


class CandidateProfile(BaseModel):
    """Canonical Candidate Profile model representing the structured AI extraction results."""

    name: str | None = Field(None, description="The full name of the candidate.")
    headline: str | None = Field(None, description="A professional headline or job title.")
    summary: str | None = Field(None, description="A short professional summary.")
    experience_years: float | None = Field(None, description="Total years of professional experience.")
    
    # Reusing existing structural models
    skills: list[Skill] = Field(default_factory=list, description="List of extracted skills.")
    projects: list[Project] = Field(default_factory=list, description="List of key projects.")
    education: list[Education] = Field(default_factory=list, description="List of educational history.")
    certifications: list[Certification] = Field(default_factory=list, description="List of professional certifications.")
    languages: list[Language] = Field(default_factory=list, description="List of languages spoken.")
    
    # Existing experience model for complete job history compatibility
    experience: list[Experience] = Field(default_factory=list, description="List of professional experiences.")

    # New collections
    preferred_roles: list[str] = Field(default_factory=list, description="Preferred roles or job titles.")
    domains: list[str] = Field(default_factory=list, description="Industry domains matching candidate experience.")
    strengths: list[str] = Field(default_factory=list, description="Key candidate strengths.")
    weaknesses: list[str] = Field(default_factory=list, description="Areas for candidate improvement/weaknesses.")
