from pydantic import BaseModel, Field


class JobProfile(BaseModel):
    """Canonical Job Profile model acting as the AI representation of a job description."""

    id: str | None = Field(None, description="The unique identifier of the job.")
    title: str = Field(..., description="The job title.")
    company: str = Field(..., description="The hiring company name.")
    summary: str | None = Field(None, description="A brief summary of the job description.")
    
    # Skills (simple list of string names for Phase 2)
    required_skills: list[str] = Field(default_factory=list, description="List of required skill names.")
    preferred_skills: list[str] = Field(default_factory=list, description="List of preferred skill names.")
    
    # Experience and Education requirements
    experience: str | None = Field(None, description="Detailed professional experience requirements.")
    education: str | None = Field(None, description="Education or degree requirements.")
    
    # Metadata
    employment_type: str | None = Field(None, description="Standardized employment type (e.g., Full-time, Contract).")
    location: str | None = Field(None, description="Job location.")
    salary: str | None = Field(None, description="Standardized or raw salary information.")
    industry: str | None = Field(None, description="The industry field of the job/company.")
    
    # Collections
    responsibilities: list[str] = Field(default_factory=list, description="List of job responsibilities and duties.")
    requirements: list[str] = Field(default_factory=list, description="List of core requirements for the candidate.")
    benefits: list[str] = Field(default_factory=list, description="List of benefits offered by the company.")
    technologies: list[str] = Field(default_factory=list, description="List of tools and technologies used.")
    keywords: list[str] = Field(default_factory=list, description="Keywords/tags extracted for categorization.")
    
    # AI Metadata
    confidence: float = Field(1.0, description="Overall AI parsing/extraction confidence score (0.0 to 1.0).")
