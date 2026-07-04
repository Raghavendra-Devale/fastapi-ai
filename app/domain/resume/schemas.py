from pydantic import BaseModel, Field
from app.domain.resume.models import (
    ResumeIntelligence,
    Skill,
    Education,
    Experience,
    Project,
    Certification,
    Language,
)


class ResumeProcessResponse(BaseModel):
    """Pydantic schema representing the processed resume output."""

    resume_text: str = Field(
        ..., description="The normalized text content extracted from the resume PDF."
    )
    embedding: list[float] = Field(
        ..., description="The generated vector embedding representing the resume content."
    )
    embedding_dimensions: int = Field(
        ..., description="The length (dimensions) of the generated embedding."
    )
    embedding_model: str = Field(
        ..., description="The model identifier used to generate the embedding."
    )
    processing_time_ms: float = Field(
        ..., description="The total pipeline processing time in milliseconds."
    )
    summary: str | None = Field(
        None, description="A short, generated professional summary of the candidate."
    )
    intelligence: ResumeIntelligence | None = Field(
        None, description="The enriched ResumeIntelligence object."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume_text": "John Doe\nSoftware Engineer\nExperience...",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dimensions": 3,
                "embedding_model": "all-MiniLM-L6-v2",
                "processing_time_ms": 123.45,
            }
        }
    }


class ResumeDetails(BaseModel):
    """Details of the processed resume, omitting raw embeddings."""

    extracted_text: str = Field(
        ..., description="The normalized text content extracted from the resume PDF."
    )
    embedding_model: str = Field(
        ..., description="The model identifier used to generate the embedding."
    )
    embedding_dimensions: int = Field(
        ..., description="The length (dimensions) of the generated embedding."
    )
    summary: str | None = Field(
        None, description="A short, generated professional summary of the candidate."
    )
    skills: list[Skill] = Field(
        default_factory=list,
        description="The list of skills extracted from the resume.",
    )
    education: list[Education] = Field(
        default_factory=list,
        description="The list of educational credentials extracted.",
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description="The list of work history entries extracted.",
    )
    projects: list[Project] = Field(
        default_factory=list,
        description="The list of projects extracted.",
    )
    certifications: list[Certification] = Field(
        default_factory=list,
        description="The list of certifications extracted.",
    )
    languages: list[Language] = Field(
        default_factory=list,
        description="The list of languages spoken by the candidate.",
    )


class ResumeProcessAPIResponse(BaseModel):
    """Pydantic schema representing the public resume processing API response."""

    success: bool = Field(True, description="Indicates if the operation was successful.")
    processing_time_ms: float = Field(
        ..., description="The total pipeline processing time in milliseconds."
    )
    resume: ResumeDetails = Field(
        ..., description="Details of the processed resume."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "processing_time_ms": 123.45,
                "resume": {
                    "extracted_text": "John Doe\nSoftware Engineer\nExperience...",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimensions": 384,
                }
            }
        }
    }


class ResumeIntelligence(BaseModel):
    """Canonical Resume Intelligence business data (excluding embeddings and AI details)."""

    extracted_text: str = Field(
        ..., description="The normalized raw text content extracted from the resume."
    )
    summary: str | None = Field(
        None, description="A short, generated professional summary of the candidate."
    )
    skills: list[Skill] = Field(
        default_factory=list,
        description="The list of skills extracted from the resume.",
    )
    education: list[Education] = Field(
        default_factory=list,
        description="The list of educational credentials extracted.",
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description="The list of work history entries extracted.",
    )
    certifications: list[Certification] = Field(
        default_factory=list,
        description="The list of certifications extracted.",
    )
    projects: list[Project] = Field(
        default_factory=list,
        description="The list of projects extracted.",
    )
    languages: list[Language] = Field(
        default_factory=list,
        description="The list of languages spoken by the candidate.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "extracted_text": "John Doe\nSoftware Engineer\nExperience...",
                "summary": "Experienced software engineer with a strong background in Python.",
                "skills": [{"name": "Python", "confidence": 0.95}],
                "education": [
                    {
                        "degree": "B.S. CS",
                        "institution": "MIT",
                        "start_date": "2018-09",
                        "end_date": "2022-06",
                    }
                ],
                "experience": [
                    {
                        "company": "Google",
                        "designation": "SWE",
                        "start_date": "2022-07",
                        "end_date": "Present",
                        "responsibilities": ["Coding"],
                    }
                ],
                "projects": [
                    {
                        "name": "AI Engine",
                        "description": "FastAPI AI Engine",
                        "technologies": ["FastAPI", "Python"],
                    }
                ],
                "certifications": [{"name": "AWS Pro", "issuer": "Amazon"}],
                "languages": [{"name": "English", "proficiency": "Native"}],
            }
        }
    }


