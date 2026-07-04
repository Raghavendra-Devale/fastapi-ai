from pydantic import BaseModel, Field


class Skill(BaseModel):
    """Represents a candidate's skill extracted from a resume."""

    name: str = Field(..., description="The name of the skill.")
    confidence: float = Field(
        ...,
        description="The confidence score of the extracted skill (between 0.0 and 1.0).",
    )


class Education(BaseModel):
    """Represents educational details extracted from a resume."""

    degree: str | None = Field(None, description="The degree or qualification earned.")
    institution: str | None = Field(
        None, description="The name of the school, college, or university."
    )
    start_date: str | None = Field(
        None, description="The start date of the education period (e.g. YYYY-MM)."
    )
    end_date: str | None = Field(
        None, description="The end date or expected completion date (e.g. YYYY-MM)."
    )


class Experience(BaseModel):
    """Represents professional experience details extracted from a resume."""

    company: str | None = Field(
        None, description="The name of the employing organization."
    )
    designation: str | None = Field(None, description="The job title or role designation.")
    start_date: str | None = Field(
        None, description="The start date of the employment period (e.g. YYYY-MM)."
    )
    end_date: str | None = Field(
        None, description="The end date of the employment period (e.g. YYYY-MM or Present)."
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="The list of responsibilities, achievements, and tasks performed.",
    )


class Project(BaseModel):
    """Represents details of a project worked on by the candidate."""

    name: str = Field(..., description="The name of the project.")
    description: str | None = Field(None, description="A brief description of the project.")
    technologies: list[str] = Field(
        default_factory=list,
        description="The list of technologies, languages, and tools utilized.",
    )


class Certification(BaseModel):
    """Represents a certification held by the candidate."""

    name: str = Field(..., description="The name of the certification.")
    issuer: str | None = Field(
        None, description="The organization or authority that issued the certification."
    )


class Language(BaseModel):
    """Represents a language and the candidate's proficiency in it."""

    name: str = Field(..., description="The name of the language.")
    proficiency: str | None = Field(
        None, description="The level of language proficiency (e.g. Native, Fluent, Basic)."
    )


class EmbeddingMetadata(BaseModel):
    """Represents metadata describing the generated vector embedding."""

    model: str = Field(
        ..., description="The identifier of the model used to generate the embedding."
    )
    dimensions: int = Field(
        ..., description="The length (dimensions) of the generated embedding vector."
    )


class ResumeIntelligence(BaseModel):
    """Canonical Resume Intelligence model representing structured extraction results.

    This acts as the single source of truth and output format for all resume processing
    within the AI engine, hiding raw vector data while presenting enriched metadata.
    """

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
    embedding: EmbeddingMetadata = Field(
        ..., description="Metadata describing the generated vector embedding."
    )
