from pydantic import BaseModel, Field
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.resume_suggestion import ResumeSuggestion


class ResumeAnalysisResult(BaseModel):
    """Result wrapper containing all output elements of the resume analysis pipeline."""

    candidate_profile: CandidateProfile = Field(..., description="The parsed/analyzed candidate profile details.")
    embedding: list[float] = Field(default_factory=list, description="The generated vector embedding for the resume.")
    suggestions: list[ResumeSuggestion] = Field(default_factory=list, description="Suggestions for resume improvement.")
    extracted_text: str = Field("", description="The normalized raw text extracted from the resume PDF.")
