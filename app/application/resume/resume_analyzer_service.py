from typing import Protocol
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.domain.resume.candidate_profile import CandidateProfile
from app.application.ai.prompt_manager import PromptManager
from app.application.ai.analyzers.base_analyzer import BaseAIAnalyzer

# Re-export exceptions for cleaner interface if needed
from app.domain.ai.exceptions import AIProviderException, AIParsingException, AIValidationException


class ResumeAnalyzer(Protocol):
    """Protocol (interface) defining the contract for resume analysis implementations."""

    async def analyze(self, cleaned_text: str) -> CandidateProfile:
        """Parse and extract a structured CandidateProfile from normalized resume text.

        Args:
            cleaned_text (str): Cleaned, normalized resume text.

        Returns:
            CandidateProfile: Structured candidate profile.
        """
        ...


class LLMResumeAnalyzer(BaseAIAnalyzer[CandidateProfile]):
    """LLM-powered structured resume analyzer building on BaseAIAnalyzer."""

    async def analyze(self, cleaned_text: str) -> CandidateProfile:
        """Parse and extract CandidateProfile from cleaned resume text using BaseAIAnalyzer.

        Args:
            cleaned_text (str): Cleaned, normalized resume text.

        Returns:
            CandidateProfile: Extracted candidate profile.

        Raises:
            AIProviderException: If the external LLM call fails.
            AIParsingException: If JSON extraction fails.
            AIValidationException: If Pydantic model validation fails.
        """
        prompt = PromptManager.resume_analysis().format(resume_text=cleaned_text)
        system_prompt = "You are an expert resume parsing system. Return ONLY valid JSON matching the CandidateProfile schema."

        return await self._analyze_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            output_model=CandidateProfile,
            temperature=0.0,
        )


class MockResumeAnalyzer:
    """Mock implementation of the ResumeAnalyzer protocol for tests and offline usage."""

    async def analyze(self, cleaned_text: str) -> CandidateProfile:
        """Mock parsing return placeholder values."""
        return CandidateProfile(
            name="John Doe",
            headline="Software Engineer",
            summary="A placeholder candidate profile summary.",
            experience_years=5.0,
            skills=[],
            projects=[],
            education=[],
            certifications=[],
            languages=[],
            preferred_roles=["Software Engineer"],
            domains=["Cloud"],
            strengths=["Communication"],
            weaknesses=["Perfectionism"],
        )


def get_resume_analyzer(
    llm_analyzer: LLMResumeAnalyzer = Depends(LLMResumeAnalyzer),
) -> ResumeAnalyzer:
    """Dependency resolver returning the active ResumeAnalyzer implementation."""
    return llm_analyzer
