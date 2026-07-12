from typing import Protocol
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_llm_provider
from app.application.ai.prompt_manager import PromptManager
from app.application.ai.structured_output_service import StructuredOutputService
from app.core.logging import get_logger

logger = get_logger(__name__)


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


class LLMResumeAnalyzer:
    """LLM-powered implementation of the ResumeAnalyzer protocol."""

    def __init__(
        self,
        llm_provider: LLMProvider = Depends(get_llm_provider),
        structured_output: StructuredOutputService = Depends(StructuredOutputService),
    ):
        self._llm_provider = llm_provider
        self._structured_output = structured_output

    async def analyze(self, cleaned_text: str) -> CandidateProfile:
        """Use LLM text generation and StructuredOutputService parser to extract CandidateProfile.

        Args:
            cleaned_text (str): Cleaned, normalized resume text.

        Returns:
            CandidateProfile: Extracted candidate profile.

        Raises:
            ValidationException: If LLM call or JSON output mapping/validation fails.
        """
        # Format the prompt from prompt manager template
        prompt = PromptManager.resume_analysis().format(resume_text=cleaned_text)
        system_prompt = "You are an expert resume parsing system. Return ONLY valid JSON matching the CandidateProfile schema."

        try:
            # Call the LLM provider
            raw_response = await self._llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Strict extraction
            )
            
            # Parse output using structured output service
            return self._structured_output.parse(raw_response, CandidateProfile)
        except Exception as e:
            # Log the error and raise to prevent silent partial profiles
            logger.error(
                event="resume_analyzer_llm_failed",
                error=str(e),
                text_length=len(cleaned_text),
            )
            raise


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
