from typing import Protocol
from fastapi import Depends

from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.job_profile import JobProfile
from app.application.ai.prompt_manager import PromptManager
from app.application.ai.analyzers.base_analyzer import BaseAIAnalyzer


class JobAnalyzer(Protocol):
    """Protocol (interface) defining the contract for job analysis implementations."""

    async def analyze(self, raw_job: RawJob) -> JobProfile:
        """Parse and extract a structured JobProfile from a RawJob.

        Args:
            raw_job (RawJob): Raw scraped job posting.

        Returns:
            JobProfile: Standardized parsed job profile.
        """
        ...


class LLMJobAnalyzer(BaseAIAnalyzer[JobProfile]):
    """LLM-powered structured job analyzer building on BaseAIAnalyzer."""

    async def analyze(self, raw_job: RawJob) -> JobProfile:
        """Parse and extract JobProfile from raw job posting context using BaseAIAnalyzer.

        Args:
            raw_job (RawJob): Raw scraped job posting.

        Returns:
            JobProfile: Extracted canonical job profile.

        Raises:
            AIProviderException: If the external LLM call fails.
            AIParsingException: If JSON extraction fails.
            AIValidationException: If Pydantic model validation fails.
        """
        # Build contextual string for job description
        desc = (
            f"Title: {raw_job.title or 'Unknown'}\n"
            f"Company: {raw_job.company or 'Unknown'}\n"
            f"Location: {raw_job.location or 'Unknown'}\n"
            f"Description:\n{raw_job.description or ''}"
        )
        prompt = PromptManager.job_analysis().format(job_description=desc)
        system_prompt = "You are an expert job parsing system. Return ONLY valid JSON matching the JobProfile schema."

        # Execute structured LLM extraction
        profile = await self._analyze_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            output_model=JobProfile,
            temperature=0.0,
        )

        # Apply fallbacks from raw_job where LLM response had blanks
        if not profile.title and raw_job.title:
            profile.title = raw_job.title
        if not profile.company and raw_job.company:
            profile.company = raw_job.company
        if not profile.location and raw_job.location:
            profile.location = raw_job.location
        if not profile.employment_type and raw_job.employment_type:
            profile.employment_type = raw_job.employment_type
        if not profile.salary and raw_job.salary:
            profile.salary = raw_job.salary

        # Map apply_url directly from raw metadata
        profile.apply_url = raw_job.apply_url

        return profile


class MockJobAnalyzer:
    """Mock implementation of the JobAnalyzer protocol for tests and offline usage."""

    async def analyze(self, raw_job: RawJob) -> JobProfile:
        """Mock parsing returning generic JobProfile values."""
        return JobProfile(
            id="mock-job-123",
            title=raw_job.title or "Software Engineer",
            company=raw_job.company or "Tech Corp",
            summary="A placeholder job profile summary.",
            required_skills=["Python", "FastAPI"],
            preferred_skills=["Docker", "AWS"],
            experience="3+ years",
            education="Bachelor's in CS or equivalent",
            employment_type=raw_job.employment_type or "Full-time",
            location=raw_job.location or "San Francisco, CA",
            salary=raw_job.salary or "$120,000 - $150,000",
            industry="Software",
            apply_url=raw_job.apply_url or "https://mockjob.com/apply",
            responsibilities=["Develop backend APIs", "Write tests"],
            requirements=["Solid Python knowledge", "API design experience"],
            benefits=["Health insurance", "Unlimited PTO"],
            technologies=["Python", "FastAPI", "PostgreSQL"],
            keywords=["Backend", "Developer", "API"],
            confidence=1.0,
        )


def get_job_analyzer(
    llm_analyzer: LLMJobAnalyzer = Depends(LLMJobAnalyzer),
) -> JobAnalyzer:
    """Dependency resolver returning the active JobAnalyzer implementation."""
    return llm_analyzer
