from typing import Protocol
from fastapi import Depends

from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.domain.recommendation.recommendation_reason import RecommendationReason
from app.application.ai.prompt_manager import PromptManager
from app.application.ai.analyzers.base_analyzer import BaseAIAnalyzer


class ReasonAnalyzer(Protocol):
    """Protocol (interface) defining the contract for recommendation reasoning analysis."""

    async def analyze(
        self,
        result: RecommendationResult,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> RecommendationReason:
        """Analyze and build explaining bullet points matching candidate, job, and precomputed scores.

        Args:
            result (RecommendationResult): Match result containing precomputed scoring.
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.

        Returns:
            RecommendationReason: Parsed matching reasons.
        """
        ...


class LLMReasonAnalyzer(BaseAIAnalyzer[RecommendationReason]):
    """LLM-powered implementation of the ReasonAnalyzer protocol."""

    async def analyze(
        self,
        result: RecommendationResult,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> RecommendationReason:
        """Format matching context and scores and parse matching reasons.

        Args:
            result (RecommendationResult): Match result containing precomputed scoring.
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.

        Returns:
            RecommendationReason: Parsed matching reasons.

        Raises:
            AIProviderException: If the external LLM call fails.
            AIParsingException: If JSON extraction fails.
            AIValidationException: If Pydantic model validation fails.
        """
        # 1. Format scores summary
        scores_summary = (
            f"- Semantic Cosine Match: {round(result.semantic_score * 100)}%\n"
            f"- Skill Match: {round(result.skill_score * 100)}%\n"
            f"- Experience Match: {round(result.experience_score * 100)}%\n"
            f"- Location Match: {round(result.location_score * 100)}%\n"
            f"- Education Match: {round(result.education_score * 100)}%\n"
            f"- Final Recommendation Match: {round(result.final_score * 100)}%\n"
            f"- Matched Skills: {', '.join(result.matched_skills) or 'None'}\n"
            f"- Missing Skills: {', '.join(result.missing_skills) or 'None'}"
        )

        # 2. Format candidate profile details
        cand_summary = (
            f"Headline: {candidate_profile.headline or 'None'}\n"
            f"Experience Years: {candidate_profile.experience_years or 0.0}\n"
            f"Skills: {', '.join(s.name for s in candidate_profile.skills) or 'None'}"
        )

        # 3. Format job profile details
        job_summary = (
            f"Title: {job_profile.title}\n"
            f"Company: {job_profile.company}\n"
            f"Experience Required: {job_profile.experience or 'None'}\n"
            f"Required Skills: {', '.join(job_profile.required_skills) or 'None'}\n"
            f"Preferred Skills: {', '.join(job_profile.preferred_skills) or 'None'}"
        )

        prompt = PromptManager.recommendation_reason().format(
            candidate_profile=cand_summary,
            job_profile=job_summary,
            scores=scores_summary,
        )
        system_prompt = "You are a professional recruiting assistant. Return ONLY a valid JSON object matching the schema."

        # Execute structured LLM extraction
        return await self._analyze_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            output_model=RecommendationReason,
            temperature=0.2,  # Wording fluidity
        )


class MockReasonAnalyzer:
    """Mock implementation of the ReasonAnalyzer protocol for tests and offline usage."""

    async def analyze(
        self,
        result: RecommendationResult,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> RecommendationReason:
        """Mock matching reasons return generic placeholder bullet points."""
        return RecommendationReason(
            bullet_points=[
                f"Matched candidate experience ({candidate_profile.experience_years or 0} yrs) against job description requirement ({job_profile.experience or 'None'}).",
                f"Strong skill alignment on matched technologies: {', '.join(result.matched_skills[:3]) or 'None'}.",
                f"Missing key required skills: {', '.join(result.missing_skills[:3]) or 'None'}.",
            ]
        )


def get_reason_analyzer(
    llm_analyzer: LLMReasonAnalyzer = Depends(LLMReasonAnalyzer),
) -> ReasonAnalyzer:
    """Dependency resolver returning the active ReasonAnalyzer implementation."""
    return llm_analyzer
