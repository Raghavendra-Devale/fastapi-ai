import re
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile


class ExperienceScoreService:
    """Service to calculate candidate experience score against job requirements."""

    def _extract_years_required(self, experience_str: str | None) -> float:
        """Parse years of experience required from string description."""
        if not experience_str:
            return 0.0
        # Matches patterns like "3+ years", "3-5 years", "3 years", "5 years of experience"
        match = re.search(r"(\d+)\s*(?:-\s*(\d+))?\s*(?:year|yr|yr\.)", experience_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0

    def calculate_score(
        self,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> float:
        """Calculate experience match score.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.

        Returns:
            float: Experience score (0.0 to 1.0).
        """
        years_required = self._extract_years_required(job_profile.experience)
        if years_required <= 0.0:
            return 1.0

        candidate_exp = candidate_profile.experience_years
        if candidate_exp is None:
            return 0.0

        if candidate_exp >= years_required:
            return 1.0

        # Calculate a fractional penalty for experience deficit
        score = candidate_exp / years_required
        return max(0.0, min(1.0, float(score)))
