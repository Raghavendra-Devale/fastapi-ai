from app.domain.resume.candidate_profile import CandidateProfile


class ResumeAnalyzerService:
    """Service responsible for parsing and analyzing normalized text to produce a CandidateProfile."""

    def __init__(self):
        pass

    async def analyze_resume(self, cleaned_text: str) -> CandidateProfile:
        """Analyze normalized resume text and extract candidate profile information.

        Args:
            cleaned_text (str): Cleaned, normalized resume text.

        Returns:
            CandidateProfile: Extracted candidate profile.
        """
        # Phase 1: Architecture Only (no AI prompts or LLM parsing implemented yet)
        # Returns a placeholder CandidateProfile for structure verification
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
            preferred_roles=["Software Engineer", "Backend Developer"],
            strengths=["Communication", "Problem Solving"],
            weaknesses=["Perfectionism"],
        )
