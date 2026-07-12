from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.resume_suggestion import ResumeSuggestion


class ResumeSuggestionService:
    """Service responsible for generating actionable suggestions from a CandidateProfile."""

    def __init__(self):
        pass

    async def generate_suggestions(self, profile: CandidateProfile) -> list[ResumeSuggestion]:
        """Generate suggestions for resume improvement based on the candidate profile.

        Args:
            profile (CandidateProfile): The candidate profile.

        Returns:
            list[ResumeSuggestion]: List of suggestions.
        """
        suggestions = []

        # Placeholder rule 1: Missing summary
        if not profile.summary:
            suggestions.append(
                ResumeSuggestion(
                    category="Content",
                    severity="Medium",
                    message="Professional summary is missing or empty.",
                    recommendation="Add a 2-3 sentence professional summary at the top of your resume.",
                )
            )

        # Placeholder rule 2: Missing skills
        if not profile.skills:
            suggestions.append(
                ResumeSuggestion(
                    category="Skills",
                    severity="High",
                    message="No skills were extracted or identified.",
                    recommendation="Include a dedicated skills section highlighting key technical/soft skills.",
                )
            )

        # Placeholder rule 3: Missing work experience
        if not profile.experience:
            suggestions.append(
                ResumeSuggestion(
                    category="Content",
                    severity="High",
                    message="Work experience entries are missing.",
                    recommendation="Ensure your work history is clearly detailed with company name, title, and dates.",
                )
            )

        return suggestions
