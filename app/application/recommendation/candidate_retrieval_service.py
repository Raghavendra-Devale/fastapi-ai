from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile


class CandidateRetrievalService:
    """Service responsible for retrieving matching job candidates for a CandidateProfile."""

    def __init__(self):
        pass

    async def retrieve_jobs(self, candidate_profile: CandidateProfile) -> list[JobProfile]:
        """Retrieve a list of potential matching JobProfile candidates.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.

        Returns:
            list[JobProfile]: List of job profiles.
        """
        # Phase 3: Architecture Only (mock candidate jobs retrieval)
        # Returns a couple of default job profiles to rank
        return [
            JobProfile(
                id="mock-job-1",
                title="Python Developer",
                company="PyCorp",
                summary="Write python application code.",
                required_skills=["Python", "FastAPI"],
                preferred_skills=["Docker"],
                location="San Francisco, CA",
            ),
            JobProfile(
                id="mock-job-2",
                title="React Frontend Developer",
                company="UI-Design",
                summary="Build user interfaces in React.",
                required_skills=["React", "JavaScript"],
                preferred_skills=["TypeScript"],
                location="Remote",
            ),
        ]
