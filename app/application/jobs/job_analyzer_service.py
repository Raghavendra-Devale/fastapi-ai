from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.job_profile import JobProfile


class JobAnalyzerService:
    """Service responsible for parsing and analyzing a RawJob to build a canonical JobProfile."""

    def __init__(self):
        pass

    async def analyze_job(self, raw_job: RawJob) -> JobProfile:
        """Analyze a raw job posting and return a canonical JobProfile.

        Args:
            raw_job (RawJob): Raw scraped job posting.

        Returns:
            JobProfile: Standardized parsed job profile.
        """
        # Phase 2: Architecture Only (no AI prompts or LLM parsing implemented yet)
        # Returns a mock JobProfile for pipeline integration testing
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
            responsibilities=["Develop backend APIs", "Write tests"],
            requirements=["Solid Python knowledge", "API design experience"],
            benefits=["Health insurance", "Unlimited PTO"],
            technologies=["Python", "FastAPI", "PostgreSQL"],
            keywords=["Backend", "Developer", "API"],
            confidence=1.0,
        )
