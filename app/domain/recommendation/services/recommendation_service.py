from fastapi import Depends

from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse, RecommendationItem
from app.domain.jobs.models.raw_job import RawJob
from app.application.resume.resume_analyzer_service import ResumeAnalyzer, get_resume_analyzer
from app.application.jobs.job_analyzer_service import JobAnalyzer, get_job_analyzer
from app.application.pipelines.recommendation_pipeline import RecommendationPipeline


class RecommendationService:
    """Orchestration service (entry point) for generating job recommendations."""

    def __init__(
        self,
        resume_analyzer = Depends(get_resume_analyzer),
        job_analyzer = Depends(get_job_analyzer),
        recommendation_pipeline: RecommendationPipeline = Depends(RecommendationPipeline),
    ):
        """Initialize the orchestration service with the new pipeline dependencies."""
        self._resume_analyzer: ResumeAnalyzer = resume_analyzer
        self._job_analyzer: JobAnalyzer = job_analyzer
        self._recommendation_pipeline = recommendation_pipeline

    async def generate_recommendations(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Coordinate embedding, similarity scoring, ranking, and explanation pipelines to
         generate job recommendations.

        Args:
            request (RecommendationRequest): Normalized resume text and job documents.

        Returns:
            RecommendationResponse: Ranked list of recommendation items.
        """
        # 1. Analyze resume text to produce CandidateProfile
        candidate_profile = await self._resume_analyzer.analyze(request.resume_text)

        # 2. Analyze job documents to produce JobProfiles
        job_profiles = []
        for jd in request.jobs:
            raw_job = RawJob(
                title=jd.title,
                company=jd.company,
                location=jd.location,
                description=jd.description,
                apply_url=jd.apply_url,
                employment_type=jd.employment_type,
            )
            job_profile = await self._job_analyzer.analyze(raw_job)
            # Map apply_url to JobProfile (it may not exist in LLM output)
            job_profile.apply_url = jd.apply_url
            job_profiles.append(job_profile)

        # 3. Execute the recommendation pipeline
        results = await self._recommendation_pipeline.run(
            candidate_profile=candidate_profile,
            job_profiles=job_profiles,
        )

        # 4. Map to RecommendationResponse items
        recommendations = []
        for res in results:
            recommendations.append(
                RecommendationItem(
                    title=res.job_profile.title,
                    company=res.job_profile.company,
                    location=res.job_profile.location,
                    description=res.job_profile.summary or "",
                    employment_type=res.job_profile.employment_type,
                    apply_url=res.job_profile.apply_url or "",
                    similarity_score=res.final_score,  # Expose final score as similarity_score for contract compatibility
                    recommendation_reason=res.recommendation_reason,
                )
            )

        return RecommendationResponse(recommendations=recommendations)
