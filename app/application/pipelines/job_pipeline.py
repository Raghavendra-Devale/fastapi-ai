from fastapi import Depends
from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.job_analysis_result import JobAnalysisResult
from app.application.jobs.job_normalizer_service import JobNormalizerService
from app.application.jobs.job_analyzer_service import JobAnalyzer, get_job_analyzer
from app.application.jobs.job_embedding_service import JobEmbeddingService
from app.application.jobs.job_persistence_service import JobPersistenceService


class JobPipeline:
    """Orchestrates the complete job processing, normalization, analysis, embedding, and persistence flow."""

    def __init__(
        self,
        normalizer: JobNormalizerService = Depends(JobNormalizerService),
        analyzer = Depends(get_job_analyzer),
        embedding_service: JobEmbeddingService = Depends(JobEmbeddingService),
        persistence: JobPersistenceService = Depends(JobPersistenceService),
    ):
        """Initialize the pipeline with the required specialized job sub-services."""
        self._normalizer = normalizer
        self._analyzer: JobAnalyzer = analyzer
        self._embedding_service = embedding_service
        self._persistence = persistence

    async def run(self, raw_job: RawJob) -> JobAnalysisResult:
        """Run the job pipeline.

        Args:
            raw_job (RawJob): Raw scraped job posting.

        Returns:
            JobAnalysisResult: Standardized job profile, embedding vector, and normalized text.
        """
        # 1. Normalize
        normalized_job = self._normalizer.normalize_job(raw_job)

        # 2. Analyze
        job_profile = await self._analyzer.analyze(raw_job)

        # 3. Embedding
        embedding = await self._embedding_service.generate_embedding(job_profile)

        # 4. Persist
        await self._persistence.save_job_analysis(
            job_profile=job_profile,
            embedding=embedding,
        )

        return JobAnalysisResult(
            job_profile=job_profile,
            embedding=embedding,
            normalized_job=normalized_job,
        )
