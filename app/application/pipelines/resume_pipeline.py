from fastapi import Depends
from app.application.resume.resume_extractor_service import ResumeExtractorService
from app.application.resume.resume_cleaner_service import ResumeCleanerService
from app.application.resume.resume_analyzer_service import ResumeAnalyzer, get_resume_analyzer
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.resume.resume_suggestion_service import ResumeSuggestionService
from app.application.resume.resume_persistence_service import ResumePersistenceService
from app.domain.resume.resume_analysis_result import ResumeAnalysisResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class ResumePipeline:
    """Orchestrates the complete resume processing flow.

    Refactored flow from Controller -> ResumePipeline -> Extractor, Cleaner, Analyzer,
    Embedding, Suggestions, Persistence.
    """

    def __init__(
        self,
        extractor: ResumeExtractorService = Depends(ResumeExtractorService),
        cleaner: ResumeCleanerService = Depends(ResumeCleanerService),
        analyzer = Depends(get_resume_analyzer),
        embedding_service: ResumeEmbeddingService = Depends(ResumeEmbeddingService),
        suggestion_service: ResumeSuggestionService = Depends(ResumeSuggestionService),
        persistence: ResumePersistenceService = Depends(ResumePersistenceService),
    ):
        """Initialize the pipeline with the required specialized sub-services."""
        self._extractor = extractor
        self._cleaner = cleaner
        self._analyzer: ResumeAnalyzer = analyzer
        self._embedding_service = embedding_service
        self._suggestion_service = suggestion_service
        self._persistence = persistence

    async def run(
        self,
        pdf_bytes: bytes,
        resume_id: int | None = None,
        user_id: int | None = None,
    ) -> ResumeAnalysisResult:
        """Run the complete resume processing, extraction, clean, analyze, embed, and suggestion flow.

        Args:
            pdf_bytes (bytes): Raw bytes of the resume PDF.
            resume_id (int | None): Spring Boot database resume entity ID.
            user_id (int | None): Spring Boot database user entity ID.

        Returns:
            ResumeAnalysisResult: Contains candidate_profile, embedding, and suggestions.
        """
        # 1. Extractor
        extracted_text = self._extractor.extract_text(pdf_bytes)

        # 2. Cleaner
        cleaned_text = self._cleaner.clean_text(extracted_text)

        # 3. Analyzer
        candidate_profile = await self._analyzer.analyze(cleaned_text)

        # 4. Embedding
        embedding = await self._embedding_service.generate_embedding(cleaned_text)

        # 5. Suggestions
        suggestions = await self._suggestion_service.generate_suggestions(candidate_profile)

        # 6. Persistence
        logger.info(f"ResumePipeline: calling save_resume_analysis with resume_id={resume_id}, user_id={user_id}")
        await self._persistence.save_resume_analysis(
            candidate_profile=candidate_profile,
            embedding=embedding,
            suggestions=suggestions,
            resume_id=resume_id,
            user_id=user_id,
        )

        return ResumeAnalysisResult(
            candidate_profile=candidate_profile,
            embedding=embedding,
            suggestions=suggestions,
            extracted_text=cleaned_text,  # Keep cleaned text as extracted_text for response mapping
        )
