import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.exceptions import ValidationException
from app.application.pipelines.resume_pipeline import ResumePipeline
from app.application.resume.resume_extractor_service import ResumeExtractorService
from app.application.resume.resume_cleaner_service import ResumeCleanerService
from app.application.resume.resume_analyzer_service import ResumeAnalyzer, MockResumeAnalyzer
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.resume.resume_suggestion_service import ResumeSuggestionService
from app.application.resume.resume_persistence_service import ResumePersistenceService
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.resume_suggestion import ResumeSuggestion
from app.domain.resume.services.pdf_extraction_service import PDFExtractionService
from app.domain.resume.services.text_normalization_service import TextNormalizationService
from app.application.ai.embedding_service import EmbeddingService
from app.domain.ai.providers.models import EmbeddingResponse


def test_resume_extractor_service():
    mock_pdf_extractor = MagicMock(spec=PDFExtractionService)
    mock_pdf_extractor.extract_text.return_value = "Extracted Text"
    
    service = ResumeExtractorService(pdf_extractor=mock_pdf_extractor)
    result = service.extract_text(b"some pdf content")
    
    assert result == "Extracted Text"
    mock_pdf_extractor.extract_text.assert_called_once_with(b"some pdf content")


def test_resume_cleaner_service():
    mock_normalizer = MagicMock(spec=TextNormalizationService)
    mock_normalizer.normalize_text.return_value = "Cleaned Text"
    
    service = ResumeCleanerService(text_normalizer=mock_normalizer)
    result = service.clean_text("Extracted Text")
    
    assert result == "Cleaned Text"
    mock_normalizer.normalize_text.assert_called_once_with("Extracted Text")


@pytest.mark.asyncio
async def test_resume_analyzer_service():
    service = MockResumeAnalyzer()
    profile = await service.analyze("Cleaned Text")
    
    assert isinstance(profile, CandidateProfile)
    assert profile.name == "John Doe"
    assert profile.experience_years == 5.0


@pytest.mark.asyncio
async def test_resume_embedding_service():
    mock_embed_service = AsyncMock(spec=EmbeddingService)
    mock_embed_service.generate_embedding.return_value = EmbeddingResponse(
        embedding=[0.1, 0.2, 0.3],
        model="test-model",
        dimensions=3,
    )
    
    service = ResumeEmbeddingService(embedding_service=mock_embed_service)
    result = await service.generate_embedding("Cleaned Text")
    
    assert result == [0.1, 0.2, 0.3]
    mock_embed_service.generate_embedding.assert_called_once_with("Cleaned Text")


@pytest.mark.asyncio
async def test_resume_suggestion_service():
    service = ResumeSuggestionService()
    
    # 1. Profile with no summary/skills/experience
    profile_empty = CandidateProfile(
        name="Empty",
        headline=None,
        summary=None,
        experience_years=None,
        skills=[],
        experience=[],
        education=[],
        certifications=[],
        languages=[],
    )
    
    suggestions = await service.generate_suggestions(profile_empty)
    assert len(suggestions) == 3
    assert any(s.category == "Content" and "summary" in s.message for s in suggestions)
    assert any(s.category == "Skills" for s in suggestions)
    assert any(s.category == "Content" and "experience" in s.message for s in suggestions)


@pytest.mark.asyncio
async def test_resume_persistence_service():
    service = ResumePersistenceService()
    profile = CandidateProfile(name="Test")
    success = await service.save_resume_analysis(
        candidate_profile=profile,
        embedding=[0.1],
        suggestions=[]
    )
    assert success is True


@pytest.mark.asyncio
async def test_resume_pipeline_run():
    mock_extractor = MagicMock(spec=ResumeExtractorService)
    mock_extractor.extract_text.return_value = "Extracted Raw"
    
    mock_cleaner = MagicMock(spec=ResumeCleanerService)
    mock_cleaner.clean_text.return_value = "Cleaned Text"
    
    mock_analyzer = AsyncMock(spec=ResumeAnalyzer)
    mock_analyzer.analyze.return_value = CandidateProfile(
        name="John Doe",
        summary="Summary text"
    )
    
    mock_embedding = AsyncMock(spec=ResumeEmbeddingService)
    mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
    
    mock_suggestion = AsyncMock(spec=ResumeSuggestionService)
    mock_suggestion.generate_suggestions.return_value = [
        ResumeSuggestion(category="Test", severity="Low", message="Test message")
    ]
    
    mock_persistence = AsyncMock(spec=ResumePersistenceService)
    mock_persistence.save_resume_analysis.return_value = True
    
    pipeline = ResumePipeline(
        extractor=mock_extractor,
        cleaner=mock_cleaner,
        analyzer=mock_analyzer,
        embedding_service=mock_embedding,
        suggestion_service=mock_suggestion,
        persistence=mock_persistence,
    )
    
    pdf_bytes = b"mock pdf"
    result = await pipeline.run(pdf_bytes)
    
    assert result.candidate_profile.name == "John Doe"
    assert result.embedding == [0.1, 0.2, 0.3]
    assert len(result.suggestions) == 1
    assert result.extracted_text == "Cleaned Text"
    
    mock_extractor.extract_text.assert_called_once_with(pdf_bytes)
    mock_cleaner.clean_text.assert_called_once_with("Extracted Raw")
    mock_analyzer.analyze.assert_called_once_with("Cleaned Text")
    mock_embedding.generate_embedding.assert_called_once_with("Cleaned Text")
    mock_suggestion.generate_suggestions.assert_called_once_with(result.candidate_profile)
    mock_persistence.save_resume_analysis.assert_called_once_with(
        candidate_profile=result.candidate_profile,
        embedding=result.embedding,
        suggestions=result.suggestions,
    )
