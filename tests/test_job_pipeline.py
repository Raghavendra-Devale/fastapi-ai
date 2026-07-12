import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.job_profile import JobProfile
from app.domain.jobs.services.job_normalization_service import JobNormalizationService
from app.application.jobs.job_normalizer_service import JobNormalizerService
from app.application.jobs.job_analyzer_service import JobAnalyzerService
from app.application.jobs.job_embedding_service import JobEmbeddingService
from app.application.jobs.job_persistence_service import JobPersistenceService
from app.application.pipelines.job_pipeline import JobPipeline
from app.application.ai.embedding_service import EmbeddingService
from app.domain.ai.providers.models import EmbeddingResponse


def test_job_normalizer_service():
    domain_normalizer = JobNormalizationService()
    service = JobNormalizerService(domain_normalizer=domain_normalizer)
    
    raw_job = RawJob(
        title="  Software   Engineer  ",
        company="  Google   Inc.  ",
        location="  Remote,   US  ",
        description="Write code.",
        apply_url="http://apply.com",
        salary="150k",
        employment_type="fulltime",
        source="LinkedIn",
    )
    
    normalized = service.normalize_job(raw_job)
    assert "Title: Software Engineer" in normalized
    assert "Company: Google Inc." in normalized
    assert "Location: Remote, US" in normalized
    assert "Employment Type: Full-time" in normalized
    assert "Salary: 150k" in normalized
    assert "Description:\nWrite code." in normalized


@pytest.mark.asyncio
async def test_job_analyzer_service():
    service = JobAnalyzerService()
    raw_job = RawJob(title="Backend Dev", company="Uber")
    profile = await service.analyze_job(raw_job)
    
    assert isinstance(profile, JobProfile)
    assert profile.title == "Backend Dev"
    assert profile.company == "Uber"
    assert profile.confidence == 1.0


@pytest.mark.asyncio
async def test_job_embedding_service():
    mock_embed = AsyncMock(spec=EmbeddingService)
    mock_embed.generate_embedding.return_value = EmbeddingResponse(
        embedding=[0.9, 0.8, 0.7],
        model="test-model",
        dimensions=3,
    )
    
    service = JobEmbeddingService(embedding_service=mock_embed)
    profile = JobProfile(
        title="Frontend",
        company="Meta",
        required_skills=["React", "JS"],
    )
    
    vector = await service.generate_embedding(profile)
    assert vector == [0.9, 0.8, 0.7]
    
    # Verify that we did not embed raw JSON but formatted text
    mock_embed.generate_embedding.assert_called_once()
    called_text = mock_embed.generate_embedding.call_args[0][0]
    assert "Job Title: Frontend" in called_text
    assert "Company: Meta" in called_text
    assert "React, JS" in called_text


@pytest.mark.asyncio
async def test_job_persistence_service():
    service = JobPersistenceService()
    profile = JobProfile(title="DevOps", company="Amazon")
    success = await service.save_job_analysis(profile, [0.1])
    assert success is True


@pytest.mark.asyncio
async def test_job_pipeline_run():
    mock_normalizer = MagicMock(spec=JobNormalizerService)
    mock_normalizer.normalize_job.return_value = "Normalized Text"
    
    mock_analyzer = AsyncMock(spec=JobAnalyzerService)
    mock_analyzer.analyze_job.return_value = JobProfile(
        title="SRE",
        company="Netflix",
    )
    
    mock_embedding = AsyncMock(spec=JobEmbeddingService)
    mock_embedding.generate_embedding.return_value = [0.1, 0.2]
    
    mock_persistence = AsyncMock(spec=JobPersistenceService)
    mock_persistence.save_job_analysis.return_value = True
    
    pipeline = JobPipeline(
        normalizer=mock_normalizer,
        analyzer=mock_analyzer,
        embedding_service=mock_embedding,
        persistence=mock_persistence,
    )
    
    raw_job = RawJob(title="SRE", company="Netflix")
    result = await pipeline.run(raw_job)
    
    assert result.job_profile.title == "SRE"
    assert result.embedding == [0.1, 0.2]
    assert result.normalized_job == "Normalized Text"
    
    mock_normalizer.normalize_job.assert_called_once_with(raw_job)
    mock_analyzer.analyze_job.assert_called_once_with(raw_job)
    mock_embedding.generate_embedding.assert_called_once_with(result.job_profile)
    mock_persistence.save_job_analysis.assert_called_once_with(
        job_profile=result.job_profile,
        embedding=result.embedding,
    )
