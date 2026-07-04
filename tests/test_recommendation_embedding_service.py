import pytest
from unittest.mock import MagicMock

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.recommendation.models.recommendation_request import JobDocument
from app.domain.recommendation.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_embedding_provider():
    return MagicMock(spec=EmbeddingProvider)


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.embedding_model = "test-embedding-model"
    return settings


@pytest.mark.asyncio
async def test_embed_resume_success(mock_embedding_provider, mock_settings):
    expected_embedding = [0.1, 0.2, 0.3]
    mock_embedding_provider.embed = MagicMock(return_value=expected_embedding)

    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)
    embedding = await service.embed_resume("This is a clean resume text.")

    assert embedding == expected_embedding
    mock_embedding_provider.embed.assert_called_once_with("This is a clean resume text.")


@pytest.mark.asyncio
async def test_embed_resume_empty_validation(mock_embedding_provider, mock_settings):
    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)

    with pytest.raises(ValidationException) as exc_info:
        await service.embed_resume("")
    assert "cannot be empty" in str(exc_info.value)

    with pytest.raises(ValidationException) as exc_info:
        await service.embed_resume("   ")
    assert "cannot be empty" in str(exc_info.value)

    mock_embedding_provider.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embed_resume_provider_failure(mock_embedding_provider, mock_settings):
    mock_embedding_provider.embed = MagicMock(
        side_effect=ExternalServiceException("Connection Timeout", error_code="TIMEOUT")
    )

    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.embed_resume("Some resume text.")
    assert exc_info.value.error_code == "TIMEOUT"


@pytest.mark.asyncio
async def test_embed_jobs_success(mock_embedding_provider, mock_settings):
    jobs = [
        JobDocument(
            title="Software Engineer",
            company="Acme Corp",
            location="Remote",
            description="Coding in Python",
            apply_url="https://acme.corp/apply",
        ),
        JobDocument(
            title="Product Manager",
            company="Beta Corp",
            location="New York",
            description="Manage product roadmap",
            apply_url="https://beta.corp/apply",
        ),
    ]

    mock_embedding_provider.embed_batch = MagicMock(
        return_value=[[0.1, 0.2], [0.3, 0.4]]
    )

    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)
    embeddings = await service.embed_jobs(jobs)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]

    mock_embedding_provider.embed_batch.assert_called_once()
    called_args = mock_embedding_provider.embed_batch.call_args[0][0]
    assert len(called_args) == 2
    assert "Title: Software Engineer" in called_args[0]
    assert "Description: Coding in Python" in called_args[0]
    assert "Title: Product Manager" in called_args[1]


@pytest.mark.asyncio
async def test_embed_jobs_empty_validation(mock_embedding_provider, mock_settings):
    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)

    with pytest.raises(ValidationException) as exc_info:
        await service.embed_jobs([])
    assert "cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embed_jobs_provider_failure(mock_embedding_provider, mock_settings):
    jobs = [
        JobDocument(
            title="Software Engineer",
            company="Acme Corp",
            location="Remote",
            description="Coding in Python",
            apply_url="https://acme.corp/apply",
        )
    ]

    mock_embedding_provider.embed_batch = MagicMock(
        side_effect=ExternalServiceException("Ollama Offline", error_code="PROVIDER_OFFLINE")
    )

    service = EmbeddingService(embedding_provider=mock_embedding_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.embed_jobs(jobs)
    assert exc_info.value.error_code == "PROVIDER_OFFLINE"
