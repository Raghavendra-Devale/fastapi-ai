import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException
from app.domain.ai.providers.implementations.sentence_transformer_provider import SentenceTransformerProvider


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.embedding_model = "all-MiniLM-L6-v2"
    return settings


def test_provider_initialization_success(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer") as mock_st_class:
        provider = SentenceTransformerProvider(mock_settings)
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")
        assert provider is not None


def test_provider_initialization_failure(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer", side_effect=Exception("Disk Error")):
        with pytest.raises(ExternalServiceException) as exc_info:
            SentenceTransformerProvider(mock_settings)
        assert exc_info.value.error_code == "EMBEDDING_MODEL_INIT_FAILED"


def test_provider_embed_success(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer") as mock_st_class:
        mock_st_instance = mock_st_class.return_value
        mock_st_instance.encode.return_value = np.array([0.1, 0.2, 0.3])

        provider = SentenceTransformerProvider(mock_settings)
        vector = provider.embed("hello world")

        assert vector == [0.1, 0.2, 0.3]
        mock_st_instance.encode.assert_called_once_with("hello world")


def test_provider_embed_empty_validation(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer"):
        provider = SentenceTransformerProvider(mock_settings)
        
        with pytest.raises(ValueError):
            provider.embed("")
        with pytest.raises(ValueError):
            provider.embed("   ")


def test_provider_embed_failure(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer") as mock_st_class:
        mock_st_instance = mock_st_class.return_value
        mock_st_instance.encode.side_effect = Exception("Out of Memory")

        provider = SentenceTransformerProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            provider.embed("hello")
        assert exc_info.value.error_code == "EMBEDDING_GENERATION_FAILED"


def test_provider_embed_batch_success(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer") as mock_st_class:
        mock_st_instance = mock_st_class.return_value
        mock_st_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        provider = SentenceTransformerProvider(mock_settings)
        vectors = provider.embed_batch(["hello", "world"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        mock_st_instance.encode.assert_called_once_with(["hello", "world"])


def test_provider_embed_batch_empty_validation(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer"):
        provider = SentenceTransformerProvider(mock_settings)
        
        with pytest.raises(ValueError):
            provider.embed_batch([])


def test_provider_embed_batch_failure(mock_settings):
    with patch("app.domain.ai.providers.implementations.sentence_transformer_provider.SentenceTransformer") as mock_st_class:
        mock_st_instance = mock_st_class.return_value
        mock_st_instance.encode.side_effect = Exception("CUDA error")

        provider = SentenceTransformerProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            provider.embed_batch(["hello"])
        assert exc_info.value.error_code == "EMBEDDING_BATCH_GENERATION_FAILED"
