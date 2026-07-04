import pytest
from unittest.mock import AsyncMock

from app.application.resume.resume_application_service import ResumeApplicationService
from app.domain.resume.services.resume_processing_service import ResumeProcessingService
from app.domain.resume.schemas import ResumeProcessResponse


@pytest.mark.asyncio
async def test_resume_application_service_success():
    """Test that ResumeApplicationService correctly calls ResumeProcessingService and returns the response."""
    # Arrange
    pdf_bytes = b"dummy pdf content"
    expected_response = ResumeProcessResponse(
        resume_text="John Doe\nSoftware Engineer",
        embedding=[0.1, 0.2, 0.3],
        embedding_dimensions=3,
        embedding_model="test-model",
        processing_time_ms=45.67,
    )

    mock_processing_service = AsyncMock(spec=ResumeProcessingService)
    mock_processing_service.process_resume.return_value = expected_response

    app_service = ResumeApplicationService(resume_processing_service=mock_processing_service)

    # Act
    response = await app_service.process_resume(pdf_bytes)

    # Assert
    assert response == expected_response
    mock_processing_service.process_resume.assert_called_once_with(pdf_bytes)


@pytest.mark.asyncio
async def test_resume_application_service_error_propagation():
    """Test that exceptions from ResumeProcessingService propagate through ResumeApplicationService."""
    # Arrange
    pdf_bytes = b"corrupted bytes"
    mock_processing_service = AsyncMock(spec=ResumeProcessingService)
    mock_processing_service.process_resume.side_effect = ValueError("Processing failed")

    app_service = ResumeApplicationService(resume_processing_service=mock_processing_service)

    # Act & Assert
    with pytest.raises(ValueError, match="Processing failed"):
        await app_service.process_resume(pdf_bytes)
