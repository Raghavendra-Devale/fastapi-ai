import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.application.ai.structured_output_service import StructuredOutputService
from app.application.ai.analyzers.base_analyzer import BaseAIAnalyzer
from app.domain.ai.exceptions import AIProviderException, AIParsingException, AIValidationException


class DummyModel(BaseModel):
    title: str = Field(..., description="Job title")
    score: float = Field(..., description="Match score")


@pytest.mark.asyncio
async def test_base_ai_analyzer_success():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"title": "Lead Dev", "score": 0.95}'

    dummy_instance = DummyModel(title="Lead Dev", score=0.95)
    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.return_value = dummy_instance

    analyzer = BaseAIAnalyzer[DummyModel](
        llm_provider=mock_llm,
        structured_output=mock_structured,
    )

    # Act
    result = await analyzer._analyze_structured(
        prompt="Rank candidate for DevOps",
        system_prompt="Return JSON",
        output_model=DummyModel,
        temperature=0.0
    )

    # Assert
    assert result == dummy_instance
    mock_llm.generate.assert_called_once_with(
        prompt="Rank candidate for DevOps",
        system_prompt="Return JSON",
        temperature=0.0
    )
    mock_structured.parse.assert_called_once_with(
        '{"title": "Lead Dev", "score": 0.95}',
        DummyModel
    )


@pytest.mark.asyncio
async def test_base_ai_analyzer_provider_failure_wrapped():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.side_effect = RuntimeError("OpenAI API Key is invalid")

    mock_structured = MagicMock(spec=StructuredOutputService)

    analyzer = BaseAIAnalyzer[DummyModel](
        llm_provider=mock_llm,
        structured_output=mock_structured,
    )

    # Act & Assert
    with pytest.raises(AIProviderException, match="LLM provider generation call failed"):
        await analyzer._analyze_structured(
            prompt="Prompt",
            system_prompt="System",
            output_model=DummyModel
        )

    mock_structured.parse.assert_not_called()


@pytest.mark.asyncio
async def test_base_ai_analyzer_parsing_failure_propagates():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = "Non JSON text response from LLM"

    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.side_effect = AIParsingException("Invalid JSON structure")

    analyzer = BaseAIAnalyzer[DummyModel](
        llm_provider=mock_llm,
        structured_output=mock_structured,
    )

    # Act & Assert
    with pytest.raises(AIParsingException, match="Invalid JSON structure"):
        await analyzer._analyze_structured(
            prompt="Prompt",
            system_prompt="System",
            output_model=DummyModel
        )


@pytest.mark.asyncio
async def test_base_ai_analyzer_validation_failure_propagates():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"title": "Dev"}'  # missing required field "score"

    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.side_effect = AIValidationException("Missing required field score")

    analyzer = BaseAIAnalyzer[DummyModel](
        llm_provider=mock_llm,
        structured_output=mock_structured,
    )

    # Act & Assert
    with pytest.raises(AIValidationException, match="Missing required field score"):
        await analyzer._analyze_structured(
            prompt="Prompt",
            system_prompt="System",
            output_model=DummyModel
        )
