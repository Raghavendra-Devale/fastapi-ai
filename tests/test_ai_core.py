import pytest
from pydantic import BaseModel, Field

from app.core.exceptions import ValidationException
from app.application.ai.prompt_manager import PromptManager
from app.application.ai.response_validator import ResponseValidator
from app.application.ai.structured_output_service import StructuredOutputService


class MockProfile(BaseModel):
    name: str = Field(..., description="Name")
    age: int = Field(..., description="Age")
    city: str | None = Field(None, description="City")


def test_prompt_manager():
    assert "{resume_text}" in PromptManager.resume_analysis()
    assert "{job_description}" in PromptManager.job_analysis()
    assert "{candidate_profile}" in PromptManager.resume_suggestions()
    assert "{candidate_profile}" in PromptManager.recommendation_reason()
    assert "{job_profile}" in PromptManager.recommendation_reason()
    assert "{user_profile}" in PromptManager.chat()
    assert "{message}" in PromptManager.chat()


def test_response_validator_json():
    validator = ResponseValidator()

    # Empty check
    with pytest.raises(ValidationException, match="empty or blank"):
        validator.validate_json("")
    with pytest.raises(ValidationException, match="empty or blank"):
        validator.validate_json("   ")

    # Invalid JSON check
    with pytest.raises(ValidationException, match="not valid JSON"):
        validator.validate_json("{invalid json}")

    # Valid JSON check
    data = validator.validate_json('{"key": "value"}')
    assert data == {"key": "value"}


def test_response_validator_required_fields():
    validator = ResponseValidator()
    data = {"name": "Alice"}

    # No required fields
    validator.validate_required_fields(data, [])

    # Missing required fields
    with pytest.raises(ValidationException, match="Missing required fields"):
        validator.validate_required_fields(data, ["name", "age"])

    # Match all required fields
    validator.validate_required_fields(data, ["name"])


def test_structured_output_service_extraction():
    service = StructuredOutputService()

    # 1. ```json block
    text_1 = "Some text before\n```json\n{\n  \"key\": \"value\"\n}\n```\nSome text after"
    assert service._extract_json_block(text_1) == "{\n  \"key\": \"value\"\n}"

    # 2. General code block
    text_2 = "```\n{\n  \"key\": \"value\"\n}\n```"
    assert service._extract_json_block(text_2) == "{\n  \"key\": \"value\"\n}"

    # 3. Boundaries matching
    text_3 = "Conversation intro {\"name\": \"Bob\"} convo outro"
    assert service._extract_json_block(text_3) == "{\"name\": \"Bob\"}"

    # 4. List boundary matching
    text_4 = "Here is the list: [1, 2, 3] enjoy!"
    assert service._extract_json_block(text_4) == "[1, 2, 3]"


def test_structured_output_service_parse_success():
    service = StructuredOutputService()
    raw_response = "```json\n{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"city\": \"NY\"\n}\n```"
    
    result = service.parse(raw_response, MockProfile)
    assert isinstance(result, MockProfile)
    assert result.name == "John"
    assert result.age == 30
    assert result.city == "NY"


def test_structured_output_service_parse_validation_failures():
    service = StructuredOutputService()

    # 1. Missing required field at dict level
    raw_missing = "```json\n{\n  \"name\": \"John\"\n}\n```"
    with pytest.raises(ValidationException, match="Missing required fields"):
        service.parse(raw_missing, MockProfile)

    # 2. Invalid types at Pydantic level
    raw_bad_type = "```json\n{\n  \"name\": \"John\",\n  \"age\": \"not-an-int\"\n}\n```"
    with pytest.raises(ValidationException, match="Failed to validate data against Pydantic model"):
        service.parse(raw_bad_type, MockProfile)
