import re
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from app.domain.ai.exceptions import AIValidationException
from app.application.ai.response_validator import ResponseValidator

T = TypeVar("T", bound=BaseModel)


class StructuredOutputService:
    """Service responsible for extracting JSON from LLM outputs and parsing them into Pydantic models."""

    def __init__(self, validator=None):
        self._validator: ResponseValidator = validator or ResponseValidator()

    def _extract_json_block(self, text: str) -> str:
        """Helper to extract a JSON block or clean text from raw LLM output.

        Attempts to locate a ```json ... ``` code block, or falls back to finding
        the first '{' or '[' and matching to the end of the text.
        """
        if not text:
            return ""

        # 1. Match code blocks with json specifier
        json_code_block_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Match general triple-backtick blocks
        code_block_pattern = r"```\s*(.*?)\s*```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 3. Search for the outermost JSON object boundaries
        start_idx = text.find("{")
        if start_idx != -1:
            end_idx = text.rfind("}")
            if end_idx != -1 and end_idx > start_idx:
                return text[start_idx : end_idx + 1].strip()

        start_idx = text.find("[")
        if start_idx != -1:
            end_idx = text.rfind("]")
            if end_idx != -1 and end_idx > start_idx:
                return text[start_idx : end_idx + 1].strip()

        return text.strip()

    def parse(self, response: str, model: Type[T]) -> T:
        """Extract JSON from response, validate structure, and convert to Pydantic model.

        Args:
            response (str): Raw string output from LLM.
            model (Type[T]): Pydantic model class to parse into.

        Returns:
            T: Parsed Pydantic model.

        Raises:
            AIParsingException: If extraction or JSON decoding fails.
            AIValidationException: If required fields are missing or model validation fails.
        """
        # 1. Extract potential JSON content
        extracted_content = self._extract_json_block(response)

        # 2. Validate JSON structure (empty check and decode)
        json_data = self._validator.validate_json(extracted_content)

        # 3. Validate required fields of the Pydantic model at the JSON level
        schema = model.model_json_schema()
        required_fields = schema.get("required", [])
        self._validator.validate_required_fields(json_data, required_fields)

        # 4. Convert/deserialize into Pydantic model
        try:
            return model.model_validate(json_data)
        except ValidationError as e:
            raise AIValidationException(f"Failed to validate data against Pydantic model {model.__name__}: {str(e)}")
