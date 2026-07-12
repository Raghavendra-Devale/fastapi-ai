import json
from typing import Any
from app.core.exceptions import ValidationException


class ResponseValidator:
    """Validator to verify the structure and validity of raw LLM responses before parsing."""

    def validate_json(self, raw_response: str) -> dict[str, Any]:
        """Validate that the raw response is non-empty and a valid JSON string.

        Args:
            raw_response (str): The raw response from the LLM.

        Returns:
            dict[str, Any]: The loaded JSON dictionary.

        Raises:
            ValidationException: If response is empty or invalid JSON.
        """
        if not raw_response or not raw_response.strip():
            raise ValidationException("LLM response is empty or blank.")

        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError as e:
            raise ValidationException(f"LLM response is not valid JSON: {str(e)}")

    def validate_required_fields(self, data: dict[str, Any], required_fields: list[str]) -> None:
        """Validate that the JSON data contains all required fields.

        Args:
            data (dict[str, Any]): Loaded JSON data.
            required_fields (list[str]): List of required field names.

        Raises:
            ValidationException: If any required field is missing or empty.
        """
        if not required_fields:
            return

        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValidationException(f"Missing required fields in LLM JSON response: {', '.join(missing)}")
