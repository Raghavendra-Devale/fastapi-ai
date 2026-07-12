from app.core.exceptions import AppException, ExternalServiceException, ValidationException


class AIProviderException(ExternalServiceException):
    """Raised when calls to LLM / Embedding providers fail (e.g. connection error, rate limits)."""

    def __init__(self, message: str, error_code: str = "AI_PROVIDER_ERROR"):
        super().__init__(message=message, error_code=error_code, status_code=502)


class AIParsingException(AppException):
    """Raised when structured JSON block extraction from LLM response fails."""

    def __init__(self, message: str, error_code: str = "AI_PARSING_ERROR"):
        super().__init__(error_code=error_code, message=message, status_code=500)


class AIValidationException(ValidationException):
    """Raised when LLM response content fails structural Pydantic validation (e.g. missing keys)."""

    def __init__(self, message: str, error_code: str = "AI_VALIDATION_ERROR"):
        super().__init__(message=message, error_code=error_code)
