class AppException(Exception):
    """Base application exception for all domain and system errors."""

    def __init__(self, error_code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


class ValidationException(AppException):
    """Raised when request payload or parameter validation fails."""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(error_code=error_code, message=message, status_code=400)


class ExternalServiceException(AppException):
    """Raised when calls to downstream AI providers or integrations fail."""

    def __init__(
        self,
        message: str,
        error_code: str = "EXTERNAL_SERVICE_ERROR",
        status_code: int = 502,
    ):
        super().__init__(error_code=error_code, message=message, status_code=status_code)


class ResourceNotFoundException(AppException):
    """Raised when a requested resource does not exist in the system."""

    def __init__(self, message: str, error_code: str = "RESOURCE_NOT_FOUND"):
        super().__init__(error_code=error_code, message=message, status_code=404)


class ConfigurationException(AppException):
    """Raised when critical configuration settings are missing or invalid."""

    def __init__(self, message: str, error_code: str = "CONFIGURATION_ERROR"):
        super().__init__(error_code=error_code, message=message, status_code=500)
