"""Exceptions for openextract."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""


class ModelError(ExtractionError):
    """Error communicating with the model API."""


class SchemaValidationError(ExtractionError):
    """Model output did not match the expected schema."""


class UrlFetchError(ExtractionError):
    """Error fetching the URL content."""
