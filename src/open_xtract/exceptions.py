"""Exceptions for open_xtract."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""

    pass


class ModelError(ExtractionError):
    """Error communicating with the model API."""

    pass


class SchemaValidationError(ExtractionError):
    """Model output did not match the expected schema."""

    pass


class UrlFetchError(ExtractionError):
    """Error fetching the URL content."""

    pass
