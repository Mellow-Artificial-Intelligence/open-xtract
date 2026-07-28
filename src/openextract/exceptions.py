"""Exceptions for openextract."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""


class ModelError(ExtractionError):
    """Error communicating with the model API."""


class ProviderNotInstalledError(ExtractionError):
    """The SDK for the requested model's provider is not installed.

    Raised when a model is requested whose provider extra was not installed,
    e.g. calling ``extract(..., model="openai:gpt-4o")`` without first running
    ``pip install 'openextract[openai]'``.
    """


class SchemaValidationError(ExtractionError):
    """Model output did not match the expected schema."""


class UrlFetchError(ExtractionError):
    """Error fetching the URL content."""


class InputTooLargeError(ExtractionError):
    """Input media exceeds the configured size limit."""
