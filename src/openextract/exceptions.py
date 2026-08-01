"""Exceptions for openextract."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""


class ModelError(ExtractionError):
    """Error communicating with the model API.

    The optional metadata is populated for provider exceptions when available.
    ``retryable`` defaults to ``True`` so manually raised ``ModelError`` values
    retain their historical retry behavior.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        if retryable is None:
            retryable = (
                status_code is None
                or status_code in {408, 409, 425, 429}
                or 500 <= status_code <= 599
            )
        self.retryable = retryable
        self.retry_after = retry_after


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
