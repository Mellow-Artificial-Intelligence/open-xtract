"""Custom exceptions for OpenXtract."""


class OpenXtractError(Exception):
    """Base exception class for all OpenXtract errors."""

    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.suggestions:
            suggestions_text = "\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)
            return f"{base_msg}{suggestions_text}"
        return base_msg


class ConfigurationError(OpenXtractError):
    """Raised when there's an error in OpenXtract configuration."""

    pass


class ProviderError(OpenXtractError):
    """Raised when there's an error with the LLM provider."""

    def __init__(
        self, message: str, provider: str | None = None, suggestions: list[str] | None = None
    ) -> None:
        super().__init__(message, suggestions)
        self.provider = provider


class InputError(OpenXtractError):
    """Raised when there's an error with the input data."""

    pass


class ProcessingError(OpenXtractError):
    """Raised when there's an error during document processing."""

    pass
