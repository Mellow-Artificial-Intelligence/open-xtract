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


class InputError(OpenXtractError):
    """Raised when there's an error with the input file or data."""

    pass


class ExtractionError(OpenXtractError):
    """Raised when extraction fails or output doesn't match schema."""

    pass
