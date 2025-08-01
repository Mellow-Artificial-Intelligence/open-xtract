"""Open Xtract - PDF extraction framework with LangGraph."""

from .processor import PDFProcessor
from .schemas.types import ExtractionResult

__all__ = ["PDFProcessor", "ExtractionResult", "main"]
__version__ = "0.1.0"


def main() -> None:
    """Command line entry point."""
    from .cli import cli_main
    cli_main()


if __name__ == "__main__":
    main() 