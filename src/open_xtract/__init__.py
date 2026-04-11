"""
open_xtract - Extract structured data from documents, images, audio, and video using LLMs.
"""

import logfire

from ._extract import extract
from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

__all__ = [
    "extract",
    "configure_logging",
    "ExtractionError",
    "ModelError",
    "SchemaValidationError",
    "UrlFetchError",
]


def configure_logging() -> None:
    """
    Configure logfire instrumentation for pydantic-ai and httpx.

    Call this function to enable detailed logging and tracing of extraction requests.
    """
    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
