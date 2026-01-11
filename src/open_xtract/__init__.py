"""
open_xtract - Extract structured data from documents, images, audio, and video using LLMs.
"""

import logfire

from ._extract import extract
from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

__all__ = [
    "extract",
    "configure_logging",
    "stop_temporal",
    "ExtractionError",
    "ModelError",
    "SchemaValidationError",
    "UrlFetchError",
]


def stop_temporal() -> None:
    """
    Stop the Temporal server stack (PostgreSQL, Temporal, and UI).

    Call this function when you're done with durable extractions to clean up
    Docker resources. This is optional - the services will continue running
    for subsequent calls if not stopped.
    """
    try:
        from ._docker import stop_temporal_server

        stop_temporal_server()
    except ImportError:
        pass


def configure_logging() -> None:
    """
    Configure logfire instrumentation for pydantic-ai and httpx.

    Call this function to enable detailed logging and tracing of extraction requests.
    """
    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
