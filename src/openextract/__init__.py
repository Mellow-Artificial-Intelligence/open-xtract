"""
openextract - Extract structured data from documents, images, audio, and video using LLMs.
"""

from ._extract import extract, extract_async, extract_many, extract_many_async
from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

__all__ = [
    "extract",
    "extract_async",
    "extract_many",
    "extract_many_async",
    "ExtractionError",
    "ModelError",
    "SchemaValidationError",
    "UrlFetchError",
]
