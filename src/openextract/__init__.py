"""
openextract - Extract structured data from documents, images, audio, and video using LLMs.
"""

from ._extract import extract
from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

__all__ = [
    "extract",
    "ExtractionError",
    "ModelError",
    "SchemaValidationError",
    "UrlFetchError",
]
